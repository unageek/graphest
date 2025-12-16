use crate::{
    block::{Block, BlockQueue, IntegerParameter, RealParameter},
    eval_cache::{EvalCacheLevel, EvalParametricCache},
    eval_result::EvalArgs,
    geom::{Box2D, Transformation1D, TransformationMode},
    graph::{
        common::*, Graph, GraphingError, GraphingErrorKind, GraphingStatistics, Padding, Ternary,
    },
    image::{Image, PixelIndex, PixelRange},
    interval_set::TupperIntervalSet,
    region::Region,
    relation::{Relation, RelationType, VarIndices},
    set_arg,
    traits::{BytesAllocated, Single},
    vars::VarSet,
};
use inari::{const_interval, interval, Decoration, Interval};
use itertools::Itertools;
use smallvec::SmallVec;
use std::{
    convert::TryFrom,
    mem::swap,
    time::{Duration, Instant},
};

/// The graphing algorithm for parametric relations.
///
/// A parametric relation is a relation of type [`RelationType::Parametric`].
pub struct Parametric {
    rel: Relation,
    var_indices: VarIndices,
    subdivision_dirs: Vec<VarSet>,
    im: Image<PixelState>,
    bs_to_subdivide: BlockQueue,
    /// The pixel-aligned region that matches the entire image.
    im_region: Region,
    /// The affine transformation from real coordinates to image coordinates.
    real_to_im_x: Transformation1D,
    real_to_im_y: Transformation1D,
    stats: GraphingStatistics,
    mem_limit: usize,
    no_cache: EvalParametricCache,
    cache: EvalParametricCache,
}

impl Parametric {
    pub fn new(
        rel: Relation,
        region: Box2D,
        im_width: u32,
        im_height: u32,
        padding: Padding,
        mem_limit: usize,
    ) -> Self {
        assert_eq!(rel.relation_type(), RelationType::Parametric);

        let vars = rel.vars();
        let var_indices = rel.var_indices().clone();
        let subdivision_dirs = [VarSet::M, VarSet::N, VarSet::T]
            .into_iter()
            .filter(|&d| vars.contains(d))
            .collect();

        let mut g = Self {
            rel,
            var_indices,
            subdivision_dirs,
            im: Image::new(im_width, im_height),
            bs_to_subdivide: BlockQueue::new(vars),
            im_region: Region::new(
                interval!(0.0, im_width as f64).unwrap(),
                interval!(0.0, im_height as f64).unwrap(),
            ),
            real_to_im_x: Transformation1D::new(
                [region.left(), region.right()],
                [
                    point_interval(padding.left as f64),
                    point_interval((im_width - padding.right) as f64),
                ],
                TransformationMode::Precise,
            ),
            real_to_im_y: Transformation1D::new(
                [region.bottom(), region.top()],
                [
                    point_interval(padding.bottom as f64),
                    point_interval((im_height - padding.top) as f64),
                ],
                TransformationMode::Precise,
            ),
            stats: GraphingStatistics {
                eval_count: 0,
                pixels: im_width as usize * im_height as usize,
                pixels_complete: 0,
                time_elapsed: Duration::ZERO,
            },
            mem_limit,
            no_cache: EvalParametricCache::new(EvalCacheLevel::None, vars),
            cache: EvalParametricCache::new(EvalCacheLevel::Full, vars),
        };

        let mut bs = vec![Block {
            t: RealParameter::new(g.rel.t_range()),
            ..Default::default()
        }];

        if vars.contains(VarSet::M) {
            let m_range = g.rel.m_range();
            bs = IntegerParameter::initial_subdivision(m_range)
                .into_iter()
                .cartesian_product(bs)
                .map(|(m, b)| Block { m, ..b })
                .collect::<Vec<_>>();
        }

        if vars.contains(VarSet::N) {
            let n_range = g.rel.n_range();
            bs = IntegerParameter::initial_subdivision(n_range)
                .into_iter()
                .cartesian_product(bs)
                .map(|(n, b)| Block { n, ..b })
                .collect::<Vec<_>>();
        }

        let last_block = bs.len() - 1;
        g.set_last_queued_block(
            &[PixelRange::new(
                PixelIndex::new(0, 0),
                PixelIndex::new(im_width, im_height),
            )],
            last_block,
            None,
        )
        .unwrap();
        for b in bs {
            g.bs_to_subdivide.push_back(b);
        }

        g
    }

    fn refine_impl(&mut self, duration: Duration, now: &Instant) -> Result<bool, GraphingError> {
        let mut sub_bs = vec![];
        let mut incomplete_pixels = vec![];
        let mut incomplete_sub_bs = vec![];
        let mut args = self.rel.create_args();
        while let Some(b) = self.bs_to_subdivide.pop_front() {
            let bi = self.bs_to_subdivide.begin_index() - 1;
            let next_dir = if (b.next_dir_index as usize) < self.subdivision_dirs.len() {
                Some(self.subdivision_dirs[b.next_dir_index as usize])
            } else {
                None
            };
            match next_dir {
                Some(VarSet::M) => subdivide_m(&mut sub_bs, &b),
                Some(VarSet::N) => subdivide_n(&mut sub_bs, &b),
                Some(VarSet::T) => subdivide_t(&mut sub_bs, &b),
                Some(_) => panic!(),
                _ => sub_bs.push(b.clone()),
            }

            let n_sub_bs = sub_bs.len();
            for sub_b in sub_bs.drain(..) {
                self.process_block(&sub_b, &mut args, &mut incomplete_pixels);
                if self.is_any_pixel_uncertain(&incomplete_pixels, bi) {
                    incomplete_sub_bs.push((
                        sub_b,
                        incomplete_pixels.drain(..).collect::<SmallVec<[_; 4]>>(),
                    ));
                }
                incomplete_pixels.clear();
            }

            let n_max = match next_dir {
                Some(VarSet::M | VarSet::N) => 3,
                Some(VarSet::T) => 1000, // Avoid repeated subdivision of t.
                Some(_) => panic!(),
                _ => 1,
            };

            let it = (0..self.subdivision_dirs.len())
                .cycle()
                .skip(
                    if n_max * incomplete_sub_bs.len() <= n_sub_bs {
                        // Subdivide in the same direction again.
                        b.next_dir_index
                    } else {
                        // Subdivide in other direction.
                        b.next_dir_index + 1
                    }
                    .into(),
                )
                .take(self.subdivision_dirs.len());

            for (mut sub_b, incomplete_pixels) in incomplete_sub_bs.drain(..) {
                let next_dir_index = next_dir.and_then(|_| {
                    it.clone().find(|&i| {
                        let d = self.subdivision_dirs[i];
                        d == VarSet::M && sub_b.m.is_subdivisible()
                            || d == VarSet::N && sub_b.n.is_subdivisible()
                            || d == VarSet::T && sub_b.t.is_subdivisible()
                    })
                });

                if let Some(i) = next_dir_index {
                    sub_b.next_dir_index = i as u8;
                } else {
                    // Cannot subdivide in any direction.
                    self.set_undisprovable(&incomplete_pixels, bi);
                    continue;
                }

                self.bs_to_subdivide.push_back(sub_b.clone());
                let last_bi = self.bs_to_subdivide.end_index() - 1;
                self.set_last_queued_block(&incomplete_pixels, last_bi, Some(bi))?;
            }

            let mut clear_cache_and_retry = true;
            while self.bytes_allocated() > self.mem_limit {
                if clear_cache_and_retry {
                    self.cache.clear();
                    clear_cache_and_retry = false;
                } else {
                    return Err(GraphingError {
                        kind: GraphingErrorKind::ReachedMemLimit,
                    });
                }
            }

            if now.elapsed() > duration {
                break;
            }
        }

        if self.bs_to_subdivide.is_empty() {
            if self
                .im
                .pixels()
                .any(|&s| s.is_uncertain_and_undisprovable())
            {
                Err(GraphingError {
                    kind: GraphingErrorKind::ReachedSubdivisionLimit,
                })
            } else {
                Ok(true)
            }
        } else {
            Ok(false)
        }
    }

    /// Tries to prove or disprove the existence of a solution in the block,
    /// and if that fails, appends pixels that may contain solutions to `incomplete_pixels`.
    fn process_block(
        &mut self,
        block: &Block,
        args: &mut EvalArgs,
        incomplete_pixels: &mut Vec<PixelRange>,
    ) {
        set_arg!(args, self.var_indices.m, block.m.interval());
        set_arg!(args, self.var_indices.n, block.n.interval());
        set_arg!(args, self.var_indices.t, block.t.interval());
        let (xs, ys, cond) = self
            .rel
            .eval_parametric(
                args,
                &self.real_to_im_x,
                &self.real_to_im_y,
                &mut self.no_cache,
            )
            .clone();
        let rs = Self::regions(&xs, &ys)
            .map(|r| Self::outer_pixels(&r))
            .collect::<SmallVec<[_; 4]>>();

        let dec = xs.decoration().min(ys.decoration());
        if dec >= Decoration::Def && cond.certainly_true() {
            let r = rs.iter().fold(Region::EMPTY, |acc, r| acc.convex_hull(r));

            if Self::is_pixel(&r) {
                // f(…) × g(…) is interior to a single pixel.
                for p in &self.pixels_in_image(&r) {
                    self.im[p] = PixelState::True;
                }
                return;
            } else if dec >= Decoration::Dac && (r.x().wid() == 1.0 || r.y().wid() == 1.0) {
                assert_eq!(rs.len(), 1);
                let r1 = {
                    set_arg!(
                        args,
                        self.var_indices.t,
                        point_interval_possibly_infinite(block.t.interval().inf())
                    );
                    let (xs, ys, _) = self
                        .rel
                        .eval_parametric(
                            args,
                            &self.real_to_im_x,
                            &self.real_to_im_y,
                            &mut self.cache,
                        )
                        .clone();
                    let rs = Self::regions(&xs, &ys);
                    rs.single().unwrap()
                };
                let r2 = {
                    set_arg!(
                        args,
                        self.var_indices.t,
                        point_interval_possibly_infinite(block.t.interval().sup())
                    );
                    let (xs, ys, _) = self
                        .rel
                        .eval_parametric(
                            args,
                            &self.real_to_im_x,
                            &self.real_to_im_y,
                            &mut self.cache,
                        )
                        .clone();
                    let rs = Self::regions(&xs, &ys);
                    rs.single().unwrap()
                };

                let mut r12 = Region::EMPTY;
                if r.x().wid() == 1.0 {
                    // `r` is a single column.
                    let mut y1 = r1.y();
                    let mut y2 = r2.y();
                    if y2.precedes(y1) {
                        swap(&mut y1, &mut y2);
                    }
                    if y1.precedes(y2) {
                        r12 = Self::outer_pixels(&Region::new(
                            r1.x(),
                            interval!(y1.sup(), y2.inf()).unwrap(),
                        ));
                        // `r12.x()` could be wider than a pixel.
                        r12 = Region::new(r.x(), r12.y());
                    }
                } else {
                    // `r` is a single row.
                    let mut x1 = r1.x();
                    let mut x2 = r2.x();
                    if x2.precedes(x1) {
                        swap(&mut x1, &mut x2);
                    }
                    if x1.precedes(x2) {
                        r12 = Self::outer_pixels(&Region::new(
                            interval!(x1.sup(), x2.inf()).unwrap(),
                            r1.y(),
                        ));
                        // `r12.y()` could be wider than a pixel.
                        r12 = Region::new(r12.x(), r.y());
                    }
                }

                // There is at least one solution per pixel of `r12`.
                for p in &self.pixels_in_image(&r12) {
                    self.im[p] = PixelState::True;
                }

                if r12 == r {
                    return;
                }
            }
        } else if cond.certainly_false() {
            return;
        }

        incomplete_pixels.extend(rs.into_iter().map(|r| self.pixels_in_image(&r)))
    }

    fn is_any_pixel_uncertain(&self, pixels: &[PixelRange], front_block_index: usize) -> bool {
        pixels
            .iter()
            .flatten()
            .any(|p| self.im[p].is_uncertain(front_block_index))
    }

    /// For the pixel-aligned region,
    /// returns `true` if both the width and the height of the region are `1.0`.
    fn is_pixel(r: &Region) -> bool {
        r.x().wid() == 1.0 && r.y().wid() == 1.0
    }

    /// Returns the smallest pixel-aligned region that contains `r` in its interior.
    fn outer_pixels(r: &Region) -> Region {
        // 5e-324 is interpreted as the smallest positive subnormal number.
        const TINY: Interval = const_interval!(-5e-324, 5e-324);
        let x = r.x() + TINY;
        let y = r.y() + TINY;
        Region::new(
            interval!(x.inf().floor(), x.sup().ceil()).unwrap(),
            interval!(y.inf().floor(), y.sup().ceil()).unwrap(),
        )
    }

    /// For the pixel-aligned region,
    /// returns the pixels in the region that are contained in the image.
    fn pixels_in_image(&self, r: &Region) -> PixelRange {
        let r = r.intersection(&self.im_region);
        if r.is_empty() {
            PixelRange::EMPTY
        } else {
            // If `r` is degenerate, the result is `PixelRange::EMPTY`.
            let x = r.x();
            let y = r.y();
            PixelRange::new(
                PixelIndex::new(x.inf() as u32, self.im.height() - y.sup() as u32),
                PixelIndex::new(x.sup() as u32, self.im.height() - y.inf() as u32),
            )
        }
    }

    /// Returns possible combinations of `x × y`.
    fn regions<'a>(
        xs: &'a TupperIntervalSet,
        ys: &'a TupperIntervalSet,
    ) -> impl 'a + Iterator<Item = Region> {
        xs.iter()
            .cartesian_product(ys.iter())
            .filter(|(x, y)| x.g.union(y.g).is_some())
            .map(|(x, y)| Region::new(x.x, y.x))
    }

    fn set_last_queued_block(
        &mut self,
        pixels: &[PixelRange],
        block_index: usize,
        parent_block_index: Option<usize>,
    ) -> Result<(), GraphingError> {
        if let Ok(block_index) = QueuedBlockIndex::try_from(block_index) {
            for p in pixels.iter().flatten() {
                if parent_block_index.is_none()
                    || self.im[p].is_uncertain_and_disprovable(parent_block_index.unwrap())
                {
                    self.im[p] = PixelState::Uncertain(Some(block_index));
                }
            }
            Ok(())
        } else {
            Err(GraphingError {
                kind: GraphingErrorKind::BlockIndexOverflow,
            })
        }
    }

    fn set_undisprovable(&mut self, pixels: &[PixelRange], parent_block_index: usize) {
        for p in pixels.iter().flatten() {
            if self.im[p].is_uncertain_and_disprovable(parent_block_index) {
                self.im[p] = PixelState::Uncertain(None);
            }
        }
    }
}

impl Graph for Parametric {
    fn get_image(&self, im: &mut Image<Ternary>) {
        assert!(im.width() == self.im.width() && im.height() == self.im.height());
        for (s, dst) in self.im.pixels().copied().zip(im.pixels_mut()) {
            *dst = match s {
                PixelState::True => Ternary::True,
                _ if s.is_uncertain(self.bs_to_subdivide.begin_index()) => Ternary::Uncertain,
                _ => Ternary::False,
            }
        }
    }

    fn get_statistics(&self) -> GraphingStatistics {
        GraphingStatistics {
            eval_count: self.rel.eval_count(),
            pixels_complete: self
                .im
                .pixels()
                .filter(|s| !s.is_uncertain(self.bs_to_subdivide.begin_index()))
                .count(),
            ..self.stats
        }
    }

    fn refine(&mut self, duration: Duration) -> Result<bool, GraphingError> {
        let now = Instant::now();
        let result = self.refine_impl(duration, &now);
        self.stats.time_elapsed += now.elapsed();
        result
    }
}

impl BytesAllocated for Parametric {
    fn bytes_allocated(&self) -> usize {
        self.im.bytes_allocated()
            + self.bs_to_subdivide.bytes_allocated()
            + self.cache.bytes_allocated()
    }
}
