use crate::{
    block::{Block, BlockQueue, RealParameter},
    eval_cache::{EvalCacheLevel, EvalParametricCache},
    eval_result::EvalArgs,
    geom::{Box2D, Transform2D, TransformMode},
    graph::{
        common::*, Graph, GraphingError, GraphingErrorKind, GraphingStatistics, Padding, Ternary,
    },
    image::{Image, PixelIndex, PixelRange},
    interval_set::TupperIntervalSet,
    region::Region,
    relation::{Relation, RelationType, VarIndices},
    set_arg,
    traits::BytesAllocated,
    vars::VarSet,
};
use inari::{const_interval, interval, Decoration, Interval};
use itertools::Itertools;
use std::{
    convert::TryFrom,
    iter::once,
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
    real_to_im: Transform2D,
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
        let subdivision_dirs = [VarSet::N, VarSet::T]
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
            real_to_im: Transform2D::new(
                [
                    Region::new(region.left(), region.bottom()),
                    Region::new(region.right(), region.top()),
                ],
                [
                    Region::new(
                        point_interval(padding.left as f64),
                        point_interval(padding.bottom as f64),
                    ),
                    Region::new(
                        point_interval((im_width - padding.right) as f64),
                        point_interval((im_height - padding.top) as f64),
                    ),
                ],
                TransformMode::Precise,
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

        g.bs_to_subdivide.push_back(Block {
            t: RealParameter::new(g.rel.t_range()),
            next_dir: g.subdivision_dirs[0],
            ..Block::default()
        });

        g
    }

    fn refine_impl(&mut self, duration: Duration, now: &Instant) -> Result<bool, GraphingError> {
        let mut sub_bs = vec![];
        let mut incomplete_sub_bs = vec![];
        let mut next_dir_candidates = vec![];
        let mut args = self.rel.create_args();
        while let Some(b) = self.bs_to_subdivide.pop_front() {
            let bi = self.bs_to_subdivide.begin_index() - 1;
            match b.next_dir {
                VarSet::N => subdivide_n(&mut sub_bs, &b),
                VarSet::T => subdivide_t(&mut sub_bs, &b),
                _ => panic!(),
            }

            let n_sub_bs = sub_bs.len();
            for sub_b in sub_bs.drain(..) {
                let incomplete_pixels = self.process_block(&sub_b, &mut args);
                if self.is_any_pixel_uncertain(&incomplete_pixels, bi) {
                    incomplete_sub_bs.push((sub_b, incomplete_pixels));
                }
            }

            let n_max = match b.next_dir {
                VarSet::N => 3,
                VarSet::T => 1000, // Avoid repeated subdivision of t.
                _ => panic!(),
            };
            let next_dir_suggestion = if n_max * incomplete_sub_bs.len() <= n_sub_bs {
                // Subdivide in the same direction again.
                b.next_dir
            } else {
                // Subdivide in other direction.
                let mut it = self.subdivision_dirs.iter().copied().cycle();
                it.find(|&d| d == b.next_dir);
                it.next().unwrap()
            };

            next_dir_candidates.splice(
                ..,
                once(next_dir_suggestion).chain(
                    self.subdivision_dirs
                        .iter()
                        .copied()
                        .filter(|&d| d != next_dir_suggestion),
                ),
            );

            for (mut sub_b, incomplete_pixels) in incomplete_sub_bs.drain(..) {
                let next_dir = next_dir_candidates.iter().copied().find(|&d| {
                    d == VarSet::N && sub_b.n.is_subdivisible()
                        || d == VarSet::T && sub_b.t.is_subdivisible()
                });

                if let Some(d) = next_dir {
                    sub_b.next_dir = d;
                } else {
                    // Cannot subdivide in any direction.
                    self.set_undisprovable(&incomplete_pixels, bi);
                    continue;
                }

                self.bs_to_subdivide.push_back(sub_b.clone());
                let last_bi = self.bs_to_subdivide.end_index() - 1;
                self.set_last_queued_block(&incomplete_pixels, last_bi, bi)?;
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

    /// Tries to prove or disprove the existence of a solution in the block
    /// and if it is unsuccessful, returns pixels that possibly contain solutions.
    fn process_block(&mut self, block: &Block, args: &mut EvalArgs) -> Vec<PixelRange> {
        set_arg!(args, self.var_indices.n, block.n.interval());
        set_arg!(args, self.var_indices.t, block.t.interval());
        let (xs, ys, cond) = self.rel.eval_parametric(args, &mut self.no_cache);
        let rs = self
            .im_regions(&xs, &ys)
            .into_iter()
            .map(|r| Self::outer_pixels(&r))
            .collect::<Vec<_>>();

        let cond = cond.result(self.rel.forms());

        let dec = xs.decoration().min(ys.decoration());
        if dec >= Decoration::Def && cond.certainly_true() {
            let r = rs.iter().fold(Region::EMPTY, |acc, r| acc.convex_hull(r));

            if Self::is_pixel(&r) {
                // f(…) × g(…) is interior to a single pixel.
                for p in &self.pixels_in_image(&r) {
                    self.im[p] = PixelState::True;
                }
                return vec![];
            } else if dec >= Decoration::Dac && (r.x().wid() == 1.0 || r.y().wid() == 1.0) {
                assert_eq!(rs.len(), 1);
                let r1 = {
                    set_arg!(
                        args,
                        self.var_indices.t,
                        point_interval_possibly_infinite(block.t.interval().inf())
                    );
                    let (xs, ys, _) = self.rel.eval_parametric(args, &mut self.cache);
                    let rs = self.im_regions(&xs, &ys);
                    assert_eq!(rs.len(), 1);
                    rs[0].clone()
                };
                let r2 = {
                    set_arg!(
                        args,
                        self.var_indices.t,
                        point_interval_possibly_infinite(block.t.interval().sup())
                    );
                    let (xs, ys, _) = self.rel.eval_parametric(args, &mut self.cache);
                    let rs = self.im_regions(&xs, &ys);
                    assert_eq!(rs.len(), 1);
                    rs[0].clone()
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
                    }
                }

                // There is at least one solution per pixel of `r12`.
                for p in &self.pixels_in_image(&r12) {
                    self.im[p] = PixelState::True;
                }

                if r12 == r {
                    return vec![];
                }
            }
        } else if cond.certainly_false() {
            return vec![];
        }

        rs.into_iter().map(|r| self.pixels_in_image(&r)).collect()
    }

    /// Returns enclosures of possible combinations of `x × y` in image coordinates.
    fn im_regions(&self, xs: &TupperIntervalSet, ys: &TupperIntervalSet) -> Vec<Region> {
        xs.iter()
            .cartesian_product(ys.iter())
            .filter(|(x, y)| x.g.union(y.g).is_some())
            .map(|(x, y)| {
                Box2D::new(
                    point_interval_possibly_infinite(x.x.inf()),
                    point_interval_possibly_infinite(x.x.sup()),
                    point_interval_possibly_infinite(y.x.inf()),
                    point_interval_possibly_infinite(y.x.sup()),
                )
                .transform(&self.real_to_im)
                .outer()
            })
            .collect::<Vec<_>>()
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

    fn set_last_queued_block(
        &mut self,
        pixels: &[PixelRange],
        block_index: usize,
        parent_block_index: usize,
    ) -> Result<(), GraphingError> {
        if let Ok(block_index) = QueuedBlockIndex::try_from(block_index) {
            for p in pixels.iter().flatten() {
                if self.im[p].is_uncertain_and_disprovable(parent_block_index) {
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
