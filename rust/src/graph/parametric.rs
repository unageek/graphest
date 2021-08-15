use super::Graph;
use crate::{
    block::{Block, BlockQueue, BlockQueueOptions},
    graph::{GraphingError, GraphingErrorKind, GraphingStatistics, PixelState, QueuedBlockIndex},
    image::{Image, PixelIndex, PixelRegion},
    interval_set::{DecSignSet, SignSet, TupperIntervalSet},
    ops::StaticForm,
    region::{InexactRegion, Region, Transform},
    relation::{EvalParametricCache, Relation, RelationType},
};
use image::{imageops, ImageBuffer, Pixel};
use inari::{const_interval, interval, Decoration, Interval};
use itertools::Itertools;
use std::{
    convert::TryFrom,
    ops::{Deref, DerefMut},
    time::{Duration, Instant},
};

/// The graphing algorithm for parametric relations.
///
/// A parametric relation is a relation of type [`RelationType::Parametric`].
pub struct Parametric {
    rel: Relation,
    forms: Vec<StaticForm>,
    im: Image<PixelState>,
    last_queued_blocks: Image<QueuedBlockIndex>,
    block_queue: BlockQueue,
    // Affine transformation from real coordinates to pixel coordinates.
    inv_transform: Transform,
    stats: GraphingStatistics,
    mem_limit: usize,
    cache: EvalParametricCache,
}

impl Parametric {
    pub fn new(
        rel: Relation,
        region: InexactRegion,
        im_width: u32,
        im_height: u32,
        mem_limit: usize,
    ) -> Self {
        assert_eq!(rel.relation_type(), RelationType::Parametric);

        let im_width_interval = Self::point_interval(im_width as f64);
        let im_height_interval = Self::point_interval(im_height as f64);
        let forms = rel.forms().clone();
        let mut g = Self {
            rel,
            forms,
            im: Image::new(im_width, im_height),
            last_queued_blocks: Image::new(im_width, im_height),
            block_queue: BlockQueue::new(BlockQueueOptions {
                store_t: true,
                ..Default::default()
            }),
            inv_transform: {
                Transform::new(
                    im_width_interval / region.width(),
                    im_height_interval / region.height(),
                    -region.left() * (im_width_interval / region.width()),
                    -region.bottom() * (im_height_interval / region.height()),
                )
            },
            stats: GraphingStatistics {
                pixels: im_width as usize * im_height as usize,
                pixels_proven: 0,
                eval_count: 0,
                time_elapsed: Duration::ZERO,
            },
            mem_limit,
            cache: EvalParametricCache::new(),
        };

        let t_range = g.rel.t_range();
        g.block_queue
            .push_back(Block::new(0, 0, 0, 0, Interval::ENTIRE, t_range));

        g
    }

    fn refine_impl(&mut self, duration: Duration, now: &Instant) -> Result<bool, GraphingError> {
        let mut sub_bs = vec![];
        while let Some(b) = self.block_queue.pop_front() {
            let incomplete_pixel_regions = self.process_block(&b);
            if !incomplete_pixel_regions.is_empty() {
                if b.is_subdivisible_on_t() {
                    Self::subdivide(&mut sub_bs, &b);
                    for sub_b in sub_bs.drain(..) {
                        self.block_queue.push_back(sub_b);
                    }
                    let last_index = self.block_queue.end_index() - 1;
                    for ps in incomplete_pixel_regions {
                        self.set_last_queued_block(&ps, last_index)?;
                    }
                } else {
                    for ps in incomplete_pixel_regions {
                        for p in &ps {
                            if self.im.get(p) == PixelState::Uncertain {
                                *self.im.get_mut(p) = PixelState::UncertainNeverFalse;
                            }
                        }
                    }
                }
            }

            let mut clear_cache_and_retry = true;
            while self.im.size_in_heap()
                + self.last_queued_blocks.size_in_heap()
                + self.block_queue.size_in_heap()
                + self.cache.size_in_heap()
                > self.mem_limit
            {
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

        if self.block_queue.is_empty() {
            if self
                .im
                .pixels()
                .any(|&s| s == PixelState::UncertainNeverFalse)
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

    fn set_last_queued_block(
        &mut self,
        r: &PixelRegion,
        block_index: usize,
    ) -> Result<(), GraphingError> {
        if let Ok(block_index) = QueuedBlockIndex::try_from(block_index) {
            for p in r.iter() {
                *self.last_queued_blocks.get_mut(p) = block_index;
            }
            Ok(())
        } else {
            Err(GraphingError {
                kind: GraphingErrorKind::BlockIndexOverflow,
            })
        }
    }

    /// Tries to prove or disprove the existence of a solution in the block
    /// and if it is unsuccessful, returns pixels that the block is interior to the union of them.
    fn process_block(&mut self, block: &Block) -> Vec<PixelRegion> {
        let (x, y, cond) = self.rel.eval_parametric(block.t, None);
        let rs = Self::regions(&x, &y, &self.inv_transform);

        let cond_is_true = cond
            .map(|DecSignSet(ss, d)| ss == SignSet::ZERO && d >= Decoration::Def)
            .eval(&self.forms[..]);
        let cond_is_false = !cond
            .map(|DecSignSet(ss, _)| ss.contains(SignSet::ZERO))
            .eval(&self.forms[..]);

        let mut incomplete_pixel_regions = vec![];

        let im_r = Region::new(
            interval!(0.0, self.im.width() as f64).unwrap(),
            interval!(0.0, self.im.height() as f64).unwrap(),
        );

        let dec = x.decoration().min(y.decoration());
        if dec >= Decoration::Def && cond_is_true {
            let r = rs.iter().fold(Region::EMPTY, |acc, r| acc.convex_hull(r));

            if Self::is_pixel(&r) {
                // f(t) × g(t) is interior to a single pixel.
                let ps = Self::to_pixel_region(&r.intersection(&im_r));
                for p in &ps {
                    *self.im.get_mut(p) = PixelState::True;
                }
                return incomplete_pixel_regions;
            } else if dec >= Decoration::Dac && (r.x().wid() == 1.0 || r.y().wid() == 1.0) {
                let r1 = {
                    let t = Self::point_interval_possibly_infinite(block.t.inf());
                    let (x, y, _) = self.rel.eval_parametric(t, Some(&mut self.cache));
                    let rs = Self::regions(&x, &y, &self.inv_transform);
                    assert_eq!(rs.len(), 1);
                    rs[0].clone()
                };
                let r2 = {
                    let t = Self::point_interval_possibly_infinite(block.t.sup());
                    let (x, y, _) = self.rel.eval_parametric(t, Some(&mut self.cache));
                    let rs = Self::regions(&x, &y, &self.inv_transform);
                    assert_eq!(rs.len(), 1);
                    rs[0].clone()
                };
                let r12 = r1.convex_hull(&r2);

                if Self::is_pixel(&r1) && Self::is_pixel(&r2) {
                    // There is at least one solution in each of the contiguous pixels
                    // from `r1` to `r2`.
                    let ps = Self::to_pixel_region(&r12.intersection(&im_r));
                    for p in &ps {
                        *self.im.get_mut(p) = PixelState::True;
                    }

                    if r12 == r {
                        return incomplete_pixel_regions;
                    }
                }
            }
        } else if cond_is_false {
            return incomplete_pixel_regions;
        }

        for r in rs {
            let ps = Self::to_pixel_region(&r.intersection(&im_r));
            if ps.iter().all(|p| self.im.get(p) == PixelState::True) {
                continue;
            } else {
                incomplete_pixel_regions.push(ps)
            }
        }

        incomplete_pixel_regions
    }

    /// Returns `true` if the region represents a pixel.
    fn is_pixel(r: &Region) -> bool {
        r.x().wid() == 1.0 && r.y().wid() == 1.0
    }

    fn point_interval(x: f64) -> Interval {
        interval!(x, x).unwrap()
    }

    fn point_interval_possibly_infinite(x: f64) -> Interval {
        if x == f64::NEG_INFINITY {
            const_interval!(f64::NEG_INFINITY, f64::MIN)
        } else if x == f64::INFINITY {
            const_interval!(f64::MAX, f64::INFINITY)
        } else {
            Self::point_interval(x)
        }
    }

    /// Returns pixel-aligned regions whose union contains the possible combinations of `x × y`
    /// in its interior.
    fn regions(x: &TupperIntervalSet, y: &TupperIntervalSet, inv_t: &Transform) -> Vec<Region> {
        /// Returns the smallest region whose bounds are integers
        /// and which contains `r` in its interior.
        fn outer_pixels(r: &Region) -> Region {
            const TINY: Interval = const_interval!(-5e-324, 5e-324);
            let r0 = r.x() + TINY;
            let r1 = r.y() + TINY;
            Region::new(
                interval!(r0.inf().floor(), r0.sup().ceil()).unwrap(),
                interval!(r1.inf().floor(), r1.sup().ceil()).unwrap(),
            )
        }

        x.iter()
            .cartesian_product(y.iter())
            .filter(|(x, y)| x.g.union(y.g).is_some())
            .map(|(x, y)| {
                let r = InexactRegion::new(
                    Self::point_interval_possibly_infinite(x.x.inf()),
                    Self::point_interval_possibly_infinite(x.x.sup()),
                    Self::point_interval_possibly_infinite(y.x.inf()),
                    Self::point_interval_possibly_infinite(y.x.sup()),
                )
                .transform(inv_t)
                .outer();
                outer_pixels(&r)
            })
            .collect::<Vec<_>>()
    }

    /// Subdivides `b.t` and appends the sub-blocks to `sub_bs`.
    /// Two sub-blocks are created.
    ///
    /// Precondition: `b.is_subdivisible_on_t()` is `true`.
    fn subdivide(sub_bs: &mut Vec<Block>, b: &Block) {
        fn bisect(x: Interval) -> (Interval, Interval) {
            let a = x.inf();
            let b = x.sup();
            let mid = if a == f64::NEG_INFINITY {
                if b < 0.0 {
                    (2.0 * b).max(f64::MIN)
                } else if b == 0.0 {
                    -1.0
                } else {
                    0.0
                }
            } else if b == f64::INFINITY {
                if a < 0.0 {
                    0.0
                } else if a == 0.0 {
                    1.0
                } else {
                    (2.0 * a).min(f64::MAX)
                }
            } else {
                x.mid()
            };
            (interval!(a, mid).unwrap(), interval!(mid, b).unwrap())
        }

        let (t1, t2) = bisect(b.t);
        sub_bs.extend(
            [t1, t2]
                .iter()
                .map(|&t| Block::new(0, 0, 0, 0, Interval::ENTIRE, t)),
        );
    }

    /// Returns the pixel region that corresponds to the pixel-aligned region.
    ///
    /// The empty pixel region is returned if `r` is empty or degenerate.
    ///
    /// Preconditions:
    /// - `r` is a subset of `[0, u32::MAX] × [0, u32::MAX]`.
    /// - `r` is empty, or each bound of `r` is an integer.
    fn to_pixel_region(r: &Region) -> PixelRegion {
        if r.is_empty() {
            PixelRegion::EMPTY
        } else {
            let x = r.x();
            let y = r.y();
            PixelRegion::new(
                PixelIndex::new(x.inf() as u32, y.inf() as u32),
                PixelIndex::new(x.sup() as u32, y.sup() as u32),
            )
        }
    }
}

impl Graph for Parametric {
    fn get_image<P, Container>(
        &self,
        im: &mut ImageBuffer<P, Container>,
        true_color: P,
        uncertain_color: P,
        false_color: P,
    ) where
        P: Pixel + 'static,
        Container: Deref<Target = [P::Subpixel]> + DerefMut,
    {
        assert!(im.width() == self.im.width() && im.height() == self.im.height());
        for ((s, bi), dst) in self
            .im
            .pixels()
            .copied()
            .zip(self.last_queued_blocks.pixels().copied())
            .zip(im.pixels_mut())
        {
            *dst = match s {
                PixelState::True => true_color,
                PixelState::Uncertain if (bi as usize) < self.block_queue.begin_index() => {
                    false_color
                }
                _ => uncertain_color,
            }
        }
        imageops::flip_vertical_in_place(im);
    }

    fn get_statistics(&self) -> GraphingStatistics {
        GraphingStatistics {
            pixels_proven: self
                .im
                .pixels()
                .copied()
                .zip(self.last_queued_blocks.pixels().copied())
                .filter(|&(s, bi)| {
                    s == PixelState::True || (bi as usize) < self.block_queue.begin_index()
                })
                .count(),
            eval_count: self.rel.eval_count(),
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
