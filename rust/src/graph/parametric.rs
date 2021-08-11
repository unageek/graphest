use super::Graph;
use crate::{
    block::{Block, BlockQueue, BlockQueueOptions},
    graph::{GraphingError, GraphingErrorKind, GraphingStatistics, PixelState, QueuedBlockIndex},
    image::{Image, PixelIndex, PixelRegion},
    interval_set::{DecSignSet, SignSet},
    ops::StaticForm,
    region::{InexactRegion, Region, Transform},
    relation::{Relation, RelationType},
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
        };

        let t_range = g.rel.t_range();
        g.block_queue
            .push_back(Block::new(0, 0, 0, 0, Interval::ENTIRE, t_range));

        g
    }

    fn refine_impl(&mut self, duration: Duration, now: &Instant) -> Result<bool, GraphingError> {
        let mut sub_bs = vec![];
        while let Some(b) = self.block_queue.pop_front() {
            let incomplete_regions = self.process_block(&b);
            if !incomplete_regions.is_empty() {
                if b.is_subdivisible_on_t() {
                    Self::subdivide(&mut sub_bs, &b);
                    for sub_b in sub_bs.drain(..) {
                        self.block_queue.push_back(sub_b);
                    }
                    let last_index = self.block_queue.end_index() - 1;
                    for r in incomplete_regions {
                        self.set_last_queued_block(&r, last_index)?;
                    }
                } else {
                    for r in incomplete_regions {
                        for p in r.iter() {
                            if self.im.get(p) == PixelState::Uncertain {
                                *self.im.get_mut(p) = PixelState::UncertainNeverFalse;
                            }
                        }
                    }
                }
            }

            if self.im.size_in_heap()
                + self.last_queued_blocks.size_in_heap()
                + self.block_queue.size_in_heap()
                > self.mem_limit
            {
                return Err(GraphingError {
                    kind: GraphingErrorKind::ReachedMemLimit,
                });
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

        let (x, y, cond) = self.rel.eval_parametric(block.t);
        let rs = x
            .iter()
            .cartesian_product(y.iter())
            .filter(|(x, y)| x.g.union(y.g).is_some())
            .map(|(x, y)| {
                let r = InexactRegion::new(
                    Self::point_interval_possibly_infinite(x.x.inf()),
                    Self::point_interval_possibly_infinite(x.x.sup()),
                    Self::point_interval_possibly_infinite(y.x.inf()),
                    Self::point_interval_possibly_infinite(y.x.sup()),
                )
                .transform(&self.inv_transform)
                .outer();
                outer_pixels(&r)
            })
            .collect::<Vec<_>>();

        let cond_is_true = cond
            .map(|DecSignSet(ss, d)| ss == SignSet::ZERO && d >= Decoration::Def)
            .eval(&self.forms[..]);
        let cond_is_false = !cond
            .map(|DecSignSet(ss, _)| ss.contains(SignSet::ZERO))
            .eval(&self.forms[..]);

        let mut incomplete_rs = vec![];

        let im_r = Region::new(
            interval!(0.0, self.im.width() as f64).unwrap(),
            interval!(0.0, self.im.height() as f64).unwrap(),
        );

        if x.decoration().min(y.decoration()) >= Decoration::Def && cond_is_true {
            let r = rs.iter().fold(Region::EMPTY, |acc, r| acc.convex_hull(r));

            let x = r.x();
            let y = r.y();
            if x.wid() == 1.0 && y.wid() == 1.0 && r.subset(&im_r) {
                // f(t) × g(t) is interior to a single pixel.
                let p = PixelIndex::new(x.inf() as u32, y.inf() as u32);
                *self.im.get_mut(p) = PixelState::True;
                return incomplete_rs;
            }
        } else if cond_is_false {
            return incomplete_rs;
        }

        for r in rs {
            let r = r.intersection(&im_r);
            if r.is_empty() {
                continue;
            }

            let x = r.x();
            let y = r.y();
            // If the region touches the image from the outside, `r` will be empty.
            let r = PixelRegion::new(
                PixelIndex::new(x.inf() as u32, y.inf() as u32),
                PixelIndex::new(x.sup() as u32, y.sup() as u32),
            );

            if r.iter().all(|p| self.im.get(p) == PixelState::True) {
                continue;
            } else {
                incomplete_rs.push(r)
            }
        }

        incomplete_rs
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
