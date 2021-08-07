use super::Graph;
use crate::{
    block::{Block, BlockQueue, BlockQueueOptions},
    graph::{
        GraphingError, GraphingErrorKind, GraphingStatistics, InexactRegion, PixelState,
        QueuedBlockIndex, Region, Transform,
    },
    image::PixelRegion,
    image::{Image, PixelIndex},
    relation::{Relation, RelationType},
};
use image::{imageops, GrayAlphaImage, LumaA, Rgb, RgbImage};
use inari::{const_interval, interval, Decoration, Interval};
use itertools::Itertools;
use std::{
    convert::TryFrom,
    time::{Duration, Instant},
};

pub struct Parametric {
    rel: Relation,
    im: Image<PixelState>,
    last_queued_blocks: Image<QueuedBlockIndex>,
    // Queue blocks that will be subdivided instead of the divided blocks to save memory.
    block_queue: BlockQueue,
    first_block_in_queue: QueuedBlockIndex,
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
        let mut g = Self {
            rel,
            im: Image::new(im_width, im_height),
            last_queued_blocks: Image::new(im_width, im_height),
            block_queue: BlockQueue::new(BlockQueueOptions {
                store_t: true,
                ..Default::default()
            }),
            first_block_in_queue: 0,
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

    fn refine_impl(&mut self, timeout: Duration, now: &Instant) -> Result<bool, GraphingError> {
        let mut sub_bs = vec![];
        while let Some((bi, b)) = self.block_queue.pop_front() {
            self.first_block_in_queue = (bi + 1) as QueuedBlockIndex;

            let incomplete_regions = self.refine_hoge(&b);
            if !incomplete_regions.is_empty() {
                if b.is_subdivisible_on_t() {
                    Self::subdivide(&mut sub_bs, &b);
                    for sub_b in sub_bs.drain(..) {
                        let index = self.block_queue.push_back(sub_b);
                        for r in &incomplete_regions {
                            self.set_last_queued_block(&r, index)?;
                        }
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

            while self.im.size_in_heap()
                + self.last_queued_blocks.size_in_heap()
                + self.block_queue.size_in_heap()
                > self.mem_limit
            {
                return Err(GraphingError {
                    kind: GraphingErrorKind::ReachedMemLimit,
                });
            }

            if now.elapsed() > timeout {
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
        pixels: &PixelRegion,
        block_index: usize,
    ) -> Result<(), GraphingError> {
        if let Ok(block_index) = QueuedBlockIndex::try_from(block_index) {
            for p in pixels.iter() {
                *self.last_queued_blocks.get_mut(p) = block_index;
            }
            Ok(())
        } else {
            Err(GraphingError {
                kind: GraphingErrorKind::BlockIndexOverflow,
            })
        }
    }

    fn refine_hoge(&mut self, b: &Block) -> Vec<PixelRegion> {
        let (x, y) = self.rel.eval_parametric(b.t);

        let rs = x
            .iter()
            .cartesian_product(y.iter())
            .filter(|(x, y)| x.g.union(y.g).is_some())
            .map(|(x, y)| {
                InexactRegion::new(
                    Self::point_interval_possibly_infinite(x.x.inf()),
                    Self::point_interval_possibly_infinite(x.x.sup()),
                    Self::point_interval_possibly_infinite(y.x.inf()),
                    Self::point_interval_possibly_infinite(y.x.sup()),
                )
                .transform(&self.inv_transform)
                .outer()
            })
            .collect::<Vec<_>>();

        let mut incomplete_regions = vec![];

        if x.decoration() >= Decoration::Def && y.decoration() >= Decoration::Def {
            let r = rs.iter().fold(Region::EMPTY, |acc, r| acc.convex_hull(r));

            // Check that `r` is interior to a single pixel.
            let px = interval!(r.0.inf().floor(), r.0.sup().ceil()).unwrap();
            let py = interval!(r.1.inf().floor(), r.1.sup().ceil()).unwrap();
            if px.wid() == 1.0
                && py.wid() == 1.0
                && r.0.interior(px)
                && r.1.interior(py)
                && px.inf() >= 0.0
                && px.inf() < self.im.width() as f64
                && py.inf() >= 0.0
                && py.inf() < self.im.height() as f64
            {
                let p = PixelIndex::new(px.inf() as u32, py.inf() as u32);
                *self.im.get_mut(p) = PixelState::True;
                return incomplete_regions;
            }
        }

        for r in rs {
            let im_r = Region(
                interval!(0.0, self.im.width() as f64).unwrap(),
                interval!(0.0, self.im.height() as f64).unwrap(),
            );

            let reg = r;
            let reg = {
                let l = reg.0.inf();
                let r = reg.0.sup();
                let b = reg.1.inf();
                let t = reg.1.sup();

                let pl = if l.floor() == l {
                    l.floor() - 1.0
                } else {
                    l.floor()
                };
                let pr = if r.ceil() == r {
                    r.ceil() + 1.0
                } else {
                    r.ceil()
                };
                let pb = if b.floor() == b {
                    b.floor() - 1.0
                } else {
                    b.floor()
                };
                let pt = if t.ceil() == t {
                    t.ceil() + 1.0
                } else {
                    t.ceil()
                };

                Region(interval!(pl, pr).unwrap(), interval!(pb, pt).unwrap())
            }
            .intersection(&im_r);

            if reg.is_empty() {
                // The region is completely outside of the image.
                continue;
            }

            let r = PixelRegion::new(
                PixelIndex::new(reg.0.inf() as u32, reg.1.inf() as u32),
                PixelIndex::new(reg.0.sup() as u32, reg.1.sup() as u32),
            );

            if r.iter().all(|p| self.im.get(p) == PixelState::True) {
                continue;
            } else {
                incomplete_regions.push(r)
            }
        }

        incomplete_regions
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
    /// two sub-blocks are created at most.
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
                .filter(|t| !t.is_singleton())
                .map(|&t| Block::new(0, 0, 0, 0, Interval::ENTIRE, t)),
        );
    }
}

impl Graph for Parametric {
    fn get_gray_alpha_image(&self, im: &mut GrayAlphaImage) {
        assert!(im.width() == self.im.width() && im.height() == self.im.height());
        for ((src, bi), dst) in self
            .im
            .pixels()
            .copied()
            .zip(self.last_queued_blocks.pixels().copied())
            .zip(im.pixels_mut())
        {
            *dst = match src {
                PixelState::True => LumaA([0, 255]),
                PixelState::Uncertain if bi < self.first_block_in_queue => LumaA([0, 0]),
                _ => LumaA([0, 128]),
            }
        }
        imageops::flip_vertical_in_place(im);
    }

    fn get_image(&self, im: &mut RgbImage) {
        assert!(im.width() == self.im.width() && im.height() == self.im.height());
        for ((src, bi), dst) in self
            .im
            .pixels()
            .copied()
            .zip(self.last_queued_blocks.pixels().copied())
            .zip(im.pixels_mut())
        {
            *dst = match src {
                PixelState::True => Rgb([0, 0, 0]),
                PixelState::Uncertain if bi < self.first_block_in_queue => Rgb([255, 255, 255]),
                _ => Rgb([64, 128, 192]),
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
                .filter(|&(s, bi)| s == PixelState::True || bi < self.first_block_in_queue)
                .count(),
            eval_count: self.rel.eval_count(),
            ..self.stats
        }
    }

    fn refine(&mut self, timeout: Duration) -> Result<bool, GraphingError> {
        let now = Instant::now();
        let result = self.refine_impl(timeout, &now);
        self.stats.time_elapsed += now.elapsed();
        result
    }
}
