use crate::{
    block::{Block, BlockQueue, BlockQueueOptions, Coordinate},
    eval_result::EvalResult,
    geom::{Box1D, Box2D, Transform1D, TransformMode},
    graph::{
        common::{
            point_interval, point_interval_possibly_infinite, simple_fraction, subpixel_outer_x,
            PixelState, QueuedBlockIndex,
        },
        Graph, GraphingError, GraphingErrorKind, GraphingStatistics, Padding, Ternary,
    },
    image::{Image, PixelIndex, PixelRange},
    interval_set::TupperIntervalSet,
    region::Region,
    relation::{EvalExplicitCache, ExplicitRelationOp, Relation, RelationType},
};
use inari::{const_interval, interval, Decoration, Interval};
use itertools::Itertools;
use smallvec::SmallVec;
use std::{
    convert::TryFrom,
    mem::swap,
    time::{Duration, Instant},
};

/// The graphing algorithm for explicit relations.
pub struct Explicit {
    rel: Relation,
    op: ExplicitRelationOp,
    transpose: bool,
    im: Image<PixelState>,
    block_queue: BlockQueue,
    im_region: Region,
    im_to_real_x: Transform1D,
    real_to_im_y: Transform1D,
    stats: GraphingStatistics,
    mem_limit: usize,
    cache: EvalExplicitCache,
}

impl Explicit {
    pub fn new(
        rel: Relation,
        region: Box2D,
        im_width: u32,
        im_height: u32,
        padding: Padding,
        mem_limit: usize,
    ) -> Self {
        let relation_type = rel.relation_type();
        assert!(matches!(
            relation_type,
            RelationType::ExplicitFunctionOfX(_) | RelationType::ExplicitFunctionOfY(_)
        ));

        let op = match relation_type {
            RelationType::ExplicitFunctionOfX(op) | RelationType::ExplicitFunctionOfY(op) => op,
            _ => unreachable!(),
        };
        let transpose = matches!(relation_type, RelationType::ExplicitFunctionOfY(_));
        let im = Image::new(im_width, im_height);

        let (region, im_width, im_height, padding) = if transpose {
            (
                region.transpose(),
                im_height,
                im_width,
                Padding {
                    bottom: padding.left,
                    left: padding.bottom,
                    right: padding.top,
                    top: padding.right,
                },
            )
        } else {
            (region, im_width, im_height, padding)
        };
        let mut g = Self {
            rel,
            op,
            transpose,
            im,
            block_queue: BlockQueue::new(BlockQueueOptions {
                store_xy: true,
                ..Default::default()
            }),
            im_region: Region::new(
                interval!(0.0, im_width as f64).unwrap(),
                interval!(0.0, im_height as f64).unwrap(),
            ),
            im_to_real_x: Transform1D::new(
                [
                    point_interval(padding.left as f64),
                    point_interval((im_width - padding.right) as f64),
                ],
                [region.left(), region.right()],
                TransformMode::Fast,
            ),
            real_to_im_y: Transform1D::new(
                [region.bottom(), region.top()],
                [
                    point_interval(padding.bottom as f64),
                    point_interval((im_height - padding.top) as f64),
                ],
                TransformMode::Precise,
            ),
            stats: GraphingStatistics {
                eval_count: 0,
                pixels: im_width as usize * im_height as usize,
                pixels_proven: 0,
                time_elapsed: Duration::ZERO,
            },
            mem_limit,
            cache: Default::default(),
        };

        let kx = (im_width as f64).log2().ceil() as i8;
        let b = Block {
            x: Coordinate::new(0, kx),
            ..Block::default()
        };
        g.block_queue.push_back(b);

        g
    }

    fn refine_impl(&mut self, duration: Duration, now: &Instant) -> Result<bool, GraphingError> {
        let mut sub_bs = vec![];
        while let Some(b) = self.block_queue.pop_front() {
            let bi = self.block_queue.begin_index() - 1;

            let incomplete_pixels = if !b.x.is_subpixel() {
                self.process_block(&b)
            } else {
                self.process_subpixel_block(&b)
            };

            if self.is_any_pixel_uncertain(&incomplete_pixels, bi) {
                if b.x.is_subdivisible() {
                    self.subdivide_on_x(&mut sub_bs, &b);
                    self.block_queue.extend(sub_bs.drain(..));
                    let last_bi = self.block_queue.end_index() - 1;
                    self.set_last_queued_block(&incomplete_pixels, last_bi, bi)?;
                } else {
                    self.set_undisprovable(&incomplete_pixels, bi);
                }
            }

            let mut clear_cache_and_retry = true;
            while self.size_in_heap() > self.mem_limit {
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
    ///
    /// Precondition: the block is either a pixel or a superpixel.
    fn process_block(&mut self, b: &Block) -> Vec<PixelRange> {
        let x_up = self.block_to_region_clipped(b).outer();
        let (ys, cond) = Self::eval_on_interval(&mut self.rel, x_up);

        let px = {
            let begin = b.pixel_index().x;
            let end = (begin + b.x.width()).min(self.im_width());
            interval!(begin as f64, end as f64).unwrap()
        };
        let im_ys = self.im_intervals(&ys);

        let cond = cond.result(self.rel.forms());
        let dec = ys.decoration();

        if dec >= Decoration::Def && cond.certainly_true() {
            let im_y = im_ys
                .iter()
                .fold(Interval::EMPTY, |acc, &y| acc.convex_hull(y));

            let pixels = self.possibly_true_pixels(px, im_y);
            let t_pixels = self.true_pixels(px, im_y);
            for p in &t_pixels {
                self.im[p] = PixelState::True;
            }
            if pixels == t_pixels {
                return vec![];
            }
        } else if cond.certainly_false() {
            return vec![];
        }

        im_ys
            .into_iter()
            .map(|im_y| self.possibly_true_pixels(px, im_y))
            .collect()
    }

    /// Tries to prove or disprove the existence of a solution in the block
    /// and if it is unsuccessful, returns pixels that possibly contain solutions.
    ///
    /// Precondition: the block is a subpixel.
    fn process_subpixel_block(&mut self, b: &Block) -> Vec<PixelRange> {
        let x_up = subpixel_outer_x(&self.block_to_region(b), b);
        let inter = {
            let p_dn = self.block_to_region(&b.pixel_block()).inner();
            x_up.intersection(p_dn)
        };
        let (ys, cond) = Self::eval_on_interval(&mut self.rel, x_up);

        let px = {
            let begin = b.pixel_index().x;
            let end = begin + 1;
            interval!(begin as f64, end as f64).unwrap()
        };
        let im_ys = self.im_intervals(&ys);

        let cond = cond.result(self.rel.forms());
        let dec = ys.decoration();

        if dec >= Decoration::Def && cond.certainly_true() {
            let im_y = im_ys
                .iter()
                .fold(Interval::EMPTY, |acc, &y| acc.convex_hull(y));

            let pixels = self.possibly_true_pixels(px, im_y);
            let t_pixels = self.true_pixels(px, im_y);
            for p in &t_pixels {
                self.im[p] = PixelState::True;
            }
            if pixels == t_pixels {
                return vec![];
            }
        } else if cond.certainly_false() {
            return vec![];
        }

        if !inter.is_empty() {
            // To dedup, points must be sorted.
            let rs = [inter.inf(), simple_fraction(inter), inter.sup()]
                .iter()
                .dedup()
                .map(|&x| Self::eval_on_point(&mut self.rel, x, Some(&mut self.cache)))
                .collect::<SmallVec<[_; 3]>>();

            if dec >= Decoration::Dac && cond.certainly_true() {
                let ys = rs
                    .into_iter()
                    .map(|(y, _)| {
                        assert_eq!(y.len(), 1);
                        y.iter().next().unwrap().x
                    })
                    .collect::<SmallVec<[_; 3]>>();

                let y0 = ys
                    .iter()
                    .map(|y| y.sup())
                    .fold(f64::INFINITY, |acc, y| acc.min(y));
                let y1 = ys
                    .iter()
                    .map(|y| y.inf())
                    .fold(f64::NEG_INFINITY, |acc, y| acc.max(y));

                if y0 <= y1 {
                    // All "possibly-true" pixels are actually true.
                    let t_pixels = self.possibly_true_pixels(
                        px,
                        Box1D::new(
                            point_interval_possibly_infinite(y0),
                            point_interval_possibly_infinite(y1),
                        )
                        .transform(&self.real_to_im_y)
                        .inner(),
                    );
                    for p in &t_pixels {
                        self.im[p] = PixelState::True;
                    }
                }
            } else {
                for (ys, cond) in rs {
                    let cond = cond.result(self.rel.forms());
                    let dec = ys.decoration();

                    if dec >= Decoration::Def && cond.certainly_true() {
                        let im_y = self
                            .im_intervals(&ys)
                            .into_iter()
                            .fold(Interval::EMPTY, |acc, y| acc.convex_hull(y));

                        let t_pixels = self.true_pixels(px, im_y);
                        for p in &t_pixels {
                            self.im[p] = PixelState::True;
                        }
                    }
                }
            }
        }

        im_ys
            .into_iter()
            .map(|im_y| self.possibly_true_pixels(px, im_y))
            .collect()
    }

    /// Returns the region that corresponds to a subpixel block `b`.
    fn block_to_region(&self, b: &Block) -> Box1D {
        let pw = b.x.widthf();
        let px = b.x.index() as f64 * pw;
        Box1D::new(point_interval(px), point_interval(px + pw)).transform(&self.im_to_real_x)
    }

    /// Returns the region that corresponds to a pixel or superpixel block `b`.
    fn block_to_region_clipped(&self, b: &Block) -> Box1D {
        let pw = b.x.widthf();
        let px = b.x.index() as f64 * pw;
        Box1D::new(
            point_interval(px),
            point_interval((px + pw).min(self.im_width() as f64)),
        )
        .transform(&self.im_to_real_x)
    }

    fn eval_on_interval(rel: &mut Relation, x: Interval) -> (TupperIntervalSet, EvalResult) {
        rel.eval_explicit(x, None)
    }

    fn eval_on_point(
        rel: &mut Relation,
        x: f64,
        cache: Option<&mut EvalExplicitCache>,
    ) -> (TupperIntervalSet, EvalResult) {
        rel.eval_explicit(point_interval(x), cache)
    }

    /// Returns enclosures of `y` in image coordinates.
    fn im_intervals(&self, ys: &TupperIntervalSet) -> Vec<Interval> {
        ys.iter()
            .map(|y| {
                Box1D::new(
                    point_interval_possibly_infinite(y.x.inf()),
                    point_interval_possibly_infinite(y.x.sup()),
                )
                .transform(&self.real_to_im_y)
                .outer()
            })
            .collect::<Vec<_>>()
    }

    fn im_width(&self) -> u32 {
        if self.transpose {
            self.im.height()
        } else {
            self.im.width()
        }
    }

    fn is_any_pixel_uncertain(&self, pixels: &[PixelRange], front_block_index: usize) -> bool {
        pixels
            .iter()
            .flatten()
            .any(|p| self.im[p].is_uncertain(front_block_index))
    }

    /// Returns the smallest pixel-aligned interval that contains `x` in its interior.
    fn outer_pixels(x: Interval) -> Interval {
        if x.is_empty() {
            x
        } else {
            const TINY: Interval = const_interval!(-5e-324, 5e-324);
            let x = x + TINY;
            interval!(x.inf().floor(), x.sup().ceil()).unwrap()
        }
    }

    /// Returns the smallest pixel-aligned interval that contains `x`.
    fn pixels(x: Interval) -> Interval {
        if x.is_empty() {
            x
        } else {
            interval!(x.inf().floor(), x.sup().ceil()).unwrap()
        }
    }

    /// For the pixel-aligned region, returns the pixels in the region that are contained in the image.
    ///
    /// If [`self.transpose`] is `true`, the x and y components of the result are swapped.
    fn pixels_in_image(&self, r: &Region) -> PixelRange {
        let r = r.intersection(&self.im_region);
        if r.is_empty() {
            PixelRange::EMPTY
        } else {
            // If `r` is degenerate, the result is `PixelRange::EMPTY`.
            let mut x = r.x();
            let mut y = r.y();
            if self.transpose {
                swap(&mut x, &mut y);
            }
            PixelRange::new(
                PixelIndex::new(x.inf() as u32, self.im.height() - y.sup() as u32),
                PixelIndex::new(x.sup() as u32, self.im.height() - y.inf() as u32),
            )
        }
    }

    /// Given an interval in image coordinates that possibly contains a solution,
    /// returns the set of pixels that possibly contain solutions.
    ///
    /// If every member of `im_y` is a solution, all the pixels contain solutions.
    fn possibly_true_pixels(&self, px: Interval, im_y: Interval) -> PixelRange {
        use ExplicitRelationOp::*;

        if im_y.is_empty() {
            return PixelRange::EMPTY;
        }

        let im_y = match self.op {
            Eq => im_y,
            Ge | Gt => interval!(im_y.inf(), f64::INFINITY).unwrap(),
            Le | Lt => interval!(f64::NEG_INFINITY, im_y.sup()).unwrap(),
        };

        let py = match self.op {
            Eq | Ge | Le => Self::outer_pixels(im_y),
            Gt | Lt => Self::pixels(im_y),
        };

        self.pixels_in_image(&Region::new(px, py))
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

    /// Subdivides the block and appends the sub-blocks to `sub_bs`.
    /// Two sub-blocks are created at most.
    ///
    /// Precondition: `b.x.is_subdivisible()` is `true`.
    fn subdivide_on_x(&self, sub_bs: &mut Vec<Block>, b: &Block) {
        let [x0, x1] = b.x.subdivide();
        if b.x.is_superpixel() {
            sub_bs.push(Block { x: x0, ..*b });
            if x1.pixel_index() < self.im_width() {
                sub_bs.push(Block { x: x1, ..*b });
            }
        } else {
            sub_bs.extend([Block { x: x0, ..*b }, Block { x: x1, ..*b }]);
        }
    }

    /// Given an interval in pixel coordinates that contains a solution,
    /// returns the set of all pixels that contain solutions.
    ///
    /// Panics if `im_y` is empty.
    fn true_pixels(&self, px: Interval, im_y: Interval) -> PixelRange {
        use ExplicitRelationOp::*;

        assert!(!im_y.is_empty());

        let im_y = match self.op {
            Eq => im_y,
            Ge | Gt => interval!(im_y.sup(), f64::INFINITY).unwrap_or(Interval::EMPTY),
            Le | Lt => interval!(f64::NEG_INFINITY, im_y.inf()).unwrap_or(Interval::EMPTY),
        };

        let py = match self.op {
            Eq => {
                let py = Self::outer_pixels(im_y);
                if im_y.is_singleton() || py.wid() == 1.0 {
                    py
                } else {
                    Interval::EMPTY
                }
            }
            Ge | Le => Self::outer_pixels(im_y),
            Gt | Lt => Self::pixels(im_y),
        };

        self.pixels_in_image(&Region::new(px, py))
    }
}

impl Graph for Explicit {
    fn get_image(&self, im: &mut Image<Ternary>) {
        assert!(im.width() == self.im.width() && im.height() == self.im.height());
        for (s, dst) in self.im.pixels().copied().zip(im.pixels_mut()) {
            *dst = match s {
                PixelState::True => Ternary::True,
                _ if s.is_uncertain(self.block_queue.begin_index()) => Ternary::Uncertain,
                _ => Ternary::False,
            }
        }
    }

    fn get_statistics(&self) -> GraphingStatistics {
        GraphingStatistics {
            eval_count: self.rel.eval_count(),
            pixels_proven: self
                .im
                .pixels()
                .copied()
                .filter(|&s| !s.is_uncertain(self.block_queue.begin_index()))
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

    fn size_in_heap(&self) -> usize {
        self.im.size_in_heap() + self.block_queue.size_in_heap() + self.cache.size_in_heap()
    }
}
