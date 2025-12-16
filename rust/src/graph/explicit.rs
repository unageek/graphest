use crate::{
    ast::ExplicitRelOp,
    block::{Block, BlockQueue, Coordinate},
    eval_cache::{EvalCacheLevel, EvalExplicitCache},
    eval_result::EvalArgs,
    geom::{Box1D, Box2D, Transform, Transformation1D, TransformationMode},
    graph::{
        common::*, Graph, GraphingError, GraphingErrorKind, GraphingStatistics, Padding, Ternary,
    },
    image::{Image, PixelIndex, PixelRange},
    region::Region,
    relation::{Relation, RelationType},
    set_arg,
    traits::BytesAllocated,
    vars::{VarIndex, VarSet},
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
    op: ExplicitRelOp,
    transpose: bool,
    x_index: Option<VarIndex>,
    im: Image<PixelState>,
    block_queue: BlockQueue,
    im_region: Region,
    im_to_real_x: Transformation1D,
    real_to_im_y: Transformation1D,
    stats: GraphingStatistics,
    mem_limit: usize,
    no_cache: EvalExplicitCache,
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

        let vars = rel.vars();
        let op = match relation_type {
            RelationType::ExplicitFunctionOfX(op) | RelationType::ExplicitFunctionOfY(op) => op,
            _ => unreachable!(),
        };
        let transpose = matches!(relation_type, RelationType::ExplicitFunctionOfY(_));
        let im = Image::new(im_width, im_height);

        let (x_index, region, im_width, im_height, padding) = if transpose {
            (
                rel.var_indices().y,
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
            (rel.var_indices().x, region, im_width, im_height, padding)
        };

        let mut g = Self {
            rel,
            op,
            transpose,
            x_index,
            im,
            block_queue: BlockQueue::new(VarSet::X),
            im_region: Region::new(
                interval!(0.0, im_width as f64).unwrap(),
                interval!(0.0, im_height as f64).unwrap(),
            ),
            im_to_real_x: Transformation1D::new(
                [
                    point_interval(padding.left as f64),
                    point_interval((im_width - padding.right) as f64),
                ],
                [region.left(), region.right()],
                TransformationMode::Fast,
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
            no_cache: EvalExplicitCache::new(EvalCacheLevel::None, vars),
            cache: EvalExplicitCache::new(EvalCacheLevel::Full, vars),
        };

        let kx = (im_width as f64).log2().ceil() as i8;
        let b = Block {
            x: Coordinate::new(0, kx),
            ..Default::default()
        };
        g.block_queue.push_back(b);

        g
    }

    fn refine_impl(&mut self, duration: Duration, now: &Instant) -> Result<bool, GraphingError> {
        let mut sub_bs = vec![];
        let mut incomplete_pixels = vec![];
        let mut args = self.rel.create_args();
        while let Some(b) = self.block_queue.pop_front() {
            let bi = self.block_queue.begin_index() - 1;

            if !b.x.is_subpixel() {
                self.process_block(&b, &mut args, &mut incomplete_pixels);
            } else {
                self.process_subpixel_block(&b, &mut args, &mut incomplete_pixels);
            }

            if self.is_any_pixel_uncertain(&incomplete_pixels, bi) {
                if b.x.is_subdivisible() {
                    self.subdivide_x(&mut sub_bs, &b);
                    self.block_queue.extend(sub_bs.drain(..));
                    let last_bi = self.block_queue.end_index() - 1;
                    self.set_last_queued_block(&incomplete_pixels, last_bi, bi)?;
                } else {
                    self.set_undisprovable(&incomplete_pixels, bi);
                }
            }
            incomplete_pixels.clear();

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

    /// Tries to prove or disprove the existence of a solution in the block,
    /// and if that fails, appends pixels that may contain solutions to `incomplete_pixels`.
    ///
    /// Precondition: the block is either a pixel or a superpixel.
    fn process_block(
        &mut self,
        b: &Block,
        args: &mut EvalArgs,
        incomplete_pixels: &mut Vec<PixelRange>,
    ) {
        let x_up = self.region_clipped(b.x).outer();
        set_arg!(args, self.x_index, x_up);
        let (ys, cond) = self
            .rel
            .eval_explicit(args, &self.real_to_im_y, &mut self.no_cache)
            .clone();

        let px = {
            let begin = b.x.pixel_index();
            let end = (begin + b.x.width()).min(self.im_width());
            interval!(begin as f64, end as f64).unwrap()
        };

        let dec = ys.decoration();
        if dec >= Decoration::Def && cond.certainly_true() {
            let y = ys
                .iter()
                .fold(Interval::EMPTY, |acc, &y| acc.convex_hull(y.x));

            let pixels = self.possibly_true_pixels(px, y);
            let t_pixels = self.true_pixels(px, y);
            for p in &t_pixels {
                self.im[p] = PixelState::True;
            }
            if pixels == t_pixels {
                return;
            }
        } else if cond.certainly_false() {
            return;
        }

        incomplete_pixels.extend(ys.into_iter().map(|y| self.possibly_true_pixels(px, y.x)))
    }

    /// Tries to prove or disprove the existence of a solution in the block,
    /// and if that fails, appends pixels that may contain solutions to `incomplete_pixels`.
    ///
    /// Precondition: the block is a subpixel.
    fn process_subpixel_block(
        &mut self,
        b: &Block,
        args: &mut EvalArgs,
        incomplete_pixels: &mut Vec<PixelRange>,
    ) {
        let x_up = subpixel_outer_x(&self.region(b.x), b);
        set_arg!(args, self.x_index, x_up);
        let (ys, cond) = self
            .rel
            .eval_explicit(args, &self.real_to_im_y, &mut self.no_cache)
            .clone();

        let px = {
            let begin = b.x.pixel_index();
            let end = begin + 1;
            interval!(begin as f64, end as f64).unwrap()
        };

        let dec = ys.decoration();

        let x_dn = self.region(b.x.pixel()).inner();
        let inter = x_up.intersection(x_dn);

        if !inter.is_empty() {
            if dec >= Decoration::Def && cond.certainly_true() {
                let y = ys
                    .iter()
                    .fold(Interval::EMPTY, |acc, &y| acc.convex_hull(y.x));

                let pixels = self.possibly_true_pixels(px, y);
                let t_pixels = self.true_pixels(px, y);
                for p in &t_pixels {
                    self.im[p] = PixelState::True;
                }
                if pixels == t_pixels {
                    return;
                }
            } else if cond.certainly_false() {
                return;
            }

            // To dedup, points must be sorted.
            let rs = [inter.inf(), simple_fraction(inter), inter.sup()]
                .into_iter()
                .dedup()
                .map(|x| {
                    set_arg!(args, self.x_index, point_interval(x));
                    self.rel
                        .eval_explicit(args, &self.real_to_im_y, &mut self.cache)
                        .clone()
                })
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
                        .inner(),
                    );
                    for p in &t_pixels {
                        self.im[p] = PixelState::True;
                    }
                }
            } else {
                for (ys, cond) in rs {
                    let dec = ys.decoration();

                    if dec >= Decoration::Def && cond.certainly_true() {
                        let y = ys
                            .iter()
                            .fold(Interval::EMPTY, |acc, y| acc.convex_hull(y.x));

                        let t_pixels = self.true_pixels(px, y);
                        for p in &t_pixels {
                            self.im[p] = PixelState::True;
                        }
                    }
                }
            }
        }

        incomplete_pixels.extend(ys.into_iter().map(|y| self.possibly_true_pixels(px, y.x)))
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
    /// If [`Self::transpose`] is `true`, the x and y components of the result are swapped.
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
    /// If every member of `y` is a solution, all the pixels contain solutions.
    fn possibly_true_pixels(&self, px: Interval, y: Interval) -> PixelRange {
        use ExplicitRelOp::*;

        if y.is_empty() {
            return PixelRange::EMPTY;
        }

        let y = match self.op {
            Eq => y,
            Ge | Gt => interval!(y.inf(), f64::INFINITY).unwrap(),
            Le | Lt => interval!(f64::NEG_INFINITY, y.sup()).unwrap(),
        };

        let py = match self.op {
            Eq | Ge | Le => Self::outer_pixels(y),
            Gt | Lt => Self::pixels(y),
        };

        self.pixels_in_image(&Region::new(px, py))
    }

    /// Returns the region that corresponds to the given subpixel.
    fn region(&self, x: Coordinate) -> Box1D {
        let pw = x.widthf();
        let px = x.index() as f64 * pw;
        Box1D::new(point_interval(px), point_interval(px + pw)).transform(&self.im_to_real_x)
    }

    /// Returns the region that corresponds to the given pixel or superpixel.
    fn region_clipped(&self, x: Coordinate) -> Box1D {
        let pw = x.widthf();
        let px = x.index() as f64 * pw;
        Box1D::new(
            point_interval(px),
            point_interval((px + pw).min(self.im_width() as f64)),
        )
        .transform(&self.im_to_real_x)
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

    /// Subdivides `b.x` and appends the sub-blocks to `sub_bs`.
    /// Two sub-blocks are created at most.
    ///
    /// Precondition: `b.x.is_subdivisible()` is `true`.
    fn subdivide_x(&self, sub_bs: &mut Vec<Block>, b: &Block) {
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
    /// Panics if `y` is empty.
    fn true_pixels(&self, px: Interval, y: Interval) -> PixelRange {
        use ExplicitRelOp::*;

        assert!(!y.is_empty());

        let y = match self.op {
            Eq => y,
            Ge | Gt => interval!(y.sup(), f64::INFINITY).unwrap_or(Interval::EMPTY),
            Le | Lt => interval!(f64::NEG_INFINITY, y.inf()).unwrap_or(Interval::EMPTY),
        };

        let py = match self.op {
            Eq => {
                let py = Self::outer_pixels(y);
                if y.is_singleton() || py.wid() == 1.0 {
                    py
                } else {
                    Interval::EMPTY
                }
            }
            Ge | Le => Self::outer_pixels(y),
            Gt | Lt => Self::pixels(y),
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
            pixels_complete: self
                .im
                .pixels()
                .filter(|s| !s.is_uncertain(self.block_queue.begin_index()))
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

impl BytesAllocated for Explicit {
    fn bytes_allocated(&self) -> usize {
        self.im.bytes_allocated()
            + self.block_queue.bytes_allocated()
            + self.cache.bytes_allocated()
    }
}
