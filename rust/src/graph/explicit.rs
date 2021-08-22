use crate::{
    block::{Block, BlockQueue, BlockQueueOptions},
    eval_result::EvalResult,
    graph::{
        common::{
            point_interval, point_interval_possibly_infinite, simple_fraction, PixelState,
            QueuedBlockIndex,
        },
        Graph, GraphingError, GraphingErrorKind, GraphingStatistics,
    },
    image::{Image, PixelIndex, PixelRegion},
    interval_set::{DecSignSet, SignSet, TupperIntervalSet},
    ops::StaticForm,
    region::{InexactRegion, Region, Transform},
    relation::{EvalExplicitCache, Relation, RelationType},
};
use image::{imageops, ImageBuffer, Pixel};
use inari::{const_interval, interval, Decoration, Interval};
use std::{
    convert::TryFrom,
    mem::swap,
    ops::{Deref, DerefMut},
    time::{Duration, Instant},
};

/// The graphing algorithm for explicit relations.
pub struct Explicit {
    rel: Relation,
    forms: Vec<StaticForm>,
    transpose: bool,
    im: Image<PixelState>,
    last_queued_blocks: Image<QueuedBlockIndex>,
    block_queue: BlockQueue,
    im_region: Region,
    im_to_real_x: Transform,
    real_to_im_x: Transform,
    real_to_im_y: Transform,
    stats: GraphingStatistics,
    mem_limit: usize,
    cache: EvalExplicitCache,
}

impl Explicit {
    pub fn new(
        rel: Relation,
        region: InexactRegion,
        im_width: u32,
        im_height: u32,
        mem_limit: usize,
    ) -> Self {
        const ONE: Interval = const_interval!(1.0, 1.0);

        assert!(matches!(
            rel.relation_type(),
            RelationType::ExplicitFunctionOfX | RelationType::ExplicitFunctionOfY
        ));

        let forms = rel.forms().clone();
        let transpose = rel.relation_type() == RelationType::ExplicitFunctionOfY;
        let im = Image::new(im_width, im_height);
        let last_queued_blocks = Image::new(im_width, im_height);

        let (im_width, im_height) = if transpose {
            (im_height, im_width)
        } else {
            (im_width, im_height)
        };
        let im_width_interval = point_interval(im_width as f64);
        let im_height_interval = point_interval(im_height as f64);
        let mut g = Self {
            rel,
            forms,
            transpose,
            im,
            last_queued_blocks,
            block_queue: BlockQueue::new(BlockQueueOptions {
                store_xy: true,
                ..Default::default()
            }),
            im_region: Region::new(
                interval!(0.0, im_width as f64).unwrap(),
                interval!(0.0, im_height as f64).unwrap(),
            ),
            im_to_real_x: Transform::with_predivision_factors(
                (region.width(), im_width_interval),
                region.left(),
                (ONE, ONE),
                const_interval!(0.0, 0.0),
            ),
            real_to_im_x: {
                Transform::with_predivision_factors(
                    (im_width_interval, region.width()),
                    -im_width_interval * (region.left() / region.width()),
                    (ONE, ONE),
                    const_interval!(0.0, 0.0),
                )
            },
            real_to_im_y: {
                Transform::with_predivision_factors(
                    (ONE, ONE),
                    const_interval!(0.0, 0.0),
                    (im_height_interval, region.height()),
                    -im_height_interval * (region.bottom() / region.height()),
                )
            },
            stats: GraphingStatistics {
                pixels: im_width as usize * im_height as usize,
                pixels_proven: 0,
                eval_count: 0,
                time_elapsed: Duration::ZERO,
            },
            mem_limit,
            cache: Default::default(),
        };

        let kx = (im_width as f64).log2().ceil() as i8;
        let b = Block::new(0, 0, kx, 0, Interval::ENTIRE, Interval::ENTIRE);
        g.block_queue.push_back(b);

        g
    }

    fn refine_impl(&mut self, duration: Duration, now: &Instant) -> Result<bool, GraphingError> {
        let mut sub_bs = vec![];
        while let Some(b) = self.block_queue.pop_front() {
            let bi = self.block_queue.begin_index() - 1;

            let incomplete_pixels = if !b.is_subpixel() {
                self.process_block(&b)
            } else {
                self.process_subpixel_block(&b)
            };

            if self.is_any_pixel_uncertain(&incomplete_pixels, bi) {
                if b.is_xy_subdivisible() {
                    self.subdivide_on_x(&mut sub_bs, &b);
                    self.block_queue.extend(sub_bs.drain(..));
                    let last_bi = self.block_queue.end_index() - 1;
                    self.set_last_queued_block(&incomplete_pixels, last_bi, bi)?;
                } else {
                    self.set_uncertain_never_false(&incomplete_pixels, bi);
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

    /// Tries to prove or disprove the existence of a solution in the block
    /// and if it is unsuccessful, returns pixels that possibly contains a solution.
    ///
    /// Precondition: the block is either a pixel or a superpixel.
    fn process_block(&mut self, b: &Block) -> Vec<PixelRegion> {
        let x = {
            let u_up = self.block_to_region_clipped(b).outer();
            u_up.x()
        };
        let (ys, cond) = Self::eval_on_interval(&mut self.rel, x);

        let px = {
            let begin = b.pixel_index().x;
            let end = (begin + b.width()).min(self.im_width());
            interval!(begin as f64, end as f64).unwrap()
        };
        let pys = self
            .im_intervals(&ys)
            .into_iter()
            .map(Self::outer_pixels)
            .collect::<Vec<_>>();

        let cond_is_true = cond
            .map(|DecSignSet(ss, d)| ss == SignSet::ZERO && d >= Decoration::Def)
            .eval(&self.forms[..]);
        let cond_is_false = !cond
            .map(|DecSignSet(ss, _)| ss.contains(SignSet::ZERO))
            .eval(&self.forms[..]);

        let dec = ys.decoration();
        if dec >= Decoration::Def && cond_is_true {
            let py = pys
                .iter()
                .fold(Interval::EMPTY, |acc, &y| acc.convex_hull(y));

            if py.wid() == 1.0 {
                // y is interior to a single row.
                for p in &self.pixels_in_image(&Region::new(px, py)) {
                    self.im[p] = PixelState::True;
                }
                return vec![];
            }
        } else if cond_is_false {
            return vec![];
        }

        pys.into_iter()
            .map(|py| self.pixels_in_image(&Region::new(px, py)))
            .filter(|ps| ps.iter().any(|p| self.im[p] != PixelState::True))
            .collect()
    }

    /// Tries to prove or disprove the existence of a solution in the block
    /// and if it is unsuccessful, returns pixels that possibly contains a solution.
    ///
    /// Precondition: the block is a subpixel.
    fn process_subpixel_block(&mut self, b: &Block) -> Vec<PixelRegion> {
        let x_up = {
            let u_up = self.block_to_region(b).subpixel_outer(b);
            u_up.x()
        };
        let x_dn = {
            let p_dn = self.block_to_region(&b.pixel_block()).inner();
            x_up.intersection(p_dn.x())
        };
        let (ys, cond) = Self::eval_on_interval(&mut self.rel, x_up);

        let px = {
            let begin = b.pixel_index().x;
            let end = begin + 1;
            interval!(begin as f64, end as f64).unwrap()
        };
        let pys = self
            .im_intervals(&ys)
            .into_iter()
            .map(Self::outer_pixels)
            .collect::<Vec<_>>();

        let cond_is_true = cond
            .map(|DecSignSet(ss, d)| ss == SignSet::ZERO && d >= Decoration::Def)
            .eval(&self.forms[..]);
        let cond_is_false = !cond
            .map(|DecSignSet(ss, _)| ss.contains(SignSet::ZERO))
            .eval(&self.forms[..]);

        let dec = ys.decoration();
        if dec >= Decoration::Def && cond_is_true {
            let py = pys
                .iter()
                .fold(Interval::EMPTY, |acc, &y| acc.convex_hull(y));

            if py.wid() == 1.0 {
                // y is interior to a single row.
                for p in &self.pixels_in_image(&Region::new(px, py)) {
                    self.im[p] = PixelState::True;
                }
                return vec![];
            } else if dec >= Decoration::Dac && !x_dn.is_empty() {
                assert_eq!(pys.len(), 1);
                let mut y1 = {
                    let (y, _) =
                        Self::eval_on_point(&mut self.rel, x_dn.inf(), Some(&mut self.cache));
                    assert_eq!(y.len(), 1);
                    y.iter().next().unwrap().x
                };
                let mut y2 = {
                    let (y, _) =
                        Self::eval_on_point(&mut self.rel, x_dn.sup(), Some(&mut self.cache));
                    assert_eq!(y.len(), 1);
                    y.iter().next().unwrap().x
                };

                let mut py12 = Interval::EMPTY;
                if y2.precedes(y1) {
                    swap(&mut y1, &mut y2);
                }
                if y1.precedes(y2) {
                    py12 = Self::outer_pixels(
                        InexactRegion::new(
                            const_interval!(0.0, 0.0),
                            const_interval!(1.0, 1.0),
                            y1,
                            y2,
                        )
                        .transform(&self.real_to_im_y)
                        .inner()
                        .y(),
                    );
                }

                for p in &self.pixels_in_image(&Region::new(px, py12)) {
                    self.im[p] = PixelState::True;
                }

                if py12 == py {
                    return vec![];
                }
            }
        } else if cond_is_false {
            return vec![];
        }

        // Try to locate true pixels.
        if !x_dn.is_empty() {
            let x = simple_fraction(x_dn);
            let px = {
                let x = point_interval(x);
                let im_x =
                    InexactRegion::new(x, x, const_interval!(0.0, 0.0), const_interval!(1.0, 1.0))
                        .transform(&self.real_to_im_x)
                        .outer()
                        .x();
                if im_x.is_singleton() || Self::outer_pixels(im_x).wid() == 1.0 {
                    Some(Self::outer_pixels(im_x))
                } else {
                    None
                }
            };

            if let Some(px) = px {
                let (ys, cond) = Self::eval_on_point(&mut self.rel, x, Some(&mut self.cache));
                let im_y = self
                    .im_intervals(&ys)
                    .into_iter()
                    .fold(Interval::EMPTY, |acc, y| acc.convex_hull(y));

                let cond_is_true = cond
                    .map(|DecSignSet(ss, d)| ss == SignSet::ZERO && d >= Decoration::Def)
                    .eval(&self.forms[..]);
                let dec = ys.decoration();

                if dec >= Decoration::Def && cond_is_true {
                    let py = Self::outer_pixels(im_y);
                    if im_y.is_singleton() || py.wid() == 1.0 {
                        for p in &self.pixels_in_image(&Region::new(px, py)) {
                            self.im[p] = PixelState::True;
                        }
                    }
                }
            }
        }

        pys.into_iter()
            .map(|py| self.pixels_in_image(&Region::new(px, py)))
            .filter(|ps| ps.iter().any(|p| self.im[p] != PixelState::True))
            .collect()
    }

    /// Returns the region that corresponds to a subpixel block `b`.
    fn block_to_region(&self, b: &Block) -> InexactRegion {
        let pw = b.widthf();
        let px = b.x as f64 * pw;
        InexactRegion::new(
            point_interval(px),
            point_interval(px + pw),
            const_interval!(0.0, 0.0),
            const_interval!(1.0, 1.0),
        )
        .transform(&self.im_to_real_x)
    }

    /// Returns the region that corresponds to a pixel or superpixel block `b`.
    fn block_to_region_clipped(&self, b: &Block) -> InexactRegion {
        let pw = b.widthf();
        let px = b.x as f64 * pw;
        InexactRegion::new(
            point_interval(px),
            point_interval((px + pw).min(self.im_width() as f64)),
            const_interval!(0.0, 0.0),
            const_interval!(1.0, 1.0),
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
                InexactRegion::new(
                    const_interval!(0.0, 0.0),
                    const_interval!(1.0, 1.0),
                    point_interval_possibly_infinite(y.x.inf()),
                    point_interval_possibly_infinite(y.x.sup()),
                )
                .transform(&self.real_to_im_y)
                .outer()
                .y()
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

    fn is_any_pixel_uncertain(&self, pixels: &[PixelRegion], parent_block_index: usize) -> bool {
        pixels.iter().flatten().any(|p| {
            let s = self.im[p];
            let bi = self.last_queued_blocks[p];
            !(s == PixelState::True
                || s == PixelState::Uncertain && (bi as usize) < parent_block_index)
        })
    }

    /// Returns the smallest pixel-aligned interval that contains `x` in its interior.
    fn outer_pixels(x: Interval) -> Interval {
        const TINY: Interval = const_interval!(-5e-324, 5e-324);
        let x = x + TINY;
        interval!(x.inf().floor(), x.sup().ceil()).unwrap()
    }

    /// For the pixel-aligned region, returns the pixels in the region that are contained in the image.
    ///
    /// If [`self.transpose`] is `true`, the x and y components of the result are swapped.
    fn pixels_in_image(&self, r: &Region) -> PixelRegion {
        let r = r.intersection(&self.im_region);
        if r.is_empty() {
            PixelRegion::EMPTY
        } else {
            // If `r` is degenerate, the result is `PixelRegion::EMPTY`.
            let mut x = r.x();
            let mut y = r.y();
            if self.transpose {
                swap(&mut x, &mut y);
            }
            PixelRegion::new(
                PixelIndex::new(x.inf() as u32, y.inf() as u32),
                PixelIndex::new(x.sup() as u32, y.sup() as u32),
            )
        }
    }

    fn set_last_queued_block(
        &mut self,
        pixels: &[PixelRegion],
        block_index: usize,
        parent_block_index: usize,
    ) -> Result<(), GraphingError> {
        if let Ok(block_index) = QueuedBlockIndex::try_from(block_index) {
            for p in pixels.iter().flatten() {
                // Check if the pixel is not already revealed to be false.
                // This is required because some interval functions are not monotonic,
                // i.e., then can return a wider interval for a narrower (refined) input.
                if self.last_queued_blocks[p] as usize >= parent_block_index {
                    self.last_queued_blocks[p] = block_index;
                }
            }
            Ok(())
        } else {
            Err(GraphingError {
                kind: GraphingErrorKind::BlockIndexOverflow,
            })
        }
    }

    fn set_uncertain_never_false(&mut self, pixels: &[PixelRegion], parent_block_index: usize) {
        for p in pixels.iter().flatten() {
            if self.im[p] == PixelState::Uncertain
                // Check if the pixel is not already revealed to be false.
                && self.last_queued_blocks[p] as usize == parent_block_index
            {
                self.im[p] = PixelState::UncertainNeverFalse;
            }
        }
    }

    /// Returns the amount of memory allocated by `self` in bytes.
    fn size_in_heap(&self) -> usize {
        self.im.size_in_heap()
            + self.last_queued_blocks.size_in_heap()
            + self.block_queue.size_in_heap()
            + self.cache.size_in_heap()
    }

    /// Subdivides the block and appends the sub-blocks to `sub_bs`.
    /// Two sub-blocks are created at most.
    ///
    /// Precondition: `b.subdivide_on_xy()` is `true`.
    fn subdivide_on_x(&self, sub_bs: &mut Vec<Block>, b: &Block) {
        let x0 = 2 * b.x;
        let x1 = x0 + 1;
        let kx = b.kx - 1;
        if b.is_superpixel() {
            let b0 = Block::new(x0, 0, kx, 0, b.n_theta, b.t);
            let b0_width = b0.width();
            sub_bs.push(b0);
            if x1 * b0_width < self.im_width() {
                sub_bs.push(Block::new(x1, 0, kx, 0, b.n_theta, b.t));
            }
        } else {
            sub_bs.push(Block::new(x0, 0, kx, 0, b.n_theta, b.t));
            sub_bs.push(Block::new(x1, 0, kx, 0, b.n_theta, b.t));
        }
    }
}

impl Graph for Explicit {
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
                    s == PixelState::True
                        || s == PixelState::Uncertain
                            && (bi as usize) < self.block_queue.begin_index()
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
