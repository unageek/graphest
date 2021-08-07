use super::Graph;
use crate::{
    block::{Block, BlockQueue, BlockQueueOptions, SubdivisionDir},
    eval_result::EvalResult,
    graph::{
        GraphingError, GraphingErrorKind, GraphingStatistics, InexactRegion, PixelState,
        QueuedBlockIndex, Region, Transform,
    },
    image::{Image, PixelIndex, PixelRegion},
    interval_set::{DecSignSet, SignSet},
    ops::StaticForm,
    relation::{EvalCache, EvalCacheLevel, Relation, RelationArgs, RelationType},
};
use image::{imageops, GrayAlphaImage, LumaA, Rgb, RgbImage};
use inari::{interval, Decoration, Interval};
use itertools::Itertools;
use std::{
    convert::TryFrom,
    time::{Duration, Instant},
};

pub struct Implicit {
    rel: Relation,
    forms: Vec<StaticForm>,
    relation_type: RelationType,
    im: Image<PixelState>,
    last_queued_blocks: Image<QueuedBlockIndex>,
    // Queue blocks that will be subdivided instead of the divided blocks to save memory.
    bs_to_subdivide: BlockQueue,
    // Affine transformation from pixel coordinates to real coordinates.
    transform: Transform,
    stats: GraphingStatistics,
    mem_limit: usize,
}

impl Implicit {
    pub fn new(
        rel: Relation,
        region: InexactRegion,
        im_width: u32,
        im_height: u32,
        mem_limit: usize,
    ) -> Self {
        assert!(matches!(
            rel.relation_type(),
            RelationType::FunctionOfX | RelationType::FunctionOfY | RelationType::Implicit
        ));

        let forms = rel.forms().clone();
        let relation_type = rel.relation_type();
        let has_n_theta = rel.has_n_theta();
        let has_t = rel.has_t();
        let mut g = Self {
            rel,
            forms,
            relation_type,
            im: Image::new(im_width, im_height),
            last_queued_blocks: Image::new(im_width, im_height),
            bs_to_subdivide: BlockQueue::new(BlockQueueOptions {
                store_xy: true,
                store_n_theta: has_n_theta,
                store_t: has_t,
                store_next_dir: has_n_theta || has_t,
            }),
            transform: Transform::new(
                region.width() / Self::point_interval(im_width as f64),
                region.height() / Self::point_interval(im_height as f64),
                region.left(),
                region.bottom(),
            ),
            stats: GraphingStatistics {
                pixels: im_width as usize * im_height as usize,
                pixels_proven: 0,
                eval_count: 0,
                time_elapsed: Duration::ZERO,
            },
            mem_limit,
        };
        let k = (im_width.max(im_height) as f64).log2().ceil() as i8;

        let t_range = g.rel.t_range();
        let mut bs = vec![Block::new(0, 0, k, k, Interval::ENTIRE, t_range)];

        if g.rel.has_n_theta() {
            let n_theta_range = g.rel.n_theta_range();
            bs = {
                let a = n_theta_range.inf();
                let b = n_theta_range.sup();
                let mid = n_theta_range.mid().round();
                vec![
                    interval!(a, a),
                    interval!(a, mid),
                    interval!(mid, mid),
                    interval!(mid, b),
                    interval!(b, b),
                ]
            }
            .into_iter()
            .filter_map(|n| n.ok()) // Remove invalid constructions, namely, [-∞, -∞] and [+∞, +∞].
            .filter(|n| n.wid() != 1.0)
            .dedup()
            .flat_map(|n| {
                bs.iter()
                    .map(|b| Block::new(b.x, b.y, b.kx, b.ky, n, b.t))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        }

        let last_block = bs.len() - 1;
        g.set_last_queued_block(&bs[last_block], last_block)
            .unwrap();
        for b in bs {
            g.bs_to_subdivide.push_back(b);
        }

        g
    }

    fn refine_impl(&mut self, timeout: Duration, now: &Instant) -> Result<bool, GraphingError> {
        let mut sub_bs = vec![];
        let mut incomplete_sub_bs = vec![];
        // Blocks are queued in the Morton order. Thanks to that, the caches should work efficiently.
        let mut cache_eval_on_region = EvalCache::new(EvalCacheLevel::PerAxis);
        let mut cache_eval_on_point = if self.rel.has_n_theta() || self.rel.has_t() {
            EvalCache::new(EvalCacheLevel::PerAxis)
        } else {
            EvalCache::new(EvalCacheLevel::Full)
        };
        while let Some((bi, b)) = self.bs_to_subdivide.pop_front() {
            match b.next_dir {
                SubdivisionDir::XY => self.subdivide_on_xy(&mut sub_bs, &b),
                SubdivisionDir::NTheta => Self::subdivide_on_n_theta(&mut sub_bs, &b),
                SubdivisionDir::T => Self::subdivide_on_t(&mut sub_bs, &b),
            }

            let n_sub_bs = sub_bs.len();
            for (sub_b, is_last_sibling) in sub_bs.drain(..) {
                let complete = if !sub_b.is_subpixel() {
                    self.refine_pixel(
                        &sub_b,
                        is_last_sibling,
                        QueuedBlockIndex::try_from(bi).unwrap(),
                        &mut cache_eval_on_region,
                    )
                } else {
                    if self.rel.has_n_theta() && !sub_b.n_theta.is_singleton() {
                        // Try finding a solution earlier.
                        let n = Self::point_interval(Self::simple_number(sub_b.n_theta));
                        self.refine_subpixel(
                            &Block::new(sub_b.x, sub_b.y, sub_b.kx, sub_b.ky, n, sub_b.t),
                            false,
                            0,
                            &mut cache_eval_on_region,
                            &mut cache_eval_on_point,
                        );
                    }
                    self.refine_subpixel(
                        &sub_b,
                        is_last_sibling,
                        QueuedBlockIndex::try_from(bi).unwrap(),
                        &mut cache_eval_on_region,
                        &mut cache_eval_on_point,
                    )
                };
                if !complete {
                    // We can't queue the block yet because we need to modify `sub_b.next_dir`
                    // after all sub-blocks are processed.
                    self.set_last_queued_block(
                        &sub_b,
                        self.bs_to_subdivide.next_back_index() + incomplete_sub_bs.len(),
                    )?;
                    incomplete_sub_bs.push(sub_b);
                }
            }

            let n_max = match b.next_dir {
                SubdivisionDir::XY => 4,
                SubdivisionDir::NTheta => 3,
                // Many bisection steps are performed to refine t.
                // So we deprioritize it.
                SubdivisionDir::T => usize::MAX,
            };
            let preferred_next_dir = if n_max * incomplete_sub_bs.len() <= n_sub_bs {
                // Subdivide in the same direction again.
                b.next_dir
            } else {
                // Subdivide in other direction.
                match b.next_dir {
                    SubdivisionDir::XY if self.rel.has_n_theta() => SubdivisionDir::NTheta,
                    SubdivisionDir::XY if self.rel.has_t() => SubdivisionDir::T,
                    SubdivisionDir::XY => SubdivisionDir::XY,
                    SubdivisionDir::NTheta if self.rel.has_t() => SubdivisionDir::T,
                    SubdivisionDir::NTheta => SubdivisionDir::XY,
                    SubdivisionDir::T => SubdivisionDir::XY,
                }
            };

            for mut sub_b in incomplete_sub_bs.drain(..) {
                sub_b.next_dir = if preferred_next_dir == SubdivisionDir::NTheta
                    && sub_b.is_subdivisible_on_n_theta()
                    || preferred_next_dir == SubdivisionDir::T && sub_b.is_subdivisible_on_t()
                {
                    preferred_next_dir
                } else if sub_b.is_subdivisible_on_xy() {
                    SubdivisionDir::XY
                } else if self.rel.has_n_theta() && sub_b.is_subdivisible_on_n_theta() {
                    SubdivisionDir::NTheta
                } else if self.rel.has_t() && sub_b.is_subdivisible_on_t() {
                    SubdivisionDir::T
                } else {
                    // Cannot subdivide in any direction.
                    assert!(sub_b.is_subpixel());
                    let pixel = b.pixel_index();
                    *self.im.get_mut(pixel) = PixelState::UncertainNeverFalse;
                    continue;
                };
                self.bs_to_subdivide.push_back(sub_b);
            }

            let mut clear_cache_and_retry = true;
            while self.im.size_in_heap()
                + self.last_queued_blocks.size_in_heap()
                + self.bs_to_subdivide.size_in_heap()
                + cache_eval_on_region.size_in_heap()
                + cache_eval_on_point.size_in_heap()
                > self.mem_limit
            {
                if clear_cache_and_retry {
                    cache_eval_on_region = EvalCache::new(EvalCacheLevel::PerAxis);
                    cache_eval_on_point = if self.rel.has_n_theta() || self.rel.has_t() {
                        EvalCache::new(EvalCacheLevel::PerAxis)
                    } else {
                        EvalCache::new(EvalCacheLevel::Full)
                    };
                    clear_cache_and_retry = false;
                } else {
                    return Err(GraphingError {
                        kind: GraphingErrorKind::ReachedMemLimit,
                    });
                }
            }

            if now.elapsed() > timeout {
                break;
            }
        }

        if self.bs_to_subdivide.is_empty() {
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
        b: &Block,
        block_index: usize,
    ) -> Result<(), GraphingError> {
        if let Ok(block_index) = QueuedBlockIndex::try_from(block_index) {
            if b.is_superpixel() {
                let pixels = {
                    let begin = b.pixel_index();
                    let end = PixelIndex::new(
                        (begin.x + b.width()).min(self.im.width()),
                        (begin.y + b.height()).min(self.im.height()),
                    );
                    PixelRegion::new(begin, end)
                };

                for p in pixels.iter() {
                    *self.last_queued_blocks.get_mut(p) = block_index;
                }
            } else {
                let p = b.pixel_index();
                *self.last_queued_blocks.get_mut(p) = block_index;
            }
            Ok(())
        } else {
            Err(GraphingError {
                kind: GraphingErrorKind::BlockIndexOverflow,
            })
        }
    }

    /// Refine the block and returns `true` if refinement is complete.
    ///
    /// Precondition: the block is either a pixel or a superpixel.
    fn refine_pixel(
        &mut self,
        b: &Block,
        b_is_last_sibling: bool,
        parent_block_index: QueuedBlockIndex,
        cache: &mut EvalCache,
    ) -> bool {
        let pixels = {
            let begin = b.pixel_index();
            let end = PixelIndex::new(
                (begin.x + b.width()).min(self.im.width()),
                (begin.y + b.height()).min(self.im.height()),
            );
            PixelRegion::new(begin, end)
        };

        if pixels.iter().all(|p| self.im.get(p) == PixelState::True) {
            // All pixels have already been proven to be true.
            return true;
        }

        let u_up = self.block_to_region_clipped(b).outer();
        let r_u_up = Self::eval_on_region(&mut self.rel, &u_up, b.n_theta, b.t, Some(cache));
        let is_true = r_u_up
            .map(|DecSignSet(ss, d)| ss == SignSet::ZERO && d >= Decoration::Def)
            .eval(&self.forms[..]);
        let is_false = !r_u_up
            .map(|DecSignSet(ss, _)| ss.contains(SignSet::ZERO))
            .eval(&self.forms[..]);

        if !(is_true || is_false) {
            return false;
        }

        for p in pixels.iter() {
            let state = self.im.get(p);
            assert_ne!(state, PixelState::False);
            if state == PixelState::True {
                // This pixel has already been proven to be true.
                continue;
            }

            if is_true {
                *self.im.get_mut(p) = PixelState::True;
            } else if is_false
                && b_is_last_sibling
                && self.last_queued_blocks.get(p) == parent_block_index
                && state != PixelState::UncertainNeverFalse
            {
                *self.im.get_mut(p) = PixelState::False;
            }
        }
        true
    }

    /// Refine the block and returns `true` if refinement is complete.
    ///
    /// Precondition: the block is a subpixel.
    fn refine_subpixel(
        &mut self,
        b: &Block,
        b_is_last_sibling: bool,
        parent_block_index: QueuedBlockIndex,
        cache_eval_on_region: &mut EvalCache,
        cache_eval_on_point: &mut EvalCache,
    ) -> bool {
        let pixel = b.pixel_index();
        let state = self.im.get(pixel);
        assert_ne!(state, PixelState::False);
        if state == PixelState::True {
            // This pixel has already been proven to be true.
            return true;
        }

        let u_up = self.block_to_region(b).subpixel_outer(b);
        let r_u_up = Self::eval_on_region(
            &mut self.rel,
            &u_up,
            b.n_theta,
            b.t,
            Some(cache_eval_on_region),
        );

        let p_dn = self.block_to_region(&b.pixel_block()).inner();
        let inter = u_up.intersection(&p_dn);

        // Save `locally_zero_mask` for later use (see the comment below).
        let locally_zero_mask =
            r_u_up.map(|DecSignSet(ss, d)| ss == SignSet::ZERO && d >= Decoration::Def);
        if locally_zero_mask.eval(&self.forms[..]) && !inter.is_empty() {
            // The relation is true everywhere in the subpixel, and the subpixel certainly overlaps
            // with the pixel. Therefore, the pixel contains a solution.
            *self.im.get_mut(pixel) = PixelState::True;
            return true;
        }
        if !r_u_up
            .map(|DecSignSet(ss, _)| ss.contains(SignSet::ZERO))
            .eval(&self.forms[..])
        {
            // The relation is false everywhere in the subpixel.
            if b_is_last_sibling
                && self.last_queued_blocks.get(pixel) == parent_block_index
                && state != PixelState::UncertainNeverFalse
            {
                *self.im.get_mut(pixel) = PixelState::False;
            }
            return true;
        }

        if inter.is_empty() {
            // We still need to refine the subpixel to show absence of solutions.
            return false;
        }

        // Evaluate the relation for some sample points within the inner bounds of the subpixel
        // and try proving existence of a solution in two ways:
        //
        // a. Test if the relation is true for any of the sample points.
        //    This is useful especially for plotting inequalities such as "lcm(x, y) ≤ 1".
        //
        // b. Use the intermediate value theorem.
        //    A note about `locally_zero_mask` (for plotting conjunction):
        //    Suppose we are plotting "y = sin(x) && x ≥ 0".
        //    If the conjunct "x ≥ 0" is true throughout the subpixel
        //    and "y - sin(x)" evaluates to `POS` for a sample point and `NEG` for another,
        //    we can conclude that there is a point within the subpixel where the entire relation holds.
        //    Such observation would not be possible by merely converting the relation to
        //    "|y - sin(x)| + |x ≥ 0 ? 0 : 1| = 0".
        let dac_mask = r_u_up.map(|DecSignSet(_, d)| d >= Decoration::Dac);

        let points = [
            (Self::simple_number(inter.0), Self::simple_number(inter.1)),
            (inter.0.inf(), inter.1.inf()), // bottom left
            (inter.0.sup(), inter.1.inf()), // bottom right
            (inter.0.inf(), inter.1.sup()), // top left
            (inter.0.sup(), inter.1.sup()), // top right
        ];

        let mut neg_mask = r_u_up.map(|_| false);
        let mut pos_mask = neg_mask.clone();
        for point in &points {
            let r = Self::eval_on_point(
                &mut self.rel,
                point.0,
                point.1,
                b.n_theta,
                b.t,
                Some(cache_eval_on_point),
            );

            // `ss` is nonempty if the decoration is ≥ `Def`, which will be ensured
            // by taking bitand with `dac_mask`.
            neg_mask |= r.map(|DecSignSet(ss, _)| (SignSet::NEG | SignSet::ZERO).contains(ss));
            pos_mask |= r.map(|DecSignSet(ss, _)| (SignSet::POS | SignSet::ZERO).contains(ss));

            if r.map(|DecSignSet(ss, d)| ss == SignSet::ZERO && d >= Decoration::Def)
                .eval(&self.forms[..])
                || (&(&neg_mask & &pos_mask) & &dac_mask)
                    .solution_certainly_exists(&self.forms[..], &locally_zero_mask)
            {
                // Found a solution.
                *self.im.get_mut(pixel) = PixelState::True;
                return true;
            }
        }

        false
    }

    fn eval_on_point(
        rel: &mut Relation,
        x: f64,
        y: f64,
        n_theta: Interval,
        t: Interval,
        cache: Option<&mut EvalCache>,
    ) -> EvalResult {
        rel.eval(
            &RelationArgs {
                x: Self::point_interval(x),
                y: Self::point_interval(y),
                n_theta,
                t,
            },
            cache,
        )
    }

    fn eval_on_region(
        rel: &mut Relation,
        r: &Region,
        n_theta: Interval,
        t: Interval,
        cache: Option<&mut EvalCache>,
    ) -> EvalResult {
        rel.eval(
            &RelationArgs {
                x: r.0,
                y: r.1,
                n_theta,
                t,
            },
            cache,
        )
    }

    /// Returns the region that corresponds to a subpixel block `b`.
    fn block_to_region(&self, b: &Block) -> InexactRegion {
        let pw = b.widthf();
        let ph = b.heightf();
        let px = b.x as f64 * pw;
        let py = b.y as f64 * ph;
        InexactRegion::new(
            Self::point_interval(px),
            Self::point_interval(px + pw),
            Self::point_interval(py),
            Self::point_interval(py + ph),
        )
        .transform(&self.transform)
    }

    /// Returns the region that corresponds to a pixel or superpixel block `b`.
    fn block_to_region_clipped(&self, b: &Block) -> InexactRegion {
        let pw = b.widthf();
        let ph = b.heightf();
        let px = b.x as f64 * pw;
        let py = b.y as f64 * ph;
        InexactRegion::new(
            Self::point_interval(px),
            Self::point_interval((px + pw).min(self.im.width() as f64)),
            Self::point_interval(py),
            Self::point_interval((py + ph).min(self.im.height() as f64)),
        )
        .transform(&self.transform)
    }

    fn point_interval(x: f64) -> Interval {
        interval!(x, x).unwrap()
    }

    /// Returns a number within the interval whose significand is as short as possible in the binary
    /// representation. For such inputs, arithmetic expressions are more likely to be evaluated
    /// exactly.
    ///
    /// Precondition: the interval is nonempty.
    fn simple_number(x: Interval) -> f64 {
        let a = x.inf();
        let b = x.sup();
        let a_bits = a.to_bits();
        let b_bits = b.to_bits();
        let diff = a_bits ^ b_bits;
        // The number of leading equal bits.
        let n = diff.leading_zeros();
        if n == 64 {
            return a;
        }
        // Set all bits from the MSB through the first differing bit.
        let mask = !0u64 << (64 - n - 1);
        if a <= 0.0 {
            f64::from_bits(a_bits & mask)
        } else {
            f64::from_bits(b_bits & mask)
        }
    }

    /// Subdivides the block both horizontally and vertically and appends the sub-blocks to `sub_bs`.
    /// Four sub-blocks are created at most.
    ///
    /// Precondition: `b.subdivide_on_xy()` is `true`.
    fn subdivide_on_xy(&self, sub_bs: &mut Vec<(Block, bool)>, b: &Block) {
        if b.is_superpixel() {
            let x0 = 2 * b.x;
            let y0 = 2 * b.y;
            let x1 = x0 + 1;
            let y1 = y0 + 1;
            let kx = b.kx - 1;
            let ky = b.ky - 1;
            let b00 = Block::new(x0, y0, kx, ky, b.n_theta, b.t);
            let b00_width = b00.width();
            let b00_height = b00.height();
            sub_bs.push((b00, true));
            if y1 * b00_height < self.im.height() {
                sub_bs.push((Block::new(x0, y1, kx, ky, b.n_theta, b.t), true));
            }
            if x1 * b00_width < self.im.width() {
                sub_bs.push((Block::new(x1, y0, kx, ky, b.n_theta, b.t), true));
            }
            if x1 * b00_width < self.im.width() && y1 * b00_height < self.im.height() {
                sub_bs.push((Block::new(x1, y1, kx, ky, b.n_theta, b.t), true));
            }
        } else {
            match self.relation_type {
                RelationType::FunctionOfX => {
                    // Subdivide only horizontally.
                    let x0 = 2 * b.x;
                    let x1 = x0 + 1;
                    let y = b.y;
                    let kx = b.kx - 1;
                    let ky = b.ky;
                    sub_bs.push((Block::new(x0, y, kx, ky, b.n_theta, b.t), false));
                    sub_bs.push((Block::new(x1, y, kx, ky, b.n_theta, b.t), true));
                }
                RelationType::FunctionOfY => {
                    // Subdivide only vertically.
                    let x = b.x;
                    let y0 = 2 * b.y;
                    let y1 = y0 + 1;
                    let kx = b.kx;
                    let ky = b.ky - 1;
                    sub_bs.push((Block::new(x, y0, kx, ky, b.n_theta, b.t), false));
                    sub_bs.push((Block::new(x, y1, kx, ky, b.n_theta, b.t), true));
                }
                _ => {
                    let x0 = 2 * b.x;
                    let y0 = 2 * b.y;
                    let x1 = x0 + 1;
                    let y1 = y0 + 1;
                    let kx = b.kx - 1;
                    let ky = b.ky - 1;
                    sub_bs.push((Block::new(x0, y0, kx, ky, b.n_theta, b.t), false));
                    sub_bs.push((Block::new(x1, y0, kx, ky, b.n_theta, b.t), false));
                    sub_bs.push((Block::new(x0, y1, kx, ky, b.n_theta, b.t), false));
                    sub_bs.push((Block::new(x1, y1, kx, ky, b.n_theta, b.t), true));
                }
            }
        }
    }

    /// Subdivides `b.n_theta` and appends the sub-blocks to `sub_bs`.
    /// Three sub-blocks are created at most.
    ///
    /// Preconditions:
    ///
    /// - `b.is_subdivisible_on_n_theta()` is `true`.
    /// - `b.n_theta` is a subset of either \[-∞, 0\] or \[0, +∞\].
    fn subdivide_on_n_theta(sub_bs: &mut Vec<(Block, bool)>, b: &Block) {
        const MULT: f64 = 2.0; // The optimal value may depend on the relation.
        let n = b.n_theta;
        let na = n.inf();
        let nb = n.sup();
        let mid = if na == f64::NEG_INFINITY {
            (MULT * nb).max(f64::MIN).min(-1.0)
        } else if nb == f64::INFINITY {
            (MULT * na).max(1.0).min(f64::MAX)
        } else {
            n.mid().round()
        };
        let ns = [
            interval!(na, mid).unwrap(),
            interval!(mid, mid).unwrap(),
            interval!(mid, nb).unwrap(),
        ];
        // Any interval with width 1 can be discarded since its endpoints are already processed
        // as point intervals and there are no integers in between them.
        sub_bs.extend(
            ns.iter()
                .filter(|n| n.wid() != 1.0)
                .map(|&n| (Block::new(b.x, b.y, b.kx, b.ky, n, b.t), false)),
        );
        if let Some(last) = sub_bs.last_mut() {
            last.1 = true;
        }
    }

    /// Subdivides `b.t` and appends the sub-blocks to `sub_bs`.
    /// Four sub-blocks are created at most.
    ///
    /// Precondition: `b.is_subdivisible_on_t()` is `true`.
    fn subdivide_on_t(sub_bs: &mut Vec<(Block, bool)>, b: &Block) {
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
        let ((t1, t2), (t3, t4)) = (bisect(t1), bisect(t2));
        sub_bs.extend(
            [t1, t2, t3, t4]
                .iter()
                .filter(|t| !t.is_singleton())
                .map(|&t| (Block::new(b.x, b.y, b.kx, b.ky, b.n_theta, t), false)),
        );
        if let Some(last) = sub_bs.last_mut() {
            last.1 = true;
        }
    }
}

impl Graph for Implicit {
    fn get_gray_alpha_image(&self, im: &mut GrayAlphaImage) {
        assert!(im.width() == self.im.width() && im.height() == self.im.height());
        for (src, dst) in self.im.pixels().copied().zip(im.pixels_mut()) {
            *dst = match src {
                PixelState::True => LumaA([0, 255]),
                PixelState::False => LumaA([0, 0]),
                _ => LumaA([0, 128]),
            }
        }
        imageops::flip_vertical_in_place(im);
    }

    fn get_image(&self, im: &mut RgbImage) {
        assert!(im.width() == self.im.width() && im.height() == self.im.height());
        for (src, dst) in self.im.pixels().copied().zip(im.pixels_mut()) {
            *dst = match src {
                PixelState::True => Rgb([0, 0, 0]),
                PixelState::False => Rgb([255, 255, 255]),
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
                .filter(|&s| s == PixelState::False || s == PixelState::True)
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