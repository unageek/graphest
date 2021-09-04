use crate::{
    block::{Block, BlockQueue, BlockQueueOptions, SubdivisionDir},
    eval_result::EvalResult,
    geom::{Box2D, Transform2D},
    graph::{
        common::{point_interval, simple_fraction, subpixel_outer, PixelState, QueuedBlockIndex},
        Graph, GraphingError, GraphingErrorKind, GraphingStatistics,
    },
    image::{Image, PixelIndex, PixelRange},
    interval_set::{DecSignSet, SignSet},
    ops::StaticForm,
    region::Region,
    relation::{EvalCache, EvalCacheLevel, Relation, RelationArgs, RelationType},
};
use image::{imageops, ImageBuffer, Pixel};
use inari::{interval, Decoration, Interval};
use itertools::Itertools;
use std::{
    convert::TryFrom,
    ops::{Deref, DerefMut},
    time::{Duration, Instant},
};

/// The graphing algorithm for implicit relations.
pub struct Implicit {
    rel: Relation,
    forms: Vec<StaticForm>,
    im: Image<PixelState>,
    // Queue blocks that will be subdivided instead of the divided blocks to save memory.
    bs_to_subdivide: BlockQueue,
    // Affine transformation from image coordinates to real coordinates.
    im_to_real: Transform2D,
    stats: GraphingStatistics,
    mem_limit: usize,
    cache_eval_on_region: EvalCache,
    cache_eval_on_point: EvalCache,
}

impl Implicit {
    pub fn new(
        rel: Relation,
        region: Box2D,
        im_width: u32,
        im_height: u32,
        mem_limit: usize,
    ) -> Self {
        assert_eq!(rel.relation_type(), RelationType::Implicit);

        let forms = rel.forms().clone();
        let has_n_theta = rel.has_n_theta();
        let has_t = rel.has_t();
        let mut g = Self {
            rel,
            forms,
            im: Image::new(im_width, im_height),
            bs_to_subdivide: BlockQueue::new(BlockQueueOptions {
                store_xy: true,
                store_n_theta: has_n_theta,
                store_t: has_t,
                store_next_dir: has_n_theta || has_t,
            }),
            im_to_real: Transform2D::new(
                region.width() / point_interval(im_width as f64),
                region.left(),
                region.height() / point_interval(im_height as f64),
                region.bottom(),
            ),
            stats: GraphingStatistics {
                eval_count: 0,
                pixels: im_width as usize * im_height as usize,
                pixels_proven: 0,
                time_elapsed: Duration::ZERO,
            },
            mem_limit,
            cache_eval_on_region: EvalCache::new(EvalCacheLevel::PerAxis),
            cache_eval_on_point: if has_n_theta || has_t {
                EvalCache::new(EvalCacheLevel::PerAxis)
            } else {
                EvalCache::new(EvalCacheLevel::Full)
            },
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
        g.set_last_queued_block(&bs[last_block], last_block, None)
            .unwrap();
        for b in bs {
            g.bs_to_subdivide.push_back(b);
        }

        g
    }

    fn refine_impl(&mut self, duration: Duration, now: &Instant) -> Result<bool, GraphingError> {
        let mut sub_bs = vec![];
        let mut incomplete_sub_bs = vec![];
        while let Some(b) = self.bs_to_subdivide.pop_front() {
            let bi = self.bs_to_subdivide.begin_index() - 1;
            match b.next_dir {
                SubdivisionDir::XY => self.subdivide_xy(&mut sub_bs, &b),
                SubdivisionDir::NTheta => Self::subdivide_n_theta(&mut sub_bs, &b),
                SubdivisionDir::T => Self::subdivide_t(&mut sub_bs, &b),
            }

            let n_sub_bs = sub_bs.len();
            for sub_b in sub_bs.drain(..) {
                let complete = if !sub_b.is_subpixel() {
                    self.process_block(&sub_b)
                } else {
                    if self.rel.has_n_theta() && !sub_b.n_theta.is_singleton() {
                        // Try finding a solution earlier.
                        let n_theta = point_interval(simple_fraction(sub_b.n_theta));
                        self.process_subpixel_block(&Block::new(
                            sub_b.x, sub_b.y, sub_b.kx, sub_b.ky, n_theta, sub_b.t,
                        ));
                    }
                    self.process_subpixel_block(&sub_b)
                };
                if !complete {
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
                    && sub_b.is_n_theta_subdivisible()
                    || preferred_next_dir == SubdivisionDir::T && sub_b.is_t_subdivisible()
                {
                    preferred_next_dir
                } else if sub_b.is_xy_subdivisible() {
                    SubdivisionDir::XY
                } else if self.rel.has_n_theta() && sub_b.is_n_theta_subdivisible() {
                    SubdivisionDir::NTheta
                } else if self.rel.has_t() && sub_b.is_t_subdivisible() {
                    SubdivisionDir::T
                } else {
                    // Cannot subdivide in any direction.
                    assert!(sub_b.is_subpixel());
                    let pixel = b.pixel_index();
                    self.im[pixel] = PixelState::Uncertain(None);
                    continue;
                };

                self.bs_to_subdivide.push_back(sub_b.clone());
                let last_bi = self.bs_to_subdivide.end_index() - 1;
                self.set_last_queued_block(&sub_b, last_bi, Some(bi))?;
            }

            let mut clear_cache_and_retry = true;
            while self.size_in_heap() > self.mem_limit {
                if clear_cache_and_retry {
                    self.cache_eval_on_region.clear();
                    self.cache_eval_on_point.clear();
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
    /// and returns `true` if it is successful.
    ///
    /// Precondition: the block is either a pixel or a superpixel.
    fn process_block(&mut self, b: &Block) -> bool {
        let pixels = {
            let begin = b.pixel_index();
            let end = PixelIndex::new(
                (begin.x + b.width()).min(self.im.width()),
                (begin.y + b.height()).min(self.im.height()),
            );
            PixelRange::new(begin, end)
        };

        if pixels.iter().all(|p| self.im[p] == PixelState::True) {
            // All pixels have already been proven to be true.
            return true;
        }

        let u_up = self.block_to_region_clipped(b).outer();
        let r_u_up = Self::eval_on_region(
            &mut self.rel,
            &u_up,
            b.n_theta,
            b.t,
            Some(&mut self.cache_eval_on_region),
        );
        let is_true = r_u_up
            .map(|DecSignSet(ss, d)| ss == SignSet::ZERO && d >= Decoration::Def)
            .eval(&self.forms[..]);
        let is_false = !r_u_up
            .map(|DecSignSet(ss, _)| ss.contains(SignSet::ZERO))
            .eval(&self.forms[..]);

        if is_true {
            for p in pixels.iter() {
                self.im[p] = PixelState::True;
            }
            return true;
        }

        is_false
    }

    /// Tries to prove or disprove the existence of a solution in the block
    /// and returns `true` if it is successful.
    ///
    /// Precondition: the block is a subpixel.
    fn process_subpixel_block(&mut self, b: &Block) -> bool {
        let p = b.pixel_index();
        if self.im[p] == PixelState::True {
            // This pixel has already been proven to be true.
            return true;
        }

        let u_up = subpixel_outer(&self.block_to_region(b), b);
        let r_u_up = Self::eval_on_region(
            &mut self.rel,
            &u_up,
            b.n_theta,
            b.t,
            Some(&mut self.cache_eval_on_region),
        );

        let p_dn = self.block_to_region(&b.pixel_block()).inner();
        let inter = u_up.intersection(&p_dn);

        // Save `locally_zero_mask` for later use (see the comment below).
        let locally_zero_mask =
            r_u_up.map(|DecSignSet(ss, d)| ss == SignSet::ZERO && d >= Decoration::Def);
        if locally_zero_mask.eval(&self.forms[..]) && !inter.is_empty() {
            // The relation is true everywhere in the subpixel, and the subpixel certainly overlaps
            // with the pixel. Therefore, the pixel contains a solution.
            self.im[p] = PixelState::True;
            return true;
        }
        if !r_u_up
            .map(|DecSignSet(ss, _)| ss.contains(SignSet::ZERO))
            .eval(&self.forms[..])
        {
            // The relation is false everywhere in the subpixel.
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

        let points = {
            let x = inter.x();
            let y = inter.y();
            [
                (simple_fraction(x), simple_fraction(y)),
                (x.inf(), y.inf()), // bottom left
                (x.sup(), y.inf()), // bottom right
                (x.inf(), y.sup()), // top left
                (x.sup(), y.sup()), // top right
            ]
        };

        let mut neg_mask = r_u_up.map(|_| false);
        let mut pos_mask = neg_mask.clone();
        for point in &points {
            let r = Self::eval_on_point(
                &mut self.rel,
                point.0,
                point.1,
                b.n_theta,
                b.t,
                Some(&mut self.cache_eval_on_point),
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
                self.im[p] = PixelState::True;
                return true;
            }
        }

        false
    }

    /// Returns the region that corresponds to a subpixel block `b`.
    fn block_to_region(&self, b: &Block) -> Box2D {
        let pw = b.widthf();
        let ph = b.heightf();
        let px = b.x as f64 * pw;
        let py = b.y as f64 * ph;
        Box2D::new(
            point_interval(px),
            point_interval(px + pw),
            point_interval(py),
            point_interval(py + ph),
        )
        .transform(&self.im_to_real)
    }

    /// Returns the region that corresponds to a pixel or superpixel block `b`.
    fn block_to_region_clipped(&self, b: &Block) -> Box2D {
        let pw = b.widthf();
        let ph = b.heightf();
        let px = b.x as f64 * pw;
        let py = b.y as f64 * ph;
        Box2D::new(
            point_interval(px),
            point_interval((px + pw).min(self.im.width() as f64)),
            point_interval(py),
            point_interval((py + ph).min(self.im.height() as f64)),
        )
        .transform(&self.im_to_real)
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
                x: point_interval(x),
                y: point_interval(y),
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
                x: r.x(),
                y: r.y(),
                n_theta,
                t,
            },
            cache,
        )
    }

    fn set_last_queued_block(
        &mut self,
        b: &Block,
        block_index: usize,
        parent_block_index: Option<usize>,
    ) -> Result<(), GraphingError> {
        if let Ok(block_index) = QueuedBlockIndex::try_from(block_index) {
            if b.is_superpixel() {
                let pixels = {
                    let begin = b.pixel_index();
                    let end = PixelIndex::new(
                        (begin.x + b.width()).min(self.im.width()),
                        (begin.y + b.height()).min(self.im.height()),
                    );
                    PixelRange::new(begin, end)
                };

                for p in pixels.iter() {
                    if parent_block_index.is_none()
                        || self.im[p].is_uncertain_and_disprovable(parent_block_index.unwrap())
                    {
                        self.im[p] = PixelState::Uncertain(Some(block_index));
                    }
                }
            } else {
                let p = b.pixel_index();
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

    /// Subdivides `b.n_theta` and appends the sub-blocks to `sub_bs`.
    /// Three sub-blocks are created at most.
    ///
    /// Preconditions:
    ///
    /// - `b.is_n_theta_subdivisible()` is `true`.
    /// - `b.n_theta` is a subset of either \[-∞, 0\] or \[0, +∞\].
    fn subdivide_n_theta(sub_bs: &mut Vec<Block>, b: &Block) {
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
                .map(|&n| Block::new(b.x, b.y, b.kx, b.ky, n, b.t)),
        );
    }

    /// Subdivides `b.t` and appends the sub-blocks to `sub_bs`.
    /// Four sub-blocks are created at most.
    ///
    /// Precondition: `b.is_t_subdivisible()` is `true`.
    fn subdivide_t(sub_bs: &mut Vec<Block>, b: &Block) {
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
                .map(|&t| Block::new(b.x, b.y, b.kx, b.ky, b.n_theta, t)),
        );
    }

    /// Subdivides the block both horizontally and vertically and appends the sub-blocks to `sub_bs`.
    /// Four sub-blocks are created at most.
    ///
    /// Precondition: `b.is_xy_subdivisible()` is `true`.
    fn subdivide_xy(&self, sub_bs: &mut Vec<Block>, b: &Block) {
        let x0 = 2 * b.x;
        let y0 = 2 * b.y;
        let x1 = x0 + 1;
        let y1 = y0 + 1;
        let kx = b.kx - 1;
        let ky = b.ky - 1;
        if b.is_superpixel() {
            let b00 = Block::new(x0, y0, kx, ky, b.n_theta, b.t);
            let b00_width = b00.width() as u64;
            let b00_height = b00.height() as u64;
            sub_bs.push(b00);
            if y1 * b00_height < self.im.height() as u64 {
                sub_bs.push(Block::new(x0, y1, kx, ky, b.n_theta, b.t));
            }
            if x1 * b00_width < self.im.width() as u64 {
                sub_bs.push(Block::new(x1, y0, kx, ky, b.n_theta, b.t));
            }
            if x1 * b00_width < self.im.width() as u64 && y1 * b00_height < self.im.height() as u64
            {
                sub_bs.push(Block::new(x1, y1, kx, ky, b.n_theta, b.t));
            }
        } else {
            sub_bs.push(Block::new(x0, y0, kx, ky, b.n_theta, b.t));
            sub_bs.push(Block::new(x1, y0, kx, ky, b.n_theta, b.t));
            sub_bs.push(Block::new(x0, y1, kx, ky, b.n_theta, b.t));
            sub_bs.push(Block::new(x1, y1, kx, ky, b.n_theta, b.t));
        }
    }
}

impl Graph for Implicit {
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
        for (s, dst) in self.im.pixels().copied().zip(im.pixels_mut()) {
            *dst = match s {
                PixelState::True => true_color,
                _ if s.is_uncertain(self.bs_to_subdivide.begin_index()) => uncertain_color,
                _ => false_color,
            }
        }
        imageops::flip_vertical_in_place(im);
    }

    fn get_statistics(&self) -> GraphingStatistics {
        GraphingStatistics {
            eval_count: self.rel.eval_count(),
            pixels_proven: self
                .im
                .pixels()
                .copied()
                .filter(|&s| !s.is_uncertain(self.bs_to_subdivide.begin_index()))
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
        self.im.size_in_heap()
            + self.bs_to_subdivide.size_in_heap()
            + self.cache_eval_on_region.size_in_heap()
            + self.cache_eval_on_point.size_in_heap()
    }
}
