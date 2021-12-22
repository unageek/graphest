use crate::{
    block::{Block, BlockQueue, Coordinate, IntegerParameter, RealParameter},
    eval_cache::{EvalCacheLevel, EvalImplicitCache},
    eval_result::EvalArgs,
    geom::{Box2D, Transform2D, TransformMode},
    graph::{
        common::*, Graph, GraphingError, GraphingErrorKind, GraphingStatistics, Padding, Ternary,
    },
    image::{Image, PixelIndex, PixelRange},
    interval_set::{DecSignSet, SignSet},
    region::Region,
    relation::{Relation, RelationType, VarIndices},
    set_arg,
    traits::BytesAllocated,
    vars,
    vars::VarSet,
};
use inari::Decoration;
use itertools::Itertools;
use smallvec::smallvec;
use std::{
    convert::TryFrom,
    default::default,
    iter::once,
    time::{Duration, Instant},
};

const XY: VarSet = vars!(VarSet::X | VarSet::Y);

/// The graphing algorithm for implicit relations.
pub struct Implicit {
    rel: Relation,
    vars: VarSet,
    var_indices: VarIndices,
    subdivision_dirs: Vec<VarSet>,
    im: Image<PixelState>,
    // Queue blocks that will be subdivided instead of the divided blocks to save memory.
    bs_to_subdivide: BlockQueue,
    // Affine transformation from image coordinates to real coordinates.
    im_to_real: Transform2D,
    stats: GraphingStatistics,
    mem_limit: usize,
    cache_eval_on_region: EvalImplicitCache,
    cache_eval_on_point: EvalImplicitCache,
}

impl Implicit {
    pub fn new(
        rel: Relation,
        region: Box2D,
        im_width: u32,
        im_height: u32,
        padding: Padding,
        mem_limit: usize,
    ) -> Self {
        assert_eq!(rel.relation_type(), RelationType::Implicit);

        let vars = rel.vars();
        let var_indices = rel.var_indices().clone();
        let subdivision_dirs = [XY, VarSet::N_THETA, VarSet::N, VarSet::T]
            .into_iter()
            .filter(|&d| (XY | vars).contains(d))
            .collect();

        let mut g = Self {
            rel,
            vars,
            var_indices,
            subdivision_dirs,
            im: Image::new(im_width, im_height),
            bs_to_subdivide: BlockQueue::new(XY | vars),
            im_to_real: Transform2D::new(
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
                [
                    Region::new(region.left(), region.bottom()),
                    Region::new(region.right(), region.top()),
                ],
                TransformMode::Fast,
            ),
            stats: GraphingStatistics {
                eval_count: 0,
                pixels: im_width as usize * im_height as usize,
                pixels_complete: 0,
                time_elapsed: Duration::ZERO,
            },
            mem_limit,
            cache_eval_on_region: EvalImplicitCache::new(EvalCacheLevel::Univariate, vars),
            cache_eval_on_point: if XY.contains(vars) {
                EvalImplicitCache::new(EvalCacheLevel::Full, vars)
            } else {
                EvalImplicitCache::new(EvalCacheLevel::Univariate, vars)
            },
        };

        let k = (im_width.max(im_height) as f64).log2().ceil() as i8;
        let mut bs = vec![Block {
            x: Coordinate::new(0, k),
            y: Coordinate::new(0, k),
            t: RealParameter::new(g.rel.t_range()),
            next_dir: g.subdivision_dirs[0],
            ..default()
        }];

        if vars.contains(VarSet::N_THETA) {
            let n_theta_range = g.rel.n_theta_range();
            bs = IntegerParameter::initial_subdivision(n_theta_range)
                .into_iter()
                .flat_map(|n| {
                    if n.is_subdivisible() {
                        n.subdivide()
                    } else {
                        smallvec![n]
                    }
                })
                .cartesian_product(bs.into_iter())
                .map(|(n, b)| Block { n_theta: n, ..b })
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
        let mut next_dir_candidates = vec![];
        let mut args = self.rel.create_args();
        while let Some(b) = self.bs_to_subdivide.pop_front() {
            let bi = self.bs_to_subdivide.begin_index() - 1;
            match b.next_dir {
                XY => self.subdivide_xy(&mut sub_bs, &b),
                VarSet::N_THETA => subdivide_n_theta(&mut sub_bs, &b),
                VarSet::N => subdivide_n(&mut sub_bs, &b),
                VarSet::T => subdivide_t_twice(&mut sub_bs, &b),
                _ => panic!(),
            }

            let n_sub_bs = sub_bs.len();
            for sub_b in sub_bs.drain(..) {
                let complete = if !sub_b.x.is_subpixel() {
                    self.process_block(&sub_b, &mut args)
                } else {
                    if self.vars.contains(VarSet::N_THETA) {
                        let n = sub_b.n_theta.interval();
                        let n_simple = simple_fraction(n);
                        if n.inf() != n_simple && n.sup() != n_simple {
                            // Try finding a solution earlier.
                            self.process_subpixel_block(
                                &Block {
                                    n_theta: IntegerParameter::new(point_interval(n_simple)),
                                    ..sub_b
                                },
                                &mut args,
                            );
                        }
                    }
                    self.process_subpixel_block(&sub_b, &mut args)
                };
                if !complete {
                    incomplete_sub_bs.push(sub_b);
                }
            }

            let n_max = match b.next_dir {
                XY => 4,
                VarSet::N_THETA | VarSet::N => 3,
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

            for mut sub_b in incomplete_sub_bs.drain(..) {
                let next_dir = next_dir_candidates.iter().copied().find(|&d| {
                    d == XY && sub_b.x.is_subdivisible()
                        || d == VarSet::N_THETA && sub_b.n_theta.is_subdivisible()
                        || d == VarSet::N && sub_b.n.is_subdivisible()
                        || d == VarSet::T && sub_b.t.is_subdivisible()
                });

                if let Some(d) = next_dir {
                    sub_b.next_dir = d;
                } else {
                    // Cannot subdivide in any direction.
                    assert!(sub_b.x.is_subpixel());
                    for p in &self.pixels_in_image(&b) {
                        self.im[p] = PixelState::Uncertain(None);
                    }
                    continue;
                }

                self.bs_to_subdivide.push_back(sub_b.clone());
                let last_bi = self.bs_to_subdivide.end_index() - 1;
                self.set_last_queued_block(&sub_b, last_bi, Some(bi))?;
            }

            let mut clear_cache_and_retry = true;
            while self.bytes_allocated() > self.mem_limit {
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
    fn process_block(&mut self, b: &Block, args: &mut EvalArgs) -> bool {
        let pixels = self.pixels_in_image(b);
        if pixels.iter().all(|p| self.im[p] == PixelState::True) {
            // All pixels have already been proven to be true.
            return true;
        }

        let u_up = self.block_to_region_clipped(b).outer();
        set_arg!(args, self.var_indices.n, b.n.interval());
        set_arg!(args, self.var_indices.n_theta, b.n_theta.interval());
        set_arg!(args, self.var_indices.t, b.t.interval());
        set_arg!(args, self.var_indices.x, u_up.x());
        set_arg!(args, self.var_indices.y, u_up.y());
        let r_u_up = self.rel.eval_implicit(args, &mut self.cache_eval_on_region);

        let result = r_u_up.result(self.rel.forms());

        if result.certainly_true() {
            for p in pixels.iter() {
                self.im[p] = PixelState::True;
            }
            return true;
        }

        result.certainly_false()
    }

    /// Tries to prove or disprove the existence of a solution in the block
    /// and returns `true` if it is successful.
    ///
    /// Precondition: the block is a subpixel.
    fn process_subpixel_block(&mut self, b: &Block, args: &mut EvalArgs) -> bool {
        let pixels = self.pixels_in_image(b);
        if pixels.iter().all(|p| self.im[p] == PixelState::True) {
            // This pixel has already been proven to be true.
            return true;
        }

        let u_up = subpixel_outer(&self.block_to_region(b), b);
        let p_dn = self.block_to_region(&b.pixel_block()).inner();
        let inter = u_up.intersection(&p_dn);

        set_arg!(args, self.var_indices.n, b.n.interval());
        set_arg!(args, self.var_indices.n_theta, b.n_theta.interval());
        set_arg!(args, self.var_indices.t, b.t.interval());
        set_arg!(args, self.var_indices.x, u_up.x());
        set_arg!(args, self.var_indices.y, u_up.y());
        let (result_mask, dac_mask, mut neg_mask, mut pos_mask) = {
            let r_u_up = self.rel.eval_implicit(args, &mut self.cache_eval_on_region);

            let result_mask = r_u_up.result_mask();
            let result = result_mask.eval(self.rel.forms());

            if result.certainly_true() && !inter.is_empty() {
                // The relation is true everywhere in the subpixel, and the subpixel certainly overlaps
                // with the pixel. Therefore, the pixel contains a solution.
                for p in &pixels {
                    self.im[p] = PixelState::True;
                }
                return true;
            }

            if result.certainly_false() {
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

            let neg_mask = r_u_up.map(|_| false);
            let pos_mask = neg_mask.clone();

            (result_mask, dac_mask, neg_mask, pos_mask)
        };

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

        for point in &points {
            set_arg!(args, self.var_indices.x, point_interval(point.0));
            set_arg!(args, self.var_indices.y, point_interval(point.1));
            let r = self.rel.eval_implicit(args, &mut self.cache_eval_on_point);

            // `ss` is nonempty if the decoration is ≥ `Def`, which will be ensured
            // by taking bitand with `dac_mask`.
            neg_mask |= r.map(|DecSignSet(ss, _)| (SignSet::NEG | SignSet::ZERO).contains(ss));
            pos_mask |= r.map(|DecSignSet(ss, _)| (SignSet::POS | SignSet::ZERO).contains(ss));

            let point_result = r.result(self.rel.forms());

            if point_result.certainly_true()
                || (&(&neg_mask & &pos_mask) & &dac_mask)
                    .solution_certainly_exists(self.rel.forms(), &result_mask)
            {
                // Found a solution.
                for p in &pixels {
                    self.im[p] = PixelState::True;
                }
                return true;
            }
        }

        false
    }

    /// Returns the region that corresponds to a subpixel block `b`.
    fn block_to_region(&self, b: &Block) -> Box2D {
        let pw = b.x.widthf();
        let ph = b.y.widthf();
        let px = b.x.index() as f64 * pw;
        let py = b.y.index() as f64 * ph;
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
        let pw = b.x.widthf();
        let ph = b.y.widthf();
        let px = b.x.index() as f64 * pw;
        let py = b.y.index() as f64 * ph;
        Box2D::new(
            point_interval(px),
            point_interval((px + pw).min(self.im.width() as f64)),
            point_interval(py),
            point_interval((py + ph).min(self.im.height() as f64)),
        )
        .transform(&self.im_to_real)
    }

    /// Returns the pixels that are contained in both the block and the image.
    fn pixels_in_image(&self, b: &Block) -> PixelRange {
        let begin = b.pixel_index();
        let end = if b.x.is_superpixel() {
            PixelIndex::new(
                (begin.x + b.x.width()).min(self.im.width()),
                (begin.y + b.y.width()).min(self.im.height()),
            )
        } else {
            PixelIndex::new(
                (begin.x + 1).min(self.im.width()),
                (begin.y + 1).min(self.im.height()),
            )
        };
        PixelRange::new(
            PixelIndex::new(begin.x, self.im.height() - end.y),
            PixelIndex::new(end.x, self.im.height() - begin.y),
        )
    }

    fn set_last_queued_block(
        &mut self,
        b: &Block,
        block_index: usize,
        parent_block_index: Option<usize>,
    ) -> Result<(), GraphingError> {
        if let Ok(block_index) = QueuedBlockIndex::try_from(block_index) {
            for p in self.pixels_in_image(b).iter() {
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

    /// Subdivides both `b.x` and `b.y` and appends the sub-blocks to `sub_bs`.
    /// Four sub-blocks are created at most.
    ///
    /// Precondition: both `b.x.is_subdivisible()` and `b.y.is_subdivisible()` are `true`.
    fn subdivide_xy(&self, sub_bs: &mut Vec<Block>, b: &Block) {
        let [x0, x1] = b.x.subdivide();
        let [y0, y1] = b.y.subdivide();
        if b.x.is_superpixel() {
            let push_x1 = x1.pixel_index() < self.im.width();
            let push_y1 = y1.pixel_index() < self.im.height();
            sub_bs.push(Block { x: x0, y: y0, ..*b });
            if push_x1 {
                sub_bs.push(Block { x: x1, y: y0, ..*b });
            }
            if push_y1 {
                sub_bs.push(Block { x: x0, y: y1, ..*b });
            }
            if push_x1 && push_y1 {
                sub_bs.push(Block { x: x1, y: y1, ..*b });
            }
        } else {
            sub_bs.extend([
                Block { x: x0, y: y0, ..*b },
                Block { x: x1, y: y0, ..*b },
                Block { x: x0, y: y1, ..*b },
                Block { x: x1, y: y1, ..*b },
            ]);
        }
    }
}

impl Graph for Implicit {
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

impl BytesAllocated for Implicit {
    fn bytes_allocated(&self) -> usize {
        self.im.bytes_allocated()
            + self.bs_to_subdivide.bytes_allocated()
            + self.cache_eval_on_region.bytes_allocated()
            + self.cache_eval_on_point.bytes_allocated()
    }
}
