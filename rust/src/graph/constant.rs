use crate::{
    eval_cache::{EvalCacheLevel, EvalImplicitCache},
    graph::{Graph, GraphingError, GraphingErrorKind, GraphingStatistics, Ternary},
    image::Image,
    relation::{Relation, RelationType},
    traits::BytesAllocated,
    vars::VarSet,
};
use std::time::{Duration, Instant};

/// Plots a constant relation with a single evaluation.
pub struct Constant {
    rel: Relation,
    im_width: u32,
    im_height: u32,
    result: Option<Ternary>,
    stats: GraphingStatistics,
}

impl Constant {
    pub fn new(rel: Relation, im_width: u32, im_height: u32) -> Self {
        assert_eq!(rel.relation_type(), RelationType::Constant);

        Self {
            rel,
            im_width,
            im_height,
            result: None,
            stats: GraphingStatistics {
                eval_count: 0,
                pixels: im_width as usize * im_height as usize,
                pixels_complete: 0,
                time_elapsed: Duration::ZERO,
            },
        }
    }

    fn refine_impl(&mut self) -> Result<bool, GraphingError> {
        if self.result.is_none() {
            let args = self.rel.create_args();
            let r = self.rel.eval_implicit(
                &args,
                &mut EvalImplicitCache::new(EvalCacheLevel::None, VarSet::EMPTY),
            );

            self.result = Some(r.result(self.rel.forms()));
        }

        if self.result == Some(Ternary::Uncertain) {
            Err(GraphingError {
                kind: GraphingErrorKind::ReachedSubdivisionLimit,
            })
        } else {
            Ok(true)
        }
    }
}

impl Graph for Constant {
    fn get_image(&self, im: &mut Image<Ternary>) {
        assert!(im.width() == self.im_width && im.height() == self.im_height);
        for dst in im.pixels_mut() {
            *dst = match self.result {
                Some(x) => x,
                _ => Ternary::Uncertain,
            }
        }
    }

    fn get_statistics(&self) -> GraphingStatistics {
        GraphingStatistics {
            eval_count: self.rel.eval_count(),
            pixels_complete: match self.result {
                Some(Ternary::False | Ternary::True) => self.stats.pixels,
                _ => 0,
            },
            ..self.stats
        }
    }

    fn refine(&mut self, _duration: Duration) -> Result<bool, GraphingError> {
        let now = Instant::now();
        let result = self.refine_impl();
        self.stats.time_elapsed += now.elapsed();
        result
    }
}

impl BytesAllocated for Constant {
    fn bytes_allocated(&self) -> usize {
        0
    }
}
