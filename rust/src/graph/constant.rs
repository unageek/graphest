use crate::{
    graph::{Graph, GraphingError, GraphingErrorKind, GraphingStatistics},
    interval_set::{DecSignSet, SignSet},
    ops::StaticForm,
    relation::{Relation, RelationArgs, RelationType},
};
use image::{ImageBuffer, Pixel};
use inari::Decoration;
use std::{
    ops::{Deref, DerefMut},
    time::{Duration, Instant},
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Ternary {
    False,
    Uncertain,
    True,
}

/// Plots a constant relation with a single evaluation.
pub struct Constant {
    rel: Relation,
    forms: Vec<StaticForm>,
    im_width: u32,
    im_height: u32,
    result: Option<Ternary>,
    stats: GraphingStatistics,
}

impl Constant {
    pub fn new(rel: Relation, im_width: u32, im_height: u32) -> Self {
        assert_eq!(rel.relation_type(), RelationType::Constant);

        let forms = rel.forms().clone();
        Self {
            rel,
            forms,
            im_width,
            im_height,
            result: None,
            stats: GraphingStatistics {
                eval_count: 0,
                pixels: im_width as usize * im_height as usize,
                pixels_proven: 0,
                time_elapsed: Duration::ZERO,
            },
        }
    }

    fn refine_impl(&mut self) -> Result<bool, GraphingError> {
        if self.result.is_none() {
            let r = self.rel.eval(&RelationArgs::default(), None);
            let is_true = r
                .map(|DecSignSet(ss, d)| ss == SignSet::ZERO && d >= Decoration::Def)
                .eval(&self.forms[..]);
            let is_false = !r
                .map(|DecSignSet(ss, _)| ss.contains(SignSet::ZERO))
                .eval(&self.forms[..]);

            self.result = Some(if is_true {
                Ternary::True
            } else if is_false {
                Ternary::False
            } else {
                Ternary::Uncertain
            });
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
        assert!(im.width() == self.im_width && im.height() == self.im_height);
        for dst in im.pixels_mut() {
            *dst = match self.result {
                Some(Ternary::True) => true_color,
                Some(Ternary::Uncertain) | None => uncertain_color,
                Some(Ternary::False) => false_color,
            }
        }
    }

    fn get_statistics(&self) -> GraphingStatistics {
        GraphingStatistics {
            eval_count: self.rel.eval_count(),
            pixels_proven: match self.result {
                Some(Ternary::False | Ternary::True) => self.stats.pixels,
                Some(Ternary::Uncertain) | None => 0,
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

    fn size_in_heap(&self) -> usize {
        0
    }
}
