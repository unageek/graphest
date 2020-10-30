use crate::{
    eval_result::EvalResult,
    interval_set::TupperIntervalSet,
    parse::parse,
    rel::{StaticExpr, StaticExprKind, StaticRel, StaticRelKind},
    visit::*,
};
use inari::{DecoratedInterval, Interval};
use std::{collections::HashMap, mem::size_of, str::FromStr};

type EvaluationCacheKey = [u64; 4];
pub struct EvaluationCache {
    cache: HashMap<EvaluationCacheKey, EvalResult>,
    size_of_values_in_heap: usize,
}

impl EvaluationCache {
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
            size_of_values_in_heap: 0,
        }
    }

    pub fn get_or_insert_with<F: FnOnce() -> EvalResult>(
        &mut self,
        x: Interval,
        y: Interval,
        f: F,
    ) -> &EvalResult {
        let res = self
            .cache
            .entry([
                x.inf().to_bits(),
                x.sup().to_bits(),
                y.inf().to_bits(),
                y.sup().to_bits(),
            ])
            .or_insert_with(f);
        self.size_of_values_in_heap += res.size_in_heap();
        res
    }

    pub fn size_in_bytes(&self) -> usize {
        // This is the lowest bound, the actual size can be much larger.
        self.cache.capacity()
            * (size_of::<u64>() + size_of::<EvaluationCacheKey>() + size_of::<EvalResult>())
            + self.size_of_values_in_heap
    }
}

#[derive(Clone, Debug)]
pub struct DynRelation {
    exprs: Vec<StaticExpr>,
    rels: Vec<StaticRel>,
    n_atom_rels: usize,
    ts: Vec<TupperIntervalSet>,
    eval_count: usize,
}

impl DynRelation {
    pub fn evaluate(
        &mut self,
        x: Interval,
        y: Interval,
        cache: Option<&mut EvaluationCache>,
    ) -> EvalResult {
        match cache {
            Some(cache) => cache
                .get_or_insert_with(x, y, || self.evaluate_impl(x, y))
                .clone(),
            _ => self.evaluate_impl(x, y),
        }
    }

    fn evaluate_impl(&mut self, x: Interval, y: Interval) -> EvalResult {
        for i in 0..self.exprs.len() {
            match &self.exprs[i].kind {
                StaticExprKind::Constant(_) => (),
                StaticExprKind::X => {
                    self.ts[i] = TupperIntervalSet::from(DecoratedInterval::new(x))
                }
                StaticExprKind::Y => {
                    self.ts[i] = TupperIntervalSet::from(DecoratedInterval::new(y))
                }
                _ => self.ts[i] = self.exprs[i].evaluate(&self.ts),
            }
        }
        self.eval_count += 1;
        EvalResult(
            self.rels
                .iter()
                .take(self.n_atom_rels)
                .map(|r| r.evaluate(&self.ts))
                .collect(),
        )
    }

    pub fn evaluation_count(&self) -> usize {
        self.eval_count
    }

    pub fn rels(&self) -> &Vec<StaticRel> {
        &self.rels
    }

    fn initialize(&mut self) {
        for i in 0..self.exprs.len() {
            if let StaticExprKind::Constant(_) = &self.exprs[i].kind {
                self.ts[i] = self.exprs[i].evaluate(&self.ts);
            }
        }
    }
}

impl FromStr for DynRelation {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, String> {
        let mut rel = parse(s)?;
        Transform.visit_rel_mut(&mut rel);
        FoldConstant.visit_rel_mut(&mut rel);
        Transform.visit_rel_mut(&mut rel);
        FoldConstant.visit_rel_mut(&mut rel);
        let mut v = AssignIdStage1::new();
        v.visit_rel(&rel);
        let mut v = AssignIdStage2::new(v);
        v.visit_rel(&rel);
        let mut v = CollectStatic::new(v);
        v.visit_rel(&rel);
        let (exprs, rels) = v.exprs_rels();
        let n_ts = exprs.len();
        let n_atom_rels = rels
            .iter()
            .filter(|r| matches!(r.kind, StaticRelKind::Atomic(_, _, _)))
            .count();
        let mut slf = Self {
            exprs,
            rels,
            n_atom_rels,
            ts: vec![TupperIntervalSet::empty(); n_ts],
            eval_count: 0,
        };
        slf.initialize();
        Ok(slf)
    }
}
