use crate::{
    ast::{AxisSet, ExprId},
    eval_result::EvalResult,
    interval_set::TupperIntervalSet,
    parse::parse,
    rel::{StaticExpr, StaticExprKind, StaticRel, StaticRelKind},
    visit::*,
};
use inari::{DecoratedInterval, Interval};
use std::{
    collections::{hash_map::Entry, HashMap},
    mem::size_of,
    str::FromStr,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EvaluationCacheLevel {
    PerAxis,
    Full,
}

pub struct EvaluationCache {
    level: EvaluationCacheLevel,
    cache: HashMap<[u64; 4], EvalResult>,
    mx_ts: HashMap<[u64; 2], Vec<TupperIntervalSet>>,
    my_ts: HashMap<[u64; 2], Vec<TupperIntervalSet>>,
    size_of_cache: usize,
    size_of_mx_ts: usize,
    size_of_my_ts: usize,
    size_of_values_in_heap: usize,
}

impl EvaluationCache {
    pub fn new(level: EvaluationCacheLevel) -> Self {
        Self {
            level,
            cache: HashMap::new(),
            mx_ts: HashMap::new(),
            my_ts: HashMap::new(),
            size_of_cache: 0,
            size_of_mx_ts: 0,
            size_of_my_ts: 0,
            size_of_values_in_heap: 0,
        }
    }

    pub fn get(&self, k: &[u64; 4]) -> Option<&EvalResult> {
        match self.level {
            EvaluationCacheLevel::PerAxis => None,
            EvaluationCacheLevel::Full => self.cache.get(k),
        }
    }

    pub fn insert_with<F: FnOnce() -> EvalResult>(&mut self, k: [u64; 4], f: F) {
        if self.level == EvaluationCacheLevel::Full {
            if let Entry::Vacant(e) = self.cache.entry(k) {
                let v = f();
                self.size_of_values_in_heap += v.size_in_heap();
                e.insert(v);
                self.size_of_cache = self.cache.capacity()
                    * (size_of::<u64>() + size_of::<[u64; 4]>() + size_of::<EvalResult>());
            }
        }
    }

    pub fn insert_mx_ts_with<F: FnOnce() -> Vec<TupperIntervalSet>>(&mut self, k: [u64; 2], f: F) {
        if let Entry::Vacant(e) = self.mx_ts.entry(k) {
            let v = f();
            self.size_of_values_in_heap += v.capacity() * size_of::<TupperIntervalSet>()
                + v.iter().map(|t| t.size_in_heap()).sum::<usize>();
            e.insert(v);
            self.size_of_mx_ts = self.mx_ts.capacity()
                * (size_of::<u64>() + size_of::<[u64; 2]>() + size_of::<Vec<TupperIntervalSet>>());
        }
    }

    pub fn insert_my_ts_with<F: FnOnce() -> Vec<TupperIntervalSet>>(&mut self, k: [u64; 2], f: F) {
        if let Entry::Vacant(e) = self.my_ts.entry(k) {
            let v = f();
            self.size_of_values_in_heap += v.capacity() * size_of::<TupperIntervalSet>()
                + v.iter().map(|t| t.size_in_heap()).sum::<usize>();
            e.insert(v);
            self.size_of_my_ts = self.my_ts.capacity()
                * (size_of::<u64>() + size_of::<[u64; 2]>() + size_of::<Vec<TupperIntervalSet>>());
        }
    }

    pub fn size_in_bytes(&self) -> usize {
        // This is the lowest bound, the actual size can be much larger.
        self.size_of_cache + self.size_of_mx_ts + self.size_of_my_ts + self.size_of_values_in_heap
    }
}

#[derive(Clone, Debug)]
pub struct DynRelation {
    exprs: Vec<StaticExpr>,
    rels: Vec<StaticRel>,
    n_atom_rels: usize,
    ts: Vec<TupperIntervalSet>,
    eval_count: usize,
    mx: Vec<ExprId>,
    my: Vec<ExprId>,
}

impl DynRelation {
    pub fn evaluate(
        &mut self,
        x: Interval,
        y: Interval,
        cache: Option<&mut EvaluationCache>,
    ) -> EvalResult {
        match cache {
            Some(cache) => self.evaluate_with_cache(x, y, cache),
            _ => self.evaluate_without_cache(x, y),
        }
    }

    fn evaluate_with_cache(
        &mut self,
        x: Interval,
        y: Interval,
        cache: &mut EvaluationCache,
    ) -> EvalResult {
        let kx = [x.inf().to_bits(), x.sup().to_bits()];
        let ky = [y.inf().to_bits(), y.sup().to_bits()];
        let kxy = [kx[0], kx[1], ky[0], ky[1]];

        if let Some(r) = cache.get(&kxy) {
            return r.clone();
        }

        let ts = &mut self.ts;
        let mx_ts = cache.mx_ts.get(&kx);
        let my_ts = cache.my_ts.get(&ky);
        if let Some(mx_ts) = mx_ts {
            for i in 0..self.mx.len() {
                ts[self.mx[i] as usize] = mx_ts[i].clone();
            }
        }
        if let Some(my_ts) = my_ts {
            for i in 0..self.my.len() {
                ts[self.my[i] as usize] = my_ts[i].clone();
            }
        }

        for i in 0..self.exprs.len() {
            let expr = &self.exprs[i];
            match expr.kind {
                StaticExprKind::X => ts[i] = TupperIntervalSet::from(DecoratedInterval::new(x)),
                StaticExprKind::Y => ts[i] = TupperIntervalSet::from(DecoratedInterval::new(y)),
                _ => match expr.dependent_axes {
                    AxisSet::X => {
                        if mx_ts == None {
                            ts[i] = expr.evaluate(&ts);
                        }
                    }
                    AxisSet::Y => {
                        if my_ts == None {
                            ts[i] = expr.evaluate(&ts);
                        }
                    }
                    AxisSet::XY => {
                        ts[i] = expr.evaluate(&ts);
                    }
                    _ => (),
                },
            }
        }

        let r = EvalResult(
            self.rels
                .iter()
                .take(self.n_atom_rels)
                .map(|r| r.evaluate(&ts))
                .collect(),
        );
        self.eval_count += 1;

        let ts = &self.ts;
        cache.insert_mx_ts_with(kx, || {
            (0..self.mx.len())
                .map(|i| ts[self.mx[i] as usize].clone())
                .collect()
        });
        cache.insert_my_ts_with(ky, || {
            (0..self.my.len())
                .map(|i| ts[self.my[i] as usize].clone())
                .collect()
        });
        cache.insert_with(kxy, || r.clone());
        r
    }

    fn evaluate_without_cache(&mut self, x: Interval, y: Interval) -> EvalResult {
        let ts = &mut self.ts;
        for i in 0..self.exprs.len() {
            let expr = &self.exprs[i];
            match expr.kind {
                StaticExprKind::Constant(_) => (),
                StaticExprKind::X => ts[i] = TupperIntervalSet::from(DecoratedInterval::new(x)),
                StaticExprKind::Y => ts[i] = TupperIntervalSet::from(DecoratedInterval::new(y)),
                _ => {
                    ts[i] = expr.evaluate(&ts);
                }
            }
        }

        self.eval_count += 1;
        EvalResult(
            self.rels
                .iter()
                .take(self.n_atom_rels)
                .map(|r| r.evaluate(&ts))
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
            if self.exprs[i].dependent_axes.is_empty() {
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
        UpdateMetadata.visit_rel_mut(&mut rel);
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

        let mut v = FindMaxima::new();
        v.visit_rel(&rel);
        let (mx, my) = v.mx_my();

        let mut slf = Self {
            exprs,
            rels,
            n_atom_rels,
            ts: vec![TupperIntervalSet::empty(); n_ts],
            eval_count: 0,
            mx,
            my,
        };
        slf.initialize();
        Ok(slf)
    }
}
