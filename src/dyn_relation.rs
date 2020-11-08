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
    cx: HashMap<[u64; 2], Vec<TupperIntervalSet>>,
    cy: HashMap<[u64; 2], Vec<TupperIntervalSet>>,
    cxy: HashMap<[u64; 4], EvalResult>,
    size_of_cx: usize,
    size_of_cy: usize,
    size_of_cxy: usize,
    size_of_values_in_heap: usize,
}

impl EvaluationCache {
    pub fn new(level: EvaluationCacheLevel) -> Self {
        Self {
            level,
            cx: HashMap::new(),
            cy: HashMap::new(),
            cxy: HashMap::new(),
            size_of_cx: 0,
            size_of_cy: 0,
            size_of_cxy: 0,
            size_of_values_in_heap: 0,
        }
    }

    pub fn get_x(&self, k: &[u64; 2]) -> Option<&Vec<TupperIntervalSet>> {
        self.cx.get(k)
    }

    pub fn get_y(&self, k: &[u64; 2]) -> Option<&Vec<TupperIntervalSet>> {
        self.cy.get(k)
    }

    pub fn get_xy(&self, k: &[u64; 4]) -> Option<&EvalResult> {
        match self.level {
            EvaluationCacheLevel::PerAxis => None,
            EvaluationCacheLevel::Full => self.cxy.get(k),
        }
    }

    pub fn insert_x_with<F: FnOnce() -> Vec<TupperIntervalSet>>(&mut self, k: [u64; 2], f: F) {
        if let Entry::Vacant(e) = self.cx.entry(k) {
            let v = f();
            self.size_of_values_in_heap += v.capacity() * size_of::<TupperIntervalSet>()
                + v.iter().map(|t| t.size_in_heap()).sum::<usize>();
            e.insert(v);
            self.size_of_cx = self.cx.capacity()
                * (size_of::<u64>() + size_of::<[u64; 2]>() + size_of::<Vec<TupperIntervalSet>>());
        }
    }

    pub fn insert_y_with<F: FnOnce() -> Vec<TupperIntervalSet>>(&mut self, k: [u64; 2], f: F) {
        if let Entry::Vacant(e) = self.cy.entry(k) {
            let v = f();
            self.size_of_values_in_heap += v.capacity() * size_of::<TupperIntervalSet>()
                + v.iter().map(|t| t.size_in_heap()).sum::<usize>();
            e.insert(v);
            self.size_of_cy = self.cy.capacity()
                * (size_of::<u64>() + size_of::<[u64; 2]>() + size_of::<Vec<TupperIntervalSet>>());
        }
    }

    pub fn insert_xy_with<F: FnOnce() -> EvalResult>(&mut self, k: [u64; 4], f: F) {
        if self.level == EvaluationCacheLevel::Full {
            if let Entry::Vacant(e) = self.cxy.entry(k) {
                let v = f();
                self.size_of_values_in_heap += v.size_in_heap();
                e.insert(v);
                self.size_of_cxy = self.cxy.capacity()
                    * (size_of::<u64>() + size_of::<[u64; 4]>() + size_of::<EvalResult>());
            }
        }
    }

    pub fn size_in_bytes(&self) -> usize {
        // This is the lowest bound, the actual size can be much larger.
        self.size_of_cx + self.size_of_cy + self.size_of_cxy + self.size_of_values_in_heap
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RelationType {
    FunctionOfX, // y = f(x)
    FunctionOfY, // x = f(y)
    Implicit,
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

    pub fn evaluation_count(&self) -> usize {
        self.eval_count
    }

    pub fn relation_type(&self) -> RelationType {
        self.relation_type_impl(self.rels.len() - 1)
    }

    fn relation_type_impl(&self, i: usize) -> RelationType {
        match self.rels[i].kind {
            StaticRelKind::Atomic(_, i, j) => {
                match (
                    &self.exprs[i as usize].kind,
                    &self.exprs[j as usize],
                    &self.exprs[i as usize],
                    &self.exprs[j as usize].kind,
                ) {
                    (StaticExprKind::Y, f, _, _) | (_, _, f, StaticExprKind::Y)
                        if !f.dependent_axes.contains(AxisSet::Y) =>
                    {
                        // y = f(x) or f(x) = y
                        RelationType::FunctionOfX
                    }
                    (StaticExprKind::X, f, _, _) | (_, _, f, StaticExprKind::X)
                        if !f.dependent_axes.contains(AxisSet::X) =>
                    {
                        // x = f(y) or f(y) = x
                        RelationType::FunctionOfY
                    }
                    _ => RelationType::Implicit,
                }
            }
            StaticRelKind::And(i, j) | StaticRelKind::Or(i, j) => {
                match (
                    self.relation_type_impl(i as usize),
                    self.relation_type_impl(j as usize),
                ) {
                    (x, y) if x == y => x,
                    _ => RelationType::Implicit,
                }
            }
        }
    }

    pub fn rels(&self) -> &Vec<StaticRel> {
        &self.rels
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

        if let Some(r) = cache.get_xy(&kxy) {
            return r.clone();
        }

        let ts = &mut self.ts;
        let mx_ts = cache.get_x(&kx);
        let my_ts = cache.get_y(&ky);
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

        self.eval_count += 1;
        let r = EvalResult(
            self.rels
                .iter()
                .take(self.n_atom_rels)
                .map(|r| r.evaluate(&ts))
                .collect(),
        );

        let ts = &self.ts;
        cache.insert_x_with(kx, || {
            (0..self.mx.len())
                .map(|i| ts[self.mx[i] as usize].clone())
                .collect()
        });
        cache.insert_y_with(ky, || {
            (0..self.my.len())
                .map(|i| ts[self.my[i] as usize].clone())
                .collect()
        });
        cache.insert_xy_with(kxy, || r.clone());
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

    fn initialize(&mut self) {
        for i in 0..self.exprs.len() {
            let expr = &self.exprs[i];
            if let StaticExprKind::Constant(_) = expr.kind {
                self.ts[i] = expr.evaluate(&self.ts);
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
