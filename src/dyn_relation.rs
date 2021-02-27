use crate::{
    ast::{TermId, VarSet},
    context::Context,
    eval_result::EvalResult,
    interval_set::TupperIntervalSet,
    parse::parse,
    rel::{StaticForm, StaticFormKind, StaticTerm, StaticTermKind},
    visit::*,
};
use inari::{DecInterval, Interval};
use std::{
    collections::{hash_map::Entry, HashMap},
    mem::size_of,
    str::FromStr,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EvalCacheLevel {
    PerAxis,
    Full,
}

pub struct EvalCache {
    level: EvalCacheLevel,
    cx: HashMap<[u64; 2], Vec<TupperIntervalSet>>,
    cy: HashMap<[u64; 2], Vec<TupperIntervalSet>>,
    cxy: HashMap<[u64; 4], EvalResult>,
    size_of_cx: usize,
    size_of_cy: usize,
    size_of_cxy: usize,
    size_of_values_in_heap: usize,
}

impl EvalCache {
    pub fn new(level: EvalCacheLevel) -> Self {
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
            EvalCacheLevel::PerAxis => None,
            EvalCacheLevel::Full => self.cxy.get(k),
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
        if self.level == EvalCacheLevel::Full {
            if let Entry::Vacant(e) = self.cxy.entry(k) {
                let v = f();
                self.size_of_values_in_heap += v.size_in_heap();
                e.insert(v);
                self.size_of_cxy = self.cxy.capacity()
                    * (size_of::<u64>() + size_of::<[u64; 4]>() + size_of::<EvalResult>());
            }
        }
    }

    pub fn size_in_heap(&self) -> usize {
        // This is a lowest bound, the actual size can be much larger.
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
    terms: Vec<StaticTerm>,
    forms: Vec<StaticForm>,
    n_atom_forms: usize,
    ts: Vec<TupperIntervalSet>,
    eval_count: usize,
    mx: Vec<TermId>,
    my: Vec<TermId>,
}

impl DynRelation {
    pub fn eval(&mut self, x: Interval, y: Interval, cache: Option<&mut EvalCache>) -> EvalResult {
        self.eval_count += 1;
        match cache {
            Some(cache) => self.eval_with_cache(x, y, cache),
            _ => self.eval_without_cache(x, y),
        }
    }

    pub fn eval_count(&self) -> usize {
        self.eval_count
    }

    pub fn forms(&self) -> &Vec<StaticForm> {
        &self.forms
    }

    pub fn relation_type(&self) -> RelationType {
        self.relation_type_impl(self.forms.len() - 1)
    }

    fn relation_type_impl(&self, i: usize) -> RelationType {
        match self.forms[i].kind {
            StaticFormKind::Atomic(_, i, j) => {
                match (
                    &self.terms[i as usize].kind,
                    &self.terms[j as usize],
                    &self.terms[i as usize],
                    &self.terms[j as usize].kind,
                ) {
                    (StaticTermKind::Y, t, _, _) | (_, _, t, StaticTermKind::Y)
                        if !t.vars.contains(VarSet::Y) =>
                    {
                        // y = f(x) or f(x) = y
                        RelationType::FunctionOfX
                    }
                    (StaticTermKind::X, t, _, _) | (_, _, t, StaticTermKind::X)
                        if !t.vars.contains(VarSet::X) =>
                    {
                        // x = f(y) or f(y) = x
                        RelationType::FunctionOfY
                    }
                    _ => RelationType::Implicit,
                }
            }
            StaticFormKind::And(_, _) => {
                // This should not be `FunctionOfX` nor `FunctionOfY`.
                // Example: "y = x && y = x + 0.0001"
                //                              /
                //                 +--+       +/-+
                //                 |  |       /  |
                //   FunctionOfX:  |  |/  ∧  /|  |   =   ?
                //                 |  / T     |  | T
                //                 +-/+       +--+
                //                  /
                //                              /
                //                 +--+       +/-+       +--+
                //                 |  | F     /  | T     |  | F
                //      Implicit:  +--+/  ∧  /+--+   =   +--+
                //                 |  / T     |  | F     |  | F
                //                 +-/+       +--+       +--+
                //                  /
                // See `EvalResultMask::solution_certainly_exists` for how conjunctions are evaluated.
                RelationType::Implicit
            }
            StaticFormKind::Or(i, j) => {
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

    fn eval_with_cache(&mut self, x: Interval, y: Interval, cache: &mut EvalCache) -> EvalResult {
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

        for i in 0..self.terms.len() {
            let t = &self.terms[i];
            match t.kind {
                StaticTermKind::X => ts[i] = TupperIntervalSet::from(DecInterval::new(x)),
                StaticTermKind::Y => ts[i] = TupperIntervalSet::from(DecInterval::new(y)),
                _ => match t.vars {
                    VarSet::X if mx_ts == None => {
                        ts[i] = t.eval(&ts);
                    }
                    VarSet::Y if my_ts == None => {
                        ts[i] = t.eval(&ts);
                    }
                    VarSet::XY => {
                        ts[i] = t.eval(&ts);
                    }
                    _ => (), // Constant or cached subexpression.
                },
            }
        }

        let r = EvalResult(
            self.forms[..self.n_atom_forms]
                .iter()
                .map(|f| f.eval(&ts))
                .collect(),
        );

        let ts = &self.ts;
        cache.insert_x_with(kx, || {
            self.mx.iter().map(|&i| ts[i as usize].clone()).collect()
        });
        cache.insert_y_with(ky, || {
            self.my.iter().map(|&i| ts[i as usize].clone()).collect()
        });
        cache.insert_xy_with(kxy, || r.clone());
        r
    }

    fn eval_without_cache(&mut self, x: Interval, y: Interval) -> EvalResult {
        let ts = &mut self.ts;
        for i in 0..self.terms.len() {
            let t = &self.terms[i];
            match t.kind {
                StaticTermKind::X => ts[i] = TupperIntervalSet::from(DecInterval::new(x)),
                StaticTermKind::Y => ts[i] = TupperIntervalSet::from(DecInterval::new(y)),
                _ => match t.vars {
                    VarSet::EMPTY => (), // Constant subexpression.
                    _ => ts[i] = t.eval(&ts),
                },
            }
        }

        EvalResult(
            self.forms[..self.n_atom_forms]
                .iter()
                .map(|f| f.eval(&ts))
                .collect(),
        )
    }

    fn initialize(&mut self) {
        for i in 0..self.terms.len() {
            let t = &self.terms[i];
            // This condition is different from `let StaticTermKind::Constant(_) = t.kind`,
            // as not all constant subexpressions are folded. See the comment on [`FoldConstant`].
            if t.vars == VarSet::EMPTY {
                self.ts[i] = t.eval(&self.ts);
            }
        }
    }
}

impl FromStr for DynRelation {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, String> {
        let ctx = Context::new();
        let mut form = parse(s, &ctx)?;
        PreTransform.visit_form_mut(&mut form);
        loop {
            let mut s = SortTerms::default();
            s.visit_form_mut(&mut form);
            let mut t = Transform::default();
            t.visit_form_mut(&mut form);
            let mut f = FoldConstant::default();
            f.visit_form_mut(&mut form);
            if !s.modified && !t.modified && !f.modified {
                break;
            }
        }
        PostTransform.visit_form_mut(&mut form);
        loop {
            let mut v = NormalizeForms::default();
            v.visit_form_mut(&mut form);
            if !v.modified {
                break;
            }
        }
        UpdateMetadata.visit_form_mut(&mut form);
        let mut v = AssignIdStage1::new();
        v.visit_form(&form);
        let mut v = AssignIdStage2::new(v);
        v.visit_form(&form);
        let mut v = CollectStatic::new(v);
        v.visit_form(&form);
        let (terms, forms) = v.terms_forms();
        let n_ts = terms.len();
        let n_atom_forms = forms
            .iter()
            .filter(|f| matches!(f.kind, StaticFormKind::Atomic(_, _, _)))
            .count();

        let mut v = FindMaximalTerms::new();
        v.visit_form(&form);
        let (mx, my) = v.mx_my();

        let mut slf = Self {
            terms,
            forms,
            n_atom_forms,
            ts: vec![TupperIntervalSet::empty(); n_ts],
            eval_count: 0,
            mx,
            my,
        };
        slf.initialize();
        Ok(slf)
    }
}
