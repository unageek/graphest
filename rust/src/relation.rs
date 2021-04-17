use crate::{
    ast::{ValueType, VarSet},
    context::Context,
    eval_result::EvalResult,
    interval_set::TupperIntervalSet,
    ops::{StaticForm, StaticFormKind, StaticTerm, StaticTermKind, StoreIndex, ValueStore},
    parse::parse_expr,
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
pub struct Relation {
    terms: Vec<StaticTerm>,
    forms: Vec<StaticForm>,
    n_atom_forms: usize,
    ts: ValueStore<TupperIntervalSet>,
    eval_count: usize,
    mx: Vec<StoreIndex>,
    my: Vec<StoreIndex>,
}

impl Relation {
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

        let terms = &self.terms;
        let ts = &mut self.ts;
        let mx_ts = cache.get_x(&kx);
        let my_ts = cache.get_y(&ky);
        if let Some(mx_ts) = mx_ts {
            #[allow(clippy::needless_range_loop)]
            for i in 0..self.mx.len() {
                ts.put(self.mx[i], mx_ts[i].clone());
            }
        }
        if let Some(my_ts) = my_ts {
            #[allow(clippy::needless_range_loop)]
            for i in 0..self.my.len() {
                ts.put(self.my[i], my_ts[i].clone());
            }
        }

        for t in terms {
            match t.kind {
                StaticTermKind::X => t.put(ts, TupperIntervalSet::from(DecInterval::new(x))),
                StaticTermKind::Y => t.put(ts, TupperIntervalSet::from(DecInterval::new(y))),
                _ => match t.vars {
                    VarSet::X if mx_ts == None => {
                        t.put_eval(terms, ts);
                    }
                    VarSet::Y if my_ts == None => {
                        t.put_eval(terms, ts);
                    }
                    VarSet::XY => {
                        t.put_eval(terms, ts);
                    }
                    _ => (), // Constant or cached subexpression.
                },
            }
        }

        let r = EvalResult(
            self.forms[..self.n_atom_forms]
                .iter()
                .map(|f| f.eval(terms, ts))
                .collect(),
        );

        let ts = &self.ts;
        cache.insert_x_with(kx, || self.mx.iter().map(|&i| ts.get(i).clone()).collect());
        cache.insert_y_with(ky, || self.my.iter().map(|&i| ts.get(i).clone()).collect());
        cache.insert_xy_with(kxy, || r.clone());
        r
    }

    fn eval_without_cache(&mut self, x: Interval, y: Interval) -> EvalResult {
        let ts = &mut self.ts;
        let terms = &self.terms;
        for t in terms {
            match t.kind {
                StaticTermKind::X => t.put(ts, TupperIntervalSet::from(DecInterval::new(x))),
                StaticTermKind::Y => t.put(ts, TupperIntervalSet::from(DecInterval::new(y))),
                _ => match t.vars {
                    VarSet::EMPTY => (), // Constant subexpression.
                    _ => t.put_eval(terms, ts),
                },
            }
        }

        EvalResult(
            self.forms[..self.n_atom_forms]
                .iter()
                .map(|f| f.eval(terms, ts))
                .collect(),
        )
    }

    fn initialize(&mut self) {
        for t in &self.terms {
            // This condition is different from `let StaticTermKind::Constant(_) = t.kind`,
            // as not all constant subexpressions are folded. See the comment on [`FoldConstant`].
            if t.vars == VarSet::EMPTY {
                t.put_eval(&self.terms, &mut self.ts);
            }
        }
    }
}

impl FromStr for Relation {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, String> {
        let mut e = parse_expr(s, Context::builtin_context())?;
        // TODO: Check types and return a pretty error message.
        PreTransform.visit_expr_mut(&mut e);
        loop {
            let mut s = SortTerms::default();
            s.visit_expr_mut(&mut e);
            let mut t = Transform::default();
            t.visit_expr_mut(&mut e);
            let mut f = FoldConstant::default();
            f.visit_expr_mut(&mut e);
            if !s.modified && !t.modified && !f.modified {
                break;
            }
        }
        PostTransform.visit_expr_mut(&mut e);
        UpdateMetadata.visit_expr_mut(&mut e);
        if e.ty != ValueType::Boolean {
            return Err("the relation must be a Boolean expression".into());
        }
        let mut v = AssignId::new();
        v.visit_expr_mut(&mut e);
        let collector = CollectStatic::new(v);
        let terms = collector.terms.clone();
        let forms = collector.forms.clone();
        let n_scalar_terms = collector.n_scalar_terms();
        let n_atom_forms = forms
            .iter()
            .filter(|f| matches!(f.kind, StaticFormKind::Atomic(_, _, _)))
            .count();

        let mut v = FindMaximalScalarTerms::new(collector);
        v.visit_expr(&e);
        let (mx, my) = v.mx_my();

        let mut slf = Self {
            terms,
            forms,
            n_atom_forms,
            ts: ValueStore::new(TupperIntervalSet::new(), n_scalar_terms),
            eval_count: 0,
            mx,
            my,
        };
        slf.initialize();
        Ok(slf)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relation_type() {
        use RelationType::*;

        fn t(rel: &str) -> RelationType {
            rel.parse::<Relation>().unwrap().relation_type()
        }

        assert_eq!(t("y = 0"), FunctionOfX);
        assert_eq!(t("0 = y"), FunctionOfX);
        assert_eq!(t("y = sin(x)"), FunctionOfX);
        assert_eq!(t("x = 0"), FunctionOfY);
        assert_eq!(t("0 = x"), FunctionOfY);
        assert_eq!(t("x = sin(y)"), FunctionOfY);
        assert_eq!(t("x y = 0"), Implicit);
        assert_eq!(t("y = sin(x y)"), Implicit);
        assert_eq!(t("sin(x) = 0"), Implicit);
        assert_eq!(t("sin(y) = 0"), Implicit);
        assert_eq!(t("y < sin(x) || sin(x) < y"), FunctionOfX);
        assert_eq!(t("y < sin(x) && sin(x) < y"), Implicit);
    }
}
