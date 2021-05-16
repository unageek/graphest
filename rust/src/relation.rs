use crate::{
    ast::{BinaryOp, Expr, ExprKind, UnaryOp, ValueType, VarSet},
    context::Context,
    eval_result::EvalResult,
    interval_set::TupperIntervalSet,
    ops::{StaticForm, StaticFormKind, StaticTerm, StaticTermKind, StoreIndex, ValueStore},
    parse::parse_expr,
    visit::*,
};
use inari::{const_dec_interval, DecInterval, Interval};
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

/// Type of the relation, which should be used to choose the optimal graphing strategy.
///
/// The following relationships hold:
///
/// - `FunctionOfX` ⟹ `Implicit`
/// - `FunctionOfY` ⟹ `Implicit`
/// - `Implicit` ⟹ `Polar`
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RelationType {
    /// y is a function of x.
    /// More generally, the relation is of the form y R_1 f_1(x) ∨ … ∨ y R_n f_n(x).
    FunctionOfX,
    /// x is a function of y.
    /// More generally, the relation is of the form x R_1 f_1(y) ∨ … ∨ x R_n f_n(y).
    FunctionOfY,
    /// Implicit relation of x and y.
    Implicit,
    /// Implicit relation of x, y and θ.
    Polar,
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
    pub fn eval(
        &mut self,
        x: Interval,
        y: Interval,
        n_theta: Interval,
        cache: Option<&mut EvalCache>,
    ) -> EvalResult {
        self.eval_count += 1;
        match cache {
            Some(cache) => self.eval_with_cache(x, y, n_theta, cache),
            _ => self.eval_without_cache(x, y, n_theta),
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
        use RelationType::*;
        match self.forms[i].kind {
            StaticFormKind::Atomic(_, i, j) => {
                match (
                    &self.terms[i as usize].kind,
                    &self.terms[j as usize],
                    &self.terms[i as usize],
                    &self.terms[j as usize].kind,
                ) {
                    (StaticTermKind::Y, t, _, _) | (_, _, t, StaticTermKind::Y)
                        if VarSet::X.contains(t.vars) =>
                    {
                        // y = f(x) or f(x) = y
                        FunctionOfX
                    }
                    (StaticTermKind::X, t, _, _) | (_, _, t, StaticTermKind::X)
                        if VarSet::Y.contains(t.vars) =>
                    {
                        // x = f(y) or f(y) = x
                        FunctionOfY
                    }
                    (_, tj, ti, _)
                        if VarSet::XY.contains(ti.vars) && VarSet::XY.contains(tj.vars) =>
                    {
                        Implicit
                    }
                    _ => Polar,
                }
            }
            StaticFormKind::And(i, j) => {
                match (
                    self.relation_type_impl(i as usize),
                    self.relation_type_impl(j as usize),
                ) {
                    (Polar, _) | (_, Polar) => Polar,
                    _ => {
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
                        Implicit
                    }
                }
            }
            StaticFormKind::Or(i, j) => {
                match (
                    self.relation_type_impl(i as usize),
                    self.relation_type_impl(j as usize),
                ) {
                    (Polar, _) | (_, Polar) => Polar,
                    (x, y) if x == y => x,
                    _ => Implicit,
                }
            }
        }
    }

    fn eval_with_cache(
        &mut self,
        x: Interval,
        y: Interval,
        n_theta: Interval,
        cache: &mut EvalCache,
    ) -> EvalResult {
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
                StaticTermKind::X => t.put(ts, DecInterval::new(x).into()),
                StaticTermKind::Y => t.put(ts, DecInterval::new(y).into()),
                StaticTermKind::NTheta => t.put(ts, DecInterval::new(n_theta).into()),
                _ if t.vars == VarSet::EMPTY
                    || t.vars == VarSet::X && mx_ts.is_some()
                    || t.vars == VarSet::Y && my_ts.is_some() =>
                {
                    // Constant or cached subexpression.
                }
                _ => t.put_eval(terms, ts),
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

    fn eval_without_cache(&mut self, x: Interval, y: Interval, n_theta: Interval) -> EvalResult {
        let ts = &mut self.ts;
        let terms = &self.terms;
        for t in terms {
            match t.kind {
                StaticTermKind::X => t.put(ts, DecInterval::new(x).into()),
                StaticTermKind::Y => t.put(ts, DecInterval::new(y).into()),
                StaticTermKind::NTheta => t.put(ts, DecInterval::new(n_theta).into()),
                _ if t.vars == VarSet::EMPTY => {
                    // Constant subexpression.
                }
                _ => t.put_eval(terms, ts),
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
        expand_polar_coords(&mut e);
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

/// Replaces polar coordinates with the equivalent family of Cartesian coordinates.
fn expand_polar_coords(e: &mut Expr) {
    // e1 = e /. {r → sqrt(x^2 + y^2), θ → atan2(y, x) + 2π n_θ}.
    let mut e1 = e.clone();
    let mut v = ReplaceAll::new(|e| match &e.kind {
        ExprKind::Var(x) if x == "r" => Some(Expr::unary(
            UnaryOp::Sqrt,
            box Expr::binary(
                BinaryOp::Add,
                box Expr::unary(UnaryOp::Sqr, box Expr::var("x")),
                box Expr::unary(UnaryOp::Sqr, box Expr::var("y")),
            ),
        )),
        ExprKind::Var(x) if x == "theta" || x == "θ" => Some(Expr::binary(
            BinaryOp::Add,
            box Expr::binary(BinaryOp::Atan2, box Expr::var("y"), box Expr::var("x")),
            box Expr::binary(
                BinaryOp::Mul,
                box Expr::constant(DecInterval::TAU.into(), None),
                box Expr::var("<n-theta>"),
            ),
        )),
        _ => None,
    });
    v.visit_expr_mut(&mut e1);
    if !v.modified {
        // `e` does not contain r nor θ.
        return;
    }

    // e2 = e /. {r → -sqrt(x^2 + y^2), θ → atan2(y, x) + 2π (1/2 + n_θ)}.
    // θ can alternatively be replaced by atan2(-y, -x) + 2π n_θ,
    // which will be a little more precise for some n_θ,
    // but much slower since we have to evaluate `atan2` separately for `e1` and `e2`.
    let mut e2 = e.clone();
    let mut v = ReplaceAll::new(|e| match &e.kind {
        ExprKind::Var(x) if x == "r" => Some(Expr::unary(
            UnaryOp::Neg,
            box Expr::unary(
                UnaryOp::Sqrt,
                box Expr::binary(
                    BinaryOp::Add,
                    box Expr::unary(UnaryOp::Sqr, box Expr::var("x")),
                    box Expr::unary(UnaryOp::Sqr, box Expr::var("y")),
                ),
            ),
        )),
        ExprKind::Var(x) if x == "theta" || x == "θ" => Some(Expr::binary(
            BinaryOp::Add,
            box Expr::binary(BinaryOp::Atan2, box Expr::var("y"), box Expr::var("x")),
            box Expr::binary(
                BinaryOp::Mul,
                box Expr::constant(DecInterval::TAU.into(), None),
                box Expr::binary(
                    BinaryOp::Add,
                    box Expr::constant(const_dec_interval!(0.5, 0.5).into(), Some((1, 2).into())),
                    box Expr::var("<n-theta>"),
                ),
            ),
        )),
        _ => None,
    });
    v.visit_expr_mut(&mut e2);

    *e = Expr::binary(BinaryOp::Or, box e1, box e2)
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
        assert_eq!(t("r = 1"), Implicit);
        assert_eq!(t("x = θ"), Polar);
        assert_eq!(t("x = theta"), Polar);
        assert_eq!(t("x = θ || θ = x"), Polar);
        assert_eq!(t("x = θ && θ = x"), Polar);
    }
}
