use crate::{
    ast::{BinaryOp, Expr, NaryOp, UnaryOp, ValueType, VarSet},
    binary, constant,
    context::Context,
    eval_result::EvalResult,
    interval_set::TupperIntervalSet,
    nary,
    ops::{StaticForm, StaticFormKind, StaticTerm, StaticTermKind, StoreIndex, ValueStore},
    parse::parse_expr,
    unary, var,
    visit::*,
};
use inari::{const_interval, interval, DecInterval, Interval};
use rug::Integer;
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
    n_theta_range: Interval,
    relation_type: RelationType,
}

impl Relation {
    /// Evaluates the relation with the given arguments.
    ///
    /// Precondition: `cache` has never been passed to other relations.
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

    /// Returns the number of calls of `self.eval` that have been made thus far.
    pub fn eval_count(&self) -> usize {
        self.eval_count
    }

    pub fn forms(&self) -> &Vec<StaticForm> {
        &self.forms
    }

    /// Returns the range of n_θ that needs to be covered to plot the graph of the relation.
    ///
    /// Each endpoint is either an integer or ±∞.
    pub fn n_theta_range(&self) -> Interval {
        self.n_theta_range
    }

    /// Returns the type of the relation.
    pub fn relation_type(&self) -> RelationType {
        self.relation_type
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
            for (i, &mx) in self.mx.iter().enumerate() {
                ts[mx] = mx_ts[i].clone();
            }
        }
        if let Some(my_ts) = my_ts {
            for (i, &my) in self.my.iter().enumerate() {
                ts[my] = my_ts[i].clone();
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
                _ => t.put_eval(ts),
            }
        }

        let r = EvalResult(
            self.forms[..self.n_atom_forms]
                .iter()
                .map(|f| f.eval(ts))
                .collect(),
        );

        let ts = &self.ts;
        cache.insert_x_with(kx, || self.mx.iter().map(|&i| ts[i].clone()).collect());
        cache.insert_y_with(ky, || self.my.iter().map(|&i| ts[i].clone()).collect());
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
                _ => t.put_eval(ts),
            }
        }

        EvalResult(
            self.forms[..self.n_atom_forms]
                .iter()
                .map(|f| f.eval(ts))
                .collect(),
        )
    }

    fn initialize(&mut self) {
        for t in &self.terms {
            // This condition is different from `let StaticTermKind::Constant(_) = t.kind`,
            // as not all constant expressions are folded. See the comment on [`FoldConstant`].
            if t.vars == VarSet::EMPTY {
                t.put_eval(&mut self.ts);
            }
        }
    }
}

impl FromStr for Relation {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, String> {
        let mut e = parse_expr(s, Context::builtin_context())?;
        // TODO: Check types and return a pretty error message.
        loop {
            let mut v = EliminateNot::default();
            v.visit_expr_mut(&mut e);
            if !v.modified {
                break;
            }
        }
        UpdateMetadata.visit_expr_mut(&mut e);
        let relation_type = relation_type(&e);
        PreTransform.visit_expr_mut(&mut e);
        simplify(&mut e);
        let period = polar_period(&e);
        let n_theta_range = if let Some(period) = &period {
            if *period == 0 {
                const_interval!(0.0, 0.0)
            } else {
                interval!(&format!("[0,{}]", Integer::from(period - 1))).unwrap()
            }
        } else {
            Interval::ENTIRE
        };
        assert_eq!(n_theta_range.trunc(), n_theta_range);
        expand_polar_coords(&mut e);
        simplify(&mut e);
        SubDivTransform.visit_expr_mut(&mut e);
        simplify(&mut e);
        PostTransform.visit_expr_mut(&mut e);
        FuseMulAdd.visit_expr_mut(&mut e);
        UpdateMetadata.visit_expr_mut(&mut e);
        if e.ty != ValueType::Boolean {
            return Err("the relation must be a Boolean expression".into());
        }
        let mut v = AssignId::new();
        v.visit_expr_mut(&mut e);
        let collector = CollectStatic::new(v);
        let terms = collector.terms.clone();
        let forms = collector.forms.clone();
        let n_terms = terms.len();
        let n_atom_forms = forms
            .iter()
            .filter(|f| matches!(f.kind, StaticFormKind::Atomic(_, _)))
            .count();

        let mut v = FindMaximalScalarTerms::new(collector);
        v.visit_expr(&e);
        let (mx, my) = v.mx_my();

        let mut slf = Self {
            terms,
            forms,
            n_atom_forms,
            ts: ValueStore::new(TupperIntervalSet::new(), n_terms),
            eval_count: 0,
            mx,
            my,
            n_theta_range,
            relation_type,
        };
        slf.initialize();
        Ok(slf)
    }
}

fn simplify(e: &mut Expr) {
    loop {
        let mut fl = Flatten::default();
        fl.visit_expr_mut(e);
        let mut s = SortTerms::default();
        s.visit_expr_mut(e);
        let mut f = FoldConstant::default();
        f.visit_expr_mut(e);
        let mut t = Transform::default();
        t.visit_expr_mut(e);
        if !fl.modified && !s.modified && !f.modified && !t.modified {
            break;
        }
    }
}

/// Transforms an expression that contains r or θ into the equivalent expression
/// that contains only x, y and n_θ. When the result contains n_θ,
/// it actually represents a disjunction of expressions indexed by n_θ.
///
/// Precondition: `e` has been pre-transformed and simplified.
fn expand_polar_coords(e: &mut Expr) {
    use {BinaryOp::*, NaryOp::*};

    // e1 = e /. {r → sqrt(x^2 + y^2), θ → atan2(y, x) + 2π n_θ}.
    let mut e1 = e.clone();
    let mut v = ReplaceAll::new(|e| match e {
        var!(x) if x == "r" => Some(Expr::binary(
            Pow,
            box Expr::nary(
                Plus,
                vec![
                    Expr::binary(Pow, box Expr::var("x"), box Expr::two()),
                    Expr::binary(Pow, box Expr::var("y"), box Expr::two()),
                ],
            ),
            box Expr::one_half(),
        )),
        var!(x) if x == "theta" || x == "θ" => Some(Expr::nary(
            Plus,
            vec![
                Expr::binary(Atan2, box Expr::var("y"), box Expr::var("x")),
                Expr::nary(
                    Times,
                    vec![
                        Expr::constant(DecInterval::TAU.into(), None),
                        Expr::var("<n-theta>"),
                    ],
                ),
            ],
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
    let mut v = ReplaceAll::new(|e| match e {
        var!(x) if x == "r" => Some(Expr::unary(
            UnaryOp::Neg,
            box Expr::binary(
                Pow,
                box Expr::nary(
                    Plus,
                    vec![
                        Expr::binary(Pow, box Expr::var("x"), box Expr::two()),
                        Expr::binary(Pow, box Expr::var("y"), box Expr::two()),
                    ],
                ),
                box Expr::one_half(),
            ),
        )),
        var!(x) if x == "theta" || x == "θ" => Some(Expr::nary(
            Plus,
            vec![
                Expr::binary(Atan2, box Expr::var("y"), box Expr::var("x")),
                Expr::nary(
                    Times,
                    vec![
                        Expr::constant(DecInterval::TAU.into(), None),
                        Expr::nary(Plus, vec![Expr::one_half(), Expr::var("<n-theta>")]),
                    ],
                ),
            ],
        )),
        _ => None,
    });
    v.visit_expr_mut(&mut e2);

    *e = Expr::binary(BinaryOp::Or, box e1, box e2);
}

/// Returns the period of a function of θ in multiples of 2π, i.e., any integer p that satisfies
/// (e /. θ → θ + 2π p) = e. If the period is 0, the expression is independent of θ.
///
/// Precondition: `e` has been pre-transformed and simplified.
pub fn polar_period(e: &Expr) -> Option<Integer> {
    use {NaryOp::*, UnaryOp::*};

    match e {
        constant!(_) => Some(0.into()),
        var!(name) if name == "theta" || name == "θ" => None,
        var!(_) => Some(0.into()),
        unary!(op, x) => {
            if let Some(p) = polar_period(x) {
                Some(p)
            } else if matches!(op, Cos | Sin | Tan) {
                match x {
                    var!(name) if name == "theta" || name == "θ" => {
                        // op(θ)
                        Some(1.into())
                    }
                    nary!(Plus, xs) => match &xs[..] {
                        [constant!(_), var!(name)] if name == "theta" || name == "θ" => {
                            // op(b + θ)
                            Some(1.into())
                        }
                        [constant!(_), nary!(Times, xs)] => match &xs[..] {
                            [constant!(a), var!(name)] if name == "theta" || name == "θ" => {
                                // op(b + a θ)
                                if let Some(a) = &a.1 {
                                    let p = a.denom().clone();
                                    if *op == Tan && p.is_divisible_u(2) {
                                        Some(p.div_exact_u(2))
                                    } else {
                                        Some(p)
                                    }
                                } else {
                                    None
                                }
                            }
                            _ => None,
                        },
                        _ => None,
                    },
                    nary!(Times, xs) => match &xs[..] {
                        [constant!(a), var!(name)] if name == "theta" || name == "θ" => {
                            // op(a θ)
                            if let Some(a) = &a.1 {
                                let p = a.denom().clone();
                                if *op == Tan && p.is_divisible_u(2) {
                                    Some(p.div_exact_u(2))
                                } else {
                                    Some(p)
                                }
                            } else {
                                None
                            }
                        }
                        _ => None,
                    },
                    _ => None,
                }
            } else {
                None
            }
        }
        binary!(_, x, y) => {
            let xp = polar_period(x)?;
            let yp = polar_period(y)?;
            Some(if xp == 0 {
                yp
            } else if yp == 0 {
                xp
            } else {
                xp.lcm(&yp)
            })
        }
        nary!(_, xs) => xs
            .iter()
            .map(|x| polar_period(x))
            .collect::<Option<Vec<_>>>()
            .map(|ps| {
                ps.into_iter().fold(Integer::from(0), |xp, yp| {
                    if xp == 0 {
                        yp
                    } else if yp == 0 {
                        xp
                    } else {
                        xp.lcm(&yp)
                    }
                })
            }),
        _ => panic!("unexpected kind of expression"),
    }
}

macro_rules! rel_op {
    () => {
        Eq | Ge | Gt | Le | Lt | Neq | Nge | Ngt | Nle | Nlt
    };
}

/// Returns the type of the relation.
///
/// Precondition: [`EliminateNot`] has been applied.
pub fn relation_type(e: &Expr) -> RelationType {
    use {BinaryOp::*, RelationType::*};
    match e {
        binary!(rel_op!(), var!(name), e) | binary!(rel_op!(), e, var!(name))
            if name == "y" && VarSet::X.contains(e.vars) =>
        {
            // y = f(x) or f(x) = y
            FunctionOfX
        }
        binary!(rel_op!(), var!(name), e) | binary!(rel_op!(), e, var!(name))
            if name == "x" && VarSet::Y.contains(e.vars) =>
        {
            // x = f(y) or f(y) = x
            FunctionOfY
        }
        binary!(rel_op!(), e1, e2)
            if VarSet::XY.contains(e1.vars) && VarSet::XY.contains(e2.vars) =>
        {
            Implicit
        }
        binary!(rel_op!(), _, _) => Polar,
        binary!(And, e1, e2) => {
            match (relation_type(e1), relation_type(e2)) {
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
        binary!(Or, e1, e2) => match (relation_type(e1), relation_type(e2)) {
            (Polar, _) | (_, Polar) => Polar,
            (x, y) if x == y => x,
            _ => Implicit,
        },
        _ => panic!(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn n_theta_range() {
        fn f(rel: &str) -> Interval {
            rel.parse::<Relation>().unwrap().n_theta_range()
        }

        assert_eq!(f("42 = 0"), const_interval!(0.0, 0.0));
        assert_eq!(f("x = 0"), const_interval!(0.0, 0.0));
        assert_eq!(f("y = 0"), const_interval!(0.0, 0.0));
        assert_eq!(f("r = 0"), const_interval!(0.0, 0.0));
        assert_eq!(f("θ = 0"), Interval::ENTIRE);
        assert_eq!(f("sin(θ) = 0"), const_interval!(0.0, 0.0));
        assert_eq!(f("cos(θ) = 0"), const_interval!(0.0, 0.0));
        assert_eq!(f("tan(θ) = 0"), const_interval!(0.0, 0.0));
        assert_eq!(f("sin(3/5θ) = 0"), const_interval!(0.0, 4.0));
        assert_eq!(f("cos(3/5θ) = 0"), const_interval!(0.0, 4.0));
        assert_eq!(f("tan(3/5θ) = 0"), const_interval!(0.0, 4.0));
        assert_eq!(f("sin(5/6θ) = 0"), const_interval!(0.0, 5.0));
        assert_eq!(f("cos(5/6θ) = 0"), const_interval!(0.0, 5.0));
        assert_eq!(f("tan(5/6θ) = 0"), const_interval!(0.0, 2.0));
        assert_eq!(f("sqrt(sin(θ)) = 0"), const_interval!(0.0, 0.0));
        assert_eq!(f("sin(θ) + θ = 0"), Interval::ENTIRE);
        assert_eq!(f("min(sin(θ), θ) = 0"), Interval::ENTIRE);
        assert_eq!(f("r = sin(θ) = 0"), const_interval!(0.0, 0.0));
        assert_eq!(f("sin(3θ/5) = 0"), const_interval!(0.0, 4.0));
        assert_eq!(f("sin(3θ/5 + 2) = 0"), const_interval!(0.0, 4.0));
        assert_eq!(f("sin(θ/2) + cos(θ/3) = 0"), const_interval!(0.0, 5.0));
        assert_eq!(f("min(sin(θ/2), cos(θ/3)) = 0"), const_interval!(0.0, 5.0));
    }

    #[test]
    fn relation_type() {
        use RelationType::*;

        fn f(rel: &str) -> RelationType {
            rel.parse::<Relation>().unwrap().relation_type()
        }

        assert_eq!(f("1 < 2"), Implicit);
        assert_eq!(f("y = 0"), FunctionOfX);
        assert_eq!(f("0 = y"), FunctionOfX);
        assert_eq!(f("y = sin(x)"), FunctionOfX);
        assert_eq!(f("!(y = sin(x))"), FunctionOfX);
        assert_eq!(f("x = 0"), FunctionOfY);
        assert_eq!(f("0 = x"), FunctionOfY);
        assert_eq!(f("x = sin(y)"), FunctionOfY);
        assert_eq!(f("!(x = sin(y))"), FunctionOfY);
        assert_eq!(f("x y = 0"), Implicit);
        assert_eq!(f("y = sin(x y)"), Implicit);
        assert_eq!(f("sin(x) = 0"), Implicit);
        assert_eq!(f("sin(y) = 0"), Implicit);
        assert_eq!(f("y = sin(x) && y = cos(x)"), Implicit);
        assert_eq!(f("y = sin(x) || y = cos(x)"), FunctionOfX);
        assert_eq!(f("!(y = sin(x) && y = cos(x))"), FunctionOfX);
        assert_eq!(f("!(y = sin(x) || y = cos(x))"), Implicit);
        assert_eq!(f("r = 1"), Implicit);
        assert_eq!(f("x = θ"), Polar);
        assert_eq!(f("x = theta"), Polar);
        assert_eq!(f("x = sin(θ) && r = cos(θ)"), Polar);
        assert_eq!(f("x = sin(θ) || r = cos(θ)"), Polar);
    }
}
