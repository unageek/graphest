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
    collections::HashMap,
    mem::{size_of, take},
    str::FromStr,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EvalCacheLevel {
    PerAxis,
    Full,
}

/// A cache for evaluation results of an implicit relation.
pub struct EvalCache {
    level: EvalCacheLevel,
    cx: HashMap<Interval, Vec<TupperIntervalSet>>,
    cy: HashMap<Interval, Vec<TupperIntervalSet>>,
    cxy: HashMap<(Interval, Interval), EvalResult>,
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

    /// Clears the cache and releases the allocated memory.
    pub fn clear(&mut self) {
        self.cx = HashMap::new();
        self.cy = HashMap::new();
        self.cxy = HashMap::new();
        self.size_of_cx = 0;
        self.size_of_cy = 0;
        self.size_of_cxy = 0;
        self.size_of_values_in_heap = 0;
    }

    pub fn get_x(&self, x: Interval) -> Option<&Vec<TupperIntervalSet>> {
        self.cx.get(&x)
    }

    pub fn get_y(&self, y: Interval) -> Option<&Vec<TupperIntervalSet>> {
        self.cy.get(&y)
    }

    pub fn get_xy(&self, x: Interval, y: Interval) -> Option<&EvalResult> {
        match self.level {
            EvalCacheLevel::PerAxis => None,
            EvalCacheLevel::Full => self.cxy.get(&(x, y)),
        }
    }

    pub fn insert_x_with<F: FnOnce() -> Vec<TupperIntervalSet>>(&mut self, x: Interval, f: F) {
        let v = f();
        self.size_of_values_in_heap += v.capacity() * size_of::<TupperIntervalSet>()
            + v.iter().map(|t| t.size_in_heap()).sum::<usize>();
        self.cx.insert(x, v);
        self.size_of_cx = self.cx.capacity()
            * (size_of::<u64>() + size_of::<Interval>() + size_of::<Vec<TupperIntervalSet>>());
    }

    pub fn insert_y_with<F: FnOnce() -> Vec<TupperIntervalSet>>(&mut self, y: Interval, f: F) {
        let v = f();
        self.size_of_values_in_heap += v.capacity() * size_of::<TupperIntervalSet>()
            + v.iter().map(|t| t.size_in_heap()).sum::<usize>();
        self.cy.insert(y, v);
        self.size_of_cy = self.cy.capacity()
            * (size_of::<u64>() + size_of::<Interval>() + size_of::<Vec<TupperIntervalSet>>());
    }

    pub fn insert_xy_with<F: FnOnce() -> EvalResult>(&mut self, x: Interval, y: Interval, f: F) {
        if self.level == EvalCacheLevel::Full {
            let v = f();
            self.size_of_values_in_heap += v.size_in_heap();
            self.cxy.insert((x, y), v);
            self.size_of_cxy = self.cxy.capacity()
                * (size_of::<u64>() + size_of::<(Interval, Interval)>() + size_of::<EvalResult>());
        }
    }

    /// Returns the approximate size allocated by the [`EvalCache`] in bytes.
    pub fn size_in_heap(&self) -> usize {
        // This is a lowest bound, the actual size can be much larger.
        self.size_of_cx + self.size_of_cy + self.size_of_cxy + self.size_of_values_in_heap
    }
}

type EvalExplicitResult = (TupperIntervalSet, EvalResult);

#[derive(Default)]
pub struct EvalExplicitCache {
    ct: HashMap<Interval, EvalExplicitResult>,
    size_of_ct: usize,
    size_of_values_in_heap: usize,
}

impl EvalExplicitCache {
    /// Clears the cache and releases the allocated memory.
    pub fn clear(&mut self) {
        self.ct = HashMap::new();
        self.size_of_ct = 0;
        self.size_of_values_in_heap = 0;
    }

    pub fn get(&self, t: Interval) -> Option<&EvalExplicitResult> {
        self.ct.get(&(t))
    }

    pub fn insert(&mut self, t: Interval, r: EvalExplicitResult) {
        self.size_of_values_in_heap += r.0.size_in_heap() + r.1.size_in_heap();
        self.ct.insert(t, r);
        self.size_of_ct = self.ct.capacity()
            * (size_of::<u64>() + size_of::<Interval>() + size_of::<EvalParametricResult>());
    }

    /// Returns the approximate size allocated by the [`EvalFunctionCache`] in bytes.
    pub fn size_in_heap(&self) -> usize {
        self.size_of_ct + self.size_of_values_in_heap
    }
}

type EvalParametricResult = (TupperIntervalSet, TupperIntervalSet, EvalResult);

/// A cache for evaluation results of a parametric relation.
#[derive(Default)]
pub struct EvalParametricCache {
    ct: HashMap<Interval, EvalParametricResult>,
    size_of_ct: usize,
    size_of_values_in_heap: usize,
}

impl EvalParametricCache {
    /// Clears the cache and releases the allocated memory.
    pub fn clear(&mut self) {
        self.ct = HashMap::new();
        self.size_of_ct = 0;
        self.size_of_values_in_heap = 0;
    }

    pub fn get(&self, t: Interval) -> Option<&EvalParametricResult> {
        self.ct.get(&(t))
    }

    pub fn insert(&mut self, t: Interval, r: EvalParametricResult) {
        self.size_of_values_in_heap += r.0.size_in_heap() + r.1.size_in_heap() + r.2.size_in_heap();
        self.ct.insert(t, r);
        self.size_of_ct = self.ct.capacity()
            * (size_of::<u64>() + size_of::<Interval>() + size_of::<EvalParametricResult>());
    }

    /// Returns the approximate size allocated by the [`EvalParametricCache`] in bytes.
    pub fn size_in_heap(&self) -> usize {
        self.size_of_ct + self.size_of_values_in_heap
    }
}

/// The type of a [`Relation`], which should be used when choosing the optimal graphing algorithm.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RelationType {
    /// A relation of the form y = f(x) ∧ P(x), where P(x) is an optional constraint on x.
    ExplicitFunctionOfX,
    /// A relation of the form x = f(y) ∧ P(y), where P(y) is an optional constraint on y.
    ExplicitFunctionOfY,
    /// y is a function of x.
    /// More generally, the relation is of the form y R_1 f_1(x) ∨ … ∨ y R_n f_n(x).
    FunctionOfX,
    /// x is a function of y.
    /// More generally, the relation is of the form x R_1 f_1(y) ∨ … ∨ x R_n f_n(y).
    FunctionOfY,
    /// An implicit relation.
    Implicit,
    /// A relation of the form x = f(t) ∧ y = g(t) ∧ P(t),
    /// where P(t) is an optional constraint on t.
    Parametric,
}

pub struct RelationArgs {
    pub x: Interval,
    pub y: Interval,
    pub n_theta: Interval,
    pub t: Interval,
}

/// A mathematical relation whose graph is to be plotted.
#[derive(Clone, Debug)]
pub struct Relation {
    terms: Vec<StaticTerm>,
    forms: Vec<StaticForm>,
    n_atom_forms: usize,
    ts: ValueStore<TupperIntervalSet>,
    eval_count: usize,
    x_explicit: Option<StoreIndex>,
    y_explicit: Option<StoreIndex>,
    mx: Vec<StoreIndex>,
    my: Vec<StoreIndex>,
    has_n_theta: bool,
    has_t: bool,
    n_theta_range: Interval,
    t_range: Interval,
    relation_type: RelationType,
}

impl Relation {
    /// Evaluates the relation with the given arguments.
    ///
    /// Precondition: `cache` has never been passed to other relations.
    pub fn eval(&mut self, args: &RelationArgs, cache: Option<&mut EvalCache>) -> EvalResult {
        self.eval_count += 1;
        match cache {
            Some(cache) => self.eval_with_cache(args, cache),
            _ => self.eval_without_cache(args),
        }
    }

    /// Returns the number of calls of `self.eval` that have been made thus far.
    pub fn eval_count(&self) -> usize {
        self.eval_count
    }

    /// Evaluates the explicit relation y = f(x) ∧ P(x) and returns (f(x), P(x)).
    ///
    /// If P(x) is absent, its value is assumed to be always true.
    pub fn eval_explicit(
        &mut self,
        x: Interval,
        cache: Option<&mut EvalExplicitCache>,
    ) -> EvalExplicitResult {
        assert!(matches!(
            self.relation_type,
            RelationType::ExplicitFunctionOfX | RelationType::ExplicitFunctionOfY
        ));

        match cache {
            Some(cache) => match cache.get(x) {
                Some(r) => r.clone(),
                _ => {
                    let r = self.eval_explicit_without_cache(x);
                    cache.insert(x, r.clone());
                    r
                }
            },
            _ => self.eval_explicit_without_cache(x),
        }
    }

    /// Evaluates the parametric relation x = f(t) ∧ y = g(t) ∧ P(t) and returns (f(t), g(t), P(t)).
    ///
    /// If P(t) is absent, its value is assumed to be always true.
    ///
    /// Precondition: `cache` has never been passed to other relations.
    pub fn eval_parametric(
        &mut self,
        t: Interval,
        cache: Option<&mut EvalParametricCache>,
    ) -> EvalParametricResult {
        assert_eq!(self.relation_type, RelationType::Parametric);

        match cache {
            Some(cache) => match cache.get(t) {
                Some(r) => r.clone(),
                _ => {
                    let r = self.eval_parametric_without_cache(t);
                    cache.insert(t, r.clone());
                    r
                }
            },
            _ => self.eval_parametric_without_cache(t),
        }
    }

    pub fn forms(&self) -> &Vec<StaticForm> {
        &self.forms
    }

    /// Returns `true` if the relation contains n_θ.
    pub fn has_n_theta(&self) -> bool {
        self.has_n_theta
    }

    /// Returns `true` if the relation contains t.
    pub fn has_t(&self) -> bool {
        self.has_t
    }

    /// Returns the range of n_θ that needs to be covered to plot the graph of the relation.
    ///
    /// Each of the bounds is either an integer or ±∞.
    pub fn n_theta_range(&self) -> Interval {
        self.n_theta_range
    }

    /// Returns the type of the relation.
    pub fn relation_type(&self) -> RelationType {
        self.relation_type
    }

    /// Returns the range of t that needs to be covered to plot the graph of the relation.
    pub fn t_range(&self) -> Interval {
        self.t_range
    }

    fn eval_with_cache(&mut self, args: &RelationArgs, cache: &mut EvalCache) -> EvalResult {
        let x = args.x;
        let y = args.y;

        if let Some(r) = cache.get_xy(x, y) {
            return r.clone();
        }

        let terms = &self.terms;
        let ts = &mut self.ts;
        let mut mx_ts_is_cached = false;
        let mut my_ts_is_cached = false;
        {
            let mx_ts = cache.get_x(x);
            let my_ts = cache.get_y(y);
            if let Some(mx_ts) = mx_ts {
                for (i, &mx) in self.mx.iter().enumerate() {
                    ts[mx] = mx_ts[i].clone();
                }
                mx_ts_is_cached = true;
            }
            if let Some(my_ts) = my_ts {
                for (i, &my) in self.my.iter().enumerate() {
                    ts[my] = my_ts[i].clone();
                }
                my_ts_is_cached = true;
            }
        }

        for t in terms {
            match t.kind {
                StaticTermKind::X => t.put(ts, DecInterval::new(x).into()),
                StaticTermKind::Y => t.put(ts, DecInterval::new(y).into()),
                StaticTermKind::NTheta => t.put(ts, DecInterval::new(args.n_theta).into()),
                StaticTermKind::T => t.put(ts, DecInterval::new(args.t).into()),
                _ if t.vars == VarSet::EMPTY
                    || t.vars == VarSet::X && mx_ts_is_cached
                    || t.vars == VarSet::Y && my_ts_is_cached =>
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
        if !mx_ts_is_cached {
            cache.insert_x_with(x, || self.mx.iter().map(|&i| ts[i].clone()).collect());
        }
        if !my_ts_is_cached {
            cache.insert_y_with(y, || self.my.iter().map(|&i| ts[i].clone()).collect());
        }
        cache.insert_xy_with(x, y, || r.clone());
        r
    }

    fn eval_without_cache(&mut self, args: &RelationArgs) -> EvalResult {
        let ts = &mut self.ts;
        let terms = &self.terms;
        for t in terms {
            match t.kind {
                StaticTermKind::X => t.put(ts, DecInterval::new(args.x).into()),
                StaticTermKind::Y => t.put(ts, DecInterval::new(args.y).into()),
                StaticTermKind::NTheta => t.put(ts, DecInterval::new(args.n_theta).into()),
                StaticTermKind::T => t.put(ts, DecInterval::new(args.t).into()),
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

    fn eval_explicit_without_cache(&mut self, x: Interval) -> EvalExplicitResult {
        match self.relation_type {
            RelationType::ExplicitFunctionOfX => {
                let p = self.eval(
                    &RelationArgs {
                        x,
                        y: Interval::ENTIRE,
                        n_theta: Interval::ENTIRE,
                        t: Interval::ENTIRE,
                    },
                    None,
                );

                (self.ts[self.y_explicit.unwrap()].clone(), p)
            }
            RelationType::ExplicitFunctionOfY => {
                let p = self.eval(
                    &RelationArgs {
                        x: Interval::ENTIRE,
                        y: x,
                        n_theta: Interval::ENTIRE,
                        t: Interval::ENTIRE,
                    },
                    None,
                );

                (self.ts[self.x_explicit.unwrap()].clone(), p)
            }
            _ => panic!(),
        }
    }

    fn eval_parametric_without_cache(&mut self, t: Interval) -> EvalParametricResult {
        let p = self.eval(
            &RelationArgs {
                x: Interval::ENTIRE,
                y: Interval::ENTIRE,
                n_theta: Interval::ENTIRE,
                t,
            },
            None,
        );

        (
            self.ts[self.x_explicit.unwrap()].clone(),
            self.ts[self.y_explicit.unwrap()].clone(),
            p,
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
        let relation_type = relation_type(&mut e);
        PreTransform.visit_expr_mut(&mut e);
        simplify(&mut e);

        let n_theta_range = {
            let period = function_period(&e, VarSet::N_THETA);
            if let Some(period) = &period {
                if *period == 0 {
                    const_interval!(0.0, 0.0)
                } else {
                    interval!(&format!("[0,{}]", Integer::from(period - 1))).unwrap()
                }
            } else {
                Interval::ENTIRE
            }
        };
        assert_eq!(n_theta_range.trunc(), n_theta_range);

        let t_range = {
            let period = function_period(&e, VarSet::T);
            if let Some(period) = &period {
                Interval::TAU * interval!(&format!("[0,{}]", period)).unwrap()
            } else {
                Interval::ENTIRE
            }
        };

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

        let mut v = FindExplicitRelation::new(&collector, VarSet::X);
        v.visit_expr(&e);
        let x_explicit = v.get();

        let mut v = FindExplicitRelation::new(&collector, VarSet::Y);
        v.visit_expr(&e);
        let y_explicit = v.get();

        let mut v = FindMaximalScalarTerms::new(collector);
        v.visit_expr(&e);
        let (mx, my) = v.mx_my();

        let mut slf = Self {
            terms,
            forms,
            n_atom_forms,
            ts: ValueStore::new(TupperIntervalSet::new(), n_terms),
            eval_count: 0,
            x_explicit,
            y_explicit,
            mx,
            my,
            has_n_theta: e.vars.contains(VarSet::N_THETA),
            has_t: e.vars.contains(VarSet::T),
            n_theta_range,
            t_range,
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

/// Returns the period of a function of a variable t in multiples of 2π,
/// i.e., an integer p that satisfies (e /. t → t + 2π p) = e.
/// If the period is 0, the expression is independent of the variable.
///
/// Precondition: `e` has been pre-transformed and simplified.
fn function_period(e: &Expr, variable: VarSet) -> Option<Integer> {
    use {NaryOp::*, UnaryOp::*};

    match e {
        constant!(_) => Some(0.into()),
        x @ var!(_) if x.vars.contains(variable) => None,
        var!(_) => Some(0.into()),
        unary!(op, x) => {
            if let Some(p) = function_period(x, variable) {
                Some(p)
            } else if matches!(op, Cos | Sin | Tan) {
                match x {
                    x @ var!(_) if x.vars.contains(variable) => {
                        // op(θ)
                        Some(1.into())
                    }
                    nary!(Plus, xs) => match &xs[..] {
                        [constant!(_), x @ var!(_)] if x.vars.contains(variable) => {
                            // op(b + θ)
                            Some(1.into())
                        }
                        [constant!(_), nary!(Times, xs)] => match &xs[..] {
                            [constant!(a), x @ var!(_)] if x.vars.contains(variable) => {
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
                        [constant!(a), x @ var!(_)] if x.vars.contains(variable) => {
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
            let xp = function_period(x, variable)?;
            let yp = function_period(y, variable)?;
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
            .map(|x| function_period(x, variable))
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

struct ExplicitRelationParts {
    y: Option<Expr>, // y = f(x)
    px: Vec<Expr>,   // P(x)
}

/// Tries to identify `e` as an explicit relation.
fn normalize_explicit_relation(e: &mut Expr, y_var: VarSet, x_var: VarSet) -> bool {
    use BinaryOp::*;

    let mut parts = ExplicitRelationParts {
        y: None,
        px: vec![],
    };

    if !normalize_explicit_relation_impl(&mut e.clone(), &mut parts, y_var, x_var) {
        return false;
    }

    if let (Some(y), px) = (parts.y, parts.px) {
        let mut conjuncts = vec![box y];
        conjuncts.extend(px.into_iter().map(|e| box e));
        let mut it = conjuncts.into_iter();
        let first = it.next().unwrap();
        *e = *it.fold(first, |acc, e| box Expr::binary(And, acc, e));
        UpdateMetadata.visit_expr_mut(e);
        true
    } else {
        false
    }
}

fn normalize_explicit_relation_impl(
    e: &mut Expr,
    parts: &mut ExplicitRelationParts,
    y_var: VarSet,
    x_var: VarSet,
) -> bool {
    use BinaryOp::*;
    match e {
        binary!(And, e1, e2) => {
            normalize_explicit_relation_impl(e1, parts, y_var, x_var)
                && normalize_explicit_relation_impl(e2, parts, y_var, x_var)
        }
        binary!(Eq, y @ var!(_), e) | binary!(Eq, e, y @ var!(_))
            if y.vars == y_var && x_var.contains(e.vars) =>
        {
            parts.y.is_none() && {
                parts.y = Some(Expr::binary(ExplicitEq, box take(y), box take(e)));
                true
            }
        }
        e if VarSet::X.contains(e.vars) => {
            parts.px.push(take(e));
            true
        }
        _ => false,
    }
}

struct ParametricRelationParts {
    xt: Option<Expr>, // x = f(t)
    yt: Option<Expr>, // y = f(t)
    pt: Vec<Expr>,    // P(t)
}

/// Tries to identify `e` as a parametric relation.
fn normalize_parametric_relation(e: &mut Expr) -> bool {
    use BinaryOp::*;

    let mut parts = ParametricRelationParts {
        xt: None,
        yt: None,
        pt: vec![],
    };

    if !e.vars.contains(VarSet::T)
        || !normalize_parametric_relation_impl(&mut e.clone(), &mut parts)
    {
        return false;
    }

    if let (Some(xt), Some(yt), pt) = (parts.xt, parts.yt, parts.pt) {
        let mut conjuncts = vec![box xt, box yt];
        conjuncts.extend(pt.into_iter().map(|e| box e));
        let mut it = conjuncts.into_iter();
        let first = it.next().unwrap();
        *e = *it.fold(first, |acc, e| box Expr::binary(And, acc, e));
        UpdateMetadata.visit_expr_mut(e);
        true
    } else {
        false
    }
}

fn normalize_parametric_relation_impl(e: &mut Expr, parts: &mut ParametricRelationParts) -> bool {
    use BinaryOp::*;
    match e {
        binary!(And, e1, e2) => {
            normalize_parametric_relation_impl(e1, parts)
                && normalize_parametric_relation_impl(e2, parts)
        }
        binary!(Eq, x @ var!(_), e) | binary!(Eq, e, x @ var!(_))
            if x.vars == VarSet::X && VarSet::T.contains(e.vars) =>
        {
            parts.xt.is_none() && {
                parts.xt = Some(Expr::binary(ExplicitEq, box take(x), box take(e)));
                true
            }
        }
        binary!(Eq, y @ var!(_), e) | binary!(Eq, e, y @ var!(_))
            if y.vars == VarSet::Y && VarSet::T.contains(e.vars) =>
        {
            parts.yt.is_none() && {
                parts.yt = Some(Expr::binary(ExplicitEq, box take(y), box take(e)));
                true
            }
        }
        e if VarSet::T.contains(e.vars) => {
            parts.pt.push(take(e));
            true
        }
        _ => false,
    }
}

macro_rules! rel_op {
    () => {
        Eq | Ge | Gt | Le | Lt | Neq | Nge | Ngt | Nle | Nlt
    };
}

/// Determines the type of the relation. If it is [`RelationType::Parametric`],
/// normalizes explicit parts of the relation to the form `(ExplicitEq x e)`.
///
/// Precondition: [`EliminateNot`] has been applied.
fn relation_type(e: &mut Expr) -> RelationType {
    use {BinaryOp::*, RelationType::*};

    if normalize_explicit_relation(e, VarSet::Y, VarSet::X) {
        return ExplicitFunctionOfX;
    }

    if normalize_explicit_relation(e, VarSet::X, VarSet::Y) {
        return ExplicitFunctionOfY;
    }

    if normalize_parametric_relation(e) {
        return Parametric;
    }

    match e {
        binary!(rel_op!(), y @ var!(_), f_x) | binary!(rel_op!(), f_x, y @ var!(_))
            if y.vars == VarSet::Y && VarSet::X.contains(f_x.vars) =>
        {
            // y = f(x) or f(x) = y
            FunctionOfX
        }
        binary!(rel_op!(), x @ var!(_), f_y) | binary!(rel_op!(), f_y, x @ var!(_))
            if x.vars == VarSet::X && VarSet::Y.contains(f_y.vars) =>
        {
            // x = f(y) or f(y) = x
            FunctionOfY
        }
        binary!(rel_op!(), _, _) => Implicit,
        binary!(
            And,
            (binary!(Eq, x @ var!(_), f_t) | binary!(Eq, f_t, x @ var!(_))),
            (binary!(Eq, y @ var!(_), g_t) | binary!(Eq, g_t, y @ var!(_)))
        ) if x.vars | y.vars == VarSet::X | VarSet::Y && f_t.vars | g_t.vars == VarSet::T => {
            *e = if x.vars == VarSet::X {
                Expr::binary(
                    And,
                    box Expr::binary(ExplicitEq, box take(x), box take(f_t)),
                    box Expr::binary(ExplicitEq, box take(y), box take(g_t)),
                )
            } else {
                Expr::binary(
                    And,
                    box Expr::binary(ExplicitEq, box take(y), box take(g_t)),
                    box Expr::binary(ExplicitEq, box take(x), box take(f_t)),
                )
            };
            // x = f(t) ∧ y = g(t)
            Parametric
        }
        binary!(And, _, _) => {
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
        binary!(Or, e1, e2) => match (relation_type(e1), relation_type(e2)) {
            (FunctionOfX, FunctionOfX) => FunctionOfX,
            (FunctionOfY, FunctionOfY) => FunctionOfY,
            _ => Implicit,
        },
        _ => panic!(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn has_n_theta() {
        fn f(rel: &str) -> bool {
            rel.parse::<Relation>().unwrap().has_n_theta()
        }

        assert!(!f("x y r t = 0"));
        assert!(f("θ = 0"));
    }

    #[test]
    fn has_t() {
        fn f(rel: &str) -> bool {
            rel.parse::<Relation>().unwrap().has_t()
        }

        assert!(!f("x y r θ = 0"));
        assert!(f("t = 0"));
    }

    #[test]
    fn n_theta_range() {
        fn f(rel: &str) -> Interval {
            rel.parse::<Relation>().unwrap().n_theta_range()
        }

        assert_eq!(f("x y r t = 0"), const_interval!(0.0, 0.0));
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
        assert_eq!(f("y = 0"), ExplicitFunctionOfX);
        assert_eq!(f("0 = y"), ExplicitFunctionOfX);
        assert_eq!(f("y = sin(x)"), ExplicitFunctionOfX);
        assert_eq!(f("!(y = sin(x))"), FunctionOfX);
        assert_eq!(f("x = 0"), ExplicitFunctionOfY);
        assert_eq!(f("0 = x"), ExplicitFunctionOfY);
        assert_eq!(f("x = sin(y)"), ExplicitFunctionOfY);
        assert_eq!(f("!(x = sin(y))"), FunctionOfY);
        assert_eq!(f("x y = 0"), Implicit);
        assert_eq!(f("y = sin(x y)"), Implicit);
        assert_eq!(f("sin(x) = 0"), Implicit);
        assert_eq!(f("sin(y) = 0"), Implicit);
        assert_eq!(f("y = sin(x) && y = cos(x)"), Implicit);
        assert_eq!(f("y = sin(x) || y = cos(x)"), Implicit);
        assert_eq!(f("!(y = sin(x) && y = cos(x))"), FunctionOfX);
        assert_eq!(f("!(y = sin(x) || y = cos(x))"), Implicit);
        assert_eq!(f("r = 1"), Implicit);
        assert_eq!(f("x = θ"), Implicit);
        assert_eq!(f("x = theta"), Implicit);
        assert_eq!(f("x = 1 && y = sin(t)"), Parametric);
        assert_eq!(f("x = cos(t) && y = 1"), Parametric);
        assert_eq!(f("x = cos(t) && y = sin(t)"), Parametric);
        assert_eq!(f("sin(t) = y && cos(t) = x"), Parametric);
        assert_eq!(f("x = cos(t) && y = sin(t) && 0 < t < 1"), Parametric);
        assert_eq!(f("0 < t < 1 && sin(t) = y && cos(t) = x"), Parametric);
        assert_eq!(f("x = t && y = t && x = 2t"), Implicit);
    }

    #[test]
    fn t_range() {
        fn f(rel: &str) -> Interval {
            rel.parse::<Relation>().unwrap().t_range()
        }

        assert_eq!(f("x y r θ = 0"), const_interval!(0.0, 0.0));
        assert_eq!(f("t = 0"), Interval::ENTIRE);
        assert_eq!(
            f("sin(t) = 0"),
            interval!(0.0, Interval::TAU.sup()).unwrap()
        );
        assert_eq!(
            f("sin(3/5t) = 0"),
            interval!(0.0, (const_interval!(5.0, 5.0) * Interval::TAU).sup()).unwrap()
        );
    }
}
