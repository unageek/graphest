use crate::{
    ast::{BinaryOp, Expr, NaryOp, UnaryOp, ValueType},
    binary, bool_constant, constant,
    context::Context,
    eval_cache::{EvalCacheGeneric, EvalExplicitCache, EvalImplicitCache, EvalParametricCache},
    eval_result::{EvalExplicitResult, EvalParametricResult, EvalResult, RelationArgs},
    interval_set::TupperIntervalSet,
    nary,
    ops::{StaticForm, StaticFormKind, StaticTerm, StaticTermKind, StoreIndex, ValueStore},
    parse::{format_error, parse_expr},
    ternary,
    traits::BytesAllocated,
    unary, var, vars,
    vars::{VarIndex, VarSet, VarType},
    visit::*,
};
use inari::{const_interval, interval, DecInterval, Decoration, Interval};
use rug::Integer;
use std::{collections::HashMap, iter::once, mem::take, str::FromStr};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExplicitRelationOp {
    Eq,
    Ge,
    Gt,
    Le,
    Lt,
}

/// The type of a [`Relation`], which decides the graphing algorithm to be used.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RelationType {
    /// The relation contains no variables.
    Constant,
    /// The relation is of the form y op f(x) ∧ P(x), where P(x) is an optional constraint on x.
    ExplicitFunctionOfX(ExplicitRelationOp),
    /// The relation is of the form x op f(y) ∧ P(y), where P(y) is an optional constraint on y.
    ExplicitFunctionOfY(ExplicitRelationOp),
    /// The relation is of a general form.
    Implicit,
    /// The relation is of the form x = f(n, t) ∧ y = g(n, t) ∧ P(n, t),
    /// where P(n, t) is an optional constraint on the parameters.
    Parametric,
}

#[derive(Clone, Debug)]
pub struct VarIndices {
    pub n: Option<VarIndex>,
    pub n_theta: Option<VarIndex>,
    pub t: Option<VarIndex>,
    pub x: Option<VarIndex>,
    pub y: Option<VarIndex>,
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
    cached_terms: Vec<Vec<StoreIndex>>,
    n_theta_range: Interval,
    t_range: Interval,
    relation_type: RelationType,
    vars: VarSet,
    vars_ordered: Vec<VarSet>,
    var_indices: VarIndices,
}

impl Relation {
    /// Creates a new [`Vec<Interval>`] with all elements initialized to [`Interval::ENTIRE`].
    pub fn create_args(&self) -> Vec<Interval> {
        vec![Interval::ENTIRE; self.vars_ordered.len()]
    }

    /// Returns the total number of times the functions [`Self::eval`], [`Self::eval_explicit`]
    /// and [`Self::eval_parametric`] are called for `self`.
    pub fn eval_count(&self) -> usize {
        self.eval_count
    }

    /// Evaluates the explicit relation y = f(x) ∧ P(x) (or x = f(y) ∧ P(y))
    /// and returns (f(x), P(x)) (or (f(y), P(y))).
    ///
    /// If P(x) (or P(y)) is absent, its value is assumed to be always true.
    ///
    /// Precondition: `cache` has never been passed to other relations.
    pub fn eval_explicit(
        &mut self,
        args: &RelationArgs,
        cache: &mut EvalExplicitCache,
    ) -> EvalExplicitResult {
        assert!(matches!(
            self.relation_type,
            RelationType::ExplicitFunctionOfX(_) | RelationType::ExplicitFunctionOfY(_)
        ));
        self.eval_count += 1;

        if let Some(r) = cache.get(args) {
            return r.clone();
        }

        let p = self.eval_generic(args, cache);
        let r = match self.relation_type {
            RelationType::ExplicitFunctionOfX(_) => (self.ts[self.y_explicit.unwrap()].clone(), p),
            RelationType::ExplicitFunctionOfY(_) => (self.ts[self.x_explicit.unwrap()].clone(), p),
            _ => unreachable!(),
        };

        cache.insert_with(args, || r.clone());
        r
    }

    /// Evaluates the implicit or constant relation.
    ///
    /// Precondition: `cache` has never been passed to other relations.
    pub fn eval_implicit(
        &mut self,
        args: &RelationArgs,
        cache: &mut EvalImplicitCache,
    ) -> EvalResult {
        assert!(matches!(
            self.relation_type,
            RelationType::Constant | RelationType::Implicit
        ));
        self.eval_count += 1;

        if let Some(r) = cache.get(args) {
            return r.clone();
        }

        let r = self.eval_generic(args, cache);

        cache.insert_with(args, || r.clone());
        r
    }

    /// Evaluates the parametric relation x = f(n, t) ∧ y = g(n, t) ∧ P(n, t)
    /// and returns (f(n, t), g(n, t), P(n, t)).
    ///
    /// If P(n, t) is absent, its value is assumed to be always true.
    ///
    /// Precondition: `cache` has never been passed to other relations.
    pub fn eval_parametric(
        &mut self,
        args: &RelationArgs,
        cache: &mut EvalParametricCache,
    ) -> EvalParametricResult {
        assert_eq!(self.relation_type, RelationType::Parametric);
        self.eval_count += 1;

        if let Some(r) = cache.get(args) {
            return r.clone();
        }

        let p = self.eval_generic(args, cache);
        let r = (
            self.ts[self.x_explicit.unwrap()].clone(),
            self.ts[self.y_explicit.unwrap()].clone(),
            p,
        );

        cache.insert_with(args, || r.clone());
        r
    }

    pub fn forms(&self) -> &Vec<StaticForm> {
        &self.forms
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

    /// Returns the set of variables that appear in the relation.
    pub fn vars(&self) -> VarSet {
        self.vars
    }

    pub fn var_indices(&self) -> &VarIndices {
        &self.var_indices
    }

    fn eval_generic<T: BytesAllocated>(
        &mut self,
        args: &RelationArgs,
        cache: &mut EvalCacheGeneric<T>,
    ) -> EvalResult {
        let ts = &mut self.ts;
        let mut cached_vars = VarSet::EMPTY;

        for i in 0..self.vars_ordered.len() {
            if let Some(mx_ts) = cache.get_x(i, args) {
                for (&i, mx) in self.cached_terms[i].iter().zip(mx_ts.iter()) {
                    ts[i] = mx.clone();
                }
                cached_vars |= self.vars_ordered[i];
            }
        }

        for t in &self.terms {
            match t.kind {
                StaticTermKind::Var(i, ty) => {
                    let x = args[i as usize];
                    let d = if ty == VarType::Integer && !x.is_singleton() {
                        Decoration::Def
                    } else {
                        Decoration::Com
                    };
                    t.put(ts, DecInterval::set_dec(x, d).into());
                }
                _ if t.vars.len() <= 1 && cached_vars.contains(t.vars) => {
                    // `t` is constant or cached.
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

        for i in 0..self.vars_ordered.len() {
            if !cached_vars.contains(self.vars_ordered[i]) {
                cache.insert_x_with(i, args, || {
                    self.cached_terms[i]
                        .iter()
                        .map(|&i| ts[i].clone())
                        .collect()
                });
            }
        }

        r
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
        let mut e = parse_expr(s, &[Context::builtin()])?;
        UpdateMetadata.visit_expr_mut(&mut e);
        if let Some(e) = find_unknown_type_expr(&e) {
            return Err(format_error(
                s,
                e.source_range.clone(),
                "cannot interpret the expression",
            ));
        }
        if e.ty != ValueType::Boolean {
            return Err(format_error(
                s,
                e.source_range.clone(),
                &format!(
                    "relation must be of type `{}` but not `{}`",
                    ValueType::Boolean,
                    e.ty
                ),
            ));
        }
        NormalizeNotExprs.visit_expr_mut(&mut e);
        PreTransform.visit_expr_mut(&mut e);
        expand_complex_functions(&mut e);
        simplify(&mut e);
        let relation_type = relation_type(&mut e);
        NormalizeRelationalExprs.visit_expr_mut(&mut e);
        ExpandBoole.visit_expr_mut(&mut e);
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
        assert_eq!(e.ty, ValueType::Boolean);
        let mut v = AssignId::new();
        v.visit_expr_mut(&mut e);

        let vars = match relation_type {
            RelationType::ExplicitFunctionOfX(_) => e.vars.difference(VarSet::Y),
            RelationType::ExplicitFunctionOfY(_) => e.vars.difference(VarSet::X),
            RelationType::Parametric => e.vars.difference(VarSet::X | VarSet::Y),
            _ => e.vars,
        };
        let vars_ordered = [VarSet::N, VarSet::N_THETA, VarSet::T, VarSet::X, VarSet::Y]
            .into_iter()
            .filter(|&v| vars.contains(v))
            .collect::<Vec<_>>();
        let var_index = vars_ordered
            .iter()
            .enumerate()
            .map(|(i, &v)| (v, i as VarIndex))
            .collect::<HashMap<VarSet, VarIndex>>();

        let collector = CollectStatic::new(v, &var_index);
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
        let cached_terms = v.get();

        let mut slf = Self {
            terms,
            forms,
            n_atom_forms,
            ts: ValueStore::new(TupperIntervalSet::new(), n_terms),
            eval_count: 0,
            x_explicit,
            y_explicit,
            cached_terms,
            n_theta_range,
            t_range,
            relation_type,
            vars,
            vars_ordered,
            var_indices: VarIndices {
                n: var_index.get(&VarSet::N).copied(),
                n_theta: var_index.get(&VarSet::N_THETA).copied(),
                t: var_index.get(&VarSet::T).copied(),
                x: var_index.get(&VarSet::X).copied(),
                y: var_index.get(&VarSet::Y).copied(),
            },
        };
        slf.initialize();
        Ok(slf)
    }
}

/// Transforms an expression that contains r or θ into the equivalent expression
/// that contains only x, y and n_θ. When the result contains n_θ,
/// it actually represents a disjunction of expressions indexed by n_θ.
///
/// Precondition: `e` has been pre-transformed and simplified.
fn expand_polar_coords(e: &mut Expr) {
    use {BinaryOp::*, NaryOp::*};
    let ctx = Context::builtin();

    // e1 = e /. {r → sqrt(x^2 + y^2), θ → atan2(y, x) + 2π n_θ}.
    let mut e1 = e.clone();
    let mut v = ReplaceAll::new(|e| match e {
        var!(x) if x == "r" => Some(Expr::binary(
            Pow,
            box Expr::nary(
                Plus,
                vec![
                    Expr::binary(Pow, box ctx.get_constant("x").unwrap(), box Expr::two()),
                    Expr::binary(Pow, box ctx.get_constant("y").unwrap(), box Expr::two()),
                ],
            ),
            box Expr::one_half(),
        )),
        var!(x) if x == "theta" => Some(Expr::nary(
            Plus,
            vec![
                Expr::binary(
                    Atan2,
                    box ctx.get_constant("y").unwrap(),
                    box ctx.get_constant("x").unwrap(),
                ),
                Expr::nary(
                    Times,
                    vec![Expr::tau(), ctx.get_constant("<n-theta>").unwrap()],
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
        var!(x) if x == "r" => Some(Expr::nary(
            Times,
            vec![
                Expr::minus_one(),
                Expr::binary(
                    Pow,
                    box Expr::nary(
                        Plus,
                        vec![
                            Expr::binary(Pow, box ctx.get_constant("x").unwrap(), box Expr::two()),
                            Expr::binary(Pow, box ctx.get_constant("y").unwrap(), box Expr::two()),
                        ],
                    ),
                    box Expr::one_half(),
                ),
            ],
        )),
        var!(x) if x == "theta" => Some(Expr::nary(
            Plus,
            vec![
                Expr::binary(
                    Atan2,
                    box ctx.get_constant("y").unwrap(),
                    box ctx.get_constant("x").unwrap(),
                ),
                Expr::nary(
                    Times,
                    vec![
                        Expr::tau(),
                        Expr::nary(
                            Plus,
                            vec![Expr::one_half(), ctx.get_constant("<n-theta>").unwrap()],
                        ),
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

    fn common_period(xp: Integer, yp: Integer) -> Integer {
        if xp == 0 {
            yp
        } else if yp == 0 {
            xp
        } else {
            xp.lcm(&yp)
        }
    }

    match e {
        bool_constant!(_) | constant!(_) => Some(0.into()),
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
                                if let Some(a) = a.rational() {
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
                            if let Some(a) = a.rational() {
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
            Some(common_period(xp, yp))
        }
        ternary!(_, x, y, z) => {
            let xp = function_period(x, variable)?;
            let yp = function_period(y, variable)?;
            let zp = function_period(z, variable)?;
            Some(common_period(common_period(xp, yp), zp))
        }
        nary!(_, xs) => xs
            .iter()
            .map(|x| function_period(x, variable))
            .collect::<Option<Vec<_>>>()
            .map(|ps| ps.into_iter().fold(Integer::from(0), common_period)),
        _ => panic!("unexpected kind of expression"),
    }
}

struct ExplicitRelationParts {
    op: ExplicitRelationOp,
    y: Option<Expr>, // y op f(x)
    px: Vec<Expr>,   // P(x)
}

/// Tries to identify `e` as an explicit relation.
fn normalize_explicit_relation(
    e: &mut Expr,
    y_var: VarSet,
    x_var: VarSet,
) -> Option<ExplicitRelationOp> {
    use BinaryOp::*;

    let mut parts = ExplicitRelationParts {
        op: ExplicitRelationOp::Eq,
        y: None,
        px: vec![],
    };

    if !normalize_explicit_relation_impl(&mut e.clone(), &mut parts, y_var, x_var) {
        return None;
    }

    if let Some(y) = parts.y {
        *e = *(once(box y)
            .chain(parts.px.into_iter().map(Box::new))
            .reduce(|acc, e| box Expr::binary(And, acc, e))
            .unwrap());
        Some(parts.op)
    } else {
        None
    }
}

fn normalize_explicit_relation_impl(
    e: &mut Expr,
    parts: &mut ExplicitRelationParts,
    y_var: VarSet,
    x_var: VarSet,
) -> bool {
    use BinaryOp::*;

    macro_rules! explicit_rel_op {
        () => {
            Eq | Ge | Gt | Le | Lt
        };
    }

    match e {
        binary!(And, e1, e2) => {
            normalize_explicit_relation_impl(e1, parts, y_var, x_var)
                && normalize_explicit_relation_impl(e2, parts, y_var, x_var)
        }
        binary!(op @ explicit_rel_op!(), y @ var!(_), e)
        | binary!(op @ explicit_rel_op!(), e, y @ var!(_))
            if y.vars == y_var && x_var.contains(e.vars) =>
        {
            parts.y.is_none() && {
                parts.op = match op {
                    Eq => ExplicitRelationOp::Eq,
                    Ge => ExplicitRelationOp::Ge,
                    Gt => ExplicitRelationOp::Gt,
                    Le => ExplicitRelationOp::Le,
                    Lt => ExplicitRelationOp::Lt,
                    _ => unreachable!(),
                };
                parts.y = Some(Expr::binary(ExplicitRel, box take(y), box take(e)));
                true
            }
        }
        e if x_var.contains(e.vars) => {
            parts.px.push(take(e));
            true
        }
        _ => false,
    }
}

struct ParametricRelationParts {
    xt: Option<Expr>, // x = f(n, t)
    yt: Option<Expr>, // y = f(n, t)
    pt: Vec<Expr>,    // P(n, t)
}

/// Tries to identify `e` as a parametric relation.
fn normalize_parametric_relation(e: &mut Expr) -> bool {
    use BinaryOp::*;

    let mut parts = ParametricRelationParts {
        xt: None,
        yt: None,
        pt: vec![],
    };

    if !normalize_parametric_relation_impl(&mut e.clone(), &mut parts) {
        return false;
    }

    if let (Some(xt), Some(yt)) = (parts.xt, parts.yt) {
        *e = *([box xt, box yt]
            .into_iter()
            .chain(parts.pt.into_iter().map(Box::new))
            .reduce(|acc, e| box Expr::binary(And, acc, e))
            .unwrap());
        true
    } else {
        false
    }
}

fn normalize_parametric_relation_impl(e: &mut Expr, parts: &mut ParametricRelationParts) -> bool {
    use BinaryOp::*;

    const PARAMS: VarSet = vars!(VarSet::N | VarSet::T);

    match e {
        binary!(And, e1, e2) => {
            normalize_parametric_relation_impl(e1, parts)
                && normalize_parametric_relation_impl(e2, parts)
        }
        binary!(Eq, x @ var!(_), e) | binary!(Eq, e, x @ var!(_))
            if x.vars == VarSet::X && PARAMS.contains(e.vars) =>
        {
            parts.xt.is_none() && {
                parts.xt = Some(Expr::binary(ExplicitRel, box take(x), box take(e)));
                true
            }
        }
        binary!(Eq, y @ var!(_), e) | binary!(Eq, e, y @ var!(_))
            if y.vars == VarSet::Y && PARAMS.contains(e.vars) =>
        {
            parts.yt.is_none() && {
                parts.yt = Some(Expr::binary(ExplicitRel, box take(y), box take(e)));
                true
            }
        }
        e if PARAMS.contains(e.vars) => {
            parts.pt.push(take(e));
            true
        }
        _ => false,
    }
}

/// Determines the type of the relation. If it is [`RelationType::ExplicitFunctionOfX`],
/// [`RelationType::ExplicitFunctionOfY`], or [`RelationType::Parametric`],
/// normalizes the explicit part(s) of the relation to the form `(ExplicitRel x f(x))`,
/// where `x` is a variable and `f(x)` is a function of `x`.
fn relation_type(e: &mut Expr) -> RelationType {
    use RelationType::*;

    UpdateMetadata.visit_expr_mut(e);

    if e.vars.is_empty() {
        Constant
    } else if let Some(op) = normalize_explicit_relation(e, VarSet::Y, VarSet::X) {
        ExplicitFunctionOfX(op)
    } else if let Some(op) = normalize_explicit_relation(e, VarSet::X, VarSet::Y) {
        ExplicitFunctionOfY(op)
    } else if normalize_parametric_relation(e) {
        Parametric
    } else {
        Implicit
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
        use {ExplicitRelationOp::*, RelationType::*};

        fn f(rel: &str) -> RelationType {
            rel.parse::<Relation>().unwrap().relation_type()
        }

        assert_eq!(f("1 < 2"), Constant);
        assert_eq!(f("y = 1"), ExplicitFunctionOfX(Eq));
        assert_eq!(f("y ≥ 1"), ExplicitFunctionOfX(Ge));
        assert_eq!(f("y > 1"), ExplicitFunctionOfX(Gt));
        assert_eq!(f("y ≤ 1"), ExplicitFunctionOfX(Le));
        assert_eq!(f("y < 1"), ExplicitFunctionOfX(Lt));
        assert_eq!(f("y = sin(x)"), ExplicitFunctionOfX(Eq));
        assert_eq!(f("y = sin(x) && 0 < x < 1 < 2"), ExplicitFunctionOfX(Eq));
        assert_eq!(f("0 < x < 1 < 2 && sin(x) = y"), ExplicitFunctionOfX(Eq));
        assert_eq!(f("x = 1"), ExplicitFunctionOfY(Eq));
        assert_eq!(f("x ≥ 1"), ExplicitFunctionOfY(Ge));
        assert_eq!(f("x > 1"), ExplicitFunctionOfY(Gt));
        assert_eq!(f("x ≤ 1"), ExplicitFunctionOfY(Le));
        assert_eq!(f("x < 1"), ExplicitFunctionOfY(Lt));
        assert_eq!(f("x = sin(y)"), ExplicitFunctionOfY(Eq));
        assert_eq!(f("x = sin(y) && 0 < y < 1 < 2"), ExplicitFunctionOfY(Eq));
        assert_eq!(f("0 < y < 1 < 2 && sin(y) = x"), ExplicitFunctionOfY(Eq));
        assert!(matches!(
            f("x = 1 && y = 1"),
            ExplicitFunctionOfX(Eq) | ExplicitFunctionOfY(Eq)
        ));
        assert_eq!(f("x y = 0"), Implicit);
        assert_eq!(f("y = sin(x y)"), Implicit);
        assert_eq!(f("sin(x) = 0"), Implicit);
        assert_eq!(f("sin(y) = 0"), Implicit);
        assert_eq!(f("!(y = sin(x))"), Implicit);
        assert_eq!(f("!(x = sin(y))"), Implicit);
        assert_eq!(f("y = sin(x) && y = cos(x)"), Implicit);
        assert_eq!(f("y = sin(x) || y = cos(x)"), Implicit);
        assert_eq!(f("r = 1"), Implicit);
        assert_eq!(f("x = θ"), Implicit);
        assert_eq!(f("x = theta"), Implicit);
        assert_eq!(f("x = n"), Implicit);
        assert_eq!(f("x = 1 && y = n"), Parametric);
        assert_eq!(f("x = n && y = 1"), Parametric);
        assert_eq!(f("x = 1 && y = sin(t)"), Parametric);
        assert_eq!(f("x = cos(t) && y = 1"), Parametric);
        assert_eq!(f("x = cos(t) && y = sin(t)"), Parametric);
        assert_eq!(f("sin(t) = y && cos(t) = x"), Parametric);
        assert_eq!(f("x = cos(t) && y = sin(t) && 0 < t < 1 < 2"), Parametric);
        assert_eq!(f("0 < t < 1 < 2 && sin(t) = y && cos(t) = x"), Parametric);
        assert_eq!(f("x = t && y = t && x = 2t"), Implicit);
        assert_eq!(f("x + i y = exp(i t)"), Parametric);
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

    #[test]
    fn vars() {
        fn f(rel: &str) -> VarSet {
            rel.parse::<Relation>().unwrap().vars()
        }

        assert_eq!(f("x = 0"), VarSet::EMPTY);
        assert_eq!(f("y = 0"), VarSet::EMPTY);
        assert_eq!(f("r = 0"), VarSet::X | VarSet::Y);
        assert_eq!(f("θ = 0"), VarSet::X | VarSet::Y | VarSet::N_THETA);
        assert_eq!(f("n = 0"), VarSet::N);
        assert_eq!(f("t = 0"), VarSet::T);
    }
}
