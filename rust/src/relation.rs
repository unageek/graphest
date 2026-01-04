use crate::{
    ast::{BinaryOp, ExplicitRelOp, Expr, NaryOp, TernaryOp, UnaryOp, ValueType},
    binary, bool_constant, constant,
    context::Context,
    error,
    eval_cache::{EvalExplicitCache, EvalImplicitCache, EvalParametricCache, MaximalTermCache},
    eval_result::{EvalArgs, EvalExplicitResult, EvalParametricResult, EvalResult},
    geom::{TransformInPlace, Transformation1D},
    interval_set::TupperIntervalSet,
    nary,
    ops::{OptionalValueStore, StaticForm, StaticFormKind, StaticTerm, StaticTermKind, StoreIndex},
    parse::{format_error, parse_expr},
    pown, rational_ops,
    real::{Real, RealUnit},
    rootn, ternary, unary, uninit, var, vars,
    vars::{VarIndex, VarSet, VarType},
    visit::*,
};
use inari::{const_dec_interval, const_interval, interval, DecInterval, Decoration, Interval};
use itertools::Itertools;
use rug::{Integer, Rational};
use std::{collections::HashMap, mem::take, str::FromStr, vec};

/// The type of a [`Relation`], which decides the graphing algorithm to be used.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RelationType {
    /// The relation is of the form y op f(x) ∧ P(x), where P(x) is an optional constraint on x.
    ExplicitFunctionOfX(ExplicitRelOp),
    /// The relation is of the form x op f(y) ∧ P(y), where P(y) is an optional constraint on y.
    ExplicitFunctionOfY(ExplicitRelOp),
    /// The relation is of a general form.
    Implicit,
    /// The relation is of the form x = f(m, n, t) ∧ y = g(m, n, t) ∧ P(m, n, t),
    /// where P(m, n, t) is an optional constraint on the parameters.
    Parametric,
}

#[derive(Clone, Debug)]
pub struct VarIndices {
    pub m: Option<VarIndex>,
    pub n: Option<VarIndex>,
    pub n_theta: Option<VarIndex>,
    pub t: Option<VarIndex>,
    pub x: Option<VarIndex>,
    pub y: Option<VarIndex>,
}

/// A mathematical relation whose graph is to be plotted.
#[derive(Clone, Debug)]
pub struct Relation {
    ast: Expr,
    terms: Vec<StaticTerm>,
    forms: Vec<StaticForm>,
    n_atom_forms: usize,
    ts: OptionalValueStore<TupperIntervalSet>,
    eval_count: usize,
    x_explicit: Option<StoreIndex>,
    y_explicit: Option<StoreIndex>,
    term_to_eval: Vec<bool>,
    m_range: Interval,
    n_range: Interval,
    n_theta_range: Interval,
    t_range: Interval,
    relation_type: RelationType,
    vars: VarSet,
    vars_ordered: Vec<VarSet>,
    var_indices: VarIndices,
}

impl Relation {
    /// Returns the processed AST of the relation.
    pub fn ast(&self) -> &Expr {
        &self.ast
    }

    /// Creates a new [`Vec<Interval>`] with all elements initialized to [`Interval::ENTIRE`].
    pub fn create_args(&self) -> Vec<Interval> {
        vec![Interval::ENTIRE; self.vars_ordered.len()]
    }

    /// Returns the total number of times either of the functions [`Self::eval_implicit`],
    /// [`Self::eval_explicit`] or [`Self::eval_parametric`] is called for `self`.
    pub fn eval_count(&self) -> usize {
        self.eval_count
    }

    /// Evaluates the explicit relation y = f(x) ∧ P(x) (or x = f(y) ∧ P(y))
    /// and returns f'(x) (or f'(y)), where f'(x) is the value of the conditional expression
    /// if(P(x), f(x), ¿) transformed by `ty`.
    ///
    /// If P(x) is absent, it is assumed to be always true.
    ///
    /// f'(x) is forced normalized.
    ///
    /// Precondition: `cache` has never been passed to other relations.
    pub fn eval_explicit<'a>(
        &mut self,
        args: &EvalArgs,
        ty: &Transformation1D,
        cache: &'a mut EvalExplicitCache,
    ) -> &'a EvalExplicitResult {
        assert!(matches!(
            self.relation_type,
            RelationType::ExplicitFunctionOfX(_) | RelationType::ExplicitFunctionOfY(_)
        ));
        self.eval_count += 1;

        cache.setup(&self.terms, &self.vars_ordered);
        cache.full.get_or_insert_with(args, || {
            self.eval(args, &mut cache.univariate);
            let mut ys = match self.relation_type {
                RelationType::ExplicitFunctionOfX(_) => {
                    self.ts.get(self.y_explicit.unwrap()).unwrap().clone()
                }
                RelationType::ExplicitFunctionOfY(_) => {
                    self.ts.get(self.x_explicit.unwrap()).unwrap().clone()
                }
                _ => unreachable!(),
            };
            ys.normalize(true);
            ys.transform_in_place(ty);
            ys
        })
    }

    /// Evaluates the implicit relation.
    ///
    /// Precondition: `cache` has never been passed to other relations.
    pub fn eval_implicit<'a>(
        &mut self,
        args: &EvalArgs,
        cache: &'a mut EvalImplicitCache,
    ) -> &'a EvalResult {
        assert_eq!(self.relation_type, RelationType::Implicit);
        self.eval_count += 1;

        cache.setup(&self.terms, &self.vars_ordered);
        cache
            .full
            .get_or_insert_with(args, || self.eval(args, &mut cache.univariate))
    }

    /// Evaluates the parametric relation x = f(m, n, t) ∧ y = g(m, n, t) ∧ P(m, n, t)
    /// and returns (f'(…), g'(…)), where f'(…) and g'(…) are the values of the conditional expressions
    /// if(P(…), f(…), ¿) and if(P(…), g(…), ¿) transformed by `tx` and `ty`, respectively.
    ///
    /// If P(…) is absent, it is assumed to be always true.
    ///
    /// f'(…) and g'(…) are forced normalized.
    ///
    /// Precondition: `cache` has never been passed to other relations.
    pub fn eval_parametric<'a>(
        &mut self,
        args: &EvalArgs,
        tx: &Transformation1D,
        ty: &Transformation1D,
        cache: &'a mut EvalParametricCache,
    ) -> &'a EvalParametricResult {
        assert_eq!(self.relation_type, RelationType::Parametric);
        self.eval_count += 1;

        cache.setup(&self.terms, &self.vars_ordered);
        cache.full.get_or_insert_with(args, || {
            self.eval(args, &mut cache.univariate);
            let mut xs = self.ts.get(self.x_explicit.unwrap()).unwrap().clone();
            let mut ys = self.ts.get(self.y_explicit.unwrap()).unwrap().clone();
            xs.normalize(true);
            ys.normalize(true);
            xs.transform_in_place(tx);
            ys.transform_in_place(ty);
            (xs, ys)
        })
    }

    pub fn forms(&self) -> &Vec<StaticForm> {
        &self.forms
    }

    /// Returns the range of the parameter m that needs to be covered to plot the graph of the relation.
    ///
    /// Each of the bounds is either an integer or ±∞.
    pub fn m_range(&self) -> Interval {
        self.m_range
    }

    /// Returns the range of the parameter n that needs to be covered to plot the graph of the relation.
    ///
    /// Each of the bounds is either an integer or ±∞.
    pub fn n_range(&self) -> Interval {
        self.n_range
    }

    /// Returns the range of the parameter n_θ that needs to be covered to plot the graph of the relation.
    ///
    /// Each of the bounds is either an integer or ±∞.
    pub fn n_theta_range(&self) -> Interval {
        self.n_theta_range
    }

    /// Returns the type of the relation.
    pub fn relation_type(&self) -> RelationType {
        self.relation_type
    }

    /// Returns the range of the parameter t that needs to be covered to plot the graph of the relation.
    pub fn t_range(&self) -> Interval {
        self.t_range
    }

    /// Returns the set of variables in the relation.
    /// For an explicit or parametric relation, the variables x and y are excluded from the set.
    pub fn vars(&self) -> VarSet {
        self.vars
    }

    /// Returns the indices of the variables in the relation
    /// that is used for indexing in [`EvalArgs`].
    pub fn var_indices(&self) -> &VarIndices {
        &self.var_indices
    }

    fn eval(
        &mut self,
        args: &EvalArgs,
        univariate_caches: &mut [MaximalTermCache<1>],
    ) -> EvalResult {
        let ts = &mut self.ts;

        for t in &self.terms {
            if !t.vars.is_empty() {
                ts.remove(t.store_index);
            }
        }

        for cache in univariate_caches.iter_mut() {
            cache.restore(args, ts);
        }

        for t in &self.terms {
            let to_eval = self.term_to_eval[t.store_index.get()];

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
                _ if to_eval => t.put_eval(&self.terms[..], ts),
                _ => (),
            }
        }

        let r = EvalResult(
            self.forms[..self.n_atom_forms]
                .iter()
                .map(|f| f.eval(ts))
                .collect(),
        );

        for cache in univariate_caches.iter_mut() {
            cache.store(args, ts);
        }

        r
    }

    fn initialize(&mut self) {
        for t in &self.terms {
            self.ts.remove(t.store_index);

            // This condition is different from `let StaticTermKind::Constant(_) = t.kind`,
            // as not all constant expressions are folded. See the comment on [`FoldConstant`].
            if t.vars.is_empty() {
                t.put_eval(&self.terms[..], &mut self.ts);
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
        let mut param_ranges = ParamRanges::new();
        let relation_type = relation_type(&mut e, &mut param_ranges);
        NormalizeRelationalExprs.visit_expr_mut(&mut e);
        ExpandBoole.visit_expr_mut(&mut e);
        simplify(&mut e);
        ModEqTransform.visit_expr_mut(&mut e);
        simplify(&mut e);

        let m_range = {
            let period = function_period(&e, VarSet::M);
            if let Some(period) = &period {
                if let Some(q) = period.rational() {
                    if q.is_zero() {
                        const_interval!(0.0, 0.0)
                    } else {
                        interval!(&format!("[0,{}]", Integer::from(q.numer() - 1))).unwrap()
                    }
                } else {
                    Interval::ENTIRE
                }
            } else {
                Interval::ENTIRE
            }
        }
        .intersection(param_ranges.m_range);

        let n_range = {
            let period = function_period(&e, VarSet::N);
            if let Some(period) = &period {
                if let Some(q) = period.rational() {
                    if q.is_zero() {
                        const_interval!(0.0, 0.0)
                    } else {
                        interval!(&format!("[0,{}]", Integer::from(q.numer() - 1))).unwrap()
                    }
                } else {
                    Interval::ENTIRE
                }
            } else {
                Interval::ENTIRE
            }
        }
        .intersection(param_ranges.n_range);

        let n_theta_range = {
            let period = function_period(&e, VarSet::N_THETA);
            if let Some(period) = &period {
                let (q, unit) = period.rational_unit().unwrap();
                if q.is_zero() {
                    const_interval!(0.0, 0.0)
                } else {
                    match unit {
                        RealUnit::One => Interval::ENTIRE,
                        RealUnit::Pi => interval!(&format!(
                            "[0,{}]",
                            Integer::from(Rational::from(q / 2).numer() - 1)
                        ))
                        .unwrap(),
                    }
                }
            } else {
                Interval::ENTIRE
            }
        };
        assert_eq!(n_theta_range.trunc(), n_theta_range);

        let t_range = {
            let period = function_period(&e, VarSet::T);
            if let Some(period) = &period {
                let (q, unit) = period.rational_unit().unwrap();
                match unit {
                    RealUnit::One => interval!(&format!("[0,{}]", q)).unwrap(),
                    RealUnit::Pi => interval!(&format!("[0,{}]", q)).unwrap() * Interval::PI,
                }
            } else {
                Interval::ENTIRE
            }
        }
        .intersection(param_ranges.t_range);

        expand_polar_coords(&mut e);
        simplify(&mut e);
        SubDivTransform.visit_expr_mut(&mut e);
        simplify(&mut e);
        PostTransform.visit_expr_mut(&mut e);
        FuseMulAdd.visit_expr_mut(&mut e);
        UpdateMetadata.visit_expr_mut(&mut e);
        assert_eq!(e.ty, ValueType::Boolean);

        let vars = match relation_type {
            RelationType::ExplicitFunctionOfX(_) => e.vars.difference(VarSet::Y),
            RelationType::ExplicitFunctionOfY(_) => e.vars.difference(VarSet::X),
            RelationType::Parametric => e.vars.difference(VarSet::X | VarSet::Y),
            _ => e.vars,
        };
        let vars_ordered = [
            VarSet::M,
            VarSet::N,
            VarSet::N_THETA,
            VarSet::T,
            VarSet::X,
            VarSet::Y,
        ]
        .into_iter()
        .filter(|&v| vars.contains(v))
        .collect::<Vec<_>>();
        let var_index = vars_ordered
            .iter()
            .enumerate()
            .map(|(i, &v)| (v, i as VarIndex))
            .collect::<HashMap<VarSet, VarIndex>>();

        let mut v = AssignSite::new();
        v.visit_expr_mut(&mut e);
        let mut collect_real_exprs = CollectRealExprs::new(vars);
        collect_real_exprs.visit_expr_mut(&mut e);
        let collector = CollectStatic::new(v, collect_real_exprs, &var_index);
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

        let mut term_to_eval = vec![false; n_terms];
        for f in &forms {
            if let StaticFormKind::Atomic(_, i) = &f.kind {
                term_to_eval[i.get()] = true;
            }
        }
        if let Some(i) = x_explicit {
            term_to_eval[i.get()] = true;
        }
        if let Some(i) = y_explicit {
            term_to_eval[i.get()] = true;
        }

        let mut slf = Self {
            ast: e,
            terms,
            forms,
            n_atom_forms,
            ts: OptionalValueStore::new(n_terms),
            eval_count: 0,
            x_explicit,
            y_explicit,
            term_to_eval,
            m_range,
            n_range,
            n_theta_range,
            t_range,
            relation_type,
            vars,
            vars_ordered,
            var_indices: VarIndices {
                m: var_index.get(&VarSet::M).copied(),
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
    use {BinaryOp::*, NaryOp::*, TernaryOp::*, UnaryOp::*};
    let ctx = Context::builtin();

    let x = ctx.get_constant("x").unwrap();
    let y = ctx.get_constant("y").unwrap();
    let minus_x = Expr::nary(Times, vec![Expr::minus_one(), x.clone()]);
    let minus_y = Expr::nary(Times, vec![Expr::minus_one(), y.clone()]);
    let hypot = Expr::binary(
        Pow,
        Expr::nary(
            Plus,
            vec![
                Expr::binary(Pow, x.clone(), Expr::two()),
                Expr::binary(Pow, y.clone(), Expr::two()),
            ],
        ),
        Expr::one_half(),
    );
    let neg_hypot = Expr::nary(Times, vec![Expr::minus_one(), hypot.clone()]);
    let atan2 = Expr::ternary(
        IfThenElse,
        // Restrict the domain to x > -ε |y| to reduce the computational cost.
        Expr::unary(
            BooleLtZero,
            Expr::nary(
                Plus,
                vec![
                    minus_x.clone(),
                    Expr::nary(
                        Times,
                        vec![
                            Expr::constant(
                                const_dec_interval!(-6.103515625e-05, -6.103515625e-05).into(),
                            ),
                            Expr::unary(Abs, y.clone()),
                        ],
                    ),
                ],
            ),
        ),
        Expr::binary(Atan2, y.clone(), x.clone()),
        Expr::undefined(),
    );
    let anti_atan2 = Expr::ternary(
        IfThenElse,
        // Restrict the domain to x < ε |y| to reduce the computational cost.
        Expr::unary(
            BooleLtZero,
            Expr::nary(
                Plus,
                vec![
                    x.clone(),
                    Expr::nary(
                        Times,
                        vec![
                            Expr::constant(
                                const_dec_interval!(-6.103515625e-05, -6.103515625e-05).into(),
                            ),
                            Expr::unary(Abs, y.clone()),
                        ],
                    ),
                ],
            ),
        ),
        Expr::binary(Atan2, minus_y.clone(), minus_x.clone()),
        Expr::undefined(),
    );
    let two_pi_n_theta = Expr::nary(
        Times,
        vec![Expr::tau(), ctx.get_constant("<n-theta>").unwrap()],
    );

    // e11 = e /. {r → sqrt(x^2 + y^2), θ → 2π n_θ + atan2(y, x)}.
    let e11 = {
        let mut e = e.clone();
        let mut v = ReplaceAll::new(|e| match e {
            var!(x) if x == "r" => Some(hypot.clone()),
            var!(x) if x == "theta" => Some(Expr::nary(
                Plus,
                vec![two_pi_n_theta.clone(), atan2.clone()],
            )),
            _ => None,
        });
        v.visit_expr_mut(&mut e);
        if !v.modified {
            // `e` contains neither r nor θ.
            return;
        }
        e
    };

    // e12 = e /. {r → sqrt(x^2 + y^2), θ → π + 2π n_θ + atan2(-y, -x)}.
    let e12 = {
        let mut e = e.clone();
        let mut v = ReplaceAll::new(|e| match e {
            var!(x) if x == "r" => Some(hypot.clone()),
            var!(x) if x == "theta" => Some(Expr::nary(
                Plus,
                vec![Expr::pi(), two_pi_n_theta.clone(), anti_atan2.clone()],
            )),
            _ => None,
        });
        v.visit_expr_mut(&mut e);
        e
    };

    // e21 = e /. {r → -sqrt(x^2 + y^2), θ → 2π n_θ + atan2(-y, -x)}.
    let e21 = {
        let mut e = e.clone();
        let mut v = ReplaceAll::new(|e| match e {
            var!(x) if x == "r" => Some(neg_hypot.clone()),
            var!(x) if x == "theta" => Some(Expr::nary(
                Plus,
                vec![two_pi_n_theta.clone(), anti_atan2.clone()],
            )),
            _ => None,
        });
        v.visit_expr_mut(&mut e);
        e
    };

    // e22 = e /. {r → -sqrt(x^2 + y^2), θ → π + 2π n_θ + atan2(y, x)}.
    let e22 = {
        let mut e = e.clone();
        let mut v = ReplaceAll::new(|e| match e {
            var!(x) if x == "r" => Some(neg_hypot.clone()),
            var!(x) if x == "theta" => Some(Expr::nary(
                Plus,
                vec![Expr::pi(), two_pi_n_theta.clone(), atan2.clone()],
            )),
            _ => None,
        });
        v.visit_expr_mut(&mut e);
        e
    };

    *e = Expr::nary(OrN, vec![e11, e12, e21, e22]);
}

/// Returns the period of a function of a variable t,
/// i.e., a real number p that satisfies (e /. t → t + p) = e.
/// If the period is 0, the expression is independent of the variable.
///
/// Precondition: `e` has been pre-transformed and simplified.
fn function_period(e: &Expr, variable: VarSet) -> Option<Real> {
    use {BinaryOp::*, NaryOp::*, UnaryOp::*};

    fn common_period(xp: Real, yp: Real) -> Option<Real> {
        match (xp.rational_unit(), yp.rational_unit()) {
            (Some((q, _)), _) if q.is_zero() => Some(yp),
            (_, Some((r, _))) if r.is_zero() => Some(xp),
            (Some((q, q_unit)), Some((r, r_unit))) if q_unit == r_unit => Some(Real::from((
                rational_ops::lcm(q.clone(), r.clone()).unwrap(),
                q_unit,
            ))),
            _ => None,
        }
    }

    fn generic_function_period(e: &Expr, variable: VarSet) -> Option<Real> {
        match e {
            bool_constant!(_) | constant!(_) => Some(Real::zero()),
            x @ var!(_) if x.vars.contains(variable) => None,
            var!(_) => Some(Real::zero()),
            unary!(_, x) => function_period(x, variable),
            binary!(_, x, y) => {
                let xp = function_period(x, variable)?;
                let yp = function_period(y, variable)?;
                common_period(xp, yp)
            }
            ternary!(_, x, y, z) => {
                let xp = function_period(x, variable)?;
                let yp = function_period(y, variable)?;
                let zp = function_period(z, variable)?;
                common_period(common_period(xp, yp)?, zp)
            }
            nary!(_, xs) => xs
                .iter()
                .map(|x| function_period(x, variable))
                .collect::<Vec<_>>()
                .into_iter()
                .try_fold(Real::zero(), |x, y| common_period(x, y?)),
            pown!(_, _) | rootn!(_, _) | error!() | uninit!() => {
                panic!("unexpected kind of expression")
            }
        }
    }

    match e {
        unary!(op @ (Cos | Sin | Tan), x) if x.vars.contains(variable) => match x {
            var!(_) => {
                // op(t)
                match op {
                    Tan => Some(Real::pi()),
                    _ => Some(Real::tau()),
                }
            }
            nary!(Plus, xs) => match xs
                .iter()
                .filter(|x| x.vars.contains(variable))
                .exactly_one()
            {
                Ok(var!(_)) => {
                    // op(… + t + …)
                    match op {
                        Tan => Some(Real::pi()),
                        _ => Some(Real::tau()),
                    }
                }
                Ok(nary!(Times, xs)) => match &xs[..] {
                    [constant!(a), var!(_)] => {
                        // op(… + a t + …)
                        if let Some((q, unit)) = a.rational_unit() {
                            let unit = match unit {
                                RealUnit::One => RealUnit::Pi,
                                RealUnit::Pi => RealUnit::One,
                            };
                            match op {
                                Tan => Some(Real::from((q.clone().recip(), unit))),
                                _ => Some(Real::from((2 * q.clone().recip(), unit))),
                            }
                        } else {
                            None
                        }
                    }
                    _ => generic_function_period(e, variable),
                },
                _ => generic_function_period(e, variable),
            },
            nary!(Times, xs) => match &xs[..] {
                [constant!(a), var!(_)] => {
                    // op(a t)
                    if let Some((q, unit)) = a.rational_unit() {
                        let unit = match unit {
                            RealUnit::One => RealUnit::Pi,
                            RealUnit::Pi => RealUnit::One,
                        };
                        match op {
                            Tan => Some(Real::from((q.clone().recip(), unit))),
                            _ => Some(Real::from((2 * q.clone().recip(), unit))),
                        }
                    } else {
                        None
                    }
                }
                _ => generic_function_period(e, variable),
            },
            _ => generic_function_period(e, variable),
        },
        binary!(Mod, x, constant!(y))
            if x.vars.contains(variable) && y.rational_unit().is_some() =>
        {
            let p = y.clone().abs();

            match x {
                var!(_) => {
                    // mod(t, y)
                    Some(p)
                }
                nary!(Plus, xs) => match xs
                    .iter()
                    .filter(|x| x.vars.contains(variable))
                    .exactly_one()
                {
                    Ok(var!(_)) => {
                        // mod(… + t + …, y)
                        Some(p)
                    }
                    Ok(nary!(Times, xs)) => match &xs[..] {
                        [constant!(a), var!(_)] => {
                            // mod(… + a t + …, y)
                            match p / a.clone().abs() {
                                p if p.rational_unit().is_some() => Some(p),
                                _ => None,
                            }
                        }
                        _ => generic_function_period(e, variable),
                    },
                    _ => generic_function_period(e, variable),
                },
                nary!(Times, xs) => match &xs[..] {
                    [constant!(a), var!(_)] => {
                        // mod(a t, y)
                        match p / a.clone().abs() {
                            p if p.rational_unit().is_some() => Some(p),
                            _ => None,
                        }
                    }
                    _ => generic_function_period(e, variable),
                },
                _ => generic_function_period(x, variable),
            }
        }
        _ => generic_function_period(e, variable),
    }
}

struct ParamRanges {
    m_range: Interval,
    n_range: Interval,
    t_range: Interval,
}

impl ParamRanges {
    fn new() -> Self {
        Self {
            m_range: Interval::ENTIRE,
            n_range: Interval::ENTIRE,
            t_range: Interval::ENTIRE,
        }
    }

    fn refine_with(&mut self, e: &Expr) {
        use BinaryOp::*;

        match e {
            binary!(Ge | Gt, var!(x), constant!(a)) | binary!(Le | Lt, constant!(a), var!(x)) => {
                if let Some(r) = get_mut(self, x) {
                    let inf = a
                        .interval()
                        .iter()
                        .fold(f64::INFINITY, |acc, x| acc.min(x.x.inf()));
                    *r = r.intersection(interval!(inf, f64::INFINITY).unwrap_or(Interval::EMPTY));
                }
            }
            binary!(Le | Lt, var!(x), constant!(a)) | binary!(Ge | Gt, constant!(a), var!(x)) => {
                if let Some(r) = get_mut(self, x) {
                    let sup = a
                        .interval()
                        .iter()
                        .fold(f64::NEG_INFINITY, |acc, x| acc.max(x.x.sup()));
                    *r = r
                        .intersection(interval!(f64::NEG_INFINITY, sup).unwrap_or(Interval::EMPTY));
                }
            }
            binary!(Eq, var!(x), constant!(a)) | binary!(Eq, constant!(a), var!(x)) => {
                if let Some(r) = get_mut(self, x) {
                    *r = r.intersection(
                        a.interval()
                            .iter()
                            .fold(Interval::EMPTY, |acc, x| acc.convex_hull(x.x)),
                    );
                }
            }
            _ => return,
        }

        self.m_range = self.m_range.trunc();
        self.n_range = self.n_range.trunc();

        // Empty ranges are not supported.
        const I_ZERO: Interval = const_interval!(0.0, 0.0);
        if self.m_range.is_empty() {
            self.m_range = I_ZERO;
        }
        if self.n_range.is_empty() {
            self.n_range = I_ZERO;
        }
        if self.t_range.is_empty() {
            self.t_range = I_ZERO;
        }

        fn get_mut<'a>(slf: &'a mut ParamRanges, param_name: &str) -> Option<&'a mut Interval> {
            match param_name {
                "m" => Some(&mut slf.m_range),
                "n" => Some(&mut slf.n_range),
                "t" => Some(&mut slf.t_range),
                _ => None,
            }
        }
    }
}

struct ExplicitRelationParts {
    op: ExplicitRelOp,
    y: Option<Expr>, // y op f(x)
    px: Vec<Expr>,   // P(x)
}

/// Tries to identify `e` as an explicit relation.
fn normalize_explicit_relation(
    e: &mut Expr,
    y_var: VarSet,
    x_var: VarSet,
) -> Option<ExplicitRelOp> {
    use {NaryOp::*, TernaryOp::*, UnaryOp::*};

    let mut parts = ExplicitRelationParts {
        op: ExplicitRelOp::Eq,
        y: None,
        px: vec![],
    };

    if !normalize_explicit_relation_impl(&mut e.clone(), &mut parts, y_var, x_var) {
        return None;
    }

    if let Some(y) = &mut parts.y {
        if !parts.px.is_empty() {
            let cond = Expr::unary(Boole, Expr::nary(AndN, parts.px));

            if let binary!(_, _, f) = y {
                *f = Expr::ternary(IfThenElse, cond, take(f), Expr::undefined());
            } else {
                unreachable!();
            }
        }

        *e = take(y);
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
    use {BinaryOp::*, NaryOp::*};

    macro_rules! explicit_rel_op {
        () => {
            Eq | Ge | Gt | Le | Lt
        };
    }

    match e {
        binary!(op @ explicit_rel_op!(), y @ var!(_), e)
            if y.vars == y_var && x_var.contains(e.vars) =>
        {
            parts.y.is_none() && {
                parts.op = match op {
                    Eq => ExplicitRelOp::Eq,
                    Ge => ExplicitRelOp::Ge,
                    Gt => ExplicitRelOp::Gt,
                    Le => ExplicitRelOp::Le,
                    Lt => ExplicitRelOp::Lt,
                    _ => unreachable!(),
                };
                parts.y = Some(Expr::binary(ExplicitRel(parts.op), take(y), take(e)));
                true
            }
        }
        binary!(op @ explicit_rel_op!(), e, y @ var!(_))
            if y.vars == y_var && x_var.contains(e.vars) =>
        {
            parts.y.is_none() && {
                parts.op = match op {
                    Eq => ExplicitRelOp::Eq,
                    Ge => ExplicitRelOp::Le,
                    Gt => ExplicitRelOp::Lt,
                    Le => ExplicitRelOp::Ge,
                    Lt => ExplicitRelOp::Gt,
                    _ => unreachable!(),
                };
                parts.y = Some(Expr::binary(ExplicitRel(parts.op), take(y), take(e)));
                true
            }
        }
        nary!(AndN, es) => es
            .iter_mut()
            .all(|e| normalize_explicit_relation_impl(e, parts, y_var, x_var)),
        e if x_var.contains(e.vars) => {
            parts.px.push(take(e));
            true
        }
        _ => false,
    }
}

struct ParametricRelationParts {
    xt: Option<Expr>, // x = f(m, n, t)
    yt: Option<Expr>, // y = f(m, n, t)
    pt: Vec<Expr>,    // P(m, n, t)
}

/// Tries to identify `e` as a parametric relation.
fn normalize_parametric_relation(e: &mut Expr, param_ranges: &mut ParamRanges) -> bool {
    use {NaryOp::*, TernaryOp::*, UnaryOp::*};

    let mut parts = ParametricRelationParts {
        xt: None,
        yt: None,
        pt: vec![],
    };

    if !normalize_parametric_relation_impl(&mut e.clone(), &mut parts) {
        return false;
    }

    if let (Some(xt), Some(yt)) = &mut (parts.xt, parts.yt) {
        for e in &parts.pt {
            param_ranges.refine_with(e);
        }

        if !parts.pt.is_empty() {
            let cond = Expr::unary(Boole, Expr::nary(AndN, parts.pt));

            if let binary!(_, _, f) = xt {
                *f = Expr::ternary(IfThenElse, cond.clone(), take(f), Expr::undefined());
            } else {
                unreachable!();
            }

            if let binary!(_, _, f) = yt {
                *f = Expr::ternary(IfThenElse, cond, take(f), Expr::undefined());
            } else {
                unreachable!();
            }
        }

        *e = Expr::nary(AndN, vec![take(xt), take(yt)]);
        true
    } else {
        false
    }
}

fn normalize_parametric_relation_impl(e: &mut Expr, parts: &mut ParametricRelationParts) -> bool {
    use {BinaryOp::*, NaryOp::*};

    const PARAMS: VarSet = vars!(VarSet::M | VarSet::N | VarSet::T);

    match e {
        binary!(Eq, x @ var!(_), e) | binary!(Eq, e, x @ var!(_))
            if x.vars == VarSet::X && PARAMS.contains(e.vars) =>
        {
            parts.xt.is_none() && {
                parts.xt = Some(Expr::binary(
                    ExplicitRel(ExplicitRelOp::Eq),
                    take(x),
                    take(e),
                ));
                true
            }
        }
        binary!(Eq, y @ var!(_), e) | binary!(Eq, e, y @ var!(_))
            if y.vars == VarSet::Y && PARAMS.contains(e.vars) =>
        {
            parts.yt.is_none() && {
                parts.yt = Some(Expr::binary(
                    ExplicitRel(ExplicitRelOp::Eq),
                    take(y),
                    take(e),
                ));
                true
            }
        }
        nary!(AndN, es) => es
            .iter_mut()
            .all(|e| normalize_parametric_relation_impl(e, parts)),
        e if PARAMS.contains(e.vars) => {
            parts.pt.push(take(e));
            true
        }
        _ => false,
    }
}

struct ImplicitRelationParts {
    eq: Option<Expr>,  // The equality relation if it is unique.
    others: Vec<Expr>, // Other relations.
}

fn normalize_implicit_relation(e: &mut Expr, param_ranges: &mut ParamRanges) {
    use {BinaryOp::*, NaryOp::*, TernaryOp::*, UnaryOp::*};

    let mut parts = ImplicitRelationParts {
        eq: None,
        others: vec![],
    };

    normalize_implicit_relation_impl(&mut e.clone(), &mut parts);

    for e in &parts.others {
        param_ranges.refine_with(e);
    }

    *e = match &mut parts.eq {
        Some(binary!(Eq, x, y)) if !parts.others.is_empty() => Expr::binary(
            Eq,
            Expr::ternary(
                IfThenElse,
                Expr::unary(Boole, Expr::nary(AndN, parts.others)),
                Expr::binary(Sub, take(x), take(y)),
                Expr::undefined(),
            ),
            Expr::zero(),
        ),
        Some(eq) => take(eq),
        None if !parts.others.is_empty() => Expr::nary(AndN, parts.others),
        _ => unreachable!(),
    };
}

fn normalize_implicit_relation_impl(e: &mut Expr, parts: &mut ImplicitRelationParts) {
    use {BinaryOp::*, NaryOp::*};

    match e {
        binary!(Eq, _, _) => match &mut parts.eq {
            Some(eq) => {
                parts.others.push(take(eq));
                parts.others.push(take(e));
                parts.eq = None;
            }
            _ => parts.eq = Some(take(e)),
        },
        nary!(AndN, xs) => {
            for x in xs {
                normalize_implicit_relation_impl(x, parts);
            }
        }
        _ => parts.others.push(take(e)),
    }
}

/// Determines the type of the relation. If it is [`RelationType::ExplicitFunctionOfX`],
/// [`RelationType::ExplicitFunctionOfY`], or [`RelationType::Parametric`],
/// normalizes the explicit part(s) of the relation to the form `(ExplicitRel x f(x))`,
/// where `x` is a variable and `f(x)` is a function of `x`.
fn relation_type(e: &mut Expr, param_ranges: &mut ParamRanges) -> RelationType {
    use RelationType::*;

    UpdateMetadata.visit_expr_mut(e);

    if normalize_parametric_relation(e, param_ranges) {
        Parametric
    } else if let Some(op) = normalize_explicit_relation(e, VarSet::Y, VarSet::X) {
        ExplicitFunctionOfX(op)
    } else if let Some(op) = normalize_explicit_relation(e, VarSet::X, VarSet::Y) {
        ExplicitFunctionOfY(op)
    } else {
        normalize_implicit_relation(e, param_ranges);
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
        use {ExplicitRelOp::*, RelationType::*};

        fn f(rel: &str) -> RelationType {
            rel.parse::<Relation>().unwrap().relation_type()
        }

        assert_eq!(f("y = 1"), ExplicitFunctionOfX(Eq));
        assert_eq!(f("y ≥ 1"), ExplicitFunctionOfX(Ge));
        assert_eq!(f("y > 1"), ExplicitFunctionOfX(Gt));
        assert_eq!(f("y ≤ 1"), ExplicitFunctionOfX(Le));
        assert_eq!(f("y < 1"), ExplicitFunctionOfX(Lt));
        assert_eq!(f("1 = y"), ExplicitFunctionOfX(Eq));
        assert_eq!(f("1 ≤ y"), ExplicitFunctionOfX(Ge));
        assert_eq!(f("1 < y"), ExplicitFunctionOfX(Gt));
        assert_eq!(f("1 ≥ y"), ExplicitFunctionOfX(Le));
        assert_eq!(f("1 > y"), ExplicitFunctionOfX(Lt));
        assert_eq!(f("y = sin(x)"), ExplicitFunctionOfX(Eq));
        assert_eq!(f("y = sin(x) && 0 < x < 1 < 2"), ExplicitFunctionOfX(Eq));
        assert_eq!(f("0 < x < 1 < 2 && sin(x) = y"), ExplicitFunctionOfX(Eq));
        assert_eq!(f("x = 1"), ExplicitFunctionOfY(Eq));
        assert_eq!(f("x ≥ 1"), ExplicitFunctionOfY(Ge));
        assert_eq!(f("x > 1"), ExplicitFunctionOfY(Gt));
        assert_eq!(f("x ≤ 1"), ExplicitFunctionOfY(Le));
        assert_eq!(f("x < 1"), ExplicitFunctionOfY(Lt));
        assert_eq!(f("1 = x"), ExplicitFunctionOfY(Eq));
        assert_eq!(f("1 ≤ x"), ExplicitFunctionOfY(Ge));
        assert_eq!(f("1 < x"), ExplicitFunctionOfY(Gt));
        assert_eq!(f("1 ≥ x"), ExplicitFunctionOfY(Le));
        assert_eq!(f("1 > x"), ExplicitFunctionOfY(Lt));
        assert_eq!(f("x = sin(y)"), ExplicitFunctionOfY(Eq));
        assert_eq!(f("x = sin(y) && 0 < y < 1 < 2"), ExplicitFunctionOfY(Eq));
        assert_eq!(f("0 < y < 1 < 2 && sin(y) = x"), ExplicitFunctionOfY(Eq));
        assert_eq!(f("1 < 2"), Implicit);
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
        assert_eq!(f("x = m + n"), Implicit);
        assert_eq!(f("x = 1 && y = 1"), Parametric); // For plotting a point with inexact coordinates.
        assert_eq!(f("x = 1 && y = m + n"), Parametric);
        assert_eq!(f("x = m + n && y = 1"), Parametric);
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
            interval!(
                0.0,
                (const_interval!(5.0, 5.0) / const_interval!(3.0, 3.0) * Interval::TAU).sup()
            )
            .unwrap()
        );
        assert_eq!(
            f("mod(2/3 t, 5/7) = 0"),
            interval!(
                0.0,
                (const_interval!(15.0, 15.0) / const_interval!(14.0, 14.0)).sup()
            )
            .unwrap()
        );
        assert_eq!(
            f("mod(2/3 π t, 5/7 π) = 0"),
            interval!(
                0.0,
                (const_interval!(15.0, 15.0) / const_interval!(14.0, 14.0)).sup()
            )
            .unwrap()
        );
        assert_eq!(
            f("mod(2/3 t, 5/7 π) = 0"),
            interval!(
                0.0,
                (const_interval!(15.0, 15.0) / const_interval!(14.0, 14.0) * Interval::PI).sup()
            )
            .unwrap()
        );
        assert_eq!(f("mod(2/3 π t, 5/7) = 0"), Interval::ENTIRE);
        assert_eq!(f("mod(-t, 1) = 0"), interval!(0.0, 1.0).unwrap());
        assert_eq!(f("mod(t, -1) = 0"), interval!(0.0, 1.0).unwrap());
        assert_eq!(f("mod(-t, -1) = 0"), interval!(0.0, 1.0).unwrap());
        assert_eq!(f("mod(sin(π t), 1) = 0"), interval!(0.0, 2.0).unwrap());
        assert_eq!(f("mod(x sin(π t), 1) = 0"), interval!(0.0, 2.0).unwrap());
        assert_eq!(f("mod(y + sin(π t), 1) = 0"), interval!(0.0, 2.0).unwrap());
        assert_eq!(
            f("mod(y + x sin(π t), 1) = 0"),
            interval!(0.0, 2.0).unwrap()
        );
        assert_eq!(f("cos(sin(π t)) = 0"), interval!(0.0, 2.0).unwrap());
        assert_eq!(f("cos(x sin(π t)) = 0"), interval!(0.0, 2.0).unwrap());
        assert_eq!(f("cos(y + sin(π t)) = 0"), interval!(0.0, 2.0).unwrap());
        assert_eq!(f("cos(y + x sin(π t)) = 0"), interval!(0.0, 2.0).unwrap());
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
        assert_eq!(f("m = 0"), VarSet::M);
        assert_eq!(f("n = 0"), VarSet::N);
        assert_eq!(f("t = 0"), VarSet::T);
    }
}
