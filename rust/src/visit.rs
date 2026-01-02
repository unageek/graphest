use crate::{
    ast::{BinaryOp, Expr, NaryOp, TernaryOp, UnaryOp, ValueType},
    binary, bool_constant, constant,
    context::{Context, Def, VarProps},
    error,
    interval_set::Site,
    nary,
    ops::{
        FormIndex, RankedMinMaxOp, RelOp, ScalarBinaryOp, ScalarTernaryOp, ScalarUnaryOp,
        StaticForm, StaticFormKind, StaticTerm, StaticTermKind, StoreIndex,
    },
    parse::parse_expr,
    pown,
    real::Real,
    rootn, ternary, unary, uninit, var,
    vars::{VarIndex, VarSet},
};
use inari::Decoration;
use rug::Rational;
use std::{
    cmp::Ordering,
    collections::{HashMap, HashSet},
    hash::{Hash, Hasher},
    iter,
    marker::Sized,
    mem::take,
    ops::Deref,
};

/// A visitor that visits AST nodes in depth-first order.
pub trait Visit<'a>
where
    Self: Sized,
{
    fn visit_expr(&mut self, e: &'a Expr) {
        traverse_expr(self, e);
    }
}

#[allow(clippy::many_single_char_names)]
fn traverse_expr<'a, V: Visit<'a>>(v: &mut V, e: &'a Expr) {
    match e {
        unary!(_, x) | pown!(x, _) | rootn!(x, _) => v.visit_expr(x),
        binary!(_, x, y) => {
            v.visit_expr(x);
            v.visit_expr(y);
        }
        ternary!(_, x, y, z) => {
            v.visit_expr(x);
            v.visit_expr(y);
            v.visit_expr(z);
        }
        nary!(_, xs) => {
            for x in xs {
                v.visit_expr(x);
            }
        }
        bool_constant!(_) | constant!(_) | var!(_) | error!() => (),
        uninit!() => panic!(),
    };
}

/// A visitor that visits AST nodes in depth-first order and possibly modifies them.
pub trait VisitMut
where
    Self: Sized,
{
    fn visit_expr_mut(&mut self, e: &mut Expr) {
        traverse_expr_mut(self, e);
    }
}

#[allow(clippy::many_single_char_names)]
fn traverse_expr_mut<V: VisitMut>(v: &mut V, e: &mut Expr) {
    match e {
        unary!(_, x) | pown!(x, _) | rootn!(x, _) => v.visit_expr_mut(x),
        binary!(_, x, y) => {
            v.visit_expr_mut(x);
            v.visit_expr_mut(y);
        }
        ternary!(_, x, y, z) => {
            v.visit_expr_mut(x);
            v.visit_expr_mut(y);
            v.visit_expr_mut(z);
        }
        nary!(_, xs) => {
            for x in xs {
                v.visit_expr_mut(x);
            }
        }
        bool_constant!(_) | constant!(_) | var!(_) | error!() => (),
        uninit!() => panic!(),
    };
}

/// A possibly dangling reference to a value.
/// All operations except `from` and `clone` are **unsafe** despite not being marked as so.
struct UnsafeRef<T: Eq + Hash> {
    ptr: *const T,
}

impl<T: Eq + Hash> UnsafeRef<T> {
    fn from(x: &T) -> Self {
        Self { ptr: x as *const T }
    }
}

impl<T: Eq + Hash> Clone for UnsafeRef<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: Eq + Hash> Copy for UnsafeRef<T> {}

impl<T: Eq + Hash> Deref for UnsafeRef<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        unsafe { &*self.ptr }
    }
}

impl<T: Eq + Hash> PartialEq for UnsafeRef<T> {
    fn eq(&self, rhs: &Self) -> bool {
        unsafe { *self.ptr == *rhs.ptr }
    }
}

impl<T: Eq + Hash> Eq for UnsafeRef<T> {}

impl<T: Eq + Hash> Hash for UnsafeRef<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        unsafe { (*self.ptr).hash(state) }
    }
}

/// Replaces the name of each [`ExprKind::Var`] that matches `params[i]` with `"#i"`.
///
/// [`ExprKind::Var`]: crate::ast::ExprKind::Var
pub struct Parametrize<'a> {
    params: &'a [&'a str],
}

impl<'a> Parametrize<'a> {
    pub fn new(params: &'a [&'a str]) -> Self {
        Self { params }
    }
}

impl<'a> VisitMut for Parametrize<'a> {
    fn visit_expr_mut(&mut self, e: &mut Expr) {
        traverse_expr_mut(self, e);

        if let var!(x) = e {
            if let Some(i) = self.params.iter().position(|p| p == x) {
                *x = format!("#{}", i);
            }
        }
    }
}

/// Replaces each [`ExprKind::Var`] with name `"#i"` with `args[i]`.
///
/// [`ExprKind::Var`]: crate::ast::ExprKind::Var
pub struct Substitute {
    args: Vec<Expr>,
}

impl Substitute {
    pub fn new(args: Vec<Expr>) -> Self {
        Self { args }
    }
}

impl VisitMut for Substitute {
    fn visit_expr_mut(&mut self, e: &mut Expr) {
        traverse_expr_mut(self, e);

        if let var!(x) = e {
            if let Some(x) = x.strip_prefix('#') {
                if let Ok(i) = x.parse::<usize>() {
                    *e = self.args[i].clone()
                }
            }
        }
    }
}

pub struct ReplaceAll<Rule>
where
    Rule: Fn(&Expr) -> Option<Expr>,
{
    pub modified: bool,
    rule: Rule,
}

impl<Rule> ReplaceAll<Rule>
where
    Rule: Fn(&Expr) -> Option<Expr>,
{
    pub fn new(rule: Rule) -> Self {
        Self {
            modified: false,
            rule,
        }
    }
}

impl<Rule> VisitMut for ReplaceAll<Rule>
where
    Rule: Fn(&Expr) -> Option<Expr>,
{
    fn visit_expr_mut(&mut self, e: &mut Expr) {
        traverse_expr_mut(self, e);

        if let Some(replacement) = (self.rule)(e) {
            *e = replacement;
            self.modified = true;
        }
    }
}

/// Distributes [`UnaryOp::Not`] over [`BinaryOp::And`] and [`BinaryOp::Or`],
/// and then eliminates double negations.
pub struct NormalizeNotExprs;

impl VisitMut for NormalizeNotExprs {
    fn visit_expr_mut(&mut self, e: &mut Expr) {
        use {NaryOp::*, UnaryOp::*};

        let mut modified = false;

        if let unary!(Not, x) = e {
            match x {
                unary!(Not, x) => {
                    // (Not (Not x)) → x
                    *e = take(x);
                    modified = true;
                }
                nary!(AndN, xs) => {
                    // (Not (AndN x1 x2 …)) → (OrN (Not x1) (Not x2) …)
                    *e = Expr::nary(OrN, xs.drain(..).map(|x| Expr::unary(Not, x)).collect());
                    modified = true;
                }
                nary!(OrN, xs) => {
                    // (Not (OrN x1 x2 …)) → (AndN (Not x1) (Not x2) …)
                    *e = Expr::nary(AndN, xs.drain(..).map(|x| Expr::unary(Not, x)).collect());
                    modified = true;
                }
                _ => (),
            }
        }

        if modified {
            // `e` has not been visited yet.
            self.visit_expr_mut(e);
        } else {
            traverse_expr_mut(self, e);
        }
    }
}

/// Does the following tasks:
///
/// - Replace arithmetic expressions that contain [`UnaryOp::Neg`], [`UnaryOp::Sqrt`],
///   [`BinaryOp::Add`], [`BinaryOp::Div`], [`BinaryOp::Mul`] or [`BinaryOp::Sub`] with their equivalents
///   with [`BinaryOp::Pow`], [`NaryOp::Plus`] and [`NaryOp::Times`].
/// - Replace [`BinaryOp::And`], [`BinaryOp::Max`], [`BinaryOp::Min`], and [`BinaryOp::Or`]
///   with their n-ary counterparts.
/// - Do some ad-hoc transformations, mainly for demonstrational purposes.
pub struct PreTransform;

impl VisitMut for PreTransform {
    fn visit_expr_mut(&mut self, e: &mut Expr) {
        use {BinaryOp::*, NaryOp::*, UnaryOp::*};
        traverse_expr_mut(self, e);

        match e {
            unary!(Neg, x) => {
                // (Neg x) → (Times -1 x)
                *e = Expr::nary(Times, vec![Expr::minus_one(), take(x)]);
            }
            unary!(Sqrt, x) => {
                // (Sqrt x) → (Pow x 1/2)
                *e = Expr::binary(Pow, take(x), Expr::one_half());
            }
            binary!(op @ (Add | And | Max | Min | Or | Mul), x, y) => {
                let nary_op = match op {
                    Add => Plus,
                    And => AndN,
                    Max => MaxN,
                    Min => MinN,
                    Or => OrN,
                    Mul => Times,
                    _ => unreachable!(),
                };
                // (op x y) → (nary-op x y)
                *e = Expr::nary(nary_op, vec![take(x), take(y)]);
            }
            binary!(Div, unary!(Sin, x), y) if x == y => {
                // Ad-hoc.
                // (Div (Sin x) x) → (Sinc (UndefAt0 x))
                *e = Expr::unary(Sinc, Expr::unary(UndefAt0, take(x)));
            }
            binary!(Div, x, unary!(Sin, y)) if y == x => {
                // Ad-hoc.
                // (Div x (Sin x)) → (Pow (Sinc (UndefAt0 x)) -1)
                *e = Expr::binary(
                    Pow,
                    Expr::unary(Sinc, Expr::unary(UndefAt0, take(x))),
                    Expr::minus_one(),
                );
            }
            binary!(Div, x, y) => {
                // (Div x y) → (Times x (Pow y -1))
                *e = Expr::nary(
                    Times,
                    vec![take(x), Expr::binary(Pow, take(y), Expr::minus_one())],
                );
            }
            binary!(Sub, x, y) => {
                // (Sub x y) → (Plus x (Times -1 y))
                *e = Expr::nary(
                    Plus,
                    vec![take(x), Expr::nary(Times, vec![Expr::minus_one(), take(y)])],
                );
            }
            _ => (),
        }
    }
}

/// Precondition: [`PreTransform`] and then [`UpdateMetadata`] have been applied
/// to the expression and it has not been modified since then.
struct ExpandComplexFunctions {
    ctx: Context,
    binary_ops: HashMap<BinaryOp, Expr>,
    unary_ops: HashMap<UnaryOp, Expr>,
}

impl ExpandComplexFunctions {
    fn new() -> Self {
        Self {
            ctx: Context::new()
                .def(
                    "a",
                    Def::var(
                        "a",
                        VarProps {
                            totally_defined: true,
                            ty: ValueType::Real,
                            ..Default::default()
                        },
                    ),
                )
                .def(
                    "b",
                    Def::var(
                        "b",
                        VarProps {
                            totally_defined: true,
                            ty: ValueType::Real,
                            ..Default::default()
                        },
                    ),
                ),
            binary_ops: HashMap::new(),
            unary_ops: HashMap::new(),
        }
    }

    fn def_binary(&mut self, op: BinaryOp, body: &str) {
        let e = self.make_def(&["a", "b", "x", "y"], body);
        self.binary_ops.insert(op, e);
    }

    fn def_unary(&mut self, op: UnaryOp, body: &str) {
        let e = self.make_def(&["x", "y"], body);
        self.unary_ops.insert(op, e);
    }

    fn make_def(&mut self, params: &[&str], body: &str) -> Expr {
        let mut e = parse_expr(body, &[Context::builtin(), &self.ctx]).unwrap();
        PreTransform.visit_expr_mut(&mut e);
        NormalizeRelationalExprs.visit_expr_mut(&mut e);
        ExpandBoole.visit_expr_mut(&mut e);
        UpdateMetadata.visit_expr_mut(&mut e);
        self.visit_expr_mut(&mut e);
        simplify(&mut e);
        Parametrize::new(params).visit_expr_mut(&mut e);
        e
    }
}

impl Default for ExpandComplexFunctions {
    fn default() -> Self {
        use {BinaryOp::*, UnaryOp::*};

        let mut v = Self::new();
        // Some of the definitions may depend on previous ones.
        v.def_unary(Abs, "sqrt(x^2 + y^2)");
        v.def_unary(Cos, "cos(x) cosh(y) - i sin(x) sinh(y)");
        v.def_unary(Cosh, "cosh(x) cos(y) + i sinh(x) sin(y)");
        v.def_unary(Exp, "exp(x) cos(y) + i exp(x) sin(y)");
        v.def_unary(Ln, "1/2 ln(x^2 + y^2) + i atan2(y, x)");
        v.def_binary(
            Pow,
            "if(a = 0 ∧ b = 0 ∧ x > 0, 0, exp((x + i y) ln(a + i b)))",
        );
        v.def_unary(Recip, "x / (x^2 + y^2) - i y / (x^2 + y^2)");
        v.def_unary(Sin, "sin(x) cosh(y) + i cos(x) sinh(y)");
        v.def_unary(Sinh, "sinh(x) cos(y) + i cosh(x) sin(y)");
        v.def_unary(Sqr, "x^2 - y^2 + 2 i x y");
        v.def_unary(Tan, "sin(x + i y) / cos(x + i y)");
        v.def_unary(Tanh, "sinh(x + i y) / cosh(x + i y)");
        // http://functions.wolfram.com/01.13.02.0001.01
        v.def_unary(Acos, "π/2 + i ln(i (x + i y) + sqrt(1 - (x + i y)^2))");
        // http://functions.wolfram.com/01.26.02.0001.01
        v.def_unary(
            Acosh,
            "ln((x + i y) + sqrt((x + i y) - 1) sqrt((x + i y) + 1))",
        );
        // http://functions.wolfram.com/01.12.02.0001.01
        v.def_unary(Asin, "-i ln(i (x + i y) + sqrt(1 - (x + i y)^2))");
        // http://functions.wolfram.com/01.25.02.0001.01
        v.def_unary(Asinh, "ln((x + i y) + sqrt((x + i y)^2 + 1))");
        // http://functions.wolfram.com/01.14.02.0001.01
        v.def_unary(Atan, "i/2 (ln(1 - i (x + i y)) - ln(1 + i (x + i y)))");
        // http://functions.wolfram.com/01.27.02.0001.01
        v.def_unary(Atanh, "1/2 (ln(1 + (x + i y)) - ln(1 - (x + i y)))");
        v.def_binary(Log, "ln(x + i y) / ln(a + i b)");
        v
    }
}

impl VisitMut for ExpandComplexFunctions {
    fn visit_expr_mut(&mut self, e: &mut Expr) {
        use {
            BinaryOp::{Complex, *},
            NaryOp::*,
            TernaryOp::*,
            UnaryOp::*,
            ValueType::{Complex as ComplexT, *},
        };
        traverse_expr_mut(self, e);

        match e {
            unary!(Arg, binary!(Complex, x, y)) => {
                *e = Expr::binary(Atan2, take(y), take(x));
            }
            unary!(Arg, x) => {
                assert_eq!(x.ty, Real);
                *e = Expr::binary(Atan2, Expr::zero(), take(x));
            }
            unary!(Conj, binary!(Complex, x, y)) => {
                *e = Expr::binary(
                    Complex,
                    take(x),
                    Expr::nary(Times, vec![Expr::minus_one(), take(y)]),
                );
            }
            unary!(Conj, x) => {
                assert_eq!(x.ty, Real);
                *e = take(x);
            }
            unary!(Im, binary!(Complex, x, y)) => {
                *e = Expr::nary(
                    Plus,
                    vec![
                        // Keep `x` to preserve the domain.
                        Expr::nary(Times, vec![Expr::zero(), take(x)]),
                        take(y),
                    ],
                );
            }
            unary!(Im, x) => {
                assert_eq!(x.ty, Real);
                // Keep `x` to preserve the domain.
                *e = Expr::nary(Times, vec![Expr::zero(), take(x)]);
            }
            unary!(Re, binary!(Complex, x, y)) => {
                *e = Expr::nary(
                    Plus,
                    vec![
                        take(x),
                        // Keep `y` to preserve the domain.
                        Expr::nary(Times, vec![Expr::zero(), take(y)]),
                    ],
                );
            }
            unary!(Re, x) => {
                assert_eq!(x.ty, Real);
                *e = take(x);
            }
            unary!(Sign, binary!(Complex, x, y)) => {
                // sgn(x + i y) = f(x, y) - f(-x, y) + i (f(y, x) - f(-y, x)), where f is `ReSignNonnegative`.
                *e = Expr::binary(
                    Complex,
                    Expr::nary(
                        Plus,
                        vec![
                            Expr::binary(ReSignNonnegative, x.clone(), y.clone()),
                            Expr::nary(
                                Times,
                                vec![
                                    Expr::minus_one(),
                                    Expr::binary(
                                        ReSignNonnegative,
                                        Expr::nary(Times, vec![Expr::minus_one(), x.clone()]),
                                        y.clone(),
                                    ),
                                ],
                            ),
                        ],
                    ),
                    Expr::nary(
                        Plus,
                        vec![
                            Expr::binary(ReSignNonnegative, y.clone(), x.clone()),
                            Expr::nary(
                                Times,
                                vec![
                                    Expr::minus_one(),
                                    Expr::binary(
                                        ReSignNonnegative,
                                        Expr::nary(Times, vec![Expr::minus_one(), take(y)]),
                                        take(x),
                                    ),
                                ],
                            ),
                        ],
                    ),
                );
            }
            unary!(Sign, x) => {
                assert_eq!(x.ty, Real);
                // sgn(x) = f(x, 0) - f(-x, 0).
                *e = Expr::nary(
                    Plus,
                    vec![
                        Expr::binary(ReSignNonnegative, x.clone(), Expr::zero()),
                        Expr::nary(
                            Times,
                            vec![
                                Expr::minus_one(),
                                Expr::binary(
                                    ReSignNonnegative,
                                    Expr::nary(Times, vec![Expr::minus_one(), take(x)]),
                                    Expr::zero(),
                                ),
                            ],
                        ),
                    ],
                );
            }
            unary!(op @ (Sinc | UndefAt0 | Zeta), binary!(Complex, x, y)) => {
                let re_op = match op {
                    Sinc => ReSinc,
                    UndefAt0 => ReUndefAt0,
                    Zeta => ReZeta,
                    _ => unreachable!(),
                };
                let im_op = match op {
                    Sinc => ImSinc,
                    UndefAt0 => ImUndefAt0,
                    Zeta => ImZeta,
                    _ => unreachable!(),
                };
                *e = Expr::binary(
                    Complex,
                    Expr::binary(re_op, x.clone(), y.clone()),
                    Expr::binary(im_op, take(x), take(y)),
                );
            }
            unary!(op, binary!(Complex, x, y)) => {
                if let Some(template) = self.unary_ops.get(op) {
                    let mut new_e = template.clone();
                    Substitute::new(vec![take(x), take(y)]).visit_expr_mut(&mut new_e);
                    *e = new_e;
                }
            }
            binary!(Eq, x, y) if x.ty == ComplexT || y.ty == ComplexT => {
                *e = match (x, y) {
                    (binary!(Complex, a, b), binary!(Complex, x, y)) => Expr::nary(
                        AndN,
                        vec![
                            Expr::binary(Eq, take(a), take(x)),
                            Expr::binary(Eq, take(b), take(y)),
                        ],
                    ),
                    (binary!(Complex, a, b), x) => Expr::nary(
                        AndN,
                        vec![
                            Expr::binary(Eq, take(a), take(x)),
                            Expr::binary(Eq, take(b), Expr::zero()),
                        ],
                    ),
                    (a, binary!(Complex, x, y)) => Expr::nary(
                        AndN,
                        vec![
                            Expr::binary(Eq, take(a), take(x)),
                            Expr::binary(Eq, Expr::zero(), take(y)),
                        ],
                    ),
                    _ => panic!(), // `x.ty` or `y.ty` is wrong.
                };
            }
            binary!(Pow, binary!(Complex, x, y), constant!(a)) if a.to_f64() == Some(-1.0) => {
                let mut new_e = self.unary_ops[&Recip].clone();
                Substitute::new(vec![take(x), take(y)]).visit_expr_mut(&mut new_e);
                *e = new_e;
            }
            binary!(Pow, binary!(Complex, x, y), constant!(a)) if a.to_f64() == Some(2.0) => {
                let mut new_e = self.unary_ops[&Sqr].clone();
                Substitute::new(vec![take(x), take(y)]).visit_expr_mut(&mut new_e);
                *e = new_e;
            }
            binary!(op, x, y) if e.ty == ComplexT => {
                if let Some(template) = self.binary_ops.get(op) {
                    let mut new_e = template.clone();
                    let mut subst = match (x, y) {
                        (binary!(Complex, a, b), binary!(Complex, x, y)) => {
                            Substitute::new(vec![take(a), take(b), take(x), take(y)])
                        }
                        (a, binary!(Complex, x, y)) => {
                            Substitute::new(vec![take(a), Expr::zero(), take(x), take(y)])
                        }
                        (binary!(Complex, a, b), x) => {
                            Substitute::new(vec![take(a), take(b), take(x), Expr::zero()])
                        }
                        _ => panic!(), // `e.ty` is wrong.
                    };
                    subst.visit_expr_mut(&mut new_e);
                    *e = new_e;
                }
            }
            ternary!(IfThenElse, cond, t, f) if e.ty == ComplexT => {
                *e = match (t, f) {
                    (binary!(Complex, a, b), binary!(Complex, x, y)) => Expr::binary(
                        Complex,
                        Expr::ternary(IfThenElse, cond.clone(), take(a), take(x)),
                        Expr::ternary(IfThenElse, take(cond), take(b), take(y)),
                    ),
                    (binary!(Complex, a, b), x) => Expr::binary(
                        Complex,
                        Expr::ternary(IfThenElse, cond.clone(), take(a), take(x)),
                        Expr::ternary(IfThenElse, take(cond), take(b), Expr::zero()),
                    ),
                    (a, binary!(Complex, x, y)) => Expr::binary(
                        Complex,
                        Expr::ternary(IfThenElse, cond.clone(), take(a), take(x)),
                        Expr::ternary(IfThenElse, take(cond), Expr::zero(), take(y)),
                    ),
                    _ => panic!(), // `t.ty` or `f.ty` is wrong.
                }
            }
            nary!(Plus, xs) if e.ty == ComplexT => {
                let mut reals = vec![];
                let mut imags = vec![];
                for x in xs {
                    match x {
                        binary!(Complex, x, y) => {
                            reals.push(take(x));
                            imags.push(take(y));
                        }
                        _ => {
                            reals.push(take(x));
                        }
                    }
                }
                *e = Expr::binary(Complex, Expr::nary(Plus, reals), Expr::nary(Plus, imags));
            }
            nary!(Times, xs) if e.ty == ComplexT => {
                let mut it = xs.drain(..);
                let mut x = it.next().unwrap();
                for mut y in it {
                    x = match (&mut x, &mut y) {
                        (binary!(Complex, a, b), binary!(Complex, x, y)) => Expr::binary(
                            Complex,
                            Expr::nary(
                                Plus,
                                vec![
                                    Expr::nary(Times, vec![a.clone(), x.clone()]),
                                    Expr::nary(
                                        Times,
                                        vec![Expr::minus_one(), b.clone(), y.clone()],
                                    ),
                                ],
                            ),
                            Expr::nary(
                                Plus,
                                vec![
                                    Expr::nary(Times, vec![take(b), take(x)]),
                                    Expr::nary(Times, vec![take(a), take(y)]),
                                ],
                            ),
                        ),
                        (a, binary!(Complex, x, y)) | (binary!(Complex, x, y), a) => Expr::binary(
                            Complex,
                            Expr::nary(Times, vec![a.clone(), take(x)]),
                            Expr::nary(Times, vec![take(a), take(y)]),
                        ),
                        _ => panic!(), // `e.ty` is wrong.
                    };
                }
                *e = x;
            }
            _ => return,
        }

        UpdateMetadata.visit_expr_mut(e);
    }
}

/// Does the following tasks:
///
/// - Replaces [`BinaryOp::Ge`] and [`BinaryOp::Gt`] with [`BinaryOp::Le`] and [`BinaryOp::Lt`]
///   by flipping the signs of both sides of the inequalities.
/// - Transposes all terms to the left-hand sides of equations and inequalities,
///   leaving zeros on the right-hand sides.
pub struct NormalizeRelationalExprs;

impl VisitMut for NormalizeRelationalExprs {
    fn visit_expr_mut(&mut self, e: &mut Expr) {
        use {BinaryOp::*, NaryOp::*};
        traverse_expr_mut(self, e);

        match e {
            binary!(op @ (Eq | Le | Lt), x, y) => {
                // (op x y) → (op (Plus x (Times -1 y)) 0)
                *e = Expr::binary(
                    *op,
                    Expr::nary(
                        Plus,
                        vec![take(x), Expr::nary(Times, vec![Expr::minus_one(), take(y)])],
                    ),
                    Expr::zero(),
                )
            }
            binary!(op @ (Ge | Gt), x, y) => {
                // (op x y) → (inv-op (Plus y (Times -1 x)) 0)
                let inv_op = match op {
                    Ge => Le,
                    Gt => Lt,
                    _ => unreachable!(),
                };
                *e = Expr::binary(
                    inv_op,
                    Expr::nary(
                        Plus,
                        vec![take(y), Expr::nary(Times, vec![Expr::minus_one(), take(x)])],
                    ),
                    Expr::zero(),
                )
            }
            _ => (),
        }
    }
}

/// Precondition: [`NormalizeRelationalExprs`] has been applied to the expression.
pub struct ExpandBoole;

impl VisitMut for ExpandBoole {
    fn visit_expr_mut(&mut self, e: &mut Expr) {
        use {BinaryOp::*, NaryOp::*, UnaryOp::*};

        let mut modified = false;

        if let unary!(Boole, x) = e {
            match x {
                bool_constant!(false) => {
                    *e = Expr::zero();
                    modified = true;
                }
                bool_constant!(true) => {
                    *e = Expr::one();
                    modified = true;
                }
                unary!(Not, x) => {
                    *e = Expr::nary(
                        Plus,
                        vec![
                            Expr::one(),
                            Expr::nary(Times, vec![Expr::minus_one(), Expr::unary(Boole, take(x))]),
                        ],
                    );
                    modified = true;
                }
                binary!(op @ (Eq | Le | Lt), x, y) => {
                    assert!(matches!(y, constant!(a) if a.to_f64() == Some(0.0)));
                    let op = match op {
                        Eq => BooleEqZero,
                        Le => BooleLeZero,
                        Lt => BooleLtZero,
                        _ => unreachable!(),
                    };
                    *e = Expr::unary(op, take(x));
                    modified = true;
                }
                nary!(op @ (AndN | OrN), xs) => {
                    let arith_op = match op {
                        AndN => MinN,
                        OrN => MaxN,
                        _ => unreachable!(),
                    };
                    *e = Expr::nary(
                        arith_op,
                        xs.drain(..).map(|x| Expr::unary(Boole, x)).collect(),
                    );
                    modified = true;
                }
                _ => (),
            }
        }

        if modified {
            // `e` has not been visited yet.
            self.visit_expr_mut(e);
        } else {
            traverse_expr_mut(self, e);
        }
    }
}

/// Transforms `M1 ... Mn = 0` into `M1' ... Mn' = 0`,
/// where `Mi = mod(xi, yi)` and `Mi' = mod(xi + yi / 2, yi) - yi / 2`.
///
/// Precondition: [`PreTransform`] and [`NormalizeRelationalExprs`] have been applied to the expression,
/// and then the expression has been simplified.
///
/// It might be better to also apply [`ExpandBoole`] before this one,
/// since transforming equations in the conditions of `if` expressions
/// will not likely to help improve quality of the output.
pub struct ModEqTransform;

impl VisitMut for ModEqTransform {
    fn visit_expr_mut(&mut self, e: &mut Expr) {
        use {BinaryOp::*, NaryOp::*};
        traverse_expr_mut(self, e);

        match e {
            binary!(Eq, z, _) if matches!(z, binary!(Mod, _, _)) => {
                if let binary!(Mod, x, y) = z {
                    *z = Expr::nary(
                        Plus,
                        vec![
                            Expr::binary(
                                Mod,
                                Expr::nary(
                                    Plus,
                                    vec![
                                        take(x),
                                        Expr::nary(Times, vec![Expr::one_half(), y.clone()]),
                                    ],
                                ),
                                y.clone(),
                            ),
                            Expr::nary(Times, vec![Expr::minus_one_half(), y.clone()]),
                        ],
                    );
                }
            }
            binary!(Eq, nary!(Times, zs), _)
                if zs.iter().all(|z| matches!(z, binary!(Mod, _, _))) =>
            {
                for z in zs.iter_mut() {
                    if let binary!(Mod, x, y) = z {
                        *z = Expr::nary(
                            Plus,
                            vec![
                                Expr::binary(
                                    Mod,
                                    Expr::nary(
                                        Plus,
                                        vec![
                                            take(x),
                                            Expr::nary(Times, vec![Expr::one_half(), y.clone()]),
                                        ],
                                    ),
                                    y.clone(),
                                ),
                                Expr::nary(Times, vec![Expr::minus_one_half(), y.clone()]),
                            ],
                        );
                    }
                }
            }
            _ => (),
        }
    }
}

/// Flattens associative n-ary operations of the same kind.
///
/// If the expression has no arguments, it is replaced by the identity element of the operation.
/// If it has exactly one argument, it is replaced by that argument.
#[derive(Default)]
struct Flatten {
    modified: bool,
}

impl VisitMut for Flatten {
    fn visit_expr_mut(&mut self, e: &mut Expr) {
        use NaryOp::*;
        traverse_expr_mut(self, e);

        if let nary!(op @ (AndN | MaxN | MinN | OrN | Plus | Times), xs) = e {
            match &mut xs[..] {
                [] => {
                    *e = match op {
                        AndN => Expr::bool_constant(true),
                        OrN => Expr::bool_constant(false),
                        Plus => Expr::zero(),
                        Times => Expr::one(),
                        MaxN | MinN => panic!(),
                        _ => unreachable!(),
                    };
                    self.modified = true;
                }
                [x] => {
                    *e = take(x);
                    self.modified = true;
                }
                _ => {
                    *xs = xs.drain(..).fold(vec![], |mut acc, x| {
                        match x {
                            nary!(opx, mut xs) if opx == *op => {
                                acc.append(&mut xs);
                                self.modified = true;
                            }
                            _ => acc.push(x),
                        }
                        acc
                    });
                }
            }
        }
    }
}

/// Sorts terms in commutative n-ary operations by their structure.
/// Terms of kind [`ExprKind::Constant`](crate::ast::ExprKind::Constant) are moved to the beginning.
#[derive(Default)]
struct SortTerms {
    modified: bool,
}

fn cmp_terms(x: &Expr, y: &Expr) -> Ordering {
    use {BinaryOp::*, NaryOp::*};
    match (x, y) {
        (constant!(x), constant!(y)) if x.rational().is_some() && y.rational_pi().is_some() => {
            Ordering::Less
        }
        (constant!(x), constant!(y)) if x.rational_pi().is_some() && y.rational().is_some() => {
            Ordering::Greater
        }
        (constant!(x), constant!(y)) => {
            let x = x
                .interval()
                .iter()
                .fold(f64::INFINITY, |inf, x| inf.min(x.x.inf()));
            let y = y
                .interval()
                .iter()
                .fold(f64::INFINITY, |inf, x| inf.min(x.x.inf()));
            x.partial_cmp(&y).unwrap()
        }
        (constant!(_), _) => Ordering::Less,
        (_, constant!(_)) => Ordering::Greater,
        (binary!(Pow, x1, x2), binary!(Pow, y1, y2)) => {
            cmp_terms(x1, y1).then_with(|| cmp_terms(x2, y2))
        }
        (binary!(Pow, x1, x2), _) => cmp_terms(x1, y).then_with(|| cmp_terms(x2, &Expr::one())),
        (_, binary!(Pow, y1, y2)) => cmp_terms(x, y1).then_with(|| cmp_terms(&Expr::one(), y2)),
        (var!(x), var!(y)) => x.cmp(y),
        (var!(_), _) => Ordering::Less,
        (_, var!(_)) => Ordering::Greater,
        (nary!(Times, xs), nary!(Times, ys)) => (|| {
            for (x, y) in xs.iter().rev().zip(ys.iter().rev()) {
                let ord = cmp_terms(x, y);
                if ord != Ordering::Equal {
                    return ord;
                }
            }
            xs.len().cmp(&ys.len())
        })(),
        (nary!(Times, _), _) => Ordering::Less,
        (_, nary!(Times, _)) => Ordering::Greater,
        _ => Ordering::Equal,
    }
}

impl VisitMut for SortTerms {
    fn visit_expr_mut(&mut self, e: &mut Expr) {
        use NaryOp::*;
        traverse_expr_mut(self, e);

        if let nary!(AndN | MaxN | MinN | OrN | Plus | Times, xs) = e {
            if xs
                .windows(2)
                .any(|xs| cmp_terms(&xs[0], &xs[1]) == Ordering::Greater)
            {
                // (op … y … x …) /; x ≺ y → (op … x … y …)
                xs.sort_by(cmp_terms);
                self.modified = true;
            }
        }
    }
}

/// Selectively merges consecutive expressions.
fn transform_vec<F>(xs: &mut Vec<Expr>, f: F) -> bool
where
    F: Fn(&mut Expr, &mut Expr) -> Option<Expr>,
{
    let mut modified = false;
    *xs = xs.drain(..).fold(vec![], |mut acc, mut next| {
        if acc.is_empty() {
            acc.push(next)
        } else {
            let last = acc.last_mut().unwrap();
            if let Some(x) = f(last, &mut next) {
                *last = x;
                modified = true;
            } else {
                acc.push(next);
            }
        }
        acc
    });
    modified
}

/// If `x` represents a rational number, returns `f` applied to the [`Rational`] value;
/// otherwise, returns `false`.
fn test_rational<F>(x: &Expr, f: F) -> bool
where
    F: Fn(&Rational) -> bool,
{
    if let constant!(x) = x {
        if let Some(x) = x.rational() {
            return f(x);
        }
    }
    false
}

/// If both `x` and `y` represent rational numbers, returns `f` applied to the [`Rational`] values;
/// otherwise, returns `false`.
fn test_rationals<F>(x: &Expr, y: &Expr, f: F) -> bool
where
    F: Fn(&Rational, &Rational) -> bool,
{
    if let (constant!(x), constant!(y)) = (x, y) {
        if let (Some(x), Some(y)) = (x.rational(), y.rational()) {
            return f(x, y);
        }
    }
    false
}

/// Transforms expressions into simpler/normalized forms.
///
/// Precondition: [`UpdateMetadata`] has been applied to the expression
/// and it has not been modified since then.
#[derive(Default)]
struct Transform {
    modified: bool,
}

impl VisitMut for Transform {
    fn visit_expr_mut(&mut self, e: &mut Expr) {
        use {BinaryOp::*, NaryOp::*, TernaryOp::*, UnaryOp::*};
        traverse_expr_mut(self, e);

        macro_rules! boole_op {
            () => {
                BooleEqZero | BooleLeZero | BooleLtZero
            };
        }

        match e {
            unary!(op @ (Cos | Sin | Tan), nary!(Times, xs)) => {
                if let [constant!(a), nary!(Plus, ys)] = &mut xs[..] {
                    if let [constant!(b), ..] = &mut ys[..] {
                        // a (b + …) → a*b + a (…) /; a*b = q π ∧ q ∈ ℚ
                        let c = a.clone() * b.clone();
                        if c.rational_pi().is_some() {
                            *e = Expr::unary(
                                *op,
                                Expr::nary(
                                    Plus,
                                    vec![
                                        Expr::constant(c),
                                        Expr::nary(
                                            Times,
                                            vec![
                                                Expr::constant(take(a)),
                                                Expr::nary(Plus, ys.drain(1..).collect()),
                                            ],
                                        ),
                                    ],
                                ),
                            );
                            self.modified = true;
                        }
                    }
                }
            }
            unary!(op @ (Cos | Sin | Tan), nary!(Plus, xs)) => {
                if let [constant!(a), xs_tail @ ..] = &xs[..] {
                    if a.rational_pi().is_some() {
                        let a_reduced = match op {
                            Cos | Sin => a.clone().modulo(Real::tau()),
                            Tan => a.clone().modulo(Real::pi()),
                            _ => unreachable!(),
                        };
                        if a_reduced != *a {
                            *e = Expr::unary(
                                *op,
                                Expr::nary(
                                    Plus,
                                    iter::once(Expr::constant(a_reduced))
                                        .chain(xs_tail.iter().cloned())
                                        .collect(),
                                ),
                            );
                            self.modified = true;
                        }
                    }
                }
            }
            binary!(Max, x @ unary!(boole_op!(), _), constant!(y))
            | binary!(Max, constant!(y), x @ unary!(boole_op!(), _))
                if y.to_f64() == Some(0.0) =>
            {
                *e = take(x);
                self.modified = true;
            }
            binary!(Min, x @ unary!(boole_op!(), _), constant!(y))
            | binary!(Min, constant!(y), x @ unary!(boole_op!(), _))
                if y.to_f64() == Some(1.0) =>
            {
                *e = take(x);
                self.modified = true;
            }
            binary!(Pow | PowRational, x, constant!(a)) if a.to_f64() == Some(1.0) => {
                // x^1 → x
                *e = take(x);
                self.modified = true;
            }
            ternary!(IfThenElse, constant!(a), _, f) if a.to_f64() == Some(0.0) => {
                *e = take(f);
                self.modified = true;
            }
            ternary!(IfThenElse, constant!(a), t, _) if a.to_f64() == Some(1.0) => {
                *e = take(t);
                self.modified = true;
            }
            ternary!(IfThenElse, _, x, y) if x == y => {
                *e = take(x);
                self.modified = true;
            }
            nary!(AndN, xs) => {
                if xs.iter().any(|x| matches!(x, bool_constant!(false))) {
                    // (AndN … false …) → false
                    *e = Expr::bool_constant(false);
                    self.modified = true;
                } else {
                    let len = xs.len();
                    // Drop `true`s.
                    xs.retain(|x| !matches!(x, bool_constant!(true)));
                    self.modified = xs.len() < len;
                }
            }
            nary!(OrN, xs) => {
                if xs.iter().any(|x| matches!(x, bool_constant!(true))) {
                    // (OrN … true …) → true
                    *e = Expr::bool_constant(true);
                    self.modified = true;
                } else {
                    let len = xs.len();
                    // Drop `false`s.
                    xs.retain(|x| !matches!(x, bool_constant!(false)));
                    self.modified = xs.len() < len;
                }
            }
            nary!(Plus, xs) => {
                let len = xs.len();

                // Drop zeros.
                xs.retain(|x| !matches!(x, constant!(a) if a.to_f64() == Some(0.0)));

                transform_vec(xs, |x, y| {
                    if x == y {
                        // x + x → 2 x
                        return Some(Expr::nary(Times, vec![Expr::two(), take(x)]));
                    }
                    if let nary!(Times, ys) = y {
                        match &mut ys[..] {
                            [y1 @ constant!(_), y2] if x == y2 => {
                                // x + a x → (1 + a) x
                                return Some(Expr::nary(
                                    Times,
                                    vec![Expr::nary(Plus, vec![Expr::one(), take(y1)]), take(x)],
                                ));
                            }
                            _ => (),
                        }
                    }
                    if let (nary!(Times, xs), nary!(Times, ys)) = (x, y) {
                        match (&mut xs[..], &mut ys[..]) {
                            ([x1 @ constant!(_), x2s @ ..], [y1 @ constant!(_), y2s @ ..])
                                if x2s == y2s =>
                            {
                                // a x… + b x… → (a + b) x…
                                let mut v = vec![Expr::nary(Plus, vec![take(x1), take(y1)])];
                                v.extend(xs.drain(1..));
                                return Some(Expr::nary(Times, v));
                            }
                            (x1s, [y1 @ constant!(_), y2s @ ..]) if x1s == y2s => {
                                // x… + a x… → (1 + a) x…
                                let mut v = vec![Expr::nary(Plus, vec![Expr::one(), take(y1)])];
                                v.append(xs);
                                return Some(Expr::nary(Times, v));
                            }
                            _ => (),
                        }
                    }
                    None
                });

                self.modified = xs.len() < len;
            }
            nary!(Times, xs) => {
                let len = xs.len();

                // (Times … 0 … x …) → (Times … 0 …), where x is totally defined.
                // Do not drop x if it is not totally defined;
                // otherwise, tha replacement will alter the domain of the expression.
                if xs.iter().any(|x| {
                    matches!(x,
                        constant!(a) if a.to_f64() == Some(0.0)
                    )
                }) {
                    xs.retain(|x| {
                        !x.totally_defined || matches!(x, constant!(a) if a.to_f64() == Some(0.0))
                    });
                }

                // Drop ones.
                xs.retain(|x| !matches!(x, constant!(a) if a.to_f64() == Some(1.0)));

                // TODO: Apply the law of exponents while preserving the domain of the expression
                // by introducing a construct like `UnaryOp::UndefAt0` but more generalized.
                transform_vec(xs, |x, y| {
                    match (x, y) {
                        (
                            binary!(Pow, x1, x2 @ constant!(_)),
                            binary!(Pow, y1, y2 @ constant!(_)),
                        ) if x1 == y1
                            && test_rationals(x2, y2, |a, b| {
                                *a.denom() == 1
                                    && *b.denom() == 1
                                    && (*a < 0 && *b < 0 || *a >= 0 && *b >= 0)
                            }) =>
                        {
                            // x^a x^b /; a, b ∈ ℤ ∧ (a, b < 0 ∨ a, b ≥ 0) → x^(a + b)
                            Some(Expr::binary(
                                Pow,
                                take(x1),
                                Expr::nary(Plus, vec![take(x2), take(y2)]),
                            ))
                        }
                        (x, binary!(Pow, y1, y2 @ constant!(_)))
                            if x == y1 && test_rational(y2, |a| *a.denom() == 1 && *a >= 0) =>
                        {
                            // x x^a /; a ∈ ℤ ∧ a ≥ 0 → x^(1 + a)
                            Some(Expr::binary(
                                Pow,
                                take(x),
                                Expr::nary(Plus, vec![Expr::one(), take(y2)]),
                            ))
                        }
                        (x, y) if x == y => {
                            // x x → x^2
                            Some(Expr::binary(Pow, take(x), Expr::two()))
                        }
                        (x, ternary!(IfThenElse, cond, constant!(a), f))
                        | (ternary!(IfThenElse, cond, constant!(a), f), x)
                            if x.totally_defined && a.to_f64() == Some(0.0) =>
                        {
                            Some(Expr::ternary(
                                IfThenElse,
                                take(cond),
                                Expr::zero(),
                                Expr::nary(Times, vec![take(x), take(f)]),
                            ))
                        }
                        (x, ternary!(IfThenElse, cond, t, constant!(a)))
                        | (ternary!(IfThenElse, cond, t, constant!(a)), x)
                            if x.totally_defined && a.to_f64() == Some(0.0) =>
                        {
                            Some(Expr::ternary(
                                IfThenElse,
                                take(cond),
                                Expr::nary(Times, vec![take(x), take(t)]),
                                Expr::zero(),
                            ))
                        }
                        _ => None,
                    }
                });

                self.modified = xs.len() < len;
            }
            _ => (),
        }
    }
}

/// Performs constant folding.
#[derive(Default)]
struct FoldConstant {
    modified: bool,
}

impl VisitMut for FoldConstant {
    fn visit_expr_mut(&mut self, e: &mut Expr) {
        use {BinaryOp::*, NaryOp::*};
        traverse_expr_mut(self, e);

        match e {
            constant!(_) => (),
            binary!(op @ (Eq | Le | Lt), constant!(a), constant!(b)) if b.to_f64() == Some(0.0) => {
                let certainly_true = match op {
                    Eq => a.to_f64() == Some(0.0),
                    Le => {
                        a.interval().decoration() >= Decoration::Def
                            && a.interval().iter().all(|a| a.x.sup() <= 0.0)
                    }
                    Lt => {
                        a.interval().decoration() >= Decoration::Def
                            && a.interval().iter().all(|a| a.x.sup() < 0.0)
                    }
                    _ => unreachable!(),
                };
                let certainly_false = match op {
                    Eq => !a.interval().iter().any(|a| a.x.contains(0.0)),
                    Le => !a.interval().iter().any(|a| a.x.inf() <= 0.0),
                    Lt => !a.interval().iter().any(|a| a.x.inf() < 0.0),
                    _ => unreachable!(),
                };
                assert!(!(certainly_true && certainly_false));

                if certainly_true {
                    *e = Expr::bool_constant(true);
                    self.modified = true;
                } else if certainly_false {
                    *e = Expr::bool_constant(false);
                    self.modified = true;
                }
            }
            nary!(op @ (MaxN | MinN | Plus | Times), xs) => {
                if let [_, constant!(_), ..] = &mut xs[..] {
                    let bin_op = match op {
                        MaxN => Max,
                        MinN => Min,
                        Plus => Add,
                        Times => Mul,
                        _ => unreachable!(),
                    };
                    self.modified = transform_vec(xs, |x, y| {
                        if let (x @ constant!(_), y @ constant!(_)) = (x, y) {
                            let e = Expr::binary(bin_op, take(x), take(y));
                            let a = e.eval().unwrap();
                            Some(Expr::constant(a))
                        } else {
                            None
                        }
                    });
                }
            }
            _ => {
                if let Some(x) = e.eval() {
                    // Only fold constants which evaluate to the empty or a single interval
                    // since the branch cut tracking is not possible with the AST.
                    if x.interval().len() <= 1 {
                        *e = Expr::constant(x);
                        self.modified = true;
                    }
                }
            }
        }
    }
}

/// Replaces arithmetic expressions with [`UnaryOp::Neg`], [`BinaryOp::Sub`], [`UnaryOp::Recip`]
/// and [`BinaryOp::Div`], whenever appropriate.
pub struct SubDivTransform;

impl VisitMut for SubDivTransform {
    fn visit_expr_mut(&mut self, e: &mut Expr) {
        use {BinaryOp::*, NaryOp::*, UnaryOp::*};
        traverse_expr_mut(self, e);

        match e {
            nary!(Plus, xs) => {
                let (lhs, rhs) =
                    xs.drain(..)
                        .fold((vec![], vec![]), |(mut lhs, mut rhs), mut e| {
                            match e {
                                nary!(Times, ref mut xs) => match &xs[..] {
                                    [constant!(a), ..] if a.to_f64() == Some(-1.0) => {
                                        rhs.push(Expr::nary(Times, xs.drain(1..).collect()));
                                    }
                                    _ => lhs.push(e),
                                },
                                _ => {
                                    lhs.push(e);
                                }
                            }
                            (lhs, rhs)
                        });

                *e = if lhs.is_empty() {
                    Expr::unary(Neg, Expr::nary(Plus, rhs))
                } else if rhs.is_empty() {
                    Expr::nary(Plus, lhs)
                } else {
                    Expr::binary(Sub, Expr::nary(Plus, lhs), Expr::nary(Plus, rhs))
                };
            }
            nary!(Times, xs) => {
                let (num, den) = xs
                    .drain(..)
                    .fold((vec![], vec![]), |(mut num, mut den), e| {
                        match e {
                            binary!(Pow, x, y @ constant!(_))
                                if test_rational(&y, |x| *x < 0.0) =>
                            {
                                if let constant!(y) = y {
                                    let factor = Expr::binary(Pow, x, Expr::constant(-y));
                                    den.push(factor)
                                } else {
                                    panic!()
                                }
                            }
                            _ => {
                                num.push(e);
                            }
                        }
                        (num, den)
                    });

                *e = if num.is_empty() {
                    Expr::unary(Recip, Expr::nary(Times, den))
                } else if den.is_empty() {
                    Expr::nary(Times, num)
                } else {
                    Expr::binary(Div, Expr::nary(Times, num), Expr::nary(Times, den))
                };
            }
            _ => (),
        }
    }
}

/// Does the following tasks:
///
/// - Replace arithmetic expressions with more optimal ones suitable for evaluation.
///   It completely removes [`NaryOp::Plus`] and [`NaryOp::Times`].
/// - Replace [`NaryOp::AndN`], [`NaryOp::MaxN`], [`NaryOp::MinN`], and [`NaryOp::OrN`]
///   with their binary counterparts.
pub struct PostTransform;

impl VisitMut for PostTransform {
    fn visit_expr_mut(&mut self, e: &mut Expr) {
        use {BinaryOp::*, NaryOp::*, UnaryOp::*};
        traverse_expr_mut(self, e);

        match e {
            binary!(Pow, x, constant!(a)) => {
                if let Some(a) = a.to_f64() {
                    if a == -1.0 {
                        *e = Expr::unary(Recip, take(x));
                    } else if a == 0.5 {
                        *e = Expr::unary(Sqrt, take(x));
                    } else if a == 1.0 {
                        *e = take(x);
                    } else if a == 2.0 {
                        *e = Expr::unary(Sqr, take(x));
                    } else if a as i32 as f64 == a {
                        *e = Expr::pown(take(x), a as i32);
                    }
                }
            }
            binary!(PowRational, x, constant!(a)) => {
                if let Some(a) = a.rational() {
                    if let (Some(n), Some(d)) = (a.numer().to_i32(), a.denom().to_u32()) {
                        let root = match d {
                            1 => take(x),
                            2 => Expr::unary(Sqrt, take(x)),
                            _ => Expr::rootn(take(x), d),
                        };
                        *e = match n {
                            -1 => Expr::unary(Recip, root),
                            1 => root,
                            2 => Expr::unary(Sqr, root),
                            _ => Expr::pown(root, n),
                        }
                    }
                }
            }
            nary!(op @ (AndN | MaxN | MinN | OrN | Plus), xs) => {
                // Assuming `e` is flattened.
                let bin_op = match op {
                    AndN => And,
                    MaxN => Max,
                    MinN => Min,
                    OrN => Or,
                    Plus => Add,
                    _ => unreachable!(),
                };
                *e = xs
                    .drain(..)
                    .reduce(|x, y| Expr::binary(bin_op, x, y))
                    .unwrap()
            }
            nary!(Times, xs) => {
                let mut it = xs.drain(..);
                // Assuming `e` is flattened.
                let first = it.next().unwrap();
                let second = it.next().unwrap();
                let init = match first {
                    constant!(a) if a.to_f64() == Some(-1.0) => Expr::unary(Neg, second),
                    _ => Expr::binary(BinaryOp::Mul, first, second),
                };
                *e = it.fold(init, |e, x| Expr::binary(BinaryOp::Mul, e, x))
            }
            _ => (),
        }
    }
}

/// Combines [`BinaryOp::Mul`] and [`BinaryOp::Add`] into [`TernaryOp::MulAdd`].
pub struct FuseMulAdd;

impl VisitMut for FuseMulAdd {
    fn visit_expr_mut(&mut self, e: &mut Expr) {
        use {BinaryOp::*, TernaryOp::*};
        traverse_expr_mut(self, e);

        match e {
            binary!(Add, binary!(Mul, x, y), z) | binary!(Add, z, binary!(Mul, x, y)) => {
                *e = Expr::ternary(MulAdd, take(x), take(y), take(z))
            }
            _ => (),
        }
    }
}

/// Precondition: [`UpdateMetadata`] have been applied to the expression
/// and it has not been modified since then.
#[derive(Default)]
struct FindUnknownTypeExpr<'a> {
    e: Option<&'a Expr>,
}

impl<'a> Visit<'a> for FindUnknownTypeExpr<'a> {
    fn visit_expr(&mut self, e: &'a Expr) {
        traverse_expr(self, e);

        if self.e.is_none() && e.ty == ValueType::Unknown {
            self.e = Some(e);
        }
    }
}

/// Updates metadata of the expression recursively.
pub struct UpdateMetadata;

impl VisitMut for UpdateMetadata {
    fn visit_expr_mut(&mut self, e: &mut Expr) {
        traverse_expr_mut(self, e);
        e.update_metadata();
    }
}

type SiteMap = HashMap<UnsafeExprRef, Site>;
type UnsafeExprRef = UnsafeRef<Expr>;

/// Assigns a branch cut site to each sub-expression which appears multiple times
/// and can perform branch cut.
///
/// It also collects boolean-valued expressions in the topological order.
///
/// Precondition: [`UpdateMetadata`] have been applied to the expression
/// and it has not been modified since then.
pub struct AssignSite {
    bool_exprs: Vec<UnsafeExprRef>,
    next_site: u8,
    real_exprs: Vec<UnsafeExprRef>,
    site_map: SiteMap,
    visited: HashSet<UnsafeExprRef>,
}

impl AssignSite {
    pub fn new() -> Self {
        AssignSite {
            bool_exprs: vec![],
            next_site: 0,
            real_exprs: vec![],
            site_map: HashMap::new(),
            visited: HashSet::new(),
        }
    }

    /// Returns `true` if the expression can perform branch cut on evaluation.
    fn expr_can_perform_cut(e: &Expr) -> bool {
        use {BinaryOp::*, UnaryOp::*};
        match e {
            unary!(
                BooleEqZero
                    | BooleLeZero
                    | BooleLtZero
                    | Ceil
                    | Digamma
                    | Floor
                    | Gamma
                    | Recip
                    | Tan,
                _
            )
            | binary!(
                Atan2
                    | Div
                    | Gcd
                    | Lcm
                    | Log
                    | Mod
                    | Pow
                    | PowRational
                    | RankedMax
                    | RankedMin
                    | ReSignNonnegative,
                _,
                _
            ) => true,
            pown!(_, n) if n % 2 == -1 => true,
            _ => false,
        }
    }
}

impl VisitMut for AssignSite {
    fn visit_expr_mut(&mut self, e: &mut Expr) {
        traverse_expr_mut(self, e);

        let e_ref = UnsafeExprRef::from(e);
        match self.visited.get(&e_ref) {
            Some(_) => {
                if !self.site_map.contains_key(&e_ref)
                    && Self::expr_can_perform_cut(e)
                    && self.next_site <= Site::MAX
                {
                    self.site_map.insert(e_ref, Site::new(self.next_site));
                    self.next_site += 1;
                }
            }
            _ => {
                match e.ty {
                    ValueType::Boolean => self.bool_exprs.push(e_ref),
                    ValueType::Real => self.real_exprs.push(e_ref),
                    _ => (),
                }
                self.visited.insert(e_ref);
            }
        }
    }
}

/// Collects real-valued expressions, ordered topologically.
///
/// Precondition: [`UpdateMetadata`] has been applied to the expression
/// and it has not been modified since then.
pub struct CollectRealExprs {
    exprs: Vec<UnsafeExprRef>,
    index_map: HashMap<UnsafeExprRef, usize>,
    vars: VarSet,
}

impl CollectRealExprs {
    pub fn new(vars: VarSet) -> Self {
        Self {
            exprs: vec![],
            index_map: HashMap::new(),
            vars,
        }
    }
}

impl VisitMut for CollectRealExprs {
    fn visit_expr_mut(&mut self, e: &mut Expr) {
        traverse_expr_mut(self, e);

        if e.ty == ValueType::Real && self.vars.contains(e.vars) {
            let e_ref = UnsafeExprRef::from(e);
            self.index_map.entry(e_ref).or_insert_with(|| {
                let index = self.exprs.len();
                self.exprs.push(e_ref);
                index
            });
        }
    }
}

/// Collects [`StaticTerm`]s and [`StaticForm`]s.
pub struct CollectStatic<'a> {
    pub forms: Vec<StaticForm>,
    pub terms: Vec<StaticTerm>,
    bool_exprs: Vec<UnsafeExprRef>,
    form_index: HashMap<UnsafeExprRef, FormIndex>,
    real_expr_index_map: HashMap<UnsafeExprRef, usize>,
    real_exprs: Vec<UnsafeExprRef>,
    site_map: SiteMap,
    var_index: &'a HashMap<VarSet, VarIndex>,
}

impl<'a> CollectStatic<'a> {
    pub fn new(
        assign_site: AssignSite,
        collect_real_exprs: CollectRealExprs,
        var_index: &'a HashMap<VarSet, VarIndex>,
    ) -> Self {
        let mut slf = Self {
            forms: vec![],
            terms: vec![],
            bool_exprs: assign_site.bool_exprs,
            form_index: HashMap::new(),
            real_expr_index_map: collect_real_exprs.index_map,
            real_exprs: collect_real_exprs.exprs,
            site_map: assign_site.site_map,
            var_index,
        };
        slf.collect_terms();
        slf.collect_atomic_forms();
        slf.collect_compound_forms();
        slf
    }

    fn collect_terms(&mut self) {
        use {BinaryOp::*, NaryOp::*, TernaryOp::*, UnaryOp::*};
        for (i, e) in self.real_exprs.iter().copied().enumerate() {
            let k = match &*e {
                bool_constant!(_) => None,
                constant!(x) => Some(StaticTermKind::Constant(Box::new(x.interval().clone()))),
                var!(_) => self
                    .var_index
                    .get(&e.vars)
                    .map(|&i| StaticTermKind::Var(i, e.vars.var_type())),
                unary!(op, x) => (|| {
                    Some(match op {
                        Abs => ScalarUnaryOp::Abs,
                        Acos => ScalarUnaryOp::Acos,
                        Acosh => ScalarUnaryOp::Acosh,
                        AiryAi => ScalarUnaryOp::AiryAi,
                        AiryAiPrime => ScalarUnaryOp::AiryAiPrime,
                        AiryBi => ScalarUnaryOp::AiryBi,
                        AiryBiPrime => ScalarUnaryOp::AiryBiPrime,
                        Asin => ScalarUnaryOp::Asin,
                        Asinh => ScalarUnaryOp::Asinh,
                        Atan => ScalarUnaryOp::Atan,
                        Atanh => ScalarUnaryOp::Atanh,
                        BooleEqZero => ScalarUnaryOp::BooleEqZero,
                        BooleLeZero => ScalarUnaryOp::BooleLeZero,
                        BooleLtZero => ScalarUnaryOp::BooleLtZero,
                        Ceil => ScalarUnaryOp::Ceil,
                        Chi => ScalarUnaryOp::Chi,
                        Ci => ScalarUnaryOp::Ci,
                        Cos => ScalarUnaryOp::Cos,
                        Cosh => ScalarUnaryOp::Cosh,
                        Digamma => ScalarUnaryOp::Digamma,
                        Ei => ScalarUnaryOp::Ei,
                        EllipticE => ScalarUnaryOp::EllipticE,
                        EllipticK => ScalarUnaryOp::EllipticK,
                        Erf => ScalarUnaryOp::Erf,
                        Erfc => ScalarUnaryOp::Erfc,
                        Erfi => ScalarUnaryOp::Erfi,
                        Exp => ScalarUnaryOp::Exp,
                        Floor => ScalarUnaryOp::Floor,
                        FresnelC => ScalarUnaryOp::FresnelC,
                        FresnelS => ScalarUnaryOp::FresnelS,
                        Gamma => ScalarUnaryOp::Gamma,
                        InverseErf => ScalarUnaryOp::InverseErf,
                        InverseErfc => ScalarUnaryOp::InverseErfc,
                        Li => ScalarUnaryOp::Li,
                        Ln => ScalarUnaryOp::Ln,
                        LnGamma => ScalarUnaryOp::LnGamma,
                        Neg => ScalarUnaryOp::Neg,
                        Recip => ScalarUnaryOp::Recip,
                        Shi => ScalarUnaryOp::Shi,
                        Si => ScalarUnaryOp::Si,
                        Sin => ScalarUnaryOp::Sin,
                        Sinc => ScalarUnaryOp::Sinc,
                        Sinh => ScalarUnaryOp::Sinh,
                        Sqr => ScalarUnaryOp::Sqr,
                        Sqrt => ScalarUnaryOp::Sqrt,
                        Tan => ScalarUnaryOp::Tan,
                        Tanh => ScalarUnaryOp::Tanh,
                        UndefAt0 => ScalarUnaryOp::UndefAt0,
                        Zeta => ScalarUnaryOp::Zeta,
                        Arg | Boole | Conj | Im | Re | Not | Sign => return None,
                    })
                })()
                .map(|op| StaticTermKind::Unary(op, self.store_index(x))),
                binary!(op @ (RankedMax | RankedMin), nary!(List, xs), n) => {
                    let op = match op {
                        RankedMax => RankedMinMaxOp::RankedMax,
                        RankedMin => RankedMinMaxOp::RankedMin,
                        _ => unreachable!(),
                    };
                    Some(StaticTermKind::RankedMinMax(
                        op,
                        xs.iter().map(|x| self.store_index(x)).collect(),
                        self.store_index(n),
                    ))
                }
                binary!(op, x, y) => (|| {
                    Some(match op {
                        Add => ScalarBinaryOp::Add,
                        Atan2 => ScalarBinaryOp::Atan2,
                        BesselI => ScalarBinaryOp::BesselI,
                        BesselJ => ScalarBinaryOp::BesselJ,
                        BesselK => ScalarBinaryOp::BesselK,
                        BesselY => ScalarBinaryOp::BesselY,
                        Div => ScalarBinaryOp::Div,
                        GammaInc => ScalarBinaryOp::GammaInc,
                        Gcd => ScalarBinaryOp::Gcd,
                        ImSinc => ScalarBinaryOp::ImSinc,
                        ImUndefAt0 => ScalarBinaryOp::ImUndefAt0,
                        ImZeta => ScalarBinaryOp::ImZeta,
                        LambertW => ScalarBinaryOp::LambertW,
                        Lcm => ScalarBinaryOp::Lcm,
                        Log => ScalarBinaryOp::Log,
                        Max => ScalarBinaryOp::Max,
                        Min => ScalarBinaryOp::Min,
                        Mod => ScalarBinaryOp::Mod,
                        Mul => ScalarBinaryOp::Mul,
                        Pow => ScalarBinaryOp::Pow,
                        PowRational => ScalarBinaryOp::PowRational,
                        ReSignNonnegative => ScalarBinaryOp::ReSignNonnegative,
                        ReSinc => ScalarBinaryOp::ReSinc,
                        ReUndefAt0 => ScalarBinaryOp::ReUndefAt0,
                        ReZeta => ScalarBinaryOp::ReZeta,
                        Sub => ScalarBinaryOp::Sub,
                        And | Complex | Eq | ExplicitRel(_) | Ge | Gt | Le | Lt | Or
                        | RankedMax | RankedMin => return None,
                    })
                })()
                .map(|op| StaticTermKind::Binary(op, self.store_index(x), self.store_index(y))),
                ternary!(op, x, y, z) => Some(match op {
                    IfThenElse => ScalarTernaryOp::IfThenElse,
                    MulAdd => ScalarTernaryOp::MulAdd,
                })
                .map(|op| {
                    StaticTermKind::Ternary(
                        op,
                        self.store_index(x),
                        self.store_index(y),
                        self.store_index(z),
                    )
                }),
                nary!(AndN | List | MaxN | MinN | OrN | Plus | Times, _) => None,
                pown!(x, n) => Some(StaticTermKind::Pown(self.store_index(x), *n)),
                rootn!(x, n) => Some(StaticTermKind::Rootn(self.store_index(x), *n)),
                error!() => panic!(),
                uninit!() => panic!(),
            };
            if let Some(k) = k {
                self.terms.push(StaticTerm {
                    kind: k,
                    site: self.site_map.get(&UnsafeExprRef::from(&e)).copied(),
                    store_index: StoreIndex::new(i),
                    vars: e.vars,
                })
            }
        }
    }

    fn collect_atomic_forms(&mut self) {
        use BinaryOp::*;
        for e in self.bool_exprs.iter().copied() {
            let k = match &*e {
                binary!(op @ (Eq | Le | Lt), x, _) => {
                    let op = match op {
                        Eq => RelOp::EqZero,
                        Le => RelOp::LeZero,
                        Lt => RelOp::LtZero,
                        _ => unreachable!(),
                    };
                    Some(StaticFormKind::Atomic(op, self.store_index(x)))
                }
                _ => None,
            };
            if let Some(k) = k {
                self.form_index
                    .insert(UnsafeExprRef::from(&e), self.forms.len() as FormIndex);
                self.forms.push(StaticForm { kind: k })
            }
        }
    }

    fn collect_compound_forms(&mut self) {
        use {BinaryOp::*, UnaryOp::*};
        for e in self.bool_exprs.iter().copied() {
            let k = match &*e {
                bool_constant!(a) => Some(StaticFormKind::Constant(*a)),
                binary!(ExplicitRel(_), _, _) => Some(StaticFormKind::Constant(true)),
                unary!(Not, x) => Some(StaticFormKind::Not(self.form_index(x))),
                binary!(And, x, y) => {
                    Some(StaticFormKind::And(self.form_index(x), self.form_index(y)))
                }
                binary!(Or, x, y) => {
                    Some(StaticFormKind::Or(self.form_index(x), self.form_index(y)))
                }
                _ => None,
            };
            if let Some(k) = k {
                self.form_index
                    .insert(UnsafeExprRef::from(&e), self.forms.len() as FormIndex);
                self.forms.push(StaticForm { kind: k })
            }
        }
    }

    fn form_index(&self, e: &Expr) -> FormIndex {
        self.form_index[&UnsafeExprRef::from(e)]
    }

    fn store_index(&self, e: &Expr) -> StoreIndex {
        let e_ref = UnsafeExprRef::from(e);
        let index = self.real_expr_index_map[&e_ref];
        StoreIndex::new(index)
    }
}

/// Finds the store index of the term `e` in `(ExplicitRel x e)`
/// which can be nested in top-level [`BinaryOp::And`] operations,
/// where `x` is the variable specified in the constructor.
pub struct FindExplicitRelation<'a> {
    collector: &'a CollectStatic<'a>,
    store_index: Option<StoreIndex>,
    variable: VarSet,
}

impl<'a> FindExplicitRelation<'a> {
    pub fn new(collector: &'a CollectStatic, variable: VarSet) -> Self {
        Self {
            collector,
            store_index: None,
            variable,
        }
    }

    pub fn get(&self) -> Option<StoreIndex> {
        self.store_index
    }
}

impl<'a> Visit<'a> for FindExplicitRelation<'a> {
    fn visit_expr(&mut self, e: &'a Expr) {
        use BinaryOp::*;
        match e {
            binary!(And, _, _) => traverse_expr(self, e),
            binary!(ExplicitRel(_), x @ var!(_), e) if x.vars == self.variable => {
                self.store_index = Some(StoreIndex::new(
                    self.collector.real_expr_index_map[&UnsafeExprRef::from(e)],
                ));
            }
            _ => (),
        }
    }
}

/// Precondition: [`PreTransform`] has been applied to the expression.
pub fn expand_complex_functions(e: &mut Expr) {
    UpdateMetadata.visit_expr_mut(e);
    ExpandComplexFunctions::default().visit_expr_mut(e);
}

pub fn find_unknown_type_expr(e: &Expr) -> Option<&Expr> {
    let mut v = FindUnknownTypeExpr::default();
    v.visit_expr(e);
    v.e
}

/// Precondition: [`PreTransform`] has been applied to the expression.
pub fn simplify(e: &mut Expr) {
    loop {
        let mut fl = Flatten::default();
        fl.visit_expr_mut(e);
        let mut s = SortTerms::default();
        s.visit_expr_mut(e);
        let mut f = FoldConstant::default();
        f.visit_expr_mut(e);
        UpdateMetadata.visit_expr_mut(e);
        let mut t = Transform::default();
        t.visit_expr_mut(e);
        if !fl.modified && !s.modified && !f.modified && !t.modified {
            break;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        context::{Context, VarProps},
        parse::parse_expr,
    };

    #[test]
    fn normalize_not_exprs() {
        let ctx = Context::new()
            .def("z", Def::var("z", VarProps::default()))
            .def("w", Def::var("w", VarProps::default()));

        let test = |input, expected| {
            let mut e = parse_expr(input, &[Context::builtin(), &ctx]).unwrap();
            PreTransform.visit_expr_mut(&mut e);
            NormalizeNotExprs.visit_expr_mut(&mut e);
            assert_eq!(format!("{}", e.dump_short()), expected);
        };

        test("!(x = y)", "(Not (Eq x y))");
        test("!!(x = y)", "(Eq x y)");
        test("!!!(x = y)", "(Not (Eq x y))");
        test("!(x = y && z = w)", "(OrN (Not (Eq x y)) (Not (Eq z w)))");
        test("!(x = y || z = w)", "(AndN (Not (Eq x y)) (Not (Eq z w)))");
    }

    #[test]
    fn pre_transform() {
        fn test(input: &str, expected: &str) {
            let mut e = parse_expr(input, &[Context::builtin()]).unwrap();
            PreTransform.visit_expr_mut(&mut e);
            assert_eq!(format!("{}", e.dump_short()), expected);
        }

        test("-x", "(Times -1 x)");
        test("sqrt(x)", "(Pow x 0.5)");
        test("x - y", "(Plus x (Times -1 y))");
        test("x/y", "(Times x (Pow y -1))");
        test("sin(x)/x", "(Sinc (UndefAt0 x))");
        test("x/sin(x)", "(Pow (Sinc (UndefAt0 x)) -1)");
        test("x y", "(Times x y)");
    }

    #[test]
    fn expand_complex_functions() {
        let ctx = Context::new()
            .def(
                "a",
                Def::var(
                    "a",
                    VarProps {
                        totally_defined: true,
                        ty: ValueType::Real,
                        ..Default::default()
                    },
                ),
            )
            .def(
                "b",
                Def::var(
                    "b",
                    VarProps {
                        totally_defined: true,
                        ty: ValueType::Real,
                        ..Default::default()
                    },
                ),
            );

        let test = |input, expected| {
            let mut e = parse_expr(input, &[Context::builtin(), &ctx]).unwrap();
            PreTransform.visit_expr_mut(&mut e);
            super::expand_complex_functions(&mut e);
            simplify(&mut e);
            assert_eq!(format!("{}", e.dump_short()), expected);
        };

        test("|x + i y|", "(Pow (Plus (Pow x 2) (Pow y 2)) 0.5)");
        test("arg(x + i y)", "(Atan2 y x)");
        test("arg(x)", "(Atan2 0 x)");
        test("~(x + i y)", "(Complex x (Times -1 y))");
        test("~x", "x");
        test("Im(x + i y)", "y");
        test("Im(x)", "0");
        test("Re(x + i y)", "x");
        test("Re(x)", "x");
        test("a + i b = x + i y", "(AndN (Eq a x) (Eq b y))");
        test("a = x + i y", "(AndN (Eq a x) (Eq 0 y))");
        test("a + i b = x", "(AndN (Eq a x) (Eq b 0))");
        test("(a + i b) + (x + i y)", "(Complex (Plus a x) (Plus b y))");
        test("a + (x + i y)", "(Complex (Plus a x) y)");
        test("(a + i b) + x", "(Complex (Plus a x) b)");
        test(
            "(a + i b) (x + i y)",
            "(Complex (Plus (Times a x) (Times -1 b y)) (Plus (Times b x) (Times a y)))",
        );
        test("a (x + i y)", "(Complex (Times a x) (Times a y))");
        test("(a + i b) x", "(Complex (Times a x) (Times b x))");
    }

    #[test]
    fn normalize_relational_exprs() {
        fn test(input: &str, expected: &str) {
            let mut e = parse_expr(input, &[Context::builtin()]).unwrap();
            PreTransform.visit_expr_mut(&mut e);
            NormalizeRelationalExprs.visit_expr_mut(&mut e);
            assert_eq!(format!("{}", e.dump_short()), expected);
        }

        test("x = y", "(Eq (Plus x (Times -1 y)) 0)");
        test("x ≥ y", "(Le (Plus y (Times -1 x)) 0)");
        test("x > y", "(Lt (Plus y (Times -1 x)) 0)");
        test("x ≤ y", "(Le (Plus x (Times -1 y)) 0)");
        test("x < y", "(Lt (Plus x (Times -1 y)) 0)");
    }

    #[test]
    fn expand_boole() {
        fn test(input: &str, expected: &str) {
            let mut e = parse_expr(&format!("if({}, x, y)", input), &[Context::builtin()]).unwrap();
            PreTransform.visit_expr_mut(&mut e);
            ExpandBoole.visit_expr_mut(&mut e);
            assert_eq!(
                format!("{}", e.dump_short()),
                format!("(IfThenElse {} x y)", expected)
            );
        }

        test("false", "0");
        test("true", "1");
        test("¬x", "(Plus 1 (Times -1 (Boole x)))");
        test("x ∧ y", "(MinN (Boole x) (Boole y))");
        test("x ∨ y", "(MaxN (Boole x) (Boole y))");
        test("x = 0", "(BooleEqZero x)");
        test("x ≤ 0", "(BooleLeZero x)");
        test("x < 0", "(BooleLtZero x)");
    }

    #[test]
    fn mod_eq_transform() {
        let ctx = Context::new()
            .def("z", Def::var("z", VarProps::default()))
            .def("w", Def::var("w", VarProps::default()));

        let test = |input, expected| {
            let mut e = parse_expr(input, &[Context::builtin(), &ctx]).unwrap();
            PreTransform.visit_expr_mut(&mut e);
            ModEqTransform.visit_expr_mut(&mut e);
            assert_eq!(format!("{}", e.dump_short()), expected);
        };

        test(
            "mod(x, y) = 0",
            "(Eq (Plus (Mod (Plus x (Times 0.5 y)) y) (Times -0.5 y)) 0)",
        );
        test(
            "mod(x, y) mod(z, w) = 0",
            "(Eq (Times (Plus (Mod (Plus x (Times 0.5 y)) y) (Times -0.5 y)) (Plus (Mod (Plus z (Times 0.5 w)) w) (Times -0.5 w))) 0)",
        );
    }

    #[test]
    fn flatten() {
        let ctx = Context::new().def("z", Def::var("z", VarProps::default()));

        let test = |input, expected| {
            let mut e = parse_expr(input, &[Context::builtin(), &ctx]).unwrap();
            PreTransform.visit_expr_mut(&mut e);
            UpdateMetadata.visit_expr_mut(&mut e);
            Transform::default().visit_expr_mut(&mut e);
            let input = format!("{}", e.dump_short());
            let mut v = Flatten::default();
            v.visit_expr_mut(&mut e);
            let output = format!("{}", e.dump_short());
            assert_eq!(format!("{}", e.dump_short()), expected);
            assert_eq!(v.modified, input != output);
        };

        test("x + y + z", "(Plus x y z)");
        test("(x + y) + z", "(Plus x y z)");
        test("x + (y + z)", "(Plus x y z)");
        test("x y z", "(Times x y z)");
        test("(x y) z", "(Times x y z)");
        test("x (y z)", "(Times x y z)");
        test("0 + 0", "0");
        test("1 1", "1");
    }

    #[test]
    fn sort_terms() {
        let ctx = Context::new().def("z", Def::var("z", VarProps::default()));

        let test = |input, expected| {
            let mut e = parse_expr(input, &[Context::builtin(), &ctx]).unwrap();
            PreTransform.visit_expr_mut(&mut e);
            FoldConstant::default().visit_expr_mut(&mut e);
            Flatten::default().visit_expr_mut(&mut e);
            FoldConstant::default().visit_expr_mut(&mut e);
            let input = format!("{}", e.dump_short());
            let mut v = SortTerms::default();
            v.visit_expr_mut(&mut e);
            let output = format!("{}", e.dump_short());
            assert_eq!(output, expected);
            assert_eq!(v.modified, input != output);
        };

        test("1 + x", "(Plus 1 x)");
        test("x + 1", "(Plus 1 x)");
        test("x + 2x", "(Plus x (Times 2 x))");
        test("2x + x", "(Plus x (Times 2 x))");
        test("x + sqrt(-1) x", "(Plus x (Times @ x))");
        test("sqrt(-1) x + x", "(Plus x (Times @ x))");
        test("2 x", "(Times 2 x)");
        test("x 2", "(Times 2 x)");
        test("x x^2", "(Times x (Pow x 2))");
        test("x^2 x", "(Times x (Pow x 2))");
        test("x x^-1", "(Times (Pow x -1) x)");
        test("x^-1 x", "(Times (Pow x -1) x)");

        test("x y", "(Times x y)");
        test("y x", "(Times x y)");

        test("y z + x y z", "(Plus (Times y z) (Times x y z))");
        test("x y z + y z", "(Plus (Times y z) (Times x y z))");

        test("x y + sin(z)", "(Plus (Times x y) (Sin z))");
        test("sin(z) + x y", "(Plus (Times x y) (Sin z))");
    }

    #[test]
    fn transform() {
        fn test(input: &str, expected: &str) {
            let mut e = parse_expr(input, &[Context::builtin()]).unwrap();
            PreTransform.visit_expr_mut(&mut e);
            ExpandBoole.visit_expr_mut(&mut e);
            Flatten::default().visit_expr_mut(&mut e);
            FoldConstant::default().visit_expr_mut(&mut e);
            Flatten::default().visit_expr_mut(&mut e);
            let input = format!("{}", e.dump_short());
            UpdateMetadata.visit_expr_mut(&mut e);
            let mut v = Transform::default();
            v.visit_expr_mut(&mut e);
            FoldConstant::default().visit_expr_mut(&mut e);
            Flatten::default().visit_expr_mut(&mut e);
            let output = format!("{}", e.dump_short());
            assert_eq!(output, expected);
            assert_eq!(v.modified, input != output);
        }

        test("false && false", "False");
        test("false && true", "False");
        test("true && false", "False");
        test("true && true", "True");

        test("false && y = x", "False");
        test("y = x && false", "False");
        test("true && y = x", "(Eq y x)");
        test("y = x && true", "(Eq y x)");

        test("false || false", "False");
        test("false || true", "True");
        test("true || false", "True");
        test("true || true", "True");

        test("false || y = x", "(Eq y x)");
        test("y = x || false", "(Eq y x)");
        test("true || y = x", "True");
        test("y = x || true", "True");

        test("x^1", "x");

        test("if(false, x, y)", "y");
        test("if(true, x, y)", "x");
        test("if(x < 0, y, y)", "y");

        test("0 + x", "x");
        test("x + x", "(Times 2 x)");
        test("x + 2x", "(Times 3 x)");
        test("2x + 3x", "(Times 5 x)");
        test("x y + x y", "(Times 2 x y)");
        test("x y + 2x y", "(Times 3 x y)");
        test("2x y + 3x y", "(Times 5 x y)");

        test("0 sin(x)", "0");
        test("0 sqrt(x)", "(Times 0 (Pow x 0.5))");
        test("0 sqrt(x) sin(x)", "(Times 0 (Pow x 0.5))");
        test("1 sqrt(x)", "(Pow x 0.5)");

        test("x x", "(Pow x 2)");
        test("x x^2", "(Pow x 3)");
        test("x^2 x^3", "(Pow x 5)");
        test("x^-2 x^-3", "(Pow x -5)");
        test("x^-2 x^3", "(Times (Pow x -2) (Pow x 3))");
        test("x^2 x^2", "(Pow x 4)");
        test("sqrt(x) sqrt(x)", "(Pow (Pow x 0.5) 2)");

        test(
            "y if(x < 0, 0, 1)",
            "(IfThenElse (BooleLtZero x) 0 (Times y 1))",
        );
        test(
            "if(x < 0, 0, 1) y",
            "(IfThenElse (BooleLtZero x) 0 (Times y 1))",
        );
        test(
            "y if(x < 0, 1, 0)",
            "(IfThenElse (BooleLtZero x) (Times y 1) 0)",
        );
        test(
            "if(x < 0, 1, 0) y",
            "(IfThenElse (BooleLtZero x) (Times y 1) 0)",
        );

        test("sin(4(π + x))", "(Sin (Plus @ (Times 4 x)))");
        test("cos(4(π + x))", "(Cos (Plus @ (Times 4 x)))");
        test("tan(3(π + x))", "(Tan (Plus @ (Times 3 x)))");
        test("sin(4π + 4x)", "(Sin (Plus 0 (Times 4 x)))");
        test("cos(4π + 4x)", "(Cos (Plus 0 (Times 4 x)))");
        test("tan(3π + 3x)", "(Tan (Plus 0 (Times 3 x)))");
    }

    #[test]
    fn post_transform() {
        let ctx = Context::new().def("z", Def::var("z", VarProps::default()));

        let test = |input, expected| {
            let mut e = parse_expr(input, &[Context::builtin(), &ctx]).unwrap();
            PreTransform.visit_expr_mut(&mut e);
            Flatten::default().visit_expr_mut(&mut e);
            FoldConstant::default().visit_expr_mut(&mut e);
            Flatten::default().visit_expr_mut(&mut e);
            PostTransform.visit_expr_mut(&mut e);
            assert_eq!(format!("{}", e.dump_short()), expected);
        };

        test("x^-1", "(Recip x)");
        test("x^0", "(Pown x 0)");
        test("x^(1/2)", "(Sqrt x)");
        test("x^1", "x");
        test("x^2", "(Sqr x)");
        test("x^3", "(Pown x 3)");

        test("x^^-1", "(Recip x)");
        test("x^^0", "(Pown x 0)");
        test("x^^1", "x");
        test("x^^2", "(Sqr x)");
        test("x^^3", "(Pown x 3)");
        test("x^^(1/2)", "(Sqrt x)");
        test("x^^(3/2)", "(Pown (Sqrt x) 3)");
        test("x^^(-2/3)", "(Pown (Rootn x 3) -2)");
        test("x^^(-1/3)", "(Recip (Rootn x 3))");
        test("x^^(1/3)", "(Rootn x 3)");
        test("x^^(2/3)", "(Sqr (Rootn x 3))");

        test("x + y", "(Add x y)");
        test("x + y + z", "(Add (Add x y) z)");

        test("x y", "(Mul x y)");
        test("x y z", "(Mul (Mul x y) z)");
    }

    #[test]
    fn fuse_mul_add() {
        let ctx = Context::new().def("z", Def::var("z", VarProps::default()));

        let test = |input, expected| {
            let mut e = parse_expr(input, &[Context::builtin(), &ctx]).unwrap();
            PreTransform.visit_expr_mut(&mut e);
            PostTransform.visit_expr_mut(&mut e);
            FuseMulAdd.visit_expr_mut(&mut e);
            assert_eq!(format!("{}", e.dump_short()), expected);
        };

        test("x y + z", "(MulAdd x y z)");
        test("z + x y", "(MulAdd x y z)");
    }
}
