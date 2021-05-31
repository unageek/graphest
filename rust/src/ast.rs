use crate::{interval_set::TupperIntervalSet, rational_ops};
use bitflags::*;
use inari::{const_dec_interval, DecInterval};
use rug::Rational;
use std::{
    collections::hash_map::DefaultHasher,
    fmt,
    hash::{Hash, Hasher},
};

pub type ExprId = u32;
pub const UNINIT_EXPR_ID: ExprId = ExprId::MAX;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum UnaryOp {
    Abs,
    Acos,
    Acosh,
    AiryAi,
    AiryAiPrime,
    AiryBi,
    AiryBiPrime,
    Asin,
    Asinh,
    Atan,
    Atanh,
    Ceil,
    Chi,
    Ci,
    Cos,
    Cosh,
    Digamma,
    Ei,
    EllipticE,
    EllipticK,
    Erf,
    Erfc,
    Erfi,
    Exp,
    Exp10,
    Exp2,
    Floor,
    FresnelC,
    FresnelS,
    Gamma,
    Li,
    Ln,
    Log10,
    Neg,
    Not,
    One,
    Recip,
    Shi,
    Si,
    Sin,
    Sinc,
    Sinh,
    Sqr,
    Sqrt,
    Tan,
    Tanh,
    UndefAt0,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum BinaryOp {
    Add,
    And,
    Atan2,
    BesselI,
    BesselJ,
    BesselK,
    BesselY,
    Div,
    Eq,
    GammaInc,
    Gcd,
    Ge,
    Gt,
    Lcm,
    Le,
    Log,
    Lt,
    Max,
    Min,
    Mod,
    Mul,
    Neq,
    Nge,
    Ngt,
    Nle,
    Nlt,
    Or,
    Pow,
    RankedMax,
    RankedMin,
    Sub,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum TernaryOp {
    MulAdd,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum NaryOp {
    List,
    Plus,
    Times,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub enum ExprKind {
    Constant(Box<(TupperIntervalSet, Option<Rational>)>),
    Var(String),
    Unary(UnaryOp, Box<Expr>),
    Binary(BinaryOp, Box<Expr>, Box<Expr>),
    Ternary(TernaryOp, Box<Expr>, Box<Expr>, Box<Expr>),
    Nary(NaryOp, Vec<Expr>),
    Pown(Box<Expr>, i32),
    Rootn(Box<Expr>, u32),
    Uninit,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ValueType {
    Scalar,
    Vector,
    Boolean,
    Unknown,
}

bitflags! {
    /// A set of free variables, subset of {x, y, n_θ}.
    pub struct VarSet: u8 {
        const EMPTY = 0;
        const X = 0b01;
        const Y = 0b10;
        const XY = 0b11;
        const N_THETA = 0b100;
    }
}

/// An AST node for an expression.
#[derive(Clone, Debug)]
pub struct Expr {
    pub id: ExprId,
    pub kind: ExprKind,
    pub ty: ValueType,
    /// The set of the free variables in the expression.
    pub vars: VarSet,
    internal_hash: u64,
}

/// Makes a pattern that matches an [`Expr`] of kind [`ExprKind::Binary`].
#[macro_export]
macro_rules! binary {
    ($($op:pat)|*, $x:pat, $y:pat) => {
        $crate::ast::Expr {
            kind: $crate::ast::ExprKind::Binary($($op)|*, box $x, box $y),
            ..
        }
    };
}

/// Makes a pattern that matches an [`Expr`] of kind [`ExprKind::Constant`].
#[macro_export]
macro_rules! constant {
    ($a:pat) => {
        $crate::ast::Expr {
            kind: $crate::ast::ExprKind::Constant(box $a),
            ..
        }
    };
}

/// Makes a pattern that matches an [`Expr`] of kind [`ExprKind::Nary`].
#[macro_export]
macro_rules! nary {
    ($($op:pat)|*, $xs:pat) => {
        $crate::ast::Expr {
            kind: $crate::ast::ExprKind::Nary($($op)|*, $xs),
            ..
        }
    };
}

/// Makes a pattern that matches an [`Expr`] of kind [`ExprKind::Pown`].
#[macro_export]
macro_rules! pown {
    ($x:pat, $n:pat) => {
        $crate::ast::Expr {
            kind: $crate::ast::ExprKind::Pown(box $x, $n),
            ..
        }
    };
}

/// Makes a pattern that matches an [`Expr`] of kind [`ExprKind::Rootn`].
#[macro_export]
macro_rules! rootn {
    ($x:pat, $n:pat) => {
        $crate::ast::Expr {
            kind: $crate::ast::ExprKind::Rootn(box $x, $n),
            ..
        }
    };
}

/// Makes a pattern that matches an [`Expr`] of kind [`ExprKind::Ternary`].
#[macro_export]
macro_rules! ternary {
    ($($op:pat)|*, $x:pat, $y:pat, $z:pat) => {
        $crate::ast::Expr {
            kind: $crate::ast::ExprKind::Ternary($($op)|*, box $x, box $y, box $z),
            ..
        }
    };
}

/// Makes a pattern that matches an [`Expr`] of kind [`ExprKind::Unary`].
#[macro_export]
macro_rules! unary {
    ($($op:pat)|*, $x:pat) => {
        $crate::ast::Expr {
            kind: $crate::ast::ExprKind::Unary($($op)|*, box $x),
            ..
        }
    };
}

/// Makes a pattern that matches an [`Expr`] of kind [`ExprKind::Uninit`].
#[macro_export]
macro_rules! uninit {
    () => {
        $crate::ast::Expr {
            kind: $crate::ast::ExprKind::Uninit,
            ..
        }
    };
}

/// Makes a pattern that matches an [`Expr`] of kind [`ExprKind::Var`].
#[macro_export]
macro_rules! var {
    ($name:pat) => {
        $crate::ast::Expr {
            kind: $crate::ast::ExprKind::Var($name),
            ..
        }
    };
}

impl Expr {
    /// Creates a new expression.
    pub fn new(kind: ExprKind) -> Self {
        Self {
            id: UNINIT_EXPR_ID,
            kind,
            ty: ValueType::Unknown,
            vars: VarSet::EMPTY,
            internal_hash: 0,
        }
    }

    /// Creates a new expression of kind [`ExprKind::Binary`].
    pub fn binary(op: BinaryOp, x: Box<Expr>, y: Box<Expr>) -> Self {
        Self::new(ExprKind::Binary(op, x, y))
    }

    /// Creates a new expression of kind [`ExprKind::Constant`].
    pub fn constant(x: TupperIntervalSet, xr: Option<Rational>) -> Self {
        Self::new(ExprKind::Constant(box (x, xr)))
    }

    /// Creates a constant node with value -1.
    pub fn minus_one() -> Self {
        Self::constant(const_dec_interval!(-1.0, -1.0).into(), Some((-1).into()))
    }

    /// Creates a new expression of kind [`ExprKind::Nary`].
    pub fn nary(op: NaryOp, xs: Vec<Expr>) -> Self {
        Self::new(ExprKind::Nary(op, xs))
    }

    /// Creates a constant node with value 1.
    pub fn one() -> Self {
        Self::constant(const_dec_interval!(1.0, 1.0).into(), Some((1).into()))
    }

    /// Creates a constant node with value 1/2.
    pub fn one_half() -> Self {
        Self::constant(const_dec_interval!(0.5, 0.5).into(), Some((1, 2).into()))
    }

    /// Creates a new expression of kind [`ExprKind::Pown`].
    pub fn pown(x: Box<Expr>, n: i32) -> Self {
        Self::new(ExprKind::Pown(x, n))
    }

    /// Creates a new expression of kind [`ExprKind::Rootn`].
    pub fn rootn(x: Box<Expr>, n: u32) -> Self {
        Self::new(ExprKind::Rootn(x, n))
    }

    /// Creates a new expression of kind [`ExprKind::Ternary`].
    pub fn ternary(op: TernaryOp, x: Box<Expr>, y: Box<Expr>, z: Box<Expr>) -> Self {
        Self::new(ExprKind::Ternary(op, x, y, z))
    }

    /// Creates a constant node with value 2.
    pub fn two() -> Self {
        Self::constant(const_dec_interval!(2.0, 2.0).into(), Some((2).into()))
    }

    /// Creates a new expression of kind [`ExprKind::Unary`].
    pub fn unary(op: UnaryOp, x: Box<Expr>) -> Self {
        Self::new(ExprKind::Unary(op, x))
    }

    /// Creates a new expression of kind [`ExprKind::Var`].
    pub fn var(name: &str) -> Self {
        Self::new(ExprKind::Var(name.into()))
    }

    /// Creates a constant node with value 0.
    pub fn zero() -> Self {
        Self::constant(const_dec_interval!(0.0, 0.0).into(), Some(0.into()))
    }

    pub fn dump_structure(&self) -> impl fmt::Display + '_ {
        DumpStructure(self)
    }

    /// Evaluates the expression.
    ///
    /// Returns [`None`] if the expression cannot be evaluated to a scalar constant
    /// or constant evaluation is not implemented for the operation.
    pub fn eval(&self) -> Option<(TupperIntervalSet, Option<Rational>)> {
        use {BinaryOp::*, NaryOp::*, TernaryOp::*, UnaryOp::*};
        match self {
            constant!(x) => Some(x.clone()),
            var!(_) => None,
            unary!(Abs, x) => x.eval1r(|x| x.abs(), |x| Some(x.abs())),
            unary!(Acos, x) => x.eval1(|x| x.acos()),
            unary!(Acosh, x) => x.eval1(|x| x.acosh()),
            unary!(AiryAi, x) => x.eval1(|x| x.airy_ai()),
            unary!(AiryAiPrime, x) => x.eval1(|x| x.airy_ai_prime()),
            unary!(AiryBi, x) => x.eval1(|x| x.airy_bi()),
            unary!(AiryBiPrime, x) => x.eval1(|x| x.airy_bi_prime()),
            unary!(Asin, x) => x.eval1(|x| x.asin()),
            unary!(Asinh, x) => x.eval1(|x| x.asinh()),
            unary!(Atan, x) => x.eval1(|x| x.atan()),
            unary!(Atanh, x) => x.eval1(|x| x.atanh()),
            unary!(Ceil, x) => x.eval1r(|x| x.ceil(None), |x| Some(x.ceil())),
            unary!(Chi, x) => x.eval1(|x| x.chi()),
            unary!(Ci, x) => x.eval1(|x| x.ci()),
            unary!(Cos, x) => x.eval1(|x| x.cos()),
            unary!(Cosh, x) => x.eval1(|x| x.cosh()),
            unary!(Digamma, x) => x.eval1(|x| x.digamma(None)),
            unary!(Ei, x) => x.eval1(|x| x.ei()),
            unary!(EllipticE, x) => x.eval1(|x| x.elliptic_e()),
            unary!(EllipticK, x) => x.eval1(|x| x.elliptic_k()),
            unary!(Erf, x) => x.eval1(|x| x.erf()),
            unary!(Erfc, x) => x.eval1(|x| x.erfc()),
            unary!(Erfi, x) => x.eval1(|x| x.erfi()),
            unary!(Exp, x) => x.eval1(|x| x.exp()),
            unary!(Floor, x) => x.eval1r(|x| x.floor(None), |x| Some(x.floor())),
            unary!(FresnelC, x) => x.eval1(|x| x.fresnel_c()),
            unary!(FresnelS, x) => x.eval1(|x| x.fresnel_s()),
            unary!(Gamma, x) => x.eval1(|x| x.gamma(None)),
            unary!(Li, x) => x.eval1(|x| x.li()),
            unary!(Ln, x) => x.eval1(|x| x.ln()),
            unary!(Log10, x) => x.eval1(|x| x.log10()),
            unary!(Neg, x) => x.eval1r(|x| -&x, |x| Some(-x)),
            unary!(One, x) => x.eval1(|x| x.one()),
            unary!(Shi, x) => x.eval1(|x| x.shi()),
            unary!(Si, x) => x.eval1(|x| x.si()),
            unary!(Sin, x) => x.eval1(|x| x.sin()),
            unary!(Sinc, x) => x.eval1(|x| x.sinc()),
            unary!(Sinh, x) => x.eval1(|x| x.sinh()),
            unary!(Sqr, x) => x.eval1r(|x| x.sqr(), |x| Some(x.square())),
            unary!(Sqrt, x) => x.eval1(|x| x.sqrt()),
            unary!(Tan, x) => x.eval1(|x| x.tan(None)),
            unary!(Tanh, x) => x.eval1(|x| x.tanh()),
            unary!(UndefAt0, x) => x.eval1(|x| x.undef_at_0()),
            binary!(Add, x, y) => x.eval2r(y, |x, y| &x + &y, |x, y| Some(x + y)),
            binary!(Atan2, y, x) => y.eval2(x, |y, x| y.atan2(&x, None)),
            binary!(BesselI, n, x) => n.eval2(x, |n, x| n.bessel_i(&x)),
            binary!(BesselJ, n, x) => n.eval2(x, |n, x| n.bessel_j(&x)),
            binary!(BesselK, n, x) => n.eval2(x, |n, x| n.bessel_k(&x)),
            binary!(BesselY, n, x) => n.eval2(x, |n, x| n.bessel_y(&x)),
            binary!(Div, x, y) => x.eval2r(y, |x, y| x.div(&y, None), rational_ops::div),
            binary!(GammaInc, a, x) => a.eval2(x, |a, x| a.gamma_inc(&x)),
            binary!(Gcd, x, y) => x.eval2r(y, |x, y| x.gcd(&y, None), rational_ops::gcd),
            binary!(Lcm, x, y) => x.eval2r(y, |x, y| x.lcm(&y, None), rational_ops::lcm),
            // Beware the order of arguments.
            binary!(Log, b, x) => b.eval2(x, |b, x| x.log(&b, None)),
            binary!(Max, x, y) => x.eval2r(y, |x, y| x.max(&y), rational_ops::max),
            binary!(Min, x, y) => x.eval2r(y, |x, y| x.min(&y), rational_ops::min),
            binary!(Mod, x, y) => {
                x.eval2r(y, |x, y| x.rem_euclid(&y, None), rational_ops::rem_euclid)
            }
            binary!(Mul, x, y) => x.eval2r(y, |x, y| &x * &y, |x, y| Some(x * y)),
            binary!(Pow, x, y) => x.eval2r(y, |x, y| x.pow(&y, None), rational_ops::pow),
            binary!(RankedMax, xs, n) => Some((
                if let nary!(List, xs) = xs {
                    let xs = xs.iter().map(|x| x.eval()).collect::<Option<Vec<_>>>()?;
                    TupperIntervalSet::ranked_max(
                        xs.iter().map(|x| &x.0).collect(),
                        &n.eval()?.0,
                        None,
                    )
                } else {
                    panic!("a list is expected")
                },
                None,
            )),
            binary!(RankedMin, xs, n) => Some((
                if let nary!(List, xs) = xs {
                    let xs = xs.iter().map(|x| x.eval()).collect::<Option<Vec<_>>>()?;
                    TupperIntervalSet::ranked_min(
                        xs.iter().map(|x| &x.0).collect(),
                        &n.eval()?.0,
                        None,
                    )
                } else {
                    panic!("a list is expected")
                },
                None,
            )),
            binary!(Sub, x, y) => x.eval2r(y, |x, y| &x - &y, |x, y| Some(x - y)),
            ternary!(MulAdd, _, _, _) => None,
            nary!(Plus | Times, _) => None,
            rootn!(x, n) => x.eval1(|x| x.rootn(*n)),
            unary!(Exp10 | Exp2 | Recip, _) | pown!(_, _) => {
                panic!("use `BinaryOp::Pow` for constant evaluation")
            }
            unary!(Not, _) => None,
            binary!(
                And | Eq | Ge | Gt | Le | Lt | Neq | Nge | Ngt | Nle | Nlt | Or,
                _,
                _
            ) => None,
            nary!(List, _) => None,
            uninit!() => panic!(),
        }
    }

    /// Updates [`Expr::ty`], [`Expr::vars`] and [`Expr::internal_hash`] of the expression.
    ///
    /// Precondition:
    ///   The function is called on all sub-expressions and they have not been changed since then.
    pub fn update_metadata(&mut self) {
        self.ty = self.value_type();
        self.vars = match self {
            constant!(_) => VarSet::EMPTY,
            var!(x) if x == "x" => VarSet::X,
            var!(x) if x == "y" => VarSet::Y,
            var!(x) if x == "<n-theta>" => VarSet::N_THETA,
            var!(_) => VarSet::EMPTY,
            unary!(_, x) | pown!(x, _) | rootn!(x, _) => x.vars,
            binary!(_, x, y) => x.vars | y.vars,
            ternary!(_, x, y, z) => x.vars | y.vars | z.vars,
            nary!(_, xs) => xs.iter().fold(VarSet::EMPTY, |vs, x| vs | x.vars),
            uninit!() => panic!(),
        };
        self.internal_hash = {
            // Use `DefaultHasher::new` so that the value of `internal_hash` will be deterministic.
            let mut hasher = DefaultHasher::new();
            self.kind.hash(&mut hasher);
            hasher.finish()
        }
    }

    pub fn value_type(&self) -> ValueType {
        use {BinaryOp::*, NaryOp::*, TernaryOp::*, UnaryOp::*, ValueType::*};
        match self {
            constant!(_) => Scalar,
            var!(x) if x == "x" || x == "y" || x == "<n-theta>" => Scalar,
            unary!(
                Abs | Acos
                    | Acosh
                    | AiryAi
                    | AiryAiPrime
                    | AiryBi
                    | AiryBiPrime
                    | Asin
                    | Asinh
                    | Atan
                    | Atanh
                    | Ceil
                    | Chi
                    | Ci
                    | Cos
                    | Cosh
                    | Digamma
                    | Ei
                    | EllipticE
                    | EllipticK
                    | Erf
                    | Erfc
                    | Erfi
                    | Exp
                    | Exp10
                    | Exp2
                    | Floor
                    | FresnelC
                    | FresnelS
                    | Gamma
                    | Li
                    | Ln
                    | Log10
                    | Neg
                    | One
                    | Recip
                    | Shi
                    | Si
                    | Sin
                    | Sinc
                    | Sinh
                    | Sqr
                    | Sqrt
                    | Tan
                    | Tanh
                    | UndefAt0,
                x
            ) if x.ty == Scalar => Scalar,
            binary!(
                Add | Atan2
                    | BesselI
                    | BesselJ
                    | BesselK
                    | BesselY
                    | Div
                    | GammaInc
                    | Gcd
                    | Lcm
                    | Log
                    | Max
                    | Min
                    | Mod
                    | Mul
                    | Pow
                    | Sub,
                x,
                y
            ) if x.ty == Scalar && y.ty == Scalar => Scalar,
            ternary!(MulAdd, x, y, z) if x.ty == Scalar && y.ty == Scalar && z.ty == Scalar => {
                Scalar
            }
            binary!(RankedMax | RankedMin, x, y) if x.ty == Vector && y.ty == Scalar => Scalar,
            pown!(x, _) | rootn!(x, _) if x.ty == Scalar => Scalar,
            nary!(List, xs) if xs.iter().all(|x| x.ty == Scalar) => Vector,
            unary!(Not, x) if x.ty == Boolean => Boolean,
            binary!(And | Or, x, y) if x.ty == Boolean && y.ty == Boolean => Boolean,
            binary!(Eq | Ge | Gt | Le | Lt | Neq | Nge | Ngt | Nle | Nlt, x, y)
                if x.ty == Scalar && y.ty == Scalar =>
            {
                Boolean
            }
            uninit!() => panic!(),
            _ => Unknown,
        }
    }

    fn eval1<F>(&self, f: F) -> Option<(TupperIntervalSet, Option<Rational>)>
    where
        F: Fn(TupperIntervalSet) -> TupperIntervalSet,
    {
        let (x, _) = self.eval()?;
        let y = f(x);
        let yr = y.to_f64().and_then(Rational::from_f64);
        Some((y, yr))
    }

    fn eval1r<F, FR>(&self, f: F, fr: FR) -> Option<(TupperIntervalSet, Option<Rational>)>
    where
        F: Fn(TupperIntervalSet) -> TupperIntervalSet,
        FR: Fn(Rational) -> Option<Rational>,
    {
        let (x, xr) = self.eval()?;
        let yr = xr.and_then(fr);
        let y = if let Some(yr) = &yr {
            TupperIntervalSet::from(DecInterval::new(rational_ops::to_interval(yr)))
        } else {
            f(x)
        };
        Some((y, yr))
    }

    fn eval2<F>(&self, y: &Self, f: F) -> Option<(TupperIntervalSet, Option<Rational>)>
    where
        F: Fn(TupperIntervalSet, TupperIntervalSet) -> TupperIntervalSet,
    {
        let (x, _) = self.eval()?;
        let (y, _) = y.eval()?;
        let z = f(x, y);
        let zr = z.to_f64().and_then(Rational::from_f64);
        Some((z, zr))
    }

    fn eval2r<F, FR>(&self, y: &Self, f: F, fr: FR) -> Option<(TupperIntervalSet, Option<Rational>)>
    where
        F: Fn(TupperIntervalSet, TupperIntervalSet) -> TupperIntervalSet,
        FR: Fn(Rational, Rational) -> Option<Rational>,
    {
        let (x, xr) = self.eval()?;
        let (y, yr) = y.eval()?;
        let zr = xr.zip(yr).and_then(|(xr, yr)| fr(xr, yr));
        let z = if let Some(zr) = &zr {
            TupperIntervalSet::from(DecInterval::new(rational_ops::to_interval(zr)))
        } else {
            f(x, y)
        };
        Some((z, zr))
    }
}

impl Default for Expr {
    fn default() -> Self {
        Self {
            id: UNINIT_EXPR_ID,
            kind: ExprKind::Uninit,
            ty: ValueType::Unknown,
            vars: VarSet::EMPTY,
            internal_hash: 0,
        }
    }
}

impl PartialEq for Expr {
    fn eq(&self, rhs: &Self) -> bool {
        self.kind == rhs.kind
    }
}

impl Eq for Expr {}

impl Hash for Expr {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.internal_hash.hash(state);
    }
}

struct DumpStructure<'a>(&'a Expr);

impl<'a> fmt::Display for DumpStructure<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self.0 {
            constant!(a) => {
                if let Some(a) = a.0.to_f64() {
                    write!(f, "{}", a)
                } else {
                    write!(f, "@")
                }
            }
            var!(name) => write!(f, "{}", name),
            unary!(op, x) => write!(f, "({:?} {})", op, x.dump_structure()),
            binary!(op, x, y) => write!(
                f,
                "({:?} {} {})",
                op,
                x.dump_structure(),
                y.dump_structure()
            ),
            ternary!(op, x, y, z) => write!(
                f,
                "({:?} {} {} {})",
                op,
                x.dump_structure(),
                y.dump_structure(),
                z.dump_structure()
            ),
            nary!(op, xs) => {
                write!(
                    f,
                    "({:?} {})",
                    op,
                    xs.iter()
                        .map(|x| format!("{}", x.dump_structure()))
                        .collect::<Vec<_>>()
                        .join(" ")
                )
            }
            pown!(x, n) => write!(f, "(Pown {} {})", x.dump_structure(), n),
            rootn!(x, n) => write!(f, "(Rootn {} {})", x.dump_structure(), n),
            uninit!() => panic!(),
        }
    }
}
