use crate::{interval_set::TupperIntervalSet, rational_ops};
use bitflags::*;
use inari::{const_dec_interval, DecInterval};
use rug::{Integer, Rational};
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
pub enum NaryOp {
    Plus,
    Times,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub enum ExprKind {
    // == Scalar-valued expressions ==
    Constant(Box<(TupperIntervalSet, Option<Rational>)>),
    Unary(UnaryOp, Box<Expr>),
    Binary(BinaryOp, Box<Expr>, Box<Expr>),
    Nary(NaryOp, Vec<Expr>),
    Pown(Box<Expr>, i32),
    Rootn(Box<Expr>, u32),
    // == Others ==
    Var(String),
    List(Vec<Expr>),
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
    /// The period of a function of θ in multiples of 2π, i.e., any p that satisfies
    /// (e /. θ → θ + 2π p) = e. If the period is 0, the expression is independent of θ.
    pub polar_period: Option<Integer>,
    /// The set of the free variables in the expression.
    pub vars: VarSet,
    internal_hash: u64,
}

impl Expr {
    /// Creates a new expression.
    pub fn new(kind: ExprKind) -> Self {
        Self {
            id: UNINIT_EXPR_ID,
            kind,
            ty: ValueType::Unknown,
            polar_period: None,
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
    /// Returns [`None`] if the expression cannot be evaluated to a scalar constant.
    pub fn eval(&self) -> Option<(TupperIntervalSet, Option<Rational>)> {
        use {BinaryOp::*, ExprKind::*, UnaryOp::*};
        match &self.kind {
            Constant(x) => Some(*x.clone()),
            Unary(Abs, x) => x.eval1r(|x| x.abs(), |x| Some(x.abs())),
            Unary(Acos, x) => x.eval1(|x| x.acos()),
            Unary(Acosh, x) => x.eval1(|x| x.acosh()),
            Unary(AiryAi, x) => x.eval1(|x| x.airy_ai()),
            Unary(AiryAiPrime, x) => x.eval1(|x| x.airy_ai_prime()),
            Unary(AiryBi, x) => x.eval1(|x| x.airy_bi()),
            Unary(AiryBiPrime, x) => x.eval1(|x| x.airy_bi_prime()),
            Unary(Asin, x) => x.eval1(|x| x.asin()),
            Unary(Asinh, x) => x.eval1(|x| x.asinh()),
            Unary(Atan, x) => x.eval1(|x| x.atan()),
            Unary(Atanh, x) => x.eval1(|x| x.atanh()),
            Unary(Ceil, x) => x.eval1r(|x| x.ceil(None), |x| Some(x.ceil())),
            Unary(Chi, x) => x.eval1(|x| x.chi()),
            Unary(Ci, x) => x.eval1(|x| x.ci()),
            Unary(Cos, x) => x.eval1(|x| x.cos()),
            Unary(Cosh, x) => x.eval1(|x| x.cosh()),
            Unary(Digamma, x) => x.eval1(|x| x.digamma(None)),
            Unary(Ei, x) => x.eval1(|x| x.ei()),
            Unary(EllipticE, x) => x.eval1(|x| x.elliptic_e()),
            Unary(EllipticK, x) => x.eval1(|x| x.elliptic_k()),
            Unary(Erf, x) => x.eval1(|x| x.erf()),
            Unary(Erfc, x) => x.eval1(|x| x.erfc()),
            Unary(Erfi, x) => x.eval1(|x| x.erfi()),
            Unary(Exp, x) => x.eval1(|x| x.exp()),
            Unary(Floor, x) => x.eval1r(|x| x.floor(None), |x| Some(x.floor())),
            Unary(FresnelC, x) => x.eval1(|x| x.fresnel_c()),
            Unary(FresnelS, x) => x.eval1(|x| x.fresnel_s()),
            Unary(Gamma, x) => x.eval1(|x| x.gamma(None)),
            Unary(Li, x) => x.eval1(|x| x.li()),
            Unary(Ln, x) => x.eval1(|x| x.ln()),
            Unary(Log10, x) => x.eval1(|x| x.log10()),
            Unary(Neg, x) => x.eval1r(|x| -&x, |x| Some(-x)),
            Unary(One, x) => x.eval1(|x| x.one()),
            Unary(Shi, x) => x.eval1(|x| x.shi()),
            Unary(Si, x) => x.eval1(|x| x.si()),
            Unary(Sin, x) => x.eval1(|x| x.sin()),
            Unary(Sinc, x) => x.eval1(|x| x.sinc()),
            Unary(Sinh, x) => x.eval1(|x| x.sinh()),
            Unary(Sqr, x) => x.eval1r(|x| x.sqr(), |x| Some(x.square())),
            Unary(Sqrt, x) => x.eval1(|x| x.sqrt()),
            Unary(Tan, x) => x.eval1(|x| x.tan(None)),
            Unary(Tanh, x) => x.eval1(|x| x.tanh()),
            Unary(UndefAt0, x) => x.eval1(|x| x.undef_at_0()),
            Binary(Add, x, y) => x.eval2r(y, |x, y| &x + &y, |x, y| Some(x + y)),
            Binary(Atan2, y, x) => y.eval2(x, |y, x| y.atan2(&x, None)),
            Binary(BesselI, n, x) => n.eval2(x, |n, x| n.bessel_i(&x)),
            Binary(BesselJ, n, x) => n.eval2(x, |n, x| n.bessel_j(&x)),
            Binary(BesselK, n, x) => n.eval2(x, |n, x| n.bessel_k(&x)),
            Binary(BesselY, n, x) => n.eval2(x, |n, x| n.bessel_y(&x)),
            Binary(Div, x, y) => x.eval2r(y, |x, y| x.div(&y, None), rational_ops::div),
            Binary(GammaInc, a, x) => a.eval2(x, |a, x| a.gamma_inc(&x)),
            Binary(Gcd, x, y) => x.eval2r(y, |x, y| x.gcd(&y, None), rational_ops::gcd),
            Binary(Lcm, x, y) => x.eval2r(y, |x, y| x.lcm(&y, None), rational_ops::lcm),
            // Beware the order of arguments.
            Binary(Log, b, x) => b.eval2(x, |b, x| x.log(&b, None)),
            Binary(Max, x, y) => x.eval2r(y, |x, y| x.max(&y), rational_ops::max),
            Binary(Min, x, y) => x.eval2r(y, |x, y| x.min(&y), rational_ops::min),
            Binary(Mod, x, y) => {
                x.eval2r(y, |x, y| x.rem_euclid(&y, None), rational_ops::rem_euclid)
            }
            Binary(Mul, x, y) => x.eval2r(y, |x, y| &x * &y, |x, y| Some(x * y)),
            Binary(Pow, x, y) => x.eval2r(y, |x, y| x.pow(&y, None), rational_ops::pow),
            Binary(RankedMax, xs, n) => Some((
                if let List(xs) = &xs.kind {
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
            Binary(RankedMin, xs, n) => Some((
                if let List(xs) = &xs.kind {
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
            Binary(Sub, x, y) => x.eval2r(y, |x, y| &x - &y, |x, y| Some(x - y)),
            Rootn(x, n) => x.eval1(|x| x.rootn(*n)),
            Unary(Exp10 | Exp2 | Recip, _) | Pown(_, _) => {
                panic!("Pow should be used instead")
            }
            Uninit => panic!(),
            _ => None,
        }
    }

    /// Updates [`Expr::ty`], [`Expr::vars`] and [`Expr::internal_hash`] of the expression.
    ///
    /// Precondition:
    ///   The function is called on all sub-expressions and they have not been changed since then.
    pub fn update_metadata(&mut self) {
        use ExprKind::*;
        self.ty = self.value_type();
        self.vars = match &self.kind {
            Constant(_) => VarSet::EMPTY,
            Var(x) if x == "x" => VarSet::X,
            Var(x) if x == "y" => VarSet::Y,
            Var(x) if x == "<n-theta>" => VarSet::N_THETA,
            Var(_) => VarSet::EMPTY,
            Unary(_, x) | Pown(x, _) | Rootn(x, _) => x.vars,
            Binary(_, x, y) => x.vars | y.vars,
            List(xs) => xs.iter().fold(VarSet::EMPTY, |vs, x| vs | x.vars),
            Nary(_, _) | Uninit => panic!(),
        };
        self.internal_hash = {
            // Use `DefaultHasher::new` so that the value of `internal_hash` will be deterministic.
            let mut hasher = DefaultHasher::new();
            self.kind.hash(&mut hasher);
            hasher.finish()
        }
    }

    pub fn value_type(&self) -> ValueType {
        use {BinaryOp::*, ExprKind::*, UnaryOp::*, ValueType::*};
        match &self.kind {
            Constant(_) => Scalar,
            Unary(
                Abs | Acos | Acosh | AiryAi | AiryAiPrime | AiryBi | AiryBiPrime | Asin | Asinh
                | Atan | Atanh | Ceil | Chi | Ci | Cos | Cosh | Digamma | Ei | EllipticE
                | EllipticK | Erf | Erfc | Erfi | Exp | Exp10 | Exp2 | Floor | FresnelC | FresnelS
                | Gamma | Li | Ln | Log10 | Neg | One | Recip | Shi | Si | Sin | Sinc | Sinh | Sqr
                | Sqrt | Tan | Tanh | UndefAt0,
                x,
            ) if x.ty == Scalar => Scalar,
            Binary(
                Add | Atan2 | BesselI | BesselJ | BesselK | BesselY | Div | GammaInc | Gcd | Lcm
                | Log | Max | Min | Mod | Mul | Pow | Sub,
                x,
                y,
            ) if x.ty == Scalar && y.ty == Scalar => Scalar,
            Binary(RankedMax | RankedMin, x, y) if x.ty == Vector && y.ty == Scalar => Scalar,
            Pown(x, _) | Rootn(x, _) if x.ty == Scalar => Scalar,
            List(xs) if xs.iter().all(|x| x.ty == Scalar) => Vector,
            Unary(Not, x) if x.ty == Boolean => Boolean,
            Binary(And | Or, x, y) if x.ty == Boolean && y.ty == Boolean => Boolean,
            Binary(Eq | Ge | Gt | Le | Lt | Neq | Nge | Ngt | Nle | Nlt, x, y)
                if x.ty == Scalar && y.ty == Scalar =>
            {
                Boolean
            }
            Var(x) if x == "x" || x == "y" || x == "<n-theta>" => Scalar,
            Uninit => panic!(),
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
            polar_period: None,
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
        match &self.0.kind {
            ExprKind::Constant(a) => {
                if let Some(a) = a.0.to_f64() {
                    write!(f, "{}", a)
                } else {
                    write!(f, "@")
                }
            }
            ExprKind::Var(name) => write!(f, "{}", name),
            ExprKind::Unary(op, x) => write!(f, "({:?} {})", op, x.dump_structure()),
            ExprKind::Binary(op, x, y) => write!(
                f,
                "({:?} {} {})",
                op,
                x.dump_structure(),
                y.dump_structure()
            ),
            ExprKind::Nary(op, xs) => {
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
            ExprKind::Pown(x, n) => write!(f, "(Pown {} {})", x.dump_structure(), n),
            ExprKind::Rootn(x, n) => write!(f, "(Rootn {} {})", x.dump_structure(), n),
            ExprKind::List(xs) => {
                write!(
                    f,
                    "(List {})",
                    xs.iter()
                        .map(|x| format!("{}", x.dump_structure()))
                        .collect::<Vec<_>>()
                        .join(" ")
                )
            }
            ExprKind::Uninit => panic!(),
        }
    }
}
