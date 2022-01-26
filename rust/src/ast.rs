use crate::{real::Real, vars::VarSet};
use inari::{const_dec_interval, DecInterval, Decoration};
use std::{
    collections::hash_map::DefaultHasher,
    fmt,
    hash::{Hash, Hasher},
    ops::Range,
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
    Arg,
    Asin,
    Asinh,
    Atan,
    Atanh,
    Boole, // Iverson bracket
    BooleEqZero,
    BooleLeZero,
    BooleLtZero,
    Ceil,
    Chi,
    Ci,
    Conj,
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
    Floor,
    FresnelC,
    FresnelS,
    Gamma,
    Im,
    Li,
    Ln,
    Neg,
    Not,
    Re,
    Recip,
    Shi,
    Si,
    Sign,
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
    Complex,
    Div,
    Eq,
    ExplicitRel,
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
    Or,
    Pow,
    PowRational,
    RankedMax,
    RankedMin,
    ReSignNonnegative,
    Sub,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum TernaryOp {
    IfThenElse,
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
    BoolConstant(bool),
    Constant(Box<Real>),
    Var(String),
    Unary(UnaryOp, Box<Expr>),
    Binary(BinaryOp, Box<Expr>, Box<Expr>),
    Ternary(TernaryOp, Box<Expr>, Box<Expr>, Box<Expr>),
    Nary(NaryOp, Vec<Expr>),
    Pown(Box<Expr>, i32),
    Rootn(Box<Expr>, u32),
    Error,
    Uninit,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ValueType {
    Boolean,
    Complex,
    Real,
    RealVector,
    Unknown,
}

impl fmt::Display for ValueType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use ValueType::*;
        match self {
            Boolean => write!(f, "boolean"),
            Complex => write!(f, "complex"),
            Real => write!(f, "real"),
            RealVector => write!(f, "real vector"),
            Unknown => write!(f, "unknown"),
        }
    }
}

/// An AST node for an expression.
#[derive(Clone, Debug)]
pub struct Expr {
    pub id: ExprId,
    pub kind: ExprKind,
    pub source_range: Range<usize>,
    pub totally_defined: bool,
    pub ty: ValueType,
    pub vars: VarSet,
    internal_hash: u64,
}

/// Makes a pattern that matches an [`Expr`] of kind [`ExprKind::Binary`].
#[macro_export]
macro_rules! binary {
    ($($op:pat_param)|*, $x:pat, $y:pat) => {
        $crate::ast::Expr {
            kind: $crate::ast::ExprKind::Binary($($op)|*, box $x, box $y),
            ..
        }
    };
}

/// Makes a pattern that matches an [`Expr`] of kind [`ExprKind::BoolConstant`].
#[macro_export]
macro_rules! bool_constant {
    ($a:pat) => {
        $crate::ast::Expr {
            kind: $crate::ast::ExprKind::BoolConstant($a),
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

/// Makes a pattern that matches an [`Expr`] of kind [`ExprKind::Error`].
#[macro_export]
macro_rules! error {
    () => {
        $crate::ast::Expr {
            kind: $crate::ast::ExprKind::Error,
            ..
        }
    };
}

/// Makes a pattern that matches an [`Expr`] of kind [`ExprKind::Nary`].
#[macro_export]
macro_rules! nary {
    ($($op:pat_param)|*, $xs:pat) => {
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
    ($($op:pat_param)|*, $x:pat, $y:pat, $z:pat) => {
        $crate::ast::Expr {
            kind: $crate::ast::ExprKind::Ternary($($op)|*, box $x, box $y, box $z),
            ..
        }
    };
}

/// Makes a pattern that matches an [`Expr`] of kind [`ExprKind::Unary`].
#[macro_export]
macro_rules! unary {
    ($($op:pat_param)|*, $x:pat) => {
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
            source_range: 0..0,
            totally_defined: false,
            ty: ValueType::Unknown,
            vars: VarSet::EMPTY,
            internal_hash: 0,
        }
    }

    /// Creates a new expression of kind [`ExprKind::Binary`].
    pub fn binary(op: BinaryOp, x: Box<Expr>, y: Box<Expr>) -> Self {
        Self::new(ExprKind::Binary(op, x, y))
    }

    /// Creates a new expression of kind [`ExprKind::BoolConstant`].
    pub fn bool_constant(x: bool) -> Self {
        Self::new(ExprKind::BoolConstant(x))
    }

    /// Creates a new expression of kind [`ExprKind::Constant`].
    pub fn constant(x: Real) -> Self {
        Self::new(ExprKind::Constant(box x))
    }

    /// Creates a new expression of kind [`ExprKind::Error`].
    pub fn error() -> Self {
        Self::new(ExprKind::Error)
    }

    /// Creates a constant node with value -1.
    pub fn minus_one() -> Self {
        Self::constant(const_dec_interval!(-1.0, -1.0).into())
    }

    /// Creates a new expression of kind [`ExprKind::Nary`].
    pub fn nary(op: NaryOp, xs: Vec<Expr>) -> Self {
        Self::new(ExprKind::Nary(op, xs))
    }

    /// Creates a constant node with value 1.
    pub fn one() -> Self {
        Self::constant(const_dec_interval!(1.0, 1.0).into())
    }

    /// Creates a constant node with value 1/2.
    pub fn one_half() -> Self {
        Self::constant(const_dec_interval!(0.5, 0.5).into())
    }

    /// Creates a new expression of kind [`ExprKind::Pown`].
    pub fn pown(x: Box<Expr>, n: i32) -> Self {
        Self::new(ExprKind::Pown(x, n))
    }

    /// Creates a new expression of kind [`ExprKind::Rootn`].
    pub fn rootn(x: Box<Expr>, n: u32) -> Self {
        Self::new(ExprKind::Rootn(x, n))
    }

    /// Creates a constant node with value 2π.
    pub fn tau() -> Self {
        Self::constant(DecInterval::TAU.into())
    }

    /// Creates a new expression of kind [`ExprKind::Ternary`].
    pub fn ternary(op: TernaryOp, x: Box<Expr>, y: Box<Expr>, z: Box<Expr>) -> Self {
        Self::new(ExprKind::Ternary(op, x, y, z))
    }

    /// Creates a constant node with value 2.
    pub fn two() -> Self {
        Self::constant(const_dec_interval!(2.0, 2.0).into())
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
        Self::constant(const_dec_interval!(0.0, 0.0).into())
    }

    /// Formats the AST in a fashion similar to S-expressions.
    ///
    /// If a real constant is convertible to a [`f64`] number with [`TupperIntervalSet::to_f64`],
    /// it is represented as the number; otherwise, by a symbol `@`.
    ///
    /// [`TupperIntervalSet::to_f64`]: crate::interval_set::TupperIntervalSet::to_f64
    pub fn dump_short(&self) -> impl fmt::Display + '_ {
        DumpShort(self)
    }

    /// Evaluates `self` if it is a real-valued constant expression.
    ///
    /// Returns [`None`] if the expression is not both real-valued and constant,
    /// or constant evaluation is not implemented for the operation.
    pub fn eval(&self) -> Option<Real> {
        use {BinaryOp::*, NaryOp::*, TernaryOp::*, UnaryOp::*};
        match self {
            bool_constant!(_) => None,
            constant!(x) => Some(x.clone()),
            var!(_) => None,
            unary!(Abs, x) => Some(x.eval()?.abs()),
            unary!(Acos, x) => Some(x.eval()?.acos()),
            unary!(Acosh, x) => Some(x.eval()?.acosh()),
            unary!(AiryAi, x) => Some(x.eval()?.airy_ai()),
            unary!(AiryAiPrime, x) => Some(x.eval()?.airy_ai_prime()),
            unary!(AiryBi, x) => Some(x.eval()?.airy_bi()),
            unary!(AiryBiPrime, x) => Some(x.eval()?.airy_bi_prime()),
            unary!(Asin, x) => Some(x.eval()?.asin()),
            unary!(Asinh, x) => Some(x.eval()?.asinh()),
            unary!(Atan, x) => Some(x.eval()?.atan()),
            unary!(Atanh, x) => Some(x.eval()?.atanh()),
            unary!(BooleEqZero, x) => Some(x.eval()?.boole_eq_zero()),
            unary!(BooleLeZero, x) => Some(x.eval()?.boole_le_zero()),
            unary!(BooleLtZero, x) => Some(x.eval()?.boole_lt_zero()),
            unary!(Ceil, x) => Some(x.eval()?.ceil()),
            unary!(Chi, x) => Some(x.eval()?.chi()),
            unary!(Ci, x) => Some(x.eval()?.ci()),
            unary!(Cos, x) => Some(x.eval()?.cos()),
            unary!(Cosh, x) => Some(x.eval()?.cosh()),
            unary!(Digamma, x) => Some(x.eval()?.digamma()),
            unary!(Ei, x) => Some(x.eval()?.ei()),
            unary!(EllipticE, x) => Some(x.eval()?.elliptic_e()),
            unary!(EllipticK, x) => Some(x.eval()?.elliptic_k()),
            unary!(Erf, x) => Some(x.eval()?.erf()),
            unary!(Erfc, x) => Some(x.eval()?.erfc()),
            unary!(Erfi, x) => Some(x.eval()?.erfi()),
            unary!(Exp, x) => Some(x.eval()?.exp()),
            unary!(Floor, x) => Some(x.eval()?.floor()),
            unary!(FresnelC, x) => Some(x.eval()?.fresnel_c()),
            unary!(FresnelS, x) => Some(x.eval()?.fresnel_s()),
            unary!(Gamma, x) => Some(x.eval()?.gamma()),
            unary!(Li, x) => Some(x.eval()?.li()),
            unary!(Ln, x) => Some(x.eval()?.ln()),
            unary!(Shi, x) => Some(x.eval()?.shi()),
            unary!(Si, x) => Some(x.eval()?.si()),
            unary!(Sin, x) => Some(x.eval()?.sin()),
            unary!(Sinc, x) => Some(x.eval()?.sinc()),
            unary!(Sinh, x) => Some(x.eval()?.sinh()),
            unary!(Tan, x) => Some(x.eval()?.tan()),
            unary!(Tanh, x) => Some(x.eval()?.tanh()),
            unary!(UndefAt0, x) => Some(x.eval()?.undef_at_0()),
            unary!(
                Arg | Boole | Conj | Im | Neg | Not | Re | Recip | Sign | Sqr | Sqrt,
                _
            ) => None,
            binary!(Add, x, y) => Some(x.eval()? + y.eval()?),
            binary!(Atan2, y, x) => Some(y.eval()?.atan2(x.eval()?)),
            binary!(BesselI, n, x) => Some(n.eval()?.bessel_i(x.eval()?)),
            binary!(BesselJ, n, x) => Some(n.eval()?.bessel_j(x.eval()?)),
            binary!(BesselK, n, x) => Some(n.eval()?.bessel_k(x.eval()?)),
            binary!(BesselY, n, x) => Some(n.eval()?.bessel_y(x.eval()?)),
            binary!(GammaInc, a, x) => Some(a.eval()?.gamma_inc(x.eval()?)),
            binary!(Gcd, x, y) => Some(x.eval()?.gcd(y.eval()?)),
            binary!(Lcm, x, y) => Some(x.eval()?.lcm(y.eval()?)),
            // Beware the order of arguments.
            binary!(Log, b, x) => Some(x.eval()?.log(b.eval()?)),
            binary!(Max, x, y) => Some(x.eval()?.max(y.eval()?)),
            binary!(Min, x, y) => Some(x.eval()?.min(y.eval()?)),
            binary!(Mod, x, y) => Some(x.eval()?.modulo(y.eval()?)),
            binary!(Mul, x, y) => Some(x.eval()? * y.eval()?),
            binary!(Pow, x, y) => Some(x.eval()?.pow(y.eval()?)),
            binary!(PowRational, x, y) => Some(x.eval()?.pow_rational(y.eval()?)),
            binary!(RankedMax, nary!(List, xs), n) => {
                let xs = xs.iter().map(|x| x.eval()).collect::<Option<Vec<_>>>()?;
                Some(Real::ranked_max(xs, n.eval()?))
            }
            binary!(RankedMin, nary!(List, xs), n) => {
                let xs = xs.iter().map(|x| x.eval()).collect::<Option<Vec<_>>>()?;
                Some(Real::ranked_min(xs, n.eval()?))
            }
            binary!(ReSignNonnegative, x, y) => Some(x.eval()?.re_sign_nonnegative(y.eval()?)),
            binary!(
                And | Complex
                    | Div
                    | Eq
                    | ExplicitRel
                    | Ge
                    | Gt
                    | Le
                    | Lt
                    | Or
                    | RankedMax
                    | RankedMin
                    | Sub,
                _,
                _
            ) => None,
            ternary!(IfThenElse, cond, t, f) => {
                Some(cond.eval()?.if_then_else(t.eval()?, f.eval()?))
            }
            ternary!(MulAdd, _, _, _) => None,
            nary!(List | Plus | Times, _) => None,
            pown!(_, _) => None,
            rootn!(_, _) => None,
            error!() => None,
            uninit!() => panic!(),
        }
    }

    /// Updates [`Expr::totally_defined`], [`Expr::ty`], [`Expr::vars`], and [`Expr::internal_hash`]
    /// of the expression.
    ///
    /// Precondition: the function is called on all sub-expressions
    /// and they have not been modified since then.
    pub fn update_metadata(&mut self) {
        self.ty = self.value_type();
        self.totally_defined = self.totally_defined(); // Requires `self.ty`.
        self.vars = self.variables();
        self.internal_hash = {
            // Use `DefaultHasher::new` so that the value of `internal_hash` will be deterministic.
            let mut hasher = DefaultHasher::new();
            self.kind.hash(&mut hasher);
            hasher.finish()
        }
    }

    pub fn with_source_range(mut self, range: Range<usize>) -> Self {
        self.source_range = range;
        self
    }

    /// Returns `true` if the expression is real-valued and is defined on the entire domain.
    ///
    /// Preconditions:
    ///
    /// - [`Expr::totally_defined`] is correctly assigned for all sub-expressions.
    /// - [`Expr::ty`] is correctly assigned for `self`.
    fn totally_defined(&self) -> bool {
        use {BinaryOp::*, NaryOp::*, TernaryOp::*, UnaryOp::*};

        // NOTE: Mathematica's `FunctionDomain` would be useful when the same definition is used.
        match self {
            constant!(a) if a.interval().decoration() >= Decoration::Def => true,
            var!(_) => self.totally_defined,
            unary!(
                Abs | AiryAi
                    | AiryAiPrime
                    | AiryBi
                    | AiryBiPrime
                    | Asinh
                    | Atan
                    | Ceil
                    | Conj
                    | Cos
                    | Cosh
                    | Erf
                    | Erfc
                    | Erfi
                    | Exp
                    | Floor
                    | FresnelC
                    | FresnelS
                    | Im
                    | Neg
                    | Re
                    | Shi
                    | Si
                    | Sign
                    | Sin
                    | Sinc
                    | Sinh
                    | Sqr
                    | Tanh,
                x
            ) => x.totally_defined,
            binary!(Add | Max | Min | Mul | ReSignNonnegative | Sub, x, y) => {
                x.totally_defined && y.totally_defined
            }
            binary!(Pow, x, constant!(y)) => {
                x.totally_defined
                    && matches!(y.rational(), Some(q) if *q >= 0 && q.denom().is_odd())
            }
            ternary!(IfThenElse, _, t, f) => t.totally_defined && f.totally_defined,
            ternary!(MulAdd, x, y, z) => {
                x.totally_defined && y.totally_defined && z.totally_defined
            }
            nary!(Plus | Times, xs) => xs.iter().all(|x| x.totally_defined),
            pown!(x, n) => x.totally_defined && *n >= 0,
            rootn!(x, n) => x.totally_defined && n % 2 == 1,
            uninit!() => panic!(),
            _ => false,
        }
    }

    /// Returns the value type of the expression.
    ///
    /// Precondition: [`Expr::ty`] is correctly assigned for `self`.
    fn value_type(&self) -> ValueType {
        use {
            BinaryOp::{Complex, *},
            NaryOp::*,
            TernaryOp::*,
            UnaryOp::*,
            ValueType::{Complex as ComplexT, *},
        };

        fn boolean(e: &Expr) -> bool {
            e.ty == Boolean
        }

        fn complex(e: &Expr) -> bool {
            e.ty == ComplexT
        }

        fn real(e: &Expr) -> bool {
            e.ty == Real
        }

        fn real_or_complex(e: &Expr) -> bool {
            real(e) || complex(e)
        }

        fn real_vector(e: &Expr) -> bool {
            e.ty == RealVector
        }

        match self {
            // Boolean
            bool_constant!(_) => Boolean,
            unary!(Not, x) if boolean(x) => Boolean,
            binary!(And | Or, x, y) if boolean(x) && boolean(y) => Boolean,
            binary!(Eq, x, y) if real_or_complex(x) && real_or_complex(y) => Boolean,
            binary!(ExplicitRel | Ge | Gt | Le | Lt, x, y) if real(x) && real(y) => Boolean,
            // Complex
            unary!(
                Acos | Acosh
                    | Asin
                    | Asinh
                    | Atan
                    | Atanh
                    | Conj
                    | Cos
                    | Cosh
                    | Exp
                    | Ln
                    | Neg
                    | Recip
                    | Sign
                    | Sin
                    | Sinh
                    | Sqr
                    | Sqrt
                    | Tan
                    | Tanh,
                x
            ) if complex(x) => ComplexT,
            binary!(Complex, x, y) if real(x) && real(y) => ComplexT,
            binary!(Add | Div | Log | Mul | Pow | Sub, x, y)
                if real_or_complex(x) && real_or_complex(y) && (complex(x) || complex(y)) =>
            {
                ComplexT
            }
            ternary!(IfThenElse, cond, t, f)
                if real(cond)
                    && real_or_complex(t)
                    && real_or_complex(f)
                    && (complex(t) || complex(f)) =>
            {
                ComplexT
            }
            nary!(Plus | Times, xs) if xs.iter().all(real_or_complex) && xs.iter().any(complex) => {
                ComplexT
            }
            // Real
            constant!(_) => Real,
            unary!(Boole, x) if boolean(x) => Real,
            unary!(Abs | Arg | Im | Re, x) if complex(x) => Real,
            unary!(
                Abs | Acos
                    | Acosh
                    | AiryAi
                    | AiryAiPrime
                    | AiryBi
                    | AiryBiPrime
                    | Arg
                    | Asin
                    | Asinh
                    | Atan
                    | Atanh
                    | BooleEqZero
                    | BooleLeZero
                    | BooleLtZero
                    | Ceil
                    | Chi
                    | Ci
                    | Conj
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
                    | Floor
                    | FresnelC
                    | FresnelS
                    | Gamma
                    | Im
                    | Li
                    | Ln
                    | Neg
                    | Re
                    | Recip
                    | Shi
                    | Si
                    | Sign
                    | Sin
                    | Sinc
                    | Sinh
                    | Sqr
                    | Sqrt
                    | Tan
                    | Tanh
                    | UndefAt0,
                x
            ) if real(x) => Real,
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
                    | PowRational
                    | ReSignNonnegative
                    | Sub,
                x,
                y
            ) if real(x) && real(y) => Real,
            binary!(RankedMax | RankedMin, x, y) if real_vector(x) && real(y) => Real,
            ternary!(IfThenElse, cond, t, f) if real(cond) && real(t) && real(f) => Real,
            ternary!(MulAdd, x, y, z) if real(x) && real(y) && real(z) => Real,
            nary!(Plus | Times, xs) if xs.iter().all(real) => Real,
            pown!(x, _) | rootn!(x, _) if real(x) => Real,
            // RealVector
            nary!(List, xs) if xs.iter().all(real) => RealVector,
            // Others
            var!(_) => self.ty,
            uninit!() => panic!(),
            _ => Unknown,
        }
    }

    /// Returns the set of free variables in the expression.
    ///
    /// Precondition: [`Expr::vars`] is correctly assigned for `self`.
    fn variables(&self) -> VarSet {
        match self {
            bool_constant!(_) | constant!(_) => VarSet::EMPTY,
            var!(_) => self.vars,
            unary!(_, x) | pown!(x, _) | rootn!(x, _) => x.vars,
            binary!(_, x, y) => x.vars | y.vars,
            ternary!(_, x, y, z) => x.vars | y.vars | z.vars,
            nary!(_, xs) => xs.iter().fold(VarSet::EMPTY, |vs, x| vs | x.vars),
            error!() => VarSet::EMPTY,
            uninit!() => panic!(),
        }
    }
}

impl Default for Expr {
    fn default() -> Self {
        Self::new(ExprKind::Uninit)
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

struct DumpShort<'a>(&'a Expr);

impl<'a> fmt::Display for DumpShort<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self.0 {
            bool_constant!(false) => {
                write!(f, "False")
            }
            bool_constant!(true) => {
                write!(f, "True")
            }
            constant!(a) => {
                if let Some(a) = a.to_f64() {
                    write!(f, "{}", a)
                } else {
                    write!(f, "@")
                }
            }
            var!(name) => write!(f, "{}", name),
            unary!(op, x) => write!(f, "({:?} {})", op, x.dump_short()),
            binary!(op, x, y) => write!(f, "({:?} {} {})", op, x.dump_short(), y.dump_short()),
            ternary!(op, x, y, z) => write!(
                f,
                "({:?} {} {} {})",
                op,
                x.dump_short(),
                y.dump_short(),
                z.dump_short()
            ),
            nary!(op, xs) => {
                write!(
                    f,
                    "({:?} {})",
                    op,
                    xs.iter()
                        .map(|x| format!("{}", x.dump_short()))
                        .collect::<Vec<_>>()
                        .join(" ")
                )
            }
            pown!(x, n) => write!(f, "(Pown {} {})", x.dump_short(), n),
            rootn!(x, n) => write!(f, "(Rootn {} {})", x.dump_short(), n),
            error!() => write!(f, "Error"),
            uninit!() => panic!(),
        }
    }
}
