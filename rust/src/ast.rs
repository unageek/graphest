use crate::real::Real;
use bitflags::*;
use inari::{const_dec_interval, DecInterval};
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
    Arg,
    Asin,
    Asinh,
    Atan,
    Atanh,
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
    Exp10,
    Exp2,
    Floor,
    FresnelC,
    FresnelS,
    Gamma,
    Im,
    Li,
    Ln,
    Log10,
    Neg,
    Not,
    One,
    Re,
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
    Complex,
    Div,
    Eq,
    /// Equality in explicit relations.
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
    Constant(Box<Real>),
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
    Boolean,
    Complex,
    Real,
    RealVector,
    Unknown,
}

bitflags! {
    /// A set of free variables; a subset of {x, y, n_θ, t}.
    pub struct VarSet: u8 {
        const EMPTY = 0;
        const X = 1;
        const Y = 2;
        const N_THETA = 4;
        const T = 8;
    }
}

/// An AST node for an expression.
#[derive(Clone, Debug)]
pub struct Expr {
    pub id: ExprId,
    pub kind: ExprKind,
    pub ty: ValueType,
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
    pub fn constant(x: Real) -> Self {
        Self::new(ExprKind::Constant(box x))
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

    pub fn dump_structure(&self) -> impl fmt::Display + '_ {
        DumpStructure(self)
    }

    /// Evaluates the expression.
    ///
    /// Returns [`None`] if the expression cannot be evaluated to a real constant
    /// or constant evaluation is not implemented for the operation.
    pub fn eval(&self) -> Option<Real> {
        use {BinaryOp::*, NaryOp::*, UnaryOp::*};
        match self {
            constant!(x) => Some(x.clone()),
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
            unary!(Log10, x) => Some(x.eval()?.log10()),
            unary!(Shi, x) => Some(x.eval()?.shi()),
            unary!(Si, x) => Some(x.eval()?.si()),
            unary!(Sin, x) => Some(x.eval()?.sin()),
            unary!(Sinc, x) => Some(x.eval()?.sinc()),
            unary!(Sinh, x) => Some(x.eval()?.sinh()),
            unary!(Tan, x) => Some(x.eval()?.tan()),
            unary!(Tanh, x) => Some(x.eval()?.tanh()),
            unary!(UndefAt0, x) => Some(x.eval()?.undef_at_0()),
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
            binary!(Mod, x, y) => Some(x.eval()?.rem_euclid(y.eval()?)),
            binary!(Mul, x, y) => Some(x.eval()? * y.eval()?),
            binary!(Pow, x, y) => Some(x.eval()?.pow(y.eval()?)),
            binary!(RankedMax, nary!(List, xs), n) => {
                let xs = xs.iter().map(|x| x.eval()).collect::<Option<Vec<_>>>()?;
                Some(Real::ranked_max(xs, n.eval()?))
            }
            binary!(RankedMin, nary!(List, xs), n) => {
                let xs = xs.iter().map(|x| x.eval()).collect::<Option<Vec<_>>>()?;
                Some(Real::ranked_min(xs, n.eval()?))
            }
            uninit!() => panic!(),
            _ => None,
        }
    }

    /// Updates [`Expr::ty`], [`Expr::vars`], and [`Expr::internal_hash`] of the expression.
    ///
    /// Precondition: The function is called on all sub-expressions
    /// and they have not been modified since then.
    pub fn update_metadata(&mut self) {
        self.ty = self.value_type();
        self.vars = self.variables();
        self.internal_hash = {
            // Use `DefaultHasher::new` so that the value of `internal_hash` will be deterministic.
            let mut hasher = DefaultHasher::new();
            self.kind.hash(&mut hasher);
            hasher.finish()
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

        fn real_vector(e: &Expr) -> bool {
            e.ty == RealVector
        }

        match self {
            // Boolean
            unary!(Not, x) if boolean(x) => Boolean,
            binary!(And | Or, x, y) if boolean(x) && boolean(y) => Boolean,
            binary!(
                Eq | ExplicitRel | Ge | Gt | Le | Lt | Neq | Nge | Ngt | Nle | Nlt,
                x,
                y
            ) if real(x) && real(y) => Boolean,
            // Complex
            unary!(Conj | Cos | Cosh | Exp | Ln | Neg | Sin | Sinh, x) if complex(x) => ComplexT,
            binary!(Complex, x, y) if real(x) && real(y) => ComplexT,
            binary!(Add | Div | Mul | Pow | Sub, x, y)
                if complex(x) && complex(y) || complex(x) && real(y) || real(x) && complex(y) =>
            {
                ComplexT
            }
            nary!(Plus | Times, xs)
                if xs.iter().all(|x| complex(x) || real(x)) && xs.iter().any(complex) =>
            {
                ComplexT
            }
            // Real
            constant!(_) => Real,
            var!(x) if x == "t" || x == "x" || x == "y" || x == "<n-theta>" => Real,
            unary!(Abs | Arg | Im | Re, x) if complex(x) => Real,
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
                    | Sub,
                x,
                y
            ) if real(x) && real(y) => Real,
            binary!(RankedMax | RankedMin, x, y) if real_vector(x) && real(y) => Real,
            ternary!(MulAdd, x, y, z) if real(x) && real(y) && real(z) => Real,
            nary!(Plus | Times, xs) if xs.iter().all(real) => Real,
            pown!(x, _) | rootn!(x, _) if real(x) => Real,
            // RealVector
            nary!(List, xs) if xs.iter().all(real) => RealVector,
            // Others
            uninit!() => panic!(),
            _ => Unknown,
        }
    }

    /// Returns the set of free variables in the expression.
    ///
    /// Precondition: [`Expr::vars`] is correctly assigned for `self`.
    fn variables(&self) -> VarSet {
        match self {
            constant!(_) => VarSet::EMPTY,
            var!(name) if name == "r" => VarSet::X | VarSet::Y,
            var!(name) if name == "t" => VarSet::T,
            var!(name) if name == "theta" || name == "θ" => {
                VarSet::X | VarSet::Y | VarSet::N_THETA
            }
            var!(name) if name == "x" => VarSet::X,
            var!(name) if name == "y" => VarSet::Y,
            var!(name) if name == "<n-theta>" => VarSet::N_THETA,
            var!(_) => VarSet::EMPTY,
            unary!(_, x) | pown!(x, _) | rootn!(x, _) => x.vars,
            binary!(_, x, y) => x.vars | y.vars,
            ternary!(_, x, y, z) => x.vars | y.vars | z.vars,
            nary!(_, xs) => xs.iter().fold(VarSet::EMPTY, |vs, x| vs | x.vars),
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

struct DumpStructure<'a>(&'a Expr);

impl<'a> fmt::Display for DumpStructure<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self.0 {
            constant!(a) => {
                if let Some(a) = a.to_f64() {
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
