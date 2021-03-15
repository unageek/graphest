use crate::interval_set::TupperIntervalSet;
use bitflags::*;
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

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub enum ExprKind {
    // == Scalar-valued expressions ==
    Constant(Box<TupperIntervalSet>),
    Unary(UnaryOp, Box<Expr>),
    Binary(BinaryOp, Box<Expr>, Box<Expr>),
    Pown(Box<Expr>, i32),
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
    /// A set of free variables: x or y.
    pub struct VarSet: u8 {
        const EMPTY = 0b00;
        const X = 0b01;
        const Y = 0b10;
        const XY = 0b11;
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

impl Expr {
    pub fn new(kind: ExprKind) -> Self {
        Self {
            id: UNINIT_EXPR_ID,
            kind,
            ty: ValueType::Unknown,
            vars: VarSet::EMPTY,
            internal_hash: 0,
        }
    }

    pub fn dump_structure(&self) -> impl fmt::Display + '_ {
        DumpStructure(self)
    }

    /// Evaluates the expression.
    ///
    /// Returns [`None`] if the expression cannot be evaluated to a scalar constant.
    pub fn eval(&self) -> Option<TupperIntervalSet> {
        use {BinaryOp::*, ExprKind::*, UnaryOp::*};
        match &self.kind {
            Constant(x) => Some(*x.clone()),
            Unary(Abs, x) => Some(x.eval()?.abs()),
            Unary(Acos, x) => Some(x.eval()?.acos()),
            Unary(Acosh, x) => Some(x.eval()?.acosh()),
            Unary(AiryAi, x) => Some(x.eval()?.airy_ai()),
            Unary(AiryAiPrime, x) => Some(x.eval()?.airy_ai_prime()),
            Unary(AiryBi, x) => Some(x.eval()?.airy_bi()),
            Unary(AiryBiPrime, x) => Some(x.eval()?.airy_bi_prime()),
            Unary(Asin, x) => Some(x.eval()?.asin()),
            Unary(Asinh, x) => Some(x.eval()?.asinh()),
            Unary(Atan, x) => Some(x.eval()?.atan()),
            Unary(Atanh, x) => Some(x.eval()?.atanh()),
            Unary(Ceil, x) => Some(x.eval()?.ceil(None)),
            Unary(Chi, x) => Some(x.eval()?.chi()),
            Unary(Ci, x) => Some(x.eval()?.ci()),
            Unary(Cos, x) => Some(x.eval()?.cos()),
            Unary(Cosh, x) => Some(x.eval()?.cosh()),
            Unary(Digamma, x) => Some(x.eval()?.digamma(None)),
            Unary(Ei, x) => Some(x.eval()?.ei()),
            Unary(Erf, x) => Some(x.eval()?.erf()),
            Unary(Erfc, x) => Some(x.eval()?.erfc()),
            Unary(Erfi, x) => Some(x.eval()?.erfi()),
            Unary(Exp, x) => Some(x.eval()?.exp()),
            Unary(Exp10, x) => Some(x.eval()?.exp10()),
            Unary(Exp2, x) => Some(x.eval()?.exp2()),
            Unary(Floor, x) => Some(x.eval()?.floor(None)),
            Unary(FresnelC, x) => Some(x.eval()?.fresnel_c()),
            Unary(FresnelS, x) => Some(x.eval()?.fresnel_s()),
            Unary(Gamma, x) => Some(x.eval()?.gamma(None)),
            Unary(Li, x) => Some(x.eval()?.li()),
            Unary(Ln, x) => Some(x.eval()?.ln()),
            Unary(Log10, x) => Some(x.eval()?.log10()),
            Unary(Neg, x) => Some(-&x.eval()?),
            Unary(One, x) => Some(x.eval()?.one()),
            Unary(Recip, x) => Some(x.eval()?.recip(None)),
            Unary(Shi, x) => Some(x.eval()?.shi()),
            Unary(Si, x) => Some(x.eval()?.si()),
            Unary(Sign, x) => Some(x.eval()?.sign(None)),
            Unary(Sin, x) => Some(x.eval()?.sin()),
            Unary(Sinc, x) => Some(x.eval()?.sinc()),
            Unary(Sinh, x) => Some(x.eval()?.sinh()),
            Unary(Sqr, x) => Some(x.eval()?.sqr()),
            Unary(Sqrt, x) => Some(x.eval()?.sqrt()),
            Unary(Tan, x) => Some(x.eval()?.tan(None)),
            Unary(Tanh, x) => Some(x.eval()?.tanh()),
            Unary(UndefAt0, x) => Some(x.eval()?.undef_at_0()),
            Binary(Add, x, y) => Some(&x.eval()? + &y.eval()?),
            Binary(Atan2, y, x) => Some(y.eval()?.atan2(&x.eval()?, None)),
            Binary(BesselI, n, x) => Some(n.eval()?.bessel_i(&x.eval()?)),
            Binary(BesselJ, n, x) => Some(n.eval()?.bessel_j(&x.eval()?)),
            Binary(BesselK, n, x) => Some(n.eval()?.bessel_k(&x.eval()?)),
            Binary(BesselY, n, x) => Some(n.eval()?.bessel_y(&x.eval()?)),
            Binary(Div, x, y) => Some(x.eval()?.div(&y.eval()?, None)),
            Binary(GammaInc, a, x) => Some(a.eval()?.gamma_inc(&x.eval()?)),
            Binary(Gcd, x, y) => Some(x.eval()?.gcd(&y.eval()?, None)),
            Binary(Lcm, x, y) => Some(x.eval()?.lcm(&y.eval()?, None)),
            // Beware the order of arguments.
            Binary(Log, b, x) => Some(x.eval()?.log(&b.eval()?, None)),
            Binary(Max, x, y) => Some(x.eval()?.max(&y.eval()?)),
            Binary(Min, x, y) => Some(x.eval()?.min(&y.eval()?)),
            Binary(Mod, x, y) => Some(x.eval()?.rem_euclid(&y.eval()?, None)),
            Binary(Mul, x, y) => Some(&x.eval()? * &y.eval()?),
            Binary(Pow, x, y) => Some(x.eval()?.pow(&y.eval()?, None)),
            Binary(RankedMax, xs, n) => Some({
                if let List(xs) = &xs.kind {
                    let xs = xs.iter().map(|x| x.eval()).collect::<Option<Vec<_>>>()?;
                    TupperIntervalSet::ranked_max(xs.iter().collect(), &n.eval()?, None)
                } else {
                    panic!("a list is expected")
                }
            }),
            Binary(RankedMin, xs, n) => Some({
                if let List(xs) = &xs.kind {
                    let xs = xs.iter().map(|x| x.eval()).collect::<Option<Vec<_>>>()?;
                    TupperIntervalSet::ranked_min(xs.iter().collect(), &n.eval()?, None)
                } else {
                    panic!("a list is expected")
                }
            }),
            Binary(Sub, x, y) => Some(&x.eval()? - &y.eval()?),
            Pown(x, y) => Some(x.eval()?.pown(*y, None)),
            Uninit => panic!(),
            _ => None,
        }
    }

    /// Updates [`Expr::ty`], [`Expr::vars`] and [`Expr::internal_hash`] of the expression.
    ///
    /// Precondition:
    ///   The function is called on all sub-expressions and they have not been changed since then.
    pub fn update_metadata(&mut self) {
        self.ty = self.value_type();
        self.vars = match &self.kind {
            ExprKind::Constant(_) => VarSet::EMPTY,
            ExprKind::Var(x) if x == "x" => VarSet::X,
            ExprKind::Var(x) if x == "y" => VarSet::Y,
            ExprKind::Var(_) => VarSet::EMPTY,
            ExprKind::Unary(_, x) | ExprKind::Pown(x, _) => x.vars,
            ExprKind::Binary(_, x, y) => x.vars | y.vars,
            ExprKind::List(xs) => xs.iter().fold(VarSet::EMPTY, |vs, x| vs | x.vars),
            ExprKind::Uninit => panic!(),
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
                | Atan | Atanh | Ceil | Chi | Ci | Cos | Cosh | Digamma | Ei | Erf | Erfc | Erfi
                | Exp | Exp10 | Exp2 | Floor | FresnelC | FresnelS | Gamma | Li | Ln | Log10 | Neg
                | One | Recip | Shi | Si | Sign | Sin | Sinc | Sinh | Sqr | Sqrt | Tan | Tanh
                | UndefAt0,
                x,
            ) if x.ty == Scalar => Scalar,
            Binary(
                Add | Atan2 | BesselI | BesselJ | BesselK | BesselY | Div | GammaInc | Gcd | Lcm
                | Log | Max | Min | Mod | Mul | Pow | Sub,
                x,
                y,
            ) if x.ty == Scalar && y.ty == Scalar => Scalar,
            Binary(RankedMax | RankedMin, x, y) if x.ty == Vector && y.ty == Scalar => Scalar,
            Pown(x, _) if x.ty == Scalar => Scalar,
            List(xs) if xs.iter().all(|x| x.ty == Scalar) => Vector,
            Unary(Not, x) if x.ty == Boolean => Boolean,
            Binary(And | Or, x, y) if x.ty == Boolean && y.ty == Boolean => Boolean,
            Binary(Eq | Ge | Gt | Le | Lt | Neq | Nge | Ngt | Nle | Nlt, x, y)
                if x.ty == Scalar && y.ty == Scalar =>
            {
                Boolean
            }
            Var(x) if x == "x" || x == "y" => Scalar,
            Uninit => panic!(),
            _ => Unknown,
        }
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
        match &self.0.kind {
            ExprKind::Constant(_) => write!(f, "@"),
            ExprKind::Var(x) => write!(f, "{}", x),
            ExprKind::Unary(op, x) => write!(f, "({:?} {})", op, x.dump_structure()),
            ExprKind::Binary(op, x, y) => write!(
                f,
                "({:?} {} {})",
                op,
                x.dump_structure(),
                y.dump_structure()
            ),
            ExprKind::Pown(x, y) => write!(f, "(Pown {} {})", x.dump_structure(), y),
            ExprKind::List(xs) => {
                let mut parts = vec!["List".to_string()];
                parts.extend(xs.iter().map(|x| format!("{}", x.dump_structure())));
                write!(f, "({})", parts.join(" "))
            }
            ExprKind::Uninit => panic!(),
        }
    }
}
