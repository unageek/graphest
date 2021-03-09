use crate::interval_set::TupperIntervalSet;
use bitflags::*;
use std::{
    collections::hash_map::DefaultHasher,
    fmt,
    hash::{Hash, Hasher},
};

pub type TermId = u32;
pub const UNINIT_TERM_ID: TermId = TermId::MAX;

pub type FormId = u32;
pub const UNINIT_FORM_ID: FormId = FormId::MAX;

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
    Atan2,
    BesselI,
    BesselJ,
    BesselK,
    BesselY,
    Div,
    GammaInc,
    Gcd,
    Lcm,
    Log,
    Max,
    Min,
    Mod,
    Mul,
    Pow,
    RankedMax,
    RankedMin,
    Sub,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum RelOp {
    Eq,
    Ge,
    Gt,
    Le,
    Lt,
    Neq,
    Nge,
    Ngt,
    Nle,
    Nlt,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub enum TermKind {
    // == Scalar terms ==
    Constant(Box<TupperIntervalSet>),
    Unary(UnaryOp, Box<Term>),
    Binary(BinaryOp, Box<Term>, Box<Term>),
    Pown(Box<Term>, i32),
    // == Others ==
    Var(String),
    List(Vec<Term>),
    Uninit,
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

/// An AST node for a term.
#[derive(Clone, Debug)]
pub struct Term {
    pub id: TermId,
    pub kind: TermKind,
    /// The set of the free variables in the term.
    pub vars: VarSet,
    internal_hash: u64,
}

impl Term {
    pub fn new(kind: TermKind) -> Self {
        Self {
            id: UNINIT_TERM_ID,
            kind,
            vars: VarSet::EMPTY,
            internal_hash: 0,
        }
    }

    pub fn dump_structure(&self) -> impl fmt::Display + '_ {
        DumpTermStructure(self)
    }

    /// Evaluates the term.
    ///
    /// Returns [`None`] if the term cannot be evaluated to a scalar constant.
    pub fn eval(&self) -> Option<TupperIntervalSet> {
        use {BinaryOp::*, TermKind::*, UnaryOp::*};
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
                    TupperIntervalSet::ranked_max(&xs, &n.eval()?, None)
                } else {
                    panic!("a list is expected")
                }
            }),
            Binary(RankedMin, xs, n) => Some({
                if let List(xs) = &xs.kind {
                    let xs = xs.iter().map(|x| x.eval()).collect::<Option<Vec<_>>>()?;
                    TupperIntervalSet::ranked_min(&xs, &n.eval()?, None)
                } else {
                    panic!("a list is expected")
                }
            }),
            Binary(Sub, x, y) => Some(&x.eval()? - &y.eval()?),
            Pown(x, y) => Some(x.eval()?.pown(*y, None)),
            Var(_) | List(_) => None,
            Uninit => panic!(),
        }
    }

    /// Updates [`Term::vars`] and [`Term::internal_hash`] fields of the term.
    ///
    /// Precondition:
    ///   The function is called on all sub-terms and they have not been changed since then.
    ///
    /// Panics if the term contains [`TermKind::Var`] with a name other than `"x"` or `"y"`.
    pub fn update_metadata(&mut self) {
        self.vars = match &self.kind {
            TermKind::Constant(_) => VarSet::EMPTY,
            TermKind::Var(x) if x == "x" => VarSet::X,
            TermKind::Var(x) if x == "y" => VarSet::Y,
            TermKind::Unary(_, x) | TermKind::Pown(x, _) => x.vars,
            TermKind::Binary(_, x, y) => x.vars | y.vars,
            TermKind::Var(x) => panic!("'{}' is undefined", x),
            TermKind::List(xs) => xs.iter().fold(VarSet::EMPTY, |vs, x| vs | x.vars),
            TermKind::Uninit => panic!(),
        };
        self.internal_hash = {
            // Use `DefaultHasher::new` so that the value of `internal_hash` will be deterministic.
            let mut hasher = DefaultHasher::new();
            self.kind.hash(&mut hasher);
            hasher.finish()
        }
    }
}

impl Default for Term {
    fn default() -> Self {
        Self {
            id: UNINIT_TERM_ID,
            kind: TermKind::Uninit,
            vars: VarSet::EMPTY,
            internal_hash: 0,
        }
    }
}

impl PartialEq for Term {
    fn eq(&self, rhs: &Self) -> bool {
        self.kind == rhs.kind
    }
}

impl Eq for Term {}

impl Hash for Term {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.internal_hash.hash(state);
    }
}

struct DumpTermStructure<'a>(&'a Term);

impl<'a> fmt::Display for DumpTermStructure<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match &self.0.kind {
            TermKind::Constant(_) => write!(f, "@"),
            TermKind::Var(x) => write!(f, "{}", x),
            TermKind::Unary(op, x) => write!(f, "({:?} {})", op, x.dump_structure()),
            TermKind::Binary(op, x, y) => write!(
                f,
                "({:?} {} {})",
                op,
                x.dump_structure(),
                y.dump_structure()
            ),
            TermKind::Pown(x, y) => write!(f, "(Pown {} {})", x.dump_structure(), y),
            TermKind::List(xs) => {
                let mut parts = vec!["List".to_string()];
                parts.extend(xs.iter().map(|x| format!("{}", x.dump_structure())));
                write!(f, "({})", parts.join(" "))
            }
            TermKind::Uninit => panic!(),
        }
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub enum FormKind {
    Atomic(RelOp, Box<Term>, Box<Term>),
    Not(Box<Form>),
    And(Box<Form>, Box<Form>),
    Or(Box<Form>, Box<Form>),
    Uninit,
}

/// An AST node for a formula.
#[derive(Clone, Debug)]
pub struct Form {
    pub id: FormId,
    pub kind: FormKind,
    internal_hash: u64,
}

impl Form {
    pub fn new(kind: FormKind) -> Self {
        Self {
            id: UNINIT_FORM_ID,
            kind,
            internal_hash: 0,
        }
    }

    pub fn dump_structure(&self) -> impl fmt::Display + '_ {
        DumpFormStructure(self)
    }

    /// Updates `internal_hash` field of `self`.
    ///
    /// Precondition:
    ///   The function is called on all sub-terms/forms and they have not been changed since then.
    pub fn update_metadata(&mut self) {
        self.internal_hash = {
            let mut hasher = DefaultHasher::new();
            self.kind.hash(&mut hasher);
            hasher.finish()
        }
    }
}

impl Default for Form {
    fn default() -> Self {
        Self {
            id: UNINIT_FORM_ID,
            kind: FormKind::Uninit,
            internal_hash: 0,
        }
    }
}

impl PartialEq for Form {
    fn eq(&self, rhs: &Self) -> bool {
        self.kind == rhs.kind
    }
}

impl Eq for Form {}

impl Hash for Form {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.kind.hash(state);
    }
}

struct DumpFormStructure<'a>(&'a Form);

impl<'a> fmt::Display for DumpFormStructure<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match &self.0.kind {
            FormKind::Atomic(op, x, y) => write!(
                f,
                "({:?} {} {})",
                op,
                x.dump_structure(),
                y.dump_structure()
            ),
            FormKind::Not(x) => write!(f, "(Not {})", x.dump_structure()),
            FormKind::And(x, y) => write!(f, "(And {} {})", x.dump_structure(), y.dump_structure()),
            FormKind::Or(x, y) => write!(f, "(Or {} {})", x.dump_structure(), y.dump_structure()),
            FormKind::Uninit => panic!(),
        }
    }
}
