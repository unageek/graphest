use crate::{
    ast::{BinaryOp, FormId, RelOp, TermId, UnaryOp, VarSet},
    interval_set::{DecSignSet, Site, TupperIntervalSet},
};

#[derive(Clone, Debug)]
pub enum StaticTermKind {
    Constant(Box<TupperIntervalSet>),
    X,
    Y,
    Unary(UnaryOp, TermId),
    Binary(BinaryOp, TermId, TermId),
    Pown(TermId, i32),
}

/// A term in a cache-efficient representation.
#[derive(Clone, Debug)]
pub struct StaticTerm {
    pub site: Option<Site>,
    pub kind: StaticTermKind,
    pub vars: VarSet,
}

impl StaticTerm {
    /// Evaluates the term.
    ///
    /// Panics if the term is of kind [`StaticTermKind::X`] or [`StaticTermKind::Y`].
    pub fn eval(&self, ts: &[TupperIntervalSet]) -> TupperIntervalSet {
        use {BinaryOp::*, StaticTermKind::*, UnaryOp::*};
        match &self.kind {
            Constant(x) => *x.clone(),
            Unary(Abs, x) => ts[*x as usize].abs(),
            Unary(Acos, x) => ts[*x as usize].acos(),
            Unary(Acosh, x) => ts[*x as usize].acosh(),
            Unary(AiryAi, x) => ts[*x as usize].airy_ai(),
            Unary(AiryAiPrime, x) => ts[*x as usize].airy_ai_prime(),
            Unary(AiryBi, x) => ts[*x as usize].airy_bi(),
            Unary(AiryBiPrime, x) => ts[*x as usize].airy_bi_prime(),
            Unary(Asin, x) => ts[*x as usize].asin(),
            Unary(Asinh, x) => ts[*x as usize].asinh(),
            Unary(Atan, x) => ts[*x as usize].atan(),
            Unary(Atanh, x) => ts[*x as usize].atanh(),
            Unary(Ceil, x) => ts[*x as usize].ceil(self.site),
            Unary(Chi, x) => ts[*x as usize].chi(),
            Unary(Ci, x) => ts[*x as usize].ci(),
            Unary(Cos, x) => ts[*x as usize].cos(),
            Unary(Cosh, x) => ts[*x as usize].cosh(),
            Unary(Ei, x) => ts[*x as usize].ei(),
            Unary(Erf, x) => ts[*x as usize].erf(),
            Unary(Erfc, x) => ts[*x as usize].erfc(),
            Unary(Erfi, x) => ts[*x as usize].erfi(),
            Unary(Exp, x) => ts[*x as usize].exp(),
            Unary(Exp10, x) => ts[*x as usize].exp10(),
            Unary(Exp2, x) => ts[*x as usize].exp2(),
            Unary(Floor, x) => ts[*x as usize].floor(self.site),
            Unary(FresnelC, x) => ts[*x as usize].fresnel_c(),
            Unary(FresnelS, x) => ts[*x as usize].fresnel_s(),
            Unary(Gamma, x) => ts[*x as usize].gamma(self.site),
            Unary(Li, x) => ts[*x as usize].li(),
            Unary(Ln, x) => ts[*x as usize].ln(),
            Unary(Log10, x) => ts[*x as usize].log10(),
            Unary(Neg, x) => -&ts[*x as usize],
            Unary(One, x) => ts[*x as usize].one(),
            Unary(Recip, x) => ts[*x as usize].recip(self.site),
            Unary(Shi, x) => ts[*x as usize].shi(),
            Unary(Si, x) => ts[*x as usize].si(),
            Unary(Sign, x) => ts[*x as usize].sign(self.site),
            Unary(Sin, x) => ts[*x as usize].sin(),
            Unary(Sinc, x) => ts[*x as usize].sinc(),
            Unary(Sinh, x) => ts[*x as usize].sinh(),
            Unary(Sqr, x) => ts[*x as usize].sqr(),
            Unary(Sqrt, x) => ts[*x as usize].sqrt(),
            Unary(Tan, x) => ts[*x as usize].tan(self.site),
            Unary(Tanh, x) => ts[*x as usize].tanh(),
            Unary(UndefAt0, x) => ts[*x as usize].undef_at_0(),
            Binary(Add, x, y) => &ts[*x as usize] + &ts[*y as usize],
            Binary(Atan2, x, y) => ts[*x as usize].atan2(&ts[*y as usize], self.site),
            Binary(BesselI, n, x) => ts[*n as usize].bessel_i(&ts[*x as usize]),
            Binary(BesselJ, n, x) => ts[*n as usize].bessel_j(&ts[*x as usize]),
            Binary(BesselK, n, x) => ts[*n as usize].bessel_k(&ts[*x as usize]),
            Binary(BesselY, n, x) => ts[*n as usize].bessel_y(&ts[*x as usize]),
            Binary(Div, x, y) => ts[*x as usize].div(&ts[*y as usize], self.site),
            // Beware the order of arguments.
            Binary(Log, b, x) => ts[*x as usize].log(&ts[*b as usize], self.site),
            Binary(Max, x, y) => ts[*x as usize].max(&ts[*y as usize]),
            Binary(Min, x, y) => ts[*x as usize].min(&ts[*y as usize]),
            Binary(Mod, x, y) => ts[*x as usize].rem_euclid(&ts[*y as usize], self.site),
            Binary(Mul, x, y) => &ts[*x as usize] * &ts[*y as usize],
            Binary(Pow, x, y) => ts[*x as usize].pow(&ts[*y as usize], self.site),
            Binary(Sub, x, y) => &ts[*x as usize] - &ts[*y as usize],
            Pown(x, y) => ts[*x as usize].pown(*y, self.site),
            X | Y => panic!("free variables cannot be evaluated"),
        }
    }
}

#[derive(Clone, Debug)]
pub enum StaticFormKind {
    Atomic(RelOp, TermId, TermId),
    And(FormId, FormId),
    Or(FormId, FormId),
}

/// A formula in a cache-efficient representation.
#[derive(Clone, Debug)]
pub struct StaticForm {
    pub kind: StaticFormKind,
}

impl StaticForm {
    /// Evaluates the formula.
    ///
    /// Panics if the formula is *not* of kind [`StaticFormKind::Atomic`].
    pub fn eval(&self, ts: &[TupperIntervalSet]) -> DecSignSet {
        use {RelOp::*, StaticFormKind::*};
        match &self.kind {
            Atomic(Eq, x, y) => ts[*x as usize].eq(&ts[*y as usize]),
            Atomic(Ge, x, y) => ts[*x as usize].ge(&ts[*y as usize]),
            Atomic(Gt, x, y) => ts[*x as usize].gt(&ts[*y as usize]),
            Atomic(Le, x, y) => ts[*x as usize].le(&ts[*y as usize]),
            Atomic(Lt, x, y) => ts[*x as usize].lt(&ts[*y as usize]),
            And(_, _) | Or(_, _) => panic!("non-atomic formulas cannot be evaluated"),
        }
    }
}
