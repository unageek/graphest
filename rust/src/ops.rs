use crate::{
    ast::VarSet,
    interval_set::{DecSignSet, Site, TupperIntervalSet},
};
use std::ops::{Index, IndexMut};

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct StoreIndex(u32);

impl StoreIndex {
    pub fn new(i: u32) -> Self {
        Self(i)
    }
}

#[derive(Clone, Debug)]
pub struct ValueStore<T: Clone>(Vec<T>);

impl<T: Clone> ValueStore<T> {
    pub fn new(init: T, size: usize) -> Self {
        Self(vec![init; size])
    }
}

impl<T: Clone> Index<StoreIndex> for ValueStore<T> {
    type Output = T;

    fn index(&self, index: StoreIndex) -> &Self::Output {
        &self.0[index.0 as usize]
    }
}

impl<T: Clone> IndexMut<StoreIndex> for ValueStore<T> {
    fn index_mut(&mut self, index: StoreIndex) -> &mut Self::Output {
        &mut self.0[index.0 as usize]
    }
}

pub type FormIndex = u32;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ScalarUnaryOp {
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ScalarBinaryOp {
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
    Sub,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ScalarTernaryOp {
    MulAdd,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RankedMinMaxOp {
    RankedMax,
    RankedMin,
}

#[allow(clippy::enum_variant_names)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RelOp {
    EqZero,
    LeZero,
    LtZero,
    NeqZero,
    NleZero,
    NltZero,
}

#[derive(Clone, Debug)]
pub enum StaticTermKind {
    Constant(Box<TupperIntervalSet>),
    X,
    Y,
    NTheta,
    T,
    Unary(ScalarUnaryOp, StoreIndex),
    Binary(ScalarBinaryOp, StoreIndex, StoreIndex),
    Ternary(ScalarTernaryOp, StoreIndex, StoreIndex, StoreIndex),
    Pown(StoreIndex, i32),
    Rootn(StoreIndex, u32),
    // Box the `Vec` to keep the enum small.
    // Operations involving lists are relatively rare, so it would be worth the cost of the extra indirection.
    #[allow(clippy::box_collection)]
    RankedMinMax(RankedMinMaxOp, Box<Vec<StoreIndex>>, StoreIndex),
}

/// A term in a cache-efficient representation.
#[derive(Clone, Debug)]
pub struct StaticTerm {
    pub site: Option<Site>,
    pub kind: StaticTermKind,
    pub vars: VarSet,
    pub store_index: StoreIndex,
}

impl StaticTerm {
    pub fn put<T: Clone>(&self, store: &mut ValueStore<T>, value: T) {
        store[self.store_index] = value;
    }

    /// Evaluates the term and puts the result in the value store.
    ///
    /// Panics if the term is of the kind [`StaticTermKind::X`], [`StaticTermKind::Y`]
    /// or [`StaticTermKind::NTheta`].
    pub fn put_eval(&self, ts: &mut ValueStore<TupperIntervalSet>) {
        use {
            RankedMinMaxOp::*, ScalarBinaryOp::*, ScalarTernaryOp::*, ScalarUnaryOp::*,
            StaticTermKind::*,
        };
        match &self.kind {
            Constant(x) => self.put(ts, *x.clone()),
            Unary(Abs, x) => self.put(ts, ts[*x].abs()),
            Unary(Acos, x) => self.put(ts, ts[*x].acos()),
            Unary(Acosh, x) => self.put(ts, ts[*x].acosh()),
            Unary(AiryAi, x) => self.put(ts, ts[*x].airy_ai()),
            Unary(AiryAiPrime, x) => self.put(ts, ts[*x].airy_ai_prime()),
            Unary(AiryBi, x) => self.put(ts, ts[*x].airy_bi()),
            Unary(AiryBiPrime, x) => self.put(ts, ts[*x].airy_bi_prime()),
            Unary(Asin, x) => self.put(ts, ts[*x].asin()),
            Unary(Asinh, x) => self.put(ts, ts[*x].asinh()),
            Unary(Atan, x) => self.put(ts, ts[*x].atan()),
            Unary(Atanh, x) => self.put(ts, ts[*x].atanh()),
            Unary(Ceil, x) => self.put(ts, ts[*x].ceil(self.site)),
            Unary(Chi, x) => self.put(ts, ts[*x].chi()),
            Unary(Ci, x) => self.put(ts, ts[*x].ci()),
            Unary(Cos, x) => self.put(ts, ts[*x].cos()),
            Unary(Cosh, x) => self.put(ts, ts[*x].cosh()),
            Unary(Digamma, x) => self.put(ts, ts[*x].digamma(self.site)),
            Unary(Ei, x) => self.put(ts, ts[*x].ei()),
            Unary(EllipticE, x) => self.put(ts, ts[*x].elliptic_e()),
            Unary(EllipticK, x) => self.put(ts, ts[*x].elliptic_k()),
            Unary(Erf, x) => self.put(ts, ts[*x].erf()),
            Unary(Erfc, x) => self.put(ts, ts[*x].erfc()),
            Unary(Erfi, x) => self.put(ts, ts[*x].erfi()),
            Unary(Exp, x) => self.put(ts, ts[*x].exp()),
            Unary(Exp10, x) => self.put(ts, ts[*x].exp10()),
            Unary(Exp2, x) => self.put(ts, ts[*x].exp2()),
            Unary(Floor, x) => self.put(ts, ts[*x].floor(self.site)),
            Unary(FresnelC, x) => self.put(ts, ts[*x].fresnel_c()),
            Unary(FresnelS, x) => self.put(ts, ts[*x].fresnel_s()),
            Unary(Gamma, x) => self.put(ts, ts[*x].gamma(self.site)),
            Unary(Li, x) => self.put(ts, ts[*x].li()),
            Unary(Ln, x) => self.put(ts, ts[*x].ln()),
            Unary(Log10, x) => self.put(ts, ts[*x].log10()),
            Unary(Neg, x) => self.put(ts, -&ts[*x]),
            Unary(One, x) => self.put(ts, ts[*x].one()),
            Unary(Recip, x) => self.put(ts, ts[*x].recip(self.site)),
            Unary(Shi, x) => self.put(ts, ts[*x].shi()),
            Unary(Si, x) => self.put(ts, ts[*x].si()),
            Unary(Sin, x) => self.put(ts, ts[*x].sin()),
            Unary(Sinc, x) => self.put(ts, ts[*x].sinc()),
            Unary(Sinh, x) => self.put(ts, ts[*x].sinh()),
            Unary(Sqr, x) => self.put(ts, ts[*x].sqr()),
            Unary(Sqrt, x) => self.put(ts, ts[*x].sqrt()),
            Unary(Tan, x) => self.put(ts, ts[*x].tan(self.site)),
            Unary(Tanh, x) => self.put(ts, ts[*x].tanh()),
            Unary(UndefAt0, x) => self.put(ts, ts[*x].undef_at_0()),
            Binary(Add, x, y) => self.put(ts, &ts[*x] + &ts[*y]),
            Binary(Atan2, y, x) => self.put(ts, ts[*y].atan2(&ts[*x], self.site)),
            Binary(BesselI, n, x) => self.put(ts, ts[*n].bessel_i(&ts[*x])),
            Binary(BesselJ, n, x) => self.put(ts, ts[*n].bessel_j(&ts[*x])),
            Binary(BesselK, n, x) => self.put(ts, ts[*n].bessel_k(&ts[*x])),
            Binary(BesselY, n, x) => self.put(ts, ts[*n].bessel_y(&ts[*x])),
            Binary(Div, x, y) => self.put(ts, ts[*x].div(&ts[*y], self.site)),
            Binary(GammaInc, a, x) => self.put(ts, ts[*a].gamma_inc(&ts[*x])),
            Binary(Gcd, x, y) => self.put(ts, ts[*x].gcd(&ts[*y], self.site)),
            Binary(Lcm, x, y) => self.put(ts, ts[*x].lcm(&ts[*y], self.site)),
            // Beware the order of arguments.
            Binary(Log, b, x) => self.put(ts, ts[*x].log(&ts[*b], self.site)),
            Binary(Max, x, y) => self.put(ts, ts[*x].max(&ts[*y])),
            Binary(Min, x, y) => self.put(ts, ts[*x].min(&ts[*y])),
            Binary(Mod, x, y) => self.put(ts, ts[*x].rem_euclid(&ts[*y], self.site)),
            Binary(Mul, x, y) => self.put(ts, &ts[*x] * &ts[*y]),
            Binary(Pow, x, y) => self.put(ts, ts[*x].pow(&ts[*y], self.site)),
            Binary(Sub, x, y) => self.put(ts, &ts[*x] - &ts[*y]),
            Ternary(MulAdd, x, y, z) => self.put(ts, ts[*x].mul_add(&ts[*y], &ts[*z])),
            Pown(x, n) => self.put(ts, ts[*x].pown(*n, self.site)),
            Rootn(x, n) => self.put(ts, ts[*x].rootn(*n)),
            RankedMinMax(RankedMax, xs, n) => {
                self.put(
                    ts,
                    TupperIntervalSet::ranked_max(
                        xs.iter().map(|x| &ts[*x]).collect(),
                        &ts[*n],
                        self.site,
                    ),
                );
            }
            RankedMinMax(RankedMin, xs, n) => {
                self.put(
                    ts,
                    TupperIntervalSet::ranked_min(
                        xs.iter().map(|x| &ts[*x]).collect(),
                        &ts[*n],
                        self.site,
                    ),
                );
            }
            X | Y | NTheta | T => panic!("this term cannot be evaluated"),
        }
    }
}

#[derive(Clone, Debug)]
pub enum StaticFormKind {
    Constant(bool),
    Atomic(RelOp, StoreIndex),
    And(FormIndex, FormIndex),
    Or(FormIndex, FormIndex),
}

/// A formula in a cache-efficient representation.
#[derive(Clone, Debug)]
pub struct StaticForm {
    pub kind: StaticFormKind,
}

impl StaticForm {
    /// Evaluates the formula.
    ///
    /// Panics if the formula is *not* of the kind [`StaticFormKind::Atomic`].
    pub fn eval(&self, ts: &ValueStore<TupperIntervalSet>) -> DecSignSet {
        use {RelOp::*, StaticFormKind::*};
        match &self.kind {
            Atomic(EqZero, x) => ts[*x].eq_zero(),
            Atomic(LeZero, x) => ts[*x].le_zero(),
            Atomic(LtZero, x) => ts[*x].lt_zero(),
            Atomic(NeqZero, x) => ts[*x].neq_zero(),
            Atomic(NleZero, x) => ts[*x].nle_zero(),
            Atomic(NltZero, x) => ts[*x].nlt_zero(),
            Constant(_) | And(_, _) | Or(_, _) => {
                panic!("constant or non-atomic formulas cannot be evaluated")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::size_of;

    #[test]
    fn struct_size() {
        assert_eq!(size_of::<StaticTermKind>(), 16);
        assert_eq!(size_of::<StaticTerm>(), 24);
        assert_eq!(size_of::<StaticFormKind>(), 12);
        assert_eq!(size_of::<StaticForm>(), 12);
    }
}
