use crate::{
    ast::VarSet,
    interval_set::{DecSignSet, Site, TupperIntervalSet},
};

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

    pub fn get(&self, store_index: StoreIndex) -> &T {
        &self.0[store_index.0 as usize]
    }

    pub fn put(&mut self, store_index: StoreIndex, value: T) {
        self.0[store_index.0 as usize] = value;
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
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

#[derive(Clone, Debug)]
pub enum StaticTermKind {
    Constant(Box<TupperIntervalSet>),
    X,
    Y,
    NTheta,
    Unary(ScalarUnaryOp, StoreIndex),
    Binary(ScalarBinaryOp, StoreIndex, StoreIndex),
    Ternary(ScalarTernaryOp, StoreIndex, StoreIndex, StoreIndex),
    Pown(StoreIndex, i32),
    Rootn(StoreIndex, u32),
    // Box the `Vec` to keep the enum small.
    // Operations involving lists are relatively rare, so it would be worth the cost of the extra indirection.
    #[allow(clippy::box_vec)]
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
        store.put(self.store_index, value);
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
            Unary(Abs, x) => self.put(ts, ts.get(*x).abs()),
            Unary(Acos, x) => self.put(ts, ts.get(*x).acos()),
            Unary(Acosh, x) => self.put(ts, ts.get(*x).acosh()),
            Unary(AiryAi, x) => self.put(ts, ts.get(*x).airy_ai()),
            Unary(AiryAiPrime, x) => self.put(ts, ts.get(*x).airy_ai_prime()),
            Unary(AiryBi, x) => self.put(ts, ts.get(*x).airy_bi()),
            Unary(AiryBiPrime, x) => self.put(ts, ts.get(*x).airy_bi_prime()),
            Unary(Asin, x) => self.put(ts, ts.get(*x).asin()),
            Unary(Asinh, x) => self.put(ts, ts.get(*x).asinh()),
            Unary(Atan, x) => self.put(ts, ts.get(*x).atan()),
            Unary(Atanh, x) => self.put(ts, ts.get(*x).atanh()),
            Unary(Ceil, x) => self.put(ts, ts.get(*x).ceil(self.site)),
            Unary(Chi, x) => self.put(ts, ts.get(*x).chi()),
            Unary(Ci, x) => self.put(ts, ts.get(*x).ci()),
            Unary(Cos, x) => self.put(ts, ts.get(*x).cos()),
            Unary(Cosh, x) => self.put(ts, ts.get(*x).cosh()),
            Unary(Digamma, x) => self.put(ts, ts.get(*x).digamma(self.site)),
            Unary(Ei, x) => self.put(ts, ts.get(*x).ei()),
            Unary(EllipticE, x) => self.put(ts, ts.get(*x).elliptic_e()),
            Unary(EllipticK, x) => self.put(ts, ts.get(*x).elliptic_k()),
            Unary(Erf, x) => self.put(ts, ts.get(*x).erf()),
            Unary(Erfc, x) => self.put(ts, ts.get(*x).erfc()),
            Unary(Erfi, x) => self.put(ts, ts.get(*x).erfi()),
            Unary(Exp, x) => self.put(ts, ts.get(*x).exp()),
            Unary(Exp10, x) => self.put(ts, ts.get(*x).exp10()),
            Unary(Exp2, x) => self.put(ts, ts.get(*x).exp2()),
            Unary(Floor, x) => self.put(ts, ts.get(*x).floor(self.site)),
            Unary(FresnelC, x) => self.put(ts, ts.get(*x).fresnel_c()),
            Unary(FresnelS, x) => self.put(ts, ts.get(*x).fresnel_s()),
            Unary(Gamma, x) => self.put(ts, ts.get(*x).gamma(self.site)),
            Unary(Li, x) => self.put(ts, ts.get(*x).li()),
            Unary(Ln, x) => self.put(ts, ts.get(*x).ln()),
            Unary(Log10, x) => self.put(ts, ts.get(*x).log10()),
            Unary(Neg, x) => self.put(ts, -ts.get(*x)),
            Unary(One, x) => self.put(ts, ts.get(*x).one()),
            Unary(Recip, x) => self.put(ts, ts.get(*x).recip(self.site)),
            Unary(Shi, x) => self.put(ts, ts.get(*x).shi()),
            Unary(Si, x) => self.put(ts, ts.get(*x).si()),
            Unary(Sin, x) => self.put(ts, ts.get(*x).sin()),
            Unary(Sinc, x) => self.put(ts, ts.get(*x).sinc()),
            Unary(Sinh, x) => self.put(ts, ts.get(*x).sinh()),
            Unary(Sqr, x) => self.put(ts, ts.get(*x).sqr()),
            Unary(Sqrt, x) => self.put(ts, ts.get(*x).sqrt()),
            Unary(Tan, x) => self.put(ts, ts.get(*x).tan(self.site)),
            Unary(Tanh, x) => self.put(ts, ts.get(*x).tanh()),
            Unary(UndefAt0, x) => self.put(ts, ts.get(*x).undef_at_0()),
            Binary(Add, x, y) => self.put(ts, ts.get(*x) + ts.get(*y)),
            Binary(Atan2, y, x) => self.put(ts, ts.get(*y).atan2(ts.get(*x), self.site)),
            Binary(BesselI, n, x) => self.put(ts, ts.get(*n).bessel_i(ts.get(*x))),
            Binary(BesselJ, n, x) => self.put(ts, ts.get(*n).bessel_j(ts.get(*x))),
            Binary(BesselK, n, x) => self.put(ts, ts.get(*n).bessel_k(ts.get(*x))),
            Binary(BesselY, n, x) => self.put(ts, ts.get(*n).bessel_y(ts.get(*x))),
            Binary(Div, x, y) => self.put(ts, ts.get(*x).div(ts.get(*y), self.site)),
            Binary(GammaInc, a, x) => self.put(ts, ts.get(*a).gamma_inc(ts.get(*x))),
            Binary(Gcd, x, y) => self.put(ts, ts.get(*x).gcd(ts.get(*y), self.site)),
            Binary(Lcm, x, y) => self.put(ts, ts.get(*x).lcm(ts.get(*y), self.site)),
            // Beware the order of arguments.
            Binary(Log, b, x) => self.put(ts, ts.get(*x).log(ts.get(*b), self.site)),
            Binary(Max, x, y) => self.put(ts, ts.get(*x).max(ts.get(*y))),
            Binary(Min, x, y) => self.put(ts, ts.get(*x).min(ts.get(*y))),
            Binary(Mod, x, y) => self.put(ts, ts.get(*x).rem_euclid(ts.get(*y), self.site)),
            Binary(Mul, x, y) => self.put(ts, ts.get(*x) * ts.get(*y)),
            Binary(Pow, x, y) => self.put(ts, ts.get(*x).pow(ts.get(*y), self.site)),
            Binary(Sub, x, y) => self.put(ts, ts.get(*x) - ts.get(*y)),
            Ternary(MulAdd, x, y, z) => self.put(ts, ts.get(*x).mul_add(ts.get(*y), ts.get(*z))),
            Pown(x, n) => self.put(ts, ts.get(*x).pown(*n, self.site)),
            Rootn(x, n) => self.put(ts, ts.get(*x).rootn(*n)),
            RankedMinMax(RankedMax, xs, n) => {
                self.put(
                    ts,
                    TupperIntervalSet::ranked_max(
                        xs.iter().map(|x| ts.get(*x)).collect(),
                        ts.get(*n),
                        self.site,
                    ),
                );
            }
            RankedMinMax(RankedMin, xs, n) => {
                self.put(
                    ts,
                    TupperIntervalSet::ranked_min(
                        xs.iter().map(|x| ts.get(*x)).collect(),
                        ts.get(*n),
                        self.site,
                    ),
                );
            }
            X | Y | NTheta => panic!("this term cannot be evaluated"),
        }
    }
}

#[derive(Clone, Debug)]
pub enum StaticFormKind {
    Atomic(RelOp, StoreIndex, StoreIndex),
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
            Atomic(Eq, x, y) => ts.get(*x).eq(ts.get(*y)),
            Atomic(Ge, x, y) => ts.get(*x).ge(ts.get(*y)),
            Atomic(Gt, x, y) => ts.get(*x).gt(ts.get(*y)),
            Atomic(Le, x, y) => ts.get(*x).le(ts.get(*y)),
            Atomic(Lt, x, y) => ts.get(*x).lt(ts.get(*y)),
            Atomic(Neq, x, y) => ts.get(*x).neq(ts.get(*y)),
            Atomic(Nge, x, y) => ts.get(*x).nge(ts.get(*y)),
            Atomic(Ngt, x, y) => ts.get(*x).ngt(ts.get(*y)),
            Atomic(Nle, x, y) => ts.get(*x).nle(ts.get(*y)),
            Atomic(Nlt, x, y) => ts.get(*x).nlt(ts.get(*y)),
            And(_, _) | Or(_, _) => panic!("non-atomic formulas cannot be evaluated"),
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
