use crate::{
    interval_set::{DecSignSet, Site, TupperIntervalSet},
    vars::{VarIndex, VarSet, VarType},
};
use inari::const_interval;
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
    BooleEqZero,
    BooleLeZero,
    BooleLtZero,
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
    Floor,
    FresnelC,
    FresnelS,
    Gamma,
    InverseErf,
    InverseErfc,
    Li,
    Ln,
    Neg,
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
    Zeta,
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
    LambertW,
    Lcm,
    Log,
    Max,
    Min,
    Mod,
    Mul,
    Pow,
    PowRational,
    ReSignNonnegative,
    Sub,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ScalarTernaryOp {
    IfThenElse,
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
}

#[derive(Clone, Debug)]
pub enum StaticTermKind {
    Constant(Box<TupperIntervalSet>),
    Var(VarIndex, VarType),
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
    pub defer: bool,
    pub kind: StaticTermKind,
    pub site: Option<Site>,
    pub store_index: StoreIndex,
    pub vars: VarSet,
}

impl StaticTerm {
    pub fn put<T: Clone>(&self, store: &mut ValueStore<T>, value: T) {
        store[self.store_index] = value;
    }

    /// Evaluates the term and puts the result in the value store.
    ///
    /// Panics if the term is of the kind [`StaticTermKind::Var`].
    pub fn put_eval(&self, terms: &[Self], ts: &mut ValueStore<TupperIntervalSet>) {
        use {ScalarTernaryOp::*, StaticTermKind::*};

        if self.defer {
            return;
        }

        match &self.kind {
            Ternary(IfThenElse, _, _, _) => self.put_eval_deferred(terms, ts),
            _ => self.put(ts, self.eval(ts)),
        }
    }

    fn eval(&self, ts: &ValueStore<TupperIntervalSet>) -> TupperIntervalSet {
        use {
            RankedMinMaxOp::*, ScalarBinaryOp::*, ScalarTernaryOp::*, ScalarUnaryOp::*,
            StaticTermKind::*,
        };

        match &self.kind {
            Constant(x) => *x.clone(),
            Unary(Abs, x) => ts[*x].abs(),
            Unary(Acos, x) => ts[*x].acos(),
            Unary(Acosh, x) => ts[*x].acosh(),
            Unary(AiryAi, x) => ts[*x].airy_ai(),
            Unary(AiryAiPrime, x) => ts[*x].airy_ai_prime(),
            Unary(AiryBi, x) => ts[*x].airy_bi(),
            Unary(AiryBiPrime, x) => ts[*x].airy_bi_prime(),
            Unary(Asin, x) => ts[*x].asin(),
            Unary(Asinh, x) => ts[*x].asinh(),
            Unary(Atan, x) => ts[*x].atan(),
            Unary(Atanh, x) => ts[*x].atanh(),
            Unary(BooleEqZero, x) => ts[*x].boole_eq_zero(self.site),
            Unary(BooleLeZero, x) => ts[*x].boole_le_zero(self.site),
            Unary(BooleLtZero, x) => ts[*x].boole_lt_zero(self.site),
            Unary(Ceil, x) => ts[*x].ceil(self.site),
            Unary(Chi, x) => ts[*x].chi(),
            Unary(Ci, x) => ts[*x].ci(),
            Unary(Cos, x) => ts[*x].cos(),
            Unary(Cosh, x) => ts[*x].cosh(),
            Unary(Digamma, x) => ts[*x].digamma(self.site),
            Unary(Ei, x) => ts[*x].ei(),
            Unary(EllipticE, x) => ts[*x].elliptic_e(),
            Unary(EllipticK, x) => ts[*x].elliptic_k(),
            Unary(Erf, x) => ts[*x].erf(),
            Unary(Erfc, x) => ts[*x].erfc(),
            Unary(Erfi, x) => ts[*x].erfi(),
            Unary(Exp, x) => ts[*x].exp(),
            Unary(Floor, x) => ts[*x].floor(self.site),
            Unary(FresnelC, x) => ts[*x].fresnel_c(),
            Unary(FresnelS, x) => ts[*x].fresnel_s(),
            Unary(Gamma, x) => ts[*x].gamma(self.site),
            Unary(InverseErf, x) => ts[*x].inverse_erf(),
            Unary(InverseErfc, x) => ts[*x].inverse_erfc(),
            Unary(Li, x) => ts[*x].li(),
            Unary(Ln, x) => ts[*x].ln(),
            Unary(Neg, x) => -&ts[*x],
            Unary(Recip, x) => ts[*x].recip(self.site),
            Unary(Shi, x) => ts[*x].shi(),
            Unary(Si, x) => ts[*x].si(),
            Unary(Sin, x) => ts[*x].sin(),
            Unary(Sinc, x) => ts[*x].sinc(),
            Unary(Sinh, x) => ts[*x].sinh(),
            Unary(Sqr, x) => ts[*x].sqr(),
            Unary(Sqrt, x) => ts[*x].sqrt(),
            Unary(Tan, x) => ts[*x].tan(self.site),
            Unary(Tanh, x) => ts[*x].tanh(),
            Unary(UndefAt0, x) => ts[*x].undef_at_0(),
            Unary(Zeta, x) => ts[*x].zeta(),
            Binary(Add, x, y) => &ts[*x] + &ts[*y],
            Binary(Atan2, y, x) => ts[*y].atan2(&ts[*x], self.site),
            Binary(BesselI, n, x) => ts[*n].bessel_i(&ts[*x]),
            Binary(BesselJ, n, x) => ts[*n].bessel_j(&ts[*x]),
            Binary(BesselK, n, x) => ts[*n].bessel_k(&ts[*x]),
            Binary(BesselY, n, x) => ts[*n].bessel_y(&ts[*x]),
            Binary(Div, x, y) => ts[*x].div(&ts[*y], self.site),
            Binary(GammaInc, a, x) => ts[*a].gamma_inc(&ts[*x]),
            Binary(Gcd, x, y) => ts[*x].gcd(&ts[*y], self.site),
            Binary(LambertW, k, x) => ts[*k].lambert_w(&ts[*x]),
            Binary(Lcm, x, y) => ts[*x].lcm(&ts[*y], self.site),
            // Beware the order of arguments.
            Binary(Log, b, x) => ts[*x].log(&ts[*b], self.site),
            Binary(Max, x, y) => ts[*x].max(&ts[*y]),
            Binary(Min, x, y) => ts[*x].min(&ts[*y]),
            Binary(Mod, x, y) => ts[*x].modulo(&ts[*y], self.site),
            Binary(Mul, x, y) => &ts[*x] * &ts[*y],
            Binary(Pow, x, y) => ts[*x].pow(&ts[*y], self.site),
            Binary(PowRational, x, y) => ts[*x].pow_rational(&ts[*y], self.site),
            Binary(ReSignNonnegative, x, y) => ts[*x].re_sign_nonnegative(&ts[*y], self.site),
            Binary(Sub, x, y) => &ts[*x] - &ts[*y],
            Ternary(IfThenElse, cond, t, f) => ts[*cond].if_then_else(&ts[*t], &ts[*f]),
            Ternary(MulAdd, x, y, z) => ts[*x].mul_add(&ts[*y], &ts[*z]),
            Pown(x, n) => ts[*x].pown(*n, self.site),
            Rootn(x, n) => ts[*x].rootn(*n),
            RankedMinMax(RankedMax, xs, n) => TupperIntervalSet::ranked_max(
                xs.iter().map(|x| &ts[*x]).collect(),
                &ts[*n],
                self.site,
            ),
            RankedMinMax(RankedMin, xs, n) => TupperIntervalSet::ranked_min(
                xs.iter().map(|x| &ts[*x]).collect(),
                &ts[*n],
                self.site,
            ),
            Var(_, _) => panic!("variables cannot be evaluated"),
        }
    }

    /// Evaluates the term and puts the result in the value store.
    ///
    /// Panics if the term is of the kind [`StaticTermKind::Var`].
    fn put_eval_deferred(&self, terms: &[Self], ts: &mut ValueStore<TupperIntervalSet>) {
        use {ScalarTernaryOp::*, StaticTermKind::*};

        if self.defer && !ts[self.store_index].is_unevaluated() {
            return;
        }

        match &self.kind {
            Constant(_) => {
                self.put(ts, self.eval(ts));
            }
            Unary(_, x) | Pown(x, _) | Rootn(x, _) => {
                terms[x.0 as usize].put_eval_deferred(terms, ts);
                self.put(ts, self.eval(ts));
            }
            Binary(_, x, y) => {
                terms[x.0 as usize].put_eval_deferred(terms, ts);
                terms[y.0 as usize].put_eval_deferred(terms, ts);
                self.put(ts, self.eval(ts));
            }
            Ternary(IfThenElse, cond, t, f) => {
                terms[cond.0 as usize].put_eval_deferred(terms, ts);
                let cond = &ts[*cond];
                let eval_t = cond.iter().any(|x| x.x == const_interval!(1.0, 1.0));
                let eval_f = cond.iter().any(|x| x.x == const_interval!(0.0, 0.0));
                if eval_t {
                    terms[t.0 as usize].put_eval_deferred(terms, ts);
                }
                if eval_f {
                    terms[f.0 as usize].put_eval_deferred(terms, ts);
                }
                self.put(ts, self.eval(ts));
            }
            Ternary(_, x, y, z) => {
                terms[x.0 as usize].put_eval_deferred(terms, ts);
                terms[y.0 as usize].put_eval_deferred(terms, ts);
                terms[z.0 as usize].put_eval_deferred(terms, ts);
                self.put(ts, self.eval(ts));
            }
            RankedMinMax(_, xs, n) => {
                for x in xs.iter() {
                    terms[x.0 as usize].put_eval_deferred(terms, ts);
                }
                terms[n.0 as usize].put_eval_deferred(terms, ts);
                self.put(ts, self.eval(ts));
            }
            Var(_, _) => (),
        }
    }
}

#[derive(Clone, Debug)]
pub enum StaticFormKind {
    Constant(bool),
    Atomic(RelOp, StoreIndex),
    Not(FormIndex),
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
            Constant(_) | Not(_) | And(_, _) | Or(_, _) => {
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
