use crate::{
    interval_set::{DecSignSet, Site, TupperIntervalSet},
    vars::{VarIndex, VarSet, VarType},
};
use inari::const_interval;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct StoreIndex(u32);

impl StoreIndex {
    pub fn new(i: u32) -> Self {
        Self(i)
    }

    pub fn get(&self) -> u32 {
        self.0
    }
}

#[derive(Clone, Debug)]
pub struct OptionalValueStore<T: Clone>(Vec<Option<T>>);

impl<T: Clone> OptionalValueStore<T> {
    pub fn new(size: usize) -> Self {
        Self(vec![None; size])
    }

    pub fn get(&self, index: StoreIndex) -> Option<&T> {
        self.0[index.0 as usize].as_ref()
    }

    pub fn get_mut(&mut self, index: StoreIndex) -> Option<&mut T> {
        self.0[index.0 as usize].as_mut()
    }

    pub fn put(&mut self, index: StoreIndex, value: T) {
        self.0[index.0 as usize] = Some(value);
    }

    pub fn remove(&mut self, index: StoreIndex) -> Option<T> {
        self.0[index.0 as usize].take()
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
    LnGamma,
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
    ImSinc,
    ImUndefAt0,
    ImZeta,
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
    ReSinc,
    ReUndefAt0,
    ReZeta,
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
    RankedMinMax(RankedMinMaxOp, Vec<StoreIndex>, StoreIndex),
}

/// A term in a cache-efficient representation.
#[derive(Clone, Debug)]
pub struct StaticTerm {
    pub kind: StaticTermKind,
    pub site: Option<Site>,
    pub store_index: StoreIndex,
    pub vars: VarSet,
}

impl StaticTerm {
    pub fn put(&self, store: &mut OptionalValueStore<TupperIntervalSet>, value: TupperIntervalSet) {
        store.put(self.store_index, value);
    }

    /// Evaluates the term and puts the result in the value store.
    ///
    /// Panics if the term is of the kind [`StaticTermKind::Var`].
    pub fn put_eval(&self, terms: &[Self], ts: &mut OptionalValueStore<TupperIntervalSet>) {
        use {
            RankedMinMaxOp::*, ScalarBinaryOp::*, ScalarTernaryOp::*, ScalarUnaryOp::*,
            StaticTermKind::*,
        };

        if ts.get(self.store_index).is_some() {
            return;
        }

        match &self.kind {
            Unary(_, x) | Pown(x, _) | Rootn(x, _) => {
                if ts.get(*x).is_none() {
                    terms[x.get() as usize].put_eval(terms, ts);
                }
            }
            Binary(_, x, y) => {
                if ts.get(*x).is_none() {
                    terms[x.get() as usize].put_eval(terms, ts);
                }
                if ts.get(*y).is_none() {
                    terms[y.get() as usize].put_eval(terms, ts);
                }
            }
            Ternary(IfThenElse, cond, t, f) => {
                if ts.get(*cond).is_none() {
                    terms[cond.get() as usize].put_eval(terms, ts);
                }
                let c = &ts.get(*cond).unwrap();
                let eval_t = c.iter().any(|x| x.x == const_interval!(1.0, 1.0));
                let eval_f = c.iter().any(|x| x.x == const_interval!(0.0, 0.0));
                if eval_t && ts.get(*t).is_none() {
                    terms[t.get() as usize].put_eval(terms, ts);
                }
                if eval_f && ts.get(*f).is_none() {
                    terms[f.get() as usize].put_eval(terms, ts);
                }
            }
            Ternary(_, x, y, z) => {
                if ts.get(*x).is_none() {
                    terms[x.get() as usize].put_eval(terms, ts);
                }
                if ts.get(*y).is_none() {
                    terms[y.get() as usize].put_eval(terms, ts);
                }
                if ts.get(*z).is_none() {
                    terms[z.get() as usize].put_eval(terms, ts);
                }
            }
            RankedMinMax(_, xs, n) => {
                for x in xs {
                    if ts.get(*x).is_none() {
                        terms[x.get() as usize].put_eval(terms, ts);
                    }
                }
                if ts.get(*n).is_none() {
                    terms[n.get() as usize].put_eval(terms, ts);
                }
            }
            Constant(_) | Var(_, _) => (),
        }

        let dummy_interval_set = TupperIntervalSet::new();

        match &self.kind {
            Constant(x) => self.put(ts, *x.clone()),
            Unary(Abs, x) => self.put(ts, ts.get(*x).unwrap().abs()),
            Unary(Acos, x) => self.put(ts, ts.get(*x).unwrap().acos()),
            Unary(Acosh, x) => self.put(ts, ts.get(*x).unwrap().acosh()),
            Unary(AiryAi, x) => self.put(ts, ts.get(*x).unwrap().airy_ai()),
            Unary(AiryAiPrime, x) => self.put(ts, ts.get(*x).unwrap().airy_ai_prime()),
            Unary(AiryBi, x) => self.put(ts, ts.get(*x).unwrap().airy_bi()),
            Unary(AiryBiPrime, x) => self.put(ts, ts.get(*x).unwrap().airy_bi_prime()),
            Unary(Asin, x) => self.put(ts, ts.get(*x).unwrap().asin()),
            Unary(Asinh, x) => self.put(ts, ts.get(*x).unwrap().asinh()),
            Unary(Atan, x) => self.put(ts, ts.get(*x).unwrap().atan()),
            Unary(Atanh, x) => self.put(ts, ts.get(*x).unwrap().atanh()),
            Unary(BooleEqZero, x) => self.put(ts, ts.get(*x).unwrap().boole_eq_zero(self.site)),
            Unary(BooleLeZero, x) => self.put(ts, ts.get(*x).unwrap().boole_le_zero(self.site)),
            Unary(BooleLtZero, x) => self.put(ts, ts.get(*x).unwrap().boole_lt_zero(self.site)),
            Unary(Ceil, x) => self.put(ts, ts.get(*x).unwrap().ceil(self.site)),
            Unary(Chi, x) => self.put(ts, ts.get(*x).unwrap().chi()),
            Unary(Ci, x) => self.put(ts, ts.get(*x).unwrap().ci()),
            Unary(Cos, x) => self.put(ts, ts.get(*x).unwrap().cos()),
            Unary(Cosh, x) => self.put(ts, ts.get(*x).unwrap().cosh()),
            Unary(Digamma, x) => self.put(ts, ts.get(*x).unwrap().digamma(self.site)),
            Unary(Ei, x) => self.put(ts, ts.get(*x).unwrap().ei()),
            Unary(EllipticE, x) => self.put(ts, ts.get(*x).unwrap().elliptic_e()),
            Unary(EllipticK, x) => self.put(ts, ts.get(*x).unwrap().elliptic_k()),
            Unary(Erf, x) => self.put(ts, ts.get(*x).unwrap().erf()),
            Unary(Erfc, x) => self.put(ts, ts.get(*x).unwrap().erfc()),
            Unary(Erfi, x) => self.put(ts, ts.get(*x).unwrap().erfi()),
            Unary(Exp, x) => self.put(ts, ts.get(*x).unwrap().exp()),
            Unary(Floor, x) => self.put(ts, ts.get(*x).unwrap().floor(self.site)),
            Unary(FresnelC, x) => self.put(ts, ts.get(*x).unwrap().fresnel_c()),
            Unary(FresnelS, x) => self.put(ts, ts.get(*x).unwrap().fresnel_s()),
            Unary(Gamma, x) => self.put(ts, ts.get(*x).unwrap().gamma(self.site)),
            Unary(InverseErf, x) => self.put(ts, ts.get(*x).unwrap().inverse_erf()),
            Unary(InverseErfc, x) => self.put(ts, ts.get(*x).unwrap().inverse_erfc()),
            Unary(Li, x) => self.put(ts, ts.get(*x).unwrap().li()),
            Unary(Ln, x) => self.put(ts, ts.get(*x).unwrap().ln()),
            Unary(LnGamma, x) => self.put(ts, ts.get(*x).unwrap().ln_gamma()),
            Unary(Neg, x) => self.put(ts, -ts.get(*x).unwrap()),
            Unary(Recip, x) => self.put(ts, ts.get(*x).unwrap().recip(self.site)),
            Unary(Shi, x) => self.put(ts, ts.get(*x).unwrap().shi()),
            Unary(Si, x) => self.put(ts, ts.get(*x).unwrap().si()),
            Unary(Sin, x) => self.put(ts, ts.get(*x).unwrap().sin()),
            Unary(Sinc, x) => self.put(ts, ts.get(*x).unwrap().sinc()),
            Unary(Sinh, x) => self.put(ts, ts.get(*x).unwrap().sinh()),
            Unary(Sqr, x) => self.put(ts, ts.get(*x).unwrap().sqr()),
            Unary(Sqrt, x) => self.put(ts, ts.get(*x).unwrap().sqrt()),
            Unary(Tan, x) => self.put(ts, ts.get(*x).unwrap().tan(self.site)),
            Unary(Tanh, x) => self.put(ts, ts.get(*x).unwrap().tanh()),
            Unary(UndefAt0, x) => self.put(ts, ts.get(*x).unwrap().undef_at_0()),
            Unary(Zeta, x) => self.put(ts, ts.get(*x).unwrap().zeta()),
            Binary(Add, x, y) => self.put(ts, ts.get(*x).unwrap() + ts.get(*y).unwrap()),
            Binary(Atan2, y, x) => self.put(
                ts,
                ts.get(*y).unwrap().atan2(ts.get(*x).unwrap(), self.site),
            ),
            Binary(BesselI, n, x) => {
                self.put(ts, ts.get(*n).unwrap().bessel_i(ts.get(*x).unwrap()))
            }
            Binary(BesselJ, n, x) => {
                self.put(ts, ts.get(*n).unwrap().bessel_j(ts.get(*x).unwrap()))
            }
            Binary(BesselK, n, x) => {
                self.put(ts, ts.get(*n).unwrap().bessel_k(ts.get(*x).unwrap()))
            }
            Binary(BesselY, n, x) => {
                self.put(ts, ts.get(*n).unwrap().bessel_y(ts.get(*x).unwrap()))
            }
            Binary(Div, x, y) => {
                self.put(ts, ts.get(*x).unwrap().div(ts.get(*y).unwrap(), self.site))
            }
            Binary(GammaInc, a, x) => {
                self.put(ts, ts.get(*a).unwrap().gamma_inc(ts.get(*x).unwrap()))
            }
            Binary(Gcd, x, y) => {
                self.put(ts, ts.get(*x).unwrap().gcd(ts.get(*y).unwrap(), self.site))
            }
            Binary(ImSinc, re_x, im_x) => {
                self.put(ts, ts.get(*re_x).unwrap().im_sinc(ts.get(*im_x).unwrap()))
            }
            Binary(ImUndefAt0, re_x, im_x) => self.put(
                ts,
                ts.get(*re_x).unwrap().im_undef_at_0(ts.get(*im_x).unwrap()),
            ),
            Binary(ImZeta, re_x, im_x) => {
                self.put(ts, ts.get(*re_x).unwrap().im_zeta(ts.get(*im_x).unwrap()))
            }
            Binary(LambertW, k, x) => {
                self.put(ts, ts.get(*k).unwrap().lambert_w(ts.get(*x).unwrap()))
            }
            Binary(Lcm, x, y) => {
                self.put(ts, ts.get(*x).unwrap().lcm(ts.get(*y).unwrap(), self.site))
            }
            // Beware the order of arguments.
            Binary(Log, b, x) => {
                self.put(ts, ts.get(*x).unwrap().log(ts.get(*b).unwrap(), self.site))
            }
            Binary(Max, x, y) => self.put(ts, ts.get(*x).unwrap().max(ts.get(*y).unwrap())),
            Binary(Min, x, y) => self.put(ts, ts.get(*x).unwrap().min(ts.get(*y).unwrap())),
            Binary(Mod, x, y) => self.put(
                ts,
                ts.get(*x).unwrap().modulo(ts.get(*y).unwrap(), self.site),
            ),
            Binary(Mul, x, y) => self.put(ts, ts.get(*x).unwrap() * ts.get(*y).unwrap()),
            Binary(Pow, x, y) => {
                self.put(ts, ts.get(*x).unwrap().pow(ts.get(*y).unwrap(), self.site))
            }
            Binary(PowRational, x, y) => self.put(
                ts,
                ts.get(*x)
                    .unwrap()
                    .pow_rational(ts.get(*y).unwrap(), self.site),
            ),
            Binary(ReSignNonnegative, x, y) => self.put(
                ts,
                ts.get(*x)
                    .unwrap()
                    .re_sign_nonnegative(ts.get(*y).unwrap(), self.site),
            ),
            Binary(ReSinc, re_x, im_x) => {
                self.put(ts, ts.get(*re_x).unwrap().re_sinc(ts.get(*im_x).unwrap()))
            }
            Binary(ReUndefAt0, re_x, im_x) => self.put(
                ts,
                ts.get(*re_x).unwrap().re_undef_at_0(ts.get(*im_x).unwrap()),
            ),
            Binary(ReZeta, re_x, im_x) => {
                self.put(ts, ts.get(*re_x).unwrap().re_zeta(ts.get(*im_x).unwrap()))
            }
            Binary(Sub, x, y) => self.put(ts, ts.get(*x).unwrap() - ts.get(*y).unwrap()),
            Ternary(IfThenElse, cond, t, f) => self.put(
                ts,
                ts.get(*cond).unwrap().if_then_else(
                    ts.get(*t).unwrap_or(&dummy_interval_set),
                    ts.get(*f).unwrap_or(&dummy_interval_set),
                ),
            ),
            Ternary(MulAdd, x, y, z) => self.put(
                ts,
                ts.get(*x)
                    .unwrap()
                    .mul_add(ts.get(*y).unwrap(), ts.get(*z).unwrap()),
            ),
            Pown(x, n) => self.put(ts, ts.get(*x).unwrap().pown(*n, self.site)),
            Rootn(x, n) => self.put(ts, ts.get(*x).unwrap().rootn(*n)),
            RankedMinMax(RankedMax, xs, n) => {
                self.put(
                    ts,
                    TupperIntervalSet::ranked_max(
                        xs.iter().map(|x| ts.get(*x).unwrap()).collect(),
                        ts.get(*n).unwrap(),
                        self.site,
                    ),
                );
            }
            RankedMinMax(RankedMin, xs, n) => {
                self.put(
                    ts,
                    TupperIntervalSet::ranked_min(
                        xs.iter().map(|x| ts.get(*x).unwrap()).collect(),
                        ts.get(*n).unwrap(),
                        self.site,
                    ),
                );
            }
            Var(_, _) => panic!("variables cannot be evaluated"),
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
    pub fn eval(&self, ts: &OptionalValueStore<TupperIntervalSet>) -> DecSignSet {
        use {RelOp::*, StaticFormKind::*};
        match &self.kind {
            Atomic(EqZero, x) => ts.get(*x).unwrap().eq_zero(),
            Atomic(LeZero, x) => ts.get(*x).unwrap().le_zero(),
            Atomic(LtZero, x) => ts.get(*x).unwrap().lt_zero(),
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
        assert_eq!(size_of::<StaticTermKind>(), 32);
        assert_eq!(size_of::<StaticTerm>(), 40);
        assert_eq!(size_of::<StaticFormKind>(), 12);
        assert_eq!(size_of::<StaticForm>(), 12);
    }
}
