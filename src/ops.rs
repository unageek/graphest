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

pub type TermIndex = u32;
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
    RankedMax,
    RankedMin,
    Sub,
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

/// TODO: Store `StoreIndex` instead of `TermIndex`.
#[derive(Clone, Debug)]
pub enum StaticTermKind {
    // == Scalar-valued terms ==
    Constant(Box<TupperIntervalSet>),
    X,
    Y,
    Unary(ScalarUnaryOp, TermIndex),
    Binary(ScalarBinaryOp, TermIndex, TermIndex),
    Pown(TermIndex, i32),
    // == Others ==
    // Box the `Vec` to keep the enum small.
    // Operations involving lists are relatively rare, so it would be worth the cost of the extra indirection.
    #[allow(clippy::box_vec)]
    List(Box<Vec<TermIndex>>),
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
    pub fn get<'a, T: Clone>(&self, store: &'a ValueStore<T>) -> &'a T {
        &store.get(self.store_index)
    }

    pub fn put<T: Clone>(&self, store: &mut ValueStore<T>, value: T) {
        store.put(self.store_index, value);
    }

    /// Evaluates the term and puts the result in the value store.
    ///
    /// Panics if the term is of the kind [`StaticTermKind::X`], [`StaticTermKind::Y`]
    /// or [`StaticTermKind::List`].
    pub fn put_eval(&self, terms: &[StaticTerm], ts: &mut ValueStore<TupperIntervalSet>) {
        use {ScalarBinaryOp::*, ScalarUnaryOp::*, StaticTermKind::*};
        match &self.kind {
            Constant(x) => self.put(ts, *x.clone()),
            Unary(Abs, x) => self.put(ts, terms[*x as usize].get(ts).abs()),
            Unary(Acos, x) => self.put(ts, terms[*x as usize].get(ts).acos()),
            Unary(Acosh, x) => self.put(ts, terms[*x as usize].get(ts).acosh()),
            Unary(AiryAi, x) => self.put(ts, terms[*x as usize].get(ts).airy_ai()),
            Unary(AiryAiPrime, x) => self.put(ts, terms[*x as usize].get(ts).airy_ai_prime()),
            Unary(AiryBi, x) => self.put(ts, terms[*x as usize].get(ts).airy_bi()),
            Unary(AiryBiPrime, x) => self.put(ts, terms[*x as usize].get(ts).airy_bi_prime()),
            Unary(Asin, x) => self.put(ts, terms[*x as usize].get(ts).asin()),
            Unary(Asinh, x) => self.put(ts, terms[*x as usize].get(ts).asinh()),
            Unary(Atan, x) => self.put(ts, terms[*x as usize].get(ts).atan()),
            Unary(Atanh, x) => self.put(ts, terms[*x as usize].get(ts).atanh()),
            Unary(Ceil, x) => self.put(ts, terms[*x as usize].get(ts).ceil(self.site)),
            Unary(Chi, x) => self.put(ts, terms[*x as usize].get(ts).chi()),
            Unary(Ci, x) => self.put(ts, terms[*x as usize].get(ts).ci()),
            Unary(Cos, x) => self.put(ts, terms[*x as usize].get(ts).cos()),
            Unary(Cosh, x) => self.put(ts, terms[*x as usize].get(ts).cosh()),
            Unary(Digamma, x) => self.put(ts, terms[*x as usize].get(ts).digamma(self.site)),
            Unary(Ei, x) => self.put(ts, terms[*x as usize].get(ts).ei()),
            Unary(Erf, x) => self.put(ts, terms[*x as usize].get(ts).erf()),
            Unary(Erfc, x) => self.put(ts, terms[*x as usize].get(ts).erfc()),
            Unary(Erfi, x) => self.put(ts, terms[*x as usize].get(ts).erfi()),
            Unary(Exp, x) => self.put(ts, terms[*x as usize].get(ts).exp()),
            Unary(Exp10, x) => self.put(ts, terms[*x as usize].get(ts).exp10()),
            Unary(Exp2, x) => self.put(ts, terms[*x as usize].get(ts).exp2()),
            Unary(Floor, x) => self.put(ts, terms[*x as usize].get(ts).floor(self.site)),
            Unary(FresnelC, x) => self.put(ts, terms[*x as usize].get(ts).fresnel_c()),
            Unary(FresnelS, x) => self.put(ts, terms[*x as usize].get(ts).fresnel_s()),
            Unary(Gamma, x) => self.put(ts, terms[*x as usize].get(ts).gamma(self.site)),
            Unary(Li, x) => self.put(ts, terms[*x as usize].get(ts).li()),
            Unary(Ln, x) => self.put(ts, terms[*x as usize].get(ts).ln()),
            Unary(Log10, x) => self.put(ts, terms[*x as usize].get(ts).log10()),
            Unary(Neg, x) => self.put(ts, -terms[*x as usize].get(ts)),
            Unary(One, x) => self.put(ts, terms[*x as usize].get(ts).one()),
            Unary(Recip, x) => self.put(ts, terms[*x as usize].get(ts).recip(self.site)),
            Unary(Shi, x) => self.put(ts, terms[*x as usize].get(ts).shi()),
            Unary(Si, x) => self.put(ts, terms[*x as usize].get(ts).si()),
            Unary(Sign, x) => self.put(ts, terms[*x as usize].get(ts).sign(self.site)),
            Unary(Sin, x) => self.put(ts, terms[*x as usize].get(ts).sin()),
            Unary(Sinc, x) => self.put(ts, terms[*x as usize].get(ts).sinc()),
            Unary(Sinh, x) => self.put(ts, terms[*x as usize].get(ts).sinh()),
            Unary(Sqr, x) => self.put(ts, terms[*x as usize].get(ts).sqr()),
            Unary(Sqrt, x) => self.put(ts, terms[*x as usize].get(ts).sqrt()),
            Unary(Tan, x) => self.put(ts, terms[*x as usize].get(ts).tan(self.site)),
            Unary(Tanh, x) => self.put(ts, terms[*x as usize].get(ts).tanh()),
            Unary(UndefAt0, x) => self.put(ts, terms[*x as usize].get(ts).undef_at_0()),
            Binary(Add, x, y) => {
                self.put(ts, terms[*x as usize].get(ts) + terms[*y as usize].get(ts))
            }
            Binary(Atan2, y, x) => self.put(
                ts,
                terms[*y as usize]
                    .get(ts)
                    .atan2(terms[*x as usize].get(ts), self.site),
            ),
            Binary(BesselI, n, x) => self.put(
                ts,
                terms[*n as usize]
                    .get(ts)
                    .bessel_i(terms[*x as usize].get(ts)),
            ),
            Binary(BesselJ, n, x) => self.put(
                ts,
                terms[*n as usize]
                    .get(ts)
                    .bessel_j(terms[*x as usize].get(ts)),
            ),
            Binary(BesselK, n, x) => self.put(
                ts,
                terms[*n as usize]
                    .get(ts)
                    .bessel_k(terms[*x as usize].get(ts)),
            ),
            Binary(BesselY, n, x) => self.put(
                ts,
                terms[*n as usize]
                    .get(ts)
                    .bessel_y(terms[*x as usize].get(ts)),
            ),
            Binary(Div, x, y) => self.put(
                ts,
                terms[*x as usize]
                    .get(ts)
                    .div(terms[*y as usize].get(ts), self.site),
            ),
            Binary(GammaInc, a, x) => self.put(
                ts,
                terms[*a as usize]
                    .get(ts)
                    .gamma_inc(terms[*x as usize].get(ts)),
            ),
            Binary(Gcd, x, y) => self.put(
                ts,
                terms[*x as usize]
                    .get(ts)
                    .gcd(terms[*y as usize].get(ts), self.site),
            ),
            Binary(Lcm, x, y) => self.put(
                ts,
                terms[*x as usize]
                    .get(ts)
                    .lcm(terms[*y as usize].get(ts), self.site),
            ),
            // Beware the order of arguments.
            Binary(Log, b, x) => self.put(
                ts,
                terms[*x as usize]
                    .get(ts)
                    .log(terms[*b as usize].get(ts), self.site),
            ),
            Binary(Max, x, y) => self.put(
                ts,
                terms[*x as usize].get(ts).max(terms[*y as usize].get(ts)),
            ),
            Binary(Min, x, y) => self.put(
                ts,
                terms[*x as usize].get(ts).min(terms[*y as usize].get(ts)),
            ),
            Binary(Mod, x, y) => self.put(
                ts,
                terms[*x as usize]
                    .get(ts)
                    .rem_euclid(terms[*y as usize].get(ts), self.site),
            ),
            Binary(Mul, x, y) => {
                self.put(ts, terms[*x as usize].get(ts) * terms[*y as usize].get(ts))
            }
            Binary(Pow, x, y) => self.put(
                ts,
                terms[*x as usize]
                    .get(ts)
                    .pow(terms[*y as usize].get(ts), self.site),
            ),
            Binary(RankedMax, xs, n) => {
                if let List(xs) = &terms[*xs as usize].kind {
                    self.put(
                        ts,
                        TupperIntervalSet::ranked_max(
                            xs.iter().map(|x| terms[*x as usize].get(ts)).collect(),
                            terms[*n as usize].get(ts),
                            self.site,
                        ),
                    );
                } else {
                    panic!("a list is expected");
                }
            }
            Binary(RankedMin, xs, n) => {
                if let List(xs) = &terms[*xs as usize].kind {
                    self.put(
                        ts,
                        TupperIntervalSet::ranked_min(
                            xs.iter().map(|x| terms[*x as usize].get(ts)).collect(),
                            terms[*n as usize].get(ts),
                            self.site,
                        ),
                    );
                } else {
                    panic!("a list is expected");
                }
            }
            Binary(Sub, x, y) => {
                self.put(ts, terms[*x as usize].get(ts) - terms[*y as usize].get(ts))
            }
            Pown(x, y) => self.put(ts, terms[*x as usize].get(ts).pown(*y, self.site)),
            List(_) => (),
            X | Y => panic!("this term cannot be evaluated"),
        }
    }
}

#[derive(Clone, Debug)]
pub enum StaticFormKind {
    // TODO: Store `StoreIndex` instead of `TermIndex`.
    Atomic(RelOp, TermIndex, TermIndex),
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
    pub fn eval(&self, terms: &[StaticTerm], ts: &ValueStore<TupperIntervalSet>) -> DecSignSet {
        use {RelOp::*, StaticFormKind::*};
        match &self.kind {
            Atomic(Eq, x, y) => terms[*x as usize].get(ts).eq(terms[*y as usize].get(ts)),
            Atomic(Ge, x, y) => terms[*x as usize].get(ts).ge(terms[*y as usize].get(ts)),
            Atomic(Gt, x, y) => terms[*x as usize].get(ts).gt(terms[*y as usize].get(ts)),
            Atomic(Le, x, y) => terms[*x as usize].get(ts).le(terms[*y as usize].get(ts)),
            Atomic(Lt, x, y) => terms[*x as usize].get(ts).lt(terms[*y as usize].get(ts)),
            Atomic(Neq, x, y) => terms[*x as usize].get(ts).neq(terms[*y as usize].get(ts)),
            Atomic(Nge, x, y) => terms[*x as usize].get(ts).nge(terms[*y as usize].get(ts)),
            Atomic(Ngt, x, y) => terms[*x as usize].get(ts).ngt(terms[*y as usize].get(ts)),
            Atomic(Nle, x, y) => terms[*x as usize].get(ts).nle(terms[*y as usize].get(ts)),
            Atomic(Nlt, x, y) => terms[*x as usize].get(ts).nlt(terms[*y as usize].get(ts)),
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
