use crate::{
    ast::{BinaryOp, Term, TermKind, UnaryOp},
    interval_set::TupperIntervalSet,
    visit::{Substitute, VisitMut},
};
use inari::{const_dec_interval, DecInterval};
use nom::{
    Compare, CompareResult, InputIter, InputLength, InputTake, Needed, Offset, Slice,
    UnspecializedInput,
};
use std::{
    collections::HashMap,
    lazy::SyncLazy,
    ops::{Range, RangeFrom, RangeFull, RangeTo},
    str::{CharIndices, Chars},
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Arity {
    Fixed(u8),
}

/// A definition of a constant or a function in terms of the AST.
#[derive(Clone, Debug)]
struct Def {
    arity: Arity,
    body: Term,
    associative: bool,
}

impl Def {
    fn constant(x: DecInterval) -> Self {
        Self {
            arity: Arity::Fixed(0),
            body: Term::new(TermKind::Constant(Box::new(TupperIntervalSet::from(x)))),
            associative: false,
        }
    }

    fn unary(op: UnaryOp) -> Self {
        Self {
            arity: Arity::Fixed(1),
            body: Term::new(TermKind::Unary(
                op,
                Box::new(Term::new(TermKind::Var("0".into()))),
            )),
            associative: false,
        }
    }

    fn binary(op: BinaryOp) -> Self {
        Self {
            arity: Arity::Fixed(2),
            body: Term::new(TermKind::Binary(
                op,
                Box::new(Term::new(TermKind::Var("0".into()))),
                Box::new(Term::new(TermKind::Var("1".into()))),
            )),
            associative: false,
        }
    }

    /// Sets the `associative` flag of `self` and returns it.
    ///
    /// Panics if the arity is not 2.
    fn associative(mut self) -> Self {
        assert!(self.arity == Arity::Fixed(2));
        self.associative = true;
        self
    }

    fn substitute(&self, args: Vec<Term>) -> Term {
        let mut t = self.body.clone();
        Substitute::new(args).visit_term_mut(&mut t);
        t
    }
}

/// A set of definitions of constants and functions.
#[derive(Clone, Debug)]
pub struct Context {
    defs: HashMap<String, Vec<Def>>,
}

impl Context {
    fn new() -> Self {
        Self {
            defs: HashMap::new(),
        }
    }

    fn def(mut self, name: &str, def: Def) -> Self {
        if let Some(defs) = self.defs.get_mut(name) {
            defs.push(def);
        } else {
            self.defs.insert(name.into(), vec![def]);
        }
        self
    }
}

static BUILTIN_CONTEXT: SyncLazy<Context> = SyncLazy::new(|| {
    const EULER_GAMMA: DecInterval = const_dec_interval!(0.5772156649015328, 0.5772156649015329);
    Context::new()
        .def("e", Def::constant(DecInterval::E))
        .def("gamma", Def::constant(EULER_GAMMA))
        .def("γ", Def::constant(EULER_GAMMA))
        .def("pi", Def::constant(DecInterval::PI))
        .def("π", Def::constant(DecInterval::PI))
        .def("abs", Def::unary(UnaryOp::Abs))
        .def("acos", Def::unary(UnaryOp::Acos))
        .def("acosh", Def::unary(UnaryOp::Acosh))
        .def("Ai", Def::unary(UnaryOp::AiryAi))
        .def("Ai'", Def::unary(UnaryOp::AiryAiPrime))
        .def("Bi", Def::unary(UnaryOp::AiryBi))
        .def("Bi'", Def::unary(UnaryOp::AiryBiPrime))
        .def("asin", Def::unary(UnaryOp::Asin))
        .def("asinh", Def::unary(UnaryOp::Asinh))
        .def("atan", Def::unary(UnaryOp::Atan))
        .def("atanh", Def::unary(UnaryOp::Atanh))
        .def("ceil", Def::unary(UnaryOp::Ceil))
        .def("Chi", Def::unary(UnaryOp::Chi))
        .def("Ci", Def::unary(UnaryOp::Ci))
        .def("cos", Def::unary(UnaryOp::Cos))
        .def("cosh", Def::unary(UnaryOp::Cosh))
        .def("psi", Def::unary(UnaryOp::Digamma))
        .def("ψ", Def::unary(UnaryOp::Digamma))
        .def("Ei", Def::unary(UnaryOp::Ei))
        .def("erf", Def::unary(UnaryOp::Erf))
        .def("erfc", Def::unary(UnaryOp::Erfc))
        .def("erfi", Def::unary(UnaryOp::Erfi))
        .def("exp", Def::unary(UnaryOp::Exp))
        .def("floor", Def::unary(UnaryOp::Floor))
        .def("C", Def::unary(UnaryOp::FresnelC))
        .def("S", Def::unary(UnaryOp::FresnelS))
        .def("Gamma", Def::unary(UnaryOp::Gamma))
        .def("Γ", Def::unary(UnaryOp::Gamma))
        .def("li", Def::unary(UnaryOp::Li))
        .def("ln", Def::unary(UnaryOp::Ln))
        .def("log", Def::unary(UnaryOp::Log10))
        .def("Shi", Def::unary(UnaryOp::Shi))
        .def("Si", Def::unary(UnaryOp::Si))
        .def("sign", Def::unary(UnaryOp::Sign))
        .def("sin", Def::unary(UnaryOp::Sin))
        .def("sinh", Def::unary(UnaryOp::Sinh))
        .def("sqrt", Def::unary(UnaryOp::Sqrt))
        .def("tan", Def::unary(UnaryOp::Tan))
        .def("tanh", Def::unary(UnaryOp::Tanh))
        .def("atan2", Def::binary(BinaryOp::Atan2))
        .def("I", Def::binary(BinaryOp::BesselI))
        .def("J", Def::binary(BinaryOp::BesselJ))
        .def("K", Def::binary(BinaryOp::BesselK))
        .def("Y", Def::binary(BinaryOp::BesselY))
        .def("Gamma", Def::binary(BinaryOp::GammaInc))
        .def("Γ", Def::binary(BinaryOp::GammaInc))
        .def("gcd", Def::binary(BinaryOp::Gcd).associative())
        .def("lcm", Def::binary(BinaryOp::Lcm).associative())
        .def("log", Def::binary(BinaryOp::Log))
        .def("max", Def::binary(BinaryOp::Max).associative())
        .def("min", Def::binary(BinaryOp::Min).associative())
        .def("mod", Def::binary(BinaryOp::Mod))
});

impl Context {
    pub fn builtin_context() -> &'static Self {
        &BUILTIN_CONTEXT
    }

    pub fn get_substitution(&self, name: &str, mut args: Vec<Term>) -> Option<Term> {
        let defs = self.defs.get(name)?;
        if let Some(def) = defs
            .iter()
            .find(|d| matches!(d.arity, Arity::Fixed(n) if n as usize == args.len()))
        {
            let t = def.substitute(args);
            Some(t)
        } else if let Some(def) = defs.iter().find(|d| d.associative) {
            if args.len() < 2 {
                return None;
            }

            let mut args = args.drain(..);
            let x0 = args.next().unwrap();
            let t = args.fold(x0, |t, x| def.substitute(vec![t, x]));
            Some(t)
        } else {
            None
        }
    }
}

#[derive(Clone)]
pub struct InputWithContext<'a> {
    pub i: &'a str,
    pub ctx: &'a Context,
}

impl<'a> InputWithContext<'a> {
    pub fn new(i: &'a str, ctx: &'a Context) -> Self {
        Self { i, ctx }
    }
}

impl<'a> Compare<&str> for InputWithContext<'a> {
    fn compare(&self, t: &str) -> CompareResult {
        self.i.compare(t)
    }

    fn compare_no_case(&self, t: &str) -> CompareResult {
        self.i.compare_no_case(t)
    }
}

impl<'a> InputIter for InputWithContext<'a> {
    type Item = char;
    type Iter = CharIndices<'a>;
    type IterElem = Chars<'a>;

    fn iter_indices(&self) -> Self::Iter {
        self.i.iter_indices()
    }

    fn iter_elements(&self) -> Self::IterElem {
        self.i.iter_elements()
    }

    fn position<P>(&self, predicate: P) -> Option<usize>
    where
        P: Fn(Self::Item) -> bool,
    {
        self.i.position(predicate)
    }

    fn slice_index(&self, count: usize) -> Result<usize, Needed> {
        self.i.slice_index(count)
    }
}

impl<'a> InputLength for InputWithContext<'a> {
    fn input_len(&self) -> usize {
        self.i.input_len()
    }
}

impl<'a> InputTake for InputWithContext<'a> {
    fn take(&self, count: usize) -> Self {
        InputWithContext {
            i: self.i.take(count),
            ctx: self.ctx,
        }
    }

    fn take_split(&self, count: usize) -> (Self, Self) {
        let (i0, i1) = self.i.take_split(count);
        let ctx = self.ctx;
        (
            InputWithContext { i: i0, ctx },
            InputWithContext { i: i1, ctx },
        )
    }
}

impl<'a> Offset for InputWithContext<'a> {
    fn offset(&self, second: &Self) -> usize {
        self.i.offset(second.i)
    }
}

impl<'a> PartialEq for InputWithContext<'a> {
    fn eq(&self, other: &Self) -> bool {
        self.i == other.i
    }
}

impl<'a> Slice<Range<usize>> for InputWithContext<'a> {
    fn slice(&self, range: Range<usize>) -> Self {
        InputWithContext {
            i: self.i.slice(range),
            ctx: self.ctx,
        }
    }
}

impl<'a> Slice<RangeFrom<usize>> for InputWithContext<'a> {
    fn slice(&self, range: RangeFrom<usize>) -> Self {
        InputWithContext {
            i: self.i.slice(range),
            ctx: self.ctx,
        }
    }
}

impl<'a> Slice<RangeTo<usize>> for InputWithContext<'a> {
    fn slice(&self, range: RangeTo<usize>) -> Self {
        InputWithContext {
            i: self.i.slice(range),
            ctx: self.ctx,
        }
    }
}

impl<'a> Slice<RangeFull> for InputWithContext<'a> {
    fn slice(&self, range: RangeFull) -> Self {
        InputWithContext {
            i: self.i.slice(range),
            ctx: self.ctx,
        }
    }
}

impl<'a> UnspecializedInput for InputWithContext<'a> {}
