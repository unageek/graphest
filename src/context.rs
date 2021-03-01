use crate::{
    ast::{Term, TermKind},
    interval_set::TupperIntervalSet,
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

const EULER_GAMMA: DecInterval = const_dec_interval!(0.5772156649015328, 0.5772156649015329);

const BUILTIN_NAMES: &[&str] = &[
    "acos", "acosh", "Ai", "asin", "asinh", "atan", "atan2", "atanh", "Bi", "C", "ceil", "Chi",
    "Ci", "cos", "cosh", "Ei", "erf", "erfc", "erfi", "exp", "floor", "Gamma", "Γ", "gcd", "I",
    "J", "K", "lcm", "li", "ln", "log", "max", "min", "mod", "psi", "ψ", "S", "Shi", "Si", "sign",
    "sin", "sinh", "sqrt", "tan", "tanh", "Y",
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Arity {
    Fixed(u8),
    Variadic,
}

#[derive(Clone, Debug)]
struct Definition {
    pub arity: Arity,
    pub body: Term,
}

impl Definition {
    fn constant(x: DecInterval) -> Self {
        Self {
            arity: Arity::Fixed(0),
            body: Term::new(TermKind::Constant(Box::new(TupperIntervalSet::from(x)))),
        }
    }
}

/// A set of definitions of constants and functions.
///
/// Builtin functions have dummy definitions at the moment.
#[derive(Clone, Debug)]
pub struct Context {
    defs: HashMap<String, Vec<Definition>>,
}

static BUILTIN_CONTEXT: SyncLazy<Context> = SyncLazy::new(|| {
    let mut defs = HashMap::new();
    defs.insert("e".into(), vec![Definition::constant(DecInterval::E)]);
    defs.insert("gamma".into(), vec![Definition::constant(EULER_GAMMA)]);
    defs.insert("γ".into(), vec![Definition::constant(EULER_GAMMA)]);
    defs.insert("pi".into(), vec![Definition::constant(DecInterval::PI)]);
    defs.insert("π".into(), vec![Definition::constant(DecInterval::PI)]);
    for name in BUILTIN_NAMES {
        defs.insert(
            name.to_string(),
            vec![Definition {
                arity: Arity::Variadic,
                body: Term::default(),
            }],
        );
    }
    Context { defs }
});

impl Context {
    pub fn builtin_context() -> &'static Self {
        &BUILTIN_CONTEXT
    }

    // TODO: Define something like `TermKind::Slot` and do substitution.
    pub fn get_substituted(&self, name: &str, args: Vec<&Term>) -> Option<Term> {
        let def = self
            .defs
            .get(name)?
            .iter()
            .find(|d| matches!(d.arity,Arity::Fixed(n) if n as usize == args.len()))?;
        Some(def.body.clone())
    }

    pub fn is_defined(&self, name: &str) -> bool {
        self.defs.contains_key(name)
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
