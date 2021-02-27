use nom::{
    Compare, CompareResult, InputIter, InputLength, InputTake, Needed, Offset, Slice,
    UnspecializedInput,
};
use std::{
    collections::HashSet,
    ops::{Range, RangeFrom, RangeFull, RangeTo},
    str::{CharIndices, Chars},
};

/// A set of definitions of constants and functions.
///
/// It only contains the names of builtin functions at the moment.
pub struct Context {
    defs: HashSet<String>,
}

impl Context {
    pub fn new() -> Self {
        let defs = [
            "acos", "acosh", "Ai", "asin", "asinh", "atan", "atan2", "atanh", "Bi", "C", "ceil",
            "Chi", "Ci", "cos", "cosh", "e", "Ei", "erf", "erfc", "erfi", "exp", "floor", "Gamma",
            "Γ", "gamma", "γ", "gcd", "I", "J", "K", "lcm", "li", "ln", "log", "max", "min", "mod",
            "pi", "π", "psi", "ψ", "S", "Shi", "Si", "sign", "sin", "sinh", "sqrt", "tan", "tanh",
            "Y",
        ]
        .iter()
        .map(|&s| s.into())
        .collect::<HashSet<String>>();
        Self { defs }
    }

    pub fn is_defined(&self, name: &str) -> bool {
        self.defs.contains(name)
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
