use crate::{
    ast::{BinaryOp, Expr, TernaryOp, UnaryOp},
    real::Real,
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

/// A definition of a constant or a function in terms of the AST.
#[derive(Clone, Debug)]
enum Def {
    Constant {
        body: Expr,
    },
    Function {
        arity: usize,
        body: Expr,
        left_associative: bool,
    },
}

impl Def {
    /// Creates a definition of a Boolean constant.
    fn bool_constant(x: bool) -> Self {
        Self::Constant {
            body: Expr::bool_constant(x),
        }
    }

    /// Creates a definition of a real constant.
    fn constant(x: Real) -> Self {
        Self::Constant {
            body: Expr::constant(x),
        }
    }

    /// Creates a definition of a unary function.
    fn unary(op: UnaryOp) -> Self {
        Self::Function {
            arity: 1,
            body: Expr::unary(op, box Expr::var("#0")),
            left_associative: false,
        }
    }

    /// Creates a definition of a binary function.
    fn binary(op: BinaryOp) -> Self {
        Self::Function {
            arity: 2,
            body: Expr::binary(op, box Expr::var("#0"), box Expr::var("#1")),
            left_associative: false,
        }
    }

    /// Marks the binary function as left-associative and returns `self`.
    ///
    /// Panics if `self` is not a function of arity 2.
    fn left_associative(mut self) -> Self {
        match self {
            Def::Function {
                arity,
                ref mut left_associative,
                ..
            } if arity == 2 => {
                *left_associative = true;
            }
            _ => panic!(),
        }
        self
    }

    /// Applies the function to the arguments.
    ///
    /// Panics if `self` is not a function or the number of arguments does not match the arity.
    fn apply(&self, args: Vec<Expr>) -> Expr {
        match self {
            Def::Function { arity, body, .. } => {
                assert!(*arity == args.len());
                let mut t = body.clone();
                Substitute::new(args).visit_expr_mut(&mut t);
                t
            }
            _ => panic!(),
        }
    }
}

/// A set of definitions of constants and functions.
#[derive(Clone, Debug)]
pub struct Context {
    defs: HashMap<String, Vec<Def>>,
}

impl Context {
    /// Creates an empty context.
    fn new() -> Self {
        Self {
            defs: HashMap::new(),
        }
    }

    /// Appends a definition to the context and returns `self`.
    fn def(mut self, name: &str, def: Def) -> Self {
        if let Some(defs) = self.defs.get_mut(name) {
            defs.push(def);
        } else {
            self.defs.insert(name.into(), vec![def]);
        }
        self
    }
}

/// The context that is used when parsing relations.
static BUILTIN_CONTEXT: SyncLazy<Context> = SyncLazy::new(|| {
    use {BinaryOp::*, TernaryOp::*, UnaryOp::*};

    const EULER_GAMMA: DecInterval = const_dec_interval!(0.5772156649015328, 0.5772156649015329);
    Context::new()
        .def("false", Def::bool_constant(false))
        .def("true", Def::bool_constant(true))
        .def("e", Def::constant(DecInterval::E.into()))
        .def("gamma", Def::constant(EULER_GAMMA.into()))
        .def("γ", Def::constant(EULER_GAMMA.into()))
        .def("pi", Def::constant(DecInterval::PI.into()))
        .def("π", Def::constant(DecInterval::PI.into()))
        .def(
            "i",
            Def::Constant {
                body: Expr::binary(Complex, box Expr::zero(), box Expr::one()),
            },
        )
        .def("abs", Def::unary(Abs))
        .def("acos", Def::unary(Acos))
        .def("acosh", Def::unary(Acosh))
        .def("Ai", Def::unary(AiryAi))
        .def("Ai'", Def::unary(AiryAiPrime))
        .def("Bi", Def::unary(AiryBi))
        .def("Bi'", Def::unary(AiryBiPrime))
        .def("arg", Def::unary(Arg))
        .def("asin", Def::unary(Asin))
        .def("asinh", Def::unary(Asinh))
        .def("atan", Def::unary(Atan))
        .def("atanh", Def::unary(Atanh))
        .def("ceil", Def::unary(Ceil))
        .def("Chi", Def::unary(Chi))
        .def("Ci", Def::unary(Ci))
        .def("~", Def::unary(Conj))
        .def("cos", Def::unary(Cos))
        .def("cosh", Def::unary(Cosh))
        .def("psi", Def::unary(Digamma))
        .def("ψ", Def::unary(Digamma))
        .def("Ei", Def::unary(Ei))
        .def("E", Def::unary(EllipticE))
        .def("K", Def::unary(EllipticK))
        .def("erf", Def::unary(Erf))
        .def("erfc", Def::unary(Erfc))
        .def("erfi", Def::unary(Erfi))
        .def("exp", Def::unary(Exp))
        .def("floor", Def::unary(Floor))
        .def("C", Def::unary(FresnelC))
        .def("S", Def::unary(FresnelS))
        .def("Gamma", Def::unary(Gamma))
        .def("Γ", Def::unary(Gamma))
        .def("Im", Def::unary(Im))
        .def("li", Def::unary(Li))
        .def("ln", Def::unary(Ln))
        .def("-", Def::unary(Neg))
        .def("!", Def::unary(Not))
        .def("Re", Def::unary(Re))
        .def("Shi", Def::unary(Shi))
        .def("Si", Def::unary(Si))
        .def("sgn", Def::unary(Sign))
        .def("sign", Def::unary(Sign))
        .def("sin", Def::unary(Sin))
        .def("sinh", Def::unary(Sinh))
        .def("sqrt", Def::unary(Sqrt))
        .def("tan", Def::unary(Tan))
        .def("tanh", Def::unary(Tanh))
        .def("+", Def::binary(Add))
        .def("&&", Def::binary(And))
        .def("atan2", Def::binary(Atan2))
        .def("I", Def::binary(BesselI))
        .def("J", Def::binary(BesselJ))
        .def("K", Def::binary(BesselK))
        .def("Y", Def::binary(BesselY))
        .def("/", Def::binary(Div))
        .def("=", Def::binary(Eq))
        .def("Gamma", Def::binary(GammaInc))
        .def("Γ", Def::binary(GammaInc))
        .def("gcd", Def::binary(Gcd).left_associative())
        .def(">=", Def::binary(Ge))
        .def(">", Def::binary(Gt))
        .def("lcm", Def::binary(Lcm).left_associative())
        .def("<=", Def::binary(Le))
        .def("log", Def::binary(Log))
        .def("<", Def::binary(Lt))
        .def("max", Def::binary(Max).left_associative())
        .def("min", Def::binary(Min).left_associative())
        .def("mod", Def::binary(Mod))
        .def("*", Def::binary(Mul))
        .def("||", Def::binary(Or))
        .def("^", Def::binary(Pow))
        .def("ranked_max", Def::binary(RankedMax))
        .def("ranked_min", Def::binary(RankedMin))
        .def("-", Def::binary(Sub))
        .def(
            "if",
            Def::Function {
                arity: 3,
                body: {
                    Expr::ternary(
                        IfThenElse,
                        box Expr::unary(Boole, box Expr::var("#0")),
                        box Expr::var("#1"),
                        box Expr::var("#2"),
                    )
                },
                left_associative: false,
            },
        )
});

impl Context {
    /// Returns the context with the builtin definitions.
    pub fn builtin_context() -> &'static Self {
        &BUILTIN_CONTEXT
    }

    pub fn apply(&self, name: &str, args: Vec<Expr>) -> Option<Expr> {
        for d in self.defs.get(name)? {
            match *d {
                Def::Function { arity, .. } if args.len() == arity => {
                    let t = d.apply(args);
                    return Some(t);
                }
                Def::Function {
                    left_associative, ..
                } if left_associative && args.len() >= 2 => {
                    let mut it = args.into_iter();
                    let x0 = it.next().unwrap();
                    let t = it.fold(x0, |t, x| d.apply(vec![t, x]));
                    return Some(t);
                }
                _ => (),
            }
        }
        None
    }

    pub fn get_constant(&self, name: &str) -> Option<Expr> {
        for d in self.defs.get(name)? {
            if let Def::Constant { body } = d {
                return Some(body.clone());
            }
        }
        None
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
