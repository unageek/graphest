use crate::{
    ast::{BinaryOp, Expr, UnaryOp},
    interval_set::TupperIntervalSet,
    parse::parse_expr,
    visit::{Parametrize, Substitute, VisitMut},
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
    /// Creates a definition of a constant.
    fn constant(x: DecInterval) -> Self {
        Self::Constant {
            body: Expr::constant(TupperIntervalSet::from(x), None),
        }
    }

    /// Creates a definition of a unary function.
    fn unary(op: UnaryOp) -> Self {
        Self::Function {
            arity: 1,
            body: Expr::unary(op, box Expr::var("0")),
            left_associative: false,
        }
    }

    /// Creates a definition of a binary function.
    fn binary(op: BinaryOp) -> Self {
        Self::Function {
            arity: 2,
            body: Expr::binary(op, box Expr::var("0"), box Expr::var("1")),
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

static BUILTIN_CONTEXT: SyncLazy<Context> = SyncLazy::new(|| {
    const EULER_GAMMA: DecInterval = const_dec_interval!(0.5772156649015328, 0.5772156649015329);
    let ctx = Context::new()
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
        .def("E", Def::unary(UnaryOp::EllipticE))
        .def("K", Def::unary(UnaryOp::EllipticK))
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
        .def("-", Def::unary(UnaryOp::Neg))
        .def("!", Def::unary(UnaryOp::Not))
        .def("Shi", Def::unary(UnaryOp::Shi))
        .def("Si", Def::unary(UnaryOp::Si))
        .def("sin", Def::unary(UnaryOp::Sin))
        .def("sinh", Def::unary(UnaryOp::Sinh))
        .def("sqrt", Def::unary(UnaryOp::Sqrt))
        .def("tan", Def::unary(UnaryOp::Tan))
        .def("tanh", Def::unary(UnaryOp::Tanh))
        .def("+", Def::binary(BinaryOp::Add))
        .def("&&", Def::binary(BinaryOp::And))
        .def("atan2", Def::binary(BinaryOp::Atan2))
        .def("I", Def::binary(BinaryOp::BesselI))
        .def("J", Def::binary(BinaryOp::BesselJ))
        .def("K", Def::binary(BinaryOp::BesselK))
        .def("Y", Def::binary(BinaryOp::BesselY))
        .def("/", Def::binary(BinaryOp::Div))
        .def("=", Def::binary(BinaryOp::Eq))
        .def("Gamma", Def::binary(BinaryOp::GammaInc))
        .def("Γ", Def::binary(BinaryOp::GammaInc))
        .def("gcd", Def::binary(BinaryOp::Gcd).left_associative())
        .def(">=", Def::binary(BinaryOp::Ge))
        .def(">", Def::binary(BinaryOp::Gt))
        .def("lcm", Def::binary(BinaryOp::Lcm).left_associative())
        .def("<=", Def::binary(BinaryOp::Le))
        .def("log", Def::binary(BinaryOp::Log))
        .def("<", Def::binary(BinaryOp::Lt))
        .def("max", Def::binary(BinaryOp::Max).left_associative())
        .def("min", Def::binary(BinaryOp::Min).left_associative())
        .def("mod", Def::binary(BinaryOp::Mod))
        .def("*", Def::binary(BinaryOp::Mul))
        .def("||", Def::binary(BinaryOp::Or))
        .def("^", Def::binary(BinaryOp::Pow))
        .def("ranked_max", Def::binary(BinaryOp::RankedMax))
        .def("ranked_min", Def::binary(BinaryOp::RankedMin))
        .def("-", Def::binary(BinaryOp::Sub));

    let mut body = parse_expr("⌊min(max(x, -0.5), 0.5)⌋ + ⌈min(max(x, -0.5), 0.5)⌉", &ctx).unwrap();
    Parametrize::new(vec!["x".into()]).visit_expr_mut(&mut body);
    let def = Def::Function {
        arity: 1,
        body,
        left_associative: false,
    };
    ctx.def("sgn", def.clone()).def("sign", def)
});

impl Context {
    /// Returns the context with the builtin definitions.
    pub fn builtin_context() -> &'static Self {
        &BUILTIN_CONTEXT
    }

    pub fn apply(&self, name: &str, mut args: Vec<Expr>) -> Option<Expr> {
        for d in self.defs.get(name)? {
            match *d {
                Def::Function { arity, .. } if args.len() == arity => {
                    let t = d.apply(args);
                    return Some(t);
                }
                Def::Function {
                    left_associative, ..
                } if left_associative && args.len() >= 2 => {
                    let mut args = args.drain(..);
                    let x0 = args.next().unwrap();
                    let t = args.fold(x0, |t, x| d.apply(vec![t, x]));
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
