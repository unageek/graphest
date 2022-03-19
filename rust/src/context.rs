use crate::{
    ast::{BinaryOp, Expr, TernaryOp, UnaryOp, ValueType},
    real::Real,
    vars::VarSet,
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
pub enum Def {
    Constant {
        body: Expr,
    },
    Function {
        arity: usize,
        body: Expr,
        left_associative: bool,
    },
}

#[derive(Clone, Debug)]
pub struct VarProps {
    pub totally_defined: bool,
    pub ty: ValueType,
    pub vars: VarSet,
}

impl Default for VarProps {
    fn default() -> Self {
        Self {
            totally_defined: false,
            ty: ValueType::Unknown,
            vars: VarSet::EMPTY,
        }
    }
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

    /// Creates a definition of a variable.
    pub fn var(name: &str, props: VarProps) -> Self {
        let mut e = Expr::var(name);
        e.totally_defined = props.totally_defined;
        e.ty = props.ty;
        e.vars = props.vars;
        Self::Constant { body: e }
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
    pub fn new() -> Self {
        Self {
            defs: HashMap::new(),
        }
    }

    /// Appends a definition to the context and returns `self`.
    ///
    /// Panics if there is already a definition that conflicts with the new one.
    pub fn def(mut self, name: &str, def: Def) -> Self {
        if let Some(defs) = self.defs.get_mut(name) {
            for d in defs.iter() {
                match (&def, d) {
                    (Def::Constant { .. }, _) | (_, Def::Constant { .. }) => panic!(),
                    (Def::Function { arity: a1, .. }, Def::Function { arity: a2, .. })
                        if a1 == a2 =>
                    {
                        panic!()
                    }
                    _ => (),
                }
            }
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
        .def(
            "m",
            Def::var(
                "m",
                VarProps {
                    totally_defined: true,
                    ty: ValueType::Real,
                    vars: VarSet::M,
                },
            ),
        )
        .def(
            "n",
            Def::var(
                "n",
                VarProps {
                    totally_defined: true,
                    ty: ValueType::Real,
                    vars: VarSet::N,
                },
            ),
        )
        .def(
            "r",
            Def::var(
                "r",
                VarProps {
                    totally_defined: true,
                    ty: ValueType::Real,
                    vars: VarSet::X | VarSet::Y,
                },
            ),
        )
        .def(
            "t",
            Def::var(
                "t",
                VarProps {
                    totally_defined: true,
                    ty: ValueType::Real,
                    vars: VarSet::T,
                },
            ),
        )
        // `theta` will be expanded to an expression that contains [`BinaryOp::Atan2`],
        // which is not totally defined.
        .def(
            "theta",
            Def::var(
                "theta",
                VarProps {
                    totally_defined: false,
                    ty: ValueType::Real,
                    vars: VarSet::X | VarSet::Y | VarSet::N_THETA,
                },
            ),
        )
        .def(
            "θ",
            Def::var(
                "theta",
                VarProps {
                    totally_defined: false,
                    ty: ValueType::Real,
                    vars: VarSet::X | VarSet::Y | VarSet::N_THETA,
                },
            ),
        )
        .def(
            "x",
            Def::var(
                "x",
                VarProps {
                    totally_defined: true,
                    ty: ValueType::Real,
                    vars: VarSet::X,
                },
            ),
        )
        .def(
            "y",
            Def::var(
                "y",
                VarProps {
                    totally_defined: true,
                    ty: ValueType::Real,
                    vars: VarSet::Y,
                },
            ),
        )
        .def(
            "<n-theta>",
            Def::var(
                "<n-theta>",
                VarProps {
                    totally_defined: true,
                    ty: ValueType::Real,
                    vars: VarSet::N_THETA,
                },
            ),
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
        .def("erfinv", Def::unary(InverseErf))
        .def("erfcinv", Def::unary(InverseErfc))
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
        .def("sinc", Def::unary(Sinc))
        .def("sinh", Def::unary(Sinh))
        .def("sqrt", Def::unary(Sqrt))
        .def("tan", Def::unary(Tan))
        .def("tanh", Def::unary(Tanh))
        .def("zeta", Def::unary(Zeta))
        .def("ζ", Def::unary(Zeta))
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
        .def(
            "W",
            Def::Function {
                arity: 1,
                body: { Expr::binary(LambertW, box Expr::zero(), box Expr::var("#0")) },
                left_associative: false,
            },
        )
        .def("W", Def::binary(LambertW))
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
        .def("^^", Def::binary(PowRational))
        .def("rankedMax", Def::binary(RankedMax))
        .def("rankedMin", Def::binary(RankedMin))
        .def("-", Def::binary(Sub))
        .def(
            "if",
            Def::Function {
                arity: 3,
                body: {
                    Expr::ternary(
                        IfThenElse(false),
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
    pub fn builtin() -> &'static Self {
        &BUILTIN_CONTEXT
    }

    /// Precondition: `name` is a function symbol.
    pub fn apply(&self, name: &str, args: Vec<Expr>) -> Expr {
        for d in self.defs.get(name).unwrap() {
            match *d {
                Def::Function { arity, .. } if args.len() == arity => {
                    return d.apply(args);
                }
                Def::Function {
                    left_associative, ..
                } if left_associative && args.len() >= 2 => {
                    return args
                        .into_iter()
                        .reduce(|xs, y| d.apply(vec![xs, y]))
                        .unwrap();
                }
                _ => (),
            }
        }
        Expr::error()
    }

    pub fn get_constant(&self, name: &str) -> Option<Expr> {
        for d in self.defs.get(name)? {
            if let Def::Constant { body } = d {
                return Some(body.clone());
            }
        }
        None
    }

    pub fn has(&self, name: &str) -> bool {
        self.defs.get(name).is_some()
    }

    pub fn is_function(&self, name: &str) -> bool {
        if let Some(defs) = self.defs.get(name) {
            matches!(defs.first().unwrap(), Def::Function { .. })
        } else {
            false
        }
    }
}

#[derive(Clone)]
pub struct InputWithContext<'a> {
    pub source: &'a str,
    pub context_stack: &'a [&'a Context],
    pub source_range: Range<usize>,
}

impl<'a> InputWithContext<'a> {
    pub fn new(source: &'a str, context_stack: &'a [&'a Context]) -> Self {
        Self {
            source,
            context_stack,
            source_range: 0..source.len(),
        }
    }
}

impl<'a> Compare<&str> for InputWithContext<'a> {
    fn compare(&self, t: &str) -> CompareResult {
        self.source.compare(t)
    }

    fn compare_no_case(&self, t: &str) -> CompareResult {
        self.source.compare_no_case(t)
    }
}

impl<'a> InputIter for InputWithContext<'a> {
    type Item = char;
    type Iter = CharIndices<'a>;
    type IterElem = Chars<'a>;

    fn iter_indices(&self) -> Self::Iter {
        self.source.iter_indices()
    }

    fn iter_elements(&self) -> Self::IterElem {
        self.source.iter_elements()
    }

    fn position<P>(&self, predicate: P) -> Option<usize>
    where
        P: Fn(Self::Item) -> bool,
    {
        self.source.position(predicate)
    }

    fn slice_index(&self, count: usize) -> Result<usize, Needed> {
        self.source.slice_index(count)
    }
}

impl<'a> InputLength for InputWithContext<'a> {
    fn input_len(&self) -> usize {
        self.source.input_len()
    }
}

impl<'a> InputTake for InputWithContext<'a> {
    fn take(&self, count: usize) -> Self {
        let start = self.source_range.start;
        let end = self.source_range.start + count;
        InputWithContext {
            source: self.source.take(count),
            source_range: start..end,
            ..*self
        }
    }

    fn take_split(&self, count: usize) -> (Self, Self) {
        // Beware the order.
        let (suffix, prefix) = self.source.take_split(count);
        let start = self.source_range.start;
        let mid = self.source_range.start + count;
        let end = self.source_range.end;
        (
            InputWithContext {
                source: suffix,
                source_range: mid..end,
                ..*self
            },
            InputWithContext {
                source: prefix,
                source_range: start..mid,
                ..*self
            },
        )
    }
}

impl<'a> Offset for InputWithContext<'a> {
    fn offset(&self, second: &Self) -> usize {
        self.source.offset(second.source)
    }
}

impl<'a> PartialEq for InputWithContext<'a> {
    fn eq(&self, other: &Self) -> bool {
        self.source == other.source
    }
}

impl<'a> Slice<Range<usize>> for InputWithContext<'a> {
    fn slice(&self, range: Range<usize>) -> Self {
        let start = self.source_range.start + range.start;
        let end = self.source_range.start + range.end;
        InputWithContext {
            source: self.source.slice(range),
            source_range: start..end,
            ..*self
        }
    }
}

impl<'a> Slice<RangeFrom<usize>> for InputWithContext<'a> {
    fn slice(&self, range: RangeFrom<usize>) -> Self {
        let start = self.source_range.start + range.start;
        let end = self.source_range.end;
        InputWithContext {
            source: self.source.slice(range),
            source_range: start..end,
            ..*self
        }
    }
}

impl<'a> Slice<RangeTo<usize>> for InputWithContext<'a> {
    fn slice(&self, range: RangeTo<usize>) -> Self {
        let start = self.source_range.start;
        let end = self.source_range.start + range.end;
        InputWithContext {
            source: self.source.slice(range),
            source_range: start..end,
            ..*self
        }
    }
}

impl<'a> Slice<RangeFull> for InputWithContext<'a> {
    fn slice(&self, range: RangeFull) -> Self {
        InputWithContext {
            source: self.source.slice(range),
            source_range: self.source_range.clone(),
            ..*self
        }
    }
}

impl<'a> UnspecializedInput for InputWithContext<'a> {}
