use crate::interval_set::{Site, TupperIntervalSet};
use bitflags::*;
use std::{
    cell::Cell,
    collections::hash_map::DefaultHasher,
    fmt,
    hash::{Hash, Hasher},
};

pub type TermId = u32;
const UNINIT_TERM_ID: TermId = TermId::MAX;

pub type FormId = u32;
const UNINIT_FORM_ID: FormId = FormId::MAX;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum UnaryOp {
    Abs,
    Acos,
    Acosh,
    Asin,
    Asinh,
    Atan,
    Atanh,
    Ceil,
    Cos,
    Cosh,
    Exp,
    Exp10,
    Exp2,
    Floor,
    Ln,
    Log10,
    Neg,
    Recip,
    Sign,
    Sin,
    SinXOverX,
    Sinh,
    Sqr,
    Sqrt,
    Tan,
    Tanh,
    XOverX,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum BinaryOp {
    Add,
    Atan2,
    Div,
    Log,
    Max,
    Min,
    Mod,
    Mul,
    Pow,
    Sub,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum RelOp {
    Eq,
    Ge,
    Gt,
    Le,
    Lt,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub enum TermKind {
    Constant(Box<TupperIntervalSet>),
    X,
    Y,
    Unary(UnaryOp, Box<Term>),
    Binary(BinaryOp, Box<Term>, Box<Term>),
    Pown(Box<Term>, i32),
    Uninit,
}

bitflags! {
    /// A set of free variables: x or y.
    pub struct VarSet: u8 {
        const EMPTY = 0b00;
        const X = 0b01;
        const Y = 0b10;
        const XY = 0b11;
    }
}

/// An AST node for a term.
#[derive(Clone, Debug)]
pub struct Term {
    pub id: Cell<TermId>,
    pub site: Cell<Option<Site>>,
    pub kind: TermKind,
    /// The set of the free variables in the term.
    pub vars: VarSet,
    internal_hash: u64,
}

impl Term {
    pub fn new(kind: TermKind) -> Self {
        Self {
            id: Cell::new(UNINIT_TERM_ID),
            site: Cell::new(None),
            kind,
            vars: VarSet::EMPTY,
            internal_hash: 0,
        }
    }

    pub fn dump_structure(&self) -> impl fmt::Display + '_ {
        DumpTermStructure(self)
    }

    /// Evaluates the term.
    ///
    /// Panics if the term contains a sub-term of kind [`TermKind::X`], [`TermKind::Y`]
    /// or [`TermKind::Uninit`].
    pub fn eval(&self) -> TupperIntervalSet {
        use {BinaryOp::*, TermKind::*, UnaryOp::*};
        match &self.kind {
            Constant(x) => *x.clone(),
            Unary(Abs, x) => x.eval().abs(),
            Unary(Acos, x) => x.eval().acos(),
            Unary(Acosh, x) => x.eval().acosh(),
            Unary(Asin, x) => x.eval().asin(),
            Unary(Asinh, x) => x.eval().asinh(),
            Unary(Atan, x) => x.eval().atan(),
            Unary(Atanh, x) => x.eval().atanh(),
            Unary(Ceil, x) => x.eval().ceil(None),
            Unary(Cos, x) => x.eval().cos(),
            Unary(Cosh, x) => x.eval().cosh(),
            Unary(Exp, x) => x.eval().exp(),
            Unary(Exp10, x) => x.eval().exp10(),
            Unary(Exp2, x) => x.eval().exp2(),
            Unary(Floor, x) => x.eval().floor(None),
            Unary(Ln, x) => x.eval().ln(),
            Unary(Log10, x) => x.eval().log10(),
            Unary(Neg, x) => -&x.eval(),
            Unary(Recip, x) => x.eval().recip(None),
            Unary(Sign, x) => x.eval().sign(None),
            Unary(Sin, x) => x.eval().sin(),
            Unary(SinXOverX, x) => x.eval().sin_x_over_x(),
            Unary(Sinh, x) => x.eval().sinh(),
            Unary(Sqr, x) => x.eval().sqr(),
            Unary(Sqrt, x) => x.eval().sqrt(),
            Unary(Tan, x) => x.eval().tan(None),
            Unary(Tanh, x) => x.eval().tanh(),
            Unary(XOverX, x) => x.eval().x_over_x(),
            Binary(Add, x, y) => &x.eval() + &y.eval(),
            Binary(Atan2, x, y) => x.eval().atan2(&y.eval(), None),
            Binary(Div, x, y) => x.eval().div(&y.eval(), None),
            // Beware the order of arguments.
            Binary(Log, b, x) => x.eval().log(&b.eval(), None),
            Binary(Max, x, y) => x.eval().max(&y.eval()),
            Binary(Min, x, y) => x.eval().min(&y.eval()),
            Binary(Mod, x, y) => x.eval().rem_euclid(&y.eval(), None),
            Binary(Mul, x, y) => &x.eval() * &y.eval(),
            Binary(Pow, x, y) => x.eval().pow(&y.eval(), None),
            Binary(Sub, x, y) => &x.eval() - &y.eval(),
            Pown(x, y) => x.eval().pown(*y, None),
            X | Y | Uninit => panic!("this term cannot be evaluated"),
        }
    }

    /// Updates `vars` and `internal_hash` fields of `self`.
    ///
    /// Precondition:
    ///   The function is called on all sub-terms and they have not been changed since then.
    pub fn update_metadata(&mut self) {
        self.vars = match &self.kind {
            TermKind::Constant(_) => VarSet::EMPTY,
            TermKind::X => VarSet::X,
            TermKind::Y => VarSet::Y,
            TermKind::Unary(_, x) | TermKind::Pown(x, _) => x.vars,
            TermKind::Binary(_, x, y) => x.vars | y.vars,
            TermKind::Uninit => panic!(),
        };
        self.internal_hash = {
            // Use `DefaultHasher::new` so that the value of `internal_hash` will be deterministic.
            let mut hasher = DefaultHasher::new();
            self.kind.hash(&mut hasher);
            hasher.finish()
        }
    }
}

impl Default for Term {
    fn default() -> Self {
        Self {
            id: Cell::new(UNINIT_TERM_ID),
            site: Cell::new(None),
            kind: TermKind::Uninit,
            vars: VarSet::EMPTY,
            internal_hash: 0,
        }
    }
}

impl PartialEq for Term {
    fn eq(&self, rhs: &Self) -> bool {
        self.kind == rhs.kind
    }
}

impl Eq for Term {}

impl Hash for Term {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.internal_hash.hash(state);
    }
}

struct DumpTermStructure<'a>(&'a Term);

impl<'a> fmt::Display for DumpTermStructure<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match &self.0.kind {
            TermKind::Constant(_) => write!(f, "{{...}}"),
            TermKind::Unary(op, x) => write!(f, "({:?} {})", op, x.dump_structure()),
            TermKind::Binary(op, x, y) => write!(
                f,
                "({:?} {} {})",
                op,
                x.dump_structure(),
                y.dump_structure()
            ),
            TermKind::Pown(x, y) => write!(f, "(Pown {} {})", x.dump_structure(), y),
            x => write!(f, "{:?}", x),
        }
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub enum FormKind {
    Atomic(RelOp, Box<Term>, Box<Term>),
    And(Box<Form>, Box<Form>),
    Or(Box<Form>, Box<Form>),
}

/// An AST node for a formula.
#[derive(Clone, Debug)]
pub struct Form {
    pub id: Cell<FormId>,
    pub kind: FormKind,
    internal_hash: u64,
}

impl Form {
    pub fn new(kind: FormKind) -> Self {
        Self {
            id: Cell::new(UNINIT_FORM_ID),
            kind,
            internal_hash: 0,
        }
    }

    pub fn dump_structure(&self) -> impl fmt::Display + '_ {
        DumpFormStructure(self)
    }

    /// Updates `internal_hash` field of `self`.
    ///
    /// Precondition:
    ///   The function is called on all sub-terms/forms and they have not been changed since then.
    pub fn update_metadata(&mut self) {
        self.internal_hash = {
            let mut hasher = DefaultHasher::new();
            self.kind.hash(&mut hasher);
            hasher.finish()
        }
    }
}

impl PartialEq for Form {
    fn eq(&self, rhs: &Self) -> bool {
        self.kind == rhs.kind
    }
}

impl Eq for Form {}

impl Hash for Form {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.kind.hash(state);
    }
}

struct DumpFormStructure<'a>(&'a Form);

impl<'a> fmt::Display for DumpFormStructure<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match &self.0.kind {
            FormKind::Atomic(op, x, y) => write!(
                f,
                "({:?} {} {})",
                op,
                x.dump_structure(),
                y.dump_structure()
            ),
            FormKind::And(x, y) => write!(f, "(And {} {})", x.dump_structure(), y.dump_structure()),
            FormKind::Or(x, y) => write!(f, "(Or {} {})", x.dump_structure(), y.dump_structure()),
        }
    }
}
