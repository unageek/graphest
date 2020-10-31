use crate::interval_set::{Site, TupperIntervalSet};
use bitflags::*;
use std::{
    cell::Cell,
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
};

pub type ExprId = u32;
const UNINIT_EXPR_ID: ExprId = ExprId::MAX;

pub type RelId = u32;
const UNINIT_REL_ID: RelId = RelId::MAX;

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
    SinOverX,
    Sinh,
    Sqr,
    Sqrt,
    Tan,
    Tanh,
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
pub enum ExprKind {
    Constant(Box<TupperIntervalSet>),
    X,
    Y,
    Unary(UnaryOp, Box<Expr>),
    Binary(BinaryOp, Box<Expr>, Box<Expr>),
    Pown(Box<Expr>, i32),
    Uninit,
}

bitflags! {
    pub struct AxisSet: u8 {
        const EMPTY = 0b00;
        const X = 0b01;
        const Y = 0b10;
        const XY = 0b11;
    }
}

impl ExprKind {
    fn dependent_axes(&self) -> AxisSet {
        match self {
            ExprKind::Constant(_) => AxisSet::EMPTY,
            ExprKind::X => AxisSet::X,
            ExprKind::Y => AxisSet::Y,
            ExprKind::Unary(_, x) | ExprKind::Pown(x, _) => x.dependent_axes,
            ExprKind::Binary(_, x, y) => x.dependent_axes | y.dependent_axes,
            ExprKind::Uninit => panic!(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Expr {
    pub id: Cell<ExprId>,
    pub site: Cell<Option<Site>>,
    pub kind: ExprKind,
    pub dependent_axes: AxisSet,
    internal_hash: u64,
}

impl Default for Expr {
    fn default() -> Self {
        Self {
            id: Cell::new(UNINIT_EXPR_ID),
            site: Cell::new(None),
            kind: ExprKind::Uninit,
            dependent_axes: AxisSet::EMPTY,
            internal_hash: 0,
        }
    }
}

impl PartialEq for Expr {
    fn eq(&self, rhs: &Self) -> bool {
        self.kind == rhs.kind
    }
}

impl Eq for Expr {}

impl Hash for Expr {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.internal_hash.hash(state);
    }
}

impl Expr {
    pub fn new(kind: ExprKind) -> Self {
        Self {
            id: Cell::new(UNINIT_EXPR_ID),
            site: Cell::new(None),
            kind,
            dependent_axes: AxisSet::EMPTY,
            internal_hash: 0,
        }
    }

    pub fn evaluate(&self) -> TupperIntervalSet {
        use {BinaryOp::*, ExprKind::*, UnaryOp::*};
        match &self.kind {
            Constant(x) => *x.clone(),
            Unary(Abs, x) => x.evaluate().abs(),
            Unary(Acos, x) => x.evaluate().acos(),
            Unary(Acosh, x) => x.evaluate().acosh(),
            Unary(Asin, x) => x.evaluate().asin(),
            Unary(Asinh, x) => x.evaluate().asinh(),
            Unary(Atan, x) => x.evaluate().atan(),
            Unary(Atanh, x) => x.evaluate().atanh(),
            Unary(Ceil, x) => x.evaluate().ceil(None),
            Unary(Cos, x) => x.evaluate().cos(),
            Unary(Cosh, x) => x.evaluate().cosh(),
            Unary(Exp, x) => x.evaluate().exp(),
            Unary(Exp10, x) => x.evaluate().exp10(),
            Unary(Exp2, x) => x.evaluate().exp2(),
            Unary(Floor, x) => x.evaluate().floor(None),
            Unary(Ln, x) => x.evaluate().ln(),
            Unary(Log10, x) => x.evaluate().log10(),
            Unary(Neg, x) => -&x.evaluate(),
            Unary(Recip, x) => x.evaluate().recip(None),
            Unary(Sign, x) => x.evaluate().sign(None),
            Unary(Sin, x) => x.evaluate().sin(),
            Unary(SinOverX, x) => x.evaluate().sin_over_x(),
            Unary(Sinh, x) => x.evaluate().sinh(),
            Unary(Sqr, x) => x.evaluate().sqr(),
            Unary(Sqrt, x) => x.evaluate().sqrt(),
            Unary(Tan, x) => x.evaluate().tan(None),
            Unary(Tanh, x) => x.evaluate().tanh(),
            Binary(Add, x, y) => &x.evaluate() + &y.evaluate(),
            Binary(Atan2, x, y) => x.evaluate().atan2(&y.evaluate(), None),
            Binary(Div, x, y) => x.evaluate().div(&y.evaluate(), None),
            // Beware the order of arguments.
            Binary(Log, b, x) => x.evaluate().log(&b.evaluate(), None),
            Binary(Max, x, y) => x.evaluate().max(&y.evaluate()),
            Binary(Min, x, y) => x.evaluate().min(&y.evaluate()),
            Binary(Mod, x, y) => x.evaluate().rem_euclid(&y.evaluate(), None),
            Binary(Mul, x, y) => &x.evaluate() * &y.evaluate(),
            Binary(Pow, x, y) => x.evaluate().pow(&y.evaluate(), None),
            Binary(Sub, x, y) => &x.evaluate() - &y.evaluate(),
            Pown(x, y) => x.evaluate().pown(*y, None),
            X | Y | Uninit => panic!("this expression cannot be evaluated"),
        }
    }

    // Precondition:
    //   The function is called on all subexpressions and they have not changed since then.
    pub fn update_metadata(&mut self) {
        self.dependent_axes = self.kind.dependent_axes();
        self.internal_hash = {
            // Use `DefaultHasher::new` so that the value of `internal_hash` will be deterministic.
            let mut hasher = DefaultHasher::new();
            self.kind.hash(&mut hasher);
            hasher.finish()
        }
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub enum RelKind {
    Atomic(RelOp, Box<Expr>, Box<Expr>),
    And(Box<Rel>, Box<Rel>),
    Or(Box<Rel>, Box<Rel>),
}

#[derive(Clone, Debug)]
pub struct Rel {
    pub id: Cell<RelId>,
    pub kind: RelKind,
}

impl PartialEq for Rel {
    fn eq(&self, rhs: &Self) -> bool {
        self.kind == rhs.kind
    }
}

impl Eq for Rel {}

impl Hash for Rel {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.kind.hash(state);
    }
}

impl Rel {
    pub fn new(kind: RelKind) -> Self {
        Self {
            id: Cell::new(UNINIT_REL_ID),
            kind,
        }
    }
}
