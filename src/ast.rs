use crate::{interval_set::*, rel::*};
use std::{
    cell::Cell,
    hash::{Hash, Hasher},
};

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

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub enum RelKind {
    Equality(EqualityOp, Box<Expr>, Box<Expr>),
    Binary(RelBinaryOp, Box<Rel>, Box<Rel>),
}

#[derive(Clone, Debug)]
pub struct Expr {
    pub id: Cell<ExprId>,
    pub site: Cell<Option<u8>>,
    pub kind: ExprKind,
}

impl Default for Expr {
    fn default() -> Self {
        Self {
            id: Cell::new(UNINIT_EXPR_ID),
            site: Cell::new(None),
            kind: ExprKind::Uninit,
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
        self.kind.hash(state);
    }
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

impl Expr {
    pub fn new(kind: ExprKind) -> Self {
        Self {
            id: Cell::new(UNINIT_EXPR_ID),
            site: Cell::new(None),
            kind,
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
            Unary(Log, x) => x.evaluate().log(),
            Unary(Log10, x) => x.evaluate().log10(),
            Unary(Log2, x) => x.evaluate().log2(),
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
            Binary(Max, x, y) => x.evaluate().max(&y.evaluate()),
            Binary(Min, x, y) => x.evaluate().min(&y.evaluate()),
            Binary(Mod, x, y) => x.evaluate().rem_euclid(&y.evaluate(), None),
            Binary(Mul, x, y) => &x.evaluate() * &y.evaluate(),
            Binary(Sub, x, y) => &x.evaluate() - &y.evaluate(),
            Pown(x, y) => x.evaluate().pown(*y, None),
            X | Y | Uninit => panic!("cannot evaluate the expression"),
        }
    }
}

impl Rel {
    pub fn new(kind: RelKind) -> Self {
        Self {
            id: Cell::new(UNINIT_REL_ID),
            kind,
        }
    }

    pub fn get_proposition(&self) -> Proposition {
        use {RelBinaryOp::*, RelKind::*};
        match &self.kind {
            Equality(_, _, _) => Proposition {
                kind: PropositionKind::Atomic,
                size: 1,
            },
            Binary(And, x, y) => {
                let px = x.get_proposition();
                let py = y.get_proposition();
                let size = px.size + py.size;
                Proposition {
                    kind: PropositionKind::And(box (px, py)),
                    size,
                }
            }
            Binary(Or, x, y) => {
                let px = x.get_proposition();
                let py = y.get_proposition();
                let size = px.size + py.size;
                Proposition {
                    kind: PropositionKind::Or(box (px, py)),
                    size,
                }
            }
        }
    }
}
