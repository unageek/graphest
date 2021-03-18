use crate::{
    ast::{BinaryOp, Expr, ExprId, ExprKind, UnaryOp, ValueType, VarSet, UNINIT_EXPR_ID},
    interval_set::{Site, TupperIntervalSet},
    ops::{
        FormIndex, RelOp, ScalarBinaryOp, ScalarUnaryOp, StaticForm, StaticFormKind, StaticTerm,
        StaticTermKind, StoreIndex, TermIndex,
    },
};
use inari::Decoration;
use std::{
    collections::{HashMap, HashSet},
    hash::{Hash, Hasher},
    marker::Sized,
    mem::{swap, take},
    ops::Deref,
};

/// A visitor that visits AST nodes in depth-first order.
pub trait Visit<'a>
where
    Self: Sized,
{
    fn visit_expr(&mut self, e: &'a Expr) {
        traverse_expr(self, e);
    }
}

fn traverse_expr<'a, V: Visit<'a>>(v: &mut V, e: &'a Expr) {
    use ExprKind::*;
    match &e.kind {
        Unary(_, x) => v.visit_expr(x),
        Binary(_, x, y) => {
            v.visit_expr(x);
            v.visit_expr(y);
        }
        Pown(x, _) => v.visit_expr(x),
        List(xs) => {
            for x in xs {
                v.visit_expr(x);
            }
        }
        Constant(_) | Var(_) | Uninit => (),
    };
}

/// A visitor that visits AST nodes in depth-first order and possibly modifies them.
pub trait VisitMut
where
    Self: Sized,
{
    fn visit_expr_mut(&mut self, e: &mut Expr) {
        traverse_expr_mut(self, e);
    }
}

fn traverse_expr_mut<V: VisitMut>(v: &mut V, e: &mut Expr) {
    use ExprKind::*;
    match &mut e.kind {
        Unary(_, x) => v.visit_expr_mut(x),
        Binary(_, x, y) => {
            v.visit_expr_mut(x);
            v.visit_expr_mut(y);
        }
        Pown(x, _) => v.visit_expr_mut(x),
        List(xs) => {
            for x in xs {
                v.visit_expr_mut(x);
            }
        }
        Constant(_) | Var(_) | Uninit => (),
    };
}

/// A possibly dangling reference to a value.
/// All operations except `from` and `clone` are unsafe.
struct UnsafeRef<T: Eq + Hash> {
    ptr: *const T,
}

impl<T: Eq + Hash> UnsafeRef<T> {
    fn from(x: &T) -> Self {
        Self { ptr: x as *const T }
    }
}

impl<T: Eq + Hash> Clone for UnsafeRef<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: Eq + Hash> Copy for UnsafeRef<T> {}

impl<T: Eq + Hash> Deref for UnsafeRef<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        unsafe { &*self.ptr }
    }
}

impl<T: Eq + Hash> PartialEq for UnsafeRef<T> {
    fn eq(&self, rhs: &Self) -> bool {
        unsafe { (*self.ptr) == (*rhs.ptr) }
    }
}

impl<T: Eq + Hash> Eq for UnsafeRef<T> {}

impl<T: Eq + Hash> Hash for UnsafeRef<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        unsafe { (*self.ptr).hash(state) }
    }
}

pub struct Substitute {
    args: Vec<Expr>,
}

impl Substitute {
    pub fn new(args: Vec<Expr>) -> Self {
        Self { args }
    }
}

impl VisitMut for Substitute {
    fn visit_expr_mut(&mut self, e: &mut Expr) {
        traverse_expr_mut(self, e);

        if let ExprKind::Var(x) = &mut e.kind {
            if let Ok(i) = x.parse::<usize>() {
                *e = self.args.get(i).unwrap().clone()
            }
        }
    }
}

/// Replaces a - b with a + (-b) and does some special transformations.
pub struct PreTransform;

impl VisitMut for PreTransform {
    fn visit_expr_mut(&mut self, e: &mut Expr) {
        use {BinaryOp::*, ExprKind::*, UnaryOp::*};
        traverse_expr_mut(self, e);

        match &mut e.kind {
            Binary(Sub, x, y) => {
                // (Sub x y) → (Add x (Neg y))
                *e = Expr::new(Binary(
                    Add,
                    take(x),
                    Box::new(Expr::new(Unary(Neg, take(y)))),
                ));
            }
            // Ad-hoc transformations mainly for demonstrational purposes.
            Binary(Div, x, y) => {
                match (&x.kind, &y.kind) {
                    (Unary(Sin, x1), _) if x1 == y => {
                        // (Div (Sin y) y) → (Sinc (UndefAt0 y))
                        *e = Expr::new(Unary(Sinc, Box::new(Expr::new(Unary(UndefAt0, take(y))))));
                    }
                    (_, Unary(Sin, y1)) if y1 == x => {
                        // (Div x (Sin x)) → (Recip (Sinc (UndefAt0 x)))
                        *e = Expr::new(Unary(
                            Recip,
                            Box::new(Expr::new(Unary(
                                Sinc,
                                Box::new(Expr::new(Unary(UndefAt0, take(x)))),
                            ))),
                        ));
                    }
                    _ if x == y => {
                        // (Div x x) → (One (UndefAt0 x))
                        *e = Expr::new(Unary(One, Box::new(Expr::new(Unary(UndefAt0, take(x))))));
                    }
                    _ => (),
                };
            }
            _ => (),
        }
    }
}

/// Moves constants to left-hand sides of addition/multiplication.
#[derive(Default)]
pub struct SortTerms {
    pub modified: bool,
}

fn precedes(x: &Expr, y: &Expr) -> bool {
    matches!(x.kind, ExprKind::Constant(_)) && !matches!(y.kind, ExprKind::Constant(_))
}

impl VisitMut for SortTerms {
    fn visit_expr_mut(&mut self, e: &mut Expr) {
        use {BinaryOp::*, ExprKind::*};
        traverse_expr_mut(self, e);

        match &mut e.kind {
            Binary(Add, x, y) | Binary(Mul, x, y) if precedes(y, x) => {
                // (op x y) /; y ≺ x → (op y x)
                swap(x, y);
                self.modified = true;
            }
            _ => (),
        }
    }
}

/// Transforms expressions into simpler/normalized forms.
#[derive(Default)]
pub struct Transform {
    pub modified: bool,
}

fn f64(x: &TupperIntervalSet) -> Option<f64> {
    if x.len() != 1 {
        return None;
    }

    let x = x.iter().next().unwrap().to_dec_interval();
    if x.is_singleton() && x.decoration() >= Decoration::Dac {
        Some(x.inf())
    } else {
        None
    }
}

impl VisitMut for Transform {
    fn visit_expr_mut(&mut self, e: &mut Expr) {
        use {BinaryOp::*, ExprKind::*, UnaryOp::*};
        traverse_expr_mut(self, e);

        match &mut e.kind {
            Unary(Neg, x) => {
                match &mut x.kind {
                    Unary(Neg, x1) => {
                        // (Neg (Neg x1)) → x1
                        *e = take(x1);
                        self.modified = true;
                    }
                    Binary(Add, x1, x2) => {
                        // (Neg (Add x1 x2)) → (Add (Neg x1) (Neg x2))
                        *e = Expr::new(Binary(
                            Add,
                            Box::new(Expr::new(Unary(Neg, take(x1)))),
                            Box::new(Expr::new(Unary(Neg, take(x2)))),
                        ));
                        self.modified = true;
                    }
                    _ => (),
                }
            }
            Binary(Add, x, y) => {
                match (&x.kind, &mut y.kind) {
                    (Constant(a), _) if f64(a) == Some(0.0) => {
                        // (Add 0 y) → y
                        *e = take(y);
                        self.modified = true;
                    }
                    (_, Binary(Add, y1, y2)) => {
                        // (Add x (Add y1 y2)) → (Add (Add x y1) y2)
                        *e = Expr::new(Binary(
                            Add,
                            Box::new(Expr::new(Binary(Add, take(x), take(y1)))),
                            take(y2),
                        ));
                        self.modified = true;
                    }
                    _ => (),
                }
            }
            Binary(Mul, x, y) => {
                match (&mut x.kind, &mut y.kind) {
                    (Constant(a), _) if f64(a) == Some(1.0) => {
                        // (Mul 1 y) → y
                        *e = take(y);
                        self.modified = true;
                    }
                    (Constant(a), _) if f64(a) == Some(-1.0) => {
                        // (Mul -1 y) → (Neg y)
                        *e = Expr::new(Unary(Neg, take(y)));
                        self.modified = true;
                    }
                    (Unary(Neg, x), _) => {
                        // (Mul (Neg x) y) → (Neg (Mul x y))
                        *e = Expr::new(Unary(
                            Neg,
                            Box::new(Expr::new(Binary(Mul, take(x), take(y)))),
                        ));
                        self.modified = true;
                    }
                    (_, Unary(Neg, y)) => {
                        // (Mul x (Neg y)) → (Neg (Mul x y))
                        *e = Expr::new(Unary(
                            Neg,
                            Box::new(Expr::new(Binary(Mul, take(x), take(y)))),
                        ));
                        self.modified = true;
                    }
                    (_, Binary(Mul, y1, y2)) => {
                        // (Mul x (Mul y1 y2)) → (Mul (Mul x y1) y2)
                        *e = Expr::new(Binary(
                            Mul,
                            Box::new(Expr::new(Binary(Mul, take(x), take(y1)))),
                            take(y2),
                        ));
                        self.modified = true;
                    }
                    _ => (),
                }
            }
            Unary(Not, x) => {
                match &mut x.kind {
                    Binary(op @ (Eq | Ge | Gt | Le | Lt | Neq | Nge | Ngt | Nle | Nlt), x1, x2) => {
                        // (Not (op x1 x2)) → (!op x1 x2)
                        let neg_op = match op {
                            Eq => Neq,
                            Ge => Nge,
                            Gt => Ngt,
                            Le => Nle,
                            Lt => Nlt,
                            Neq => Eq,
                            Nge => Ge,
                            Ngt => Gt,
                            Nle => Le,
                            Nlt => Lt,
                            _ => unreachable!(),
                        };
                        *e = Expr::new(Binary(neg_op, take(x1), take(x2)));
                        self.modified = true;
                    }
                    Binary(And, x1, x2) => {
                        // (And (x1 x2)) → (Or (Not x1) (Not x2))
                        *e = Expr::new(Binary(
                            Or,
                            Box::new(Expr::new(Unary(Not, take(x1)))),
                            Box::new(Expr::new(Unary(Not, take(x2)))),
                        ));
                        self.modified = true;
                    }
                    Binary(Or, x1, x2) => {
                        // (Or (x1 x2)) → (And (Not x1) (Not x2))
                        *e = Expr::new(Binary(
                            And,
                            Box::new(Expr::new(Unary(Not, take(x1)))),
                            Box::new(Expr::new(Unary(Not, take(x2)))),
                        ));
                        self.modified = true;
                    }
                    _ => (),
                }
            }
            _ => (),
        }
    }
}

/// Performs constant folding.
#[derive(Default)]
pub struct FoldConstant {
    pub modified: bool,
}

impl VisitMut for FoldConstant {
    fn visit_expr_mut(&mut self, e: &mut Expr) {
        use ExprKind::*;
        traverse_expr_mut(self, e);

        if !matches!(e.kind, ExprKind::Constant(_)) {
            if let Some(val) = e.eval() {
                // Only fold constants which evaluate to the empty or a single interval
                // since the branch cut tracking is not possible with the AST.
                if val.len() <= 1 {
                    *e = Expr::new(Constant(Box::new(val)));
                    self.modified = true;
                }
            }
        }
    }
}

/// Substitutes generic exponentiations with specialized functions.
pub struct PostTransform;

impl VisitMut for PostTransform {
    fn visit_expr_mut(&mut self, e: &mut Expr) {
        use {BinaryOp::*, ExprKind::*, UnaryOp::*};
        traverse_expr_mut(self, e);

        if let Binary(Pow, x, y) = &mut e.kind {
            match (&x.kind, &y.kind) {
                (Constant(a), _) => {
                    if let Some(a) = f64(a) {
                        if a == 2.0 {
                            // (Pow 2 x) → (Exp2 x)
                            *e = Expr::new(Unary(Exp2, take(y)));
                        } else if a == 10.0 {
                            // (Pow 10 x) → (Exp10 x)
                            *e = Expr::new(Unary(Exp10, take(y)));
                        }
                    }
                }
                (_, Constant(a)) => {
                    if let Some(a) = f64(a) {
                        if a == -1.0 {
                            // (Pow x -1) → (Recip x)
                            *e = Expr::new(Unary(Recip, take(x)));
                        } else if a == 0.0 {
                            // (Pow x 0) → (One x)
                            *e = Expr::new(Unary(One, take(x)));
                        } else if a == 0.5 {
                            // (Pow x 1/2) → (Sqrt x)
                            *e = Expr::new(Unary(Sqrt, take(x)));
                        } else if a == 1.0 {
                            // (Pow x 1) → x
                            *e = take(x);
                        } else if a == 2.0 {
                            // (Pow x 2) → (Sqr x)
                            *e = Expr::new(Unary(Sqr, take(x)));
                        } else if a == a as i32 as f64 {
                            // (Pow x a) /; a ∈ i32 → (Pown x a)
                            *e = Expr::new(Pown(take(x), a as i32));
                        }
                    }
                }
                _ => (),
            }
        }
    }
}

/// Updates metadata of terms and formulas.
pub struct UpdateMetadata;

impl VisitMut for UpdateMetadata {
    fn visit_expr_mut(&mut self, e: &mut Expr) {
        traverse_expr_mut(self, e);
        e.update_metadata();
    }
}

type SiteMap = HashMap<ExprId, Site>;
type UnsafeExprRef = UnsafeRef<Expr>;

/// Assigns [`ExprId`]s to unique expressions in topological order.
pub struct AssignId {
    next_id: ExprId,
    next_site: u8,
    site_map: SiteMap,
    exprs: Vec<UnsafeExprRef>,
    visited: HashSet<UnsafeExprRef>,
}

impl AssignId {
    pub fn new() -> Self {
        AssignId {
            next_id: 0,
            next_site: 0,
            site_map: HashMap::new(),
            exprs: vec![],
            visited: HashSet::new(),
        }
    }

    /// Returns `true` if the expression can perform branch cut on evaluation.
    fn term_can_perform_cut(kind: &ExprKind) -> bool {
        use {BinaryOp::*, ExprKind::*, UnaryOp::*};
        matches!(
            kind,
            Unary(Ceil, _)
                | Unary(Digamma, _)
                | Unary(Floor, _)
                | Unary(Gamma, _)
                | Unary(Recip, _)
                | Unary(Sign, _)
                | Unary(Tan, _)
                | Binary(Atan2, _, _)
                | Binary(Div, _, _)
                | Binary(Gcd, _, _)
                | Binary(Lcm, _, _)
                | Binary(Log, _, _)
                | Binary(Mod, _, _)
                | Binary(Pow, _, _)
                | Binary(RankedMax, _, _)
                | Binary(RankedMin, _, _)
                | Pown(_, _)
        )
    }
}

impl VisitMut for AssignId {
    fn visit_expr_mut(&mut self, e: &mut Expr) {
        traverse_expr_mut(self, e);

        match self.visited.get(&UnsafeExprRef::from(e)) {
            Some(visited) => {
                let id = visited.id;
                e.id = id;

                if !self.site_map.contains_key(&id)
                    && Self::term_can_perform_cut(&e.kind)
                    && self.next_site <= Site::MAX
                {
                    self.site_map.insert(id, Site::new(self.next_site));
                    self.next_site += 1;
                }
            }
            _ => {
                assert!(self.next_id != UNINIT_EXPR_ID);
                e.id = self.next_id;
                self.next_id += 1;
                let r = UnsafeExprRef::from(e);
                self.exprs.push(r);
                self.visited.insert(r);
            }
        }
    }
}

/// Collects [`StaticTerm`]s and [`StaticForm`]s in ascending order of the IDs.
pub struct CollectStatic {
    pub terms: Vec<StaticTerm>,
    pub forms: Vec<StaticForm>,
    site_map: SiteMap,
    exprs: Vec<UnsafeExprRef>,
    term_index: HashMap<ExprId, TermIndex>,
    form_index: HashMap<ExprId, FormIndex>,
    next_scalar_store_index: u32,
}

impl CollectStatic {
    pub fn new(v: AssignId) -> Self {
        let mut slf = Self {
            terms: vec![],
            forms: vec![],
            site_map: v.site_map,
            exprs: v.exprs,
            term_index: HashMap::new(),
            form_index: HashMap::new(),
            next_scalar_store_index: 0,
        };
        slf.collect_terms();
        slf.collect_atomic_forms();
        slf.collect_non_atomic_forms();
        slf
    }

    pub fn n_scalar_terms(&self) -> usize {
        self.exprs
            .iter()
            .filter(|t| t.ty == ValueType::Scalar)
            .count()
    }

    fn collect_terms(&mut self) {
        use {BinaryOp::*, ExprKind::*, UnaryOp::*};
        for t in self.exprs.iter().map(|t| &*t) {
            let k = match &t.kind {
                Constant(x) => Some(StaticTermKind::Constant(x.clone())),
                Var(x) if x == "x" => Some(StaticTermKind::X),
                Var(x) if x == "y" => Some(StaticTermKind::Y),
                Unary(Abs, x) => Some(StaticTermKind::Unary(ScalarUnaryOp::Abs, self.ti(x))),
                Unary(Acos, x) => Some(StaticTermKind::Unary(ScalarUnaryOp::Acos, self.ti(x))),
                Unary(Acosh, x) => Some(StaticTermKind::Unary(ScalarUnaryOp::Acosh, self.ti(x))),
                Unary(AiryAi, x) => Some(StaticTermKind::Unary(ScalarUnaryOp::AiryAi, self.ti(x))),
                Unary(AiryAiPrime, x) => Some(StaticTermKind::Unary(
                    ScalarUnaryOp::AiryAiPrime,
                    self.ti(x),
                )),
                Unary(AiryBi, x) => Some(StaticTermKind::Unary(ScalarUnaryOp::AiryBi, self.ti(x))),
                Unary(AiryBiPrime, x) => Some(StaticTermKind::Unary(
                    ScalarUnaryOp::AiryBiPrime,
                    self.ti(x),
                )),
                Unary(Asin, x) => Some(StaticTermKind::Unary(ScalarUnaryOp::Asin, self.ti(x))),
                Unary(Asinh, x) => Some(StaticTermKind::Unary(ScalarUnaryOp::Asinh, self.ti(x))),
                Unary(Atan, x) => Some(StaticTermKind::Unary(ScalarUnaryOp::Atan, self.ti(x))),
                Unary(Atanh, x) => Some(StaticTermKind::Unary(ScalarUnaryOp::Atanh, self.ti(x))),
                Unary(Ceil, x) => Some(StaticTermKind::Unary(ScalarUnaryOp::Ceil, self.ti(x))),
                Unary(Chi, x) => Some(StaticTermKind::Unary(ScalarUnaryOp::Chi, self.ti(x))),
                Unary(Ci, x) => Some(StaticTermKind::Unary(ScalarUnaryOp::Ci, self.ti(x))),
                Unary(Cos, x) => Some(StaticTermKind::Unary(ScalarUnaryOp::Cos, self.ti(x))),
                Unary(Cosh, x) => Some(StaticTermKind::Unary(ScalarUnaryOp::Cosh, self.ti(x))),
                Unary(Digamma, x) => {
                    Some(StaticTermKind::Unary(ScalarUnaryOp::Digamma, self.ti(x)))
                }
                Unary(Ei, x) => Some(StaticTermKind::Unary(ScalarUnaryOp::Ei, self.ti(x))),
                Unary(Erf, x) => Some(StaticTermKind::Unary(ScalarUnaryOp::Erf, self.ti(x))),
                Unary(Erfc, x) => Some(StaticTermKind::Unary(ScalarUnaryOp::Erfc, self.ti(x))),
                Unary(Erfi, x) => Some(StaticTermKind::Unary(ScalarUnaryOp::Erfi, self.ti(x))),
                Unary(Exp, x) => Some(StaticTermKind::Unary(ScalarUnaryOp::Exp, self.ti(x))),
                Unary(Exp10, x) => Some(StaticTermKind::Unary(ScalarUnaryOp::Exp10, self.ti(x))),
                Unary(Exp2, x) => Some(StaticTermKind::Unary(ScalarUnaryOp::Exp2, self.ti(x))),
                Unary(Floor, x) => Some(StaticTermKind::Unary(ScalarUnaryOp::Floor, self.ti(x))),
                Unary(FresnelC, x) => {
                    Some(StaticTermKind::Unary(ScalarUnaryOp::FresnelC, self.ti(x)))
                }
                Unary(FresnelS, x) => {
                    Some(StaticTermKind::Unary(ScalarUnaryOp::FresnelS, self.ti(x)))
                }
                Unary(Gamma, x) => Some(StaticTermKind::Unary(ScalarUnaryOp::Gamma, self.ti(x))),
                Unary(Li, x) => Some(StaticTermKind::Unary(ScalarUnaryOp::Li, self.ti(x))),
                Unary(Ln, x) => Some(StaticTermKind::Unary(ScalarUnaryOp::Ln, self.ti(x))),
                Unary(Log10, x) => Some(StaticTermKind::Unary(ScalarUnaryOp::Log10, self.ti(x))),
                Unary(Neg, x) => Some(StaticTermKind::Unary(ScalarUnaryOp::Neg, self.ti(x))),
                Unary(One, x) => Some(StaticTermKind::Unary(ScalarUnaryOp::One, self.ti(x))),
                Unary(Recip, x) => Some(StaticTermKind::Unary(ScalarUnaryOp::Recip, self.ti(x))),
                Unary(Shi, x) => Some(StaticTermKind::Unary(ScalarUnaryOp::Shi, self.ti(x))),
                Unary(Si, x) => Some(StaticTermKind::Unary(ScalarUnaryOp::Si, self.ti(x))),
                Unary(Sign, x) => Some(StaticTermKind::Unary(ScalarUnaryOp::Sign, self.ti(x))),
                Unary(Sin, x) => Some(StaticTermKind::Unary(ScalarUnaryOp::Sin, self.ti(x))),
                Unary(Sinc, x) => Some(StaticTermKind::Unary(ScalarUnaryOp::Sinc, self.ti(x))),
                Unary(Sinh, x) => Some(StaticTermKind::Unary(ScalarUnaryOp::Sinh, self.ti(x))),
                Unary(Sqr, x) => Some(StaticTermKind::Unary(ScalarUnaryOp::Sqr, self.ti(x))),
                Unary(Sqrt, x) => Some(StaticTermKind::Unary(ScalarUnaryOp::Sqrt, self.ti(x))),
                Unary(Tan, x) => Some(StaticTermKind::Unary(ScalarUnaryOp::Tan, self.ti(x))),
                Unary(Tanh, x) => Some(StaticTermKind::Unary(ScalarUnaryOp::Tanh, self.ti(x))),
                Unary(UndefAt0, x) => {
                    Some(StaticTermKind::Unary(ScalarUnaryOp::UndefAt0, self.ti(x)))
                }
                Binary(Add, x, y) => Some(StaticTermKind::Binary(
                    ScalarBinaryOp::Add,
                    self.ti(x),
                    self.ti(y),
                )),
                Binary(Atan2, y, x) => Some(StaticTermKind::Binary(
                    ScalarBinaryOp::Atan2,
                    self.ti(y),
                    self.ti(x),
                )),
                Binary(BesselI, n, x) => Some(StaticTermKind::Binary(
                    ScalarBinaryOp::BesselI,
                    self.ti(n),
                    self.ti(x),
                )),
                Binary(BesselJ, n, x) => Some(StaticTermKind::Binary(
                    ScalarBinaryOp::BesselJ,
                    self.ti(n),
                    self.ti(x),
                )),
                Binary(BesselK, n, x) => Some(StaticTermKind::Binary(
                    ScalarBinaryOp::BesselK,
                    self.ti(n),
                    self.ti(x),
                )),
                Binary(BesselY, n, x) => Some(StaticTermKind::Binary(
                    ScalarBinaryOp::BesselY,
                    self.ti(n),
                    self.ti(x),
                )),
                Binary(Div, x, y) => Some(StaticTermKind::Binary(
                    ScalarBinaryOp::Div,
                    self.ti(x),
                    self.ti(y),
                )),
                Binary(GammaInc, a, x) => Some(StaticTermKind::Binary(
                    ScalarBinaryOp::GammaInc,
                    self.ti(a),
                    self.ti(x),
                )),
                Binary(Gcd, x, y) => Some(StaticTermKind::Binary(
                    ScalarBinaryOp::Gcd,
                    self.ti(x),
                    self.ti(y),
                )),
                Binary(Lcm, x, y) => Some(StaticTermKind::Binary(
                    ScalarBinaryOp::Lcm,
                    self.ti(x),
                    self.ti(y),
                )),
                Binary(Log, b, x) => Some(StaticTermKind::Binary(
                    ScalarBinaryOp::Log,
                    self.ti(b),
                    self.ti(x),
                )),
                Binary(Max, x, y) => Some(StaticTermKind::Binary(
                    ScalarBinaryOp::Max,
                    self.ti(x),
                    self.ti(y),
                )),
                Binary(Min, x, y) => Some(StaticTermKind::Binary(
                    ScalarBinaryOp::Min,
                    self.ti(x),
                    self.ti(y),
                )),
                Binary(Mod, x, y) => Some(StaticTermKind::Binary(
                    ScalarBinaryOp::Mod,
                    self.ti(x),
                    self.ti(y),
                )),
                Binary(Mul, x, y) => Some(StaticTermKind::Binary(
                    ScalarBinaryOp::Mul,
                    self.ti(x),
                    self.ti(y),
                )),
                Binary(Pow, x, y) => Some(StaticTermKind::Binary(
                    ScalarBinaryOp::Pow,
                    self.ti(x),
                    self.ti(y),
                )),
                Binary(RankedMax, x, y) => Some(StaticTermKind::Binary(
                    ScalarBinaryOp::RankedMax,
                    self.ti(x),
                    self.ti(y),
                )),
                Binary(RankedMin, x, y) => Some(StaticTermKind::Binary(
                    ScalarBinaryOp::RankedMin,
                    self.ti(x),
                    self.ti(y),
                )),
                Binary(Sub, x, y) => Some(StaticTermKind::Binary(
                    ScalarBinaryOp::Sub,
                    self.ti(x),
                    self.ti(y),
                )),
                Pown(x, n) => Some(StaticTermKind::Pown(self.ti(x), *n)),
                List(xs) => Some(StaticTermKind::List(Box::new(
                    xs.iter().map(|x| self.ti(x)).collect(),
                ))),
                Var(_) | Uninit => panic!(),
                _ => None,
            };
            if let Some(k) = k {
                self.term_index.insert(t.id, self.terms.len() as TermIndex);
                let store_index = match &t.kind {
                    List(_) => StoreIndex::new(0), // List values are not stored.
                    _ => {
                        let i = self.next_scalar_store_index;
                        self.next_scalar_store_index += 1;
                        StoreIndex::new(i)
                    }
                };
                self.terms.push(StaticTerm {
                    site: self.site_map.get(&t.id).copied(),
                    kind: k,
                    vars: t.vars,
                    store_index,
                })
            }
        }
    }

    fn collect_atomic_forms(&mut self) {
        use {BinaryOp::*, ExprKind::*};
        for t in self.exprs.iter().map(|t| &*t) {
            let k = match &t.kind {
                Binary(Eq, x, y) => Some(StaticFormKind::Atomic(RelOp::Eq, self.ti(x), self.ti(y))),
                Binary(Ge, x, y) => Some(StaticFormKind::Atomic(RelOp::Ge, self.ti(x), self.ti(y))),
                Binary(Gt, x, y) => Some(StaticFormKind::Atomic(RelOp::Gt, self.ti(x), self.ti(y))),
                Binary(Le, x, y) => Some(StaticFormKind::Atomic(RelOp::Le, self.ti(x), self.ti(y))),
                Binary(Lt, x, y) => Some(StaticFormKind::Atomic(RelOp::Lt, self.ti(x), self.ti(y))),
                Binary(Neq, x, y) => {
                    Some(StaticFormKind::Atomic(RelOp::Neq, self.ti(x), self.ti(y)))
                }
                Binary(Nge, x, y) => {
                    Some(StaticFormKind::Atomic(RelOp::Nge, self.ti(x), self.ti(y)))
                }
                Binary(Ngt, x, y) => {
                    Some(StaticFormKind::Atomic(RelOp::Ngt, self.ti(x), self.ti(y)))
                }
                Binary(Nle, x, y) => {
                    Some(StaticFormKind::Atomic(RelOp::Nle, self.ti(x), self.ti(y)))
                }
                Binary(Nlt, x, y) => {
                    Some(StaticFormKind::Atomic(RelOp::Nlt, self.ti(x), self.ti(y)))
                }
                _ => None,
            };
            if let Some(k) = k {
                self.form_index.insert(t.id, self.forms.len() as FormIndex);
                self.forms.push(StaticForm { kind: k })
            }
        }
    }

    fn collect_non_atomic_forms(&mut self) {
        use {BinaryOp::*, ExprKind::*};
        for t in self.exprs.iter().map(|t| &*t) {
            let k = match &t.kind {
                Binary(And, x, y) => Some(StaticFormKind::And(self.fi(x), self.fi(y))),
                Binary(Or, x, y) => Some(StaticFormKind::Or(self.fi(x), self.fi(y))),
                _ => None,
            };
            if let Some(k) = k {
                self.form_index.insert(t.id, self.forms.len() as FormIndex);
                self.forms.push(StaticForm { kind: k })
            }
        }
    }

    fn ti(&self, e: &Expr) -> TermIndex {
        self.term_index[&e.id]
    }

    fn fi(&self, e: &Expr) -> FormIndex {
        self.form_index[&e.id]
    }
}

/// Collects the store indices of maximal scalar sub-expressions that contain exactly one free variable.
/// Expressions of the kind [`ExprKind::Var`] are excluded from collection.
pub struct FindMaximalScalarTerms {
    mx: Vec<StoreIndex>,
    my: Vec<StoreIndex>,
    terms: Vec<StaticTerm>,
    term_index: HashMap<ExprId, TermIndex>,
}

impl FindMaximalScalarTerms {
    pub fn new(collector: CollectStatic) -> Self {
        Self {
            mx: vec![],
            my: vec![],
            terms: collector.terms,
            term_index: collector.term_index,
        }
    }

    pub fn mx_my(mut self) -> (Vec<StoreIndex>, Vec<StoreIndex>) {
        self.mx.sort_unstable();
        self.mx.dedup();
        self.my.sort_unstable();
        self.my.dedup();
        (self.mx, self.my)
    }
}

impl<'a> Visit<'a> for FindMaximalScalarTerms {
    fn visit_expr(&mut self, e: &'a Expr) {
        match e.vars {
            VarSet::EMPTY => {
                // Stop traversal.
            }
            VarSet::X if e.ty == ValueType::Scalar => {
                if !matches!(e.kind, ExprKind::Var(_)) {
                    self.mx
                        .push(self.terms[self.term_index[&e.id] as usize].store_index);
                }
                // Stop traversal.
            }
            VarSet::Y if e.ty == ValueType::Scalar => {
                if !matches!(e.kind, ExprKind::Var(_)) {
                    self.my
                        .push(self.terms[self.term_index[&e.id] as usize].store_index);
                }
                // Stop traversal.
            }
            _ => traverse_expr(self, e),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{context::Context, parse::parse_expr};

    fn test_pre_transform(input: &str, expected: &str) {
        let mut f = parse_expr(input, Context::builtin_context()).unwrap();
        PreTransform.visit_expr_mut(&mut f);
        assert_eq!(format!("{}", f.dump_structure()), expected);
    }

    #[test]
    fn pre_transform() {
        test_pre_transform("x - y", "(Add x (Neg y))");
        test_pre_transform("sin(x)/x", "(Sinc (UndefAt0 x))");
        test_pre_transform("x/sin(x)", "(Recip (Sinc (UndefAt0 x)))");
        test_pre_transform("x/x", "(One (UndefAt0 x))");
    }

    fn test_sort_terms(input: &str, expected: &str) {
        let mut f = parse_expr(input, Context::builtin_context()).unwrap();
        let input = format!("{}", f.dump_structure());
        let mut v = SortTerms::default();
        v.visit_expr_mut(&mut f);
        let output = format!("{}", f.dump_structure());
        assert_eq!(output, expected);
        assert_eq!(v.modified, input != output);
    }

    #[test]
    fn sort_terms() {
        test_sort_terms("1 + x", "(Add @ x)");
        test_sort_terms("x + 1", "(Add @ x)");
        test_sort_terms("2 x", "(Mul @ x)");
        test_sort_terms("x 2", "(Mul @ x)");
    }

    fn test_transform(input: &str, expected: &str) {
        let mut f = parse_expr(input, Context::builtin_context()).unwrap();
        FoldConstant::default().visit_expr_mut(&mut f);
        let input = format!("{}", f.dump_structure());
        let mut v = Transform::default();
        v.visit_expr_mut(&mut f);
        let output = format!("{}", f.dump_structure());
        assert_eq!(output, expected);
        assert_eq!(v.modified, input != output);
    }

    #[test]
    fn transform() {
        test_transform("--x", "x");
        test_transform("-(x + y)", "(Add (Neg x) (Neg y))");
        test_transform("0 + x", "x");
        test_transform("x + (y + z)", "(Add (Add x y) z)");
        test_transform("1 x", "x");
        test_transform("-1 x", "(Neg x)"); // Needs constant folding
        test_transform("(-x) y", "(Neg (Mul x y))");
        test_transform("x (-y)", "(Neg (Mul x y))");
        test_transform("x (y z)", "(Mul (Mul x y) z)");
        test_transform("!(x = y)", "(Neq x y)");
        test_transform("!(x ≤ y)", "(Nle x y)");
        test_transform("!(x < y)", "(Nlt x y)");
        test_transform("!(x ≥ y)", "(Nge x y)");
        test_transform("!(x > y)", "(Ngt x y)");
        test_transform("!!(x = y)", "(Eq x y)");
        test_transform("!!(x ≤ y)", "(Le x y)");
        test_transform("!!(x < y)", "(Lt x y)");
        test_transform("!!(x ≥ y)", "(Ge x y)");
        test_transform("!!(x > y)", "(Gt x y)");
        test_transform("!(x = y && y = z)", "(Or (Not (Eq x y)) (Not (Eq y z)))");
        test_transform("!(x = y || y = z)", "(And (Not (Eq x y)) (Not (Eq y z)))");
    }

    fn test_post_transform(input: &str, expected: &str) {
        let mut f = parse_expr(input, Context::builtin_context()).unwrap();
        FoldConstant::default().visit_expr_mut(&mut f);
        PostTransform.visit_expr_mut(&mut f);
        assert_eq!(format!("{}", f.dump_structure()), expected);
    }

    #[test]
    fn post_transform() {
        test_post_transform("2^x", "(Exp2 x)");
        test_post_transform("10^x", "(Exp10 x)");
        test_post_transform("x^-1", "(Recip x)"); // Needs constant folding
        test_post_transform("x^0", "(One x)");
        test_post_transform("x^0.5", "(Sqrt x)");
        test_post_transform("x^1", "x");
        test_post_transform("x^2", "(Sqr x)");
        test_post_transform("x^3", "(Pown x 3)");
    }
}
