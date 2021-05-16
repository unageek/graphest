use crate::{
    ast::{BinaryOp, Expr, ExprId, ExprKind, UnaryOp, ValueType, VarSet, UNINIT_EXPR_ID},
    interval_set::Site,
    ops::{
        FormIndex, RelOp, ScalarBinaryOp, ScalarUnaryOp, StaticForm, StaticFormKind, StaticTerm,
        StaticTermKind, StoreIndex, TermIndex,
    },
};
use inari::const_dec_interval;
use std::{
    cmp::Ordering,
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
        Rootn(x, _) => v.visit_expr(x),
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
        Rootn(x, _) => v.visit_expr_mut(x),
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

/// Replaces the names of [`ExprKind::Var`]s that are equal to `params[i]` with `i.to_string()`.
pub struct Parametrize {
    params: Vec<String>,
}

impl Parametrize {
    pub fn new(params: Vec<String>) -> Self {
        Self { params }
    }
}

impl VisitMut for Parametrize {
    fn visit_expr_mut(&mut self, e: &mut Expr) {
        traverse_expr_mut(self, e);

        if let ExprKind::Var(x) = &mut e.kind {
            if let Some(i) = self.params.iter().position(|p| p == x) {
                *x = i.to_string();
            }
        }
    }
}

/// Replaces all expressions of the kind [`ExprKind::Var`] with name "0", "1", …
/// with `args[0]`, `args[1]`, …, respectively.
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

pub struct ReplaceAll<Rule>
where
    Rule: Fn(&Expr) -> Option<Expr>,
{
    pub modified: bool,
    rule: Rule,
}

impl<Rule> ReplaceAll<Rule>
where
    Rule: Fn(&Expr) -> Option<Expr>,
{
    pub fn new(rule: Rule) -> Self {
        Self {
            modified: false,
            rule,
        }
    }
}

impl<Rule> VisitMut for ReplaceAll<Rule>
where
    Rule: Fn(&Expr) -> Option<Expr>,
{
    fn visit_expr_mut(&mut self, e: &mut Expr) {
        traverse_expr_mut(self, e);

        if let Some(replacement) = (self.rule)(e) {
            *e = replacement;
            self.modified = true;
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
                *e = Expr::binary(Add, take(x), box Expr::unary(Neg, take(y)));
            }
            // Ad-hoc transformations mainly for demonstrational purposes.
            Binary(Div, x, y) => {
                match (&x.kind, &y.kind) {
                    (Unary(Sin, x1), _) if x1 == y => {
                        // (Div (Sin y) y) → (Sinc (UndefAt0 y))
                        *e = Expr::unary(Sinc, box Expr::unary(UndefAt0, take(y)));
                    }
                    (_, Unary(Sin, y1)) if y1 == x => {
                        // (Div x (Sin x)) → (Pow (Sinc (UndefAt0 x)) -1)
                        *e = Expr::binary(
                            Pow,
                            box Expr::unary(Sinc, box Expr::unary(UndefAt0, take(x))),
                            box Expr::constant(
                                const_dec_interval!(-1.0, -1.0).into(),
                                Some((-1).into()),
                            ),
                        );
                    }
                    _ if x == y => {
                        // (Div x x) → (One (UndefAt0 x))
                        *e = Expr::unary(One, box Expr::unary(UndefAt0, take(x)));
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

fn cmp_terms(x: &Expr, y: &Expr) -> Ordering {
    use {BinaryOp::*, ExprKind::*};
    match (&x.kind, &y.kind) {
        (Constant(_), Constant(_)) => Ordering::Equal,
        (Constant(_), _) => Ordering::Less,
        (_, Constant(_)) => Ordering::Greater,
        (Var(x), Var(y)) => x.cmp(y),
        (Var(_), _) => Ordering::Less,
        (_, Var(_)) => Ordering::Greater,
        (Binary(Mul, x1, y1), Binary(Mul, x2, y2)) => {
            cmp_terms(y1, y2).then_with(|| cmp_terms(x1, x2))
        }
        (Binary(Pow, x1, y1), Binary(Pow, x2, y2)) => {
            cmp_terms(x1, x2).then_with(|| cmp_terms(y1, y2))
        }
        _ => Ordering::Equal,
    }
}

impl VisitMut for SortTerms {
    fn visit_expr_mut(&mut self, e: &mut Expr) {
        use {BinaryOp::*, ExprKind::*};
        traverse_expr_mut(self, e);

        match &mut e.kind {
            Binary(Add | Mul, x, y) if cmp_terms(y, x) == Ordering::Less => {
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
                        *e = Expr::binary(
                            Add,
                            box Expr::unary(Neg, take(x1)),
                            box Expr::unary(Neg, take(x2)),
                        );
                        self.modified = true;
                    }
                    _ => (),
                }
            }
            Binary(Add, x, y) => {
                match (&x.kind, &mut y.kind) {
                    (Constant(a), _) if a.0.to_f64() == Some(0.0) => {
                        // (Add 0 y) → y
                        *e = take(y);
                        self.modified = true;
                    }
                    (_, Binary(Add, y1, y2)) => {
                        // (Add x (Add y1 y2)) → (Add (Add x y1) y2)
                        *e = Expr::binary(Add, box Expr::binary(Add, take(x), take(y1)), take(y2));
                        self.modified = true;
                    }
                    _ => (),
                }
            }
            Binary(Mul, x, y) => {
                match (&mut x.kind, &mut y.kind) {
                    (Constant(a), _) if a.0.to_f64() == Some(1.0) => {
                        // (Mul 1 y) → y
                        *e = take(y);
                        self.modified = true;
                    }
                    (Constant(a), _) if a.0.to_f64() == Some(-1.0) => {
                        // (Mul -1 y) → (Neg y)
                        *e = Expr::unary(Neg, take(y));
                        self.modified = true;
                    }
                    (Unary(Neg, x), _) => {
                        // (Mul (Neg x) y) → (Neg (Mul x y))
                        *e = Expr::unary(Neg, box Expr::binary(Mul, take(x), take(y)));
                        self.modified = true;
                    }
                    (_, Unary(Neg, y)) => {
                        // (Mul x (Neg y)) → (Neg (Mul x y))
                        *e = Expr::unary(Neg, box Expr::binary(Mul, take(x), take(y)));
                        self.modified = true;
                    }
                    (_, Binary(Mul, y1, y2)) => {
                        // (Mul x (Mul y1 y2)) → (Mul (Mul x y1) y2)
                        *e = Expr::binary(Mul, box Expr::binary(Mul, take(x), take(y1)), take(y2));
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
                        *e = Expr::binary(neg_op, take(x1), take(x2));
                        self.modified = true;
                    }
                    Binary(And, x1, x2) => {
                        // (And (x1 x2)) → (Or (Not x1) (Not x2))
                        *e = Expr::binary(
                            Or,
                            box Expr::unary(Not, take(x1)),
                            box Expr::unary(Not, take(x2)),
                        );
                        self.modified = true;
                    }
                    Binary(Or, x1, x2) => {
                        // (Or (x1 x2)) → (And (Not x1) (Not x2))
                        *e = Expr::binary(
                            And,
                            box Expr::unary(Not, take(x1)),
                            box Expr::unary(Not, take(x2)),
                        );
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
        traverse_expr_mut(self, e);

        if !matches!(e.kind, ExprKind::Constant(_)) {
            if let Some((x, xr)) = e.eval() {
                // Only fold constants which evaluate to the empty or a single interval
                // since the branch cut tracking is not possible with the AST.
                if x.len() <= 1 {
                    *e = Expr::constant(x, xr);
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
                    if let Some(a) = a.0.to_f64() {
                        if a == 2.0 {
                            // (Pow 2 x) → (Exp2 x)
                            *e = Expr::unary(Exp2, take(y));
                        } else if a == 10.0 {
                            // (Pow 10 x) → (Exp10 x)
                            *e = Expr::unary(Exp10, take(y));
                        }
                    }
                }
                (_, Constant(a)) => {
                    if let Some(a) = &a.1 {
                        if let (Some(n), Some(d)) = (a.numer().to_i32(), a.denom().to_u32()) {
                            let root = match d {
                                1 => take(x),
                                2 => box Expr::unary(Sqrt, take(x)),
                                _ => box Expr::rootn(take(x), d),
                            };
                            *e = match n {
                                -1 => Expr::unary(Recip, root),
                                0 => Expr::unary(One, root),
                                1 => *root,
                                2 => Expr::unary(Sqr, root),
                                _ => Expr::pown(root, n),
                            }
                        }
                    }
                }
                _ => (),
            }
        }
    }
}

pub struct UpdatePolarPeriod;

impl VisitMut for UpdatePolarPeriod {
    fn visit_expr_mut(&mut self, e: &mut Expr) {
        use {BinaryOp::*, ExprKind::*, UnaryOp::*};
        traverse_expr_mut(self, e);

        match &mut e.kind {
            Constant(_) => e.polar_period = Some(0.into()),
            Var(name) if name != "theta" && name != "θ" => e.polar_period = Some(0.into()),
            Unary(_, x) if x.polar_period.is_some() => {
                e.polar_period = x.polar_period.clone();
            }
            Binary(_, x, y) if x.polar_period.is_some() && y.polar_period.is_some() => {
                let xp = x.polar_period.as_ref().unwrap();
                let yp = y.polar_period.as_ref().unwrap();
                e.polar_period = Some(xp.gcd_ref(yp).into());
            }
            Unary(Cos | Sin | Tan, x) => match &x.kind {
                Var(name) if name == "theta" || name == "θ" => {
                    // sin(θ)
                    e.polar_period = Some(1.into());
                }
                Binary(Mul, x, y) => match (&x.kind, &y.kind) {
                    (Constant(a), Var(name)) if name == "theta" || name == "θ" => {
                        // sin(a θ)
                        if let Some(a) = &a.1 {
                            e.polar_period = Some(a.denom().clone())
                        }
                    }
                    _ => (),
                },
                Binary(Add, x, y) => match (&x.kind, &y.kind) {
                    (Constant(_), Var(name)) if name == "theta" || name == "θ" => {
                        // sin(b + θ)
                        e.polar_period = Some(1.into());
                    }
                    (Constant(_), Binary(Mul, x, y)) => match (&x.kind, &y.kind) {
                        (Constant(a), Var(name)) if name == "theta" || name == "θ" => {
                            // sin(b + a θ)
                            if let Some(a) = &a.1 {
                                e.polar_period = Some(a.denom().clone())
                            }
                        }
                        _ => (),
                    },
                    _ => (),
                },
                _ => (),
            },
            _ => (),
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
            Unary(Ceil | Digamma | Floor | Gamma | Recip | Tan, _)
                | Binary(
                    Atan2 | Div | Gcd | Lcm | Log | Mod | Pow | RankedMax | RankedMin,
                    _,
                    _
                )
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
                Constant(x) => Some(StaticTermKind::Constant(box x.0.clone())),
                Var(x) if x == "x" => Some(StaticTermKind::X),
                Var(x) if x == "y" => Some(StaticTermKind::Y),
                Var(x) if x == "<n-theta>" => Some(StaticTermKind::NTheta),
                Unary(op, x) => match op {
                    Abs => Some(ScalarUnaryOp::Abs),
                    Acos => Some(ScalarUnaryOp::Acos),
                    Acosh => Some(ScalarUnaryOp::Acosh),
                    AiryAi => Some(ScalarUnaryOp::AiryAi),
                    AiryAiPrime => Some(ScalarUnaryOp::AiryAiPrime),
                    AiryBi => Some(ScalarUnaryOp::AiryBi),
                    AiryBiPrime => Some(ScalarUnaryOp::AiryBiPrime),
                    Asin => Some(ScalarUnaryOp::Asin),
                    Asinh => Some(ScalarUnaryOp::Asinh),
                    Atan => Some(ScalarUnaryOp::Atan),
                    Atanh => Some(ScalarUnaryOp::Atanh),
                    Ceil => Some(ScalarUnaryOp::Ceil),
                    Chi => Some(ScalarUnaryOp::Chi),
                    Ci => Some(ScalarUnaryOp::Ci),
                    Cos => Some(ScalarUnaryOp::Cos),
                    Cosh => Some(ScalarUnaryOp::Cosh),
                    Digamma => Some(ScalarUnaryOp::Digamma),
                    Ei => Some(ScalarUnaryOp::Ei),
                    EllipticE => Some(ScalarUnaryOp::EllipticE),
                    EllipticK => Some(ScalarUnaryOp::EllipticK),
                    Erf => Some(ScalarUnaryOp::Erf),
                    Erfc => Some(ScalarUnaryOp::Erfc),
                    Erfi => Some(ScalarUnaryOp::Erfi),
                    Exp => Some(ScalarUnaryOp::Exp),
                    Exp10 => Some(ScalarUnaryOp::Exp10),
                    Exp2 => Some(ScalarUnaryOp::Exp2),
                    Floor => Some(ScalarUnaryOp::Floor),
                    FresnelC => Some(ScalarUnaryOp::FresnelC),
                    FresnelS => Some(ScalarUnaryOp::FresnelS),
                    Gamma => Some(ScalarUnaryOp::Gamma),
                    Li => Some(ScalarUnaryOp::Li),
                    Ln => Some(ScalarUnaryOp::Ln),
                    Log10 => Some(ScalarUnaryOp::Log10),
                    Neg => Some(ScalarUnaryOp::Neg),
                    One => Some(ScalarUnaryOp::One),
                    Recip => Some(ScalarUnaryOp::Recip),
                    Shi => Some(ScalarUnaryOp::Shi),
                    Si => Some(ScalarUnaryOp::Si),
                    Sin => Some(ScalarUnaryOp::Sin),
                    Sinc => Some(ScalarUnaryOp::Sinc),
                    Sinh => Some(ScalarUnaryOp::Sinh),
                    Sqr => Some(ScalarUnaryOp::Sqr),
                    Sqrt => Some(ScalarUnaryOp::Sqrt),
                    Tan => Some(ScalarUnaryOp::Tan),
                    Tanh => Some(ScalarUnaryOp::Tanh),
                    UndefAt0 => Some(ScalarUnaryOp::UndefAt0),
                    _ => None,
                }
                .map(|op| StaticTermKind::Unary(op, self.ti(x))),
                Binary(op, x, y) => match op {
                    Add => Some(ScalarBinaryOp::Add),
                    Atan2 => Some(ScalarBinaryOp::Atan2),
                    BesselI => Some(ScalarBinaryOp::BesselI),
                    BesselJ => Some(ScalarBinaryOp::BesselJ),
                    BesselK => Some(ScalarBinaryOp::BesselK),
                    BesselY => Some(ScalarBinaryOp::BesselY),
                    Div => Some(ScalarBinaryOp::Div),
                    GammaInc => Some(ScalarBinaryOp::GammaInc),
                    Gcd => Some(ScalarBinaryOp::Gcd),
                    Lcm => Some(ScalarBinaryOp::Lcm),
                    Log => Some(ScalarBinaryOp::Log),
                    Max => Some(ScalarBinaryOp::Max),
                    Min => Some(ScalarBinaryOp::Min),
                    Mod => Some(ScalarBinaryOp::Mod),
                    Mul => Some(ScalarBinaryOp::Mul),
                    Pow => Some(ScalarBinaryOp::Pow),
                    RankedMax => Some(ScalarBinaryOp::RankedMax),
                    RankedMin => Some(ScalarBinaryOp::RankedMin),
                    Sub => Some(ScalarBinaryOp::Sub),
                    _ => None,
                }
                .map(|op| StaticTermKind::Binary(op, self.ti(x), self.ti(y))),
                Pown(x, n) => Some(StaticTermKind::Pown(self.ti(x), *n)),
                Rootn(x, n) => Some(StaticTermKind::Rootn(self.ti(x), *n)),
                List(xs) => Some(StaticTermKind::List(
                    box xs.iter().map(|x| self.ti(x)).collect(),
                )),
                Var(_) | Uninit => panic!(),
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
                Binary(op, x, y) => match op {
                    Eq => Some(RelOp::Eq),
                    Ge => Some(RelOp::Ge),
                    Gt => Some(RelOp::Gt),
                    Le => Some(RelOp::Le),
                    Lt => Some(RelOp::Lt),
                    Neq => Some(RelOp::Neq),
                    Nge => Some(RelOp::Nge),
                    Ngt => Some(RelOp::Ngt),
                    Nle => Some(RelOp::Nle),
                    Nlt => Some(RelOp::Nlt),
                    _ => None,
                }
                .map(|op| StaticFormKind::Atomic(op, self.ti(x), self.ti(y))),
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
    use rug::Integer;

    #[test]
    fn pre_transform() {
        fn test(input: &str, expected: &str) {
            let mut f = parse_expr(input, Context::builtin_context()).unwrap();
            PreTransform.visit_expr_mut(&mut f);
            assert_eq!(format!("{}", f.dump_structure()), expected);
        }

        test("x - y", "(Add x (Neg y))");
        test("sin(x)/x", "(Sinc (UndefAt0 x))");
        test("x/sin(x)", "(Pow (Sinc (UndefAt0 x)) @)");
        test("x/x", "(One (UndefAt0 x))");
    }

    #[test]
    fn sort_terms() {
        fn test(input: &str, expected: &str) {
            let mut f = parse_expr(input, Context::builtin_context()).unwrap();
            let input = format!("{}", f.dump_structure());
            let mut v = SortTerms::default();
            v.visit_expr_mut(&mut f);
            let output = format!("{}", f.dump_structure());
            assert_eq!(output, expected);
            assert_eq!(v.modified, input != output);
        }

        test("1 + x", "(Add @ x)");
        test("x + 1", "(Add @ x)");
        test("2 x", "(Mul @ x)");
        test("x 2", "(Mul @ x)");
    }

    #[test]
    fn transform() {
        fn test(input: &str, expected: &str) {
            let mut f = parse_expr(input, Context::builtin_context()).unwrap();
            FoldConstant::default().visit_expr_mut(&mut f);
            let input = format!("{}", f.dump_structure());
            let mut v = Transform::default();
            v.visit_expr_mut(&mut f);
            let output = format!("{}", f.dump_structure());
            assert_eq!(output, expected);
            assert_eq!(v.modified, input != output);
        }

        test("--x", "x");
        test("-(x + y)", "(Add (Neg x) (Neg y))");
        test("0 + x", "x");
        test("x + (y + z)", "(Add (Add x y) z)");
        test("1 x", "x");
        test("-1 x", "(Neg x)"); // Needs constant folding
        test("(-x) y", "(Neg (Mul x y))");
        test("x (-y)", "(Neg (Mul x y))");
        test("x (y z)", "(Mul (Mul x y) z)");
        test("!(x = y)", "(Neq x y)");
        test("!(x ≤ y)", "(Nle x y)");
        test("!(x < y)", "(Nlt x y)");
        test("!(x ≥ y)", "(Nge x y)");
        test("!(x > y)", "(Ngt x y)");
        test("!!(x = y)", "(Eq x y)");
        test("!!(x ≤ y)", "(Le x y)");
        test("!!(x < y)", "(Lt x y)");
        test("!!(x ≥ y)", "(Ge x y)");
        test("!!(x > y)", "(Gt x y)");
        test("!(x = y && y = z)", "(Or (Not (Eq x y)) (Not (Eq y z)))");
        test("!(x = y || y = z)", "(And (Not (Eq x y)) (Not (Eq y z)))");
    }

    #[test]
    fn post_transform() {
        fn test(input: &str, expected: &str) {
            let mut f = parse_expr(input, Context::builtin_context()).unwrap();
            FoldConstant::default().visit_expr_mut(&mut f);
            PostTransform.visit_expr_mut(&mut f);
            assert_eq!(format!("{}", f.dump_structure()), expected);
        }

        test("2^x", "(Exp2 x)");
        test("10^x", "(Exp10 x)");
        test("x^-1", "(Recip x)");
        test("x^0", "(One x)");
        test("x^1", "x");
        test("x^2", "(Sqr x)");
        test("x^3", "(Pown x 3)");
        test("x^(1/2)", "(Sqrt x)");
        test("x^(3/2)", "(Pown (Sqrt x) 3)");
        test("x^(-2/3)", "(Pown (Rootn x 3) -2)");
        test("x^(-1/3)", "(Recip (Rootn x 3))");
        test("x^(1/3)", "(Rootn x 3)");
        test("x^(2/3)", "(Sqr (Rootn x 3))");
    }

    #[test]
    fn update_polar_period() {
        fn test(input: &str, expected_period: Option<Integer>) {
            let mut f = parse_expr(input, Context::builtin_context()).unwrap();
            FoldConstant::default().visit_expr_mut(&mut f);
            UpdatePolarPeriod.visit_expr_mut(&mut f);
            assert_eq!(f.polar_period, expected_period);
        }

        test("42", Some(0.into()));
        test("x", Some(0.into()));
        test("y", Some(0.into()));
        test("r", Some(0.into()));
        test("θ", None);
        test("sin(θ)", Some(1.into()));
        test("cos(θ)", Some(1.into()));
        test("tan(θ)", Some(1.into()));
        test("sin(3/5θ)", Some(5.into()));
        test("sin(θ) + θ", None);
        test("r = sin(θ)", Some(1.into()));
        // TODO
        // test("sin(3θ/5)", Some(5.into()));
    }
}
