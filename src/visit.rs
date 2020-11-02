use crate::{
    ast::{AxisSet, BinaryOp, Expr, ExprId, ExprKind, Rel, RelId, RelKind, UnaryOp},
    interval_set::{Site, MAX_SITE},
    rel::{StaticExpr, StaticExprKind, StaticRel, StaticRelKind},
};
use inari::const_dec_interval;
use std::{
    collections::{HashMap, HashSet},
    marker::Sized,
};

pub trait Visit<'a>
where
    Self: Sized,
{
    fn visit_expr(&mut self, expr: &'a Expr) {
        traverse_expr(self, expr);
    }

    fn visit_rel(&mut self, rel: &'a Rel) {
        traverse_rel(self, rel)
    }
}

fn traverse_expr<'a, V: Visit<'a>>(v: &mut V, expr: &'a Expr) {
    use ExprKind::*;
    match &expr.kind {
        Unary(_, x) => v.visit_expr(x),
        Binary(_, x, y) => {
            v.visit_expr(x);
            v.visit_expr(y);
        }
        Pown(x, _) => v.visit_expr(x),
        _ => (),
    };
}

fn traverse_rel<'a, V: Visit<'a>>(v: &mut V, rel: &'a Rel) {
    use RelKind::*;
    match &rel.kind {
        Atomic(_, x, y) => {
            v.visit_expr(x);
            v.visit_expr(y);
        }
        And(x, y) | Or(x, y) => {
            v.visit_rel(x);
            v.visit_rel(y);
        }
    };
}

pub trait VisitMut
where
    Self: Sized,
{
    fn visit_expr_mut(&mut self, expr: &mut Expr) {
        traverse_expr_mut(self, expr);
    }

    fn visit_rel_mut(&mut self, rel: &mut Rel) {
        traverse_rel_mut(self, rel);
    }
}

fn traverse_expr_mut<V: VisitMut>(v: &mut V, expr: &mut Expr) {
    use ExprKind::*;
    match &mut expr.kind {
        Unary(_, x) => v.visit_expr_mut(x),
        Binary(_, x, y) => {
            v.visit_expr_mut(x);
            v.visit_expr_mut(y);
        }
        Pown(x, _) => v.visit_expr_mut(x),
        _ => (),
    };
}

fn traverse_rel_mut<V: VisitMut>(v: &mut V, rel: &mut Rel) {
    use RelKind::*;
    match &mut rel.kind {
        Atomic(_, x, y) => {
            v.visit_expr_mut(x);
            v.visit_expr_mut(y);
        }
        And(x, y) | Or(x, y) => {
            v.visit_rel_mut(x);
            v.visit_rel_mut(y);
        }
    };
}

type SiteMap = HashMap<ExprId, Option<Site>>;

pub struct Transform;

impl VisitMut for Transform {
    fn visit_expr_mut(&mut self, expr: &mut Expr) {
        use {BinaryOp::*, ExprKind::*, UnaryOp::*};
        traverse_expr_mut(self, expr);

        match &mut expr.kind {
            Unary(Neg, x) => {
                if let Unary(Neg, x) = &mut x.kind {
                    // (Neg (Neg x)) => x
                    *expr = std::mem::take(x);
                }
            }
            Binary(Div, x, y) => {
                match (&x.kind, &y.kind) {
                    (Unary(Sin, z), _) if z == y => {
                        // (Div (Sin z) y) => (SinOverX y) if z == y
                        *expr = Expr::new(Unary(SinOverX, std::mem::take(y)));
                    }
                    (_, Unary(Sin, z)) if z == x => {
                        // (Div x (Sin z)) => (Recip (SinOverX x)) if z == x
                        *expr = Expr::new(Unary(
                            Recip,
                            Box::new(Expr::new(Unary(SinOverX, std::mem::take(x)))),
                        ));
                    }
                    _ => (),
                };
            }
            Binary(Pow, x, y) => {
                if let Constant(x) = &x.kind {
                    if x.len() == 1 {
                        let x = x.iter().next().unwrap().to_dec_interval();
                        if x == const_dec_interval!(2.0, 2.0) {
                            *expr = Expr::new(Unary(Exp2, std::mem::take(y)));
                        } else if x == const_dec_interval!(10.0, 10.0) {
                            *expr = Expr::new(Unary(Exp10, std::mem::take(y)));
                        }
                    }
                } else if let Constant(y) = &y.kind {
                    if y.len() == 1 {
                        let y = y.iter().next().unwrap().to_dec_interval();
                        // Do not transform x^0 to 1 as that can discard the decoration (e.g. sqrt(x)^0).
                        if y == const_dec_interval!(-1.0, -1.0) {
                            *expr = Expr::new(Unary(Recip, std::mem::take(x)));
                        } else if y == const_dec_interval!(0.5, 0.5) {
                            *expr = Expr::new(Unary(Sqrt, std::mem::take(x)));
                        } else if y == const_dec_interval!(1.0, 1.0) {
                            *expr = std::mem::take(x);
                        } else if y == const_dec_interval!(2.0, 2.0) {
                            *expr = Expr::new(Unary(Sqr, std::mem::take(x)));
                        } else if y.is_singleton() {
                            let y = y.inf();
                            let iy = y as i32;
                            if y == iy as f64 {
                                *expr = Expr::new(Pown(std::mem::take(x), iy));
                            }
                        }
                    }
                }
            }
            _ => (),
        }
    }
}

pub struct FoldConstant;

// Only fold constants which evaluate to an empty or a single interval
// since the sites are not assigned and branch cut tracking is not possible
// at this moment.
impl VisitMut for FoldConstant {
    fn visit_expr_mut(&mut self, expr: &mut Expr) {
        use ExprKind::*;
        traverse_expr_mut(self, expr);

        match &mut expr.kind {
            Unary(_, x) => {
                if let Constant(_) = &x.kind {
                    let val = expr.evaluate();
                    if val.len() <= 1 {
                        *expr = Expr::new(Constant(Box::new(val)));
                    }
                }
            }
            Binary(_, x, y) => {
                if let (Constant(_), Constant(_)) = (&x.kind, &y.kind) {
                    let val = expr.evaluate();
                    if val.len() <= 1 {
                        *expr = Expr::new(Constant(Box::new(val)));
                    }
                }
            }
            Pown(x, _) => {
                if let Constant(_) = &x.kind {
                    let val = expr.evaluate();
                    if val.len() <= 1 {
                        *expr = Expr::new(Constant(Box::new(val)));
                    }
                }
            }
            _ => (),
        }
    }
}

pub struct UpdateMetadata;

impl VisitMut for UpdateMetadata {
    fn visit_expr_mut(&mut self, expr: &mut Expr) {
        traverse_expr_mut(self, expr);
        expr.update_metadata();
    }
}

// Does the following tasks:
// - Assign ids to `Expr`s.
// - Assign ids to atomic `Rel`s so that they can be used as indices for `EvalResult`.
pub struct AssignIdStage1<'a> {
    next_expr_id: ExprId,
    next_site: Site,
    site_map: SiteMap,
    visited_exprs: HashSet<&'a Expr>,
    next_rel_id: RelId,
    visited_rels: HashSet<&'a Rel>,
}

impl<'a> AssignIdStage1<'a> {
    pub fn new() -> Self {
        AssignIdStage1 {
            next_expr_id: 0,
            next_site: 0,
            site_map: HashMap::new(),
            visited_exprs: HashSet::new(),
            next_rel_id: 0,
            visited_rels: HashSet::new(),
        }
    }

    fn expr_can_perform_cut(kind: &ExprKind) -> bool {
        use {BinaryOp::*, ExprKind::*, UnaryOp::*};
        matches!(kind,
            Unary(Ceil, _)
            | Unary(Floor, _)
            | Unary(Recip, _)
            | Unary(Sign, _)
            | Unary(Tan, _)
            | Binary(Atan2, _, _)
            | Binary(Div, _, _)
            | Binary(Log, _, _)
            | Binary(Mod, _, _)
            | Binary(Pow, _, _)
            | Pown(_, _))
    }
}

impl<'a> Visit<'a> for AssignIdStage1<'a> {
    fn visit_expr(&mut self, expr: &'a Expr) {
        traverse_expr(self, expr);

        match self.visited_exprs.get(expr) {
            Some(visited) => {
                let id = visited.id.get();
                expr.id.set(id);

                if let Some(site) = self.site_map.get_mut(&id) {
                    if site.is_none() && self.next_site <= MAX_SITE {
                        *site = Some(self.next_site);
                        self.next_site += 1;
                    }
                }
            }
            _ => {
                let id = self.next_expr_id;
                expr.id.set(id);
                self.next_expr_id += 1;

                if Self::expr_can_perform_cut(&expr.kind) {
                    self.site_map.insert(id, None);
                }

                self.visited_exprs.insert(expr);
            }
        }
    }

    fn visit_rel(&mut self, rel: &'a Rel) {
        traverse_rel(self, rel);

        if let RelKind::Atomic(_, _, _) = rel.kind {
            match self.visited_rels.get(rel) {
                Some(visited) => {
                    rel.id.set(visited.id.get());
                }
                _ => {
                    rel.id.set(self.next_rel_id);
                    self.next_rel_id += 1;
                    self.visited_rels.insert(rel);
                }
            }
        }
    }
}

// Does the following tasks:
// - Assign sites to `Expr`s.
// - Assign ids to the rest of the `Rel`s.
pub struct AssignIdStage2<'a> {
    next_expr_id: ExprId,
    site_map: SiteMap,
    next_rel_id: RelId,
    visited_rels: HashSet<&'a Rel>,
}

impl<'a> AssignIdStage2<'a> {
    pub fn new(stage1: AssignIdStage1<'a>) -> Self {
        AssignIdStage2 {
            next_expr_id: stage1.next_expr_id,
            site_map: stage1.site_map,
            next_rel_id: stage1.next_rel_id,
            visited_rels: stage1.visited_rels,
        }
    }
}

impl<'a> Visit<'a> for AssignIdStage2<'a> {
    fn visit_expr(&mut self, expr: &'a Expr) {
        traverse_expr(self, expr);

        if let Some(site) = self.site_map.get(&expr.id.get()) {
            expr.site.set(*site);
        }
    }

    fn visit_rel(&mut self, rel: &'a Rel) {
        traverse_rel(self, rel);

        match self.visited_rels.get(rel) {
            Some(visited) => {
                rel.id.set(visited.id.get());
            }
            _ => {
                rel.id.set(self.next_rel_id);
                self.next_rel_id += 1;
                self.visited_rels.insert(rel);
            }
        }
    }
}

// Collects `StaticExpr`s and `StaticRel`s in the topological order.
pub struct CollectStatic {
    exprs: Vec<Option<StaticExpr>>,
    rels: Vec<Option<StaticRel>>,
}

impl CollectStatic {
    pub fn new(stage2: AssignIdStage2) -> Self {
        Self {
            exprs: vec![None; stage2.next_expr_id as usize],
            rels: vec![None; stage2.next_rel_id as usize],
        }
    }

    pub fn exprs_rels(self) -> (Vec<StaticExpr>, Vec<StaticRel>) {
        (
            self.exprs.into_iter().collect::<Option<Vec<_>>>().unwrap(),
            self.rels.into_iter().collect::<Option<Vec<_>>>().unwrap(),
        )
    }
}

impl<'a> Visit<'a> for CollectStatic {
    fn visit_expr(&mut self, expr: &'a Expr) {
        use ExprKind::*;
        traverse_expr(self, expr);

        let i = expr.id.get() as usize;
        if self.exprs[i].is_none() {
            self.exprs[i] = Some(StaticExpr {
                site: expr.site.get(),
                kind: match &expr.kind {
                    Constant(x) => StaticExprKind::Constant(x.clone()),
                    X => StaticExprKind::X,
                    Y => StaticExprKind::Y,
                    Unary(op, x) => StaticExprKind::Unary(*op, x.id.get()),
                    Binary(op, x, y) => StaticExprKind::Binary(*op, x.id.get(), y.id.get()),
                    Pown(x, y) => StaticExprKind::Pown(x.id.get(), *y),
                    Uninit => panic!(),
                },
                dependent_axes: expr.dependent_axes,
            });
        }
    }

    fn visit_rel(&mut self, rel: &'a Rel) {
        use RelKind::*;
        traverse_rel(self, rel);

        let i = rel.id.get() as usize;
        if self.rels[i].is_none() {
            self.rels[i] = Some(StaticRel {
                kind: match &rel.kind {
                    Atomic(op, x, y) => StaticRelKind::Atomic(*op, x.id.get(), y.id.get()),
                    And(x, y) => StaticRelKind::And(x.id.get(), y.id.get()),
                    Or(x, y) => StaticRelKind::Or(x.id.get(), y.id.get()),
                },
            });
        }
    }
}

// Collects ids of maximal subexpressions that depend only on X or Y for caching.
// Atomic expressions X and Y are excluded.
pub struct FindMaxima {
    mx: Vec<ExprId>,
    my: Vec<ExprId>,
}

impl FindMaxima {
    pub fn new() -> FindMaxima {
        Self {
            mx: Vec::new(),
            my: Vec::new(),
        }
    }

    pub fn mx_my(mut self) -> (Vec<ExprId>, Vec<ExprId>) {
        self.mx.sort();
        self.mx.dedup();
        self.my.sort();
        self.my.dedup();
        (self.mx, self.my)
    }
}

impl<'a> Visit<'a> for FindMaxima {
    fn visit_expr(&mut self, expr: &'a Expr) {
        match expr.dependent_axes {
            AxisSet::X => {
                if !matches!(expr.kind, ExprKind::X) {
                    self.mx.push(expr.id.get());
                }
            }
            AxisSet::Y => {
                if !matches!(expr.kind, ExprKind::Y) {
                    self.my.push(expr.id.get());
                }
            }
            AxisSet::XY => traverse_expr(self, expr),
            _ => (),
        }
    }

    fn visit_rel(&mut self, rel: &'a Rel) {
        traverse_rel(self, rel);
    }
}
