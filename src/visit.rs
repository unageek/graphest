use crate::{
    ast::{BinaryOp, Form, FormId, FormKind, Term, TermId, TermKind, UnaryOp, VarSet},
    interval_set::{Site, TupperIntervalSet},
    rel::{StaticForm, StaticFormKind, StaticTerm, StaticTermKind},
};
use inari::Decoration;
use std::{
    collections::{HashMap, HashSet},
    marker::Sized,
};

/// A type that traverses formulas and terms in depth-first order.
pub trait Visit<'a>
where
    Self: Sized,
{
    fn visit_term(&mut self, t: &'a Term) {
        traverse_term(self, t);
    }

    fn visit_form(&mut self, f: &'a Form) {
        traverse_form(self, f)
    }
}

fn traverse_term<'a, V: Visit<'a>>(v: &mut V, t: &'a Term) {
    use TermKind::*;
    match &t.kind {
        Unary(_, x) => v.visit_term(x),
        Binary(_, x, y) => {
            v.visit_term(x);
            v.visit_term(y);
        }
        Pown(x, _) => v.visit_term(x),
        _ => (),
    };
}

fn traverse_form<'a, V: Visit<'a>>(v: &mut V, f: &'a Form) {
    use FormKind::*;
    match &f.kind {
        Atomic(_, x, y) => {
            v.visit_term(x);
            v.visit_term(y);
        }
        And(x, y) | Or(x, y) => {
            v.visit_form(x);
            v.visit_form(y);
        }
    };
}

/// A type that traverses formulas and terms and possibly modifies them
/// in depth-first order.
pub trait VisitMut
where
    Self: Sized,
{
    fn visit_term_mut(&mut self, t: &mut Term) {
        traverse_term_mut(self, t);
    }

    fn visit_form_mut(&mut self, f: &mut Form) {
        traverse_form_mut(self, f);
    }
}

fn traverse_term_mut<V: VisitMut>(v: &mut V, t: &mut Term) {
    use TermKind::*;
    match &mut t.kind {
        Unary(_, x) => v.visit_term_mut(x),
        Binary(_, x, y) => {
            v.visit_term_mut(x);
            v.visit_term_mut(y);
        }
        Pown(x, _) => v.visit_term_mut(x),
        _ => (),
    };
}

fn traverse_form_mut<V: VisitMut>(v: &mut V, f: &mut Form) {
    use FormKind::*;
    match &mut f.kind {
        Atomic(_, x, y) => {
            v.visit_term_mut(x);
            v.visit_term_mut(y);
        }
        And(x, y) | Or(x, y) => {
            v.visit_form_mut(x);
            v.visit_form_mut(y);
        }
    };
}

type SiteMap = HashMap<TermId, Option<Site>>;

/// Transforms terms into more efficient forms.
pub struct Transform;

fn f64(x: &TupperIntervalSet) -> Option<f64> {
    if x.len() != 1 {
        return None;
    }

    let x = x.iter().next().unwrap().to_dec_interval();
    if x.is_singleton() && x.decoration_part() >= Decoration::Dac {
        Some(x.inf())
    } else {
        None
    }
}

impl VisitMut for Transform {
    fn visit_term_mut(&mut self, t: &mut Term) {
        use {std::mem::take, BinaryOp::*, TermKind::*, UnaryOp::*};
        traverse_term_mut(self, t);

        match &mut t.kind {
            Unary(Neg, x) => {
                if let Unary(Neg, x) = &mut x.kind {
                    // (Neg (Neg x)) => x
                    *t = take(x);
                }
            }
            Binary(Add, x, y) => {
                if let Constant(x) = &x.kind {
                    if f64(x) == Some(0.0) {
                        // (Add 0 y) => y
                        *t = take(y);
                    }
                } else if let Constant(y) = &y.kind {
                    if f64(y) == Some(0.0) {
                        // (Add x 0) => x
                        *t = take(x);
                    }
                }
            }
            Binary(Mul, x, y) => {
                if let Constant(x) = &x.kind {
                    if f64(x) == Some(1.0) {
                        // (Mul 1 y) => y
                        *t = take(y);
                    }
                } else if let Constant(y) = &y.kind {
                    if f64(y) == Some(1.0) {
                        // (Mul x 1) => x
                        *t = take(x);
                    }
                }
            }
            Binary(Div, x, y) => {
                match (&x.kind, &y.kind) {
                    _ if x == y => {
                        // (Div x y) => (XOverX x) if x == y
                        *t = Term::new(Unary(XOverX, take(x)));
                    }
                    (Unary(Sin, z), _) if z == y => {
                        // (Div (Sin z) y) => (SinXOverX y) if z == y
                        *t = Term::new(Unary(SinXOverX, take(y)));
                    }
                    (_, Unary(Sin, z)) if z == x => {
                        // (Div x (Sin z)) => (Recip (SinXOverX x)) if z == x
                        *t =
                            Term::new(Unary(Recip, Box::new(Term::new(Unary(SinXOverX, take(x))))));
                    }
                    _ => (),
                };
            }
            Binary(Pow, x, y) => {
                if let Constant(x) = &x.kind {
                    if let Some(x) = f64(x) {
                        if x == 2.0 {
                            // (Pow 2 x) => (Exp2 x)
                            *t = Term::new(Unary(Exp2, take(y)));
                        } else if x == 10.0 {
                            // (Pow 10 x) => (Exp10 x)
                            *t = Term::new(Unary(Exp10, take(y)));
                        }
                    }
                } else if let Constant(y) = &y.kind {
                    if let Some(y) = f64(y) {
                        // Do not transform x^0 to 1 as that can discard the decoration (e.g. sqrt(x)^0).
                        if y == -1.0 {
                            // (Pow x -1) => (Recip x)
                            *t = Term::new(Unary(Recip, take(x)));
                        } else if y == 0.5 {
                            // (Pow x 1/2) => (Sqrt x)
                            *t = Term::new(Unary(Sqrt, take(x)));
                        } else if y == 1.0 {
                            // (Pow x 1) => x
                            *t = take(x);
                        } else if y == 2.0 {
                            // (Pow x 2) => (Sqr x)
                            *t = Term::new(Unary(Sqr, take(x)));
                        } else if y == y as i32 as f64 {
                            // (Pow x y) => (Pown x y) if y ∈ i32
                            *t = Term::new(Pown(take(x), y as i32));
                        }
                    }
                }
            }
            _ => (),
        }
    }
}

/// Performs constant folding.
pub struct FoldConstant;

// Only fold constants which evaluate to an empty or a single interval
// since the sites are not assigned and branch cut tracking is not possible
// at this moment.
impl VisitMut for FoldConstant {
    fn visit_term_mut(&mut self, t: &mut Term) {
        use TermKind::*;
        traverse_term_mut(self, t);

        match &mut t.kind {
            Unary(_, x) => {
                if let Constant(_) = &x.kind {
                    let val = t.eval();
                    if val.len() <= 1 {
                        *t = Term::new(Constant(Box::new(val)));
                    }
                }
            }
            Binary(_, x, y) => {
                if let (Constant(_), Constant(_)) = (&x.kind, &y.kind) {
                    let val = t.eval();
                    if val.len() <= 1 {
                        *t = Term::new(Constant(Box::new(val)));
                    }
                }
            }
            Pown(x, _) => {
                if let Constant(_) = &x.kind {
                    let val = t.eval();
                    if val.len() <= 1 {
                        *t = Term::new(Constant(Box::new(val)));
                    }
                }
            }
            _ => (),
        }
    }
}

/// Calls [`Term::update_metadata`]/[`Form::update_metadata`] on each term/formula
/// in the topological order.
pub struct UpdateMetadata;

impl VisitMut for UpdateMetadata {
    fn visit_term_mut(&mut self, t: &mut Term) {
        traverse_term_mut(self, t);
        t.update_metadata();
    }

    fn visit_form_mut(&mut self, f: &mut Form) {
        traverse_form_mut(self, f);
        f.update_metadata();
    }
}

/// Assigns [`TermId`]s to all of the terms, and assigns [`FormId`]s to the atomic formulas
/// so that they serve as indices in [`EvalResult`][`crate::eval_result::EvalResult`].
pub struct AssignIdStage1<'a> {
    next_term_id: TermId,
    next_site: u8,
    site_map: SiteMap,
    visited_terms: HashSet<&'a Term>,
    next_form_id: FormId,
    visited_forms: HashSet<&'a Form>,
}

impl<'a> AssignIdStage1<'a> {
    pub fn new() -> Self {
        AssignIdStage1 {
            next_term_id: 0,
            next_site: 0,
            site_map: HashMap::new(),
            visited_terms: HashSet::new(),
            next_form_id: 0,
            visited_forms: HashSet::new(),
        }
    }

    /// Returns `true` if the term can perform branch cut on evaluation.
    fn term_can_perform_cut(kind: &TermKind) -> bool {
        use {BinaryOp::*, TermKind::*, UnaryOp::*};
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
    fn visit_term(&mut self, t: &'a Term) {
        traverse_term(self, t);

        match self.visited_terms.get(t) {
            Some(visited) => {
                let id = visited.id.get();
                t.id.set(id);

                if let Some(site) = self.site_map.get_mut(&id) {
                    if site.is_none() && self.next_site <= Site::MAX {
                        *site = Some(Site::new(self.next_site));
                        self.next_site += 1;
                    }
                }
            }
            _ => {
                let id = self.next_term_id;
                t.id.set(id);
                self.next_term_id += 1;

                if Self::term_can_perform_cut(&t.kind) {
                    self.site_map.insert(id, None);
                }

                self.visited_terms.insert(t);
            }
        }
    }

    fn visit_form(&mut self, f: &'a Form) {
        traverse_form(self, f);

        if let FormKind::Atomic(_, _, _) = f.kind {
            match self.visited_forms.get(f) {
                Some(visited) => {
                    f.id.set(visited.id.get());
                }
                _ => {
                    f.id.set(self.next_form_id);
                    self.next_form_id += 1;
                    self.visited_forms.insert(f);
                }
            }
        }
    }
}

/// Assigns [`Site`]s to the terms if they are necessary for branch cut tracking,
/// and assigns [`TermId`]s to the non-atomic formulas.
pub struct AssignIdStage2<'a> {
    next_term_id: TermId,
    site_map: SiteMap,
    next_form_id: FormId,
    visited_forms: HashSet<&'a Form>,
}

impl<'a> AssignIdStage2<'a> {
    pub fn new(stage1: AssignIdStage1<'a>) -> Self {
        AssignIdStage2 {
            next_term_id: stage1.next_term_id,
            site_map: stage1.site_map,
            next_form_id: stage1.next_form_id,
            visited_forms: HashSet::new(),
        }
    }
}

impl<'a> Visit<'a> for AssignIdStage2<'a> {
    fn visit_term(&mut self, t: &'a Term) {
        traverse_term(self, t);

        if let Some(site) = self.site_map.get(&t.id.get()) {
            t.site.set(*site);
        }
    }

    fn visit_form(&mut self, f: &'a Form) {
        traverse_form(self, f);

        if !matches!(f.kind, FormKind::Atomic(_, _, _)) {
            match self.visited_forms.get(f) {
                Some(visited) => {
                    f.id.set(visited.id.get());
                }
                _ => {
                    f.id.set(self.next_form_id);
                    self.next_form_id += 1;
                    self.visited_forms.insert(f);
                }
            }
        }
    }
}

/// Collects [`StaticTerm`]s and [`StaticForm`]s in the topological order.
pub struct CollectStatic {
    terms: Vec<Option<StaticTerm>>,
    forms: Vec<Option<StaticForm>>,
}

impl CollectStatic {
    pub fn new(stage2: AssignIdStage2) -> Self {
        Self {
            terms: vec![None; stage2.next_term_id as usize],
            forms: vec![None; stage2.next_form_id as usize],
        }
    }

    /// Returns the collected terms and formulas.
    pub fn terms_forms(self) -> (Vec<StaticTerm>, Vec<StaticForm>) {
        (
            self.terms.into_iter().collect::<Option<Vec<_>>>().unwrap(),
            self.forms.into_iter().collect::<Option<Vec<_>>>().unwrap(),
        )
    }
}

impl<'a> Visit<'a> for CollectStatic {
    fn visit_term(&mut self, t: &'a Term) {
        use TermKind::*;
        traverse_term(self, t);

        let i = t.id.get() as usize;
        if self.terms[i].is_none() {
            self.terms[i] = Some(StaticTerm {
                site: t.site.get(),
                kind: match &t.kind {
                    Constant(x) => StaticTermKind::Constant(x.clone()),
                    X => StaticTermKind::X,
                    Y => StaticTermKind::Y,
                    Unary(op, x) => StaticTermKind::Unary(*op, x.id.get()),
                    Binary(op, x, y) => StaticTermKind::Binary(*op, x.id.get(), y.id.get()),
                    Pown(x, y) => StaticTermKind::Pown(x.id.get(), *y),
                    Uninit => panic!(),
                },
                vars: t.vars,
            });
        }
    }

    fn visit_form(&mut self, f: &'a Form) {
        use FormKind::*;
        traverse_form(self, f);

        let i = f.id.get() as usize;
        if self.forms[i].is_none() {
            self.forms[i] = Some(StaticForm {
                kind: match &f.kind {
                    Atomic(op, x, y) => StaticFormKind::Atomic(*op, x.id.get(), y.id.get()),
                    And(x, y) => StaticFormKind::And(x.id.get(), y.id.get()),
                    Or(x, y) => StaticFormKind::Or(x.id.get(), y.id.get()),
                },
            });
        }
    }
}

/// Collects the ids of maximal sub-terms that contain exactly one free variable.
/// Terms of kind [`TermKind::X`] and [`TermKind::Y`] are excluded from collection.
pub struct FindMaxima {
    mx: Vec<TermId>,
    my: Vec<TermId>,
}

impl FindMaxima {
    pub fn new() -> FindMaxima {
        Self {
            mx: Vec::new(),
            my: Vec::new(),
        }
    }

    pub fn mx_my(mut self) -> (Vec<TermId>, Vec<TermId>) {
        self.mx.sort_unstable();
        self.mx.dedup();
        self.my.sort_unstable();
        self.my.dedup();
        (self.mx, self.my)
    }
}

impl<'a> Visit<'a> for FindMaxima {
    fn visit_term(&mut self, t: &'a Term) {
        match t.vars {
            VarSet::X => {
                if !matches!(t.kind, TermKind::X) {
                    self.mx.push(t.id.get());
                }
            }
            VarSet::Y => {
                if !matches!(t.kind, TermKind::Y) {
                    self.my.push(t.id.get());
                }
            }
            VarSet::XY => traverse_term(self, t),
            _ => (),
        }
    }

    fn visit_form(&mut self, f: &'a Form) {
        traverse_form(self, f);
    }
}
