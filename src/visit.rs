use crate::{
    ast::{
        BinaryOp, Form, FormId, FormKind, NaryOp, RelOp, Term, TermId, TermKind, UnaryOp, VarSet,
        UNINIT_FORM_ID, UNINIT_TERM_ID,
    },
    interval_set::{Site, TupperIntervalSet},
    rel::{StaticForm, StaticFormKind, StaticTerm, StaticTermKind},
};
use inari::Decoration;
use std::{
    collections::{HashMap, HashSet},
    marker::Sized,
    mem::{swap, take},
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
        Nary(_, xs) => {
            for x in xs {
                v.visit_term(x);
            }
        }
        Constant(_) | Var(_) | Uninit => (),
    };
}

fn traverse_form<'a, V: Visit<'a>>(v: &mut V, f: &'a Form) {
    use FormKind::*;
    match &f.kind {
        Atomic(_, x, y) => {
            v.visit_term(x);
            v.visit_term(y);
        }
        Not(x) => {
            v.visit_form(x);
        }
        And(x, y) | Or(x, y) => {
            v.visit_form(x);
            v.visit_form(y);
        }
        Uninit => (),
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
        Nary(_, xs) => {
            for x in xs {
                v.visit_term_mut(x);
            }
        }
        Constant(_) | Var(_) | Uninit => (),
    };
}

fn traverse_form_mut<V: VisitMut>(v: &mut V, f: &mut Form) {
    use FormKind::*;
    match &mut f.kind {
        Atomic(_, x, y) => {
            v.visit_term_mut(x);
            v.visit_term_mut(y);
        }
        Not(x) => {
            v.visit_form_mut(x);
        }
        And(x, y) | Or(x, y) => {
            v.visit_form_mut(x);
            v.visit_form_mut(y);
        }
        Uninit => (),
    };
}

/// Replaces a - b with a + (-b) and does some special transformations.
pub struct PreTransform;

impl VisitMut for PreTransform {
    fn visit_term_mut(&mut self, t: &mut Term) {
        use {BinaryOp::*, TermKind::*, UnaryOp::*};
        traverse_term_mut(self, t);

        match &mut t.kind {
            Binary(Sub, x, y) => {
                // (Sub x y) → (Add x (Neg y))
                *t = Term::new(Binary(
                    Add,
                    take(x),
                    Box::new(Term::new(Unary(Neg, take(y)))),
                ));
            }
            // Ad-hoc transformations mainly for demonstrational purposes.
            Binary(Div, x, y) => {
                match (&x.kind, &y.kind) {
                    (Unary(Sin, x1), _) if x1 == y => {
                        // (Div (Sin y) y) → (Sinc (UndefAt0 y))
                        *t = Term::new(Unary(Sinc, Box::new(Term::new(Unary(UndefAt0, take(y))))));
                    }
                    (_, Unary(Sin, y1)) if y1 == x => {
                        // (Div x (Sin x)) → (Recip (Sinc (UndefAt0 x)))
                        *t = Term::new(Unary(
                            Recip,
                            Box::new(Term::new(Unary(
                                Sinc,
                                Box::new(Term::new(Unary(UndefAt0, take(x)))),
                            ))),
                        ));
                    }
                    _ if x == y => {
                        // (Div x x) → (One (UndefAt0 x))
                        *t = Term::new(Unary(One, Box::new(Term::new(Unary(UndefAt0, take(x))))));
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

fn precedes(x: &Term, y: &Term) -> bool {
    matches!(x.kind, TermKind::Constant(_)) && !matches!(y.kind, TermKind::Constant(_))
}

impl VisitMut for SortTerms {
    fn visit_term_mut(&mut self, t: &mut Term) {
        use {BinaryOp::*, TermKind::*};
        traverse_term_mut(self, t);

        match &mut t.kind {
            Binary(Add, x, y) | Binary(Mul, x, y) if precedes(y, x) => {
                // (op x y) /; y ≺ x → (op y x)
                swap(x, y);
                self.modified = true;
            }
            _ => (),
        }
    }
}

/// Transforms terms into simpler forms.
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
    fn visit_term_mut(&mut self, t: &mut Term) {
        use {BinaryOp::*, TermKind::*, UnaryOp::*};
        traverse_term_mut(self, t);

        match &mut t.kind {
            Unary(Neg, x) => {
                match &mut x.kind {
                    Unary(Neg, x1) => {
                        // (Neg (Neg x1)) → x1
                        *t = take(x1);
                        self.modified = true;
                    }
                    Binary(Add, x1, x2) => {
                        // (Neg (Add x1 x2)) → (Add (Neg x1) (Neg x2))
                        *t = Term::new(Binary(
                            Add,
                            Box::new(Term::new(Unary(Neg, take(x1)))),
                            Box::new(Term::new(Unary(Neg, take(x2)))),
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
                        *t = take(y);
                        self.modified = true;
                    }
                    (_, Binary(Add, y1, y2)) => {
                        // (Add x (Add y1 y2)) → (Add (Add x y1) y2)
                        *t = Term::new(Binary(
                            Add,
                            Box::new(Term::new(Binary(Add, take(x), take(y1)))),
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
                        *t = take(y);
                        self.modified = true;
                    }
                    (Constant(a), _) if f64(a) == Some(-1.0) => {
                        // (Mul -1 y) → (Neg y)
                        *t = Term::new(Unary(Neg, take(y)));
                        self.modified = true;
                    }
                    (Unary(Neg, x), _) => {
                        // (Mul (Neg x) y) → (Neg (Mul x y))
                        *t = Term::new(Unary(
                            Neg,
                            Box::new(Term::new(Binary(Mul, take(x), take(y)))),
                        ));
                        self.modified = true;
                    }
                    (_, Unary(Neg, y)) => {
                        // (Mul x (Neg y)) → (Neg (Mul x y))
                        *t = Term::new(Unary(
                            Neg,
                            Box::new(Term::new(Binary(Mul, take(x), take(y)))),
                        ));
                        self.modified = true;
                    }
                    (_, Binary(Mul, y1, y2)) => {
                        // (Mul x (Mul y1 y2)) → (Mul (Mul x y1) y2)
                        *t = Term::new(Binary(
                            Mul,
                            Box::new(Term::new(Binary(Mul, take(x), take(y1)))),
                            take(y2),
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
                        self.modified = true;
                    }
                }
            }
            Binary(_, x, y) => {
                if let (Constant(_), Constant(_)) = (&x.kind, &y.kind) {
                    let val = t.eval();
                    if val.len() <= 1 {
                        *t = Term::new(Constant(Box::new(val)));
                        self.modified = true;
                    }
                }
            }
            Pown(x, _) => {
                if let Constant(_) = &x.kind {
                    let val = t.eval();
                    if val.len() <= 1 {
                        *t = Term::new(Constant(Box::new(val)));
                        self.modified = true;
                    }
                }
            }
            Nary(_, xs) => {
                if xs.iter().all(|x| matches!(x.kind, Constant(_))) {
                    let val = t.eval();
                    if val.len() <= 1 {
                        *t = Term::new(Constant(Box::new(val)));
                        self.modified = true;
                    }
                }
            }
            _ => (),
        }
    }
}

/// Substitutes generic exponentiations with specialized functions.
pub struct PostTransform;

impl VisitMut for PostTransform {
    fn visit_term_mut(&mut self, t: &mut Term) {
        use {BinaryOp::*, TermKind::*, UnaryOp::*};
        traverse_term_mut(self, t);

        if let Binary(Pow, x, y) = &mut t.kind {
            match (&x.kind, &y.kind) {
                (Constant(a), _) => {
                    if let Some(a) = f64(a) {
                        if a == 2.0 {
                            // (Pow 2 x) → (Exp2 x)
                            *t = Term::new(Unary(Exp2, take(y)));
                        } else if a == 10.0 {
                            // (Pow 10 x) → (Exp10 x)
                            *t = Term::new(Unary(Exp10, take(y)));
                        }
                    }
                }
                (_, Constant(a)) => {
                    if let Some(a) = f64(a) {
                        if a == -1.0 {
                            // (Pow x -1) → (Recip x)
                            *t = Term::new(Unary(Recip, take(x)));
                        } else if a == 0.0 {
                            // (Pow x 0) → (One x)
                            *t = Term::new(Unary(One, take(x)));
                        } else if a == 0.5 {
                            // (Pow x 1/2) → (Sqrt x)
                            *t = Term::new(Unary(Sqrt, take(x)));
                        } else if a == 1.0 {
                            // (Pow x 1) → x
                            *t = take(x);
                        } else if a == 2.0 {
                            // (Pow x 2) → (Sqr x)
                            *t = Term::new(Unary(Sqr, take(x)));
                        } else if a == a as i32 as f64 {
                            // (Pow x a) /; a ∈ i32 → (Pown x a)
                            *t = Term::new(Pown(take(x), a as i32));
                        }
                    }
                }
                _ => (),
            }
        }
    }
}

/// Applies some normalization to formulas.
#[derive(Default)]
pub struct NormalizeForms {
    pub modified: bool,
}

impl VisitMut for NormalizeForms {
    fn visit_form_mut(&mut self, f: &mut Form) {
        use FormKind::*;
        traverse_form_mut(self, f);

        if let FormKind::Not(x) = &mut f.kind {
            match &mut x.kind {
                Atomic(op, x1, x2) => {
                    // (Not (op x1 x2)) → (!op x1 x2)
                    let neg_op = match op {
                        RelOp::Eq => RelOp::Neq,
                        RelOp::Ge => RelOp::Nge,
                        RelOp::Gt => RelOp::Ngt,
                        RelOp::Le => RelOp::Nle,
                        RelOp::Lt => RelOp::Nlt,
                        RelOp::Neq => RelOp::Eq,
                        RelOp::Nge => RelOp::Ge,
                        RelOp::Ngt => RelOp::Gt,
                        RelOp::Nle => RelOp::Le,
                        RelOp::Nlt => RelOp::Lt,
                    };
                    *f = Form::new(Atomic(neg_op, take(x1), take(x2)));
                    self.modified = true;
                }
                And(x1, x2) => {
                    // (And (x1 x2)) → (Or (Not x1) (Not x2))
                    *f = Form::new(Or(
                        Box::new(Form::new(Not(take(x1)))),
                        Box::new(Form::new(Not(take(x2)))),
                    ));
                    self.modified = true;
                }
                Or(x1, x2) => {
                    // (Or (x1 x2)) → (And (Not x1) (Not x2))
                    *f = Form::new(And(
                        Box::new(Form::new(Not(take(x1)))),
                        Box::new(Form::new(Not(take(x2)))),
                    ));
                    self.modified = true;
                }
                _ => (),
            };
        }
    }
}

/// Updates metadata of terms and formulas.
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

type SiteMap = HashMap<TermId, Site>;

/// Assigns [`TermId`]s to all of the terms, and assigns [`FormId`]s to the atomic formulas
/// so that they serve as indices in [`EvalResult`][`crate::eval_result::EvalResult`].
pub struct AssignIdStage1 {
    next_form_id: FormId,
    next_term_id: TermId,
    next_site: u8,
    site_map: SiteMap,
    visited_forms: HashSet<*const Form>,
    visited_terms: HashSet<*const Term>,
}

impl AssignIdStage1 {
    pub fn new() -> Self {
        AssignIdStage1 {
            next_form_id: 0,
            next_term_id: 0,
            next_site: 0,
            site_map: HashMap::new(),
            visited_forms: HashSet::new(),
            visited_terms: HashSet::new(),
        }
    }

    /// Returns `true` if the term can perform branch cut on evaluation.
    fn term_can_perform_cut(kind: &TermKind) -> bool {
        use {BinaryOp::*, NaryOp::*, TermKind::*, UnaryOp::*};
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
                | Pown(_, _)
                | Nary(RankedMax, _)
                | Nary(RankedMin, _)
        )
    }
}

impl VisitMut for AssignIdStage1 {
    fn visit_term_mut(&mut self, t: &mut Term) {
        traverse_term_mut(self, t);

        match self.visited_terms.get(&(t as *const Term)) {
            Some(visited) => {
                let id = unsafe { (**visited).id };
                t.id = id;

                if !self.site_map.contains_key(&id)
                    && Self::term_can_perform_cut(&t.kind)
                    && self.next_site <= Site::MAX
                {
                    self.site_map.insert(id, Site::new(self.next_site));
                    self.next_site += 1;
                }
            }
            _ => {
                assert!(self.next_term_id != UNINIT_TERM_ID);
                t.id = self.next_term_id;
                self.next_term_id += 1;
                self.visited_terms.insert(t);
            }
        }
    }

    fn visit_form_mut(&mut self, f: &mut Form) {
        traverse_form_mut(self, f);

        if let FormKind::Atomic(_, _, _) = f.kind {
            match self.visited_forms.get(&(f as *const Form)) {
                Some(visited) => {
                    f.id = unsafe { (**visited).id };
                }
                _ => {
                    assert!(self.next_form_id != UNINIT_FORM_ID);
                    f.id = self.next_form_id;
                    self.next_form_id += 1;
                    self.visited_forms.insert(f);
                }
            }
        }
    }
}

/// Assigns [`TermId`]s to the non-atomic formulas.
pub struct AssignIdStage2 {
    next_form_id: FormId,
    next_term_id: TermId,
    site_map: SiteMap,
    visited_forms: HashSet<*const Form>,
}

impl AssignIdStage2 {
    pub fn new(stage1: AssignIdStage1) -> Self {
        AssignIdStage2 {
            next_form_id: stage1.next_form_id,
            next_term_id: stage1.next_term_id,
            site_map: stage1.site_map,
            visited_forms: HashSet::new(),
        }
    }
}

impl VisitMut for AssignIdStage2 {
    fn visit_form_mut(&mut self, f: &mut Form) {
        traverse_form_mut(self, f);

        if !matches!(f.kind, FormKind::Atomic(_, _, _)) {
            match self.visited_forms.get(&(f as *const Form)) {
                Some(visited) => {
                    f.id = unsafe { (**visited).id };
                }
                _ => {
                    assert!(self.next_form_id != UNINIT_FORM_ID);
                    f.id = self.next_form_id;
                    self.next_form_id += 1;
                    self.visited_forms.insert(f);
                }
            }
        }
    }
}

/// Collects [`StaticTerm`]s and [`StaticForm`]s in the topological order.
pub struct CollectStatic {
    site_map: SiteMap,
    terms: Vec<Option<StaticTerm>>,
    forms: Vec<Option<StaticForm>>,
}

impl CollectStatic {
    pub fn new(stage2: AssignIdStage2) -> Self {
        Self {
            site_map: stage2.site_map,
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

        let i = t.id as usize;
        if self.terms[i].is_none() {
            self.terms[i] = Some(StaticTerm {
                site: self.site_map.get(&t.id).copied(),
                kind: match &t.kind {
                    Constant(x) => StaticTermKind::Constant(x.clone()),
                    Var(x) if x == "x" => StaticTermKind::X,
                    Var(x) if x == "y" => StaticTermKind::Y,
                    Unary(op, x) => StaticTermKind::Unary(*op, x.id),
                    Binary(op, x, y) => StaticTermKind::Binary(*op, x.id, y.id),
                    Pown(x, y) => StaticTermKind::Pown(x.id, *y),
                    Nary(op, xs) => {
                        StaticTermKind::Nary(*op, Box::new(xs.iter().map(|x| x.id).collect()))
                    }
                    Var(_) | Uninit => panic!(),
                },
                vars: t.vars,
            });
        }
    }

    fn visit_form(&mut self, f: &'a Form) {
        use FormKind::*;
        traverse_form(self, f);

        let i = f.id as usize;
        if self.forms[i].is_none() {
            self.forms[i] = Some(StaticForm {
                kind: match &f.kind {
                    Atomic(op, x, y) => StaticFormKind::Atomic(*op, x.id, y.id),
                    And(x, y) => StaticFormKind::And(x.id, y.id),
                    Or(x, y) => StaticFormKind::Or(x.id, y.id),
                    Not(_) | Uninit => panic!(),
                },
            });
        }
    }
}

/// Collects the ids of maximal sub-terms that contain exactly one free variable.
/// Terms of kind [`TermKind::Var`] are excluded from collection.
pub struct FindMaximalTerms {
    mx: Vec<TermId>,
    my: Vec<TermId>,
}

impl FindMaximalTerms {
    pub fn new() -> Self {
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

impl<'a> Visit<'a> for FindMaximalTerms {
    fn visit_term(&mut self, t: &'a Term) {
        match t.vars {
            VarSet::X => {
                if !matches!(t.kind, TermKind::Var(_)) {
                    self.mx.push(t.id);
                }
            }
            VarSet::Y => {
                if !matches!(t.kind, TermKind::Var(_)) {
                    self.my.push(t.id);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{context::Context, parse::parse};

    fn test_pre_transform(input: &str, expected: &str) {
        let mut f = parse(&format!("{} = 0", input), Context::builtin_context()).unwrap();
        PreTransform.visit_form_mut(&mut f);
        assert_eq!(
            format!("{}", f.dump_structure()),
            format!("(Eq {} @)", expected)
        );
    }

    #[test]
    fn pre_transform() {
        test_pre_transform("x - y", "(Add x (Neg y))");
        test_pre_transform("sin(x)/x", "(Sinc (UndefAt0 x))");
        test_pre_transform("x/sin(x)", "(Recip (Sinc (UndefAt0 x)))");
        test_pre_transform("x/x", "(One (UndefAt0 x))");
    }

    fn test_sort_terms(input: &str, expected: &str) {
        let mut f = parse(&format!("{} = 0", input), Context::builtin_context()).unwrap();
        let input = format!("{}", f.dump_structure());
        let mut v = SortTerms::default();
        v.visit_form_mut(&mut f);
        let output = format!("{}", f.dump_structure());
        assert_eq!(output, format!("(Eq {} @)", expected));
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
        let mut f = parse(&format!("{} = 0", input), Context::builtin_context()).unwrap();
        FoldConstant::default().visit_form_mut(&mut f);
        let input = format!("{}", f.dump_structure());
        let mut v = Transform::default();
        v.visit_form_mut(&mut f);
        let output = format!("{}", f.dump_structure());
        assert_eq!(output, format!("(Eq {} @)", expected));
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
    }

    fn test_post_transform(input: &str, expected: &str) {
        let mut f = parse(&format!("{} = 0", input), Context::builtin_context()).unwrap();
        FoldConstant::default().visit_form_mut(&mut f);
        PostTransform.visit_form_mut(&mut f);
        assert_eq!(
            format!("{}", f.dump_structure()),
            format!("(Eq {} @)", expected)
        );
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

    fn test_normalize_forms(input: &str, expected: &str) {
        let mut f = parse(input, Context::builtin_context()).unwrap();
        let input = format!("{}", f.dump_structure());
        let mut v = NormalizeForms::default();
        v.visit_form_mut(&mut f);
        let output = format!("{}", f.dump_structure());
        assert_eq!(output, expected);
        assert_eq!(v.modified, input != output);
    }

    #[test]
    fn normalize_forms() {
        test_normalize_forms("!x = y", "(Neq x y)");
        test_normalize_forms("!x <= y", "(Nle x y)");
        test_normalize_forms("!x < y", "(Nlt x y)");
        test_normalize_forms("!x >= y", "(Nge x y)");
        test_normalize_forms("!x > y", "(Ngt x y)");
        test_normalize_forms("!!x = y", "(Eq x y)");
        test_normalize_forms("!!x <= y", "(Le x y)");
        test_normalize_forms("!!x < y", "(Lt x y)");
        test_normalize_forms("!!x >= y", "(Ge x y)");
        test_normalize_forms("!!x > y", "(Gt x y)");
        test_normalize_forms("!(x = y && y = z)", "(Or (Not (Eq x y)) (Not (Eq y z)))");
        test_normalize_forms("!(x = y || y = z)", "(And (Not (Eq x y)) (Not (Eq y z)))");
    }
}
