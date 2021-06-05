use crate::{
    interval_set::DecSignSet,
    ops::{StaticForm, StaticFormKind},
};
use core::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign};
use smallvec::SmallVec;
use std::mem::size_of;

/// A sequence of evaluation results of atomic formulas.
#[derive(Clone, Debug)]
pub struct EvalResult(pub SmallVec<[DecSignSet; 32]>);

impl EvalResult {
    /// Applies the given boolean-valued function on each result.
    pub fn map<F>(&self, f: F) -> EvalResultMask
    where
        F: Fn(DecSignSet) -> bool,
    {
        EvalResultMask(self.0.iter().copied().map(f).collect())
    }

    /// Returns the size allocated by the [`EvalResult`] in bytes.
    pub fn size_in_heap(&self) -> usize {
        if self.0.spilled() {
            self.0.capacity() * size_of::<DecSignSet>()
        } else {
            0
        }
    }
}

/// A sequence of Boolean values assigned to atomic formulas.
#[derive(Clone, Debug)]
pub struct EvalResultMask(pub SmallVec<[bool; 32]>);

impl EvalResultMask {
    /// Evaluates the last formula to a Boolean value.
    pub fn eval(&self, forms: &[StaticForm]) -> bool {
        Self::eval_impl(&self.0[..], forms, forms.len() - 1)
    }

    fn eval_impl(slf: &[bool], forms: &[StaticForm], i: usize) -> bool {
        use StaticFormKind::*;
        match &forms[i].kind {
            Atomic(_, _) => slf[i],
            And(x, y) => {
                Self::eval_impl(slf, forms, *x as usize) && Self::eval_impl(slf, forms, *y as usize)
            }
            Or(x, y) => {
                Self::eval_impl(slf, forms, *x as usize) || Self::eval_impl(slf, forms, *y as usize)
            }
        }
    }

    /// Returns `true` if the existence of a solution is concluded by the arguments.
    /// See the actual usage for details.
    pub fn solution_certainly_exists(
        &self,
        forms: &[StaticForm],
        locally_zero_mask: &Self,
    ) -> bool {
        Self::solution_certainly_exists_impl(
            &self.0[..],
            forms,
            forms.len() - 1,
            &locally_zero_mask.0[..],
        )
    }

    fn solution_certainly_exists_impl(
        slf: &[bool],
        forms: &[StaticForm],
        i: usize,
        locally_zero_mask: &[bool],
    ) -> bool {
        use StaticFormKind::*;
        match &forms[i].kind {
            Atomic(_, _) => slf[i],
            And(x, y) => {
                if Self::eval_impl(locally_zero_mask, forms, *x as usize) {
                    Self::solution_certainly_exists_impl(slf, forms, *y as usize, locally_zero_mask)
                } else if Self::eval_impl(locally_zero_mask, forms, *y as usize) {
                    Self::solution_certainly_exists_impl(slf, forms, *x as usize, locally_zero_mask)
                } else {
                    // Cannot tell the existence of a solution by a normal conjunction.
                    false
                }
            }
            Or(x, y) => {
                Self::solution_certainly_exists_impl(slf, forms, *x as usize, locally_zero_mask)
                    || Self::solution_certainly_exists_impl(
                        slf,
                        forms,
                        *y as usize,
                        locally_zero_mask,
                    )
            }
        }
    }
}

impl BitAnd for &EvalResultMask {
    type Output = EvalResultMask;

    fn bitand(self, rhs: Self) -> Self::Output {
        assert_eq!(self.0.len(), rhs.0.len());
        EvalResultMask(
            self.0
                .iter()
                .zip(rhs.0.iter())
                .map(|(&x, &y)| x && y)
                .collect(),
        )
    }
}

impl BitAndAssign for EvalResultMask {
    fn bitand_assign(&mut self, rhs: Self) {
        *self = self.bitand(&rhs)
    }
}

impl BitOr for &EvalResultMask {
    type Output = EvalResultMask;

    fn bitor(self, rhs: Self) -> Self::Output {
        assert_eq!(self.0.len(), rhs.0.len());
        EvalResultMask(
            self.0
                .iter()
                .zip(rhs.0.iter())
                .map(|(&x, &y)| x || y)
                .collect(),
        )
    }
}

impl BitOrAssign for EvalResultMask {
    fn bitor_assign(&mut self, rhs: Self) {
        *self = self.bitor(&rhs)
    }
}
