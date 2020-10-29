use crate::{
    interval_set::{DecSignSet, SignSet},
    rel::{StaticRel, StaticRelKind},
};
use core::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign};
use inari::Decoration;
use smallvec::{smallvec, SmallVec};
use std::mem::size_of;

#[derive(Clone, Debug)]
pub struct EvalResult(pub SmallVec<[DecSignSet; 32]>);

impl EvalResult {
    pub fn size_in_heap(&self) -> usize {
        if self.0.spilled() {
            self.0.capacity() * size_of::<DecSignSet>()
        } else {
            0
        }
    }

    pub fn map<F>(&self, rels: &[StaticRel], f: &F) -> EvalResultMask
    where
        F: Fn(SignSet, Decoration) -> bool,
    {
        let mut m = EvalResultMask(smallvec![false; self.0.len()]);
        Self::map_impl(&self.0[..], rels, rels.len() - 1, f, &mut m.0[..]);
        m
    }

    #[allow(clippy::many_single_char_names)]
    fn map_impl<F>(slf: &[DecSignSet], rels: &[StaticRel], i: usize, f: &F, m: &mut [bool])
    where
        F: Fn(SignSet, Decoration) -> bool,
    {
        use StaticRelKind::*;
        match &rels[i].kind {
            Atomic(_, _, _) => {
                m[i] = f(slf[i].0, slf[i].1);
            }
            And(x, y) => {
                Self::map_impl(&slf, rels, *x as usize, f, m);
                Self::map_impl(&slf, rels, *y as usize, f, m);
            }
            Or(x, y) => {
                Self::map_impl(&slf, rels, *x as usize, f, m);
                Self::map_impl(&slf, rels, *y as usize, f, m);
            }
        }
    }

    pub fn map_reduce<F>(&self, rels: &[StaticRel], f: &F) -> bool
    where
        F: Fn(SignSet, Decoration) -> bool,
    {
        Self::map_reduce_impl(&self.0[..], rels, rels.len() - 1, f)
    }

    fn map_reduce_impl<F>(slf: &[DecSignSet], rels: &[StaticRel], i: usize, f: &F) -> bool
    where
        F: Fn(SignSet, Decoration) -> bool,
    {
        use StaticRelKind::*;
        match &rels[i].kind {
            Atomic(_, _, _) => f(slf[i].0, slf[i].1),
            And(x, y) => {
                Self::map_reduce_impl(&slf, rels, *x as usize, f)
                    && Self::map_reduce_impl(&slf, rels, *y as usize, f)
            }
            Or(x, y) => {
                Self::map_reduce_impl(&slf, rels, *x as usize, f)
                    || Self::map_reduce_impl(&slf, rels, *y as usize, f)
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct EvalResultMask(pub SmallVec<[bool; 32]>);

impl EvalResultMask {
    pub fn reduce(&self, rels: &[StaticRel]) -> bool {
        Self::reduce_impl(&self.0[..], rels, rels.len() - 1)
    }

    fn reduce_impl(slf: &[bool], rels: &[StaticRel], i: usize) -> bool {
        use StaticRelKind::*;
        match &rels[i].kind {
            Atomic(_, _, _) => slf[i],
            And(x, y) => {
                Self::reduce_impl(&slf, rels, *x as usize)
                    && Self::reduce_impl(&slf, rels, *y as usize)
            }
            Or(x, y) => {
                Self::reduce_impl(&slf, rels, *x as usize)
                    || Self::reduce_impl(&slf, rels, *y as usize)
            }
        }
    }

    pub fn solution_certainly_exists(&self, rels: &[StaticRel], locally_zero_mask: &Self) -> bool {
        Self::solution_certainly_exists_impl(
            &self.0[..],
            rels,
            rels.len() - 1,
            &locally_zero_mask.0[..],
        )
    }

    fn solution_certainly_exists_impl(
        slf: &[bool],
        rels: &[StaticRel],
        i: usize,
        locally_zero_mask: &[bool],
    ) -> bool {
        use StaticRelKind::*;
        match &rels[i].kind {
            Atomic(_, _, _) => slf[i],
            And(x, y) => {
                if Self::reduce_impl(&locally_zero_mask, rels, *x as usize) {
                    Self::solution_certainly_exists_impl(
                        &slf,
                        rels,
                        *y as usize,
                        &locally_zero_mask,
                    )
                } else if Self::reduce_impl(&locally_zero_mask, rels, *y as usize) {
                    Self::solution_certainly_exists_impl(
                        &slf,
                        rels,
                        *x as usize,
                        &locally_zero_mask,
                    )
                } else {
                    // Cannot tell the existence of a solution by a normal conjunction.
                    false
                }
            }
            Or(x, y) => {
                Self::solution_certainly_exists_impl(&slf, rels, *x as usize, &locally_zero_mask)
                    || Self::solution_certainly_exists_impl(
                        &slf,
                        rels,
                        *y as usize,
                        &locally_zero_mask,
                    )
            }
        }
    }
}

impl BitAnd for &EvalResultMask {
    type Output = EvalResultMask;

    fn bitand(self, rhs: Self) -> Self::Output {
        #[allow(clippy::suspicious_arithmetic_impl)]
        EvalResultMask(
            self.0
                .iter()
                .zip(rhs.0.iter())
                .map(|(x, y)| *x && *y)
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
        #[allow(clippy::suspicious_arithmetic_impl)]
        EvalResultMask(
            self.0
                .iter()
                .zip(rhs.0.iter())
                .map(|(x, y)| *x || *y)
                .collect(),
        )
    }
}

impl BitOrAssign for EvalResultMask {
    fn bitor_assign(&mut self, rhs: Self) {
        *self = self.bitor(&rhs)
    }
}
