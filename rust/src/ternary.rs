use std::ops::{BitAnd, BitOr, Not};
use Ternary::*;

/// A ternary value which could be either [`False`], [`Uncertain`], or [`True`].
///
/// The values are ordered as: [`False`] < [`Uncertain`] < [`True`].
///
/// The default value is [`Uncertain`].
#[derive(Clone, Copy, Debug, Default, Eq, Ord, PartialEq, PartialOrd)]
pub enum Ternary {
    False,
    #[default]
    Uncertain,
    True,
}

impl Ternary {
    /// Returns `true` if `self` is [`False`].
    pub fn certainly_false(self) -> bool {
        self == False
    }

    /// Returns `true` if `self` is [`True`].
    pub fn certainly_true(self) -> bool {
        self == True
    }

    /// Returns `true` if `self` is either [`False`] or [`Uncertain`].
    pub fn possibly_false(self) -> bool {
        !self.certainly_true()
    }

    /// Returns `true` if `self` is either [`True`] or [`Uncertain`].
    pub fn possibly_true(self) -> bool {
        !self.certainly_false()
    }
}

impl BitAnd for Ternary {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        self.min(rhs)
    }
}

impl BitOr for Ternary {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        self.max(rhs)
    }
}

impl From<bool> for Ternary {
    fn from(x: bool) -> Self {
        if x {
            True
        } else {
            False
        }
    }
}

impl From<(bool, bool)> for Ternary {
    fn from(x: (bool, bool)) -> Self {
        match x {
            (true, true) => True,
            (false, true) => Uncertain,
            (false, false) => False,
            _ => panic!(),
        }
    }
}

impl Not for Ternary {
    type Output = Ternary;

    fn not(self) -> Self::Output {
        match self {
            True => False,
            False => True,
            _ => Uncertain,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ternary() {
        assert_eq!(Ternary::default(), Uncertain);
        assert_eq!(Ternary::from(false), False);
        assert_eq!(Ternary::from(true), True);
        assert_eq!(Ternary::from((false, false)), False);
        assert_eq!(Ternary::from((false, true)), Uncertain);
        assert_eq!(Ternary::from((true, true)), True);

        assert!(False.certainly_false());
        assert!(!False.certainly_true());
        assert!(False.possibly_false());
        assert!(!False.possibly_true());

        assert!(!Uncertain.certainly_false());
        assert!(!Uncertain.certainly_true());
        assert!(Uncertain.possibly_false());
        assert!(Uncertain.possibly_true());

        assert!(!True.certainly_false());
        assert!(True.certainly_true());
        assert!(!True.possibly_false());
        assert!(True.possibly_true());

        assert!(False < Uncertain);
        assert!(Uncertain < True);

        assert_eq!(False & False, False);
        assert_eq!(False & Uncertain, False);
        assert_eq!(False & True, False);
        assert_eq!(Uncertain & Uncertain, Uncertain);
        assert_eq!(Uncertain & True, Uncertain);
        assert_eq!(True & True, True);

        assert_eq!(False | False, False);
        assert_eq!(False | Uncertain, Uncertain);
        assert_eq!(False | True, True);
        assert_eq!(Uncertain | Uncertain, Uncertain);
        assert_eq!(Uncertain | True, True);
        assert_eq!(True | True, True);

        assert_eq!(!False, True);
        assert_eq!(!Uncertain, Uncertain);
        assert_eq!(!True, False);
    }
}
