use std::ops::{BitAnd, BitOr, Not};

/// A ternary value which could be either [`False`], [`Uncertain`], or [`True`].
///
/// The values are ordered as: [`False`] < [`Uncertain`] < [`True`].
///
/// The default value is [`Uncertain`].
///
/// [`False`]: Ternary::False
/// [`True`]: Ternary::True
/// [`Uncertain`]: Ternary::Uncertain
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum Ternary {
    False,
    Uncertain,
    True,
}

impl Ternary {
    pub fn certainly_false(self) -> bool {
        self == Self::False
    }

    pub fn certainly_true(self) -> bool {
        self == Self::True
    }

    pub fn possibly_false(self) -> bool {
        !self.certainly_true()
    }

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

impl Default for Ternary {
    fn default() -> Self {
        Ternary::Uncertain
    }
}

impl From<bool> for Ternary {
    fn from(x: bool) -> Self {
        if x {
            Self::True
        } else {
            Self::False
        }
    }
}

impl From<(bool, bool)> for Ternary {
    fn from(x: (bool, bool)) -> Self {
        match x {
            (true, true) => Ternary::True,
            (false, true) => Ternary::Uncertain,
            (false, false) => Ternary::False,
            _ => panic!(),
        }
    }
}

impl Not for Ternary {
    type Output = Ternary;

    fn not(self) -> Self::Output {
        use Ternary::*;
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
    use Ternary::*;

    #[test]
    fn ternary() {
        assert_eq!(Ternary::default(), Uncertain);
        assert_eq!(Ternary::from(false), False);
        assert_eq!(Ternary::from(true), True);
        assert_eq!(Ternary::from((false, false)), False);
        assert_eq!(Ternary::from((false, true)), Uncertain);
        assert_eq!(Ternary::from((true, true)), True);

        assert!(Ternary::False.certainly_false());
        assert!(!Ternary::False.certainly_true());
        assert!(Ternary::False.possibly_false());
        assert!(!Ternary::False.possibly_true());

        assert!(!Ternary::Uncertain.certainly_false());
        assert!(!Ternary::Uncertain.certainly_true());
        assert!(Ternary::Uncertain.possibly_false());
        assert!(Ternary::Uncertain.possibly_true());

        assert!(!Ternary::True.certainly_false());
        assert!(Ternary::True.certainly_true());
        assert!(!Ternary::True.possibly_false());
        assert!(Ternary::True.possibly_true());

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
