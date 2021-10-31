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

    #[test]
    fn ternary() {
        assert_eq!(Ternary::default(), Ternary::Uncertain);

        assert!(Ternary::False < Ternary::Uncertain);
        assert!(Ternary::Uncertain < Ternary::True);
    }
}
