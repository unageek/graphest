use bitflags::*;

bitflags! {
    /// A set of free variables.
    #[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
    pub struct VarSet: u8 {
        const EMPTY = 0;
        const M = 1;
        const N = 2;
        const N_THETA = 4;
        const T = 8;
        const X = 16;
        const Y = 32;
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VarType {
    Integer,
    Real,
}

impl VarSet {
    pub fn len(&self) -> usize {
        self.bits().count_ones() as usize
    }

    pub fn var_type(&self) -> VarType {
        match *self {
            VarSet::M | VarSet::N | VarSet::N_THETA => VarType::Integer,
            VarSet::T | VarSet::X | VarSet::Y => VarType::Real,
            _ => panic!(),
        }
    }
}

/// Creates a set of variables in a const context.
///
/// See <https://github.com/bitflags/bitflags/issues/180>
#[macro_export]
macro_rules! vars {
    ($($var:path)|*) => {
        $crate::vars::VarSet::from_bits_truncate($($var.bits())|*)
    };
}

pub type VarIndex = u8;
