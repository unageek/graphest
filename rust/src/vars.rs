use bitflags::*;

bitflags! {
    /// A set of free variables.
    #[derive(Default)]
    pub struct VarSet: u8 {
        const EMPTY = 0;
        const X = 1;
        const Y = 2;
        const N_THETA = 4;
        const N = 8;
        const T = 16;
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
