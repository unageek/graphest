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
