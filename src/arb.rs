use crate::arb_sys::*;
use inari::{interval, Interval};
use std::{mem::MaybeUninit, ops::Drop};

pub enum Round {
    // Down = 0,
    // Up = 1,
    Floor = 2,
    Ceil = 3,
    // Near = 4,
}

pub struct Arf(arf_struct);

impl Arf {
    pub fn new() -> Self {
        unsafe {
            let mut x = MaybeUninit::uninit();
            arf_init(x.as_mut_ptr());
            Self(x.assume_init())
        }
    }

    pub fn with_val(val: f64) -> Self {
        let mut x = Self::new();
        unsafe {
            arf_set_d(x.as_raw_mut(), val);
        }
        x
    }

    pub fn as_raw_mut(&mut self) -> arf_ptr {
        &mut self.0 as arf_ptr
    }

    #[allow(clippy::wrong_self_convention)]
    pub fn to_f64_round(&mut self, round: Round) -> f64 {
        unsafe { arf_get_d(self.as_raw_mut(), round as i32) }
    }
}

impl Drop for Arf {
    fn drop(&mut self) {
        unsafe {
            arf_clear(self.as_raw_mut());
        }
    }
}

pub struct Arb(arb_struct);

impl Arb {
    pub fn new() -> Self {
        unsafe {
            let mut x = MaybeUninit::uninit();
            arb_init(x.as_mut_ptr());
            Self(x.assume_init())
        }
    }

    pub fn as_raw_mut(&mut self) -> arb_ptr {
        &mut self.0 as arb_ptr
    }

    pub fn from_interval(x: Interval) -> Self {
        let mut y = Self::new();
        if !x.is_common_interval() {
            unsafe {
                arb_zero_pm_inf(y.as_raw_mut());
            }
        } else {
            let mut a = Arf::with_val(x.inf());
            let mut b = Arf::with_val(x.sup());
            unsafe {
                arb_set_interval_arf(
                    y.as_raw_mut(),
                    a.as_raw_mut(),
                    b.as_raw_mut(),
                    f64::MANTISSA_DIGITS.into(),
                );
            }
        }
        y
    }

    #[allow(clippy::wrong_self_convention)]
    pub fn to_interval(&mut self) -> Interval {
        let mut a = Arf::new();
        let mut b = Arf::new();
        unsafe {
            arb_get_interval_arf(
                a.as_raw_mut(),
                b.as_raw_mut(),
                self.as_raw_mut(),
                f64::MANTISSA_DIGITS.into(),
            );
        }
        interval!(a.to_f64_round(Round::Floor), b.to_f64_round(Round::Ceil))
            .unwrap_or(Interval::ENTIRE) // [+∞ ± c], [-∞ ± c] or [NaN ± c]
    }
}

impl Drop for Arb {
    fn drop(&mut self) {
        unsafe {
            arb_clear(self.as_raw_mut());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use inari::{const_interval, Interval};

    #[test]
    fn inclusion_property() {
        let xs = [
            Interval::EMPTY,
            const_interval!(1.0, 1.0),
            Interval::PI,
            const_interval!(0.0, 1.0),
            const_interval!(0.0, f64::INFINITY),
            const_interval!(f64::NEG_INFINITY, 0.0),
            Interval::ENTIRE,
        ];
        for x in xs.iter().copied() {
            assert!(x.subset(Arb::from_interval(x).to_interval()));
        }
    }
}
