use crate::arb_sys::*;
use inari::{interval, Interval};
use std::{mem::MaybeUninit, ops::Drop};

const MAG_BITS: u32 = 30;

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
            // Construct an `Arb` interval faster and more precisely than
            // using `arb_set_interval_arf`.

            let mid = x.mid();
            unsafe {
                arf_set_d(&mut y.0.mid as arf_ptr, mid);
            }

            let rad = x.rad();
            if rad != 0.0 {
                let (man, mut exp) = frexp(rad);
                let mut man = (man * (1 << MAG_BITS) as f64).ceil() as u32;
                if man == 1 << MAG_BITS {
                    // Restrict the mantissa within 30 bits:
                    //   100...000 ≤ `man` ≤ 111...111 (30 1's).
                    man = 1 << (MAG_BITS - 1);
                    exp += 1;
                }
                // For safer construction, see `mag_set_ui_2exp_si`.
                // https://github.com/fredrik-johansson/arb/blob/master/mag/set_ui_2exp_si.c
                y.0.rad.exp = exp.into();
                y.0.rad.man = man.into();
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

// A copy-paste of https://github.com/rust-lang/libm/blob/master/src/math/frexp.rs
fn frexp(x: f64) -> (f64, i32) {
    let mut y = x.to_bits();
    let ee = ((y >> 52) & 0x7ff) as i32;

    if ee == 0 {
        if x != 0.0 {
            let x1p64 = f64::from_bits(0x43f0000000000000);
            let (x, e) = frexp(x * x1p64);
            return (x, e - 64);
        }
        return (x, 0);
    } else if ee == 0x7ff {
        return (x, 0);
    }

    let e = ee - 0x3fe;
    y &= 0x800fffffffffffff;
    y |= 0x3fe0000000000000;
    (f64::from_bits(y), e)
}

#[cfg(test)]
mod tests {
    use super::*;
    use inari::{const_interval, Interval};

    #[test]
    fn inclusion_property() {
        let xs = [
            Interval::EMPTY,
            const_interval!(0.0, 0.0),
            const_interval!(1.0, 1.0),
            Interval::PI,
            const_interval!(0.0, f64::INFINITY),
            const_interval!(f64::NEG_INFINITY, 0.0),
            Interval::ENTIRE,
            // The case where rounding up the interval radius (`mag_t`) produces a carry:
            // As opposed to `f64`, the hidden bit is not used in the mantissa of a `mag_t`.
            //         a =  0.0₂
            //         b =  0.111...111 1₂ × 2^1
            //                ^^^^^^^^^^^ 31 1's (1-bit larger than what can fit in the mantissa)
            // (b - a)/2 =  0.111...111 1₂ × 2^0
            //       rad =  1.000...00₂    × 2^0  <- Round the mantissa of (b - a)/2 up to
            //           =  0.100...000₂   × 2^1     the nearest 30-bit number. (produces a carry)
            //                ^^^^^^^^^ the mantissa of a `mag_t` (30-bit)
            const_interval!(0.0, 1.9999999990686774),
        ];
        for x in xs.iter().copied() {
            let y = Arb::from_interval(x).to_interval();
            assert!(x.subset(y));
        }
    }
}
