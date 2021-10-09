use gmp_mpfr_sys::{mpfr, mpfr::rnd_t};
use inari::{interval, Interval};
use rug::{Float, Rational};

pub fn div(x: Rational, y: Rational) -> Option<Rational> {
    if y == 0 {
        None
    } else {
        Some(x / y)
    }
}

pub fn gcd(mut x: Rational, mut y: Rational) -> Option<Rational> {
    while y != 0 {
        let rem = rem_euclid(x, y.clone())?;
        x = y;
        y = rem;
    }
    Some(x.abs())
}

pub fn lcm(x: Rational, y: Rational) -> Option<Rational> {
    if x == 0 && y == 0 {
        Some(Rational::new())
    } else {
        let xy = Rational::from(&x * &y);
        div(xy.abs(), gcd(x, y)?)
    }
}

pub fn max(x: Rational, y: Rational) -> Option<Rational> {
    if x < y {
        Some(y)
    } else {
        Some(x)
    }
}

pub fn min(x: Rational, y: Rational) -> Option<Rational> {
    if x < y {
        Some(x)
    } else {
        Some(y)
    }
}

pub fn pow(x: Rational, y: Rational) -> Option<Rational> {
    let xn = x.numer().to_i32()?;
    let xd = x.denom().to_u32()?;
    let yn = y.numer().to_i32()?;
    let yd = y.denom().to_u32()?;
    if yd == 1 {
        // y ∈ ℤ.
        if yn >= 0 {
            let n = yn as u32;
            let zn = xn.checked_pow(n)?;
            let zd = xd.checked_pow(n)?;
            Some((zn, zd).into())
        } else if xn != 0 {
            let n = -yn as u32;
            let zn = xd.checked_pow(n)?;
            let zd = xn.checked_pow(n)?;
            Some((zn, zd).into())
        } else {
            // y < 0 ∧ x = 0.
            None
        }
    } else {
        // y ∉ ℤ.
        if xn == 0 && yn > 0 {
            Some(0.into())
        } else {
            None
        }
    }
}

pub fn rem_euclid(x: Rational, y: Rational) -> Option<Rational> {
    if y == 0 {
        None
    } else {
        let y = y.abs();
        Some(&x - &y * Rational::from(&x / &y).floor())
    }
}

// Based on `inari::parse::rational_to_f64`.
#[allow(clippy::many_single_char_names)]
pub fn to_interval(r: &Rational) -> Interval {
    let mut f = Float::new(f64::MANTISSA_DIGITS);

    unsafe {
        let orig_emin = mpfr::get_emin();
        let orig_emax = mpfr::get_emax();
        mpfr::set_emin((f64::MIN_EXP - (f64::MANTISSA_DIGITS as i32) + 1).into());
        mpfr::set_emax(f64::MAX_EXP.into());
        let rnd = rnd_t::RNDD;
        let t = mpfr::set_q(f.as_raw_mut(), r.as_raw(), rnd);
        mpfr::subnormalize(f.as_raw_mut(), t, rnd);
        let a = mpfr::get_d(f.as_raw(), rnd);
        let rnd = rnd_t::RNDU;
        let t = mpfr::set_q(f.as_raw_mut(), r.as_raw(), rnd);
        mpfr::subnormalize(f.as_raw_mut(), t, rnd);
        let b = mpfr::get_d(f.as_raw(), rnd);
        mpfr::set_emin(orig_emin);
        mpfr::set_emax(orig_emax);
        interval!(a, b).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! r {
        ($i:literal) => {
            Rational::from($i)
        };

        ($p:literal / $q:literal) => {
            Rational::from(($p, $q))
        };
    }

    macro_rules! test {
        ($f:path, $x:expr, $y:expr, $z:expr) => {
            assert_eq!($f($x, $y), $z);
        };

        (@commut $(@$af:ident)* $f:path, $(@$ax:ident)* $x:expr, $(@$ay:ident)* $y:expr, $z:expr) => {
            test!($(@$af)* $f, $(@$ax)* $x, $(@$ay)* $y, $z);
            test!($(@$af)* $f, $(@$ax)* $y, $(@$ay)* $x, $z);
        };

        ($(@$af:ident)* $f:path, @even $(@$ax:ident)* $x:expr, $(@$ay:ident)* $y:expr, $z:expr) => {
            test!($(@$af)* $f, $(@$ax)* $x, $(@$ay)* $y, $z);
            test!($(@$af)* $f, $(@$ax)* -$x, $(@$ay)* $y, $z);
        };

        ($(@$af:ident)* $f:path, @odd $(@$ax:ident)* $x:expr, $(@$ay:ident)* $y:expr, $z:expr) => {
            test!($(@$af)* $f, $(@$ax)* $x, $(@$ay)* $y, $z);
            test!($(@$af)* $f, $(@$ax)* -$x, $(@$ay)* $y, $z.map(|z: Rational| -z));
        };

        ($(@$af:ident)* $f:path, $(@$ax:ident)* $x:expr, @even $(@$ay:ident)* $y:expr, $z:expr) => {
            test!($(@$af)* $f, $(@$ax)* $x, $(@$ay)* $y, $z);
            test!($(@$af)* $f, $(@$ax)* $x, $(@$ay)* -$y, $z);
        };

        ($(@$af:ident)* $f:path, $(@$ax:ident)* $x:expr, @odd $(@$ay:ident)* $y:expr, $z:expr) => {
            test!($(@$af)* $f, $(@$ax)* $x, $(@$ay)* $y, $z);
            test!($(@$af)* $f, $(@$ax)* $x, $(@$ay)* -$y, $z.map(|z: Rational| -z));
        };
    }

    #[test]
    fn div() {
        use super::div;
        test!(div, r!(0), r!(0), None);
        test!(div, @odd r!(2 / 3), r!(0), None);
        test!(div, @odd r!(2 / 3), @odd r!(4 / 5), Some(r!(5 / 6)));
    }

    #[test]
    fn gcd() {
        use super::gcd;
        test!(gcd, r!(0), r!(0), Some(r!(0)));
        test!(@commut gcd, @even r!(2 / 3), r!(0), Some(r!(2 / 3)));
        test!(@commut gcd, @even r!(2 / 3), @even r!(4 / 5), Some(r!(2 / 15)));
    }

    #[test]
    fn lcm() {
        use super::lcm;
        test!(lcm, r!(0), r!(0), Some(r!(0)));
        test!(@commut lcm, @even r!(2 / 3), r!(0), Some(r!(0)));
        test!(@commut lcm, @even r!(2 / 3), @even r!(4 / 5), Some(r!(4)));
    }

    #[test]
    fn max() {
        use super::max;
        test!(@commut max, r!(2 / 3), r!(4 / 5), Some(r!(4 / 5)));
    }

    #[test]
    fn min() {
        use super::min;
        test!(@commut min, r!(2 / 3), r!(4 / 5), Some(r!(2 / 3)));
    }

    #[test]
    fn pow() {
        use super::pow;
        test!(pow, r!(0), r!(-4), None);
        test!(pow, r!(0), r!(-3), None);
        test!(pow, r!(0), r!(-4 / 5), None);
        test!(pow, r!(0), r!(-3 / 5), None);
        test!(pow, r!(0), r!(0), Some(r!(1)));
        test!(pow, r!(0), r!(3 / 5), Some(r!(0)));
        test!(pow, r!(0), r!(4 / 5), Some(r!(0)));
        test!(pow, r!(0), r!(3), Some(r!(0)));
        test!(pow, r!(0), r!(4), Some(r!(0)));
        test!(pow, @even r!(2 / 3), r!(-4), Some(r!(81 / 16)));
        test!(pow, @odd r!(2 / 3), r!(-3), Some(r!(27 / 8)));
        test!(pow, @even r!(2 / 3), r!(-4 / 5), None);
        test!(pow, @odd r!(2 / 3), r!(-3 / 5), None);
        test!(pow, @even r!(2 / 3), r!(0), Some(r!(1)));
        test!(pow, @odd r!(2 / 3), r!(3 / 5), None);
        test!(pow, @even r!(2 / 3), r!(4 / 5), None);
        test!(pow, @odd r!(2 / 3), r!(3), Some(r!(8 / 27)));
        test!(pow, @even r!(2 / 3), r!(4), Some(r!(16 / 81)));

        // The result is rational, but not computed.
        test!(pow, r!(1), r!(1 / 2), None);
    }

    #[test]
    fn rem_euclid() {
        use super::rem_euclid;
        test!(rem_euclid, r!(0), r!(0), None);
        test!(rem_euclid, r!(2 / 3), r!(0), None);
        test!(rem_euclid, r!(2 / 3), @even r!(4 / 5), Some(r!(2 / 3)));
        test!(rem_euclid, r!(4 / 5), @even r!(2 / 3), Some(r!(2 / 15)));
        test!(rem_euclid, r!(-2 / 3), @even r!(4 / 5), Some(r!(2 / 15)));
        test!(rem_euclid, r!(-4 / 5), @even r!(2 / 3), Some(r!(8 / 15)));
    }
}
