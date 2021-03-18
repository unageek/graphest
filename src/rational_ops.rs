use gmp_mpfr_sys::{mpfr, mpfr::rnd_t};
use inari::{interval, Interval};
use rug::{ops::Pow, Float, Rational};

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

pub fn pown(x: Rational, n: i32) -> Option<Rational> {
    if n < 0 && x == 0 {
        None
    } else {
        Some(x.pow(n))
    }
}

pub fn recip(x: Rational) -> Option<Rational> {
    if x == 0 {
        None
    } else {
        Some(x.recip())
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

#[allow(clippy::many_single_char_names)]
pub fn to_interval(r: &Rational) -> Interval {
    let mut f = Float::new(f64::MANTISSA_DIGITS);

    unsafe {
        let orig_emin = mpfr::get_emin();
        let orig_emax = mpfr::get_emax();
        mpfr::set_emin((f64::MIN_EXP - (f64::MANTISSA_DIGITS as i32) + 1) as i64);
        mpfr::set_emax(f64::MAX_EXP as i64);
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
