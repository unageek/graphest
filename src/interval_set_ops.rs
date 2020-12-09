use crate::interval_set::{Branch, DecSignSet, SignSet, Site, TupperInterval, TupperIntervalSet};
use gmp_mpfr_sys::mpfr;
use inari::{const_dec_interval, const_interval, interval, DecInterval, Decoration, Interval};
use rug::Float;
use std::{
    convert::From,
    ops::{Add, Mul, Neg, Sub},
};

impl Neg for &TupperIntervalSet {
    type Output = TupperIntervalSet;

    fn neg(self) -> Self::Output {
        let mut rs = Self::Output::empty();
        for x in self {
            rs.insert(TupperInterval::new(-x.to_dec_interval(), x.g));
        }
        rs.normalize()
    }
}

macro_rules! impl_arith_op {
    ($Op:ident, $op:ident) => {
        impl<'a, 'b> $Op<&'b TupperIntervalSet> for &'a TupperIntervalSet {
            type Output = TupperIntervalSet;

            fn $op(self, rhs: &'b TupperIntervalSet) -> Self::Output {
                let mut rs = Self::Output::empty();
                for x in self {
                    for y in rhs {
                        if let Some(g) = x.g.union(y.g) {
                            rs.insert(TupperInterval::new(
                                x.to_dec_interval().$op(y.to_dec_interval()),
                                g,
                            ));
                        }
                    }
                }
                rs.normalize()
            }
        }
    };
}

impl_arith_op!(Add, add);
impl_arith_op!(Sub, sub);
impl_arith_op!(Mul, mul);

/// The parity of a function.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Parity {
    None,
    Even,
    Odd,
}

impl TupperIntervalSet {
    pub fn atan2(&self, rhs: &Self, site: Option<Site>) -> Self {
        let mut rs = Self::empty();
        for y in self {
            for x in rhs {
                if let Some(g) = x.g.union(y.g) {
                    let a = x.x.inf();
                    let b = x.x.sup();
                    let c = y.x.inf();
                    let d = y.x.sup();

                    if a == 0.0 && b == 0.0 && c < 0.0 && d > 0.0 {
                        let dec = Decoration::Trv;

                        rs.insert(TupperInterval::new(
                            DecInterval::set_dec(-Interval::FRAC_PI_2, dec),
                            match site {
                                Some(site) => g.inserted(site, Branch::new(0)),
                                _ => g,
                            },
                        ));
                        rs.insert(TupperInterval::new(
                            DecInterval::set_dec(Interval::FRAC_PI_2, dec),
                            match site {
                                Some(site) => g.inserted(site, Branch::new(1)),
                                _ => g,
                            },
                        ));
                    } else if a < 0.0 && b <= 0.0 && c < 0.0 && d >= 0.0 {
                        let dec = if b == 0.0 {
                            Decoration::Trv
                        } else {
                            Decoration::Def.min(x.d).min(y.d)
                        };

                        // y < 0 (thus z < 0) part.
                        let x0 = interval!(b, b).unwrap();
                        let y0 = interval!(c, c).unwrap();
                        rs.insert(TupperInterval::new(
                            DecInterval::set_dec(
                                interval!(-Interval::PI.sup(), y0.atan2(x0).sup()).unwrap(),
                                dec,
                            ),
                            match site {
                                Some(site) => g.inserted(site, Branch::new(0)),
                                _ => g,
                            },
                        ));

                        // y ≥ 0 (thus z > 0) part.
                        let x1 = interval!(a, b).unwrap();
                        let y1 = interval!(0.0, d).unwrap();
                        rs.insert(TupperInterval::new(
                            DecInterval::set_dec(y1.atan2(x1), dec),
                            match site {
                                Some(site) => g.inserted(site, Branch::new(1)),
                                _ => g,
                            },
                        ));
                    } else {
                        // a = b = c = d = 0 goes here.
                        rs.insert(TupperInterval::new(
                            y.to_dec_interval().atan2(x.to_dec_interval()),
                            g,
                        ));
                    }
                }
            }
        }
        rs.normalize()
    }

    pub fn div(&self, rhs: &Self, site: Option<Site>) -> Self {
        let mut rs = Self::empty();
        for x in self {
            for y in rhs {
                if let Some(g) = x.g.union(y.g) {
                    let c = y.x.inf();
                    let d = y.x.sup();
                    if c < 0.0 && d > 0.0 {
                        let y0 = DecInterval::set_dec(interval!(c, 0.0).unwrap(), y.d);
                        rs.insert(TupperInterval::new(
                            x.to_dec_interval() / y0,
                            match site {
                                Some(site) => g.inserted(site, Branch::new(0)),
                                _ => g,
                            },
                        ));
                        let y1 = DecInterval::set_dec(interval!(0.0, d).unwrap(), y.d);
                        rs.insert(TupperInterval::new(
                            x.to_dec_interval() / y1,
                            match site {
                                Some(site) => g.inserted(site, Branch::new(1)),
                                _ => g,
                            },
                        ));
                    } else {
                        rs.insert(TupperInterval::new(
                            x.to_dec_interval() / y.to_dec_interval(),
                            g,
                        ));
                    }
                }
            }
        }
        rs.normalize()
    }

    #[cfg(not(feature = "arb"))]
    pub fn erf(&self) -> Self {
        let mut rs = Self::empty();
        for x in self {
            rs.insert(TupperInterval::new(
                DecInterval::set_dec(erf(x.x), x.d),
                x.g,
            ));
        }
        rs
    }

    #[cfg(not(feature = "arb"))]
    pub fn erfc(&self) -> Self {
        let mut rs = Self::empty();
        for x in self {
            rs.insert(TupperInterval::new(
                DecInterval::set_dec(erfc(x.x), x.d),
                x.g,
            ));
        }
        rs
    }

    pub fn gamma(&self, site: Option<Site>) -> Self {
        // argmin_{x > 0} Γ(x), rounded down/up.
        const ARGMIN_RD: f64 = 1.4616321449683622;
        const ARGMIN_RU: f64 = 1.4616321449683625;
        // min_{x > 0} Γ(x), rounded down.
        const MIN_RD: f64 = 0.8856031944108886;
        let mut rs = Self::empty();
        for x in self {
            let mut a = x.x.inf();
            let b = x.x.sup();
            if a == 0.0 && b == 0.0 {
                // empty.
            } else if a >= 0.0 {
                let dec = if a == 0.0 { Decoration::Trv } else { x.d };

                if a == 0.0 {
                    // gamma_rd/ru(±0.0) returns ±∞.
                    a = 0.0;
                }

                let y = if b <= ARGMIN_RD {
                    // b < x0, where x0 = argmin_{x > 0} Γ(x).
                    interval!(gamma_rd(b), gamma_ru(a)).unwrap()
                } else if a >= ARGMIN_RU {
                    // x0 < a.
                    interval!(gamma_rd(a), gamma_ru(b)).unwrap()
                } else {
                    // a < x0 < b.
                    interval!(MIN_RD, gamma_ru(a).max(gamma_ru(b))).unwrap()
                };
                rs.insert(TupperInterval::new(DecInterval::set_dec(y, dec), x.g));
            } else if b <= 0.0 {
                // Γ(x) = π / (sin(π x) Γ(1 - x)).
                let one = TupperIntervalSet::from(const_dec_interval!(1.0, 1.0));
                let pi = TupperIntervalSet::from(DecInterval::PI);
                let mut xs = Self::empty();
                xs.insert(*x);
                rs = pi.div(&(&(&pi * &xs).sin() * &(&one - &xs).gamma(None)), site);
            } else {
                // a < 0 < b.
                let dec = Decoration::Trv;

                let mut xs = Self::empty();
                xs.insert(TupperInterval::new(
                    DecInterval::set_dec(interval!(a, 0.0).unwrap(), dec),
                    match site {
                        Some(site) => x.g.inserted(site, Branch::new(0)),
                        _ => x.g,
                    },
                ));
                xs.insert(TupperInterval::new(
                    DecInterval::set_dec(interval!(0.0, b).unwrap(), dec),
                    match site {
                        Some(site) => x.g.inserted(site, Branch::new(1)),
                        _ => x.g,
                    },
                ));
                rs = xs.gamma(None);
            }
        }
        rs.normalize()
    }

    pub fn log(&self, rhs: &Self, site: Option<Site>) -> Self {
        self.log2().div(&rhs.log2(), site)
    }

    pub fn mul_add(&self, rhs: &Self, addend: &Self) -> Self {
        let mut rs = Self::empty();
        for x in self {
            for y in rhs {
                if let Some(g) = x.g.union(y.g) {
                    for z in addend {
                        if let Some(g) = g.union(z.g) {
                            rs.insert(TupperInterval::new(
                                x.to_dec_interval()
                                    .mul_add(y.to_dec_interval(), z.to_dec_interval()),
                                g,
                            ));
                        }
                    }
                }
            }
        }
        rs.normalize()
    }

    // f(x) = 1.
    pub fn one(&self) -> Self {
        let mut rs = Self::empty();
        for x in self {
            rs.insert(TupperInterval::new(
                DecInterval::set_dec(const_interval!(1.0, 1.0), x.d),
                x.g,
            ));
        }
        rs.normalize()
    }

    /// Returns the parity of the function f(x) = x^y.
    ///
    /// Precondition: y is neither ±∞ nor NaN.
    fn exponentiation_parity(y: f64) -> Parity {
        if y == y.trunc() {
            // y ∈ ℤ.
            if y == 2.0 * (y / 2.0).trunc() {
                // y ∈ 2ℤ.
                Parity::Even
            } else {
                // y ∈ 2ℤ + 1.
                Parity::Odd
            }
        } else {
            // y is a rational number of the form odd / even.
            Parity::None
        }
    }

    // - For any integer m and positive odd integer n, we define
    //     x^(m/n) = surd(x, n)^m,
    //   where surd(x, n) is the real-valued nth root of x.
    //   Therefore, for x < 0,
    //           | (-x)^y     if y = (even)/(odd)
    //           |            (x^y is an even function of x),
    //     x^y = | -(-x)^y    if y = (odd)/(odd)
    //           |            (x^y is an odd function of x),
    //           | undefined  otherwise (y = (odd)/(even) or irrational).
    // - We define 0^0 = 1.
    // The original `Interval::pow` is not defined for x < 0 nor x = y = 0.
    #[allow(clippy::many_single_char_names)]
    pub fn pow(&self, rhs: &Self, site: Option<Site>) -> Self {
        let mut rs = Self::empty();
        for x in self {
            for y in rhs {
                if let Some(g) = x.g.union(y.g) {
                    let a = x.x.inf();
                    let b = x.x.sup();
                    let c = y.x.inf();
                    let d = y.x.sup();

                    // | {1}  if (0, 0) ∈ x × y,
                    // | ∅    otherwise.
                    let one_or_empty = if a <= 0.0 && b >= 0.0 && c <= 0.0 && d >= 0.0 {
                        const_interval!(1.0, 1.0)
                    } else {
                        Interval::EMPTY
                    };

                    if c == d {
                        // y is a singleton.
                        match Self::exponentiation_parity(c) {
                            Parity::None => {
                                rs.insert(TupperInterval::new(
                                    x.to_dec_interval().pow(y.to_dec_interval()),
                                    g,
                                ));
                            }
                            Parity::Even => {
                                let dec = if a <= 0.0 && b >= 0.0 && c < 0.0 {
                                    // Undefined for x = 0.
                                    Decoration::Trv
                                } else {
                                    // The restriction is continuous, thus `Dac`.
                                    // It can be `Com`, but that does not matter for graphing.
                                    Decoration::Dac.min(x.d).min(y.d)
                                };

                                let x = x.x.abs();
                                let z = x.pow(y.x);
                                rs.insert(TupperInterval::new(
                                    DecInterval::set_dec(z.convex_hull(one_or_empty), dec),
                                    g,
                                ));
                            }
                            Parity::Odd => {
                                let dec = if a <= 0.0 && b >= 0.0 && c < 0.0 {
                                    Decoration::Trv
                                } else {
                                    Decoration::Dac.min(x.d).min(y.d)
                                };

                                // xn or xp can be empty.
                                let xn = x.x.intersection(const_interval!(f64::NEG_INFINITY, 0.0));
                                let zn = -(-xn).pow(y.x);
                                let xp = x.x.intersection(const_interval!(0.0, f64::INFINITY));
                                let zp = xp.pow(y.x);
                                if c < 0.0 {
                                    rs.insert(TupperInterval::new(
                                        DecInterval::set_dec(zn, dec),
                                        match site {
                                            Some(site) => g.inserted(site, Branch::new(0)),
                                            _ => g,
                                        },
                                    ));
                                    rs.insert(TupperInterval::new(
                                        DecInterval::set_dec(zp, dec),
                                        match site {
                                            Some(site) => g.inserted(site, Branch::new(1)),
                                            _ => g,
                                        },
                                    ));
                                } else {
                                    let z = zn.convex_hull(zp);
                                    rs.insert(TupperInterval::new(DecInterval::set_dec(z, dec), g));
                                }
                            }
                        }
                    } else if a < 0.0 {
                        // a < 0.
                        let dec = Decoration::Trv;

                        // x^y ≥ 0 part from
                        //   x ≥ 0 (incl. 0^0), and
                        //   x < 0 for those ys where f(x) = x^y is an even function.
                        let x0 = x.x.abs();
                        let z = x0.pow(y.x);
                        rs.insert(TupperInterval::new(
                            DecInterval::set_dec(z.convex_hull(one_or_empty), dec),
                            match site {
                                Some(site) => g.inserted(site, Branch::new(0)),
                                _ => g,
                            },
                        ));

                        // x^y < 0 part from
                        //   x < 0 for those ys where f(x) = x^y is an odd function.
                        let x1 = x.x.min(const_interval!(0.0, 0.0));
                        let z = -(-x1).pow(y.x);
                        rs.insert(TupperInterval::new(
                            DecInterval::set_dec(z, dec),
                            match site {
                                Some(site) => g.inserted(site, Branch::new(1)),
                                _ => g,
                            },
                        ));
                    } else {
                        // a ≥ 0.

                        // If a = b = 0 ∧ c ≤ 0 ≤ d, we need to add {1} to z manually.
                        // In that case, the decoration of z is already `Trv`.
                        let z = x.to_dec_interval().pow(y.to_dec_interval());
                        rs.insert(TupperInterval::new(
                            DecInterval::set_dec(
                                z.interval().unwrap().convex_hull(one_or_empty),
                                z.decoration(),
                            ),
                            g,
                        ));
                    }
                }
            }
        }
        rs.normalize()
    }

    pub fn pown(&self, rhs: i32, site: Option<Site>) -> Self {
        let mut rs = Self::empty();
        for x in self {
            let a = x.x.inf();
            let b = x.x.sup();
            if rhs < 0 && rhs % 2 == 1 && a < 0.0 && b > 0.0 {
                let x0 = DecInterval::set_dec(interval!(a, 0.0).unwrap(), x.d);
                rs.insert(TupperInterval::new(
                    x0.pown(rhs),
                    match site {
                        Some(site) => x.g.inserted(site, Branch::new(0)),
                        _ => x.g,
                    },
                ));
                let x1 = DecInterval::set_dec(interval!(0.0, b).unwrap(), x.d);
                rs.insert(TupperInterval::new(
                    x1.pown(rhs),
                    match site {
                        Some(site) => x.g.inserted(site, Branch::new(1)),
                        _ => x.g,
                    },
                ));
            } else {
                rs.insert(TupperInterval::new(x.to_dec_interval().pown(rhs), x.g));
            }
        }
        rs.normalize()
    }

    pub fn recip(&self, site: Option<Site>) -> Self {
        let mut rs = Self::empty();
        for x in self {
            let a = x.x.inf();
            let b = x.x.sup();
            if a < 0.0 && b > 0.0 {
                let x0 = DecInterval::set_dec(interval!(a, 0.0).unwrap(), x.d);
                rs.insert(TupperInterval::new(
                    x0.recip(),
                    match site {
                        Some(site) => x.g.inserted(site, Branch::new(0)),
                        _ => x.g,
                    },
                ));
                let x1 = DecInterval::set_dec(interval!(0.0, b).unwrap(), x.d);
                rs.insert(TupperInterval::new(
                    x1.recip(),
                    match site {
                        Some(site) => x.g.inserted(site, Branch::new(1)),
                        _ => x.g,
                    },
                ));
            } else {
                rs.insert(TupperInterval::new(x.to_dec_interval().recip(), x.g));
            }
        }
        rs.normalize()
    }

    pub fn rem_euclid(&self, rhs: &Self, site: Option<Site>) -> Self {
        let zero = TupperIntervalSet::from(const_dec_interval!(0.0, 0.0));
        let y = rhs.abs();
        (self - &(&y * &self.div(&y, None).floor(site)))
            .max(&zero)
            .min(&y)
    }

    // f(x) = | sin(x)/x  if x ≠ 0,
    //        | 1         otherwise.
    #[cfg(not(feature = "arb"))]
    pub fn sinc(&self) -> Self {
        let mut rs = Self::empty();
        for x in self {
            rs.insert(TupperInterval::new(
                DecInterval::set_dec(sinc(x.x), x.d),
                x.g,
            ));
        }
        rs
    }

    pub fn tan(&self, site: Option<Site>) -> Self {
        let mut rs = Self::empty();
        for x in self {
            let a = x.x.inf();
            let b = x.x.sup();
            let q_nowrap = (x.x / Interval::FRAC_PI_2).floor();
            let qa = q_nowrap.inf();
            let qb = q_nowrap.sup();
            let n = if a == b { 0.0 } else { qb - qa };
            let q = qa.rem_euclid(2.0);

            let cont = qb != f64::INFINITY
                && b <= (interval!(qb, qb).unwrap() * Interval::FRAC_PI_2).inf();
            if q == 0.0 && (n < 1.0 || n == 1.0 && cont)
                || q == 1.0 && (n < 2.0 || n == 2.0 && cont)
            {
                rs.insert(TupperInterval::new(x.to_dec_interval().tan(), x.g));
            } else if q == 0.0 && (n < 2.0 || n == 2.0 && cont)
                || q == 1.0 && (n < 3.0 || n == 3.0 && cont)
            {
                rs.insert(TupperInterval::new(
                    DecInterval::set_dec(
                        interval!(interval!(a, a).unwrap().tan().inf(), f64::INFINITY).unwrap(),
                        Decoration::Trv,
                    ),
                    match site {
                        Some(site) => x.g.inserted(site, Branch::new(0)),
                        _ => x.g,
                    },
                ));
                rs.insert(TupperInterval::new(
                    DecInterval::set_dec(
                        interval!(f64::NEG_INFINITY, interval!(b, b).unwrap().tan().sup()).unwrap(),
                        Decoration::Trv,
                    ),
                    match site {
                        Some(site) => x.g.inserted(site, Branch::new(1)),
                        _ => x.g,
                    },
                ));
            } else {
                rs.insert(TupperInterval::new(x.to_dec_interval().tan(), x.g));
            }
        }
        rs.normalize()
    }

    // f(x) = | x          if x ≠ 0,
    //        | undefined  otherwise.
    pub fn undef_at_0(&self) -> Self {
        let mut rs = Self::empty();
        for x in self {
            let a = x.x.inf();
            let b = x.x.sup();
            if a <= 0.0 && b >= 0.0 {
                if a < b {
                    // x ≠ {0}.
                    rs.insert(TupperInterval::new(
                        DecInterval::set_dec(x.x, Decoration::Trv),
                        x.g,
                    ));
                }
            } else {
                rs.insert(*x);
            }
        }
        rs.normalize()
    }
}

macro_rules! impl_no_cut_op {
    ($op:ident) => {
        pub fn $op(&self) -> Self {
            let mut rs = Self::empty();
            for x in self {
                rs.insert(TupperInterval::new(x.to_dec_interval().$op(), x.g));
            }
            rs.normalize()
        }
    };
}

macro_rules! impl_no_cut_op2 {
    ($op:ident) => {
        pub fn $op(&self, rhs: &Self) -> Self {
            let mut rs = Self::empty();
            for x in self {
                for y in rhs {
                    if let Some(g) = x.g.union(y.g) {
                        rs.insert(TupperInterval::new(
                            x.to_dec_interval().$op(y.to_dec_interval()),
                            g,
                        ));
                    }
                }
            }
            rs.normalize()
        }
    };
}

macro_rules! impl_integer_op {
    ($op:ident) => {
        pub fn $op(&self, site: Option<Site>) -> Self {
            let mut rs = Self::empty();
            for x in self {
                let y = TupperInterval::new(x.to_dec_interval().$op(), x.g);
                let a = y.x.inf();
                let b = y.x.sup();
                if b - a == 1.0 {
                    rs.insert(TupperInterval::new(
                        DecInterval::set_dec(interval!(a, a).unwrap(), y.d),
                        match site {
                            Some(site) => y.g.inserted(site, Branch::new(0)),
                            _ => y.g,
                        },
                    ));
                    rs.insert(TupperInterval::new(
                        DecInterval::set_dec(interval!(b, b).unwrap(), y.d),
                        match site {
                            Some(site) => y.g.inserted(site, Branch::new(1)),
                            _ => y.g,
                        },
                    ));
                } else {
                    rs.insert(y);
                }
            }
            rs.normalize()
        }
    };
}

#[cfg(not(feature = "arb"))]
macro_rules! requires_arb {
    ($op:ident) => {
        pub fn $op(&self) -> Self {
            panic!(concat!(
                "function `",
                stringify!($op),
                "` is only available when the feature `arb` is enabled"
            ))
        }
    };
}

macro_rules! impl_arb_op {
    ($op:ident($x:ident), $main:expr, $fallback_cond:expr, $fallback:expr) => {
        #[cfg(feature = "arb")]
        pub fn $op(&self) -> Self {
            let mut rs = Self::empty();
            for x in self {
                let $x = x.x;
                let y = if $fallback_cond { $fallback } else { $main };
                rs.insert(TupperInterval::new(DecInterval::set_dec(y, x.d), x.g));
            }
            rs.normalize()
        }
    };

    ($op:ident($x:ident), $main:expr) => {
        impl_arb_op!($op($x), $main, false, panic!());
    };
}

impl TupperIntervalSet {
    // absmax
    impl_no_cut_op!(abs);
    impl_no_cut_op2!(max);
    impl_no_cut_op2!(min);

    // basic
    impl_no_cut_op!(sqr);
    impl_no_cut_op!(sqrt);

    // elementary
    impl_no_cut_op!(acos);
    impl_no_cut_op!(acosh);
    impl_no_cut_op!(asin);
    impl_no_cut_op!(asinh);
    #[cfg(not(feature = "arb"))]
    impl_no_cut_op!(atan);
    impl_no_cut_op!(atanh);
    #[cfg(not(feature = "arb"))]
    impl_no_cut_op!(cos);
    impl_no_cut_op!(cosh);
    impl_no_cut_op!(exp);
    impl_no_cut_op!(exp10);
    impl_no_cut_op!(exp2);
    impl_no_cut_op!(ln);
    impl_no_cut_op!(log10);
    impl_no_cut_op!(log2);
    #[cfg(not(feature = "arb"))]
    impl_no_cut_op!(sin);
    impl_no_cut_op!(sinh);
    impl_no_cut_op!(tanh);

    // integer
    impl_integer_op!(ceil);
    impl_integer_op!(floor);
    impl_integer_op!(round);
    impl_integer_op!(round_ties_to_even);
    impl_integer_op!(sign);
    impl_integer_op!(trunc);

    #[cfg(not(feature = "arb"))]
    requires_arb!(fresnel_c);
    #[cfg(not(feature = "arb"))]
    requires_arb!(fresnel_s);

    // Use Arb for bounded functions for the moment,
    // since half-bounded intervals cannot be represented by mid-rad IA that Arb provides.
    impl_arb_op!(atan(x), arb_atan(x), !x.is_common_interval(), x.atan());
    impl_arb_op!(cos(x), arb_cos(x));
    impl_arb_op!(erf(x), arb_erf(x), !x.is_common_interval(), erf(x));
    impl_arb_op!(erfc(x), arb_erfc(x), !x.is_common_interval(), erfc(x));
    impl_arb_op!(fresnel_c(x), arb_fresnel_c(x));
    impl_arb_op!(fresnel_s(x), arb_fresnel_s(x));
    impl_arb_op!(sin(x), arb_sin(x));
    impl_arb_op!(sinc(x), arb_sinc(x), !x.is_common_interval(), sinc(x));
}

macro_rules! impl_rel_op {
    ($op:ident, $map_neg:expr, $map_zero:expr, $map_pos:expr) => {
        pub fn $op(&self, rhs: &Self) -> DecSignSet {
            let xs = self - rhs;
            if xs.is_empty() {
                return DecSignSet::empty();
            }

            let mut ss = SignSet::empty();
            let mut d = Decoration::Com;
            for x in xs {
                let a = x.x.inf();
                let b = x.x.sup();
                if a < 0.0 {
                    ss |= $map_neg;
                }
                if a <= 0.0 && b >= 0.0 {
                    ss |= $map_zero;
                }
                if b > 0.0 {
                    ss |= $map_pos;
                }
                d = d.min(x.d);
            }

            DecSignSet(ss, d)
        }
    };
}

impl TupperIntervalSet {
    impl_rel_op!(eq, SignSet::NEG, SignSet::ZERO, SignSet::POS);

    // f ≥ 0 ⟺ (f ≥ 0 ? 0 : 1) = 0, etc.
    impl_rel_op!(ge, SignSet::POS, SignSet::ZERO, SignSet::ZERO);
    impl_rel_op!(gt, SignSet::POS, SignSet::POS, SignSet::ZERO);
    impl_rel_op!(le, SignSet::ZERO, SignSet::ZERO, SignSet::POS);
    impl_rel_op!(lt, SignSet::ZERO, SignSet::POS, SignSet::POS);
}

// Copy-paste from inari/src/elementary.rs

fn mpfr_fn(
    f: unsafe extern "C" fn(*mut mpfr::mpfr_t, *const mpfr::mpfr_t, mpfr::rnd_t) -> i32,
    x: f64,
    rnd: mpfr::rnd_t,
) -> f64 {
    let mut x = Float::with_val(f64::MANTISSA_DIGITS, x);
    unsafe {
        f(x.as_raw_mut(), x.as_raw(), rnd);
        mpfr::get_d(x.as_raw(), rnd)
    }
}

macro_rules! mpfr_fn {
    ($mpfr_f:ident, $f_rd:ident, $f_ru:ident) => {
        fn $f_rd(x: f64) -> f64 {
            mpfr_fn(mpfr::$mpfr_f, x, mpfr::rnd_t::RNDD)
        }

        fn $f_ru(x: f64) -> f64 {
            mpfr_fn(mpfr::$mpfr_f, x, mpfr::rnd_t::RNDU)
        }
    };
}

mpfr_fn!(erf, erf_rd, erf_ru);
mpfr_fn!(erfc, erfc_rd, erfc_ru);
mpfr_fn!(gamma, gamma_rd, gamma_ru);

/// `x` must be nonempty.
fn erf(x: Interval) -> Interval {
    interval!(erf_rd(x.inf()), erf_ru(x.sup())).unwrap()
}

/// `x` must be nonempty.
fn erfc(x: Interval) -> Interval {
    interval!(erfc_rd(x.sup()), erfc_ru(x.inf())).unwrap()
}

/// `x` must be nonempty.
fn sinc(x: Interval) -> Interval {
    // argmin_{x > 0} sinc(x), rounded down.
    const ARGMIN_RD: f64 = 4.493409457909063;
    // min_{x > 0} sinc(x), rounded down.
    const MIN_RD: f64 = -0.21723362821122166;
    let a = x.inf();
    let b = x.sup();
    if a <= 0.0 && b >= 0.0 {
        let b2 = (-a).max(b);
        if b2 <= ARGMIN_RD {
            let x2 = interval!(b2, b2).unwrap();
            interval!((x2.sin() / x2).inf().max(MIN_RD).min(1.0), 1.0).unwrap()
        } else {
            interval!(MIN_RD, 1.0).unwrap()
        }
    } else {
        x.sin() / x
    }
}

macro_rules! arb_fn {
    ($f:ident($x:ident $(,$y:ident)*), $arb_f:ident($($args:expr),*), $range:expr) => {
        #[cfg(feature = "arb")]
        fn $f($x: Interval, $($y: Interval,)*) -> Interval {
            use crate::arb::Arb;
            let mut $x = Arb::from_interval($x);
            $(let mut $y = Arb::from_interval($y);)*
            unsafe {
                #[allow(unused_imports)]
                use std::ptr::null_mut as null;
                let $x = $x.as_raw_mut();
                $(let $y = $y.as_raw_mut();)*
                crate::arb_sys::$arb_f($($args),*);
            }
            $x.to_interval().intersection($range)
        }
    };
}

arb_fn!(
    arb_atan(x),
    arb_atan(x, x, f64::MANTISSA_DIGITS.into()),
    const_interval!(-1.5707963267948968, 1.5707963267948968)
);
arb_fn!(
    arb_cos(x),
    arb_cos(x, x, f64::MANTISSA_DIGITS.into()),
    const_interval!(-1.0, 1.0)
);
arb_fn!(
    arb_erf(x),
    // `+ 3` completes the graphing of "y = erf(1/x^21)".
    arb_hypgeom_erf(x, x, (f64::MANTISSA_DIGITS + 3).into()),
    const_interval!(-1.0, 1.0)
);
arb_fn!(
    arb_erfc(x),
    // `+ 3` completes the graphing of "y = erfc(1/x^21) + 1".
    arb_hypgeom_erfc(x, x, (f64::MANTISSA_DIGITS + 3).into()),
    const_interval!(0.0, 2.0)
);
arb_fn!(
    arb_fresnel_c(x),
    arb_hypgeom_fresnel(null(), x, x, 1, f64::MANTISSA_DIGITS.into()),
    const_interval!(-0.7798934003768229, 0.7798934003768229)
);
arb_fn!(
    arb_fresnel_s(x),
    arb_hypgeom_fresnel(x, null(), x, 1, f64::MANTISSA_DIGITS.into()),
    const_interval!(-0.7139722140219397, 0.7139722140219397)
);
arb_fn!(
    arb_sin(x),
    arb_sin(x, x, f64::MANTISSA_DIGITS.into()),
    const_interval!(-1.0, 1.0)
);
arb_fn!(
    arb_sinc(x),
    arb_sinc(x, x, f64::MANTISSA_DIGITS.into()),
    const_interval!(-0.21723362821122166, 1.0)
);
