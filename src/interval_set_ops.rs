use crate::interval_set::{
    Branch, BranchMap, DecSignSet, SignSet, Site, TupperInterval, TupperIntervalSet,
};
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
        rs // Skip normalization since negation does not produce new overlapping intervals.
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
                rs.normalize(false);
                rs
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

macro_rules! impl_op {
    ($op:ident($x:ident), $result:expr) => {
        pub fn $op(&self) -> Self {
            let mut rs = Self::empty();
            for x in self {
                let $x = x.to_dec_interval();
                rs.insert(TupperInterval::new($result, x.g));
            }
            rs.normalize(false);
            rs
        }
    };

    ($op:ident($x:ident, $y:ident), $result:expr) => {
        pub fn $op(&self, rhs: &Self) -> Self {
            let mut rs = Self::empty();
            for x in self {
                for y in rhs {
                    if let Some(g) = x.g.union(y.g) {
                        let $x = x.to_dec_interval();
                        let $y = y.to_dec_interval();
                        rs.insert(TupperInterval::new($result, g));
                    }
                }
            }
            rs.normalize(false);
            rs
        }
    };
}

fn insert_intervals(
    rs: &mut TupperIntervalSet,
    r: (DecInterval, Option<DecInterval>),
    g: BranchMap,
    site: Option<Site>,
) {
    match r {
        (r0, None) => {
            rs.insert(TupperInterval::new(r0, g));
        }
        (r0, Some(r1)) => {
            if !r0.disjoint(r1) {
                rs.insert(TupperInterval::new(
                    DecInterval::set_dec(
                        r0.interval().unwrap().convex_hull(r1.interval().unwrap()),
                        r0.decoration().min(r1.decoration()),
                    ),
                    g,
                ));
            } else {
                rs.insert(TupperInterval::new(
                    r0,
                    match site {
                        Some(site) => g.inserted(site, Branch::new(0)),
                        _ => g,
                    },
                ));
                rs.insert(TupperInterval::new(
                    r1,
                    match site {
                        Some(site) => g.inserted(site, Branch::new(1)),
                        _ => g,
                    },
                ));
            }
        }
    }
}

macro_rules! impl_op_cut {
    ($op:ident($x:ident $(,$p:ident: $pt:ty)*), $result:expr) => {
        pub fn $op(&self, $($p: $pt,)* site: Option<Site>) -> Self {
            let mut rs = Self::empty();
            for x in self {
                let $x = x.to_dec_interval();
                insert_intervals(&mut rs, $result, x.g, site);
            }
            rs.normalize(false);
            rs
        }
    };

    ($(#[$meta:meta])* $op:ident($x:ident, $y:ident), $result:expr) => {
        $(#[$meta])*
        pub fn $op(&self, rhs: &Self, site: Option<Site>) -> Self {
            let mut rs = Self::empty();
            for x in self {
                for y in rhs {
                    if let Some(g) = x.g.union(y.g) {
                        let $x = x.to_dec_interval();
                        let $y = y.to_dec_interval();
                        insert_intervals(&mut rs, $result, g, site);
                    }
                }
            }
            rs.normalize(false);
            rs
        }
    };
}

impl TupperIntervalSet {
    impl_op!(abs(x), x.abs());

    #[cfg(not(feature = "arb"))]
    impl_op!(acos(x), x.acos());

    #[cfg(not(feature = "arb"))]
    impl_op!(acosh(x), x.acosh());

    #[cfg(not(feature = "arb"))]
    impl_op!(asin(x), x.asin());

    #[cfg(not(feature = "arb"))]
    impl_op!(asinh(x), x.asinh());

    #[cfg(not(feature = "arb"))]
    impl_op!(atan(x), x.atan());

    impl_op_cut!(atan2(y, x), {
        let a = x.inf();
        let b = x.sup();
        let c = y.inf();
        let d = y.sup();
        if a == 0.0 && b == 0.0 && c < 0.0 && d > 0.0 {
            let dec = Decoration::Trv;
            (
                DecInterval::set_dec(-Interval::FRAC_PI_2, dec),
                Some(DecInterval::set_dec(Interval::FRAC_PI_2, dec)),
            )
        } else if a < 0.0 && b <= 0.0 && c < 0.0 && d >= 0.0 {
            let dec = if b == 0.0 {
                Decoration::Trv
            } else {
                Decoration::Def.min(x.decoration()).min(y.decoration())
            };
            // y < 0 (thus z < 0) part.
            let x0 = interval!(b, b).unwrap();
            let y0 = interval!(c, c).unwrap();
            let z0 = interval!(-Interval::PI.sup(), y0.atan2(x0).sup()).unwrap();
            // y ≥ 0 (thus z > 0) part.
            let x1 = interval!(a, b).unwrap();
            let y1 = interval!(0.0, d).unwrap();
            let z1 = y1.atan2(x1);
            (
                DecInterval::set_dec(z0, dec),
                Some(DecInterval::set_dec(z1, dec)),
            )
        } else {
            // a = b = c = d = 0 goes here.
            (y.atan2(x), None)
        }
    });

    #[cfg(not(feature = "arb"))]
    impl_op!(atanh(x), x.atanh());

    #[cfg(not(feature = "arb"))]
    impl_op!(cos(x), x.cos());

    #[cfg(not(feature = "arb"))]
    impl_op!(cosh(x), x.cosh());

    impl_op_cut!(digamma(x), {
        let a = x.inf();
        let b = x.sup();
        let ia = a.ceil();
        let ib = b.floor();
        if ia == ib && a <= 0.0 {
            // ∃i ∈ S : x ∩ S = {i}, where S = {0, -1, …}.
            let dec = Decoration::Trv;
            let x0 = interval!(a, ia).unwrap();
            let x1 = interval!(ia, b).unwrap();
            (
                DecInterval::set_dec(digamma(x0), dec),
                Some(DecInterval::set_dec(digamma(x1), dec)),
            )
        } else {
            let dec = if ia < ib && a <= 0.0 {
                // x ∩ S ≠ ∅.
                Decoration::Trv
            } else {
                Decoration::Com.min(x.decoration())
            };
            let x = x.interval().unwrap();
            (DecInterval::set_dec(digamma(x), dec), None)
        }
    });

    impl_op_cut!(div(x, y), {
        let c = y.inf();
        let d = y.sup();
        if c < 0.0 && d > 0.0 {
            let y0 = DecInterval::set_dec(interval!(c, 0.0).unwrap(), y.decoration());
            let y1 = DecInterval::set_dec(interval!(0.0, d).unwrap(), y.decoration());
            (x / y0, Some(x / y1))
        } else {
            (x / y, None)
        }
    });

    #[cfg(not(feature = "arb"))]
    impl_op!(erf(x), {
        DecInterval::set_dec(erf(x.interval().unwrap()), x.decoration())
    });

    #[cfg(not(feature = "arb"))]
    impl_op!(erfc(x), {
        DecInterval::set_dec(erfc(x.interval().unwrap()), x.decoration())
    });

    #[cfg(not(feature = "arb"))]
    impl_op!(exp(x), x.exp());

    #[cfg(not(feature = "arb"))]
    impl_op!(exp10(x), x.exp10());

    #[cfg(not(feature = "arb"))]
    impl_op!(exp2(x), x.exp2());

    pub fn gamma(&self, site: Option<Site>) -> Self {
        // argmin_{x > 0} Γ(x), rounded down/up.
        const ARGMIN_RD: f64 = 1.4616321449683622;
        const ARGMIN_RU: f64 = 1.4616321449683625;
        // min_{x > 0} Γ(x), rounded down.
        const MIN_RD: f64 = 0.8856031944108886;
        let mut rs = Self::empty();
        for x in self {
            let a = x.x.inf();
            let b = x.x.sup();
            if a == 0.0 && b == 0.0 {
                // empty.
            } else if a >= 0.0 {
                let dec = if a == 0.0 { Decoration::Trv } else { x.d };
                // gamma_rd/ru(±0.0) returns ±∞.
                let a = if a == 0.0 { 0.0 } else { a };

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
                let one = Self::from(const_dec_interval!(1.0, 1.0));
                let pi = Self::from(DecInterval::PI);
                let mut xs = Self::empty();
                xs.insert(*x);
                for y in pi.div(&(&(&pi * &xs).sin() * &(&one - &xs).gamma(None)), site) {
                    rs.insert(y);
                }
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
                for y in xs.gamma(None) {
                    rs.insert(y);
                }
            }
        }
        rs.normalize(false);
        rs
    }

    // For x, y ∈ ℚ, gcd(x, y) is defined recursively (the Euclidean algorithm) as:
    //
    //   gcd(x, y) = | |x|              if y = 0,
    //               | gcd(y, x mod y)  otherwise,
    //
    // assuming the recursion terminates. We leave gcd undefined For irrational numbers.
    // Here is an interval extension of it:
    //
    //                | |X|                    if Y = {0} ∨ (X = Y ∧ 0 ∈ X ∧ X mod X ⊆ |X|),
    //   gcd(X, Y) := | gcd(Y, X mod Y)        if 0 ∉ Y,
    //                | |X| ∪ gcd(Y, X mod Y)  otherwise.
    //
    // The second condition of the first case is justified by the following proposition.
    // Let X ⊆ ℚ such that 0 ∈ X ∧ X mod X ⊆ |X|. Then
    //
    //   gcd(X, X) ⊆ |X|.
    //
    // Proof: From the definition of gcd,
    //
    //   ∀x, y ∈ X : ∃n ∈ ℕ_{≥1} : ∃z_0, …, z_n ∈ |X| :
    //       z_0 = |x| ∧ z_1 = |y|
    //     ∧ ∀k ∈ {2, …, n} : z_k = z_{k-2} mod z_{k-1}
    //     ∧ z_n = 0 (thus z_{n-1} = gcd(x, y)).
    //
    // Thus ∀x, y ∈ X : gcd(x, y) ∈ |X|. ■
    //
    // TODO: Implement branch cut tracking.
    pub fn gcd(&self, rhs: &Self, _site: Option<Site>) -> Self {
        const ZERO: Interval = const_interval!(0.0, 0.0);
        let mut rs = Self::empty();
        // gcd(X, Y)
        //   = gcd(|X|, |Y|)
        //   = {gcd(max(x, y), min(x, y)) | x ∈ |X|, y ∈ |Y|}
        //   ⊆ {gcd(x, y) | x ∈ max(|X|, |Y|), y ∈ min(|X|, |Y|)}.
        let xs = &self.abs();
        let ys = &rhs.abs();
        for x in xs {
            for y in ys {
                if let Some(g) = x.g.union(y.g) {
                    let dec = if x.x.is_singleton() && y.x.is_singleton() {
                        Decoration::Dac.min(x.d).min(y.d)
                    } else {
                        Decoration::Trv
                    };
                    let x = DecInterval::set_dec(x.x, dec);
                    let y = DecInterval::set_dec(y.x, dec);
                    let mut xs = Self::from(TupperInterval::new(x.max(y), g));
                    let mut ys = Self::from(TupperInterval::new(x.min(y), g));
                    loop {
                        if ys.iter().any(|y| y.x.contains(0.0)) {
                            for x in &xs {
                                rs.insert(*x);
                            }

                            if xs == ys {
                                // Here, in the first iteration, `xs` and `ys` consists of
                                // the same, single, nonnegative interval X = |X| = [0, a].
                                // Let x, y ∈ X. Then
                                //
                                //       0 ≤ x mod y < y ≤ a
                                //   ⟹ x mod y ∈ X.
                                //
                                // Therefore, 0 ∈ X ∧ X mod X ⊆ |X|.
                                break;
                            }
                        }

                        ys.retain(|y| y.x != ZERO);
                        if ys.is_empty() {
                            break;
                        }

                        let xs_rem_ys = xs.rem_euclid(&ys, None);
                        xs = ys;
                        ys = xs_rem_ys;
                        ys.normalize(true);
                    }
                }
            }
        }
        rs.normalize(false);
        rs
    }

    // For x, y ∈ ℚ, lcm(x, y) is defined as:
    //
    //   lcm(x, y) = | 0                  if x = y = 0,
    //               | |x y| / gcd(x, y)  otherwise.
    //
    // We leave lcm undefined for irrational numbers.
    // Here is an interval extension of it:
    //
    //   lcm(X, Y) := | {0}                if X = Y = {0},
    //                | |X Y| / gcd(X, Y)  otherwise.
    pub fn lcm(&self, rhs: &Self, site: Option<Site>) -> Self {
        const ZERO: Interval = const_interval!(0.0, 0.0);
        let mut rs = (self * rhs).abs().div(&self.gcd(rhs, site), None);
        for x in self.iter().filter(|x| x.x == ZERO) {
            for y in rhs.iter().filter(|y| y.x == ZERO) {
                if let Some(g) = x.g.union(y.g) {
                    let dec = Decoration::Dac.min(x.d).min(y.d);
                    rs.insert(TupperInterval::new(DecInterval::set_dec(ZERO, dec), g));
                }
            }
        }
        rs.normalize(false);
        rs
    }

    #[cfg(not(feature = "arb"))]
    impl_op!(ln(x), x.ln());

    pub fn log(&self, rhs: &Self, site: Option<Site>) -> Self {
        self.log2().div(&rhs.log2(), site)
    }

    #[cfg(not(feature = "arb"))]
    impl_op!(log10(x), x.log10());

    #[cfg(not(feature = "arb"))]
    impl_op!(log2(x), x.log2());

    impl_op!(max(x, y), x.max(y));

    impl_op!(min(x, y), x.min(y));

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
        rs.normalize(false);
        rs
    }

    // f(x) = 1.
    impl_op!(one(x), {
        DecInterval::set_dec(const_interval!(1.0, 1.0), x.decoration())
    });

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
    impl_op_cut!(
        #[allow(clippy::many_single_char_names)]
        pow(x, y),
        {
            let a = x.inf();
            let c = y.inf();

            // | {1}  if (0, 0) ∈ x × y,
            // | ∅    otherwise.
            let one_or_empty = if x.contains(0.0) && y.contains(0.0) {
                const_interval!(1.0, 1.0)
            } else {
                Interval::EMPTY
            };

            if y.is_singleton() {
                match Self::exponentiation_parity(c) {
                    Parity::None => (x.pow(y), None),
                    Parity::Even => {
                        let z = x.abs().pow(y);
                        (
                            DecInterval::set_dec(
                                z.interval().unwrap().convex_hull(one_or_empty),
                                // It could be `Com`, but that does not matter in graphing.
                                Decoration::Dac.min(z.decoration()),
                            ),
                            None,
                        )
                    }
                    Parity::Odd => {
                        let dec = if x.contains(0.0) && c < 0.0 {
                            Decoration::Trv
                        } else {
                            Decoration::Dac.min(x.decoration()).min(y.decoration())
                        };

                        let x = x.interval().unwrap();
                        let y = y.interval().unwrap();
                        // Either x0 or x1 can be empty.
                        let x0 = x.intersection(const_interval!(f64::NEG_INFINITY, 0.0));
                        let z0 = -(-x0).pow(y);
                        let x1 = x.intersection(const_interval!(0.0, f64::INFINITY));
                        let z1 = x1.pow(y);
                        if c < 0.0 {
                            (
                                DecInterval::set_dec(z0, dec),
                                Some(DecInterval::set_dec(z1, dec)),
                            )
                        } else {
                            let z = z0.convex_hull(z1);
                            (DecInterval::set_dec(z, dec), None)
                        }
                    }
                }
            } else if a < 0.0 {
                // a < 0.
                let dec = Decoration::Trv;

                let x = x.interval().unwrap();
                let y = y.interval().unwrap();

                // x^y < 0 part from
                //   x < 0 for those ys where f(x) = x^y is an odd function.
                let x0 = x.min(const_interval!(0.0, 0.0));
                let z0 = -(-x0).pow(y);

                // x^y ≥ 0 part from
                //   x ≥ 0 (incl. 0^0), and
                //   x < 0 for those ys where f(x) = x^y is an even function.
                let z1 = x.abs().pow(y).convex_hull(one_or_empty);

                (
                    DecInterval::set_dec(z0, dec),
                    Some(DecInterval::set_dec(z1, dec)),
                )
            } else {
                // a ≥ 0.

                // If 0 ∈ x ∧ c ≤ 0 ≤ d, we need to add {1} to z manually.
                // In that case, the decoration of z is already `Trv`.
                let z = x.pow(y);
                (
                    DecInterval::set_dec(
                        z.interval().unwrap().convex_hull(one_or_empty),
                        z.decoration(),
                    ),
                    None,
                )
            }
        }
    );

    impl_op_cut!(pown(x, n: i32), {
        let a = x.inf();
        let b = x.sup();
        if n < 0 && n % 2 == -1 && a < 0.0 && b > 0.0 {
            let x0 = DecInterval::set_dec(interval!(a, 0.0).unwrap(), x.decoration());
            let x1 = DecInterval::set_dec(interval!(0.0, b).unwrap(), x.decoration());
            (x0.pown(n), Some(x1.pown(n)))
        } else {
            (x.pown(n), None)
        }
    });

    pub fn ranked_max(xs: Vec<&Self>, n: &Self, site: Option<Site>) -> Self {
        Self::ranked_min_max(xs, n, site, true)
    }

    pub fn ranked_min(xs: Vec<&Self>, n: &Self, site: Option<Site>) -> Self {
        Self::ranked_min_max(xs, n, site, false)
    }

    fn ranked_min_max(xs: Vec<&Self>, n: &Self, site: Option<Site>, max: bool) -> Self {
        use itertools::Itertools;
        assert!(!xs.is_empty());
        let mut rs = Self::empty();
        let mut infs = vec![];
        let mut sups = vec![];
        for n in n {
            // `n` uses 1-based indexing.
            let n0 = n.x - const_interval!(1.0, 1.0);
            let n0_rest = n0.intersection(interval!(0.0, (xs.len() - 1) as f64).unwrap());
            let na = n0_rest.inf().ceil() as usize;
            let nb = n0_rest.sup().floor() as usize;
            if na > nb {
                continue;
            }
            let dec = if n0.is_singleton() && na == nb {
                Decoration::Dac.min(n.d)
            } else {
                Decoration::Trv
            };
            for xs in xs.iter().copied().multi_cartesian_product() {
                if let Some(g) = xs.iter().try_fold(n.g, |g, x| g.union(x.g)) {
                    let dec = xs.iter().fold(dec, |d, x| d.min(x.d));
                    infs.splice(.., xs.iter().map(|x| x.x.inf()));
                    infs.sort_by(|x, y| x.partial_cmp(y).unwrap());
                    sups.splice(.., xs.iter().map(|x| x.x.sup()));
                    sups.sort_by(|x, y| x.partial_cmp(y).unwrap());
                    if nb == na + 1 {
                        let y0 = DecInterval::set_dec(interval!(infs[na], sups[na]).unwrap(), dec);
                        let y1 = DecInterval::set_dec(interval!(infs[nb], sups[nb]).unwrap(), dec);
                        insert_intervals(&mut rs, (y0, Some(y1)), g, site);
                    } else {
                        for i in na..=nb {
                            let i = if max { xs.len() - 1 - i } else { i };
                            let y = DecInterval::set_dec(interval!(infs[i], sups[i]).unwrap(), dec);
                            rs.insert(TupperInterval::new(y, g));
                        }
                    }
                }
            }
        }
        rs.normalize(false);
        rs
    }

    impl_op_cut!(recip(x), {
        let a = x.inf();
        let b = x.sup();
        if a < 0.0 && b > 0.0 {
            let x0 = DecInterval::set_dec(interval!(a, 0.0).unwrap(), x.decoration());
            let x1 = DecInterval::set_dec(interval!(0.0, b).unwrap(), x.decoration());
            (x0.recip(), Some(x1.recip()))
        } else {
            (x.recip(), None)
        }
    });

    impl_op_cut!(rem_euclid(x, y), {
        // Compute x - |y| ⌊x / |y|⌋.
        const ZERO: DecInterval = const_dec_interval!(0.0, 0.0);
        let y = y.abs(); // Take abs, normalize, then iterate would be better.
        let q = (x / y).floor();
        let a = q.inf();
        let b = q.sup();
        if b - a == 1.0 {
            let q0 = DecInterval::set_dec(interval!(a, a).unwrap(), q.decoration());
            let q1 = DecInterval::set_dec(interval!(b, b).unwrap(), q.decoration());
            (
                (-y).mul_add(q0, x).max(ZERO).min(y),
                Some((-y).mul_add(q1, x).max(ZERO).min(y)),
            )
        } else {
            ((-y).mul_add(q, x).max(ZERO).min(y), None)
        }
    });

    #[cfg(not(feature = "arb"))]
    impl_op!(sin(x), x.sin());

    // f(x) = | sin(x)/x  if x ≠ 0,
    //        | 1         otherwise.
    #[cfg(not(feature = "arb"))]
    impl_op!(sinc(x), {
        DecInterval::set_dec(sinc(x.interval().unwrap()), x.decoration())
    });

    #[cfg(not(feature = "arb"))]
    impl_op!(sinh(x), x.sinh());

    impl_op!(sqr(x), x.sqr());

    impl_op!(sqrt(x), x.sqrt());

    impl_op_cut!(tan(x), {
        let a = x.inf();
        let b = x.sup();
        let q_nowrap = (x.interval().unwrap() / Interval::FRAC_PI_2).floor();
        let qa = q_nowrap.inf();
        let qb = q_nowrap.sup();
        let n = if a == b { 0.0 } else { qb - qa };
        let q = qa.rem_euclid(2.0);

        let cont =
            qb != f64::INFINITY && b <= (interval!(qb, qb).unwrap() * Interval::FRAC_PI_2).inf();
        if q == 0.0 && (n < 1.0 || n == 1.0 && cont) || q == 1.0 && (n < 2.0 || n == 2.0 && cont) {
            (x.tan(), None)
        } else if q == 0.0 && (n < 2.0 || n == 2.0 && cont)
            || q == 1.0 && (n < 3.0 || n == 3.0 && cont)
        {
            let dec = Decoration::Trv;
            let y0 = interval!(interval!(a, a).unwrap().tan().inf(), f64::INFINITY).unwrap();
            let y1 = interval!(f64::NEG_INFINITY, interval!(b, b).unwrap().tan().sup()).unwrap();
            (
                DecInterval::set_dec(y0, dec),
                Some(DecInterval::set_dec(y1, dec)),
            )
        } else {
            (x.tan(), None)
        }
    });

    #[cfg(not(feature = "arb"))]
    impl_op!(tanh(x), x.tanh());

    // f(x) = | x          if x ≠ 0,
    //        | undefined  otherwise.
    impl_op!(undef_at_0(x), {
        if x.contains(0.0) {
            if x.is_singleton() {
                // x = {0}.
                DecInterval::EMPTY
            } else {
                DecInterval::set_dec(x.interval().unwrap(), Decoration::Trv)
            }
        } else {
            x
        }
    });
}

macro_rules! impl_integer_op {
    ($op:ident) => {
        impl_op_cut!($op(x), {
            let y = x.$op();
            let a = y.inf();
            let b = y.sup();
            if b - a == 1.0 {
                let y0 = interval!(a, a).unwrap();
                let y1 = interval!(b, b).unwrap();
                (
                    DecInterval::set_dec(y0, y.decoration()),
                    Some(DecInterval::set_dec(y1, y.decoration())),
                )
            } else {
                (y, None)
            }
        });
    };
}

impl TupperIntervalSet {
    impl_integer_op!(ceil);
    impl_integer_op!(floor);
    impl_integer_op!(round);
    impl_integer_op!(round_ties_to_even);
    impl_integer_op!(sign);
    impl_integer_op!(trunc);
}

macro_rules! requires_arb {
    ($op:ident($x:ident $(,$y:ident)*)) => {
        #[cfg(not(feature = "arb"))]
        #[allow(unused_variables)]
        pub fn $op(&self $(,$y: &Self)*) -> Self {
            panic!(concat!(
                "function `",
                stringify!($op),
                "` is only available when the feature `arb` is enabled"
            ))
        }
    };
}

impl TupperIntervalSet {
    requires_arb!(airy_ai(x));
    requires_arb!(airy_ai_prime(x));
    requires_arb!(airy_bi(x));
    requires_arb!(airy_bi_prime(x));
    requires_arb!(bessel_i(n, x));
    requires_arb!(bessel_j(n, x));
    requires_arb!(bessel_k(n, x));
    requires_arb!(bessel_y(n, x));
    requires_arb!(chi(x));
    requires_arb!(ci(x));
    requires_arb!(ei(x));
    requires_arb!(erfi(x));
    requires_arb!(fresnel_c(x));
    requires_arb!(fresnel_s(x));
    requires_arb!(gamma_inc(a, x));
    requires_arb!(li(x));
    requires_arb!(shi(x));
    requires_arb!(si(x));
}

impl TupperIntervalSet {
    pub fn eq(&self, rhs: &Self) -> DecSignSet {
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
                ss |= SignSet::NEG;
            }
            if a <= 0.0 && b >= 0.0 {
                ss |= SignSet::ZERO;
            }
            if b > 0.0 {
                ss |= SignSet::POS;
            }
            d = d.min(x.d);
        }

        DecSignSet(ss, d)
    }
}

macro_rules! impl_rel_op {
    ($op:ident, $map_neg:expr, $map_zero:expr, $map_pos:expr, $map_undef:expr) => {
        pub fn $op(&self, rhs: &Self) -> DecSignSet {
            fn bool_to_sign(b: bool) -> SignSet {
                if b {
                    SignSet::ZERO
                } else {
                    SignSet::POS
                }
            }

            let xs = self - rhs;
            let ss = if xs.is_empty() {
                bool_to_sign($map_undef)
            } else {
                let mut ss = SignSet::empty();
                for x in xs {
                    let a = x.x.inf();
                    let b = x.x.sup();
                    if a < 0.0 {
                        ss |= bool_to_sign($map_neg);
                    }
                    if a <= 0.0 && b >= 0.0 {
                        ss |= bool_to_sign($map_zero);
                    }
                    if b > 0.0 {
                        ss |= bool_to_sign($map_pos);
                    }
                    if x.d == Decoration::Trv {
                        ss |= bool_to_sign($map_undef);
                    }
                }
                ss
            };
            let d = match ss {
                SignSet::ZERO => Decoration::Dac,
                _ => Decoration::Def,
            };

            DecSignSet(ss, d)
        }
    };
}

impl TupperIntervalSet {
    impl_rel_op!(ge, false, true, true, false);
    impl_rel_op!(gt, false, false, true, false);
    impl_rel_op!(le, true, true, false, false);
    impl_rel_op!(lt, true, false, false, false);
    impl_rel_op!(neq, true, false, true, true);
    impl_rel_op!(nge, true, false, false, true);
    impl_rel_op!(ngt, true, true, false, true);
    impl_rel_op!(nle, false, false, true, true);
    impl_rel_op!(nlt, false, true, true, true);
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

mpfr_fn!(digamma, digamma_rd, digamma_ru);
mpfr_fn!(erf, erf_rd, erf_ru);
mpfr_fn!(erfc, erfc_rd, erfc_ru);
mpfr_fn!(gamma, gamma_rd, gamma_ru);

/// `x` must be nonempty.
pub(crate) fn digamma(x: Interval) -> Interval {
    let a = x.inf();
    let b = x.sup();
    let ia = a.ceil();
    let ib = b.floor();
    if x.is_singleton() && a == ia && a <= 0.0 {
        // ∃i ∈ S : x = {i}, where S = {0, -1, …}.
        Interval::EMPTY
    } else if a < 0.0 && (a < ia && ia <= ib && ib < b || x.wid() >= 1.0) {
        // (∃i ∈ S : a < i < b) ∨ (a < 0 ∧ b - a ≥ 1).
        Interval::ENTIRE
    } else {
        let inf = if a == ia && a <= 0.0 {
            f64::NEG_INFINITY
        } else {
            digamma_rd(a)
        };
        let sup = if b == ib && b <= 0.0 {
            f64::INFINITY
        } else {
            digamma_ru(b)
        };
        interval!(inf, sup).unwrap()
    }
}

/// `x` must be nonempty.
pub(crate) fn erf(x: Interval) -> Interval {
    interval!(erf_rd(x.inf()), erf_ru(x.sup())).unwrap()
}

/// `x` must be nonempty.
pub(crate) fn erfc(x: Interval) -> Interval {
    interval!(erfc_rd(x.sup()), erfc_ru(x.inf())).unwrap()
}

/// `x` must be nonempty.
pub(crate) fn sinc(x: Interval) -> Interval {
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
