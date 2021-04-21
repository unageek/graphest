use crate::interval_set::{
    Branch, BranchMap, DecSignSet, SignSet, Site, TupperInterval, TupperIntervalSet,
};
use gmp_mpfr_sys::mpfr;
use inari::{const_dec_interval, const_interval, interval, DecInterval, Decoration, Interval};
use rug::Float;
use smallvec::{smallvec, SmallVec};
use std::{
    convert::From,
    ops::{Add, Mul, Neg, Sub},
};

impl Neg for &TupperIntervalSet {
    type Output = TupperIntervalSet;

    fn neg(self) -> Self::Output {
        let mut rs = Self::Output::new();
        for x in self {
            rs.insert(TupperInterval::new(-x.dec_interval(), x.g));
        }
        rs // Skip normalization since negation does not produce new overlapping intervals.
    }
}

macro_rules! impl_arith_op {
    ($Op:ident, $op:ident) => {
        impl<'a, 'b> $Op<&'b TupperIntervalSet> for &'a TupperIntervalSet {
            type Output = TupperIntervalSet;

            fn $op(self, rhs: &'b TupperIntervalSet) -> Self::Output {
                let mut rs = Self::Output::new();
                for x in self {
                    for y in rhs {
                        if let Some(g) = x.g.union(y.g) {
                            rs.insert(TupperInterval::new(
                                x.dec_interval().$op(y.dec_interval()),
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
    ($op:ident($x:ident $(,$p:ident: $pt:ty)*), $result:expr) => {
        pub fn $op(&self, $($p: $pt,)*) -> Self {
            let mut rs = Self::new();
            for x in self {
                let $x = x.dec_interval();
                rs.insert(TupperInterval::new($result, x.g));
            }
            rs.normalize(false);
            rs
        }
    };

    ($op:ident($x:ident, $y:ident), $result:expr) => {
        pub fn $op(&self, rhs: &Self) -> Self {
            let mut rs = Self::new();
            for x in self {
                for y in rhs {
                    if let Some(g) = x.g.union(y.g) {
                        let $x = x.dec_interval();
                        let $y = y.dec_interval();
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
            let mut rs = Self::new();
            for x in self {
                let $x = x.dec_interval();
                insert_intervals(&mut rs, $result, x.g, site);
            }
            rs.normalize(false);
            rs
        }
    };

    ($(#[$meta:meta])* $op:ident($x:ident, $y:ident), $result:expr) => {
        $(#[$meta])*
        pub fn $op(&self, rhs: &Self, site: Option<Site>) -> Self {
            let mut rs = Self::new();
            for x in self {
                for y in rhs {
                    if let Some(g) = x.g.union(y.g) {
                        let $x = x.dec_interval();
                        let $y = y.dec_interval();
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
        } else if a < 0.0 && b > 0.0 && c == 0.0 && d == 0.0 {
            let dec = Decoration::Trv;
            (
                DecInterval::set_dec(const_interval!(0.0, 0.0), dec),
                Some(DecInterval::set_dec(Interval::PI, dec)),
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
        let mut rs = Self::new();
        for x in self {
            let a = x.x.inf();
            let b = x.x.sup();
            if b <= 0.0 {
                // b ≤ 0.
                if a == b && a == a.floor() {
                    // Empty.
                } else {
                    // Γ(x) = π / (sin(π x) Γ(1 - x)).
                    let one = Self::from(const_dec_interval!(1.0, 1.0));
                    let pi = Self::from(DecInterval::PI);
                    let mut xs = Self::new();
                    xs.insert(*x);
                    let mut sin = (&pi * &xs).sin();
                    // `a.floor() + 1.0` can be inexact when the first condition is not met.
                    if x.x.wid() <= 1.0 && b <= a.floor() + 1.0 {
                        let zero = Self::from(const_dec_interval!(0.0, 0.0));
                        sin = if a.floor() % 2.0 == 0.0 {
                            // ∃k ∈ ℤ : 2k ≤ x ≤ 2k + 1 ⟹ sin(π x) ≥ 0.
                            sin.max(&zero)
                        } else {
                            // ∃k ∈ ℤ : 2k - 1 ≤ x ≤ 2k ⟹ sin(π x) ≤ 0.
                            sin.min(&zero)
                        };
                    }
                    let gamma = pi.div(&(&sin * &(&one - &xs).gamma(None)), site);
                    rs.extend(gamma);
                }
            } else if a < 0.0 {
                // a < 0 < b.
                let mut xs = Self::new();
                let dec = Decoration::Trv;
                // We cannot use `insert_intervals` here as it merges overlapping intervals.
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
                rs.extend(xs.gamma(None));
            } else {
                // 0 ≤ a.
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
            }
        }
        rs.normalize(false);
        rs
    }

    // For any x, y ∈ ℚ, the GCD (greatest common divisor) of x and y is defined as an extension
    // to the integer GCD as:
    //
    //   gcd(x, y) := gcd(q x, q y) / q,
    //
    // where q is any positive integer that satisfies q x, q y ∈ ℤ (the trivial one is the product
    // of the denominators of |x| and |y|).  We leave the function undefined for irrational numbers.
    // The Euclidean algorithm can be applied to compute the GCD of rational numbers:
    //
    //   gcd(x, y) = | |x|              if y = 0,
    //               | gcd(y, x mod y)  otherwise,
    //
    // which can be seen from a simple observation:
    //
    //   gcd(q x, q y) / q = | |q x| / q                      if y = 0,
    //                       | gcd(q y, (q x) mod (q y)) / q  otherwise,
    //                           (the Euclidean algorithm for the integer GCD)
    //                     = | |x|                            if y = 0,
    //                       | gcd(y, ((q x) mod (q y)) / q)  otherwise.
    //                                ^^^^^^^^^^^^^^^^^^^^^ = x mod y
    //
    // We construct an interval extension of the function as follows:
    //
    //   R_0(X, Y) := X,
    //   R_1(X, Y) := Y,
    //   R_k(X, Y) := R_{k-2}(X, Y) mod R_{k-1}(X, Y) for any k ∈ ℕ_{≥2},
    //
    //   Z_0(X, Y) := ∅,
    //   Z_k(X, Y) := | Z_{k-1}(X, Y) ∪ |R_{k-1}(X, Y)|  if 0 ∈ R_k(X, Y),
    //                | Z_{k-1}(X, Y)                    otherwise,
    //                for any k ∈ ℕ_{≥1},
    //
    //   gcd(X, Y) :=     ⋃      Z_k(X, Y).
    //                k ∈ ℕ_{≥0}
    //
    // We will denote R_k(X, Y) and Z_k(X, Y) just by R_k and Z_k, respectively.
    //
    // Proposition.  gcd(X, Y) is an interval extension of gcd(x, y).
    //
    // Proof.  Let X and Y be any intervals.  There are two possibilities:
    //
    //   (1):  X ∩ ℚ = ∅ ∨ Y ∩ ℚ = ∅,
    //   (2):  X ∩ ℚ ≠ ∅ ∧ Y ∩ ℚ ≠ ∅.
    //
    // Suppose (1).  Then, gcd[X, Y] = ∅ ⊆ gcd(X, Y).
    // Suppose (2).  Let x ∈ X ∩ ℚ, y ∈ Y ∩ ℚ.
    // Let r_0 := x, r_1 := y, r_k := r_{k-2} mod r_{k-1} for k ≥ 2.
    // Let P(k) :⟺ r_k ∈ R_k.  We show that P(k) holds for every k ≥ 0 by induction on k.
    // Base cases:  From r_0 = x ∈ X = R_0 and r_1 = y ∈ Y = R_1, P(0) and P(1) holds.
    // Inductive step:  Let k ≥ 0.  Suppose P(k), P(k + 1).
    // Since X mod Y is an interval extension of x mod y, the following holds:
    //
    //   r_{k+2} = r_k mod r_{k+1} ∈ R_k mod R_{k+1} = R_{k+2}.
    //
    // Thus, P(k + 2) holds.  Therefore, P(k) holds for every k ∈ ℕ_{≥0}.
    // Since the Euclidean algorithm halts on any input, there exists n ≥ 1 such that
    // r_n = 0 ∧ ∀k ∈ {2, …, n-1} : r_k ≠ 0, which leads to |r_{n-1}| = gcd(x, y).
    // Let n be such a number.  Then from r_n = 0 and r_n ∈ R_n, 0 ∈ R_n.  Therefore:
    //
    //   gcd(x, y) = |r_{n-1}| ∈ |R_{n-1}| ⊆ Z_n ⊆ gcd(X, Y).
    //
    // Therefore, for any intervals X and Y, gcd[X, Y] ⊆ gcd(X, Y).  ■
    //
    // Proposition.  For any intervals X and Y, and any k ≥ 2,
    // ∃i ∈ {1, …, k - 1} : R_{i-1} = R_{k-1} ∧ R_i = R_k ∧ Z_{i-1} = Z_{k-1} ⟹ gcd(X, Y) = Z_{k-1}.
    // The statement may look a bit awkward, but it makes the implementation easier.
    //
    // Proof.  For any j ≥ 1, Z_j can be written in the form:
    //
    //   Z_j = f(Z_{j-1}, R_{j-1}, R_j),
    //
    // where f is common for every j.
    // Suppose ∃i ∈ {1, …, k - 1} : R_{i-1} = R{k-1} ∧ R_i = R_k ∧ Z_{i-1} = Z_{k-1}.
    // Let i be such a number.  Let n := k - i.  Then:
    //
    //   Z_{i+n} = f(Z_{i+n-1}, R_{i+n-1}, R_{i+n})
    //           = f(Z_{i-1}, R_{i-1}, R_i)
    //           = Z_i.
    //
    // By repeating the process, we get ∀m ∈ ℕ_{≥0} : Z_{i+mn} = Z_i.
    // Therefore, ∀j ∈ ℕ_{≥i} : Z_j = Z_i.
    // Therefore, gcd(X, Y) = Z_i = Z_{k-1}.  ■
    pub fn gcd(&self, rhs: &Self, site: Option<Site>) -> Self {
        let mut rs = Self::new();
        // {gcd(x, y) | x ∈ X, y ∈ Y}
        //   = {gcd(x, y) | x ∈ |X|, y ∈ |Y|}
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
                    let mut zs = TupperIntervalSet::new();
                    let mut zs_prev = zs.clone();
                    let mut rems: SmallVec<[_; 4]> = smallvec![
                        Self::from(TupperInterval::new(x.max(y), g)),
                        Self::from(TupperInterval::new(x.min(y), g))
                    ];
                    'outer: loop {
                        // The iteration starts with k = 1.
                        let xs = &rems[rems.len() - 2];
                        let ys = &rems[rems.len() - 1];
                        // R_{k-1} = `xs`, R_k = `ys`, Z_{k-1} = `zs_prev`.
                        for i_prime in 0..rems.len() - 2 {
                            // `i_prime` is i with some offset subtracted.
                            // We have  (1): 1 ≤ i < k,  (2): Z_{i-1} = Z_i = … = Z_{k-1}.
                            if &rems[i_prime] == xs && &rems[i_prime + 1] == ys {
                                // We have R_{i-1} = R_{k-1} ∧ R_i = R_k.
                                // Therefore, gcd(X, Y) = Z_{k-1}.
                                break 'outer;
                            }
                        }

                        // (used later) R_{k+1} = `rem`.
                        let mut rem = xs.rem_euclid(&ys, None);
                        rem.normalize(true);

                        if ys.iter().any(|y| y.x.contains(0.0)) {
                            zs.extend(xs);
                            zs.normalize(true);
                            // Z_k = `zs`.
                            if zs != zs_prev {
                                // Z_k ≠ Z_{k-1}.
                                // Retain only R_k so that both (1) and (2) will hold
                                // in subsequent iterations.
                                rems = rems[rems.len() - 1..].into();
                                zs_prev = zs.clone();
                            }
                        }
                        rems.push(rem); // […, R_k, R_{k+1}]
                    }
                    rs.extend(zs_prev);
                }
            }
        }
        rs.normalize(true);
        if let Some(site) = site {
            if rs.len() == 2 {
                // Assign branches.
                rs = rs
                    .into_iter()
                    .enumerate()
                    .map(|(i, x)| {
                        TupperInterval::new(
                            x.dec_interval(),
                            x.g.inserted(site, Branch::new(i as u8)),
                        )
                    })
                    .collect();
            }
        }
        rs
    }

    // For x, y ∈ ℚ, the LCM (least common multiple) of x and y is defined as:
    //
    //   lcm(x, y) = | 0                  if x = y = 0,
    //               | |x y| / gcd(x, y)  otherwise.
    //
    // We leave the function undefined for irrational numbers.
    // Here is an interval extension of the function:
    //
    //   lcm(X, Y) := | {0}                if X = Y = {0},
    //                | |X Y| / gcd(X, Y)  otherwise.
    //
    // Proposition.  lcm(X, Y) is an interval extension of lcm(x, y).
    //
    // Proof.  Let X and Y be any intervals.  There are five possibilities:
    //
    //   (1):  X ∩ ℚ = ∅ ∨ Y ∩ ℚ = ∅,
    //   (2):  X = Y = {0},
    //   (3):  X = {0} ∧ Y ∩ ℚ\{0} ≠ ∅,
    //   (4):  X ∩ ℚ\{0} ≠ ∅ ∧ Y = {0},
    //   (5):  X ∩ ℚ\{0} ≠ ∅ ∧ Y ∩ ℚ\{0} ≠ ∅.
    //
    // Suppose (1).  Then, lcm[X, Y] = ∅ ⊆ lcm(X, Y).
    // Suppose (2).  Then, lcm[X, Y] = lcm(X, Y) = {0}.
    // Suppose (3).  As Y ≠ {0}, lcm(X, Y) = |X Y| / gcd(X, Y).
    // Therefore, from 0 ∈ |X Y| and ∃y ∈ Y ∩ ℚ\{0} : |y| ∈ gcd(X, Y), 0 ∈ lcm(X, Y).
    // Therefore, lcm[X, Y] = {0} ⊆ lcm(X, Y).
    // Suppose (4).  In the same manner, lcm[X, Y] ⊆ lcm(X, Y).
    // Suppose (5).  Let x ∈ X ∩ ℚ\{0}, y ∈ Y ∩ ℚ\{0} ≠ ∅.
    // Then, |x y| / gcd(x, y) ∈ lcm(X, Y) = |X Y| / gcd(X, Y).
    // Therefore, lcm[X, Y] ⊆ lcm(X, Y).
    //
    // Hence, the result.  ■
    pub fn lcm(&self, rhs: &Self, site: Option<Site>) -> Self {
        const ZERO: Interval = const_interval!(0.0, 0.0);
        let mut rs = TupperIntervalSet::new();
        for x in self {
            for y in rhs {
                if let Some(g) = x.g.union(y.g) {
                    if x.x == ZERO && y.x == ZERO {
                        let dec = Decoration::Dac.min(x.d).min(y.d);
                        rs.insert(TupperInterval::new(DecInterval::set_dec(ZERO, dec), g));
                    } else {
                        let xs = &TupperIntervalSet::from(*x);
                        let ys = &TupperIntervalSet::from(*y);
                        rs.extend((xs * ys).abs().div(&xs.gcd(ys, site), None).into_iter());
                    }
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
        let mut rs = Self::new();
        for x in self {
            for y in rhs {
                if let Some(g) = x.g.union(y.g) {
                    for z in addend {
                        if let Some(g) = g.union(z.g) {
                            rs.insert(TupperInterval::new(
                                x.dec_interval().mul_add(y.dec_interval(), z.dec_interval()),
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

    // For any integer m and any positive odd integer n, we define
    //
    //   x^(m/n) = rootn(x, n)^m,
    //
    // where rootn(x, n) is the real-valued nth root of x.  Therefore, for x < 0,
    //
    //         | (-x)^y     if y = (even)/(odd)
    //         |            (x^y is an even function of x),
    //   x^y = | -(-x)^y    if y = (odd)/(odd)
    //         |            (x^y is an odd function of x),
    //         | undefined  otherwise (y = (odd)/(even) or irrational).
    //
    // We also define 0^0 = 1.  `Interval::pow` is defined neither for x < 0 nor for x = y = 0,
    // so we extend it here.
    impl_op_cut!(
        #[allow(clippy::many_single_char_names)]
        pow(x, y),
        {
            let a = x.inf();
            let c = y.inf();
            if y.is_singleton() {
                match Self::exponentiation_parity(c) {
                    Parity::None => (x.pow(y), None),
                    Parity::Even => {
                        let dec = if x.contains(0.0) && c < 0.0 {
                            Decoration::Trv
                        } else {
                            if a > 0.0 || a >= 0.0 && c > 0.0 {
                                Decoration::Com
                            } else {
                                Decoration::Dac
                            }
                            .min(x.decoration())
                            .min(y.decoration())
                        };

                        let x = x.interval().unwrap();
                        let y = y.interval().unwrap();
                        let mut z = x.abs().pow(y);
                        if x.contains(0.0) && c == 0.0 {
                            z = z.convex_hull(const_interval!(1.0, 1.0));
                        }
                        let z = DecInterval::set_dec(z, dec);
                        (z, None)
                    }
                    Parity::Odd => {
                        let dec = if x.contains(0.0) && c < 0.0 {
                            Decoration::Trv
                        } else {
                            if a >= 0.0 {
                                Decoration::Com
                            } else {
                                Decoration::Dac
                            }
                            .min(x.decoration())
                            .min(y.decoration())
                        };

                        let x = x.interval().unwrap();
                        let y = y.interval().unwrap();
                        let x0 = x.intersection(const_interval!(f64::NEG_INFINITY, 0.0));
                        let z0 = if x0.is_empty() {
                            None
                        } else {
                            Some(DecInterval::set_dec(-(-x0).pow(y), dec))
                        };
                        let x1 = x.intersection(const_interval!(0.0, f64::INFINITY));
                        let z1 = if x1.is_empty() {
                            None
                        } else {
                            Some(DecInterval::set_dec(x1.pow(y), dec))
                        };

                        if c < 0.0 {
                            match (z0, z1) {
                                (Some(z0), _) => (z0, z1),
                                (_, Some(z1)) => (z1, z0),
                                _ => panic!(),
                            }
                        } else {
                            let z = z0
                                .unwrap_or(DecInterval::EMPTY)
                                .convex_hull(z1.unwrap_or(DecInterval::EMPTY));
                            let z = DecInterval::set_dec(z.interval().unwrap(), dec);
                            (z, None)
                        }
                    }
                }
            } else if a < 0.0 {
                // a < 0.
                let dec = Decoration::Trv;

                let x = x.interval().unwrap();
                let y = y.interval().unwrap();

                // x^y < 0 part, which comes from
                //   x < 0, y = (odd)/(odd) (x^y is an odd function of x).
                let x0 = x.min(const_interval!(0.0, 0.0));
                let z0 = DecInterval::set_dec(-(-x0).pow(y), dec);

                // x^y ≥ 0 part, which comes from
                //   x ≥ 0;
                //   x < 0, y = (even)/(odd) (x^y is an even function of x).
                let mut z1 = DecInterval::set_dec(x.abs().pow(y), dec);
                if x.contains(0.0) && y.contains(0.0) {
                    z1 = z1.convex_hull(const_dec_interval!(1.0, 1.0));
                }

                (z0, Some(z1))
            } else {
                // a ≥ 0.
                if x.contains(0.0) && y.contains(0.0) {
                    let dec = if c < 0.0 {
                        Decoration::Trv
                    } else {
                        // y is not a singleton, thus there is a discontinuity.
                        Decoration::Def.min(x.decoration()).min(y.decoration())
                    };
                    let z0 = x.pow(y);
                    let z0 = DecInterval::set_dec(z0.interval().unwrap(), dec);
                    let z1 = if x.is_singleton() {
                        // a = b = 0.
                        Some(DecInterval::set_dec(const_interval!(1.0, 1.0), dec))
                    } else {
                        None
                    };
                    (z0, z1)
                } else {
                    (x.pow(y), None)
                }
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
        let mut rs = Self::new();
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
        let y = y.abs(); // Take abs, normalize, then iterate over could be better.
        let q = (x / y).floor();
        let qa = q.inf();
        let qb = q.sup();
        let range = interval!(0.0, y.sup()).unwrap();
        if qb - qa == 1.0 {
            let q0 = DecInterval::set_dec(interval!(qa, qa).unwrap(), q.decoration());
            let q1 = DecInterval::set_dec(interval!(qb, qb).unwrap(), q.decoration());
            let z0 = (-y).mul_add(q0, x);
            let z1 = (-y).mul_add(q1, x);
            let z0 =
                DecInterval::set_dec(z0.interval().unwrap().intersection(range), z0.decoration());
            let z1 =
                DecInterval::set_dec(z1.interval().unwrap().intersection(range), z1.decoration());
            (z0, Some(z1))
        } else {
            let z = (-y).mul_add(q, x);
            let z = DecInterval::set_dec(z.interval().unwrap().intersection(range), z.decoration());
            (z, None)
        }
    });

    impl_op!(rootn(x, n: u32), {
        if n == 0 {
            DecInterval::EMPTY
        } else if n % 2 == 0 {
            const DOM: Interval = const_interval!(0.0, f64::INFINITY);
            let dec = if x.interval().unwrap().subset(DOM) {
                x.decoration()
            } else {
                Decoration::Trv
            };
            DecInterval::set_dec(rootn(x.interval().unwrap(), n), dec)
        } else {
            DecInterval::set_dec(rootn(x.interval().unwrap(), n), x.decoration())
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
    requires_arb!(elliptic_e(x));
    requires_arb!(elliptic_k(x));
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
            return DecSignSet(SignSet::empty(), Decoration::Trv);
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

fn mpfr_fn_ui(
    f: unsafe extern "C" fn(*mut mpfr::mpfr_t, *const mpfr::mpfr_t, u64, mpfr::rnd_t) -> i32,
    x: f64,
    y: u64,
    rnd: mpfr::rnd_t,
) -> f64 {
    let mut x = Float::with_val(f64::MANTISSA_DIGITS, x);
    unsafe {
        f(x.as_raw_mut(), x.as_raw(), y, rnd);
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

macro_rules! mpfr_fn_ui {
    ($mpfr_f:ident, $f_rd:ident, $f_ru:ident) => {
        fn $f_rd(x: f64, y: u32) -> f64 {
            mpfr_fn_ui(mpfr::$mpfr_f, x, y as u64, mpfr::rnd_t::RNDD)
        }

        fn $f_ru(x: f64, y: u32) -> f64 {
            mpfr_fn_ui(mpfr::$mpfr_f, x, y as u64, mpfr::rnd_t::RNDU)
        }
    };
}

mpfr_fn!(digamma, digamma_rd, digamma_ru);
mpfr_fn!(erf, erf_rd, erf_ru);
mpfr_fn!(erfc, erfc_rd, erfc_ru);
mpfr_fn!(gamma, gamma_rd, gamma_ru);
mpfr_fn_ui!(rootn_ui, rootn_rd, rootn_ru);

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
pub(crate) fn rootn(x: Interval, n: u32) -> Interval {
    if n == 0 {
        Interval::EMPTY
    } else if n % 2 == 0 {
        const DOM: Interval = const_interval!(0.0, f64::INFINITY);
        let x = x.intersection(DOM);
        if x.is_empty() {
            return Interval::EMPTY;
        }
        let a = x.inf();
        let b = x.sup();
        interval!(rootn_rd(a, n), rootn_ru(b, n)).unwrap()
    } else {
        let a = x.inf();
        let b = x.sup();
        interval!(rootn_rd(a, n), rootn_ru(b, n)).unwrap()
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use inari::{Decoration::*, *};

    macro_rules! i {
        ($a:expr) => {
            const_interval!($a, $a)
        };

        ($a:expr, $b:expr) => {
            const_interval!($a, $b)
        };
    }

    fn test1<F>(f: F, x: Interval, expected: (Vec<Interval>, Decoration))
    where
        F: Fn(TupperIntervalSet) -> TupperIntervalSet,
    {
        let decs = [Com, Dac, Def, Trv];
        let mut y_exp = expected
            .0
            .into_iter()
            .map(|x| TupperInterval::new(DecInterval::new(x), BranchMap::new()))
            .collect::<TupperIntervalSet>();
        y_exp.normalize(true);
        for &dx in &decs {
            let x = TupperIntervalSet::from(DecInterval::set_dec(x, dx));
            let mut y = f(x);
            y.normalize(true);
            let dy = if y.is_empty() {
                Decoration::Trv
            } else {
                y.iter().next().unwrap().d
            };
            let dy_exp = expected.1.min(dx);
            assert_eq!(y, y_exp);
            assert_eq!(dy, dy_exp);
        }
    }

    fn test2<F>(f: F, x: Interval, y: Interval, expected: (Vec<Interval>, Decoration))
    where
        F: Fn(TupperIntervalSet, TupperIntervalSet) -> TupperIntervalSet,
    {
        let decs = [Com, Dac, Def, Trv];
        let mut z_exp = expected
            .0
            .into_iter()
            .map(|x| TupperInterval::new(DecInterval::new(x), BranchMap::new()))
            .collect::<TupperIntervalSet>();
        z_exp.normalize(true);
        for &dx in &decs {
            for &dy in &decs {
                let x = TupperIntervalSet::from(DecInterval::set_dec(x, dx));
                let y = TupperIntervalSet::from(DecInterval::set_dec(y, dy));
                let mut z = f(x, y);
                z.normalize(true);
                let dz = if z.is_empty() {
                    Decoration::Trv
                } else {
                    z.iter().next().unwrap().d
                };
                let dz_exp = expected.1.min(dx).min(dy);
                assert_eq!(z, z_exp);
                assert_eq!(dz, dz_exp);
            }
        }
    }

    fn neg(x: (Vec<Interval>, Decoration)) -> (Vec<Interval>, Decoration) {
        (x.0.iter().map(|&x| -x).collect(), x.1)
    }

    macro_rules! test {
        ($f:expr, $x:expr, $expected:expr) => {
            test1($f, $x, $expected);
        };

        ($(@$af:ident)* $f:expr, @even $(@$ax:ident)* $x:expr, $expected:expr) => {
            test!($(@$af)* $f, $(@$ax)* $x, $expected);
            test!($(@$af)* $f, $(@$ax)* -$x, $expected);
        };

        ($(@$af:ident)* $f:expr, @odd $(@$ax:ident)* $x:expr, $expected:expr) => {
            test!($(@$af)* $f, $(@$ax)* $x, $expected);
            test!($(@$af)* $f, $(@$ax)* -$x, neg($expected));
        };

        ($f:expr, $x:expr, $y:expr, $expected:expr) => {
            test2($f, $x, $y, $expected);
        };

        (@commut $(@$af:ident)* $f:expr, $(@$ax:ident)* $x:expr, $(@$ay:ident)* $y:expr, $expected:expr) => {
            test!($(@$af)* $f, $(@$ax)* $x, $(@$ay)* $y, $expected);
            test!($(@$af)* $f, $(@$ax)* $y, $(@$ay)* $x, $expected);
        };

        ($(@$af:ident)* $f:expr, @even $(@$ax:ident)* $x:expr, $(@$ay:ident)* $y:expr, $expected:expr) => {
            test!($(@$af)* $f, $(@$ax)* $x, $(@$ay)* $y, $expected);
            test!($(@$af)* $f, $(@$ax)* -$x, $(@$ay)* $y, $expected);
        };

        ($(@$af:ident)* $f:expr, @odd $(@$ax:ident)* $x:expr, $(@$ay:ident)* $y:expr, $expected:expr) => {
            test!($(@$af)* $f, $(@$ax)* $x, $(@$ay)* $y, $expected);
            test!($(@$af)* $f, $(@$ax)* -$x, $(@$ay)* $y, neg($expected));
        };

        ($(@$af:ident)* $f:expr, $(@$ax:ident)* $x:expr, @even $(@$ay:ident)* $y:expr, $expected:expr) => {
            test!($(@$af)* $f, $(@$ax)* $x, $(@$ay)* $y, $expected);
            test!($(@$af)* $f, $(@$ax)* $x, $(@$ay)* -$y, $expected);
        };

        ($(@$af:ident)* $f:expr, $(@$ax:ident)* $x:expr, @odd $(@$ay:ident)* $y:expr, $expected:expr) => {
            test!($(@$af)* $f, $(@$ax)* $x, $(@$ay)* $y, $expected);
            test!($(@$af)* $f, $(@$ax)* $x, $(@$ay)* -$y, neg($expected));
        };
    }

    #[test]
    fn atan2() {
        fn f(x: TupperIntervalSet, y: TupperIntervalSet) -> TupperIntervalSet {
            x.atan2(&y, None)
        }

        let y = i!(0.0);
        test!(f, y, i!(0.0), (vec![], Trv));
        test!(f, y, i!(-1.0, -0.5), (vec![Interval::PI], Dac));
        test!(f, y, i!(-1.0, 0.0), (vec![Interval::PI], Trv));
        test!(f, y, i!(0.5, 1.0), (vec![i!(0.0)], Com));
        test!(f, y, i!(0.0, 1.0), (vec![i!(0.0)], Trv));
        test!(f, y, i!(-1.0, 1.0), (vec![i!(0.0), Interval::PI], Trv));

        let y = i!(-1.0, -0.5);
        test!(f, y, i!(0.0), (vec![-Interval::FRAC_PI_2], Com));

        let y = i!(-1.0, 0.0);
        test!(
            f,
            y,
            i!(-1.0),
            (
                vec![
                    -Interval::PI.convex_hull(i!(3.0) * Interval::FRAC_PI_4),
                    Interval::PI,
                ],
                Def,
            )
        );
        test!(f, y, i!(0.0), (vec![-Interval::FRAC_PI_2], Trv));

        let y = i!(0.5, 1.0);
        test!(f, y, i!(0.0), (vec![Interval::FRAC_PI_2], Com));

        let y = i!(0.0, 1.0);
        test!(
            f,
            y,
            i!(-1.0),
            (
                vec![Interval::PI.convex_hull(i!(3.0) * Interval::FRAC_PI_4)],
                Dac,
            )
        );
        test!(f, y, i!(0.0), (vec![Interval::FRAC_PI_2], Trv));

        let y = i!(-1.0, 1.0);
        test!(
            f,
            y,
            i!(-1.0),
            (
                vec![
                    -Interval::PI.convex_hull(i!(3.0) * Interval::FRAC_PI_4),
                    Interval::PI.convex_hull(i!(3.0) * Interval::FRAC_PI_4),
                ],
                Def,
            )
        );
        test!(
            f,
            y,
            i!(0.0),
            (vec![-Interval::FRAC_PI_2, Interval::FRAC_PI_2], Trv)
        );
        test!(
            f,
            y,
            i!(1.0),
            (
                vec![(-Interval::FRAC_PI_4).convex_hull(Interval::FRAC_PI_4)],
                Com,
            )
        );
        test!(
            f,
            y,
            i!(-1.0, 0.0),
            (
                vec![
                    -Interval::FRAC_PI_2.convex_hull(Interval::PI),
                    Interval::FRAC_PI_2.convex_hull(Interval::PI),
                ],
                Trv,
            )
        );
        test!(
            f,
            y,
            i!(0.0, 1.0),
            (
                vec![(-Interval::FRAC_PI_2).convex_hull(Interval::FRAC_PI_2)],
                Trv,
            )
        );
        test!(
            f,
            y,
            i!(-1.0, 1.0),
            (vec![(-Interval::PI).convex_hull(Interval::PI)], Trv)
        );
    }

    #[test]
    fn ceil() {
        fn f(x: TupperIntervalSet) -> TupperIntervalSet {
            x.ceil(None)
        }

        test!(f, i!(-1.5), (vec![i!(-1.0)], Com));
        test!(f, i!(-1.0), (vec![i!(-1.0)], Dac));
        test!(f, i!(-0.5), (vec![i!(0.0)], Com));
        test!(f, i!(0.0), (vec![i!(0.0)], Dac));
        test!(f, i!(0.5), (vec![i!(1.0)], Com));
        test!(f, i!(1.0), (vec![i!(1.0)], Dac));
        test!(f, i!(1.5), (vec![i!(2.0)], Com));

        test!(f, i!(-0.5, 0.5), (vec![i!(0.0), i!(1.0)], Def));
    }

    #[test]
    fn digamma() {
        fn f(x: TupperIntervalSet) -> TupperIntervalSet {
            x.digamma(None)
        }

        test!(f, i!(-3.0), (vec![], Trv));
        test!(f, i!(-2.0), (vec![], Trv));
        test!(f, i!(-1.0), (vec![], Trv));
        test!(f, i!(0.0), (vec![], Trv));
        test!(
            f,
            i!(1.0),
            (vec![i!(-0.5772156649015329, -0.5772156649015328)], Com)
        );

        test!(
            f,
            i!(-2.5, -1.5),
            (
                vec![
                    i!(-f64::INFINITY, 0.7031566406452432),
                    i!(1.103156640645243, f64::INFINITY),
                ],
                Trv,
            )
        );
        test!(
            f,
            i!(-1.5, -0.5),
            (
                vec![
                    i!(-f64::INFINITY, 3.648997397857653e-2),
                    i!(0.7031566406452431, f64::INFINITY),
                ],
                Trv,
            )
        );
        test!(
            f,
            i!(-0.5, 0.5),
            (
                vec![
                    i!(-f64::INFINITY, -1.9635100260214233),
                    i!(3.648997397857652e-2, f64::INFINITY),
                ],
                Trv,
            )
        );
    }

    #[test]
    fn div() {
        fn f(x: TupperIntervalSet, y: TupperIntervalSet) -> TupperIntervalSet {
            x.div(&y, None)
        }

        // x / 0
        let y = i!(0.0);
        test!(f, i!(-1.0, 1.0), y, (vec![], Trv));

        // x / 2
        let y = i!(2.0);
        test!(f, i!(0.0), y, (vec![i!(0.0)], Com));
        test!(f, @odd i!(1.0), @odd y, (vec![i!(0.5)], Com));
        test!(f, @odd i!(0.0, 1.0), @odd y, (vec![i!(0.0, 0.5)], Com));
        test!(f, i!(-1.0, 1.0), y, (vec![i!(-0.5, 0.5)], Com));

        // x / [0, 2]
        let y = i!(0.0, 2.0);
        test!(f, i!(0.0), y, (vec![i!(0.0)], Trv));
        test!(f, @odd i!(1.0), @odd y, (vec![i!(0.5, f64::INFINITY)], Trv));
        test!(f, @odd i!(0.0, 1.0), @odd y, (vec![i!(0.0, f64::INFINITY)], Trv));
        test!(
            f,
            i!(-1.0, 1.0),
            y,
            (vec![i!(-f64::INFINITY, f64::INFINITY)], Trv)
        );

        // x / [-2, 2]
        let y = i!(-2.0, 2.0);
        test!(f, i!(0.0), y, (vec![i!(0.0)], Trv));
        test!(
            f,
            @odd i!(1.0),
            @odd y,
            (vec![i!(-f64::INFINITY, -0.5), i!(0.5, f64::INFINITY)], Trv)
        );
        test!(
            f,
            @odd i!(0.0, 1.0),
            @odd y,
            (vec![i!(-f64::INFINITY, f64::INFINITY)], Trv)
        );
        test!(
            f,
            i!(-1.0, 1.0),
            y,
            (vec![i!(-f64::INFINITY, f64::INFINITY)], Trv)
        );
    }

    #[test]
    fn floor() {
        fn f(x: TupperIntervalSet) -> TupperIntervalSet {
            x.floor(None)
        }

        test!(f, i!(-1.5), (vec![i!(-2.0)], Com));
        test!(f, i!(-1.0), (vec![i!(-1.0)], Dac));
        test!(f, i!(-0.5), (vec![i!(-1.0)], Com));
        test!(f, i!(0.0), (vec![i!(0.0)], Dac));
        test!(f, i!(0.5), (vec![i!(0.0)], Com));
        test!(f, i!(1.0), (vec![i!(1.0)], Dac));
        test!(f, i!(1.5), (vec![i!(1.0)], Com));

        test!(f, i!(-0.5, 0.5), (vec![i!(-1.0), i!(0.0)], Def));
    }

    #[test]
    fn gamma() {
        fn f(x: TupperIntervalSet) -> TupperIntervalSet {
            x.gamma(None)
        }

        test!(f, i!(-3.0), (vec![], Trv));
        test!(f, i!(-2.0), (vec![], Trv));
        test!(f, i!(-1.0), (vec![], Trv));
        test!(f, i!(0.0), (vec![], Trv));
        test!(f, i!(1.0), (vec![i!(1.0)], Com));
        test!(f, i!(2.0), (vec![i!(1.0)], Com));
        test!(f, i!(3.0), (vec![i!(2.0)], Com));
        test!(
            f,
            i!(30.0),
            (
                vec![interval!("[8.841761993739701954543616e30]").unwrap()],
                Com,
            )
        );

        let x = TupperIntervalSet::from(dec_interval!("[-2.0000000000000000001]").unwrap());
        assert!(f(x).iter().all(|x| x.x.sup() < 0.0));
        let x = TupperIntervalSet::from(dec_interval!("[-1.9999999999999999999]").unwrap());
        assert!(f(x).iter().all(|x| x.x.inf() > 0.0));
        let x = TupperIntervalSet::from(dec_interval!("[-1.0000000000000000001]").unwrap());
        assert!(f(x).iter().all(|x| x.x.inf() > 0.0));
        let x = TupperIntervalSet::from(dec_interval!("[-0.99999999999999999999]").unwrap());
        assert!(f(x).iter().all(|x| x.x.sup() < 0.0));
        let x = TupperIntervalSet::from(dec_interval!("[-1e-500]").unwrap());
        assert!(f(x).iter().all(|x| x.x.sup() < 0.0));
        let x = TupperIntervalSet::from(dec_interval!("[1e-500]").unwrap());
        assert!(f(x).iter().all(|x| x.x.inf() > 0.0));
    }

    #[test]
    fn gcd() {
        fn f(x: TupperIntervalSet, y: TupperIntervalSet) -> TupperIntervalSet {
            x.gcd(&y, None)
        }

        test!(f, i!(0.0), i!(0.0), (vec![i!(0.0)], Dac));
        test!(@commut f, i!(0.0), @even i!(5.0), (vec![i!(5.0)], Dac));
        test!(@commut f, @even i!(7.5), @even i!(10.5), (vec![i!(1.5)], Dac));
        test!(@commut f, @even i!(15.0), @even i!(17.0), (vec![i!(1.0)], Dac));
        test!(@commut f, @even i!(15.0), @even i!(21.0), (vec![i!(3.0)], Dac));
        test!(@commut f, @even i!(15.0), @even i!(30.0), (vec![i!(15.0)], Dac));
        test!(
            @commut f,
            @even i!(1348500621.0),
            @even i!(18272779829.0),
            (vec![i!(150991.0)], Dac)
        );
        test!(
            @commut f,
            @even i!(4.0, 6.0),
            @even i!(5.0, 6.0),
            (vec![i!(0.0, 2.0), i!(4.0, 6.0)], Trv)
        );
    }

    #[test]
    fn lcm() {
        fn f(x: TupperIntervalSet, y: TupperIntervalSet) -> TupperIntervalSet {
            x.lcm(&y, None)
        }

        test!(f, i!(0.0), i!(0.0), (vec![i!(0.0)], Dac));
        test!(@commut f, i!(0.0), @even i!(5.0), (vec![i!(0.0)], Dac));
        test!(@commut f, @even i!(1.5), @even i!(2.5), (vec![i!(7.5)], Dac));
        test!(@commut f, @even i!(3.0), @even i!(5.0), (vec![i!(15.0)], Dac));
    }

    #[test]
    fn one() {
        fn f(x: TupperIntervalSet) -> TupperIntervalSet {
            x.one()
        }

        test!(f, i!(-1.0, 1.0), (vec![i!(1.0)], Com));
    }

    #[test]
    fn pow() {
        fn f(x: TupperIntervalSet, y: TupperIntervalSet) -> TupperIntervalSet {
            x.pow(&y, None)
        }

        // x^-3
        let y = i!(-3.0);
        test!(f, i!(-1.0), y, (vec![i!(-1.0)], Dac));
        test!(f, i!(0.0), y, (vec![], Trv));
        test!(f, i!(1.0), y, (vec![i!(1.0)], Com));
        test!(f, i!(-1.0, 0.0), y, (vec![i!(-f64::INFINITY, -1.0)], Trv));
        test!(f, i!(0.0, 1.0), y, (vec![i!(1.0, f64::INFINITY)], Trv));
        test!(
            f,
            i!(-1.0, 1.0),
            y,
            (vec![i!(-f64::INFINITY, -1.0), i!(1.0, f64::INFINITY)], Trv)
        );
        test!(
            f,
            i!(-3.0, -2.0),
            y,
            (vec![interval!("[-1/8, -1/27]").unwrap()], Dac)
        );
        test!(
            f,
            i!(2.0, 3.0),
            y,
            (vec![interval!("[1/27, 1/8]").unwrap()], Com)
        );

        // x^-2
        let y = i!(-2.0);
        test!(f, i!(-1.0), y, (vec![i!(1.0)], Dac));
        test!(f, i!(0.0), y, (vec![], Trv));
        test!(f, i!(1.0), y, (vec![i!(1.0)], Com));
        test!(f, i!(-1.0, 0.0), y, (vec![i!(1.0, f64::INFINITY)], Trv));
        test!(f, i!(0.0, 1.0), y, (vec![i!(1.0, f64::INFINITY)], Trv));
        test!(f, i!(-1.0, 1.0), y, (vec![i!(1.0, f64::INFINITY)], Trv));
        test!(
            f,
            i!(-3.0, -2.0),
            y,
            (vec![interval!("[1/9, 1/4]").unwrap()], Dac)
        );
        test!(
            f,
            i!(2.0, 3.0),
            y,
            (vec![interval!("[1/9, 1/4]").unwrap()], Com)
        );

        // x^(-1/2)
        let y = i!(-0.5);
        test!(f, i!(-1.0), y, (vec![], Trv));
        test!(f, i!(0.0), y, (vec![], Trv));
        test!(f, i!(1.0), y, (vec![i!(1.0)], Com));
        test!(f, i!(-1.0, 0.0), y, (vec![], Trv));
        test!(f, i!(0.0, 1.0), y, (vec![i!(1.0, f64::INFINITY)], Trv));
        test!(f, i!(-1.0, 1.0), y, (vec![i!(1.0, f64::INFINITY)], Trv));
        test!(
            f,
            i!(4.0, 9.0),
            y,
            (vec![interval!("[1/3, 1/2]").unwrap()], Com)
        );

        // x^0
        let y = i!(0.0);
        test!(f, i!(-1.0), y, (vec![i!(1.0)], Dac));
        test!(f, i!(0.0), y, (vec![i!(1.0)], Dac));
        test!(f, i!(1.0), y, (vec![i!(1.0)], Com));
        test!(f, i!(-1.0, 0.0), y, (vec![i!(1.0)], Dac));
        test!(f, i!(0.0, 1.0), y, (vec![i!(1.0)], Dac));
        test!(f, i!(-1.0, 1.0), y, (vec![i!(1.0)], Dac));
        test!(f, i!(-3.0, -2.0), y, (vec![i!(1.0)], Dac));
        test!(f, i!(2.0, 3.0), y, (vec![i!(1.0)], Com));

        // x^(1/2)
        let y = i!(0.5);
        test!(f, i!(-1.0), y, (vec![], Trv));
        test!(f, i!(0.0), y, (vec![i!(0.0)], Com));
        test!(f, i!(1.0), y, (vec![i!(1.0)], Com));
        test!(f, i!(-1.0, 0.0), y, (vec![i!(0.0)], Trv));
        test!(f, i!(0.0, 1.0), y, (vec![i!(0.0, 1.0)], Com));
        test!(f, i!(-1.0, 1.0), y, (vec![i!(0.0, 1.0)], Trv));
        test!(f, i!(4.0, 9.0), y, (vec![i!(2.0, 3.0)], Com));

        // x^2
        let y = i!(2.0);
        test!(f, i!(-1.0), y, (vec![i!(1.0)], Dac));
        test!(f, i!(0.0), y, (vec![i!(0.0)], Com));
        test!(f, i!(1.0), y, (vec![i!(1.0)], Com));
        test!(f, i!(-1.0, 0.0), y, (vec![i!(0.0, 1.0)], Dac));
        test!(f, i!(0.0, 1.0), y, (vec![i!(0.0, 1.0)], Com));
        test!(f, i!(-1.0, 1.0), y, (vec![i!(0.0, 1.0)], Dac));
        test!(f, i!(-3.0, -2.0), y, (vec![i!(4.0, 9.0)], Dac));
        test!(f, i!(2.0, 3.0), y, (vec![i!(4.0, 9.0)], Com));

        // x^3
        let y = i!(3.0);
        test!(f, i!(-1.0), y, (vec![i!(-1.0)], Dac));
        test!(f, i!(0.0), y, (vec![i!(0.0)], Com));
        test!(f, i!(1.0), y, (vec![i!(1.0)], Com));
        test!(f, i!(-1.0, 0.0), y, (vec![i!(-1.0, 0.0)], Dac));
        test!(f, i!(0.0, 1.0), y, (vec![i!(0.0, 1.0)], Com));
        test!(f, i!(-1.0, 1.0), y, (vec![i!(-1.0, 1.0)], Dac));
        test!(f, i!(-3.0, -2.0), y, (vec![i!(-27.0, -8.0)], Dac));
        test!(f, i!(2.0, 3.0), y, (vec![i!(8.0, 27.0)], Com));

        // x^e (or any inexact positive number)
        let y = Interval::E;
        test!(f, i!(-1.0), y, (vec![i!(-1.0), i!(1.0)], Trv));
        test!(f, i!(0.0), y, (vec![i!(0.0)], Com));
        test!(f, i!(1.0), y, (vec![i!(1.0)], Com));
        test!(f, i!(-1.0, 0.0), y, (vec![i!(-1.0, 1.0)], Trv));
        test!(f, i!(0.0, 1.0), y, (vec![i!(0.0, 1.0)], Com));
        test!(f, i!(-1.0, 1.0), y, (vec![i!(-1.0, 1.0)], Trv));

        // x^-e (or any inexact negative number)
        let y = -Interval::E;
        test!(f, i!(-1.0), y, (vec![i!(-1.0), i!(1.0)], Trv));
        test!(f, i!(0.0), y, (vec![], Trv));
        test!(f, i!(1.0), y, (vec![i!(1.0)], Com));
        test!(
            f,
            i!(-1.0, 0.0),
            y,
            (vec![i!(-f64::INFINITY, -1.0), i!(1.0, f64::INFINITY)], Trv)
        );
        test!(f, i!(0.0, 1.0), y, (vec![i!(1.0, f64::INFINITY)], Trv));
        test!(
            f,
            i!(-1.0, 1.0),
            y,
            (vec![i!(-f64::INFINITY, -1.0), i!(1.0, f64::INFINITY)], Trv)
        );

        // 0^y
        let x = i!(0.0);
        test!(f, x, i!(-1.0), (vec![], Trv));
        test!(f, x, i!(0.0), (vec![i!(1.0)], Dac));
        test!(f, x, i!(1.0), (vec![i!(0.0)], Com));
        test!(f, x, i!(-1.0, 0.0), (vec![i!(1.0)], Trv));
        test!(f, x, i!(0.0, 1.0), (vec![i!(0.0), i!(1.0)], Def));
        test!(f, x, i!(-1.0, 1.0), (vec![i!(0.0), i!(1.0)], Trv));

        // Others
        let x = i!(0.0, 1.0);
        test!(f, x, i!(-1.0, 0.0), (vec![i!(1.0, f64::INFINITY)], Trv));
        test!(f, x, i!(0.0, 1.0), (vec![i!(0.0, 1.0)], Def));
        test!(f, x, i!(-1.0, 1.0), (vec![i!(0.0, f64::INFINITY)], Trv));
    }

    #[test]
    fn pown() {
        fn pown(x: TupperIntervalSet, n: i32) -> TupperIntervalSet {
            x.pown(n, None)
        }

        // x^-2
        let f = |x| pown(x, -2);
        test!(f, i!(0.0), (vec![], Trv));
        test!(f, @even i!(1.0), (vec![i!(1.0)], Com));
        test!(f, @even i!(0.0, 1.0), (vec![i!(1.0, f64::INFINITY)], Trv));
        test!(f, i!(-1.0, 1.0), (vec![i!(1.0, f64::INFINITY)], Trv));
        test!(
            f,
            @even i!(2.0, 3.0),
            (vec![interval!("[1/9, 1/4]").unwrap()], Com)
        );

        // x^-1
        let f = |x| pown(x, -1);
        test!(f, i!(0.0), (vec![], Trv));
        test!(f, @odd i!(1.0), (vec![i!(1.0)], Com));
        test!(f, @odd i!(0.0, 1.0), (vec![i!(1.0, f64::INFINITY)], Trv));
        test!(
            f,
            i!(-1.0, 1.0),
            (vec![i!(-f64::INFINITY, -1.0), i!(1.0, f64::INFINITY)], Trv)
        );
        test!(
            f,
            @odd i!(2.0, 3.0),
            (vec![interval!("[1/3, 1/2]").unwrap()], Com)
        );

        // x^0
        let f = |x| pown(x, 0);
        test!(f, i!(-1.0, 1.0), (vec![i!(1.0)], Com));
        test!(f, @even i!(2.0, 3.0), (vec![i!(1.0, 1.0)], Com));

        // x^2
        let f = |x| pown(x, 2);
        test!(f, i!(-1.0, 1.0), (vec![i!(0.0, 1.0)], Com));
        test!(f, @even i!(2.0, 3.0), (vec![i!(4.0, 9.0)], Com));

        // x^3
        let f = |x| pown(x, 3);
        test!(f, i!(-1.0, 1.0), (vec![i!(-1.0, 1.0)], Com));
        test!(f, @odd i!(2.0, 3.0), (vec![i!(8.0, 27.0)], Com));
    }

    #[test]
    fn recip() {
        fn f(x: TupperIntervalSet) -> TupperIntervalSet {
            x.recip(None)
        }

        test!(f, i!(0.0), (vec![], Trv));
        test!(f, @odd i!(1.0), (vec![i!(1.0)], Com));
        test!(f, @odd i!(0.0, 1.0), (vec![i!(1.0, f64::INFINITY)], Trv));
        test!(
            f,
            i!(-1.0, 1.0),
            (vec![i!(-f64::INFINITY, -1.0), i!(1.0, f64::INFINITY)], Trv)
        );
        test!(
            f,
            @odd i!(2.0, 3.0),
            (vec![interval!("[1/3, 1/2]").unwrap()], Com)
        );
    }

    #[test]
    fn rem_euclid() {
        fn f(x: TupperIntervalSet, y: TupperIntervalSet) -> TupperIntervalSet {
            x.rem_euclid(&y, None)
        }

        let y = i!(0.0);
        test!(f, i!(-1.0, 1.0), y, (vec![], Trv));

        let y = i!(3.0);
        test!(f, i!(-1.0), @even y, (vec![i!(2.0)], Com));
        test!(f, i!(0.0), @even y, (vec![i!(0.0)], Dac));
        test!(f, i!(1.0), @even y, (vec![i!(1.0)], Com));
        test!(f, i!(-1.0, 1.0), @even y, (vec![i!(0.0, 1.0), i!(2.0, 3.0)], Def));

        let y = i!(1.5, 2.5);
        // NOTE: This is not the tightest enclosure, which is {[0.0, 0.5], [1.0, 2.0]}.
        test!(f, i!(-2.0), @even y, (vec![i!(0.0, 0.5), i!(1.0, 2.5)], Def));
        test!(f, i!(2.0), @even y, (vec![i!(0.0, 0.5), i!(2.0)], Def));

        let y = i!(-3.0, 3.0);
        test!(f, i!(0.0), y, (vec![i!(0.0)], Trv));
        test!(f, i!(-3.0, -3.0), y, (vec![i!(0.0, 3.0)], Trv));
    }

    #[test]
    fn rootn() {
        fn rootn(x: TupperIntervalSet, n: u32) -> TupperIntervalSet {
            x.rootn(n)
        }

        // x^1/0
        let f = |x| rootn(x, 0);
        test!(f, i!(-1.0, 1.0), (vec![], Trv));

        // x^1/2
        let f = |x| rootn(x, 2);
        test!(f, i!(-1.0), (vec![], Trv));
        test!(f, i!(0.0), (vec![i!(0.0)], Com));
        test!(f, i!(1.0), (vec![i!(1.0)], Com));
        test!(f, i!(-1.0, 0.0), (vec![i!(0.0)], Trv));
        test!(f, i!(0.0, 1.0), (vec![i!(0.0, 1.0)], Com));
        test!(f, i!(-1.0, 1.0), (vec![i!(0.0, 1.0)], Trv));
        test!(f, i!(4.0, 9.0), (vec![i!(2.0, 3.0)], Com));

        // x^1/3
        let f = |x| rootn(x, 3);
        test!(f, i!(-1.0, 1.0), (vec![i!(-1.0, 1.0)], Com));
        test!(f, @odd i!(8.0, 27.0), (vec![i!(2.0, 3.0)], Com));
    }

    #[test]
    fn undef_at_0() {
        fn f(x: TupperIntervalSet) -> TupperIntervalSet {
            x.undef_at_0()
        }

        test!(f, i!(0.0), (vec![], Trv));
        test!(f, @odd i!(1.0), (vec![i!(1.0)], Com));
        test!(f, @odd i!(0.0, 1.0), (vec![i!(0.0, 1.0)], Trv));
        test!(f, i!(-1.0, 1.0), (vec![i!(-1.0, 1.0)], Trv));
    }
}
