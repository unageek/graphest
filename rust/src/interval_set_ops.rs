use crate::{
    interval_set::{
        Branch, BranchMap, DecSignSet, SignSet, Site, TupperInterval, TupperIntervalSet,
    },
    Ternary,
};
use gmp_mpfr_sys::mpfr;
use inari::{
    const_dec_interval, const_interval, dec_interval, interval, DecInterval, Decoration, Interval,
};
use itertools::Itertools;
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
        impl $Op for &TupperIntervalSet {
            type Output = TupperIntervalSet;

            fn $op(self, rhs: &TupperIntervalSet) -> Self::Output {
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

const I_ZERO: Interval = const_interval!(0.0, 0.0);
const I_ONE: Interval = const_interval!(1.0, 1.0);
const DI_ZERO: DecInterval = const_dec_interval!(0.0, 0.0);
const DI_ONE: DecInterval = const_dec_interval!(1.0, 1.0);

/// Returns the parity of the function f(x) = x^y.
///
/// Precondition: `y` is finite.
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

fn ternary_to_intervals(t: Ternary) -> (DecInterval, Option<DecInterval>) {
    match t {
        Ternary::False => (DI_ZERO, None),
        Ternary::True => (DI_ONE, None),
        _ => (
            DecInterval::set_dec(I_ZERO, Decoration::Def),
            Some(DecInterval::set_dec(I_ONE, Decoration::Def)),
        ),
    }
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

    #[cfg(not(feature = "arb"))]
    pub fn atan2(&self, rhs: &Self, site: Option<Site>) -> Self {
        self.atan2_impl(rhs, site)
    }

    impl_op_cut!(atan2_impl(y, x), {
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
                DecInterval::set_dec(I_ZERO, dec),
                Some(DecInterval::set_dec(Interval::PI, dec)),
            )
        } else if a < 0.0 && b <= 0.0 && c < 0.0 && d >= 0.0 {
            let dec = if b == 0.0 {
                Decoration::Trv
            } else {
                Decoration::Def.min(x.decoration()).min(y.decoration())
            };
            // y < 0 (thus z < 0) part.
            let z0 = if c == f64::NEG_INFINITY {
                interval!(-Interval::PI.sup(), -Interval::FRAC_PI_2.inf()).unwrap()
            } else {
                let x0 = interval!(b, b).unwrap();
                let y0 = interval!(c, c).unwrap();
                interval!(-Interval::PI.sup(), y0.atan2(x0).sup()).unwrap()
            };
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

    pub fn boole_eq_zero(&self, site: Option<Site>) -> Self {
        if self.is_empty() {
            Self::from(DI_ZERO)
        } else {
            let mut rs = self.boole_eq_zero_nonempty(site);
            rs.normalize_boole();
            rs
        }
    }

    impl_op_cut!(boole_eq_zero_nonempty(x), {
        ternary_to_intervals(Ternary::from((
            x == DI_ZERO && x.decoration() >= Decoration::Def,
            x.contains(0.0),
        )))
    });

    pub fn boole_le_zero(&self, site: Option<Site>) -> Self {
        if self.is_empty() {
            Self::from(DI_ZERO)
        } else {
            let mut rs = self.boole_le_zero_nonempty(site);
            rs.normalize_boole();
            rs
        }
    }

    impl_op_cut!(boole_le_zero_nonempty(x), {
        ternary_to_intervals(Ternary::from((
            x.sup() <= 0.0 && x.decoration() >= Decoration::Def,
            x.inf() <= 0.0,
        )))
    });

    pub fn boole_lt_zero(&self, site: Option<Site>) -> Self {
        if self.is_empty() {
            Self::from(DI_ZERO)
        } else {
            let mut rs = self.boole_lt_zero_nonempty(site);
            rs.normalize_boole();
            rs
        }
    }

    impl_op_cut!(boole_lt_zero_nonempty(x), {
        ternary_to_intervals(Ternary::from((
            x.sup() < 0.0 && x.decoration() >= Decoration::Def,
            x.inf() < 0.0,
        )))
    });

    fn normalize_boole(&mut self) {
        let has_zero = self.iter().any(|&x| x.x == I_ZERO);
        let has_one = self.iter().any(|&x| x.x == I_ONE);
        match (has_zero, has_one) {
            (true, false) => *self = TupperIntervalSet::from(DI_ZERO),
            (false, true) => *self = TupperIntervalSet::from(DI_ONE),
            _ => (),
        };
    }

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
    pub fn gamma(&self, site: Option<Site>) -> Self {
        self.gamma_impl(site)
    }

    pub fn gamma_impl(&self, site: Option<Site>) -> Self {
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
                    let one = Self::from(DI_ONE);
                    let pi = Self::from(DecInterval::PI);
                    let mut xs = Self::new();
                    xs.insert(*x);
                    let mut sin = (&pi * &xs).sin();
                    // `a.floor() + 1.0` can be inexact when the first condition is not met.
                    if x.x.wid() <= 1.0 && b <= a.floor() + 1.0 {
                        let zero = Self::from(DI_ZERO);
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
    // Therefore, for any intervals X and Y, gcd[X, Y] ⊆ gcd(X, Y).  ∎
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
    // Therefore, gcd(X, Y) = Z_i = Z_{k-1}.  ∎
    pub fn gcd(&self, rhs: &Self, site: Option<Site>) -> Self {
        let mut rs = Self::new();
        // {gcd(x, y) ∣ x ∈ X, y ∈ Y}
        //   = {gcd(x, y) ∣ x ∈ |X|, y ∈ |Y|}
        //   = {gcd(max(x, y), min(x, y)) ∣ x ∈ |X|, y ∈ |Y|}
        //   ⊆ {gcd(x, y) ∣ x ∈ max(|X|, |Y|), y ∈ min(|X|, |Y|)}.
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
                        let mut rem = xs.modulo(ys, None);
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

    pub fn if_then_else(&self, t: &Self, f: &Self) -> Self {
        assert!(self.decoration() >= Decoration::Def);
        let mut rs = Self::new();
        for cond in self {
            let xs = if cond.x == I_ZERO {
                f
            } else if cond.x == I_ONE {
                t
            } else {
                panic!();
            };

            if xs.is_empty() {
                rs.insert(TupperInterval::from(DecInterval::EMPTY));
            } else {
                for x in xs {
                    if let Some(g) = cond.g.union(x.g) {
                        rs.insert(TupperInterval::new(
                            DecInterval::set_dec(x.x, cond.d.min(x.d)),
                            g,
                        ))
                    }
                }
            }
        }
        rs.normalize(false);
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
    //   (3):  X = {0} ∧ Y ∩ ℚ∖{0} ≠ ∅,
    //   (4):  X ∩ ℚ∖{0} ≠ ∅ ∧ Y = {0},
    //   (5):  X ∩ ℚ∖{0} ≠ ∅ ∧ Y ∩ ℚ∖{0} ≠ ∅.
    //
    // Suppose (1).  Then, lcm[X, Y] = ∅ ⊆ lcm(X, Y).
    // Suppose (2).  Then, lcm[X, Y] = lcm(X, Y) = {0}.
    // Suppose (3).  As Y ≠ {0}, lcm(X, Y) = |X Y| / gcd(X, Y).
    // Therefore, from 0 ∈ |X Y| and ∃y ∈ Y ∩ ℚ∖{0} : |y| ∈ gcd(X, Y), 0 ∈ lcm(X, Y).
    // Therefore, lcm[X, Y] = {0} ⊆ lcm(X, Y).
    // Suppose (4).  In the same manner, lcm[X, Y] ⊆ lcm(X, Y).
    // Suppose (5).  Let x ∈ X ∩ ℚ∖{0}, y ∈ Y ∩ ℚ∖{0} ≠ ∅.
    // Then, |x y| / gcd(x, y) ∈ lcm(X, Y) = |X Y| / gcd(X, Y).
    // Therefore, lcm[X, Y] ⊆ lcm(X, Y).
    //
    // Hence, the result.  ∎
    pub fn lcm(&self, rhs: &Self, site: Option<Site>) -> Self {
        let mut rs = TupperIntervalSet::new();
        for x in self {
            for y in rhs {
                if let Some(g) = x.g.union(y.g) {
                    if x.x == I_ZERO && y.x == I_ZERO {
                        let dec = Decoration::Dac.min(x.d).min(y.d);
                        rs.insert(TupperInterval::new(DecInterval::set_dec(I_ZERO, dec), g));
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
        self.ln().div(&rhs.ln(), site)
    }

    impl_op!(max(x, y), x.max(y));

    impl_op!(min(x, y), x.min(y));

    // f(x, y) = x - y ⌊x / y⌋.
    impl_op_cut!(modulo(x, y), {
        let q = (x / y).floor();
        let qa = q.inf();
        let qb = q.sup();
        let range = y.interval().unwrap().convex_hull(I_ZERO);
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

    #[cfg(not(feature = "arb"))]
    pub fn pow(&self, rhs: &Self, site: Option<Site>) -> Self {
        self.pow_impl(rhs, site)
    }

    // For any integer n,
    //
    //         | x × ⋯ × x (n copies)  if n > 0,
    //   x^n = | 1                     if n = 0 ∧ x ≠ 0,
    //         | 1 / x^-n              if n < 0,
    //
    // and for any non-integer y,
    //
    //   x^y = | 1             if x = 0 ∧ y > 0,
    //         | exp(y ln(x))  otherwise.
    //
    // 0^0 is left undefined.
    impl_op_cut!(pow_impl(x, y), {
        let a = x.inf();
        let b = x.sup();
        let c = y.inf();
        let d = y.sup();
        if y.is_singleton() {
            Self::pow_singleton(x, y)
        } else if a < 0.0 {
            // a < 0.
            let dec = Decoration::Trv;

            let nc = c.ceil();
            let nd = d.floor();
            let x_neg = x.min(DI_ZERO);
            let (z0, z1) = if nc > nd {
                (DecInterval::EMPTY, None)
            } else if nc == nd {
                let y = dec_interval!(nc, nc).unwrap();
                let z = Self::pow_singleton(x_neg, y).0;
                let z = DecInterval::set_dec(z.interval().unwrap(), dec);
                (z, None)
            } else if nd - nc == 1.0 {
                let y0 = dec_interval!(nc, nc).unwrap();
                let y1 = dec_interval!(nd, nd).unwrap();
                let z0 = Self::pow_singleton(x_neg, y0).0;
                let z1 = Self::pow_singleton(x_neg, y1).0;
                let z0 = DecInterval::set_dec(z0.interval().unwrap(), dec);
                let z1 = DecInterval::set_dec(z1.interval().unwrap(), dec);
                (z0, Some(z1))
            } else {
                //                    |x^y|
                //     ⋮    |    ⋮    ↑
                // y = 1    |   -1    |
                //     0    |    0   -+- 1
                //    -1    |    1    |
                //     ⋮    |    ⋮    |
                // ---------+---------+- 0 -→
                //         -1         0     x

                let z0 = if nd == f64::INFINITY {
                    DecInterval::ENTIRE
                } else {
                    let x = x.intersection(const_dec_interval!(f64::NEG_INFINITY, -1.0));
                    if x.is_empty() {
                        DecInterval::EMPTY
                    } else {
                        let y = dec_interval!(nd, nd).unwrap();
                        let z = Self::pow_singleton(x, y).0;
                        z.convex_hull(-z)
                    }
                };
                let z1 = if nc == f64::NEG_INFINITY {
                    DecInterval::ENTIRE
                } else {
                    let x = x.intersection(const_dec_interval!(-1.0, 0.0));
                    if x.is_empty() {
                        DecInterval::EMPTY
                    } else {
                        let y = dec_interval!(nc, nc).unwrap();
                        let z = Self::pow_singleton(x, y).0;
                        z.convex_hull(-z)
                    }
                };
                let z = z0.convex_hull(z1);
                (z, None)
            };

            if b < 0.0 {
                (z0, z1)
            } else {
                let z = z0
                    .convex_hull(z1.unwrap_or(DecInterval::EMPTY))
                    .convex_hull(x.pow(y));
                (z, None)
            }
        } else {
            // a ≥ 0.
            (x.pow(y), None)
        }
    });

    #[cfg(not(feature = "arb"))]
    pub fn pow_rational(&self, rhs: &Self, site: Option<Site>) -> Self {
        self.pow_rational_impl(rhs, site)
    }

    // For any rational number y = p/q where p and q (> 0) are coprime integers,
    //
    //   x^y = surd(x, q)^p.
    //
    // surd(x, q) is the real-valued qth root of x for odd q,
    // and is the principal qth root of x ≥ 0 for even q. Therefore, for x < 0,
    //
    //         | (-x)^y     if y = (even)/(odd)
    //         |            (x^y is an even function of x),
    //   x^y = | -(-x)^y    if y = (odd)/(odd)
    //         |            (x^y is an odd function of x),
    //         | undefined  otherwise (y = (odd)/(even) or irrational).
    //
    // And for any irrational number y,
    //
    //   x^y = | 1             if x = 0 ∧ y > 0,
    //         | exp(y ln(x))  otherwise.
    //
    // 0^0 is left undefined.
    impl_op_cut!(pow_rational_impl(x, y), {
        let a = x.inf();
        if y.is_singleton() {
            Self::pow_singleton(x, y)
        } else if a < 0.0 {
            // a < 0.
            let dec = Decoration::Trv;

            let x = x.interval().unwrap();
            let y = y.interval().unwrap();

            // x^y < 0 part, which comes from
            //   x < 0, y = (odd)/(odd) (x^y is an odd function of x).
            let x0 = x.min(I_ZERO);
            let z0 = DecInterval::set_dec(-(-x0).pow(y), dec);

            // x^y ≥ 0 part, which comes from
            //   x ≥ 0;
            //   x < 0, y = (even)/(odd) (x^y is an even function of x).
            let z1 = DecInterval::set_dec(x.abs().pow(y), dec);

            (z0, Some(z1))
        } else {
            // a ≥ 0.
            (x.pow(y), None)
        }
    });

    #[allow(clippy::many_single_char_names)]
    fn pow_singleton(x: DecInterval, y: DecInterval) -> (DecInterval, Option<DecInterval>) {
        assert!(y.is_singleton());
        let a = x.inf();
        let c = y.inf();
        match exponentiation_parity(c) {
            Parity::None => (x.pow(y), None),
            Parity::Even => {
                let dec = if x.contains(0.0) && c <= 0.0 {
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
                let z = DecInterval::set_dec(x.abs().pow(y), dec);
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
    }

    //       | x × ⋯ × x (n copies)  if n > 0,
    // x^n = | 1                     if n = 0 ∧ x ≠ 0,
    //       | 1 / x^-n              if n < 0.
    impl_op_cut!(pown(x, n: i32), {
        let a = x.inf();
        let b = x.sup();
        if n == 0 {
            if x.contains(0.0) {
                if x.is_singleton() {
                    // x = {0}.
                    (DecInterval::EMPTY, None)
                } else {
                    (DecInterval::set_dec(I_ONE, Decoration::Trv), None)
                }
            } else {
                (DecInterval::set_dec(I_ONE, x.decoration()), None)
            }
        } else if n < 0 && n % 2 == -1 && a < 0.0 && b > 0.0 {
            let x0 = DecInterval::set_dec(interval!(a, 0.0).unwrap(), x.decoration());
            let x1 = DecInterval::set_dec(interval!(0.0, b).unwrap(), x.decoration());
            (x0.powi(n), Some(x1.powi(n)))
        } else {
            (x.powi(n), None)
        }
    });

    pub fn ranked_max(xs: Vec<&Self>, n: &Self, site: Option<Site>) -> Self {
        Self::ranked_min_max(xs, n, site, true)
    }

    pub fn ranked_min(xs: Vec<&Self>, n: &Self, site: Option<Site>) -> Self {
        Self::ranked_min_max(xs, n, site, false)
    }

    fn ranked_min_max(xs: Vec<&Self>, n: &Self, site: Option<Site>, max: bool) -> Self {
        assert!(!xs.is_empty());
        let mut rs = Self::new();
        let mut infs = vec![];
        let mut sups = vec![];
        for n in n {
            // `n` uses 1-based indexing.
            let n0 = n.x - I_ONE;
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

    // f(x, y) = | Re(sgn(x + i y))  if x > 0,
    //           | 0                 otherwise,
    //         = | x / sqrt(x^2 + y^2)  if x > 0,
    //           | 0                    otherwise.
    impl_op_cut!(re_sign_nonnegative(x, y), {
        let x = x.max(DI_ZERO);
        let y = y.abs();
        // 0 ≤ a ∧ 0 ≤ c.
        let a = x.inf();
        let b = x.sup();
        let c = y.inf();
        let d = y.sup();
        let dec = if a == 0.0 && b > 0.0 && c == 0.0 {
            Decoration::Def
        } else {
            Decoration::Dac
        }
        .min(x.decoration())
        .min(y.decoration());
        if d == 0.0 {
            if b == 0.0 {
                let z = DecInterval::set_dec(I_ZERO, dec);
                (z, None)
            } else if a == 0.0 {
                let z0 = DecInterval::set_dec(I_ZERO, dec);
                let z1 = DecInterval::set_dec(I_ONE, dec);
                (z0, Some(z1))
            } else {
                let z = DecInterval::set_dec(I_ONE, dec);
                (z, None)
            }
        } else {
            let inf = if d == f64::INFINITY {
                0.0
            } else {
                let a = interval!(a, a).unwrap();
                let d = interval!(d, d).unwrap();
                (a / (a.sqr() + d.sqr()).sqrt()).inf()
            };
            let sup = if b == 0.0 {
                0.0
            } else if b == f64::INFINITY {
                1.0
            } else {
                let b = interval!(b, b).unwrap();
                let c = interval!(c, c).unwrap();
                (b / (b.sqr() + c.sqr()).sqrt()).sup()
            };
            let z = DecInterval::set_dec(interval!(inf, sup).unwrap(), dec);
            (z, None)
        }
    });

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

    #[cfg(not(feature = "arb"))]
    pub fn tan(&self, site: Option<Site>) -> Self {
        self.tan_impl(site)
    }

    impl_op_cut!(tan_impl(x), {
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
            let y0 = interval!(tan_rd(a), f64::INFINITY).unwrap();
            let y1 = interval!(f64::NEG_INFINITY, tan_ru(b)).unwrap();
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
    requires_arb!(inverse_erf(x));
    requires_arb!(inverse_erfc(x));
    requires_arb!(lambert_w(k, x));
    requires_arb!(li(x));
    requires_arb!(shi(x));
    requires_arb!(si(x));
    requires_arb!(zeta(x));
}

impl TupperIntervalSet {
    pub fn eq_zero(&self) -> DecSignSet {
        let mut ss = SignSet::empty();
        for x in self {
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
        }

        DecSignSet(ss, self.decoration())
    }
}

macro_rules! impl_rel_op {
    ($op:ident, $map_neg:expr, $map_zero:expr, $map_pos:expr, $map_undef:expr) => {
        pub fn $op(&self) -> DecSignSet {
            fn bool_to_sign(b: bool) -> SignSet {
                if b {
                    SignSet::ZERO
                } else {
                    SignSet::POS
                }
            }

            let mut ss = SignSet::empty();
            for x in self {
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
            }
            if self.decoration() == Decoration::Trv {
                ss |= bool_to_sign($map_undef);
            }

            let d = match ss {
                SignSet::ZERO => Decoration::Dac,
                _ => Decoration::Def,
            };

            DecSignSet(ss, d)
        }
    };
}

impl TupperIntervalSet {
    impl_rel_op!(le_zero, true, true, false, false);
    impl_rel_op!(lt_zero, true, false, false, false);
}

// Copied from https://github.com/unageek/inari/blob/b398df0609ea96c28574f8b1acdabbc87cb7cf78/src/elementary.rs
macro_rules! mpfr_fn {
    ($mpfr_f:ident, $f_rd:ident, $f_ru:ident) => {
        fn $f_rd(x: f64) -> f64 {
            mpfr_fn!($mpfr_f(x, RNDD))
        }

        fn $f_ru(x: f64) -> f64 {
            mpfr_fn!($mpfr_f(x, RNDU))
        }
    };

    ($mpfr_f:ident($x:ident, $rnd:ident)) => {{
        let mut x = Float::with_val(f64::MANTISSA_DIGITS, $x);
        let rnd = mpfr::rnd_t::$rnd;
        unsafe {
            mpfr::$mpfr_f(x.as_raw_mut(), x.as_raw(), rnd);
            mpfr::get_d(x.as_raw(), rnd)
        }
    }};
}

macro_rules! mpfr_fn_ui {
    ($mpfr_f:ident, $f_rd:ident, $f_ru:ident) => {
        fn $f_rd(x: f64, y: u32) -> f64 {
            mpfr_fn_ui!($mpfr_f(x, y, RNDD))
        }

        fn $f_ru(x: f64, y: u32) -> f64 {
            mpfr_fn_ui!($mpfr_f(x, y, RNDU))
        }
    };

    ($mpfr_f:ident($x:ident, $y:ident, $rnd:ident)) => {{
        let mut x = Float::with_val(f64::MANTISSA_DIGITS, $x);
        let rnd = mpfr::rnd_t::$rnd;
        unsafe {
            mpfr::$mpfr_f(x.as_raw_mut(), x.as_raw(), $y.into(), rnd);
            mpfr::get_d(x.as_raw(), rnd)
        }
    }};
}

mpfr_fn!(digamma, digamma_rd, digamma_ru);
mpfr_fn!(erf, erf_rd, erf_ru);
mpfr_fn!(erfc, erfc_rd, erfc_ru);
mpfr_fn!(gamma, gamma_rd, gamma_ru);
mpfr_fn_ui!(rootn_ui, rootn_rd, rootn_ru);
mpfr_fn!(tan, tan_rd, tan_ru);

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
    use inari::Decoration::*;
    use Ternary::*;

    macro_rules! i {
        ($a:expr) => {
            const_interval!($a, $a)
        };

        ($a:expr, $b:expr) => {
            const_interval!($a, $b)
        };
    }

    fn test1<F>(f: F, x: Interval, expected: (Vec<Interval>, Decoration), loose: bool)
    where
        F: Fn(&TupperIntervalSet) -> TupperIntervalSet,
    {
        let empty = TupperIntervalSet::new();
        let decs = [Com, Dac, Def, Trv];
        for &dx in &decs {
            let x = TupperIntervalSet::from(DecInterval::set_dec(x, dx));
            let y = {
                let mut y = f(&x);
                y.normalize(true);
                y
            };

            let y_exp = {
                let dy = expected.1.min(x.decoration());
                let mut y = expected
                    .0
                    .iter()
                    .map(|&y| TupperInterval::from(DecInterval::set_dec(y, dy)))
                    .chain((if loose { &y } else { &empty }).iter().copied())
                    .collect::<TupperIntervalSet>();
                y.normalize(true);
                y
            };

            assert_eq!(y, y_exp);
        }
    }

    fn test2<F>(f: F, x: Interval, y: Interval, expected: (Vec<Interval>, Decoration), loose: bool)
    where
        F: Fn(&TupperIntervalSet, &TupperIntervalSet) -> TupperIntervalSet,
    {
        let empty = TupperIntervalSet::new();
        let decs = [Com, Dac, Def, Trv];
        for &dx in &decs {
            let x = TupperIntervalSet::from(DecInterval::set_dec(x, dx));
            for &dy in &decs {
                let y = TupperIntervalSet::from(DecInterval::set_dec(y, dy));
                let z = {
                    let mut z = f(&x, &y);
                    z.normalize(true);
                    z
                };

                let z_exp = {
                    let dz = expected.1.min(x.decoration()).min(y.decoration());
                    let mut z = expected
                        .0
                        .iter()
                        .map(|&z| TupperInterval::from(DecInterval::set_dec(z, dz)))
                        .chain((if loose { &z } else { &empty }).iter().copied())
                        .collect::<TupperIntervalSet>();
                    z.normalize(true);
                    z
                };

                assert_eq!(z, z_exp);
            }
        }
    }

    fn neg(x: (Vec<Interval>, Decoration)) -> (Vec<Interval>, Decoration) {
        (x.0.iter().map(|&x| -x).collect(), x.1)
    }

    macro_rules! test {
        ($f:expr, $x:expr, $expected:expr) => {
            test1($f, $x, $expected, false);
        };

        ($f:expr, $x:expr, @loose_for_arb $expected:expr) => {
            test1($f, $x, $expected, cfg!(feature = "arb"));
        };

        ($(@$af:ident)* $f:expr, @even $(@$ax:ident)* $x:expr, $(@$aexp:ident)* $expected:expr) => {
            test!($(@$af)* $f, $(@$ax)* $x, $(@$aexp)* $expected);
            test!($(@$af)* $f, $(@$ax)* -$x, $(@$aexp)* $expected);
        };

        ($(@$af:ident)* $f:expr, @odd $(@$ax:ident)* $x:expr, $(@$aexp:ident)* $expected:expr) => {
            test!($(@$af)* $f, $(@$ax)* $x, $(@$aexp)* $expected);
            test!($(@$af)* $f, $(@$ax)* -$x, $(@$aexp)* neg($expected));
        };

        ($f:expr, $x:expr, $y:expr, $expected:expr) => {
            test2($f, $x, $y, $expected, false);
        };

        ($f:expr, $x:expr, $y:expr, @loose_for_arb $expected:expr) => {
            test2($f, $x, $y, $expected, cfg!(feature = "arb"));
        };

        (@commut $(@$af:ident)* $f:expr, $(@$ax:ident)* $x:expr, $(@$ay:ident)* $y:expr, $(@$aexp:ident)* $expected:expr) => {
            test!($(@$af)* $f, $(@$ax)* $x, $(@$ay)* $y, $(@$aexp)* $expected);
            test!($(@$af)* $f, $(@$ax)* $y, $(@$ay)* $x, $(@$aexp)* $expected);
        };

        ($(@$af:ident)* $f:expr, @even $(@$ax:ident)* $x:expr, $(@$ay:ident)* $y:expr, $(@$aexp:ident)* $expected:expr) => {
            test!($(@$af)* $f, $(@$ax)* $x, $(@$ay)* $y, $(@$aexp)* $expected);
            test!($(@$af)* $f, $(@$ax)* -$x, $(@$ay)* $y, $(@$aexp)* $expected);
        };

        ($(@$af:ident)* $f:expr, @odd $(@$ax:ident)* $x:expr, $(@$ay:ident)* $y:expr, $(@$aexp:ident)* $expected:expr) => {
            test!($(@$af)* $f, $(@$ax)* $x, $(@$ay)* $y, $(@$aexp)* $expected);
            test!($(@$af)* $f, $(@$ax)* -$x, $(@$ay)* $y, $(@$aexp)* neg($expected));
        };

        ($(@$af:ident)* $f:expr, $(@$ax:ident)* $x:expr, @even $(@$ay:ident)* $y:expr, $(@$aexp:ident)* $expected:expr) => {
            test!($(@$af)* $f, $(@$ax)* $x, $(@$ay)* $y, $(@$aexp)* $expected);
            test!($(@$af)* $f, $(@$ax)* $x, $(@$ay)* -$y, $(@$aexp)* $expected);
        };

        ($(@$af:ident)* $f:expr, $(@$ax:ident)* $x:expr, @odd $(@$ay:ident)* $y:expr, $(@$aexp:ident)* $expected:expr) => {
            test!($(@$af)* $f, $(@$ax)* $x, $(@$ay)* $y, $(@$aexp)* $expected);
            test!($(@$af)* $f, $(@$ax)* $x, $(@$ay)* -$y, $(@$aexp)* neg($expected));
        };
    }

    fn test_boole_op<F>(f: F, x: Interval, t_expected: Ternary)
    where
        F: Fn(&TupperIntervalSet) -> TupperIntervalSet,
    {
        let decs = [Com, Dac, Def, Trv];
        for &dx in &decs {
            let x = TupperIntervalSet::from(DecInterval::set_dec(x, dx));
            let y = {
                let mut y = f(&x);
                y.normalize(true);
                y
            };

            let y_exp = {
                let mut y = match (dx, t_expected) {
                    (Trv, True) | (_, Uncertain) => vec![
                        TupperInterval::from(DecInterval::set_dec(I_ZERO, Def)),
                        TupperInterval::from(DecInterval::set_dec(I_ONE, Def)),
                    ]
                    .into_iter()
                    .collect(),
                    (_, False) => TupperIntervalSet::from(DI_ZERO),
                    (_, True) => TupperIntervalSet::from(DI_ONE),
                };
                y.normalize(true);
                y
            };

            assert_eq!(y, y_exp);
        }
    }

    #[test]
    fn atan2() {
        fn f(x: &TupperIntervalSet, y: &TupperIntervalSet) -> TupperIntervalSet {
            x.atan2(y, None)
        }

        let y = i!(0.0);
        test!(f, y, i!(0.0), (vec![], Trv));
        test!(f, y, i!(-1.0, -0.5), (vec![Interval::PI], Dac));
        test!(f, y, i!(-1.0, 0.0), (vec![Interval::PI], Trv));
        test!(f, y, i!(0.5, 1.0), (vec![i!(0.0)], Com));
        test!(f, y, i!(0.0, 1.0), (vec![i!(0.0)], Trv));
        test!(f, y, i!(-1.0, 1.0), (vec![i!(0.0), Interval::PI], Trv));

        let y = i!(-1.0, -0.5);
        test!(f, y, i!(0.0), @loose_for_arb (vec![-Interval::FRAC_PI_2], Com));

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
        test!(f, y, i!(0.0), @loose_for_arb (vec![Interval::FRAC_PI_2], Com));

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
            @loose_for_arb (
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
    fn boole_eq_zero() {
        fn f(x: &TupperIntervalSet) -> TupperIntervalSet {
            x.boole_eq_zero(None)
        }

        test_boole_op(f, Interval::EMPTY, False);
        test_boole_op(f, i!(-1.0), False);
        test_boole_op(f, i!(0.0), True);
        test_boole_op(f, i!(1.0), False);
        test_boole_op(f, i!(-1.0, 0.0), Uncertain);
        test_boole_op(f, i!(0.0, 1.0), Uncertain);
        test_boole_op(f, i!(-1.0, 1.0), Uncertain);
    }

    #[test]
    fn boole_le_zero() {
        fn f(x: &TupperIntervalSet) -> TupperIntervalSet {
            x.boole_le_zero(None)
        }

        test_boole_op(f, Interval::EMPTY, False);
        test_boole_op(f, i!(-1.0), True);
        test_boole_op(f, i!(0.0), True);
        test_boole_op(f, i!(1.0), False);
        test_boole_op(f, i!(-1.0, 0.0), True);
        test_boole_op(f, i!(0.0, 1.0), Uncertain);
        test_boole_op(f, i!(-1.0, 1.0), Uncertain);
    }

    #[test]
    fn boole_lt_zero() {
        fn f(x: &TupperIntervalSet) -> TupperIntervalSet {
            x.boole_lt_zero(None)
        }

        test_boole_op(f, Interval::EMPTY, False);
        test_boole_op(f, i!(-1.0), True);
        test_boole_op(f, i!(0.0), False);
        test_boole_op(f, i!(1.0), False);
        test_boole_op(f, i!(-1.0, 0.0), Uncertain);
        test_boole_op(f, i!(0.0, 1.0), False);
        test_boole_op(f, i!(-1.0, 1.0), Uncertain);
    }

    #[test]
    fn ceil() {
        fn f(x: &TupperIntervalSet) -> TupperIntervalSet {
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
        fn f(x: &TupperIntervalSet) -> TupperIntervalSet {
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
        fn f(x: &TupperIntervalSet, y: &TupperIntervalSet) -> TupperIntervalSet {
            x.div(y, None)
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
        fn f(x: &TupperIntervalSet) -> TupperIntervalSet {
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
        fn f(x: &TupperIntervalSet) -> TupperIntervalSet {
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
            @loose_for_arb (
                vec![interval!("[8.841761993739701954543616e30]").unwrap()],
                Com,
            )
        );

        let x = TupperIntervalSet::from(dec_interval!("[-2.0000000000000000001]").unwrap());
        assert!(f(&x).iter().all(|x| x.x.sup() < 0.0));
        let x = TupperIntervalSet::from(dec_interval!("[-1.9999999999999999999]").unwrap());
        assert!(f(&x).iter().all(|x| x.x.inf() > 0.0));
        let x = TupperIntervalSet::from(dec_interval!("[-1.0000000000000000001]").unwrap());
        assert!(f(&x).iter().all(|x| x.x.inf() > 0.0));
        let x = TupperIntervalSet::from(dec_interval!("[-0.99999999999999999999]").unwrap());
        assert!(f(&x).iter().all(|x| x.x.sup() < 0.0));
        let x = TupperIntervalSet::from(dec_interval!("[-1e-500]").unwrap());
        assert!(f(&x).iter().all(|x| x.x.sup() < 0.0));
        let x = TupperIntervalSet::from(dec_interval!("[1e-500]").unwrap());
        assert!(f(&x).iter().all(|x| x.x.inf() > 0.0));
    }

    #[test]
    fn gcd() {
        fn f(x: &TupperIntervalSet, y: &TupperIntervalSet) -> TupperIntervalSet {
            x.gcd(y, None)
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
    fn if_then_else() {
        fn test(x: Interval, t: Interval, f: Interval, expected: (Vec<Interval>, Decoration)) {
            let decs = [Com, Dac, Def, Trv];
            let x = TupperIntervalSet::from(DecInterval::new(x));
            let cond = x.boole_eq_zero(None);
            for &dt in &decs {
                let t = TupperIntervalSet::from(DecInterval::set_dec(t, dt));
                for &df in &decs {
                    let f = TupperIntervalSet::from(DecInterval::set_dec(f, df));
                    let y = {
                        let mut y = cond.if_then_else(&t, &f);
                        y.normalize(true);
                        y
                    };

                    let y_exp = {
                        let dy = expected.1.min(if cond.to_f64() == Some(0.0) {
                            f.decoration()
                        } else if cond.to_f64() == Some(1.0) {
                            t.decoration()
                        } else {
                            t.decoration().min(f.decoration())
                        });
                        let mut y = expected
                            .0
                            .iter()
                            .map(|&y| TupperInterval::from(DecInterval::set_dec(y, dy)))
                            .collect::<TupperIntervalSet>();
                        y.normalize(true);
                        y
                    };

                    assert_eq!(y, y_exp);
                }
            }
        }

        // True cases
        let x = i!(0.0);
        test(x, i!(2.0), i!(3.0), (vec![i!(2.0)], Com));
        test(x, Interval::EMPTY, i!(3.0), (vec![], Trv));
        test(x, i!(2.0), Interval::EMPTY, (vec![i!(2.0)], Com));

        // False cases
        let x = i!(1.0);
        test(x, i!(2.0), i!(3.0), (vec![i!(3.0)], Com));
        test(x, Interval::EMPTY, i!(3.0), (vec![i!(3.0)], Com));
        test(x, i!(2.0), Interval::EMPTY, (vec![], Trv));

        // Uncertain cases
        let x = i!(0.0, 1.0);
        test(x, i!(2.0), i!(3.0), (vec![i!(2.0), i!(3.0)], Def));
        test(x, Interval::EMPTY, i!(3.0), (vec![i!(3.0)], Trv));
        test(x, i!(2.0), Interval::EMPTY, (vec![i!(2.0)], Trv));
    }

    #[test]
    fn lcm() {
        fn f(x: &TupperIntervalSet, y: &TupperIntervalSet) -> TupperIntervalSet {
            x.lcm(y, None)
        }

        test!(f, i!(0.0), i!(0.0), (vec![i!(0.0)], Dac));
        test!(@commut f, i!(0.0), @even i!(5.0), (vec![i!(0.0)], Dac));
        test!(@commut f, @even i!(1.5), @even i!(2.5), (vec![i!(7.5)], Dac));
        test!(@commut f, @even i!(3.0), @even i!(5.0), (vec![i!(15.0)], Dac));
    }

    #[test]
    fn modulo() {
        fn f(x: &TupperIntervalSet, y: &TupperIntervalSet) -> TupperIntervalSet {
            x.modulo(y, None)
        }

        let y = i!(-3.0);
        test!(f, i!(-1.0), y, (vec![i!(-1.0)], Com));
        test!(f, i!(0.0), y, (vec![i!(0.0)], Dac));
        test!(f, i!(1.0), y, (vec![i!(-2.0)], Com));
        test!(
            f,
            i!(-1.0, 1.0),
            y,
            (vec![i!(-3.0, -2.0), i!(-1.0, 0.0)], Def)
        );

        let y = i!(0.0);
        test!(f, i!(-1.0, 1.0), y, (vec![], Trv));

        let y = i!(3.0);
        test!(f, i!(-1.0), y, (vec![i!(2.0)], Com));
        test!(f, i!(0.0), y, (vec![i!(0.0)], Dac));
        test!(f, i!(1.0), y, (vec![i!(1.0)], Com));
        test!(f, i!(-1.0, 1.0), y, (vec![i!(0.0, 1.0), i!(2.0, 3.0)], Def));

        let y = i!(-2.5, -1.5);
        test!(f, i!(-2.0), y, (vec![i!(-2.0), i!(-0.5, 0.0)], Def));
        // NOTE: This is not the tightest enclosure,
        // which is {2} mod ([-2.5, -2] ∪ (-2, -1.5]) = [-0.5, 0] ∪ (-2, -1].
        test!(f, i!(2.0), y, (vec![i!(-2.5, -1.0), i!(-0.5, 0.0)], Def));

        let y = i!(1.5, 2.5);
        // NOTE: This is not the tightest enclosure,
        // which is {-2} mod ([1.5, 2) ∪ [2, 2.5]) = [1, 2) ∪ [0, 0.5].
        test!(f, i!(-2.0), y, (vec![i!(0.0, 0.5), i!(1.0, 2.5)], Def));
        test!(f, i!(2.0), y, (vec![i!(0.0, 0.5), i!(2.0)], Def));

        let y = i!(-3.0, 3.0);
        test!(f, i!(0.0), y, (vec![i!(0.0)], Trv));
        test!(f, i!(-3.0, 3.0), y, (vec![i!(-3.0, 3.0)], Trv));
    }

    #[test]
    fn pow() {
        fn f(x: &TupperIntervalSet, y: &TupperIntervalSet) -> TupperIntervalSet {
            x.pow(y, None)
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
            @loose_for_arb (vec![interval!("[-1/8, -1/27]").unwrap()], Dac)
        );
        test!(
            f,
            i!(2.0, 3.0),
            y,
            @loose_for_arb (vec![interval!("[1/27, 1/8]").unwrap()], Com)
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
            @loose_for_arb (vec![interval!("[1/9, 1/4]").unwrap()], Dac)
        );
        test!(
            f,
            i!(2.0, 3.0),
            y,
            @loose_for_arb (vec![interval!("[1/9, 1/4]").unwrap()], Com)
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
            @loose_for_arb (vec![interval!("[1/3, 1/2]").unwrap()], Com)
        );

        // x^0
        let y = i!(0.0);
        test!(f, i!(-1.0), y, (vec![i!(1.0)], Dac));
        test!(f, i!(0.0), y, (vec![], Trv));
        test!(f, i!(1.0), y, (vec![i!(1.0)], Com));
        test!(f, i!(-1.0, 0.0), y, (vec![i!(1.0)], Trv));
        test!(f, i!(0.0, 1.0), y, (vec![i!(1.0)], Trv));
        test!(f, i!(-1.0, 1.0), y, (vec![i!(1.0)], Trv));
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
        test!(f, i!(4.0, 9.0), y, @loose_for_arb (vec![i!(2.0, 3.0)], Com));

        // x^2
        let y = i!(2.0);
        test!(f, i!(-1.0), y, (vec![i!(1.0)], Dac));
        test!(f, i!(0.0), y, (vec![i!(0.0)], Com));
        test!(f, i!(1.0), y, (vec![i!(1.0)], Com));
        test!(f, i!(-1.0, 0.0), y, (vec![i!(0.0, 1.0)], Dac));
        test!(f, i!(0.0, 1.0), y, (vec![i!(0.0, 1.0)], Com));
        test!(f, i!(-1.0, 1.0), y, (vec![i!(0.0, 1.0)], Dac));
        test!(f, i!(-3.0, -2.0), y, (vec![i!(4.0, 9.0)], Dac));
        test!(f, i!(2.0, 3.0), y, @loose_for_arb (vec![i!(4.0, 9.0)], Com));

        // x^3
        let y = i!(3.0);
        test!(f, i!(-1.0), y, (vec![i!(-1.0)], Dac));
        test!(f, i!(0.0), y, (vec![i!(0.0)], Com));
        test!(f, i!(1.0), y, (vec![i!(1.0)], Com));
        test!(f, i!(-1.0, 0.0), y, (vec![i!(-1.0, 0.0)], Dac));
        test!(f, i!(0.0, 1.0), y, (vec![i!(0.0, 1.0)], Com));
        test!(f, i!(-1.0, 1.0), y, (vec![i!(-1.0, 1.0)], Dac));
        test!(f, i!(-3.0, -2.0), y, (vec![i!(-27.0, -8.0)], Dac));
        test!(f, i!(2.0, 3.0), y, @loose_for_arb (vec![i!(8.0, 27.0)], Com));

        // x^e (or any inexact positive number)
        let y = Interval::E;
        test!(f, i!(-1.0), y, (vec![], Trv));
        test!(f, i!(0.0), y, (vec![i!(0.0)], Com));
        test!(f, i!(1.0), y, (vec![i!(1.0)], Com));
        test!(f, i!(-1.0, 0.0), y, (vec![i!(0.0)], Trv));
        test!(f, i!(0.0, 1.0), y, (vec![i!(0.0, 1.0)], Com));
        test!(f, i!(-1.0, 1.0), y, (vec![i!(0.0, 1.0)], Trv));

        // x^-e (or any inexact negative number)
        let y = -Interval::E;
        test!(f, i!(-1.0), y, (vec![], Trv));
        test!(f, i!(0.0), y, (vec![], Trv));
        test!(f, i!(1.0), y, (vec![i!(1.0)], Com));
        test!(f, i!(-1.0, 0.0), y, (vec![], Trv));
        test!(f, i!(0.0, 1.0), y, (vec![i!(1.0, f64::INFINITY)], Trv));
        test!(f, i!(-1.0, 1.0), y, (vec![i!(1.0, f64::INFINITY)], Trv));

        // 0^y
        let x = i!(0.0);
        test!(f, x, i!(-1.0), (vec![], Trv));
        test!(f, x, i!(0.0), (vec![], Trv));
        test!(f, x, i!(1.0), (vec![i!(0.0)], Com));
        test!(f, x, i!(-1.0, 0.0), (vec![], Trv));
        test!(f, x, i!(0.0, 1.0), (vec![i!(0.0)], Trv));
        test!(f, x, i!(-1.0, 1.0), (vec![i!(0.0)], Trv));

        // Others
        let x = i!(0.0, 1.0);
        test!(f, x, i!(-1.0, 0.0), (vec![i!(1.0, f64::INFINITY)], Trv));
        test!(f, x, i!(0.0, 1.0), (vec![i!(0.0, 1.0)], Trv));
        test!(f, x, i!(-1.0, 1.0), (vec![i!(0.0, f64::INFINITY)], Trv));
    }

    #[test]
    fn pow_rational() {
        fn f(x: &TupperIntervalSet, y: &TupperIntervalSet) -> TupperIntervalSet {
            x.pow_rational(y, None)
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
            @loose_for_arb (vec![interval!("[-1/8, -1/27]").unwrap()], Dac)
        );
        test!(
            f,
            i!(2.0, 3.0),
            y,
            @loose_for_arb (vec![interval!("[1/27, 1/8]").unwrap()], Com)
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
            @loose_for_arb (vec![interval!("[1/9, 1/4]").unwrap()], Dac)
        );
        test!(
            f,
            i!(2.0, 3.0),
            y,
            @loose_for_arb (vec![interval!("[1/9, 1/4]").unwrap()], Com)
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
            @loose_for_arb (vec![interval!("[1/3, 1/2]").unwrap()], Com)
        );

        // x^0
        let y = i!(0.0);
        test!(f, i!(-1.0), y, (vec![i!(1.0)], Dac));
        test!(f, i!(0.0), y, (vec![], Trv));
        test!(f, i!(1.0), y, (vec![i!(1.0)], Com));
        test!(f, i!(-1.0, 0.0), y, (vec![i!(1.0)], Trv));
        test!(f, i!(0.0, 1.0), y, (vec![i!(1.0)], Trv));
        test!(f, i!(-1.0, 1.0), y, (vec![i!(1.0)], Trv));
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
        test!(f, i!(4.0, 9.0), y, @loose_for_arb (vec![i!(2.0, 3.0)], Com));

        // x^2
        let y = i!(2.0);
        test!(f, i!(-1.0), y, (vec![i!(1.0)], Dac));
        test!(f, i!(0.0), y, (vec![i!(0.0)], Com));
        test!(f, i!(1.0), y, (vec![i!(1.0)], Com));
        test!(f, i!(-1.0, 0.0), y, (vec![i!(0.0, 1.0)], Dac));
        test!(f, i!(0.0, 1.0), y, (vec![i!(0.0, 1.0)], Com));
        test!(f, i!(-1.0, 1.0), y, (vec![i!(0.0, 1.0)], Dac));
        test!(f, i!(-3.0, -2.0), y, (vec![i!(4.0, 9.0)], Dac));
        test!(f, i!(2.0, 3.0), y, @loose_for_arb (vec![i!(4.0, 9.0)], Com));

        // x^3
        let y = i!(3.0);
        test!(f, i!(-1.0), y, (vec![i!(-1.0)], Dac));
        test!(f, i!(0.0), y, (vec![i!(0.0)], Com));
        test!(f, i!(1.0), y, (vec![i!(1.0)], Com));
        test!(f, i!(-1.0, 0.0), y, (vec![i!(-1.0, 0.0)], Dac));
        test!(f, i!(0.0, 1.0), y, (vec![i!(0.0, 1.0)], Com));
        test!(f, i!(-1.0, 1.0), y, (vec![i!(-1.0, 1.0)], Dac));
        test!(f, i!(-3.0, -2.0), y, @loose_for_arb (vec![i!(-27.0, -8.0)], Dac));
        test!(f, i!(2.0, 3.0), y, @loose_for_arb (vec![i!(8.0, 27.0)], Com));

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
        test!(f, x, i!(0.0), (vec![], Trv));
        test!(f, x, i!(1.0), (vec![i!(0.0)], Com));
        test!(f, x, i!(-1.0, 0.0), (vec![], Trv));
        test!(f, x, i!(0.0, 1.0), (vec![i!(0.0)], Trv));
        test!(f, x, i!(-1.0, 1.0), (vec![i!(0.0)], Trv));

        // Others
        let x = i!(0.0, 1.0);
        test!(f, x, i!(-1.0, 0.0), (vec![i!(1.0, f64::INFINITY)], Trv));
        test!(f, x, i!(0.0, 1.0), (vec![i!(0.0, 1.0)], Trv));
        test!(f, x, i!(-1.0, 1.0), (vec![i!(0.0, f64::INFINITY)], Trv));
    }

    #[test]
    fn pown() {
        fn pown(x: &TupperIntervalSet, n: i32) -> TupperIntervalSet {
            x.pown(n, None)
        }

        // x^-2
        let f = |x: &_| pown(x, -2);
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
        let f = |x: &_| pown(x, -1);
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
        let f = |x: &_| pown(x, 0);
        test!(f, i!(0.0), (vec![], Trv));
        test!(f, @even i!(1.0), (vec![i!(1.0)], Com));
        test!(f, @even i!(0.0, 1.0), (vec![i!(1.0)], Trv));
        test!(f, i!(-1.0, 1.0), (vec![i!(1.0)], Trv));
        test!(f, @even i!(2.0, 3.0), (vec![i!(1.0)], Com));

        // x^2
        let f = |x: &_| pown(x, 2);
        test!(f, i!(0.0), (vec![i!(0.0)], Com));
        test!(f, @even i!(1.0), (vec![i!(1.0)], Com));
        test!(f, @even i!(0.0, 1.0), (vec![i!(0.0, 1.0)], Com));
        test!(f, i!(-1.0, 1.0), (vec![i!(0.0, 1.0)], Com));
        test!(f, @even i!(2.0, 3.0), (vec![i!(4.0, 9.0)], Com));

        // x^3
        let f = |x: &_| pown(x, 3);
        test!(f, i!(0.0), (vec![i!(0.0)], Com));
        test!(f, @odd i!(1.0), (vec![i!(1.0)], Com));
        test!(f, @odd i!(0.0, 1.0), (vec![i!(0.0, 1.0)], Com));
        test!(f, i!(-1.0, 1.0), (vec![i!(-1.0, 1.0)], Com));
        test!(f, @odd i!(2.0, 3.0), (vec![i!(8.0, 27.0)], Com));
    }

    #[test]
    fn re_sign_nonnegative() {
        fn f(x: &TupperIntervalSet, y: &TupperIntervalSet) -> TupperIntervalSet {
            x.re_sign_nonnegative(y, None)
        }

        let y = i!(0.0);
        test!(f, i!(-1.0), y, (vec![i!(0.0)], Dac));
        test!(f, i!(0.0), y, (vec![i!(0.0)], Dac));
        test!(f, i!(1.0), y, (vec![i!(1.0)], Dac));
        test!(f, i!(-1.0, 0.0), y, (vec![i!(0.0)], Dac));
        test!(f, i!(0.0, 1.0), y, (vec![i!(0.0), i!(1.0)], Def));
        test!(f, i!(-1.0, 1.0), y, (vec![i!(0.0), i!(1.0)], Def));

        let y = i!(1.0);
        test!(f, i!(-1.0), @even y, (vec![i!(0.0)], Dac));
        test!(f, i!(0.0), @even y, (vec![i!(0.0)], Dac));
        test!(f, i!(1.0), @even y, (vec![i!(0.7071067811865475, 0.7071067811865477)], Dac));
        test!(f, i!(-1.0, 0.0), @even y, (vec![i!(0.0)], Dac));
        test!(f, i!(0.0, 1.0), @even y, (vec![i!(0.0, 0.7071067811865477)], Dac));
        test!(f, i!(-1.0, 1.0), @even y, (vec![i!(0.0, 0.7071067811865477)], Dac));

        let y = i!(0.0, 1.0);
        test!(f, i!(-1.0), @even y, (vec![i!(0.0)], Dac));
        test!(f, i!(0.0), @even y, (vec![i!(0.0)], Dac));
        test!(f, i!(1.0), @even y, (vec![i!(0.7071067811865475, 1.0)], Dac));
        test!(f, i!(-1.0, 0.0), @even y, (vec![i!(0.0)], Dac));
        test!(f, i!(0.0, 1.0), @even y, (vec![i!(0.0, 1.0)], Def));
        test!(f, i!(-1.0, 1.0), @even y, (vec![i!(0.0, 1.0)], Def));
    }

    #[test]
    fn recip() {
        fn f(x: &TupperIntervalSet) -> TupperIntervalSet {
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
    fn rootn() {
        fn rootn(x: &TupperIntervalSet, n: u32) -> TupperIntervalSet {
            x.rootn(n)
        }

        // x^1/0
        let f = |x: &_| rootn(x, 0);
        test!(f, i!(-1.0, 1.0), (vec![], Trv));

        // x^1/2
        let f = |x: &_| rootn(x, 2);
        test!(f, i!(-1.0), (vec![], Trv));
        test!(f, i!(0.0), (vec![i!(0.0)], Com));
        test!(f, i!(1.0), (vec![i!(1.0)], Com));
        test!(f, i!(-1.0, 0.0), (vec![i!(0.0)], Trv));
        test!(f, i!(0.0, 1.0), (vec![i!(0.0, 1.0)], Com));
        test!(f, i!(-1.0, 1.0), (vec![i!(0.0, 1.0)], Trv));
        test!(f, i!(4.0, 9.0), (vec![i!(2.0, 3.0)], Com));

        // x^1/3
        let f = |x: &_| rootn(x, 3);
        test!(f, i!(-1.0, 1.0), (vec![i!(-1.0, 1.0)], Com));
        test!(f, @odd i!(8.0, 27.0), (vec![i!(2.0, 3.0)], Com));
    }

    #[test]
    fn undef_at_0() {
        fn f(x: &TupperIntervalSet) -> TupperIntervalSet {
            x.undef_at_0()
        }

        test!(f, i!(0.0), (vec![], Trv));
        test!(f, @odd i!(1.0), (vec![i!(1.0)], Com));
        test!(f, @odd i!(0.0, 1.0), (vec![i!(0.0, 1.0)], Trv));
        test!(f, i!(-1.0, 1.0), (vec![i!(-1.0, 1.0)], Trv));
    }

    #[test]
    fn ops_sanity() {
        // Check that operations do not panic due to invalid construction of an interval.
        let xs = [
            TupperIntervalSet::from(const_dec_interval!(0.0, 0.0)),
            TupperIntervalSet::from(const_dec_interval!(0.0, 1.0)),
            TupperIntervalSet::from(const_dec_interval!(-1.0, 0.0)),
            TupperIntervalSet::from(const_dec_interval!(-1.0, 1.0)),
            TupperIntervalSet::from(const_dec_interval!(f64::NEG_INFINITY, 0.0)),
            TupperIntervalSet::from(const_dec_interval!(f64::NEG_INFINITY, 1.0)),
            TupperIntervalSet::from(const_dec_interval!(0.0, f64::INFINITY)),
            TupperIntervalSet::from(const_dec_interval!(1.0, f64::INFINITY)),
            TupperIntervalSet::from(DecInterval::ENTIRE),
        ];

        let fs = [
            |x: &_| -x,
            TupperIntervalSet::abs,
            TupperIntervalSet::acos,
            TupperIntervalSet::acosh,
            TupperIntervalSet::asin,
            TupperIntervalSet::asinh,
            TupperIntervalSet::atan,
            TupperIntervalSet::atanh,
            TupperIntervalSet::cos,
            TupperIntervalSet::cosh,
            TupperIntervalSet::erf,
            TupperIntervalSet::erfc,
            TupperIntervalSet::exp,
            TupperIntervalSet::ln,
            TupperIntervalSet::sin,
            TupperIntervalSet::sinc,
            TupperIntervalSet::sinh,
            TupperIntervalSet::sqr,
            TupperIntervalSet::sqrt,
            TupperIntervalSet::tanh,
            TupperIntervalSet::undef_at_0,
        ];
        for f in &fs {
            for x in &xs {
                f(x);
            }
        }

        let fs = [
            TupperIntervalSet::boole_eq_zero,
            TupperIntervalSet::boole_le_zero,
            TupperIntervalSet::boole_lt_zero,
            TupperIntervalSet::ceil,
            TupperIntervalSet::digamma,
            TupperIntervalSet::floor,
            TupperIntervalSet::gamma,
            TupperIntervalSet::recip,
            TupperIntervalSet::tan,
        ];
        for f in &fs {
            for x in &xs {
                f(x, None);
            }
        }

        let fs = [
            |x: &_, y: &_| x + y,
            |x: &_, y: &_| x - y,
            |x: &_, y: &_| x * y,
            TupperIntervalSet::max,
            TupperIntervalSet::min,
        ];
        for f in &fs {
            for x in &xs {
                for y in &xs {
                    f(x, y);
                }
            }
        }

        let fs = [
            TupperIntervalSet::atan2,
            TupperIntervalSet::div,
            TupperIntervalSet::gcd,
            TupperIntervalSet::lcm,
            TupperIntervalSet::log,
            TupperIntervalSet::modulo,
            TupperIntervalSet::pow,
            TupperIntervalSet::pow_rational,
            TupperIntervalSet::re_sign_nonnegative,
        ];
        for f in &fs {
            for x in &xs {
                for y in &xs {
                    f(x, y, None);
                }
            }
        }

        let fs = [TupperIntervalSet::mul_add];
        for f in &fs {
            for x in &xs {
                for y in &xs {
                    for z in &xs {
                        f(x, y, z);
                    }
                }
            }
        }

        let ns = [-3, -2, -1, 0, 1, 2, 3];
        for x in &xs {
            for n in &ns {
                x.pown(*n, None);
            }
        }

        let ns = [0, 1, 2, 3];
        for x in &xs {
            for n in &ns {
                x.rootn(*n);
            }
        }

        let fs = [TupperIntervalSet::ranked_max, TupperIntervalSet::ranked_min];
        let ns = [
            TupperIntervalSet::from(const_dec_interval!(1.0, 1.0)),
            TupperIntervalSet::from(const_dec_interval!(2.0, 2.0)),
            TupperIntervalSet::from(const_dec_interval!(3.0, 3.0)),
        ];
        for f in &fs {
            for x in &xs {
                for y in &xs {
                    for z in &xs {
                        for n in &ns {
                            f(vec![x, y, z], n, None);
                        }
                        for n in &xs {
                            f(vec![x, y, z], n, None);
                        }
                    }
                }
            }
        }
    }
}
