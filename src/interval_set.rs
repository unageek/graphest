#![allow(clippy::float_cmp)]

use crate::rel::{StaticRel, StaticRelKind};
use bitflags::*;
use core::ops::{Add, BitAnd, BitAndAssign, BitOr, BitOrAssign, Mul, Neg, Sub};
use hexf::*;
use inari::{
    const_dec_interval, const_interval, interval, DecoratedInterval, Decoration, Interval,
};
use smallvec::{smallvec, SmallVec};
use std::{
    convert::From,
    hash::{Hash, Hasher},
    mem::transmute,
    slice::Iter,
};

// Represents a partial function {0, ..., 31} -> {0, 1}, the domain of which is
// the set of branch cut sites and the codomain is the set of branch indices.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(C)]
struct IntervalBranch {
    // A bit field that keeps track of at which sites
    // branch cut has been performed during the derivation of the interval.
    cut: u32,
    // A bit field that records the chosen branch (0 or 1)
    // at each site, when the corresponding bit of `cut` is set.
    chosen: u32,
}

impl IntervalBranch {
    fn new() -> Self {
        Self { cut: 0, chosen: 0 }
    }

    fn inserted(self, site: u8, branch: u8) -> Self {
        assert!(site <= 31 && branch <= 1 && self.cut & 1 << site == 0);
        Self {
            cut: self.cut | 1 << site,
            chosen: self.chosen | (branch as u32) << site,
        }
    }

    fn union(self, rhs: Self) -> Option<Self> {
        // Tests if Graph(self) ∪ Graph(rhs) is a valid graph of a partial function.
        let mask = self.cut & rhs.cut;
        let valid = (self.chosen & mask) == (rhs.chosen & mask);

        if valid {
            Some(Self {
                cut: self.cut | rhs.cut,
                chosen: self.chosen | rhs.chosen,
            })
        } else {
            None
        }
    }
}

// Relationship between the decoration system and the properties of Tupper IA:
//  Decoration | Properties
// ------------+---------------------------
//  com, dac   | def: [T,T]
//             | cont: [T,T]
//  def        | def: [T,T]
//             | cont: [F,F], [F,T]
//  trv        | def: [F,F], [F,T]
//             | cont: [F,F], [F,T], [T,T]

#[repr(C)]
struct _DecoratedInterval {
    x: Interval,
    d: Decoration,
}

// We don't store `DecoratedInterval` directly as that would make
// the size of `TupperInterval` 48 bytes instead of 32 due to the alignment.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct TupperInterval {
    x: Interval,
    d: Decoration,
    g: IntervalBranch,
}

impl TupperInterval {
    /// Creates a new `TupperInterval` with the given `DecoratedInterval` and `IntervalBranch`.
    ///
    /// Panics if The interval is NaI.
    fn new(x: DecoratedInterval, g: IntervalBranch) -> Self {
        // nai is prohibited.
        assert!(!x.is_nai());
        let x = unsafe { transmute::<_, _DecoratedInterval>(x) };
        Self { x: x.x, d: x.d, g }
    }

    /// Returns the `DecoratedInterval` part of the `TupperInterval`.
    pub fn to_dec_interval(self) -> DecoratedInterval {
        unsafe {
            transmute(_DecoratedInterval {
                x: self.x,
                d: self.d,
            })
        }
    }
}

// NOTE: Hash, PartialEq and Eq look only the interval part
// as these are used solely to discriminate constants with
// the maximum decorations.

impl PartialEq for TupperInterval {
    fn eq(&self, rhs: &Self) -> bool {
        self.x == rhs.x
    }
}

impl Eq for TupperInterval {}

impl Hash for TupperInterval {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.x.inf().to_bits().hash(state);
        self.x.sup().to_bits().hash(state);
    }
}

type TupperIntervalVecBackingArray = [TupperInterval; 2];
type TupperIntervalVec = SmallVec<TupperIntervalVecBackingArray>;

// NOTE: Equality is order-sensitive.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct TupperIntervalSet(TupperIntervalVec);

impl TupperIntervalSet {
    /// Creates a new, empty `TupperIntervalSet`.
    pub fn empty() -> Self {
        Self(TupperIntervalVec::new())
    }

    pub fn iter(&self) -> Iter<'_, TupperInterval> {
        self.0.iter()
    }

    /// Returns the number of intervals in `self`.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Inserts an interval to `self`. If the interval is empty, `self` remains intact.
    fn insert(&mut self, x: TupperInterval) {
        if !x.x.is_empty() {
            self.0.push(x);
        }
    }

    /// Returns `true` if `self` is empty.
    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Expected to be called after modifying `self`. It is no-op for the moment.
    fn normalize(self) -> Self {
        // TODO: Merge overlapping intervals.
        self
    }
}

impl From<DecoratedInterval> for TupperIntervalSet {
    fn from(x: DecoratedInterval) -> Self {
        let mut xs = Self::empty();
        xs.insert(TupperInterval::new(x, IntervalBranch::new()));
        xs
    }
}

impl IntoIterator for TupperIntervalSet {
    type Item = TupperInterval;
    type IntoIter = smallvec::IntoIter<TupperIntervalVecBackingArray>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<'a> IntoIterator for &'a TupperIntervalSet {
    type Item = &'a TupperInterval;
    type IntoIter = Iter<'a, TupperInterval>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

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
        pub fn $op(&self, site: Option<u8>) -> Self {
            let mut rs = Self::empty();
            for x in self {
                let y = TupperInterval::new(x.to_dec_interval().$op(), x.g);
                let a = y.x.inf();
                let b = y.x.sup();
                if b - a == 1.0 {
                    rs.insert(TupperInterval::new(
                        DecoratedInterval::set_dec(interval!(a, a).unwrap(), y.d),
                        match site {
                            Some(site) => y.g.inserted(site, 0),
                            _ => y.g,
                        },
                    ));
                    rs.insert(TupperInterval::new(
                        DecoratedInterval::set_dec(interval!(b, b).unwrap(), y.d),
                        match site {
                            Some(site) => y.g.inserted(site, 1),
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Parity {
    None,
    Even,
    Odd,
}

impl TupperIntervalSet {
    pub fn atan2(&self, rhs: &Self, site: Option<u8>) -> Self {
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
                            DecoratedInterval::set_dec(-Interval::FRAC_PI_2, dec),
                            match site {
                                Some(site) => g.inserted(site, 0),
                                _ => g,
                            },
                        ));
                        rs.insert(TupperInterval::new(
                            DecoratedInterval::set_dec(Interval::FRAC_PI_2, dec),
                            match site {
                                Some(site) => g.inserted(site, 1),
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
                            DecoratedInterval::set_dec(
                                interval!(-Interval::PI.sup(), y0.atan2(x0).sup()).unwrap(),
                                dec,
                            ),
                            match site {
                                Some(site) => g.inserted(site, 0),
                                _ => g,
                            },
                        ));

                        // y ≥ 0 (thus z > 0) part.
                        let x1 = interval!(a, b).unwrap();
                        let y1 = interval!(0.0, d).unwrap();
                        rs.insert(TupperInterval::new(
                            DecoratedInterval::set_dec(y1.atan2(x1), dec),
                            match site {
                                Some(site) => g.inserted(site, 1),
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

    pub fn div(&self, rhs: &Self, site: Option<u8>) -> Self {
        let mut rs = Self::empty();
        for x in self {
            for y in rhs {
                if let Some(g) = x.g.union(y.g) {
                    let c = y.x.inf();
                    let d = y.x.sup();
                    if c < 0.0 && d > 0.0 {
                        let y0 = DecoratedInterval::set_dec(interval!(c, 0.0).unwrap(), y.d);
                        rs.insert(TupperInterval::new(
                            x.to_dec_interval() / y0,
                            match site {
                                Some(site) => g.inserted(site, 0),
                                _ => g,
                            },
                        ));
                        let y1 = DecoratedInterval::set_dec(interval!(0.0, d).unwrap(), y.d);
                        rs.insert(TupperInterval::new(
                            x.to_dec_interval() / y1,
                            match site {
                                Some(site) => g.inserted(site, 1),
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

    pub fn log(&self, rhs: &Self, site: Option<u8>) -> Self {
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

    // Returns the parity of the function f(x) = x^y.
    // Precondition: y is neither ±∞ nor NaN.
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
    //     x^(m / n) = surd(x, n)^m,
    //   where surd(x, n) is the real-valued nth root of x.
    //   Therefore, for x < 0,
    //           | (-x)^y     if y = even / odd
    //           |            (x^y is an even function of x),
    //     x^y = | -(-x)^y    if y = odd / odd
    //           |            (x^y is an odd function of x),
    //           | undefined  otherwise (y = odd / even or irrational).
    // - We define 0^0 = 1.
    // The original `Interval::pow` is not defined for x < 0 nor x = y = 0.
    pub fn pow(&self, rhs: &Self, site: Option<u8>) -> Self {
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
                                    DecoratedInterval::set_dec(z.convex_hull(one_or_empty), dec),
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
                                        DecoratedInterval::set_dec(zn, dec),
                                        match site {
                                            Some(site) => g.inserted(site, 0),
                                            _ => g,
                                        },
                                    ));
                                    rs.insert(TupperInterval::new(
                                        DecoratedInterval::set_dec(zp, dec),
                                        match site {
                                            Some(site) => g.inserted(site, 1),
                                            _ => g,
                                        },
                                    ));
                                } else {
                                    let z = zn.convex_hull(zp);
                                    rs.insert(TupperInterval::new(
                                        DecoratedInterval::set_dec(z, dec),
                                        g,
                                    ));
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
                            DecoratedInterval::set_dec(z.convex_hull(one_or_empty), dec),
                            match site {
                                Some(site) => g.inserted(site, 0),
                                _ => g,
                            },
                        ));

                        // x^y < 0 part from
                        //   x < 0 for those ys where f(x) = x^y is an odd function.
                        let x1 = x.x.min(const_interval!(0.0, 0.0));
                        let z = -(-x1).pow(y.x);
                        rs.insert(TupperInterval::new(
                            DecoratedInterval::set_dec(z, dec),
                            match site {
                                Some(site) => g.inserted(site, 1),
                                _ => g,
                            },
                        ));
                    } else {
                        // a ≥ 0.

                        // If a = b = 0 ∧ c ≤ 0 ≤ d, we need to add {1} to z manually.
                        // In that case, the decoration of z is already `Trv`.
                        let z = x.to_dec_interval().pow(y.to_dec_interval());
                        rs.insert(TupperInterval::new(
                            DecoratedInterval::set_dec(
                                z.interval_part().unwrap().convex_hull(one_or_empty),
                                z.decoration_part(),
                            ),
                            g,
                        ));
                    }
                }
            }
        }
        rs.normalize()
    }

    pub fn pown(&self, rhs: i32, site: Option<u8>) -> Self {
        let mut rs = Self::empty();
        for x in self {
            let a = x.x.inf();
            let b = x.x.sup();
            if rhs < 0 && rhs % 2 == 1 && a < 0.0 && b > 0.0 {
                let x0 = DecoratedInterval::set_dec(interval!(a, 0.0).unwrap(), x.d);
                rs.insert(TupperInterval::new(
                    x0.pown(rhs),
                    match site {
                        Some(site) => x.g.inserted(site, 0),
                        _ => x.g,
                    },
                ));
                let x1 = DecoratedInterval::set_dec(interval!(0.0, b).unwrap(), x.d);
                rs.insert(TupperInterval::new(
                    x1.pown(rhs),
                    match site {
                        Some(site) => x.g.inserted(site, 1),
                        _ => x.g,
                    },
                ));
            } else {
                rs.insert(TupperInterval::new(x.to_dec_interval().pown(rhs), x.g));
            }
        }
        rs.normalize()
    }

    pub fn recip(&self, site: Option<u8>) -> Self {
        let mut rs = Self::empty();
        for x in self {
            let a = x.x.inf();
            let b = x.x.sup();
            if a < 0.0 && b > 0.0 {
                let x0 = DecoratedInterval::set_dec(interval!(a, 0.0).unwrap(), x.d);
                rs.insert(TupperInterval::new(
                    x0.recip(),
                    match site {
                        Some(site) => x.g.inserted(site, 0),
                        _ => x.g,
                    },
                ));
                let x1 = DecoratedInterval::set_dec(interval!(0.0, b).unwrap(), x.d);
                rs.insert(TupperInterval::new(
                    x1.recip(),
                    match site {
                        Some(site) => x.g.inserted(site, 1),
                        _ => x.g,
                    },
                ));
            } else {
                rs.insert(TupperInterval::new(x.to_dec_interval().recip(), x.g));
            }
        }
        rs.normalize()
    }

    pub fn rem_euclid(&self, rhs: &Self, site: Option<u8>) -> Self {
        let zero = TupperIntervalSet::from(const_dec_interval!(0.0, 0.0));
        let y = rhs.abs();
        (self - &(&y * &self.div(&y, None).floor(site)))
            .max(&zero)
            .min(&y)
    }

    // Like the (unnormalized) sinc function, but undefined for 0.
    // Less precise for an interval near zero but does not contain zero.
    pub fn sin_over_x(&self) -> Self {
        const ARGMIN_RD: f64 = hexf64!("0x4.7e50150d41abp+0");
        const MIN_RD: f64 = hexf64!("-0x3.79c9f80c234ecp-4");
        let mut rs = Self::empty();
        for x in self {
            let a = x.x.inf();
            let b = x.x.sup();
            if a <= 0.0 && b >= 0.0 {
                let yn = if a < 0.0 {
                    if -a < ARGMIN_RD {
                        let x = interval!(a, a).unwrap();
                        interval!((x.sin() / x).inf(), 1.0).unwrap()
                    } else {
                        interval!(MIN_RD, 1.0).unwrap()
                    }
                } else {
                    Interval::EMPTY
                };
                let yp = if b > 0.0 {
                    if b < ARGMIN_RD {
                        let x = interval!(b, b).unwrap();
                        interval!((x.sin() / x).inf(), 1.0).unwrap()
                    } else {
                        interval!(MIN_RD, 1.0).unwrap()
                    }
                } else {
                    Interval::EMPTY
                };
                let y = DecoratedInterval::set_dec(yn.convex_hull(yp), Decoration::Trv);
                rs.insert(TupperInterval::new(y, x.g));
            } else {
                rs.insert(TupperInterval::new(
                    x.to_dec_interval().sin() / x.to_dec_interval(),
                    x.g,
                ));
            }
        }
        rs.normalize()
    }

    pub fn tan(&self, site: Option<u8>) -> Self {
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
                    DecoratedInterval::set_dec(
                        interval!(interval!(a, a).unwrap().tan().inf(), f64::INFINITY).unwrap(),
                        Decoration::Trv,
                    ),
                    match site {
                        Some(site) => x.g.inserted(site, 0),
                        _ => x.g,
                    },
                ));
                rs.insert(TupperInterval::new(
                    DecoratedInterval::set_dec(
                        interval!(f64::NEG_INFINITY, interval!(b, b).unwrap().tan().sup()).unwrap(),
                        Decoration::Trv,
                    ),
                    match site {
                        Some(site) => x.g.inserted(site, 1),
                        _ => x.g,
                    },
                ));
            } else {
                rs.insert(TupperInterval::new(x.to_dec_interval().tan(), x.g));
            }
        }
        rs.normalize()
    }

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
    impl_no_cut_op!(atan);
    impl_no_cut_op!(atanh);
    impl_no_cut_op!(cos);
    impl_no_cut_op!(cosh);
    impl_no_cut_op!(exp);
    impl_no_cut_op!(exp10);
    impl_no_cut_op!(exp2);
    impl_no_cut_op!(ln);
    impl_no_cut_op!(log10);
    impl_no_cut_op!(log2);
    impl_no_cut_op!(sin);
    impl_no_cut_op!(sinh);
    impl_no_cut_op!(tanh);

    // integer
    impl_integer_op!(ceil);
    impl_integer_op!(floor);
    impl_integer_op!(round_ties_to_away);
    impl_integer_op!(round_ties_to_even);
    impl_integer_op!(sign);
    impl_integer_op!(trunc);
}

bitflags! {
    pub struct SignSet: u8 {
        const NEG = 1;
        const ZERO = 2;
        const POS = 4;
    }
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

#[derive(Clone, Debug)]
pub struct DecSignSet(pub SignSet, pub Decoration);

impl DecSignSet {
    pub fn empty() -> Self {
        Self(SignSet::empty(), Decoration::Trv)
    }
}

#[derive(Clone, Debug)]
pub struct EvalResult(pub SmallVec<[DecSignSet; 32]>);

impl EvalResult {
    pub fn get_size_of_payload(&self) -> usize {
        self.0.capacity() * std::mem::size_of::<DecSignSet>()
    }

    pub fn map<F>(&self, rels: &[StaticRel], f: &F) -> EvalResultMask
    where
        F: Fn(SignSet, Decoration) -> bool,
    {
        let mut m = EvalResultMask(smallvec![false; self.0.len()]);
        Self::map_impl(&self.0[..], rels, rels.len() - 1, f, &mut m.0[..]);
        m
    }

    #[allow(clippy::many_single_char_names)]
    fn map_impl<F>(slf: &[DecSignSet], rels: &[StaticRel], i: usize, f: &F, m: &mut [bool])
    where
        F: Fn(SignSet, Decoration) -> bool,
    {
        use StaticRelKind::*;
        match &rels[i].kind {
            Atomic(_, _, _) => {
                m[i] = f(slf[i].0, slf[i].1);
            }
            And(x, y) => {
                Self::map_impl(&slf, rels, *x as usize, f, m);
                Self::map_impl(&slf, rels, *y as usize, f, m);
            }
            Or(x, y) => {
                Self::map_impl(&slf, rels, *x as usize, f, m);
                Self::map_impl(&slf, rels, *y as usize, f, m);
            }
        }
    }

    pub fn map_reduce<F>(&self, rels: &[StaticRel], f: &F) -> bool
    where
        F: Fn(SignSet, Decoration) -> bool,
    {
        Self::map_reduce_impl(&self.0[..], rels, rels.len() - 1, f)
    }

    fn map_reduce_impl<F>(slf: &[DecSignSet], rels: &[StaticRel], i: usize, f: &F) -> bool
    where
        F: Fn(SignSet, Decoration) -> bool,
    {
        use StaticRelKind::*;
        match &rels[i].kind {
            Atomic(_, _, _) => f(slf[i].0, slf[i].1),
            And(x, y) => {
                Self::map_reduce_impl(&slf, rels, *x as usize, f)
                    && Self::map_reduce_impl(&slf, rels, *y as usize, f)
            }
            Or(x, y) => {
                Self::map_reduce_impl(&slf, rels, *x as usize, f)
                    || Self::map_reduce_impl(&slf, rels, *y as usize, f)
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct EvalResultMask(pub SmallVec<[bool; 32]>);

impl EvalResultMask {
    pub fn reduce(&self, rels: &[StaticRel]) -> bool {
        Self::reduce_impl(&self.0[..], rels, rels.len() - 1)
    }

    fn reduce_impl(slf: &[bool], rels: &[StaticRel], i: usize) -> bool {
        use StaticRelKind::*;
        match &rels[i].kind {
            Atomic(_, _, _) => slf[i],
            And(x, y) => {
                Self::reduce_impl(&slf, rels, *x as usize)
                    && Self::reduce_impl(&slf, rels, *y as usize)
            }
            Or(x, y) => {
                Self::reduce_impl(&slf, rels, *x as usize)
                    || Self::reduce_impl(&slf, rels, *y as usize)
            }
        }
    }

    pub fn solution_certainly_exists(&self, rels: &[StaticRel], locally_zero_mask: &Self) -> bool {
        Self::solution_certainly_exists_impl(
            &self.0[..],
            rels,
            rels.len() - 1,
            &locally_zero_mask.0[..],
        )
    }

    fn solution_certainly_exists_impl(
        slf: &[bool],
        rels: &[StaticRel],
        i: usize,
        locally_zero_mask: &[bool],
    ) -> bool {
        use StaticRelKind::*;
        match &rels[i].kind {
            Atomic(_, _, _) => slf[i],
            And(x, y) => {
                if Self::reduce_impl(&locally_zero_mask, rels, *x as usize) {
                    Self::solution_certainly_exists_impl(
                        &slf,
                        rels,
                        *y as usize,
                        &locally_zero_mask,
                    )
                } else if Self::reduce_impl(&locally_zero_mask, rels, *y as usize) {
                    Self::solution_certainly_exists_impl(
                        &slf,
                        rels,
                        *x as usize,
                        &locally_zero_mask,
                    )
                } else {
                    // Cannot tell the existence of a solution by a normal conjunction.
                    false
                }
            }
            Or(x, y) => {
                Self::solution_certainly_exists_impl(&slf, rels, *x as usize, &locally_zero_mask)
                    || Self::solution_certainly_exists_impl(
                        &slf,
                        rels,
                        *y as usize,
                        &locally_zero_mask,
                    )
            }
        }
    }
}

impl BitAnd for &EvalResultMask {
    type Output = EvalResultMask;

    fn bitand(self, rhs: Self) -> Self::Output {
        #[allow(clippy::suspicious_arithmetic_impl)]
        EvalResultMask(
            self.0
                .iter()
                .zip(rhs.0.iter())
                .map(|(x, y)| *x && *y)
                .collect(),
        )
    }
}

impl BitAndAssign for EvalResultMask {
    fn bitand_assign(&mut self, rhs: Self) {
        *self = self.bitand(&rhs)
    }
}

impl BitOr for &EvalResultMask {
    type Output = EvalResultMask;

    fn bitor(self, rhs: Self) -> Self::Output {
        #[allow(clippy::suspicious_arithmetic_impl)]
        EvalResultMask(
            self.0
                .iter()
                .zip(rhs.0.iter())
                .map(|(x, y)| *x || *y)
                .collect(),
        )
    }
}

impl BitOrAssign for EvalResultMask {
    fn bitor_assign(&mut self, rhs: Self) {
        *self = self.bitor(&rhs)
    }
}
