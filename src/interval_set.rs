#![allow(clippy::float_cmp)]

use bitflags::*;
use core::ops::{Add, BitAnd, BitAndAssign, BitOr, BitOrAssign, Mul, Neg, Sub};
use hexf::*;
use inari::*;
use smallvec::SmallVec;
use std::{
    convert::From,
    hash::{Hash, Hasher},
};

// Represents a partial function {0, ..., 31} -> {0, 1}
// which domain is the set of branch cut site ids
// and range is the set of branch indices.
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

#[derive(Clone, Copy)]
#[repr(C)]
struct _DecoratedInterval {
    x: Interval,
    d: Decoration,
}

// We don't store `DecoratedInterval` directly as that would make
// the size of `TupperInterval` 48 bytes instead of 32 due to the alignment.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
struct TupperInterval {
    x: Interval,
    d: Decoration,
    g: IntervalBranch,
}

impl TupperInterval {
    fn new(x: DecoratedInterval, g: IntervalBranch) -> Self {
        // nai is prohibited.
        assert!(!x.is_nai());
        let x = unsafe { std::mem::transmute::<DecoratedInterval, _DecoratedInterval>(x) };
        Self { x: x.x, d: x.d, g }
    }

    fn to_dec_interval(self) -> DecoratedInterval {
        let x = _DecoratedInterval {
            x: self.x,
            d: self.d,
        };
        unsafe { std::mem::transmute(x) }
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

type TupperIntervalVec = SmallVec<[TupperInterval; 2]>;

// NOTE: Equality is order-sensitive.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct TupperIntervalSet(TupperIntervalVec);

impl TupperIntervalSet {
    pub fn empty() -> Self {
        Self(TupperIntervalVec::new())
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    fn insert(&mut self, x: TupperInterval) {
        if !x.x.is_empty() {
            self.0.push(x);
        }
    }

    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

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

impl Neg for &TupperIntervalSet {
    type Output = TupperIntervalSet;

    fn neg(self) -> Self::Output {
        let mut rs = Self::Output::empty();
        for x in &self.0 {
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
                for x in &self.0 {
                    for y in &rhs.0 {
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
            for x in &self.0 {
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
            for x in &self.0 {
                for y in &rhs.0 {
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
            for x in &self.0 {
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

impl TupperIntervalSet {
    pub fn atan2(&self, rhs: &Self, site: Option<u8>) -> Self {
        let mut rs = Self::empty();
        for y in &self.0 {
            for x in &rhs.0 {
                if let Some(g) = x.g.union(y.g) {
                    let a = x.x.inf();
                    let b = x.x.sup();
                    let c = y.x.inf();
                    let d = y.x.sup();
                    if a == 0.0 && b == 0.0 && c < 0.0 && d > 0.0 {
                        rs.insert(TupperInterval::new(
                            DecoratedInterval::set_dec(-Interval::FRAC_PI_2, Decoration::Trv),
                            match site {
                                Some(site) => g.inserted(site, 0),
                                _ => g,
                            },
                        ));
                        rs.insert(TupperInterval::new(
                            DecoratedInterval::set_dec(Interval::FRAC_PI_2, Decoration::Trv),
                            match site {
                                Some(site) => g.inserted(site, 1),
                                _ => g,
                            },
                        ));
                    } else if b <= 0.0 && c < 0.0 && d >= 0.0 {
                        let dec = if b == 0.0 {
                            Decoration::Trv
                        } else {
                            Decoration::Def
                        };
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
                        if d == 0.0 {
                            rs.insert(TupperInterval::new(
                                DecoratedInterval::set_dec(Interval::PI, dec),
                                match site {
                                    Some(site) => g.inserted(site, 1),
                                    _ => g,
                                },
                            ));
                        } else {
                            let x1 = interval!(b, b).unwrap();
                            let y1 = interval!(d, d).unwrap();
                            rs.insert(TupperInterval::new(
                                DecoratedInterval::set_dec(
                                    interval!(y1.atan2(x1).inf(), Interval::PI.sup()).unwrap(),
                                    dec,
                                ),
                                match site {
                                    Some(site) => g.inserted(site, 1),
                                    _ => g,
                                },
                            ));
                        }
                    } else {
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
        for x in &self.0 {
            for y in &rhs.0 {
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

    pub fn mul_add(&self, rhs: &Self, addend: &Self) -> Self {
        let mut rs = Self::empty();
        for x in &self.0 {
            for y in &rhs.0 {
                if let Some(g) = x.g.union(y.g) {
                    for z in &addend.0 {
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

    pub fn pown(&self, rhs: i32, site: Option<u8>) -> Self {
        let mut rs = Self::empty();
        for x in &self.0 {
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
        for x in &self.0 {
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

    // Like the sinc function, but undefined for 0.
    // Less precise for an interval near zero but does not contain zero.
    pub fn sin_over_x(&self) -> Self {
        const ARGMIN_RD: f64 = hexf64!("0x4.7e50150d41abp+0");
        const MIN_RD: f64 = hexf64!("-0x3.79c9f80c234ecp-4");
        let mut rs = Self::empty();
        for x in &self.0 {
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
        for x in &self.0 {
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
    impl_no_cut_op!(log);
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
        pub fn $op(&self, rhs: &Self) -> EvalResult {
            let xs = self - rhs;
            if xs.is_empty() {
                return EvalResult(vec![(SignSet::empty(), Decoration::Trv)]);
            }

            let mut ss = SignSet::empty();
            let mut d = Decoration::Com;
            for x in xs.0.iter() {
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

            EvalResult(vec![(ss, d)])
        }
    };
}

impl TupperIntervalSet {
    impl_rel_op!(eq, SignSet::NEG, SignSet::ZERO, SignSet::POS);

    // f >= 0 ⇔ (f >= 0 ? 0 : 1) == 0, etc.
    impl_rel_op!(ge, SignSet::POS, SignSet::ZERO, SignSet::ZERO);
    impl_rel_op!(gt, SignSet::POS, SignSet::POS, SignSet::ZERO);
    impl_rel_op!(le, SignSet::ZERO, SignSet::ZERO, SignSet::POS);
    impl_rel_op!(lt, SignSet::ZERO, SignSet::POS, SignSet::POS);
}

#[derive(Clone, Debug)]
pub enum PropositionKind {
    Atomic,
    And(Box<(Proposition, Proposition)>),
    Or(Box<(Proposition, Proposition)>),
}

#[derive(Clone, Debug)]
pub struct Proposition {
    pub kind: PropositionKind,
    pub size: usize,
}

#[derive(Clone, Debug)]
pub struct EvalResult(pub Vec<(SignSet, Decoration)>);

#[derive(Clone, Debug)]
pub struct EvalResultMask(pub Vec<bool>);

impl EvalResult {
    pub fn get_size_of_payload(&self) -> usize {
        self.0.capacity() * std::mem::size_of::<(SignSet, Decoration)>()
    }

    pub fn map<F>(&self, p: &Proposition, f: &F) -> EvalResultMask
    where
        F: Fn(SignSet, Decoration) -> bool,
    {
        let mut m = EvalResultMask(vec![false; p.size]);
        Self::map_impl(&self.0[..], p, f, &mut m.0[..]);
        m
    }

    #[allow(clippy::many_single_char_names)]
    fn map_impl<F>(slf: &[(SignSet, Decoration)], p: &Proposition, f: &F, m: &mut [bool])
    where
        F: Fn(SignSet, Decoration) -> bool,
    {
        use PropositionKind::*;
        match &p.kind {
            Atomic => {
                m[0] = f(slf[0].0, slf[0].1);
            }
            And(box (x, y)) => {
                Self::map_impl(&slf[..x.size], &x, f, &mut m[..x.size]);
                Self::map_impl(&slf[x.size..], &y, f, &mut m[x.size..]);
            }
            Or(box (x, y)) => {
                Self::map_impl(&slf[..x.size], &x, f, &mut m[..x.size]);
                Self::map_impl(&slf[x.size..], &y, f, &mut m[x.size..]);
            }
        }
    }

    pub fn map_reduce<F>(&self, p: &Proposition, f: &F) -> bool
    where
        F: Fn(SignSet, Decoration) -> bool,
    {
        Self::map_reduce_impl(&self.0[..], p, f)
    }

    fn map_reduce_impl<F>(slf: &[(SignSet, Decoration)], p: &Proposition, f: &F) -> bool
    where
        F: Fn(SignSet, Decoration) -> bool,
    {
        use PropositionKind::*;
        match &p.kind {
            Atomic => f(slf[0].0, slf[0].1),
            And(box (x, y)) => {
                Self::map_reduce_impl(&slf[..x.size], &x, f)
                    && Self::map_reduce_impl(&slf[x.size..], &y, f)
            }
            Or(box (x, y)) => {
                Self::map_reduce_impl(&slf[..x.size], &x, f)
                    || Self::map_reduce_impl(&slf[x.size..], &y, f)
            }
        }
    }
}

impl Default for EvalResult {
    fn default() -> Self {
        Self(Vec::new())
    }
}

impl EvalResultMask {
    pub fn implies_solution(&self, p: &Proposition, locally_zero_mask: &Self) -> bool {
        Self::implies_solution_impl(&self.0[..], p, &locally_zero_mask.0[..])
    }

    fn implies_solution_impl(slf: &[bool], p: &Proposition, locally_zero_mask: &[bool]) -> bool {
        use PropositionKind::*;
        match &p.kind {
            Atomic => slf[0],
            And(box (x, y)) => {
                if Self::reduce_impl(&locally_zero_mask[..x.size], &x) {
                    Self::implies_solution_impl(&slf[x.size..], &y, &locally_zero_mask[x.size..])
                } else if Self::reduce_impl(&locally_zero_mask[x.size..], &y) {
                    Self::implies_solution_impl(&slf[..x.size], &x, &locally_zero_mask[..x.size])
                } else {
                    // Cannot tell the existence of a solution by a normal conjunction.
                    false
                }
            }
            Or(box (x, y)) => {
                Self::implies_solution_impl(&slf[..x.size], &x, &locally_zero_mask[..x.size])
                    || Self::implies_solution_impl(&slf[x.size..], &y, &locally_zero_mask[x.size..])
            }
        }
    }

    pub fn reduce(&self, p: &Proposition) -> bool {
        Self::reduce_impl(&self.0[..], p)
    }

    fn reduce_impl(slf: &[bool], p: &Proposition) -> bool {
        use PropositionKind::*;
        match &p.kind {
            Atomic => slf[0],
            And(box (x, y)) => {
                Self::reduce_impl(&slf[..x.size], &x) && Self::reduce_impl(&slf[x.size..], &y)
            }
            Or(box (x, y)) => {
                Self::reduce_impl(&slf[..x.size], &x) || Self::reduce_impl(&slf[x.size..], &y)
            }
        }
    }
}

impl BitAnd for &EvalResultMask {
    type Output = EvalResultMask;

    fn bitand(self, rhs: Self) -> Self::Output {
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
