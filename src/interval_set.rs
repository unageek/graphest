use bitflags::*;
use inari::{DecInterval, Decoration, Interval};
use smallvec::SmallVec;
use std::{
    convert::From,
    hash::{Hash, Hasher},
    mem::{size_of, transmute},
    slice::Iter,
};

/// A branch cut site.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Site(u8);

impl Site {
    pub const MAX: u8 = 31;

    pub fn new(site: u8) -> Self {
        assert!(site <= Self::MAX);
        Self(site)
    }
}

/// A branch index.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Branch(u8);

impl Branch {
    pub const MAX: u8 = 1;

    pub fn new(branch: u8) -> Self {
        assert!(branch <= Self::MAX);
        Self(branch)
    }
}

/// A partial function from the set of branch cut sites to the set of branch indices.
///
/// For example, `BranchMap { cut: 0b00101110, chosen: 0b00001010 }`
/// represents a function `{1 ↦ 1, 2 ↦ 0, 3 ↦ 1, 5 ↦ 0}`.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(C)]
pub struct BranchMap {
    /// A bit field that keeps track of at which sites
    /// branch cuts have been performed during evaluation.
    cut: u32,
    /// A bit field that records the branch chosen (0 or 1)
    /// at each site, when the corresponding bit of `cut` is set.
    chosen: u32,
}

impl BranchMap {
    /// Creates an empty [`BranchMap`].
    pub fn new() -> Self {
        Self { cut: 0, chosen: 0 }
    }

    /// Creates a [`BranchMap`] defined by `self ∪ {site ↦ branch}`.
    ///
    /// Panics if `site ∈ dom(self)`.
    pub fn inserted(self, site: Site, branch: Branch) -> Self {
        assert!(self.cut & 1 << site.0 == 0);
        Self {
            cut: self.cut | 1 << site.0,
            chosen: self.chosen | (branch.0 as u32) << site.0,
        }
    }

    /// Returns `Some(self ∪ rhs)` if `self` and `rhs` are compatible, i.e.,
    /// `∀x ∈ dom(self) ∩ dom(rhs) : self(x) = rhs(x)`;
    /// otherwise, `None`.
    pub fn union(self, rhs: Self) -> Option<Self> {
        let mask = self.cut & rhs.cut;
        let compatible = self.chosen & mask == rhs.chosen & mask;

        if compatible {
            Some(Self {
                cut: self.cut | rhs.cut,
                chosen: self.chosen | rhs.chosen,
            })
        } else {
            None
        }
    }
}

impl Default for BranchMap {
    fn default() -> Self {
        Self::new()
    }
}

// Used for type punning. The layout must be exactly the same with `DecInterval`.
#[repr(C)]
struct _DecInterval {
    x: Interval,
    d: Decoration,
}

/// An interval with additional properties that are required by Tupper interval arithmetic.
///
/// The decoration system is used instead of the interval properties `def` and `cont`,
/// introduced by the original Tupper IA. Here is the relationship between them:
///
/// | Decoration   | def            | cont                   |
/// | ------------ | -------------- | ---------------------- |
/// | `Com`, `Dac` | [T, T]         | [T, T]                 |
/// | `Def`        | [T, T]         | [F, F]; [F, T]         |
/// | `Trv`        | [F, F]; [F, T] | [F, F]; [F, T]; [T, T] |
///
/// I'm not 100% certain if the above mapping is correct, but there should be no problem
/// on implementing the graphing algorithms.
///
/// [`Interval`] and [`Decoration`] are stored directly rather than through [`DecInterval`]
/// to reduce the size of the struct to 32 bytes from 48, which is due to the alignment.
///
/// NOTE: [`Hash`], [`PartialEq`] and [`Eq`] look only the interval part and ignores
/// the decoration and the branch map.
/// This is because these traits are only used for discriminating interval constants,
/// and those constants always have the maximum decorations and the empty branch maps.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct TupperInterval {
    pub x: Interval,
    pub d: Decoration,
    pub g: BranchMap,
}

impl TupperInterval {
    /// Creates a new [`TupperInterval`] with the given [`DecInterval`] and [`BranchMap`].
    ///
    /// Panics if the interval is NaI.
    pub fn new(x: DecInterval, g: BranchMap) -> Self {
        assert!(!x.is_nai());
        let x = unsafe { transmute::<_, _DecInterval>(x) };
        Self { x: x.x, d: x.d, g }
    }

    /// Returns the [`DecInterval`] part of the interval.
    pub fn to_dec_interval(self) -> DecInterval {
        unsafe {
            transmute(_DecInterval {
                x: self.x,
                d: self.d,
            })
        }
    }
}

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

/// A set of [`TupperInterval`]s.
///
/// NOTE: [`Hash`], [`PartialEq`] and [`Eq`] are sensitive to the order by which the intervals
/// are inserted to. See also the note in [`TupperInterval`].
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct TupperIntervalSet(TupperIntervalVec);

impl TupperIntervalSet {
    /// Creates an empty [`TupperIntervalSet`].
    pub fn empty() -> Self {
        Self(TupperIntervalVec::new())
    }

    /// Returns an iterator.
    pub fn iter(&self) -> Iter<'_, TupperInterval> {
        self.0.iter()
    }

    /// Returns the number of intervals in the set.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Inserts an interval to the set. If the interval is empty, it does nothing.
    pub fn insert(&mut self, x: TupperInterval) {
        if !x.x.is_empty() {
            self.0.push(x);
        }
    }

    /// Returns `true` if the set is empty.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Sorts intervals in a consistent order, takes the least decoration
    /// and merges overlapping intervals with the same branch maps.
    ///
    /// It does nothing when the set is small enough and `force` is set to `false`.
    pub fn normalize(&mut self, force: bool) {
        let xs = &mut self.0;
        if !force && !xs.spilled() || xs.is_empty() {
            return;
        }

        xs.sort_by(|x, y| {
            (x.g.cut.cmp(&y.g.cut))
                .then(x.g.chosen.cmp(&y.g.chosen))
                .then(x.x.inf().partial_cmp(&y.x.inf()).unwrap())
        });
        let dec = xs.iter().map(|x| x.d).min().unwrap();

        let mut hull = Interval::EMPTY;
        let mut g = BranchMap::new();
        let mut write: usize = 0;
        for read in 0..xs.len() {
            let x = xs[read];
            if x.g == g && !x.x.disjoint(hull) {
                hull = hull.convex_hull(x.x);
            } else {
                if !hull.is_empty() {
                    xs[write] = TupperInterval::new(DecInterval::set_dec(hull, dec), g);
                    write += 1;
                }
                hull = x.x;
                g = x.g;
            }
        }
        if !hull.is_empty() {
            xs[write] = TupperInterval::new(DecInterval::set_dec(hull, dec), g);
            write += 1;
        }
        xs.truncate(write);
        xs.shrink_to_fit();
    }

    /// Retains only the intervals specified by the predicate.
    ///
    /// It preserves the order of the retained elements.
    pub fn retain<F>(&mut self, f: F)
    where
        F: FnMut(&mut TupperInterval) -> bool,
    {
        self.0.retain(f)
    }

    pub fn size_in_heap(&self) -> usize {
        if self.0.spilled() {
            self.0.capacity() * size_of::<TupperInterval>()
        } else {
            0
        }
    }
}

impl From<DecInterval> for TupperIntervalSet {
    fn from(x: DecInterval) -> Self {
        let mut xs = Self::empty();
        xs.insert(TupperInterval::new(x, BranchMap::new()));
        xs
    }
}

impl From<TupperInterval> for TupperIntervalSet {
    fn from(x: TupperInterval) -> Self {
        let mut xs = Self::empty();
        xs.insert(x);
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

bitflags! {
    /// A set of signs: negative, positive or zero.
    pub struct SignSet: u8 {
        const NEG = 1;
        const ZERO = 2;
        const POS = 4;
    }
}

/// A pair of [`SignSet`] and [`Decoration`].
///
/// It is used as an efficient version of [`DecInterval`] when only the sign of the interval
/// is of interest.
#[derive(Clone, Copy, Debug)]
pub struct DecSignSet(pub SignSet, pub Decoration);

impl DecSignSet {
    pub fn empty() -> Self {
        Self(SignSet::empty(), Decoration::Trv)
    }
}
