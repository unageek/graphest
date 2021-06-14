use bitflags::*;
use inari::{DecInterval, Decoration, Interval};
use smallvec::SmallVec;
use std::{
    convert::From,
    hash::{Hash, Hasher},
    iter::{Extend, FromIterator},
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
/// which are used by Tupper IA. The following table describes the relationship between them:
///
/// | Decoration   | def            | cont                   |
/// | ------------ | -------------- | ---------------------- |
/// | `Com`, `Dac` | [T, T]         | [T, T]                 |
/// | `Def`        | [T, T]         | [F, F]; [F, T]         |
/// | `Trv`        | [F, F]; [F, T] | [F, F]; [F, T]; [T, T] |
///
/// [`Interval`] and [`Decoration`] are stored directly rather than through [`DecInterval`]
/// to reduce the size of the struct to 32 bytes from 48, which is due to the alignment.
#[derive(Clone, Copy, Debug)]
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
    pub fn dec_interval(self) -> DecInterval {
        unsafe {
            transmute(_DecInterval {
                x: self.x,
                d: self.d,
            })
        }
    }
}

type TupperIntervalVecBackingArray = [TupperInterval; 2];
type TupperIntervalVec = SmallVec<TupperIntervalVecBackingArray>;

/// A set of [`TupperInterval`]s.
///
/// The traits [`Hash`], [`PartialEq`] and [`Eq`] discriminate interval sets only by their intervals
/// and decorations, and the branch maps are ignored. This is because these traits are only used
/// during construction of [`crate::relation::Relation`]s, where branch maps are always empty.
/// Also note that the traits are sensitive to the order by which the intervals are inserted to.
/// To compare interval sets, you first need to call `normalize(true)` on them.
#[derive(Clone, Debug)]
pub struct TupperIntervalSet {
    xs: TupperIntervalVec,

    /// The decoration of the intervals.
    ///
    /// The same decoration is also stored in each interval.
    /// However, this is the only place where we can keep track of the decoration [`Decoration::Trv`]
    /// if the first intervals being inserted are empty, since they will not be stored in `xs`.
    d: Decoration,
}

impl TupperIntervalSet {
    /// Creates an empty [`TupperIntervalSet`].
    pub fn new() -> Self {
        Self {
            xs: TupperIntervalVec::new(),
            d: Decoration::Com,
        }
    }

    /// Returns the decoration of the intervals.
    pub fn decoration(&self) -> Decoration {
        if self.is_empty() {
            Decoration::Trv
        } else {
            self.d
        }
    }

    /// Inserts an interval to the set and weakens the decoration of the intervals
    /// if the interval being inserted has a weaker decoration than that.
    pub fn insert(&mut self, x: TupperInterval) {
        if !x.x.is_empty() {
            self.xs.push(x);
        }

        let d = self.d.min(x.d);
        if self.d != d {
            self.d = d;
            for x in self.xs.iter_mut() {
                x.d = d;
            }
        }
    }

    /// Returns `true` if the set is empty.
    pub fn is_empty(&self) -> bool {
        self.xs.is_empty()
    }

    /// Returns an iterator over the intervals.
    pub fn iter(&self) -> Iter<'_, TupperInterval> {
        self.xs.iter()
    }

    /// Returns the number of intervals in the set.
    pub fn len(&self) -> usize {
        self.xs.len()
    }

    /// Sorts intervals in a consistent order and merges overlapping intervals
    /// with the same branch map.
    ///
    /// It does nothing when the set is small enough and `force` is set to `false`.
    pub fn normalize(&mut self, force: bool) {
        let xs = &mut self.xs;
        if !force && !xs.spilled() || xs.is_empty() {
            return;
        }

        xs.sort_by(|x, y| {
            (x.g.cut.cmp(&y.g.cut))
                .then(x.g.chosen.cmp(&y.g.chosen))
                .then(x.x.inf().partial_cmp(&y.x.inf()).unwrap())
        });

        let mut hull = Interval::EMPTY;
        let mut g = BranchMap::new();
        let mut write: usize = 0;
        for read in 0..xs.len() {
            let x = xs[read];
            if x.g == g && !x.x.disjoint(hull) {
                hull = hull.convex_hull(x.x);
            } else {
                if !hull.is_empty() {
                    xs[write] = TupperInterval::new(DecInterval::set_dec(hull, self.d), g);
                    write += 1;
                }
                hull = x.x;
                g = x.g;
            }
        }
        if !hull.is_empty() {
            xs[write] = TupperInterval::new(DecInterval::set_dec(hull, self.d), g);
            write += 1;
        }
        xs.truncate(write);
        xs.shrink_to_fit();
    }

    /// Returns the size allocated by the [`TupperIntervalSet`] in bytes.
    pub fn size_in_heap(&self) -> usize {
        if self.xs.spilled() {
            self.xs.capacity() * size_of::<TupperInterval>()
        } else {
            0
        }
    }

    /// Returns the `f64` value if `self` contains exactly one interval
    /// which is singleton and has a decoration ≥ [`Decoration::Def`]; otherwise, `None`.
    ///
    /// If a `f64` value is returned, it implies that the exact result is obtained from evaluation.
    pub fn to_f64(&self) -> Option<f64> {
        if self.len() != 1 {
            return None;
        }

        let x = self.xs[0].x;
        if x.is_singleton() && self.d >= Decoration::Def {
            // Use `sup` instead of `inf` to return +0.0, which is more suitable for formatting.
            Some(x.sup())
        } else {
            None
        }
    }
}

impl PartialEq for TupperIntervalSet {
    fn eq(&self, rhs: &Self) -> bool {
        self.len() == rhs.len()
            && self.iter().zip(rhs.iter()).all(|(x, y)| x.x == y.x)
            && self.decoration() == rhs.decoration()
    }
}

impl Eq for TupperIntervalSet {}

impl Hash for TupperIntervalSet {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for x in self.iter() {
            x.x.inf().to_bits().hash(state);
            x.x.sup().to_bits().hash(state);
        }
        (self.decoration() as u8).hash(state);
    }
}

impl Default for TupperIntervalSet {
    fn default() -> Self {
        Self::new()
    }
}

impl Extend<TupperInterval> for TupperIntervalSet {
    fn extend<T: IntoIterator<Item = TupperInterval>>(&mut self, iter: T) {
        for x in iter {
            self.insert(x);
        }
    }
}

impl<'a> Extend<&'a TupperInterval> for TupperIntervalSet {
    fn extend<T: IntoIterator<Item = &'a TupperInterval>>(&mut self, iter: T) {
        for x in iter {
            self.insert(*x);
        }
    }
}

impl From<DecInterval> for TupperIntervalSet {
    fn from(x: DecInterval) -> Self {
        let mut xs = Self::new();
        xs.insert(TupperInterval::new(x, BranchMap::new()));
        xs
    }
}

impl From<TupperInterval> for TupperIntervalSet {
    fn from(x: TupperInterval) -> Self {
        let mut xs = Self::new();
        xs.insert(x);
        xs
    }
}

impl FromIterator<TupperInterval> for TupperIntervalSet {
    fn from_iter<T: IntoIterator<Item = TupperInterval>>(iter: T) -> Self {
        let mut xs = Self::new();
        xs.extend(iter);
        xs
    }
}

impl IntoIterator for TupperIntervalSet {
    type Item = TupperInterval;
    type IntoIter = smallvec::IntoIter<TupperIntervalVecBackingArray>;

    fn into_iter(self) -> Self::IntoIter {
        self.xs.into_iter()
    }
}

impl<'a> IntoIterator for &'a TupperIntervalSet {
    type Item = &'a TupperInterval;
    type IntoIter = Iter<'a, TupperInterval>;

    fn into_iter(self) -> Self::IntoIter {
        self.xs.iter()
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
/// It is used as a compact version of [`DecInterval`] when only the sign of the interval
/// is of interest.
#[derive(Clone, Copy, Debug)]
pub struct DecSignSet(pub SignSet, pub Decoration);

#[cfg(test)]
mod tests {
    use super::*;
    use inari::{const_dec_interval, const_interval};

    #[test]
    fn decoration() {
        use Decoration::*;

        let mut xs = TupperIntervalSet::new();
        assert_eq!(xs.decoration(), Trv);

        xs.insert(TupperInterval::new(
            const_dec_interval!(0.0, 0.0),
            BranchMap::new(),
        ));
        assert_eq!(xs.decoration(), Com);

        xs.insert(TupperInterval::new(
            DecInterval::set_dec(const_interval!(0.0, 0.0), Def),
            BranchMap::new(),
        ));
        assert_eq!(xs.decoration(), Def);

        let mut xs = TupperIntervalSet::new();
        assert_eq!(xs.decoration(), Trv);

        xs.insert(TupperInterval::new(DecInterval::EMPTY, BranchMap::new()));
        assert_eq!(xs.decoration(), Trv);

        xs.insert(TupperInterval::new(
            const_dec_interval!(0.0, 0.0),
            BranchMap::new(),
        ));
        assert_eq!(xs.decoration(), Trv);
    }
}
