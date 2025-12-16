use crate::{
    geom::{Transform, TransformInPlace, Transformation1D},
    traits::BytesAllocated,
};
use bitflags::*;
use inari::{DecInterval, Decoration, Interval};
use smallvec::SmallVec;
use std::{
    convert::From,
    hash::{Hash, Hasher},
    iter::{Extend, FromIterator},
    mem::transmute,
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
        assert!(self.cut & (1 << site.0) == 0);
        Self {
            cut: self.cut | (1 << site.0),
            chosen: self.chosen | ((branch.0 as u32) << site.0),
        }
    }

    /// Returns `self ∪ rhs` if `self` and `rhs` are compatible, i.e., they satisfy
    /// `∀x ∈ dom(self) ∩ dom(rhs) : self(x) = rhs(x)`; otherwise, [`None`].
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

// Used for type punning. The layout must be exactly the same as `DecInterval`.
#[repr(C)]
struct _DecInterval {
    x: Interval,
    d: Decoration,
}

/// An interval augmented with properties that are required by Tupper interval arithmetic.
///
/// The decoration system is used instead of the Tupper IA's interval properties: `def` and `cont`.
/// For a nonempty interval, the relationship between them is:
///
/// | Decoration   | `def`          | `cont`                 |
/// | ------------ | -------------- | ---------------------- |
/// | `Com`, `Dac` | [T, T]         | [T, T]                 |
/// | `Def`        | [T, T]         | [F, F], [F, T]         |
/// | `Trv`        | [F, F], [F, T] | [F, F], [F, T], [T, T] |
///
/// Tupper IA primarily works with sets of intervals.
/// The empty set is represented by the empty set of intervals, instead of the empty interval.
///
/// The interval and the decoration are stored directly rather than through [`DecInterval`]
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
        let x = unsafe { transmute::<inari::DecInterval, _DecInterval>(x) };
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

impl From<DecInterval> for TupperInterval {
    fn from(x: DecInterval) -> Self {
        Self::new(x, BranchMap::new())
    }
}

type TupperIntervalVecBackingArray = [TupperInterval; 2];
type TupperIntervalVec = SmallVec<TupperIntervalVecBackingArray>;

/// A set of [`TupperInterval`]s.
///
/// Notes on the traits [`PartialEq`], [`Eq`] and [`Hash`]:
///
/// - Unlike [`DecInterval`], the traits distinguish interval sets with different decorations.
///
/// - The traits are sensitive to the order by which the intervals have been inserted.
///   To compare interval sets, you first need to call `normalize(true)` on them.
#[derive(Clone, Debug)]
pub struct TupperIntervalSet {
    xs: TupperIntervalVec,

    /// The decoration of the interval set.
    ///
    /// The same decoration is also stored in each interval.
    /// However, this is the only place where we can keep track of the decoration [`Decoration::Trv`]
    /// if the first intervals being inserted are empty, since they will not be stored in `xs`.
    d: Decoration,
}

impl TupperIntervalSet {
    /// The maximum number of intervals left after normalization.
    ///
    /// See [`Self::normalize`].
    pub const MAX_INTERVALS: usize = 16;

    /// Creates an empty [`TupperIntervalSet`].
    pub fn new() -> Self {
        Self {
            xs: TupperIntervalVec::new(),
            d: Decoration::Com,
        }
    }

    /// Returns the decoration of the interval set.
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

        self.d = self.d.min(x.d);
        for x in self.xs.iter_mut() {
            x.d = self.d;
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

    /// Sorts the intervals in a consistent order, and merges overlapping ones
    /// which share the same branch map. If there are more than [`Self::MAX_INTERVALS`] intervals,
    /// merges all of them regardless of the branch maps by taking the convex hull,
    /// leaving exactly one interval.
    ///
    /// It does nothing when the set is small enough and `force` is `false`.
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

        if xs.len() > Self::MAX_INTERVALS {
            let hull = xs
                .drain(..)
                .map(|x| x.x)
                .reduce(|acc, x| acc.convex_hull(x))
                .unwrap();
            xs.push(DecInterval::set_dec(hull, self.d).into());
        }

        xs.shrink_to_fit();
    }

    /// Returns the only [`f64`] number in the set if `self` contains exactly one interval
    /// which is a singleton and has a decoration ≥ [`Decoration::Def`]; otherwise, [`None`].
    /// Zero is returned as `+0.0`.
    ///
    /// If a [`f64`] number is obtained, that is the exact value of the evaluated expression.
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
            && self
                .iter()
                .zip(rhs.iter())
                .all(|(x, y)| x.x == y.x && x.g == y.g)
            && self.decoration() == rhs.decoration()
    }
}

impl Eq for TupperIntervalSet {}

impl Hash for TupperIntervalSet {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for x in self.iter() {
            x.x.hash(state);
            x.g.hash(state);
        }
        self.decoration().hash(state);
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
        xs.insert(x.into());
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

impl BytesAllocated for TupperIntervalSet {
    fn bytes_allocated(&self) -> usize {
        self.xs.bytes_allocated()
    }
}

impl TransformInPlace<Transformation1D> for TupperIntervalSet {
    fn transform_in_place(&mut self, t: &Transformation1D) {
        self.d = self.d.min(Decoration::Dac);
        for x in &mut self.xs {
            x.x = x.x.transform(t);
            x.d = self.d;
        }
    }
}

bitflags! {
    /// A set of signs; a subset of {−, 0, +}.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
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

        xs.insert(TupperInterval::from(const_dec_interval!(0.0, 0.0)));
        assert_eq!(xs.decoration(), Com);

        xs.insert(TupperInterval::from(DecInterval::set_dec(
            const_interval!(0.0, 0.0),
            Def,
        )));
        assert_eq!(xs.decoration(), Def);

        let mut xs = TupperIntervalSet::new();
        assert_eq!(xs.decoration(), Trv);

        xs.insert(TupperInterval::from(DecInterval::EMPTY));
        assert_eq!(xs.decoration(), Trv);

        xs.insert(TupperInterval::from(const_dec_interval!(0.0, 0.0)));
        assert_eq!(xs.decoration(), Trv);
    }

    #[test]
    fn normalize() {
        fn test(input: Vec<TupperInterval>, output: Vec<TupperInterval>) {
            let mut input = input.into_iter().collect::<TupperIntervalSet>();
            let output = output.into_iter().collect::<TupperIntervalSet>();
            input.normalize(true);
            assert_eq!(input, output);
        }

        macro_rules! i {
            ($a:expr, $b:expr) => {
                const_dec_interval!($a, $b)
            };
        }

        let g = BranchMap::new().inserted(Site::new(0), Branch::new(0));
        let g2 = BranchMap::new().inserted(Site::new(0), Branch::new(1));

        test(vec![], vec![]);

        test(
            vec![
                TupperInterval::new(i!(0.0, 2.0), g),
                TupperInterval::new(i!(1.0, 3.0), g),
            ],
            vec![TupperInterval::new(i!(0.0, 3.0), g)],
        );

        test(
            vec![
                TupperInterval::new(i!(1.0, 3.0), g),
                TupperInterval::new(i!(0.0, 2.0), g),
            ],
            vec![TupperInterval::new(i!(0.0, 3.0), g)],
        );

        // Non-overlapping intervals are not merged.
        test(
            vec![
                TupperInterval::new(i!(0.0, 1.0), g),
                TupperInterval::new(i!(2.0, 3.0), g),
            ],
            vec![
                TupperInterval::new(i!(0.0, 1.0), g),
                TupperInterval::new(i!(2.0, 3.0), g),
            ],
        );

        // Intervals with different branch maps are not merged.
        test(
            vec![
                TupperInterval::new(i!(0.0, 2.0), g),
                TupperInterval::new(i!(1.0, 3.0), g2),
            ],
            vec![
                TupperInterval::new(i!(0.0, 2.0), g),
                TupperInterval::new(i!(1.0, 3.0), g2),
            ],
        );

        test(
            vec![
                TupperInterval::new(i!(0.0, 2.0), g),
                TupperInterval::new(i!(1.0, 3.0), g2),
                TupperInterval::new(i!(2.0, 4.0), g),
            ],
            vec![
                TupperInterval::new(i!(0.0, 4.0), g),
                TupperInterval::new(i!(1.0, 3.0), g2),
            ],
        );
    }

    #[test]
    fn struct_size() {
        assert_eq!(size_of::<TupperIntervalSet>(), 112);
        assert_eq!(size_of::<Option<TupperIntervalSet>>(), 112);
    }

    #[test]
    fn to_f64() {
        let xs = TupperIntervalSet::new();
        assert_eq!(xs.to_f64(), None);

        for d in [Decoration::Com, Decoration::Dac, Decoration::Def] {
            let mut xs = TupperIntervalSet::new();
            xs.insert(TupperInterval::from(DecInterval::set_dec(
                const_interval!(0.1, 0.1),
                d,
            )));
            assert_eq!(xs.to_f64(), Some(0.1));
        }

        let mut xs = TupperIntervalSet::new();
        xs.insert(TupperInterval::from(DecInterval::set_dec(
            const_interval!(0.1, 0.1),
            Decoration::Trv,
        )));
        assert_eq!(xs.to_f64(), None);

        let mut xs = TupperIntervalSet::new();
        xs.insert(TupperInterval::from(DecInterval::PI));
        assert_eq!(xs.to_f64(), None);

        // The sign bit of 0.0 is positive.
        let mut xs = TupperIntervalSet::new();
        xs.insert(TupperInterval::from(const_dec_interval!(0.0, 0.0)));
        assert_eq!(xs.to_f64(), Some(0.0));
        if let Some(zero) = xs.to_f64() {
            assert!(zero.is_sign_positive());
        }
    }
}
