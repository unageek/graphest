use inari::Interval;

/// The Cartesian product of two [`Interval`]s.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Region(Interval, Interval);

impl Region {
    /// The empty region.
    pub const EMPTY: Self = Self(Interval::EMPTY, Interval::EMPTY);

    /// Creates a new [`Region`].
    pub fn new(x: Interval, y: Interval) -> Self {
        if x.is_empty() || y.is_empty() {
            Self::EMPTY
        } else {
            Self(x, y)
        }
    }

    /// Returns the convex hull of the regions.
    pub fn convex_hull(&self, rhs: &Self) -> Self {
        Self::new(self.0.convex_hull(rhs.0), self.1.convex_hull(rhs.1))
    }

    /// Returns the intersection of the regions.
    pub fn intersection(&self, rhs: &Self) -> Self {
        Self::new(self.0.intersection(rhs.0), self.1.intersection(rhs.1))
    }

    /// Returns `true` if the region is empty.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Returns `true` if `self` is a subset of `rhs`.
    pub fn subset(&self, rhs: &Self) -> bool {
        self.0.subset(rhs.0) && self.1.subset(rhs.1)
    }

    /// Returns the x component (the first factor) of the region, i.e., `{x ∣ (x, y) ∈ R}`.
    pub fn x(&self) -> Interval {
        self.0
    }

    /// Returns the y component (the second factor) of the region, i.e., `{y ∣ (x, y) ∈ R}`.
    pub fn y(&self) -> Interval {
        self.1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use inari::const_interval;

    #[test]
    fn region() {
        let r = Region::new(Interval::EMPTY, Interval::EMPTY);
        assert_eq!(r, Region::EMPTY);
        assert!(r.is_empty());
        assert!(r.subset(&r));

        let r = Region::new(Interval::EMPTY, const_interval!(2.0, 4.0));
        assert_eq!(r, Region::EMPTY);
        assert!(r.is_empty());

        let r = Region::new(const_interval!(2.0, 4.0), Interval::EMPTY);
        assert_eq!(r, Region::EMPTY);
        assert!(r.is_empty());

        let r = Region::new(const_interval!(1.0, 4.0), const_interval!(2.0, 5.0));
        let s = Region::new(const_interval!(2.0, 3.0), const_interval!(3.0, 4.0));
        assert!(r.subset(&r));
        assert!(Region::EMPTY.subset(&r));
        assert!(!r.subset(&s));
        assert!(s.subset(&r));

        let r = Region::new(const_interval!(1.0, 3.0), const_interval!(2.0, 4.0));
        let s = Region::new(const_interval!(2.0, 4.0), const_interval!(3.0, 5.0));
        assert_eq!(
            r.convex_hull(&s),
            Region::new(const_interval!(1.0, 4.0), const_interval!(2.0, 5.0))
        );
        assert_eq!(
            r.intersection(&s),
            Region::new(const_interval!(2.0, 3.0), const_interval!(3.0, 4.0))
        );
        assert!(!r.subset(&s));
        assert!(!s.subset(&r));

        let r = Region::new(const_interval!(1.0, 2.0), const_interval!(2.0, 3.0));
        let s = Region::new(const_interval!(3.0, 4.0), const_interval!(4.0, 5.0));
        assert_eq!(
            r.convex_hull(&s),
            Region::new(const_interval!(1.0, 4.0), const_interval!(2.0, 5.0))
        );
        assert!(r.intersection(&s).is_empty());
        assert!(!r.subset(&s));
        assert!(!s.subset(&r));
    }
}
