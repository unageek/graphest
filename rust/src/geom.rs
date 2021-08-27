use crate::region::Region;
use inari::{const_interval, interval, Interval};

/// A one-dimensional geometric region that represents a line segment.
///
/// Conceptually, it is a pair of two [`Interval`]s `inner` and `outer`
/// that satisfy `inner ⊆ outer`. `inner` can be empty, while `outer` cannot.
#[derive(Clone, Debug)]
pub struct Box1D {
    l: Interval,
    r: Interval,
}

impl Box1D {
    /// Creates a new [`Box1D`] with the given bounds.
    pub fn new(l: Interval, r: Interval) -> Self {
        assert!(l.inf() <= r.sup());
        Self { l, r }
    }

    /// Returns the inner region.
    pub fn inner(&self) -> Interval {
        let l = self.l.sup();
        let r = self.r.inf();
        if l <= r {
            interval!(l, r).unwrap()
        } else {
            Interval::EMPTY
        }
    }

    /// Returns the left bound of the region.
    pub fn left(&self) -> Interval {
        self.l
    }

    /// Returns the outer region.
    pub fn outer(&self) -> Interval {
        interval!(self.l.inf(), self.r.sup()).unwrap()
    }

    /// Returns the right bound of the region.
    pub fn right(&self) -> Interval {
        self.r
    }

    /// Returns the region transformed by `t`.
    pub fn transform(&self, t: &Transform1D) -> Self {
        Self::new(
            match t.sxd {
                Some(sxd) => self.l / sxd,
                _ => self.l,
            }
            .mul_add(t.sx, t.tx),
            match t.sxd {
                Some(sxd) => self.r / sxd,
                _ => self.r,
            }
            .mul_add(t.sx, t.tx),
        )
    }

    /// Returns the width of the region.
    pub fn width(&self) -> Interval {
        self.r - self.l
    }
}

/// A two-dimensional geometric region that represents an axis-aligned rectangle.
///
/// Conceptually, it is a pair of two [`Region`]s `inner` and `outer`
/// that satisfy `inner ⊆ outer`. `inner` can be empty, while `outer` cannot.
#[derive(Clone, Debug)]
pub struct Box2D(Box1D, Box1D);

impl Box2D {
    /// Creates a new [`Box2D`] with the given bounds.
    pub fn new(l: Interval, r: Interval, b: Interval, t: Interval) -> Self {
        Self(Box1D::new(l, r), Box1D::new(b, t))
    }

    /// Returns the bottom bound of the region.
    pub fn bottom(&self) -> Interval {
        self.1.left()
    }

    /// Returns the height of the region.
    pub fn height(&self) -> Interval {
        self.1.width()
    }

    /// Returns the inner region.
    pub fn inner(&self) -> Region {
        Region::new(self.0.inner(), self.1.inner())
    }

    /// Returns the left bound of the region.
    pub fn left(&self) -> Interval {
        self.0.left()
    }

    /// Returns the outer region.
    pub fn outer(&self) -> Region {
        Region::new(self.0.outer(), self.1.outer())
    }

    /// Returns the right bound of the region.
    pub fn right(&self) -> Interval {
        self.0.right()
    }

    /// Returns the top bound of the region.
    pub fn top(&self) -> Interval {
        self.1.right()
    }

    /// Returns the region transformed by `t`.
    pub fn transform(&self, t: &Transform2D) -> Self {
        Self(self.0.transform(&t.0), self.1.transform(&t.1))
    }

    /// Returns the width of the region.
    pub fn width(&self) -> Interval {
        self.0.width()
    }
}

/// A one-dimensional affine geometric transformation that consists of only scaling and translation.
pub struct Transform1D {
    sx: Interval,
    sxd: Option<Interval>,
    tx: Interval,
}

impl Transform1D {
    /// Creates a transformation that maps `x` to `sx x + tx`.
    pub fn new(sx: Interval, tx: Interval) -> Self {
        Self { sx, sxd: None, tx }
    }

    /// Creates a transformation that maps `x` to `sx (x / sxd) + tx`,
    /// where the division is carried out first.
    pub fn with_predivision_factors((sx, sxd): (Interval, Interval), tx: Interval) -> Self {
        const ONE: Interval = const_interval!(1.0, 1.0);
        Self {
            sx,
            sxd: if sxd == ONE { None } else { Some(sxd) },
            tx,
        }
    }
}

/// A two-dimensional affine geometric transformation that consists of only scaling and translation.
pub struct Transform2D(Transform1D, Transform1D);

impl Transform2D {
    /// Creates a transformation that maps `(x, y)` to `(sx x + tx, sy y + ty)`.
    pub fn new(sx: Interval, tx: Interval, sy: Interval, ty: Interval) -> Self {
        Self(Transform1D::new(sx, tx), Transform1D::new(sy, ty))
    }

    /// Creates a transformation that maps `(x, y)` to `(sx (x / sxd) + tx, sy (y / syd) + ty)`,
    /// where the divisions are carried out first.
    pub fn with_predivision_factors(
        (sx, sxd): (Interval, Interval),
        tx: Interval,
        (sy, syd): (Interval, Interval),
        ty: Interval,
    ) -> Self {
        Self(
            Transform1D::with_predivision_factors((sx, sxd), tx),
            Transform1D::with_predivision_factors((sy, syd), ty),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use inari::const_interval;

    #[test]
    fn box2d() {
        let u = Box2D::new(
            const_interval!(0.33, 0.34),
            const_interval!(0.66, 0.67),
            const_interval!(1.33, 1.34),
            const_interval!(1.66, 1.67),
        );

        assert_eq!(
            u.inner(),
            Region::new(const_interval!(0.34, 0.66), const_interval!(1.34, 1.66))
        );

        assert_eq!(
            u.outer(),
            Region::new(const_interval!(0.33, 0.67), const_interval!(1.33, 1.67))
        );

        let u = Box2D::new(
            const_interval!(0.33, 0.66),
            const_interval!(0.34, 0.67),
            const_interval!(1.33, 1.66),
            const_interval!(1.34, 1.67),
        );

        assert_eq!(u.inner(), Region::EMPTY);

        assert_eq!(
            u.outer(),
            Region::new(const_interval!(0.33, 0.67), const_interval!(1.33, 1.67))
        );
    }
}
