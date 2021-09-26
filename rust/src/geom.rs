use crate::region::Region;
use inari::{interval, Interval};

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

    /// Transforms the region by `t`.
    pub fn transform(&self, t: &Transform1D) -> Self {
        match *t {
            Transform1D::Fast { s, t } => Self::new(self.l.mul_add(s, t), self.r.mul_add(s, t)),
            Transform1D::Precise { a0, a01, x0, x01 } => Self::new(
                ((self.l - a0) / a01).mul_add(x01, x0),
                ((self.r - a0) / a01).mul_add(x01, x0),
            ),
        }
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

    /// Transforms the region by `t`.
    pub fn transform(&self, t: &Transform2D) -> Self {
        Self(self.0.transform(&t.0), self.1.transform(&t.1))
    }

    /// Swaps the axes of the region.
    pub fn transpose(&self) -> Self {
        Self(self.1.clone(), self.0.clone())
    }
}

/// The type of the formula that should be used for performing geometric transformations.
#[derive(Clone, Copy, Debug)]
pub enum TransformMode {
    /// Suitable for transformation from image coordinates to real coordinates,
    /// which usually involves exact divisions (division by image dimensions).
    Fast,
    /// Suitable for transformation from real coordinates to image coordinates,
    /// which usually involves inexact divisions (division by lengths of the plot range).
    Precise,
}

/// A one-dimensional affine geometric transformation that consists of only scaling and translation.
#[derive(Clone, Debug)]
pub enum Transform1D {
    Fast {
        s: Interval,
        t: Interval,
    },
    Precise {
        a0: Interval,
        a01: Interval,
        x0: Interval,
        x01: Interval,
    },
}

impl Transform1D {
    /// Creates a transformation that maps each source point to the corresponding destination point.
    pub fn new(from_points: [Interval; 2], to_points: [Interval; 2], mode: TransformMode) -> Self {
        let [a0, a1] = from_points;
        let [x0, x1] = to_points;
        match mode {
            TransformMode::Fast => Self::Fast {
                s: (x1 - x0) / (a1 - a0),
                t: (-a0).mul_add((x1 - x0) / (a1 - a0), x0),
            },
            TransformMode::Precise => Self::Precise {
                a0,
                a01: a1 - a0,
                x0,
                x01: x1 - x0,
            },
        }
    }
}

/// A two-dimensional affine geometric transformation that consists of only scaling and translation.
#[derive(Clone, Debug)]
pub struct Transform2D(Transform1D, Transform1D);

impl Transform2D {
    /// Creates a transformation that maps each source point to the corresponding destination point.
    pub fn new(from_points: [Region; 2], to_points: [Region; 2], mode: TransformMode) -> Self {
        Self(
            Transform1D::new(
                [from_points[0].x(), from_points[1].x()],
                [to_points[0].x(), to_points[1].x()],
                mode,
            ),
            Transform1D::new(
                [from_points[0].y(), from_points[1].y()],
                [to_points[0].y(), to_points[1].y()],
                mode,
            ),
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
