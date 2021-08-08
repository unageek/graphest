use crate::block::Block;
use inari::{interval, Interval};

/// A possibly empty rectangular region of the Cartesian plane.
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
        Self(self.0.convex_hull(rhs.0), self.1.convex_hull(rhs.1))
    }

    /// Returns the intersection of the regions.
    pub fn intersection(&self, rhs: &Self) -> Self {
        Self(self.0.intersection(rhs.0), self.1.intersection(rhs.1))
    }

    /// Returns `true` if the region is empty.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Returns `true` if `self` is a subset of `rhs`.
    pub fn subset(&self, rhs: &Self) -> bool {
        self.0.subset(rhs.0) && self.1.subset(rhs.1)
    }

    /// Returns the projection of the region onto the x-axis.
    pub fn x(&self) -> Interval {
        self.0
    }

    /// Returns the projection of the region onto the y-axis.
    pub fn y(&self) -> Interval {
        self.1
    }
}

/// A rectangular region of the Cartesian plane with inexact bounds.
///
/// Conceptually, an inexact region is a pair of two rectangular regions: *inner* and *outer*
/// that satisfy `inner ⊆ outer`. The inner region can be empty, while the outer one cannot.
#[derive(Clone, Debug)]
pub struct InexactRegion {
    l: Interval,
    r: Interval,
    b: Interval,
    t: Interval,
}

impl InexactRegion {
    /// Creates a new [`InexactRegion`] with the given bounds.
    pub fn new(l: Interval, r: Interval, b: Interval, t: Interval) -> Self {
        assert!(l.inf() <= r.sup() && b.inf() <= t.sup());
        Self { l, r, b, t }
    }

    /// Returns the bottom bound of the region.
    pub fn bottom(&self) -> Interval {
        self.b
    }

    /// Returns the height of the region.
    pub fn height(&self) -> Interval {
        self.t - self.b
    }

    /// Returns the inner region.
    pub fn inner(&self) -> Region {
        Region::new(
            {
                let l = self.l.sup();
                let r = self.r.inf();
                if l <= r {
                    interval!(l, r).unwrap()
                } else {
                    Interval::EMPTY
                }
            },
            {
                let b = self.b.sup();
                let t = self.t.inf();
                if b <= t {
                    interval!(b, t).unwrap()
                } else {
                    Interval::EMPTY
                }
            },
        )
    }

    /// Returns the left bound of the region.
    pub fn left(&self) -> Interval {
        self.l
    }

    /// Returns the outer region.
    pub fn outer(&self) -> Region {
        Region::new(
            interval!(self.l.inf(), self.r.sup()).unwrap(),
            interval!(self.b.inf(), self.t.sup()).unwrap(),
        )
    }

    /// Returns the right bound of the region.
    pub fn right(&self) -> Interval {
        self.r
    }

    /// Returns a subset of the outer region.
    ///
    /// It is assumed that the region is obtained from the given block.
    /// When applied to a set of regions/blocks which form a partition of a pixel,
    /// the results form a partition of the outer boundary of the pixel.
    ///
    /// Precondition: the block is a subpixel.
    pub fn subpixel_outer(&self, blk: &Block) -> Region {
        let mask_x = blk.pixel_align_x() - 1;
        let mask_y = blk.pixel_align_y() - 1;

        let l = if blk.x & mask_x == 0 {
            self.l.inf()
        } else {
            self.l.mid()
        };
        let r = if (blk.x + 1) & mask_x == 0 {
            self.r.sup()
        } else {
            self.r.mid()
        };
        let b = if blk.y & mask_y == 0 {
            self.b.inf()
        } else {
            self.b.mid()
        };
        let t = if (blk.y + 1) & mask_y == 0 {
            self.t.sup()
        } else {
            self.t.mid()
        };
        Region::new(interval!(l, r).unwrap(), interval!(b, t).unwrap())
    }

    /// Returns the top bound of the region.
    pub fn top(&self) -> Interval {
        self.t
    }

    /// Returns the region transformed by `t`.
    pub fn transform(&self, t: &Transform) -> Self {
        Self::new(
            self.l.mul_add(t.sx, t.tx),
            self.r.mul_add(t.sx, t.tx),
            self.b.mul_add(t.sy, t.ty),
            self.t.mul_add(t.sy, t.ty),
        )
    }

    /// Returns the width of the region.
    pub fn width(&self) -> Interval {
        self.r - self.l
    }
}

/// A 2-D affine geometric transformation composed of scaling and translation.
pub struct Transform {
    sx: Interval,
    sy: Interval,
    tx: Interval,
    ty: Interval,
}

impl Transform {
    /// Creates a transformation that maps `(x, y)` to `(sx x + tx, sy y + ty)`.
    pub fn new(sx: Interval, sy: Interval, tx: Interval, ty: Interval) -> Self {
        Self { sx, sy, tx, ty }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use inari::const_interval;

    #[test]
    fn inexact_region() {
        let u = InexactRegion::new(
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

        // The bottom/left sides are pixel boundaries.
        let b = Block::new(4, 8, -2, -2, Interval::ENTIRE, Interval::ENTIRE);
        assert_eq!(
            u.subpixel_outer(&b),
            Region::new(
                interval!(u.left().inf(), u.right().mid()).unwrap(),
                interval!(u.bottom().inf(), u.top().mid()).unwrap()
            )
        );

        // The top/right sides are pixel boundaries.
        let b = Block::new(b.x + 3, b.y + 3, -2, -2, Interval::ENTIRE, Interval::ENTIRE);
        assert_eq!(
            u.subpixel_outer(&b),
            Region::new(
                interval!(u.left().mid(), u.right().sup()).unwrap(),
                interval!(u.bottom().mid(), u.top().sup()).unwrap()
            )
        );

        let u = InexactRegion::new(
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
