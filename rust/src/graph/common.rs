use inari::{const_interval, interval, Interval};

use crate::{
    block::Block,
    geom::{Box1D, Box2D},
    region::Region,
};

/// The graphing status of a pixel.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PixelState {
    /// The pixel may or may not contain a solution.
    Uncertain,
    /// The same as [`PixelState::Uncertain`] but the pixel has reached the subdivision limit,
    /// so we cannot prove absence of solutions.
    UncertainNeverFalse,
    /// The pixel does not contain a solution.
    False,
    /// The pixel contains a solution.
    True,
}

impl Default for PixelState {
    fn default() -> Self {
        PixelState::Uncertain
    }
}

/// The index of a [`Block`](crate::block::Block) in a [`BlockQueue`](crate::block::BlockQueue).
///
/// While [`BlockQueue::begin_index`](crate::block::BlockQueue::begin_index)/[`BlockQueue::end_index`](crate::block::BlockQueue::begin_index) return [`usize`],
/// [`u32`] would be large enough.
pub type QueuedBlockIndex = u32;

/// Returns the interval \[`x`, `x`\].
///
/// Panics if `x` is infinite or NaN.
pub fn point_interval(x: f64) -> Interval {
    interval!(x, x).unwrap()
}

/// Returns the interval:
///
/// - \[`x`, `x`\] if `x` is finite,
/// - \[−∞, [`f64::MIN`]\] if `x` is −∞, or
/// - \[[`f64::MAX`], +∞\] if `x` is +∞.
///
/// Panics if `x` is NaN.
pub fn point_interval_possibly_infinite(x: f64) -> Interval {
    if x == f64::NEG_INFINITY {
        const_interval!(f64::NEG_INFINITY, f64::MIN)
    } else if x == f64::INFINITY {
        const_interval!(f64::MAX, f64::INFINITY)
    } else {
        point_interval(x)
    }
}

/// Returns a number within the interval whose significand is as short as possible in the binary
/// representation. For such inputs, arithmetic expressions are more likely to be evaluated
/// exactly.
///
/// Precondition: the interval is nonempty.
pub fn simple_fraction(x: Interval) -> f64 {
    let a = x.inf();
    let b = x.sup();
    let a_bits = a.to_bits();
    let b_bits = b.to_bits();
    let diff = a_bits ^ b_bits;
    // The number of leading equal bits.
    let n = diff.leading_zeros();
    if n == 64 {
        return a;
    }
    // Set all bits from the MSB through the first differing bit.
    let mask = !0u64 << (64 - n - 1);
    if a <= 0.0 {
        f64::from_bits(a_bits & mask)
    } else {
        f64::from_bits(b_bits & mask)
    }
}

/// Returns a subset of the outer region.
///
/// It is assumed that the region is obtained from the given block.
/// When applied to a set of regions/blocks which form a partition of a pixel,
/// the results form a partition of the outer boundary of the pixel.
///
/// Precondition: the block is a subpixel.
pub fn subpixel_outer_x(r: &Box1D, b: &Block) -> Interval {
    let mask_x = b.pixel_align_x() - 1;

    let left = if b.x & mask_x == 0 {
        r.left().inf()
    } else {
        r.left().mid()
    };
    let right = if (b.x + 1) & mask_x == 0 {
        r.right().sup()
    } else {
        r.right().mid()
    };

    interval!(left, right).unwrap()
}

/// Returns a subset of the outer region.
///
/// It is assumed that the region is obtained from the given block.
/// When applied to a set of regions/blocks which form a partition of a pixel,
/// the results form a partition of the outer boundary of the pixel.
///
/// Precondition: the block is a subpixel.
pub fn subpixel_outer(r: &Box2D, b: &Block) -> Region {
    let mask_x = b.pixel_align_x() - 1;
    let mask_y = b.pixel_align_y() - 1;

    let left = if b.x & mask_x == 0 {
        r.left().inf()
    } else {
        r.left().mid()
    };
    let right = if (b.x + 1) & mask_x == 0 {
        r.right().sup()
    } else {
        r.right().mid()
    };
    let bottom = if b.y & mask_y == 0 {
        r.bottom().inf()
    } else {
        r.bottom().mid()
    };
    let top = if (b.y + 1) & mask_y == 0 {
        r.top().sup()
    } else {
        r.top().mid()
    };

    Region::new(
        interval!(left, right).unwrap(),
        interval!(bottom, top).unwrap(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use inari::const_interval;

    #[test]
    fn test_subpixel_outer() {
        let r = Box2D::new(
            const_interval!(0.33, 0.34),
            const_interval!(0.66, 0.67),
            const_interval!(1.33, 1.34),
            const_interval!(1.66, 1.67),
        );

        // The bottom/left sides are pixel boundaries.
        let b = Block::new(4, 8, -2, -2, Interval::ENTIRE, Interval::ENTIRE);
        assert_eq!(
            subpixel_outer(&r, &b),
            Region::new(
                interval!(r.left().inf(), r.right().mid()).unwrap(),
                interval!(r.bottom().inf(), r.top().mid()).unwrap()
            )
        );

        // The top/right sides are pixel boundaries.
        let b = Block::new(b.x + 3, b.y + 3, -2, -2, Interval::ENTIRE, Interval::ENTIRE);
        assert_eq!(
            subpixel_outer(&r, &b),
            Region::new(
                interval!(r.left().mid(), r.right().sup()).unwrap(),
                interval!(r.bottom().mid(), r.top().sup()).unwrap()
            )
        );
    }
}
