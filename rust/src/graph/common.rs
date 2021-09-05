use crate::{
    block::Block,
    geom::{Box1D, Box2D},
    region::Region,
};
use inari::{interval, Interval};

/// The index of a [`Block`] in a [`BlockQueue`].
///
/// While [`BlockQueue::begin_index`]/[`BlockQueue::end_index`] return [`usize`],
/// [`u32`] would be large enough.
///
/// [`Block`]: crate::block::Block
/// [`BlockQueue`]: crate::block::BlockQueue
/// [`BlockQueue::begin_index`]: crate::block::BlockQueue::begin_index
/// [`BlockQueue::end_index`]: crate::block::BlockQueue::end_index
pub type QueuedBlockIndex = u32;

/// The graphing status of a pixel.
///
/// # Overview of Pixel States
///
/// ```text
///                                     Found
///                                   a solution     +---------------+
///                              +------------------>|     True      |
///                              |                   +---------------+
///                              |                           Λ    Found
///                              |  No subdivisible          |  a solution
///        +---------------+     |  blocks are left  +-------+-------+
/// ●----->|   Uncertain   +-----+------------------>|   Uncertain   |
///        |  disprovable  |     |                   | undisprovable |
///        +---------------+     |                   +---------------+
///                              |    No solution
///                              |   in all blocks   +---------------+
///                              +------------------>|     False     |
///                                                  +---------------+
/// ```
///
/// The false state is represented as a special case of the variant [`PixelState::Uncertain`],
/// as explained in its description.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PixelState {
    /// The pixel contains a solution.
    True,
    /// The pixel may or may not contain a solution.
    ///
    /// It holds the index of the last block in the queue that intersects with the pixel.
    /// If it is [`None`], no subdivisible block is left for the pixel,
    /// thus we cannot prove absence of solutions.
    ///
    /// If the index is less than that of the front element of the queue,
    /// that implies the pixel does not contain a solution.
    Uncertain(Option<QueuedBlockIndex>),
}

impl PixelState {
    pub fn is_uncertain(self, front_block_index: usize) -> bool {
        match self {
            PixelState::Uncertain(Some(bi)) => bi as usize >= front_block_index,
            PixelState::Uncertain(None) => true,
            _ => false,
        }
    }

    pub fn is_uncertain_and_disprovable(self, front_block_index: usize) -> bool {
        match self {
            PixelState::Uncertain(Some(bi)) => bi as usize >= front_block_index,
            _ => false,
        }
    }

    pub fn is_uncertain_and_undisprovable(self) -> bool {
        self == PixelState::Uncertain(None)
    }
}

impl Default for PixelState {
    fn default() -> Self {
        PixelState::Uncertain(Some(0))
    }
}

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
    interval!(x.min(f64::MAX), x.max(f64::MIN)).unwrap()
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

/// One-dimensional version of [`subpixel_outer`].
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

#[cfg(test)]
mod tests {
    use super::*;
    use inari::const_interval;

    #[test]
    fn test_subpixel_outer_x() {
        let r = Box1D::new(const_interval!(0.33, 0.34), const_interval!(0.66, 0.67));

        // The left side is pixel boundary.
        let b = Block::new(4, 0, -2, 0, Interval::ENTIRE, Interval::ENTIRE);
        assert_eq!(
            subpixel_outer_x(&r, &b),
            interval!(r.left().inf(), r.right().mid()).unwrap()
        );

        // The right side is pixel boundary.
        let b = Block::new(b.x + 3, 0, -2, 0, Interval::ENTIRE, Interval::ENTIRE);
        assert_eq!(
            subpixel_outer_x(&r, &b),
            interval!(r.left().mid(), r.right().sup()).unwrap(),
        );
    }
}
