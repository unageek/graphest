use crate::image::PixelIndex;
use inari::{interval, Interval};
use itertools::Itertools;
use smallvec::SmallVec;
use std::{collections::VecDeque, mem::size_of, ptr::copy_nonoverlapping};

/// Represents a subset of an axis of an [`Image`].
///
/// A [`Coordinate`] with index `i` and level `k` represents the interval `[i 2^k, (i + 1) 2^k]`,
/// where the endpoints represent pixel coordinates.
///
/// [`Image`]: crate::image::Image
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Coordinate {
    i: u64,
    k: i8,
}

impl Coordinate {
    /// The smallest level of horizontal/vertical subdivision.
    ///
    /// A smaller value can be used, as long as the following condition is met:
    ///
    /// - [`Image::MAX_WIDTH`]` / 2^`[`Self::MIN_LEVEL`]` ≤ 2^53`,
    ///
    /// which is required for keeping `block_to_region` operations exact.
    ///
    /// [`Image::MAX_WIDTH`]: crate::image::Image::MAX_WIDTH
    const MIN_LEVEL: i8 = -32;

    pub fn new(index: u64, level: i8) -> Self {
        assert!(level >= Self::MIN_LEVEL);
        Self { i: index, k: level }
    }

    /// Returns the index of the block in multiples of the block width.
    pub fn index(&self) -> u64 {
        self.i
    }

    /// Returns `true` if the block is a subpixel.
    pub fn is_subpixel(&self) -> bool {
        self.k < 0
    }

    /// Returns `true` if the block is a superpixel.
    pub fn is_superpixel(&self) -> bool {
        self.k > 0
    }

    /// Returns `true` if the block can be subdivided both horizontally and vertically.
    pub fn is_subdivisible(&self) -> bool {
        self.k > Self::MIN_LEVEL
    }

    #[allow(dead_code)]
    /// Returns the subdivision level of the block.
    pub fn level(&self) -> i8 {
        self.k
    }

    /// Returns the width of a pixel divided by the block's width.
    ///
    /// Panics if `self.k > 0`.
    pub fn pixel_align(&self) -> u64 {
        assert!(self.k <= 0);
        1u64 << -self.k
    }

    /// Returns the index of the pixel that contains the block.
    /// If the block spans multiple pixels, the least index is returned.
    pub fn pixel_index(&self) -> u32 {
        if self.k >= 0 {
            (self.i << self.k) as u32
        } else {
            (self.i >> -self.k) as u32
        }
    }

    pub fn subdivide(&self) -> [Self; 2] {
        let i0 = 2 * self.i;
        let i1 = i0 + 1;
        let k = self.k - 1;
        [Self { i: i0, k }, Self { i: i1, k }]
    }

    /// Returns the width of the block in pixels.
    ///
    /// Panics if `self.k < 0`.
    pub fn width(&self) -> u32 {
        assert!(self.k >= 0);
        1u32 << self.k
    }

    /// Returns the width of the block in pixels.
    pub fn widthf(&self) -> f64 {
        Self::exp2(self.k)
    }

    /// Returns `2^k`.
    fn exp2(k: i8) -> f64 {
        f64::from_bits(((1023 + k as i32) as u64) << 52)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct IntegerParameter(Interval);

impl IntegerParameter {
    pub fn initial_subdivision(x: Interval) -> Vec<Self> {
        assert!(!x.is_empty() && x == x.trunc());
        let a = x.inf();
        let b = x.sup();
        [interval!(a, a), interval!(a, b), interval!(b, b)]
            .into_iter()
            .filter_map(|x| x.ok()) // Remove invalid constructions, namely, [-∞, -∞] and [+∞, +∞].
            .filter(|x| x.wid() != 1.0)
            .dedup()
            .map(Self::new)
            .collect()
    }

    pub fn new(x: Interval) -> Self {
        assert!(!x.is_empty() && x == x.trunc());
        Self(x)
    }

    pub fn interval(&self) -> Interval {
        self.0
    }

    pub fn is_subdivisible(&self) -> bool {
        let x = self.0;
        let mid = x.mid().round();
        x.inf() != mid && x.sup() != mid
    }

    pub fn subdivide(&self) -> SmallVec<[Self; 3]> {
        let x = self.0;
        let a = x.inf();
        let b = x.sup();
        let mid = if a == f64::NEG_INFINITY {
            (2.0 * b).max(f64::MIN).min(-1.0)
        } else if b == f64::INFINITY {
            (2.0 * a).max(1.0).min(f64::MAX)
        } else {
            x.mid().round()
        };
        [
            interval!(a, mid).unwrap(),
            interval!(mid, mid).unwrap(),
            interval!(mid, b).unwrap(),
        ]
        .into_iter()
        // Any interval with width 1 can be discarded since its endpoints are already processed
        // as point intervals and there are no integers in between them.
        .filter(|x| x.wid() != 1.0)
        .map(Self)
        .collect()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RealParameter(Interval);

impl RealParameter {
    pub fn new(x: Interval) -> Self {
        assert!(!x.is_empty());
        Self(x)
    }

    pub fn interval(&self) -> Interval {
        self.0
    }

    pub fn is_subdivisible(&self) -> bool {
        let x = self.0;
        let mid = x.mid();
        x.inf() != mid && x.sup() != mid
    }

    pub fn subdivide(&self) -> SmallVec<[Self; 2]> {
        let x = self.0;
        let a = x.inf();
        let b = x.sup();
        let mid = if a == f64::NEG_INFINITY {
            if b < 0.0 {
                (2.0 * b).max(f64::MIN)
            } else if b == 0.0 {
                -1.0
            } else {
                0.0
            }
        } else if b == f64::INFINITY {
            if a < 0.0 {
                0.0
            } else if a == 0.0 {
                1.0
            } else {
                (2.0 * a).min(f64::MAX)
            }
        } else {
            x.mid()
        };
        [interval!(a, mid).unwrap(), interval!(mid, b).unwrap()]
            .into_iter()
            .filter(|x| !x.is_singleton())
            .map(Self)
            .collect()
    }
}

/// The direction of subdivision.
#[repr(u8)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SubdivisionDir {
    XY = 0,
    NTheta = 1,
    T = 2,
}

/// A subset of the domain of a relation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Block {
    /// The horizontal coordinate.
    pub x: Coordinate,
    /// The vertical coordinate.
    pub y: Coordinate,
    /// The parameter n_θ for polar coordinates.
    pub n_theta: IntegerParameter,
    /// The parameter t.
    pub t: RealParameter,
    /// The direction that should be chosen when subdividing this block.
    // TODO: It is awkward that a block has this field.
    pub next_dir: SubdivisionDir,
}

impl Block {
    /// Returns the pixel-level block that contains the given block.
    ///
    /// Panics if `self` is a superpixel.
    pub fn pixel_block(&self) -> Self {
        assert!(!self.x.is_superpixel() && !self.y.is_superpixel());
        let pixel = self.pixel_index();
        Self {
            x: Coordinate::new(pixel.x as u64, 0),
            y: Coordinate::new(pixel.y as u64, 0),
            ..*self
        }
    }

    /// Returns the index of the pixel that contains the block.
    /// If the block spans multiple pixels, the least index is returned.
    pub fn pixel_index(&self) -> PixelIndex {
        PixelIndex::new(self.x.pixel_index(), self.y.pixel_index())
    }
}

impl Default for Block {
    fn default() -> Self {
        Self {
            x: Coordinate { i: 0, k: 0 },
            y: Coordinate { i: 0, k: 0 },
            n_theta: IntegerParameter(Interval::ENTIRE),
            t: RealParameter(Interval::ENTIRE),
            next_dir: SubdivisionDir::XY,
        }
    }
}

/// Specifies which fields of [`Block`]s should be stored in a [`BlockQueue`].
#[derive(Clone, Debug, Default)]
pub struct BlockQueueOptions {
    pub store_xy: bool,
    pub store_n_theta: bool,
    pub store_t: bool,
    pub store_next_dir: bool,
}

/// A queue that stores [`Block`]s.
///
/// The [`Block`]s are entropy-encoded internally so that the closer the indices of consecutive
/// blocks are (which is expected by using the Morton order), the less memory it consumes.
pub struct BlockQueue {
    opts: BlockQueueOptions,
    seq: VecDeque<u8>,
    begin_index: usize,
    end_index: usize,
    x_front: u64,
    x_back: u64,
    y_front: u64,
    y_back: u64,
    n_theta_front: Interval,
    n_theta_back: Interval,
    t_front: Interval,
    t_back: Interval,
}

impl BlockQueue {
    /// Creates an empty queue.
    pub fn new(opts: BlockQueueOptions) -> Self {
        Self {
            opts,
            seq: VecDeque::new(),
            begin_index: 0,
            end_index: 0,
            x_front: 0,
            x_back: 0,
            y_front: 0,
            y_back: 0,
            n_theta_front: Interval::ENTIRE,
            n_theta_back: Interval::ENTIRE,
            t_front: Interval::ENTIRE,
            t_back: Interval::ENTIRE,
        }
    }

    /// Returns the index of the first block in the queue.
    ///
    /// Initially, the index is zero, and is incremented by and only by calling to [`Self::pop_front`].
    /// Therefore, the index is tied to a block in the queue and never reused for another block.
    ///
    /// You can obtain the index of the block right **after** it is returned by [`Self::pop_front`],
    /// by `queue.begin_index() - 1`. Beware the off-by-one error.
    pub fn begin_index(&self) -> usize {
        self.begin_index
    }

    /// Returns the index of one past the last block in the queue.
    ///
    /// Initially, the index is zero, and is incremented by and only by calling to [`Self::push_back`].
    ///
    /// You can obtain the index of the block right **after** it is passed to [`Self::push_back`],
    /// by `queue.end_index() - 1`. Beware the off-by-one error.
    ///
    /// See also [`Self::begin_index`].
    pub fn end_index(&self) -> usize {
        self.end_index
    }

    /// Returns `true` if the queue is empty.
    pub fn is_empty(&self) -> bool {
        self.seq.is_empty()
    }

    /// Removes the first block from the queue and returns it.
    /// [`None`] is returned if the queue is empty.
    pub fn pop_front(&mut self) -> Option<Block> {
        let (x, y) = if self.opts.store_xy {
            let x = self.x_front ^ self.pop_small_u64()?;
            self.x_front = x;

            let y = self.y_front ^ self.pop_small_u64()?;
            self.y_front = y;

            let kx = self.pop_i8()?;
            let ky = self.pop_i8()?;

            (Coordinate { i: x, k: kx }, Coordinate { i: y, k: ky })
        } else {
            (Coordinate { i: 0, k: 0 }, Coordinate { i: 0, k: 0 })
        };

        let n_theta = IntegerParameter(if self.opts.store_n_theta {
            self.n_theta_front = self.pop_interval(self.n_theta_front)?;
            self.n_theta_front
        } else {
            Interval::ENTIRE
        });

        let t = RealParameter(if self.opts.store_t {
            self.t_front = self.pop_interval(self.t_front)?;
            self.t_front
        } else {
            Interval::ENTIRE
        });

        let next_dir = if self.opts.store_next_dir {
            self.pop_subdivision_dir()?
        } else {
            SubdivisionDir::XY
        };

        self.begin_index += 1;

        Some(Block {
            x,
            y,
            n_theta,
            t,
            next_dir,
        })
    }

    /// Appends the block to the back of the queue.
    pub fn push_back(&mut self, b: Block) {
        if self.opts.store_xy {
            self.push_small_u64(b.x.i ^ self.x_back);
            self.x_back = b.x.i;

            self.push_small_u64(b.y.i ^ self.y_back);
            self.y_back = b.y.i;

            self.push_i8(b.x.k);
            self.push_i8(b.y.k);
        }

        if self.opts.store_n_theta {
            let n_theta = b.n_theta.interval();
            self.push_interval(n_theta, self.n_theta_back);
            self.n_theta_back = n_theta;
        }

        if self.opts.store_t {
            let t = b.t.interval();
            self.push_interval(t, self.t_back);
            self.t_back = t;
        }

        if self.opts.store_next_dir {
            self.push_subdivision_dir(b.next_dir);
        }

        self.end_index += 1;
    }

    /// Returns the approximate size allocated by the [`BlockQueue`] in bytes.
    pub fn size_in_heap(&self) -> usize {
        self.seq.capacity() * size_of::<u8>()
    }

    fn pop_i8(&mut self) -> Option<i8> {
        Some(self.seq.pop_front()? as i8)
    }

    fn pop_interval(&mut self, front: Interval) -> Option<Interval> {
        if self.seq.is_empty() {
            return None;
        }

        let mut bytes = [0u8; 16];
        for (src, dst) in self.seq.drain(..2).zip(bytes.iter_mut()) {
            *dst = src;
        }
        if (bytes[0], bytes[1]) != (0xff, 0xff) {
            for (src, dst) in self.seq.drain(..14).zip(bytes.iter_mut().skip(2)) {
                *dst = src;
            }
            Some(Interval::try_from_be_bytes(bytes).unwrap())
        } else {
            Some(front)
        }
    }

    // `u64` values up to 2^56 - 1 are encoded with PrefixVarint[1,2] so that smaller numbers take less space:
    //
    //    Range  `zeros`  Encoded bytes in `seq`
    //   ------  -------  ------------------------------------------------------------
    //                       6     0 -- Bit place in the original number
    //    < 2^7        0  [0bxxxxxxx1]
    //                       5    0     13      6
    //   < 2^14        1  [0bxxxxxx10, 0byyyyyyyy]
    //                       4   0      12      5   20     13
    //   < 2^21        2  [0bxxxxx100, 0byyyyyyyy, 0byyyyyyyy]
    //                       3  0       11      4   19     12   27     20
    //   < 2^28        3  [0bxxxx1000, 0byyyyyyyy, 0byyyyyyyy, 0byyyyyyyy]
    //                       2 0        10      3   18     11   26     19   34     27
    //   < 2^35        4  [0bxxx10000, 0byyyyyyyy, 0byyyyyyyy, 0byyyyyyyy, 0byyyyyyyy]
    //                      1┐┌0         9      2   17     10               41     34
    //   < 2^42        5  [0bxx100000, 0byyyyyyyy, 0byyyyyyyy,  «2 bytes», 0byyyyyyyy]
    //                       0           8      1   16      9               48     41
    //   < 2^49        6  [0bx1000000, 0byyyyyyyy, 0byyyyyyyy,  «3 bytes», 0byyyyyyyy]
    //                                   7      0   15      8               55     48
    //   < 2^56        7  [0b10000000, 0byyyyyyyy, 0byyyyyyyy,  «4 bytes», 0byyyyyyyy]
    //                 |               -----------------------v----------------------
    //                 |               Padded zeros to the right, these bytes can be
    //                 |               interpreted as a `u64` value in little endian.
    //                 The number of trailing zeros in the first byte.
    //
    // [1]: https://github.com/stoklund/varint#prefixvarint
    // [2]: https://news.ycombinator.com/item?id=11263667
    fn pop_small_u64(&mut self) -> Option<u64> {
        let head = self.seq.pop_front()?;
        let zeros = head.trailing_zeros();
        let tail_len = zeros as usize;
        let (tail1, tail2) = {
            let (mut t1, mut t2) = self.seq.as_slices();
            t1 = &t1[..tail_len.min(t1.len())];
            t2 = &t2[..(tail_len - t1.len())];
            (t1, t2)
        };
        // Shift twice to avoid overflow by `head >> 8`.
        let x = ((head >> zeros) >> 1) as u64;
        let y = {
            let mut y = 0u64;
            let y_ptr = &mut y as *mut u64 as *mut u8;
            unsafe {
                copy_nonoverlapping(tail1.as_ptr(), y_ptr, tail1.len());
                copy_nonoverlapping(tail2.as_ptr(), y_ptr.add(tail1.len()), tail2.len());
            }
            y = u64::from_le(y);
            y << (7 - zeros)
        };
        self.seq.drain(..tail_len);
        Some(x | y)
    }

    fn pop_subdivision_dir(&mut self) -> Option<SubdivisionDir> {
        let axis = match self.seq.pop_front()? {
            0 => SubdivisionDir::XY,
            1 => SubdivisionDir::NTheta,
            2 => SubdivisionDir::T,
            _ => panic!(),
        };
        Some(axis)
    }

    fn push_i8(&mut self, x: i8) {
        self.seq.push_back(x as u8);
    }

    fn push_interval(&mut self, x: Interval, back: Interval) {
        if x != back {
            self.seq.extend(x.to_be_bytes());
        } else {
            // A `f64` datum that starts with 0xffff is NaN, which never appears in interval bounds.
            self.seq.extend([0xff, 0xff]);
        }
    }

    fn push_small_u64(&mut self, x: u64) {
        assert!(x <= 0xffffffffffffff);
        let zeros = match x {
            0..=0x7f => 0,
            0x80..=0x3fff => 1,
            0x4000..=0x1fffff => 2,
            0x200000..=0xfffffff => 3,
            0x10000000..=0x7ffffffff => 4,
            0x800000000..=0x3ffffffffff => 5,
            0x40000000000..=0x1ffffffffffff => 6,
            0x2000000000000..=0xffffffffffffff => 7,
            _ => unreachable!(),
        };
        self.seq.push_back((((x << 1) | 0x1) << zeros) as u8);
        let y = x >> (7 - zeros);
        let tail_len = zeros;
        self.seq.extend(y.to_le_bytes()[..tail_len].iter());
    }

    fn push_subdivision_dir(&mut self, axis: SubdivisionDir) {
        self.seq.push_back(axis as u8);
    }
}

impl Extend<Block> for BlockQueue {
    fn extend<T: IntoIterator<Item = Block>>(&mut self, iter: T) {
        for b in iter {
            self.push_back(b);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use inari::const_interval;

    #[test]
    fn block() {
        let b = Block {
            x: Coordinate::new(42, 3),
            y: Coordinate::new(42, 5),
            ..Block::default()
        };
        assert_eq!(b.x.width(), 8);
        assert_eq!(b.y.width(), 32);
        assert_eq!(b.x.widthf(), 8.0);
        assert_eq!(b.y.widthf(), 32.0);
        assert_eq!(b.pixel_index(), PixelIndex::new(336, 1344));
        assert!(b.x.is_superpixel());
        assert!(b.y.is_superpixel());
        assert!(!b.x.is_subpixel());
        assert!(!b.y.is_subpixel());

        let b = Block {
            x: Coordinate::new(42, 0),
            y: Coordinate::new(42, 0),
            ..Block::default()
        };
        assert_eq!(b.x.width(), 1);
        assert_eq!(b.y.width(), 1);
        assert_eq!(b.x.widthf(), 1.0);
        assert_eq!(b.y.widthf(), 1.0);
        assert_eq!(b.x.pixel_align(), 1);
        assert_eq!(b.y.pixel_align(), 1);
        assert_eq!(b.pixel_block(), b);
        assert_eq!(b.pixel_index(), PixelIndex::new(42, 42));
        assert!(!b.x.is_superpixel());
        assert!(!b.y.is_superpixel());
        assert!(!b.x.is_subpixel());
        assert!(!b.y.is_subpixel());

        let b = Block {
            x: Coordinate::new(42, -3),
            y: Coordinate::new(42, -5),
            ..Block::default()
        };
        assert_eq!(b.x.widthf(), 0.125);
        assert_eq!(b.y.widthf(), 0.03125);
        assert_eq!(b.x.pixel_align(), 8);
        assert_eq!(b.y.pixel_align(), 32);
        assert_eq!(
            b.pixel_block(),
            Block {
                x: Coordinate::new(5, 0),
                y: Coordinate::new(1, 0),
                ..Block::default()
            }
        );
        assert_eq!(b.pixel_index(), PixelIndex::new(5, 1));
        assert!(!b.x.is_superpixel());
        assert!(!b.y.is_superpixel());
        assert!(b.x.is_subpixel());
        assert!(b.y.is_subpixel());
    }

    #[test]
    fn block_queue() {
        let mut queue = BlockQueue::new(BlockQueueOptions {
            store_xy: true,
            ..Default::default()
        });
        let blocks = [
            Block {
                x: Coordinate::new(0, -32),
                y: Coordinate::new(0xffffffffffffff, 0),
                ..Block::default()
            },
            Block {
                x: Coordinate::new(0x7f, 64),
                y: Coordinate::new(0x2000000000000, 127),
                ..Block::default()
            },
            Block {
                x: Coordinate::new(0x80, 0),
                y: Coordinate::new(0x1ffffffffffff, 0),
                ..Block::default()
            },
            Block {
                x: Coordinate::new(0x3fff, 0),
                y: Coordinate::new(0x40000000000, 0),
                ..Block::default()
            },
            Block {
                x: Coordinate::new(0x4000, 0),
                y: Coordinate::new(0x3ffffffffff, 0),
                ..Block::default()
            },
            Block {
                x: Coordinate::new(0x1fffff, 0),
                y: Coordinate::new(0x800000000, 0),
                ..Block::default()
            },
            Block {
                x: Coordinate::new(0x200000, 0),
                y: Coordinate::new(0x7ffffffff, 0),
                ..Block::default()
            },
            Block {
                x: Coordinate::new(0xfffffff, 0),
                y: Coordinate::new(0x10000000, 0),
                ..Block::default()
            },
            Block {
                x: Coordinate::new(0x10000000, 0),
                y: Coordinate::new(0xfffffff, 0),
                ..Block::default()
            },
            Block {
                x: Coordinate::new(0x7ffffffff, 0),
                y: Coordinate::new(0x200000, 0),
                ..Block::default()
            },
            Block {
                x: Coordinate::new(0x800000000, 0),
                y: Coordinate::new(0x1fffff, 0),
                ..Block::default()
            },
            Block {
                x: Coordinate::new(0x3ffffffffff, 0),
                y: Coordinate::new(0x4000, 0),
                ..Block::default()
            },
            Block {
                x: Coordinate::new(0x40000000000, 0),
                y: Coordinate::new(0x3fff, 0),
                ..Block::default()
            },
            Block {
                x: Coordinate::new(0x1ffffffffffff, 0),
                y: Coordinate::new(0x80, 0),
                ..Block::default()
            },
            Block {
                x: Coordinate::new(0x2000000000000, 0),
                y: Coordinate::new(0x7f, 0),
                ..Block::default()
            },
            Block {
                x: Coordinate::new(0xffffffffffffff, 0),
                y: Coordinate::new(0, 0),
                ..Block::default()
            },
        ];
        assert_eq!(queue.begin_index(), 0);
        assert_eq!(queue.end_index(), 0);
        for (i, b) in blocks.iter().cloned().enumerate() {
            queue.push_back(b);
            assert_eq!(queue.begin_index(), 0);
            assert_eq!(queue.end_index(), i + 1);
        }
        for (i, b) in blocks.iter().cloned().enumerate() {
            assert_eq!(queue.pop_front(), Some(b));
            assert_eq!(queue.begin_index(), i + 1);
            assert_eq!(queue.end_index(), blocks.len());
        }
        assert_eq!(queue.pop_front(), None);
        assert_eq!(queue.begin_index(), blocks.len());
        assert_eq!(queue.end_index(), blocks.len());

        let mut queue = BlockQueue::new(BlockQueueOptions {
            store_xy: true,
            store_n_theta: true,
            store_next_dir: true,
            ..Default::default()
        });
        let b1 = Block {
            n_theta: IntegerParameter::new(const_interval!(-2.0, 3.0)),
            ..Block::default()
        };
        let b2 = Block {
            next_dir: SubdivisionDir::NTheta,
            ..b1
        };
        queue.push_back(b1.clone());
        queue.push_back(b2.clone());
        assert_eq!(queue.pop_front(), Some(b1));
        assert_eq!(queue.pop_front(), Some(b2));

        let mut queue = BlockQueue::new(BlockQueueOptions {
            store_xy: true,
            store_t: true,
            store_next_dir: true,
            ..Default::default()
        });
        let b1 = Block {
            t: RealParameter::new(const_interval!(-2.0, 3.0)),
            ..Block::default()
        };
        let b2 = Block {
            next_dir: SubdivisionDir::T,
            ..b1
        };
        queue.push_back(b1.clone());
        queue.push_back(b2.clone());
        assert_eq!(queue.pop_front(), Some(b1));
        assert_eq!(queue.pop_front(), Some(b2));
    }
}
