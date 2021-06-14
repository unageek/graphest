use crate::image::PixelIndex;
use inari::Interval;
use std::{collections::VecDeque, mem::size_of, ptr::copy_nonoverlapping};

/// The level of the smallest horizontal/vertical subdivision.
///
/// The value is currently fixed, but it could be determined based on the size of the image.
const MIN_K: i8 = -15;

/// The direction of subdivision.
#[repr(u8)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SubdivisionDir {
    XY = 0,
    NTheta = 1,
}

/// A rectangular region of an [`Image`](crate::image::Image) with the following bounds in pixels:
/// `[x 2^kx, (x + 1) 2^kx] × [y 2^ky, (y + 1) 1^ky]`.
/// It also contains additional information that are required by certain kinds of graphs.
///
/// A block is said to be:
///
/// - a *superpixel* iff `∀k ∈ K : k ≥ 0 ∧ ∃k ∈ K : k > 0`,
/// - a *pixel* iff `∀k ∈ K : k = 0`,
/// - a *subpixel* iff `∀k ∈ K : k ≤ 0 ∧ ∃k ∈ K : k < 0`,
///
/// where `K = {kx, ky}`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Block {
    /// The horizontal index of the block in multiples of the block width.
    pub x: u32,
    /// The vertical index of the block in multiples of the block height.
    pub y: u32,
    /// The horizontal subdivision level.
    pub kx: i8,
    /// The vertical subdivision level.
    pub ky: i8,
    /// The parameter n_θ for polar coordinates.
    pub n_theta: Interval,
    /// The direction that should be chosen when subdividing this block.
    pub next_dir: SubdivisionDir,
}

impl Block {
    /// Creates a new block.
    pub fn new(x: u32, y: u32, kx: i8, ky: i8, n_theta: Interval) -> Self {
        assert!(
            (kx >= 0 && ky >= 0 || kx <= 0 && ky <= 0)
                && !n_theta.is_empty()
                && n_theta == n_theta.trunc()
        );
        Self {
            x,
            y,
            kx,
            ky,
            n_theta,
            next_dir: SubdivisionDir::XY,
        }
    }

    /// Returns the height of the block in pixels.
    ///
    /// Panics if `self.ky < 0`.
    pub fn height(&self) -> u32 {
        assert!(self.ky >= 0);
        1u32 << self.ky
    }

    /// Returns the height of the block in pixels.
    pub fn heightf(&self) -> f64 {
        Self::exp2(self.ky)
    }

    /// Returns `true` if [`self.n_theta`] can be subdivided.
    pub fn is_subdivisible_on_n_theta(&self) -> bool {
        let n = self.n_theta;
        let mid = n.mid().round();
        n.inf() != mid && n.sup() != mid
    }

    /// Returns `true` if the block can be subdivided both horizontally and vertically.
    pub fn is_subdivisible_on_xy(&self) -> bool {
        self.kx > MIN_K && self.ky > MIN_K
    }

    /// Returns `true` if the block is a subpixel.
    pub fn is_subpixel(&self) -> bool {
        self.kx < 0 || self.ky < 0
    }

    /// Returns `true` if the block is a superpixel.
    pub fn is_superpixel(&self) -> bool {
        self.kx > 0 || self.ky > 0
    }

    /// Returns the width of a pixel divided by the block's width.
    ///
    /// Panics if `self.kx > 0`.
    pub fn pixel_align_x(&self) -> u32 {
        assert!(self.kx <= 0);
        1u32 << -self.kx
    }

    /// Returns the height of a pixel divided by the block's height.
    ///
    /// Panics if `self.ky > 0`.
    pub fn pixel_align_y(&self) -> u32 {
        assert!(self.ky <= 0);
        1u32 << -self.ky
    }

    /// Returns the pixel-level block that contains the given block.
    ///
    /// Panics if `self` is a superpixel.
    pub fn pixel_block(&self) -> Self {
        assert!(!self.is_superpixel());
        let pixel = self.pixel_index();
        Self {
            x: pixel.x,
            y: pixel.y,
            kx: 0,
            ky: 0,
            ..*self
        }
    }

    /// Returns the index of the pixel that contains the block.
    /// If the block spans multiple pixels, the least index is returned.
    pub fn pixel_index(&self) -> PixelIndex {
        PixelIndex::new(
            if self.kx >= 0 {
                self.x << self.kx
            } else {
                self.x >> -self.kx
            },
            if self.ky >= 0 {
                self.y << self.ky
            } else {
                self.y >> -self.ky
            },
        )
    }

    /// Returns the width of the block in pixels.
    ///
    /// Panics if `self.kx < 0`.
    pub fn width(&self) -> u32 {
        assert!(self.kx >= 0);
        1u32 << self.kx
    }

    /// Returns the width of the block in pixels.
    pub fn widthf(&self) -> f64 {
        Self::exp2(self.kx)
    }

    /// Returns `2^k`.
    fn exp2(k: i8) -> f64 {
        f64::from_bits(((1023 + k as i32) as u64) << 52)
    }
}

/// A queue that stores [`Block`]s.
///
/// The [`Block`]s are entropy-encoded internally so that the closer the indices of consecutive
/// blocks are (which is expected by using the Morton order), the less memory it consumes.
pub struct BlockQueue {
    seq: VecDeque<u8>,
    x_front: u32,
    x_back: u32,
    y_front: u32,
    y_back: u32,
    n_theta_front: Interval,
    n_theta_back: Interval,
    front_index: usize,
    back_index: usize,
    polar: bool,
}

impl BlockQueue {
    /// Creates an empty queue.
    pub fn new(polar: bool) -> Self {
        Self {
            seq: VecDeque::new(),
            x_front: 0,
            x_back: 0,
            y_front: 0,
            y_back: 0,
            n_theta_front: Interval::EMPTY,
            n_theta_back: Interval::EMPTY,
            front_index: 0,
            back_index: 0,
            polar,
        }
    }

    /// Returns `true` if the queue is empty.
    pub fn is_empty(&self) -> bool {
        self.seq.is_empty()
    }

    /// Returns the index that will be returned by the next call of `self.push_back`.
    pub fn next_back_index(&self) -> usize {
        self.back_index
    }

    /// Removes the first block from the queue and returns it with its original index.
    /// It returns `None` if the queue is empty.
    pub fn pop_front(&mut self) -> Option<(usize, Block)> {
        let x = self.x_front ^ self.pop_small_u32()?;
        let y = self.y_front ^ self.pop_small_u32()?;
        let kx = self.pop_i8()?;
        let ky = self.pop_i8()?;
        let (n_theta, axis) = if self.polar {
            (self.pop_n_theta()?, self.pop_subdivision_dir()?)
        } else {
            (Interval::ENTIRE, SubdivisionDir::XY)
        };
        self.x_front = x;
        self.y_front = y;
        let front_index = self.front_index;
        self.front_index += 1;
        Some((
            front_index,
            Block {
                x,
                y,
                kx,
                ky,
                n_theta,
                next_dir: axis,
            },
        ))
    }

    /// Appends the block to the back of the queue and returns the index where it is stored.
    pub fn push_back(&mut self, b: Block) -> usize {
        self.push_small_u32(b.x ^ self.x_back);
        self.push_small_u32(b.y ^ self.y_back);
        self.push_i8(b.kx);
        self.push_i8(b.ky);
        if self.polar {
            self.push_n_theta(b.n_theta);
            self.push_subdivision_dir(b.next_dir);
        }
        self.x_back = b.x;
        self.y_back = b.y;
        let back_index = self.back_index;
        self.back_index += 1;
        back_index
    }

    /// Returns the approximate size allocated by the [`BlockQueue`] in bytes.
    pub fn size_in_heap(&self) -> usize {
        self.seq.capacity() * size_of::<u8>()
    }

    fn pop_i8(&mut self) -> Option<i8> {
        Some(self.seq.pop_front()? as i8)
    }

    fn pop_n_theta(&mut self) -> Option<Interval> {
        let mut bytes = [0u8; 16];
        for (src, dst) in self.seq.drain(..2).zip(bytes.iter_mut()) {
            *dst = src;
        }
        if bytes[0] != 0xff || bytes[1] != 0xff {
            for (src, dst) in self.seq.drain(..14).zip(bytes.iter_mut().skip(2)) {
                *dst = src;
            }
            self.n_theta_front = Interval::try_from_be_bytes(bytes).unwrap();
        }
        Some(self.n_theta_front)
    }

    // PrefixVarint[1,2] is used to encode unsigned numbers:
    //
    //    Range   `zeros`  Encoded bytes in `seq`
    //   ------  --------  ----------------------------------------------------------------------
    //                        6     0 -- Bit place in the original number
    //    < 2^7         0  [0bxxxxxxx1]
    //                        5    0     13      6
    //   < 2^14         1  [0bxxxxxx10, 0byyyyyyyy]
    //                        4   0      12      5   20     13
    //   < 2^21         2  [0bxxxxx100, 0byyyyyyyy, 0byyyyyyyy]
    //                        3  0       11      4   19     12   27     20
    //   < 2^28         3  [0bxxxx1000, 0byyyyyyyy, 0byyyyyyyy, 0byyyyyyyy]
    //                        2 0        10      3   18     11   26     19      31  27
    //   < 2^32         4  [0bxxx10000, 0byyyyyyyy, 0byyyyyyyy, 0byyyyyyyy, 0b000yyyyy]
    //                  |               -----------------------v----------------------
    //                  |               These bytes can be interpreted as a part of a `u32` value
    //                  |               in little endian.
    //                  The number of trailing zeros in the first byte.
    //
    // [1]: https://github.com/stoklund/varint#prefixvarint
    // [2]: https://news.ycombinator.com/item?id=11263667
    fn pop_small_u32(&mut self) -> Option<u32> {
        let head = self.seq.pop_front()?;
        let zeros = head.trailing_zeros();
        let tail_len = zeros as usize;
        let (tail1, tail2) = {
            let (mut t1, mut t2) = self.seq.as_slices();
            t1 = &t1[..tail_len.min(t1.len())];
            t2 = &t2[..(tail_len - t1.len())];
            (t1, t2)
        };
        let x = (head >> (zeros + 1)) as u32;
        let y = {
            let mut y = 0u32;
            let y_ptr = &mut y as *mut u32 as *mut u8;
            unsafe {
                copy_nonoverlapping(tail1.as_ptr(), y_ptr, tail1.len());
                copy_nonoverlapping(tail2.as_ptr(), y_ptr.add(tail1.len()), tail2.len());
            }
            y = u32::from_le(y);
            y << (7 - zeros)
        };
        self.seq.drain(..tail_len);
        Some(x | y)
    }

    fn pop_subdivision_dir(&mut self) -> Option<SubdivisionDir> {
        let axis = match self.seq.pop_front()? {
            0 => SubdivisionDir::XY,
            1 => SubdivisionDir::NTheta,
            _ => panic!(),
        };
        Some(axis)
    }

    fn push_i8(&mut self, x: i8) {
        self.seq.push_back(x as u8);
    }

    fn push_n_theta(&mut self, x: Interval) {
        if x == self.n_theta_back {
            // A `f64` that starts with 0xffff is NaN, which never appears in interval bounds.
            self.seq.extend([0xff, 0xff]);
        } else {
            self.seq.extend(x.to_be_bytes());
            self.n_theta_back = x;
        }
    }

    fn push_small_u32(&mut self, x: u32) {
        let zeros = match x {
            0..=0x7f => 0,
            0x80..=0x3fff => 1,
            0x4000..=0x1fffff => 2,
            0x200000..=0xfffffff => 3,
            0x10000000..=0xffffffff => 4,
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

#[cfg(test)]
mod tests {
    use super::*;
    use inari::const_interval;

    #[test]
    fn block() {
        let b = Block::new(42, 42, 3, 5, Interval::ENTIRE);
        assert_eq!(b.width(), 8);
        assert_eq!(b.height(), 32);
        assert_eq!(b.widthf(), 8.0);
        assert_eq!(b.heightf(), 32.0);
        assert_eq!(b.pixel_index(), PixelIndex::new(336, 1344));
        assert!(b.is_superpixel());
        assert!(!b.is_subpixel());

        let b = Block::new(42, 42, 0, 0, Interval::ENTIRE);
        assert_eq!(b.width(), 1);
        assert_eq!(b.height(), 1);
        assert_eq!(b.widthf(), 1.0);
        assert_eq!(b.heightf(), 1.0);
        assert_eq!(b.pixel_align_x(), 1);
        assert_eq!(b.pixel_align_y(), 1);
        assert_eq!(b.pixel_block(), b);
        assert_eq!(b.pixel_index(), PixelIndex::new(42, 42));
        assert!(!b.is_superpixel());
        assert!(!b.is_subpixel());

        let b = Block::new(42, 42, -3, -5, Interval::ENTIRE);
        assert_eq!(b.widthf(), 0.125);
        assert_eq!(b.heightf(), 0.03125);
        assert_eq!(b.pixel_align_x(), 8);
        assert_eq!(b.pixel_align_y(), 32);
        assert_eq!(b.pixel_block(), Block::new(5, 1, 0, 0, Interval::ENTIRE));
        assert_eq!(b.pixel_index(), PixelIndex::new(5, 1));
        assert!(!b.is_superpixel());
        assert!(b.is_subpixel());
    }

    #[test]
    fn block_queue() {
        let mut queue = BlockQueue::new(false);
        let blocks = [
            Block::new(0, 0xffffffff, -128, -64, Interval::ENTIRE),
            Block::new(0x7f, 0x10000000, -32, 0, Interval::ENTIRE),
            Block::new(0x80, 0xfffffff, 0, 32, Interval::ENTIRE),
            Block::new(0x3fff, 0x200000, 64, 127, Interval::ENTIRE),
            Block::new(0x4000, 0x1fffff, 0, 0, Interval::ENTIRE),
            Block::new(0x1fffff, 0x4000, 0, 0, Interval::ENTIRE),
            Block::new(0x200000, 0x3fff, 0, 0, Interval::ENTIRE),
            Block::new(0xfffffff, 0x80, 0, 0, Interval::ENTIRE),
            Block::new(0x10000000, 0x7f, 0, 0, Interval::ENTIRE),
            Block::new(0xffffffff, 0, 0, 0, Interval::ENTIRE),
        ];
        for (i, b) in blocks.iter().copied().enumerate() {
            let back_index = queue.push_back(b);
            assert_eq!(back_index, i);
        }
        for (i, b) in blocks.iter().copied().enumerate() {
            let (front_index, front) = queue.pop_front().unwrap();
            assert_eq!(front_index, i);
            assert_eq!(front, b);
        }

        let mut queue = BlockQueue::new(true);
        let b1 = Block::new(0, 0, 0, 0, const_interval!(-2.0, 3.0));
        let b2 = Block {
            next_dir: SubdivisionDir::NTheta,
            ..b1
        };
        queue.push_back(b1);
        queue.push_back(b2);
        assert_eq!(queue.pop_front().unwrap().1, b1);
        assert_eq!(queue.pop_front().unwrap().1, b2);
    }
}
