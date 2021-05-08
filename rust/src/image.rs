use std::{collections::VecDeque, mem::size_of, ptr::copy_nonoverlapping};

/// The limit of the width/height of [`Image`]s in pixels.
const MAX_IMAGE_WIDTH: u32 = 32768;

/// The level of the smallest subdivision.
///
/// The value is currently fixed, but it could be determined based on the size of the image.
const MIN_K: i8 = -15;

/// The graphing status of a pixel.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PixelState {
    /// There may be or may not be a solution in the pixel.
    Uncertain,
    /// There are no solutions in the pixel.
    False,
    /// There is at least one solution in the pixel.
    True,
}

/// The index of an [`ImageBlock`] in an [`ImageBlockQueue`].
///
/// Indices returned by [`ImageBlockQueue`] are `usize`, but `u32` would be large enough.
pub type QueuedBlockIndex = u32;

/// A rendering of a graph. Each pixel stores the existence or absence of the solution:
#[derive(Debug)]
pub struct Image {
    pub width: u32,
    pub height: u32,
    last_queued_blocks: Vec<QueuedBlockIndex>,
    pub states: Vec<PixelState>,
}

impl Image {
    /// Creates a new [`Image`] with all pixels set to [`PixelStat::Uncertain`].
    pub fn new(width: u32, height: u32) -> Self {
        assert!(width > 0 && width <= MAX_IMAGE_WIDTH && height > 0 && height <= MAX_IMAGE_WIDTH);
        Self {
            width,
            height,
            last_queued_blocks: vec![0; height as usize * width as usize],
            states: vec![PixelState::Uncertain; height as usize * width as usize],
        }
    }

    /// Returns the index of the last-queued block of the pixel.
    pub fn last_queued_block(&self, p: PixelIndex) -> QueuedBlockIndex {
        self.last_queued_blocks[self.index(p)]
    }

    /// Returns a mutable reference to the index of the last-queued block of the pixel.
    pub fn last_queued_block_mut(&mut self, p: PixelIndex) -> &mut QueuedBlockIndex {
        let i = self.index(p);
        &mut self.last_queued_blocks[i]
    }

    /// Returns the size allocated by the [`Image`] in bytes.
    pub fn size_in_heap(&self) -> usize {
        self.states.capacity() * size_of::<PixelState>()
            + self.last_queued_blocks.capacity() * size_of::<QueuedBlockIndex>()
    }

    /// Returns the graphing status of the pixel.
    pub fn state(&self, p: PixelIndex) -> PixelState {
        self.states[self.index(p)]
    }

    /// Returns a mutable reference to the graphing status of the pixel.
    pub fn state_mut(&mut self, p: PixelIndex) -> &mut PixelState {
        let i = self.index(p);
        &mut self.states[i]
    }

    /// Returns the flattened index of the pixel.
    fn index(&self, p: PixelIndex) -> usize {
        p.y as usize * self.width as usize + p.x as usize
    }
}

/// The index of a pixel of an [`Image`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PixelIndex {
    /// The horizontal index of the pixel.
    pub x: u32,
    /// The vertical index of the pixel.
    pub y: u32,
}

impl PixelIndex {
    /// Creates a new [`PixelIndex`] with the coordinates.
    pub fn new(x: u32, y: u32) -> Self {
        Self { x, y }
    }
}

/// A rectangular region of an [`Image`] with the following bounds in pixels:
/// `[x 2^kx, (x + 1) 2^kx] × [y 2^ky, (y + 1) 1^ky]`.
///
/// A block is said to be:
///
/// - a *superpixel* iff `∀k ∈ K : k ≥ 0 ∧ ∃k ∈ K : k > 0`,
/// - a *pixel* iff `∀k ∈ K : k = 0`,
/// - a *subpixel* iff `∀k ∈ K : k ≤ 0 ∧ ∃k ∈ K : k < 0`,
///
/// where `K = {kx, ky}`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ImageBlock {
    /// The horizontal index of the block in multiples of the block width.
    pub x: u32,
    /// The vertical index of the block in multiples of the block height.
    pub y: u32,
    /// The horizontal subdivision level.
    pub kx: i8,
    /// The vertical subdivision level.
    pub ky: i8,
}

impl ImageBlock {
    /// Creates a new block.
    pub fn new(x: u32, y: u32, kx: i8, ky: i8) -> Self {
        assert!(kx >= 0 && ky >= 0 || kx <= 0 && ky <= 0);
        Self { x, y, kx, ky }
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

    /// Returns `true` if the block is a pixel.
    pub fn is_pixel(&self) -> bool {
        self.kx == 0 && self.ky == 0
    }

    /// Returns `true` if the block can be divided both horizontally and vertically.
    pub fn is_subdivisible(&self) -> bool {
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

    /// Returns the width of a pixel in multiples of the block's width.
    ///
    /// Panics if `self.kx > 0`.
    pub fn pixel_align_x(&self) -> u32 {
        assert!(self.kx <= 0);
        1u32 << -self.kx
    }

    /// Returns the height of a pixel in multiples of the block's height.
    ///
    /// Panics if `self.ky > 0`.
    pub fn pixel_align_y(&self) -> u32 {
        assert!(self.ky <= 0);
        1u32 << -self.ky
    }

    /// Returns the pixel-level block that contains the given block.
    ///
    /// Panics if the block is a superpixel.
    pub fn pixel_block(&self) -> Self {
        assert!(!self.is_superpixel());
        let pixel = self.pixel_index();
        Self {
            x: pixel.x,
            y: pixel.y,
            kx: 0,
            ky: 0,
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

/// A queue for storing [`ImageBlock`]s.
/// The closer the indices of consecutive blocks are (which is expected by using the Morton order),
/// the less memory it consumes.
pub struct ImageBlockQueue {
    seq: VecDeque<u8>,
    x_front: u32,
    y_front: u32,
    x_back: u32,
    y_back: u32,
    front_index: usize,
    back_index: usize,
}

impl ImageBlockQueue {
    /// Creates an empty queue.
    pub fn new() -> Self {
        Self {
            seq: VecDeque::new(),
            x_front: 0,
            y_front: 0,
            x_back: 0,
            y_back: 0,
            front_index: 0,
            back_index: 0,
        }
    }

    /// Returns `true` if the queue is empty.
    pub fn is_empty(&self) -> bool {
        self.seq.is_empty()
    }

    /// Removes the first element and returns it with its original index
    /// if the queue is nonempty; otherwise `None`.
    pub fn pop_front(&mut self) -> Option<(usize, ImageBlock)> {
        let x = self.x_front ^ self.pop_small_u32()?;
        let y = self.y_front ^ self.pop_small_u32()?;
        let kx = self.pop_i8()?;
        let ky = self.pop_i8()?;
        self.x_front = x;
        self.y_front = y;
        let front_index = self.front_index;
        self.front_index += 1;
        Some((front_index, ImageBlock { x, y, kx, ky }))
    }

    /// Appends an element to the back of the queue and returns the unique index where it is stored.
    pub fn push_back(&mut self, b: ImageBlock) -> usize {
        self.push_small_u32(b.x ^ self.x_back);
        self.push_small_u32(b.y ^ self.y_back);
        self.push_i8(b.kx);
        self.push_i8(b.ky);
        self.x_back = b.x;
        self.y_back = b.y;
        let back_index = self.back_index;
        self.back_index += 1;
        back_index
    }

    /// Returns the approximate size allocated by the [`ImageBlockQueue`] in bytes.
    pub fn size_in_heap(&self) -> usize {
        self.seq.capacity() * size_of::<u8>()
    }

    fn pop_i8(&mut self) -> Option<i8> {
        Some(self.seq.pop_front()? as i8)
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

    fn push_i8(&mut self, x: i8) {
        self.seq.push_back(x as u8);
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_block() {
        let b = ImageBlock::new(42, 42, 3, 5);
        assert_eq!(b.width(), 8);
        assert_eq!(b.height(), 32);
        assert_eq!(b.widthf(), 8.0);
        assert_eq!(b.heightf(), 32.0);
        assert_eq!(b.pixel_index(), PixelIndex::new(336, 1344));
        assert!(b.is_superpixel());
        assert!(!b.is_pixel());
        assert!(!b.is_subpixel());

        let b = ImageBlock::new(42, 42, 0, 0);
        assert_eq!(b.width(), 1);
        assert_eq!(b.height(), 1);
        assert_eq!(b.widthf(), 1.0);
        assert_eq!(b.heightf(), 1.0);
        assert_eq!(b.pixel_align_x(), 1);
        assert_eq!(b.pixel_align_y(), 1);
        assert_eq!(b.pixel_block(), b);
        assert_eq!(b.pixel_index(), PixelIndex::new(42, 42));
        assert!(!b.is_superpixel());
        assert!(b.is_pixel());
        assert!(!b.is_subpixel());

        let b = ImageBlock::new(42, 42, -3, -5);
        assert_eq!(b.widthf(), 0.125);
        assert_eq!(b.heightf(), 0.03125);
        assert_eq!(b.pixel_align_x(), 8);
        assert_eq!(b.pixel_align_y(), 32);
        assert_eq!(b.pixel_block(), ImageBlock::new(5, 1, 0, 0));
        assert_eq!(b.pixel_index(), PixelIndex::new(5, 1));
        assert!(!b.is_superpixel());
        assert!(!b.is_pixel());
        assert!(b.is_subpixel());
    }

    #[test]
    fn image_block_queue() {
        let mut queue = ImageBlockQueue::new();
        let blocks = [
            ImageBlock {
                x: 0,
                y: 0xffffffff,
                kx: -128,
                ky: 127,
            },
            ImageBlock {
                x: 0x7f,
                y: 0x10000000,
                kx: -128,
                ky: 127,
            },
            ImageBlock {
                x: 0x80,
                y: 0xfffffff,
                kx: -127,
                ky: 64,
            },
            ImageBlock {
                x: 0x3fff,
                y: 0x200000,
                kx: -64,
                ky: 63,
            },
            ImageBlock {
                x: 0x4000,
                y: 0x1fffff,
                kx: -63,
                ky: 0,
            },
            ImageBlock {
                x: 0x1fffff,
                y: 0x4000,
                kx: 0,
                ky: -63,
            },
            ImageBlock {
                x: 0x200000,
                y: 0x3fff,
                kx: 63,
                ky: -64,
            },
            ImageBlock {
                x: 0xfffffff,
                y: 0x80,
                kx: 64,
                ky: -127,
            },
            ImageBlock {
                x: 0x10000000,
                y: 0x7f,
                kx: 127,
                ky: -128,
            },
            ImageBlock {
                x: 0xffffffff,
                y: 0,
                kx: -128,
                ky: 127,
            },
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
    }
}
