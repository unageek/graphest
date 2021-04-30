use crate::graph::ImageBlock;
use std::{collections::VecDeque, mem::size_of};

/// A queue for storing [`ImageBlock`]s.
/// The closer the indices of consecutive blocks are (which is expected by using the Morton order),
/// the less memory it consumes.
pub struct ImageBlockQueue {
    seq: VecDeque<u8>,
    x_front: u32,
    y_front: u32,
    x_back: u32,
    y_back: u32,
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
        }
    }

    /// Returns `true` if the queue is empty.
    pub fn is_empty(&self) -> bool {
        self.seq.is_empty()
    }

    /// Removes the first element and returns it if the queue is nonempty; otherwise `None`.
    pub fn pop_front(&mut self) -> Option<ImageBlock> {
        let x = self.x_front ^ self.pop_small_u32()?;
        let y = self.y_front ^ self.pop_small_u32()?;
        let kx = self.pop_i8()?;
        let ky = self.pop_i8()?;
        self.x_front = x;
        self.y_front = y;
        Some(ImageBlock { x, y, kx, ky })
    }

    /// Appends an element to the back of the queue.
    pub fn push_back(&mut self, b: ImageBlock) {
        self.push_small_u32(b.x ^ self.x_back);
        self.push_small_u32(b.y ^ self.y_back);
        self.push_i8(b.kx);
        self.push_i8(b.ky);
        self.x_back = b.x;
        self.y_back = b.y;
    }

    /// Returns the approximate size in bytes allocated by the [`ImageBlockQueue`].
    pub fn size_in_heap(&self) -> usize {
        self.seq.capacity() * size_of::<u8>()
    }

    fn pop_i8(&mut self) -> Option<i8> {
        Some(self.seq.pop_front()? as i8)
    }

    fn pop_small_u32(&mut self) -> Option<u32> {
        let mut byte = self.seq.pop_front()?;
        let mut shift = 0;
        let mut x = (byte & 0x7f) as u32;
        while byte & 0x80 != 0 {
            byte = self.seq.pop_front()?;
            shift += 7;
            x |= ((byte & 0x7f) as u32) << shift;
        }
        Some(x)
    }

    fn push_i8(&mut self, x: i8) {
        self.seq.push_back(x as u8);
    }

    // The value is divided into 7-bit chunks and only the least significant chunks are stored.
    // For example, a 16-bit value is stored as:
    //
    //   0b00_0000000_00000xx → [0b0_00000xx]
    //   0b00_0000000_xxxxxxx → [0b0_xxxxxxx]
    //   0b00_yyyyyyy_xxxxxxx → [0b1_xxxxxxx, 0b0_yyyyyyy]
    //   0bzz_yyyyyyy_xxxxxxx → [0b1_xxxxxxx, 0b1_yyyyyyy, 0b0_00000zz]
    fn push_small_u32(&mut self, mut x: u32) {
        while x >= 0x80 {
            self.seq.push_back(0x80 | (x & 0x7f) as u8);
            x >>= 7;
        }
        self.seq.push_back(x as u8);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_pop() {
        let mut queue = ImageBlockQueue::new();
        let blocks = [
            ImageBlock {
                x: 0x7F,
                y: 0x10000000,
                kx: -128,
                ky: 127,
            },
            ImageBlock {
                x: 0x80,
                y: 0xFFFFFFF,
                kx: -127,
                ky: 64,
            },
            ImageBlock {
                x: 0x3FFF,
                y: 0x200000,
                kx: -64,
                ky: 63,
            },
            ImageBlock {
                x: 0x4000,
                y: 0x1FFFFF,
                kx: -63,
                ky: 0,
            },
            ImageBlock {
                x: 0x1FFFFF,
                y: 0x4000,
                kx: 0,
                ky: -63,
            },
            ImageBlock {
                x: 0x200000,
                y: 0x3FFF,
                kx: 63,
                ky: -64,
            },
            ImageBlock {
                x: 0xFFFFFFF,
                y: 0x80,
                kx: 64,
                ky: -127,
            },
            ImageBlock {
                x: 0x10000000,
                y: 0x7F,
                kx: 127,
                ky: -128,
            },
        ];
        for b in blocks.iter().copied() {
            queue.push_back(b);
        }
        for b in blocks.iter().copied() {
            assert_eq!(queue.pop_front().unwrap(), b);
        }
    }
}
