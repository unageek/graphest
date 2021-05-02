use crate::graph::ImageBlock;
use std::{collections::VecDeque, mem::size_of, ptr::copy_nonoverlapping};

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

    // PrefixVarint[0][1] is used to encode unsigned numbers:
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
    // [0]: https://github.com/stoklund/varint#prefixvarint
    // [1]: https://news.ycombinator.com/item?id=11263667
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
            0x10000000..=0xFFFFFFFF => 4,
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
    fn push_pop() {
        let mut queue = ImageBlockQueue::new();
        let blocks = [
            ImageBlock {
                x: 0,
                y: 0xFFFFFFFF,
                kx: -128,
                ky: 127,
            },
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
            ImageBlock {
                x: 0xFFFFFFFF,
                y: 0,
                kx: -128,
                ky: 127,
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
