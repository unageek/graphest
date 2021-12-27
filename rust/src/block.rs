use crate::{traits::BytesAllocated, vars::VarSet};
use inari::{interval, Interval};
use itertools::Itertools;
use smallvec::SmallVec;
use std::{collections::VecDeque, ptr::copy_nonoverlapping};

/// A component of a [`Block`] that corresponds to the horizontal or vertical axis of an [`Image`].
///
/// A [`Coordinate`] with index `i` and level `k` represents the interval `[i 2^k, (i + 1) 2^k]`,
/// where the endpoints are in pixel coordinates.
///
/// [`Image`]: crate::image::Image
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Coordinate {
    i: u64,
    k: i8,
}

impl Coordinate {
    /// The smallest level of blocks.
    ///
    /// A smaller value can be used, as long as the following condition is met:
    ///
    /// - [`Image::MAX_WIDTH`]` / 2^`[`Self::MIN_LEVEL`]` ≤ 2^53`,
    ///
    /// which is required for keeping `block_to_region` operations exact.
    ///
    /// [`Image::MAX_WIDTH`]: crate::image::Image::MAX_WIDTH
    pub const MIN_LEVEL: i8 = -32;

    /// Creates a new [`Coordinate`].
    ///
    /// Panics if `level` is less than [`Self::MIN_LEVEL`].
    pub fn new(index: u64, level: i8) -> Self {
        assert!(level >= Self::MIN_LEVEL);
        Self { i: index, k: level }
    }

    /// Returns the index of the block in multiples of the block width.
    pub fn index(&self) -> u64 {
        self.i
    }

    /// Returns `true` if `self.level() < 0`, i.e., the block width is smaller than the pixel width.
    pub fn is_subpixel(&self) -> bool {
        self.k < 0
    }

    /// Returns `true` if `self.level() > 0`, i.e., the block width is larger than the pixel width.
    pub fn is_superpixel(&self) -> bool {
        self.k > 0
    }

    /// Returns `true` if the block can be subdivided.
    pub fn is_subdivisible(&self) -> bool {
        self.k > Self::MIN_LEVEL
    }

    /// Returns the level of the block.
    #[allow(dead_code)]
    pub fn level(&self) -> i8 {
        self.k
    }

    /// Returns the pixel-level block that contains the given block.
    /// If the block spans multiple pixels, the one with the least index is returned.
    pub fn pixel(&self) -> Self {
        Self {
            i: self.pixel_index() as u64,
            k: 0,
        }
    }

    /// Returns the pixel width divided by the block width.
    ///
    /// Panics if `self.level() > 0`.
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

    /// Returns the subdivided blocks.
    ///
    /// Two blocks are returned.
    ///
    /// Precondition: [`self.is_subdivisible()`] is `true`.
    pub fn subdivide(&self) -> [Self; 2] {
        let i0 = 2 * self.i;
        let i1 = i0 + 1;
        let k = self.k - 1;
        [Self { i: i0, k }, Self { i: i1, k }]
    }

    /// Returns the block width in pixels.
    ///
    /// Panics if `self.level() < 0`.
    pub fn width(&self) -> u32 {
        assert!(self.k >= 0);
        1u32 << self.k
    }

    /// Returns the block width in pixels.
    pub fn widthf(&self) -> f64 {
        Self::exp2(self.k)
    }

    /// Returns `2^k`.
    fn exp2(k: i8) -> f64 {
        f64::from_bits(((1023 + k as i32) as u64) << 52)
    }
}

fn subdivision_point(x: Interval, integer: bool) -> f64 {
    let a = x.inf();
    let b = x.sup();
    if a == f64::NEG_INFINITY {
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
    } else if integer {
        x.mid().round()
    } else {
        x.mid()
    }
}

/// A component of a [`Block`] that corresponds to a integer parameter.
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

    /// Creates a new [`IntegerParameter`].
    ///
    /// Panics if `x` is empty or an endpoint of `x` is a finite non-integer number.
    pub fn new(x: Interval) -> Self {
        assert!(!x.is_empty() && x == x.trunc());
        Self(x)
    }

    /// Returns the interval that the block spans.
    pub fn interval(&self) -> Interval {
        self.0
    }

    /// Returns `true` if the block can be subdivided.
    pub fn is_subdivisible(&self) -> bool {
        Self::is_subdivisible_impl(self.0)
    }

    fn is_subdivisible_impl(x: Interval) -> bool {
        let mid = x.mid().round();
        x.inf() != mid && x.sup() != mid
    }

    /// Returns the subdivided blocks.
    ///
    /// Three blocks are returned at most.
    ///
    /// Precondition: [`self.is_subdivisible()`] is `true`.
    pub fn subdivide(&self) -> SmallVec<[Self; 3]> {
        let x = self.0;
        let a = x.inf();
        let b = x.sup();
        let mid = subdivision_point(x, true);
        [
            interval!(a, mid).unwrap(),
            interval!(mid, mid).unwrap(),
            interval!(mid, b).unwrap(),
        ]
        .into_iter()
        .filter_map(|x| {
            let w = x.wid();
            if w == 1.0 {
                // Discard the interval since both of the endpoints are already taken
                // as point intervals and there is no integer between them.
                None
            } else if w == 2.0 && Self::is_subdivisible_impl(x) {
                let m = x.mid();
                Some(Self(interval!(m, m).unwrap()))
            } else {
                Some(Self(x))
            }
        })
        .collect()
    }
}

impl Default for IntegerParameter {
    fn default() -> Self {
        Self(Interval::ENTIRE)
    }
}

/// A component of a [`Block`] that corresponds to a real parameter.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RealParameter(Interval);

impl RealParameter {
    /// Creates a new [`RealParameter`].
    ///
    /// Panics if `x` is empty.
    pub fn new(x: Interval) -> Self {
        assert!(!x.is_empty());
        Self(x)
    }

    /// Returns the interval that the block spans.
    pub fn interval(&self) -> Interval {
        self.0
    }

    /// Returns `true` if the block can be subdivided.
    pub fn is_subdivisible(&self) -> bool {
        let x = self.0;
        let mid = x.mid();
        x.inf() != mid && x.sup() != mid
    }

    /// Returns the subdivided blocks.
    ///
    /// Two blocks are returned at most.
    ///
    /// Precondition: [`self.is_subdivisible()`] is `true`.
    pub fn subdivide(&self) -> SmallVec<[Self; 2]> {
        let x = self.0;
        let a = x.inf();
        let b = x.sup();
        let mid = subdivision_point(x, false);
        [interval!(a, mid).unwrap(), interval!(mid, b).unwrap()]
            .into_iter()
            .filter(|x| !x.is_singleton())
            .map(Self)
            .collect()
    }
}

impl Default for RealParameter {
    fn default() -> Self {
        Self(Interval::ENTIRE)
    }
}

/// A subset of the domain of a relation.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Block {
    /// The horizontal coordinate.
    pub x: Coordinate,
    /// The vertical coordinate.
    pub y: Coordinate,
    /// The parameter m.
    pub m: IntegerParameter,
    /// The parameter n.
    pub n: IntegerParameter,
    /// The parameter n_θ for polar coordinates.
    pub n_theta: IntegerParameter,
    /// The parameter t.
    pub t: RealParameter,
    /// The index of the subdivision direction, defined by the graphing algorithm,
    /// that should be chosen to subdivide this block.
    pub next_dir_index: u8,
}

const DEFAULT_BLOCK: Block = Block {
    x: Coordinate { i: 0, k: 0 },
    y: Coordinate { i: 0, k: 0 },
    m: IntegerParameter(Interval::ENTIRE),
    n: IntegerParameter(Interval::ENTIRE),
    n_theta: IntegerParameter(Interval::ENTIRE),
    t: RealParameter(Interval::ENTIRE),
    next_dir_index: 0,
};

/// A queue that stores [`Block`]s.
///
/// The [`Block`]s are entropy-encoded internally so that the closer the indices of consecutive
/// blocks are (which is expected by using the Morton order), the less memory it consumes.
pub struct BlockQueue {
    seq: VecDeque<u8>,
    store_vars: VarSet,
    begin_index: usize,
    end_index: usize,
    x_front: u64,
    x_back: u64,
    y_front: u64,
    y_back: u64,
    m_front: Interval,
    m_back: Interval,
    n_front: Interval,
    n_back: Interval,
    n_theta_front: Interval,
    n_theta_back: Interval,
    t_front: Interval,
    t_back: Interval,
}

impl BlockQueue {
    /// Creates an empty queue.
    pub fn new(store_vars: VarSet) -> Self {
        Self {
            seq: VecDeque::new(),
            store_vars,
            begin_index: 0,
            end_index: 0,
            x_front: 0,
            x_back: 0,
            y_front: 0,
            y_back: 0,
            m_front: Interval::ENTIRE,
            m_back: Interval::ENTIRE,
            n_front: Interval::ENTIRE,
            n_back: Interval::ENTIRE,
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
        // This is (somehow) faster than Block::default().
        let mut b = DEFAULT_BLOCK;

        if self.store_vars.contains(VarSet::X) {
            self.x_front ^= self.pop_small_u64()?;
            b.x = Coordinate {
                i: self.x_front,
                k: self.pop_i8()?,
            };
        }

        if self.store_vars.contains(VarSet::Y) {
            self.y_front ^= self.pop_small_u64()?;
            b.y = Coordinate {
                i: self.y_front,
                k: self.pop_i8()?,
            };
        }

        if self.store_vars.contains(VarSet::M) {
            self.m_front = self.pop_interval(self.m_front)?;
            b.m = IntegerParameter(self.m_front)
        }

        if self.store_vars.contains(VarSet::N) {
            self.n_front = self.pop_interval(self.n_front)?;
            b.n = IntegerParameter(self.n_front);
        }

        if self.store_vars.contains(VarSet::N_THETA) {
            self.n_theta_front = self.pop_interval(self.n_theta_front)?;
            b.n_theta = IntegerParameter(self.n_theta_front);
        }

        if self.store_vars.contains(VarSet::T) {
            self.t_front = self.pop_interval(self.t_front)?;
            b.t = RealParameter(self.t_front);
        }

        b.next_dir_index = self.pop_u8()?;

        self.begin_index += 1;

        Some(b)
    }

    /// Appends the block to the back of the queue.
    pub fn push_back(&mut self, b: Block) {
        if self.store_vars.contains(VarSet::X) {
            self.push_small_u64(b.x.i ^ self.x_back);
            self.push_i8(b.x.k);
            self.x_back = b.x.i;
        }

        if self.store_vars.contains(VarSet::Y) {
            self.push_small_u64(b.y.i ^ self.y_back);
            self.push_i8(b.y.k);
            self.y_back = b.y.i;
        }

        if self.store_vars.contains(VarSet::M) {
            let m = b.m.interval();
            self.push_interval(m, self.m_back);
            self.m_back = m;
        }

        if self.store_vars.contains(VarSet::N) {
            let n = b.n.interval();
            self.push_interval(n, self.n_back);
            self.n_back = n;
        }

        if self.store_vars.contains(VarSet::N_THETA) {
            let n_theta = b.n_theta.interval();
            self.push_interval(n_theta, self.n_theta_back);
            self.n_theta_back = n_theta;
        }

        if self.store_vars.contains(VarSet::T) {
            let t = b.t.interval();
            self.push_interval(t, self.t_back);
            self.t_back = t;
        }

        self.push_u8(b.next_dir_index);

        self.end_index += 1;
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

    fn pop_u8(&mut self) -> Option<u8> {
        self.seq.pop_front()
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

    fn push_u8(&mut self, x: u8) {
        self.seq.push_back(x)
    }
}

impl Extend<Block> for BlockQueue {
    fn extend<T: IntoIterator<Item = Block>>(&mut self, iter: T) {
        for b in iter {
            self.push_back(b);
        }
    }
}

impl BytesAllocated for BlockQueue {
    fn bytes_allocated(&self) -> usize {
        self.seq.bytes_allocated()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use inari::const_interval;
    use std::default::default;

    #[test]
    fn coordinate() {
        let x = Coordinate::new(42, 3);
        assert!(!x.is_subpixel());
        assert!(x.is_superpixel());
        assert_eq!(x.pixel(), Coordinate::new(336, 0));
        assert_eq!(x.pixel_index(), 336);
        assert_eq!(
            x.subdivide(),
            [Coordinate::new(84, 2), Coordinate::new(85, 2)]
        );
        assert_eq!(x.width(), 8);
        assert_eq!(x.widthf(), 8.0);

        let x = Coordinate::new(42, 0);
        assert!(!x.is_subpixel());
        assert!(!x.is_superpixel());
        assert_eq!(x.pixel(), x);
        assert_eq!(x.pixel_align(), 1);
        assert_eq!(x.pixel_index(), 42);
        assert_eq!(
            x.subdivide(),
            [Coordinate::new(84, -1), Coordinate::new(85, -1)]
        );
        assert_eq!(x.width(), 1);
        assert_eq!(x.widthf(), 1.0);

        let x = Coordinate::new(42, -3);
        assert!(x.is_subpixel());
        assert!(!x.is_superpixel());
        assert_eq!(x.pixel(), Coordinate::new(5, 0));
        assert_eq!(x.pixel_align(), 8);
        assert_eq!(x.pixel_index(), 5);
        assert_eq!(
            x.subdivide(),
            [Coordinate::new(84, -4), Coordinate::new(85, -4)]
        );
        assert_eq!(x.widthf(), 0.125);
    }

    #[test]
    fn integer_parameter() {
        fn test(x: Interval, ys: Vec<Interval>) {
            let n = IntegerParameter::new(x);
            assert_eq!(
                n.subdivide(),
                ys.iter()
                    .copied()
                    .map(IntegerParameter::new)
                    .collect::<SmallVec<[_; 2]>>()
            );

            let n = IntegerParameter::new(-x);
            assert_eq!(
                n.subdivide(),
                ys.into_iter()
                    .map(|y| IntegerParameter::new(-y))
                    .rev()
                    .collect::<SmallVec<[_; 2]>>()
            );
        }

        test(
            Interval::ENTIRE,
            vec![
                const_interval!(f64::NEG_INFINITY, 0.0),
                const_interval!(0.0, 0.0),
                const_interval!(0.0, f64::INFINITY),
            ],
        );

        test(
            const_interval!(0.0, f64::INFINITY),
            vec![
                const_interval!(1.0, 1.0),
                const_interval!(1.0, f64::INFINITY),
            ],
        );

        test(
            const_interval!(1.0, f64::INFINITY),
            vec![
                const_interval!(2.0, 2.0),
                const_interval!(2.0, f64::INFINITY),
            ],
        );

        test(
            const_interval!(2.0, f64::INFINITY),
            vec![
                const_interval!(3.0, 3.0),
                const_interval!(4.0, 4.0),
                const_interval!(4.0, f64::INFINITY),
            ],
        );

        test(
            const_interval!(4.0, f64::INFINITY),
            vec![
                const_interval!(4.0, 8.0),
                const_interval!(8.0, 8.0),
                const_interval!(8.0, f64::INFINITY),
            ],
        );

        test(const_interval!(0.0, 2.0), vec![const_interval!(1.0, 1.0)]);

        test(
            const_interval!(0.0, 3.0),
            vec![const_interval!(1.0, 1.0), const_interval!(2.0, 2.0)],
        );

        test(
            const_interval!(0.0, 4.0),
            vec![
                const_interval!(1.0, 1.0),
                const_interval!(2.0, 2.0),
                const_interval!(3.0, 3.0),
            ],
        );
    }

    #[test]
    fn real_parameter() {
        fn test(x: Interval, ys: Vec<Interval>) {
            let n = RealParameter::new(x);
            assert_eq!(
                n.subdivide(),
                ys.iter()
                    .copied()
                    .map(RealParameter::new)
                    .collect::<SmallVec<[_; 3]>>()
            );

            let n = RealParameter::new(-x);
            assert_eq!(
                n.subdivide(),
                ys.into_iter()
                    .map(|y| RealParameter::new(-y))
                    .rev()
                    .collect::<SmallVec<[_; 3]>>()
            );
        }

        test(
            Interval::ENTIRE,
            vec![
                const_interval!(f64::NEG_INFINITY, 0.0),
                const_interval!(0.0, f64::INFINITY),
            ],
        );

        test(
            const_interval!(0.0, f64::INFINITY),
            vec![
                const_interval!(0.0, 1.0),
                const_interval!(1.0, f64::INFINITY),
            ],
        );

        test(
            const_interval!(2.0, f64::INFINITY),
            vec![
                const_interval!(2.0, 4.0),
                const_interval!(4.0, f64::INFINITY),
            ],
        );

        test(
            const_interval!(2.0, 3.0),
            vec![const_interval!(2.0, 2.5), const_interval!(2.5, 3.0)],
        );
    }

    #[test]
    fn block_queue() {
        let mut queue = BlockQueue::new(VarSet::X | VarSet::Y);
        let blocks = [
            Block {
                x: Coordinate::new(0, -32),
                y: Coordinate::new(0xffffffffffffff, 0),
                ..default()
            },
            Block {
                x: Coordinate::new(0x7f, 64),
                y: Coordinate::new(0x2000000000000, 127),
                ..default()
            },
            Block {
                x: Coordinate::new(0x80, 0),
                y: Coordinate::new(0x1ffffffffffff, 0),
                ..default()
            },
            Block {
                x: Coordinate::new(0x3fff, 0),
                y: Coordinate::new(0x40000000000, 0),
                ..default()
            },
            Block {
                x: Coordinate::new(0x4000, 0),
                y: Coordinate::new(0x3ffffffffff, 0),
                ..default()
            },
            Block {
                x: Coordinate::new(0x1fffff, 0),
                y: Coordinate::new(0x800000000, 0),
                ..default()
            },
            Block {
                x: Coordinate::new(0x200000, 0),
                y: Coordinate::new(0x7ffffffff, 0),
                ..default()
            },
            Block {
                x: Coordinate::new(0xfffffff, 0),
                y: Coordinate::new(0x10000000, 0),
                ..default()
            },
            Block {
                x: Coordinate::new(0x10000000, 0),
                y: Coordinate::new(0xfffffff, 0),
                ..default()
            },
            Block {
                x: Coordinate::new(0x7ffffffff, 0),
                y: Coordinate::new(0x200000, 0),
                ..default()
            },
            Block {
                x: Coordinate::new(0x800000000, 0),
                y: Coordinate::new(0x1fffff, 0),
                ..default()
            },
            Block {
                x: Coordinate::new(0x3ffffffffff, 0),
                y: Coordinate::new(0x4000, 0),
                ..default()
            },
            Block {
                x: Coordinate::new(0x40000000000, 0),
                y: Coordinate::new(0x3fff, 0),
                ..default()
            },
            Block {
                x: Coordinate::new(0x1ffffffffffff, 0),
                y: Coordinate::new(0x80, 0),
                ..default()
            },
            Block {
                x: Coordinate::new(0x2000000000000, 0),
                y: Coordinate::new(0x7f, 0),
                ..default()
            },
            Block {
                x: Coordinate::new(0xffffffffffffff, 0),
                y: Coordinate::new(0, 0),
                ..default()
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

        fn test(mut queue: BlockQueue, b: Block) {
            queue.push_back(b.clone());
            queue.push_back(b.clone());
            assert_eq!(queue.pop_front(), Some(b.clone()));
            assert_eq!(queue.pop_front(), Some(b));
        }

        test(
            BlockQueue::new(VarSet::M),
            Block {
                m: IntegerParameter::new(const_interval!(-2.0, 3.0)),
                ..default()
            },
        );

        test(
            BlockQueue::new(VarSet::N),
            Block {
                n: IntegerParameter::new(const_interval!(-2.0, 3.0)),
                ..default()
            },
        );

        test(
            BlockQueue::new(VarSet::N_THETA),
            Block {
                n_theta: IntegerParameter::new(const_interval!(-2.0, 3.0)),
                ..default()
            },
        );

        test(
            BlockQueue::new(VarSet::T),
            Block {
                t: RealParameter::new(const_interval!(-2.0, 3.0)),
                ..default()
            },
        );
    }
}
