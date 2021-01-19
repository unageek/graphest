use crate::{
    dyn_relation::{DynRelation, EvalCache, EvalCacheLevel, RelationType},
    eval_result::EvalResult,
    interval_set::{DecSignSet, SignSet},
    rel::StaticForm,
};
use image::{imageops, Rgb, RgbImage};
use inari::{interval, Decoration, Interval};
use std::{
    collections::VecDeque,
    error, fmt,
    mem::size_of,
    time::{Duration, Instant},
};

/// The (approximate) maximum amount of memory that the graphing algorithm can use in bytes.
///
/// The current value is 1GiB. You can pick an arbitrary value.
const MEM_LIMIT: usize = 1usize << 30;

/// The maximum limit of the width (or height) of an [`Image`] in pixels.
///
/// The current value is 32768.
/// An arbitrary value that satisfies `MAX_IMAGE_WIDTH * 2^(-MIN_K) < u32::MAX` should be safe.
const MAX_IMAGE_WIDTH: u32 = 1u32 << 15;

/// The level of the smallest subpixels.
///
/// The current value is chosen so that [`C_UNCERTAIN`] fits in `u32`.
const MIN_K: i8 = -15;

/// The fractional pixel width of the smallest subpixels.
const MIN_WIDTH: f64 = 1.0 / ((1u32 << -MIN_K) as f64);

/// The width of the pixels in multiples of [`MIN_WIDTH`].
const PIXEL_ALIGNMENT: u32 = 1u32 << -MIN_K;

const C_FALSE: u32 = 0;
const C_UNCERTAIN: u32 = 1u32 << (2 * -MIN_K);
const C_TRUE: u32 = !0u32;

/// A rendering of a graph.
///
/// Each pixel stores the existence or absence of the solution:
///
/// - [`C_FALSE`] : There are no solutions in the pixel.
/// - `1..C_UNCERTAIN` : Uncertain, but we have shown that there are no solutions
///   in some parts of the pixel.
///   `value / C_UNCERTAIN` gives the ratio of the parts remaining uncertain.
/// - [`C_UNCERTAIN`] : Uncertain.
/// - [`C_TRUE`] : There is at least one solution in the pixel.
#[derive(Debug)]
struct Image {
    width: u32,
    height: u32,
    data: Vec<u32>,
}

impl Image {
    /// Creates a new [`Image`] with all pixels set to [`C_UNCERTAIN`].
    fn new(width: u32, height: u32) -> Self {
        assert!(width > 0 && width <= MAX_IMAGE_WIDTH && height > 0 && height <= MAX_IMAGE_WIDTH);
        Self {
            width,
            height,
            data: vec![C_UNCERTAIN; height as usize * width as usize],
        }
    }

    /// Returns the index in `self.data` where the value of the pixel is stored.
    fn index(&self, p: PixelIndex) -> usize {
        p.y as usize * self.width as usize + p.x as usize
    }

    /// Returns the value of the pixel.
    fn pixel(&self, p: PixelIndex) -> u32 {
        self.data[self.index(p)]
    }

    /// Returns a mutable reference to the value of the pixel.
    fn pixel_mut(&mut self, p: PixelIndex) -> &mut u32 {
        let i = self.index(p);
        &mut self.data[i]
    }

    fn size_in_heap(&self) -> usize {
        self.data.capacity() * size_of::<u32>()
    }
}

/// The index of a pixel of an [`Image`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PixelIndex {
    x: u32,
    y: u32,
}

impl PixelIndex {
    /// Returns the [`ImageBlock`] that represents the same area as the pixel.
    fn to_block(&self) -> ImageBlock {
        ImageBlock {
            x: self.x << -MIN_K,
            y: self.y << -MIN_K,
            kx: 0,
            ky: 0,
        }
    }
}

/// A rectangular region of an [`Image`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ImageBlock {
    /// The horizontal index of the first subpixel in the block.
    /// The index is represented in multiples of [`MIN_WIDTH`].
    x: u32,
    /// The vertical index of the first subpixel.
    y: u32,
    kx: i8,
    ky: i8,
}

impl ImageBlock {
    /// Returns the area of the block in multiples of `MIN_WIDTH^2`.
    ///
    /// Precondition: k ≤ 0, to prevent overflow.
    fn area(&self) -> u32 {
        1u32 << ((self.kx - MIN_K) + (self.ky - MIN_K))
    }

    /// Returns the height of the block in multiples of [`MIN_WIDTH`].
    fn height(&self) -> u32 {
        1u32 << (self.ky - MIN_K)
    }

    /// Returns `true` if the block can be divided both horizontally and vertically.
    fn is_subdivisible(&self) -> bool {
        self.kx > MIN_K && self.ky > MIN_K
    }

    /// Returns `true` if the width *or* the height of the block is smaller than a pixel.
    fn is_subpixel(&self) -> bool {
        self.kx < 0 || self.ky < 0
    }

    /// Returns `true` if the width *or* the height of the block is larger than a pixel.
    fn is_superpixel(&self) -> bool {
        self.kx > 0 || self.ky > 0
    }

    /// Returns the index of the pixel that contains the block.
    /// If the block spans multiple pixels, the least index is returned.
    fn pixel_index(&self) -> PixelIndex {
        PixelIndex {
            x: self.x >> -MIN_K,
            y: self.y >> -MIN_K,
        }
    }

    /// Returns the width of the block in pixels.
    ///
    /// Precondition: `k ≥ 0`.
    fn pixel_width(&self) -> u32 {
        1u32 << self.kx
    }

    /// Returns the width of the block in multiples of [`MIN_WIDTH`].
    fn width(&self) -> u32 {
        1u32 << (self.kx - MIN_K)
    }
}

/// A possibly empty rectangular region of the Cartesian plane.
#[derive(Debug, Clone)]
pub struct Region(Interval, Interval);

impl Region {
    /// Creates a new [`Region`] with the specified bounds.
    pub fn new(l: f64, r: f64, b: f64, t: f64) -> Self {
        // Regions constructed directly do not need to satisfy these conditions.
        assert!(l < r && b < t && l.is_finite() && r.is_finite() && b.is_finite() && t.is_finite());
        Self(interval!(l, r).unwrap(), interval!(b, t).unwrap())
    }

    /// Returns the tightest enclusure of the height of the region.
    fn height(&self) -> Interval {
        interval!(self.1.sup(), self.1.sup()).unwrap()
            - interval!(self.1.inf(), self.1.inf()).unwrap()
    }

    /// Returns the intersection of the two regions.
    fn intersection(&self, rhs: &Self) -> Self {
        Self(self.0.intersection(rhs.0), self.1.intersection(rhs.1))
    }

    /// Returns `true` if the region is empty.
    fn is_empty(&self) -> bool {
        self.0.is_empty() || self.1.is_empty()
    }

    /// Returns the tightest enclosure of the width of the region.
    fn width(&self) -> Interval {
        interval!(self.0.sup(), self.0.sup()).unwrap()
            - interval!(self.0.inf(), self.0.inf()).unwrap()
    }
}

/// A rectangular region of the Cartesian plane with inexact bounds.
///
/// Conceptually, it is a pair of two rectangular regions *inner* and *outer*
/// that satisfy `inner ⊆ outer`. The inner region can be empty, while the outer one cannot.
#[derive(Debug)]
struct InexactRegion {
    l: Interval,
    r: Interval,
    b: Interval,
    t: Interval,
}

impl InexactRegion {
    /// Returns the inner region.
    fn inner(&self) -> Region {
        Region(
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

    /// Returns the outer region.
    fn outer(&self) -> Region {
        Region(
            interval!(self.l.inf(), self.r.sup()).unwrap(),
            interval!(self.b.inf(), self.t.sup()).unwrap(),
        )
    }

    /// Returns a subset of the outer region.
    ///
    /// It is assumed that the region is obtained from the given block.
    /// When applied to a set of regions/blocks which form a partition of a pixel,
    /// the results form a partition of the outer boundary of the pixel.
    fn subpixel_outer(&self, blk: ImageBlock) -> Region {
        // `PIXEL_ALIGNMENT` is a power of two.
        // Therefore, `x & mask` is equivalent to `x % PIXEL_ALIGNMENT`.
        let mask = PIXEL_ALIGNMENT - 1;

        let l = if blk.x & mask == 0 {
            self.l.inf()
        } else {
            self.l.mid()
        };
        let r = if (blk.x + blk.width()) & mask == 0 {
            self.r.sup()
        } else {
            self.r.mid()
        };
        let b = if blk.y & mask == 0 {
            self.b.inf()
        } else {
            self.b.mid()
        };
        let t = if (blk.y + blk.height()) & mask == 0 {
            self.t.sup()
        } else {
            self.t.mid()
        };
        Region(interval!(l, r).unwrap(), interval!(b, t).unwrap())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use inari::*;

    #[test]
    fn subpixel_outer() {
        let u = InexactRegion {
            l: const_interval!(0.33, 0.34),
            r: const_interval!(0.66, 0.67),
            b: const_interval!(1.33, 1.34),
            t: const_interval!(1.66, 1.67),
        };

        // The bottom/left sides are pixel boundaries.
        let b = ImageBlock {
            x: 5 * PIXEL_ALIGNMENT,
            y: 7 * PIXEL_ALIGNMENT,
            kx: -2,
            ky: -2,
        };
        let u_up = u.subpixel_outer(b);
        assert_eq!(u_up.0.inf(), u.l.inf());
        assert_eq!(u_up.0.sup(), u.r.mid());
        assert_eq!(u_up.1.inf(), u.b.inf());
        assert_eq!(u_up.1.sup(), u.t.mid());

        // The top/right sides are pixel boundaries.
        let b = ImageBlock {
            x: b.x + 3 * PIXEL_ALIGNMENT / 4,
            y: b.y + 3 * PIXEL_ALIGNMENT / 4,
            ..b
        };
        let u_up = u.subpixel_outer(b);
        assert_eq!(u_up.0.inf(), u.l.mid());
        assert_eq!(u_up.0.sup(), u.r.sup());
        assert_eq!(u_up.1.inf(), u.b.mid());
        assert_eq!(u_up.1.sup(), u.t.sup());
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GraphingErrorKind {
    ReachedMemLimit,
    ReachedSubdivisionLimit,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GraphingError {
    pub kind: GraphingErrorKind,
}

impl fmt::Display for GraphingError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self.kind {
            GraphingErrorKind::ReachedMemLimit => write!(f, "reached memory usage limit"),
            GraphingErrorKind::ReachedSubdivisionLimit => write!(f, "reached subdivision limit"),
        }
    }
}

impl error::Error for GraphingError {}

#[derive(Clone, Debug)]
pub struct GraphingStatistics {
    pub pixels: usize,
    pub pixels_proven: usize,
    pub eval_count: usize,
    pub time_elapsed: Duration,
}

#[derive(Debug)]
pub struct Graph {
    rel: DynRelation,
    forms: Vec<StaticForm>,
    relation_type: RelationType,
    im: Image,
    bs_to_subdivide: VecDeque<ImageBlock>,
    bx_end: u32,
    by_end: u32,
    // Affine transformation from subpixel coordinates (bx, by) to real coordinates (x, y):
    //
    //   ⎛ x ⎞   ⎛ sx   0  tx ⎞ ⎛ bx ⎞
    //   ⎜ y ⎟ = ⎜  0  sy  ty ⎟ ⎜ by ⎟.
    //   ⎝ 1 ⎠   ⎝  0   0   1 ⎠ ⎝  1 ⎠
    sx: Interval,
    sy: Interval,
    tx: Interval,
    ty: Interval,
    stats: GraphingStatistics,
}

impl Graph {
    pub fn new(rel: DynRelation, region: Region, im_width: u32, im_height: u32) -> Self {
        let forms = rel.forms().clone();
        let relation_type = rel.relation_type();
        let mut g = Self {
            rel,
            forms,
            relation_type,
            im: Image::new(im_width, im_height),
            bs_to_subdivide: VecDeque::new(),
            bx_end: im_width << -MIN_K,
            by_end: im_height << -MIN_K,
            sx: region.width() / Self::point_interval(im_width as f64 / MIN_WIDTH),
            sy: region.height() / Self::point_interval(im_height as f64 / MIN_WIDTH),
            tx: Self::point_interval(region.0.inf()),
            ty: Self::point_interval(region.1.inf()),
            stats: GraphingStatistics {
                pixels: im_width as usize * im_height as usize,
                pixels_proven: 0,
                eval_count: 0,
                time_elapsed: Duration::new(0, 0),
            },
        };
        let k = (im_width.max(im_height) as f64).log2().ceil() as i8;
        g.bs_to_subdivide.push_back(ImageBlock {
            x: 0,
            y: 0,
            kx: k,
            ky: k,
        });
        g
    }

    pub fn get_image(&self, im: &mut RgbImage) {
        assert!(im.width() == self.im.width && im.height() == self.im.height);
        for (src, dst) in self.im.data.iter().zip(im.pixels_mut()) {
            *dst = match *src {
                C_TRUE => Rgb([0, 0, 0]),
                C_FALSE => Rgb([255, 255, 255]),
                _ => Rgb([64, 128, 192]),
            }
        }
        imageops::flip_vertical_in_place(im);
    }

    pub fn get_statistics(&self) -> GraphingStatistics {
        GraphingStatistics {
            pixels_proven: self
                .im
                .data
                .iter()
                .filter(|&&s| s == C_TRUE || s == C_FALSE)
                .count(),
            eval_count: self.rel.eval_count(),
            ..self.stats
        }
    }

    /// Refines the graph for a given amount of time.
    ///
    /// Returns `Ok(true)`/`Ok(false)` if graphing is complete/incomplete after refinement.
    pub fn refine(&mut self, timeout: Duration) -> Result<bool, GraphingError> {
        let now = Instant::now();
        let result = self.refine_impl(timeout, &now);
        self.stats.time_elapsed += now.elapsed();
        result
    }

    fn refine_impl(&mut self, timeout: Duration, now: &Instant) -> Result<bool, GraphingError> {
        let mut sub_bs = Vec::<ImageBlock>::new();
        // The blocks are queued in the Morton order. Thus, the cache should work effectively.
        let mut cache_eval_on_region = EvalCache::new(EvalCacheLevel::PerAxis);
        let mut cache_eval_on_point = EvalCache::new(EvalCacheLevel::Full);
        while let Some(b) = self.bs_to_subdivide.pop_front() {
            if b.is_superpixel() {
                self.push_sub_blocks_clipped(&mut sub_bs, b);
            } else {
                self.push_sub_blocks(&mut sub_bs, b);
            }

            for sub_b in sub_bs.drain(..) {
                if !sub_b.is_subpixel() {
                    self.refine_pixel(sub_b, &mut cache_eval_on_region);
                } else {
                    self.refine_subpixel(
                        sub_b,
                        &mut cache_eval_on_region,
                        &mut cache_eval_on_point,
                    )?;
                };
            }

            if self.im.size_in_heap()
                + self.bs_to_subdivide.capacity() * size_of::<ImageBlock>()
                + cache_eval_on_region.size_in_heap()
                + cache_eval_on_point.size_in_heap()
                > MEM_LIMIT
            {
                return Err(GraphingError {
                    kind: GraphingErrorKind::ReachedMemLimit,
                });
            }

            if now.elapsed() > timeout {
                break;
            }
        }

        Ok(self.bs_to_subdivide.is_empty())
    }

    fn refine_pixel(&mut self, b: ImageBlock, cache: &mut EvalCache) {
        let u_up = self.block_to_region_clipped(b).outer();
        let r_u_up = Self::eval_on_region(&mut self.rel, &u_up, Some(cache));

        let is_true = r_u_up
            .map(|DecSignSet(ss, d)| d >= Decoration::Def && ss == SignSet::ZERO)
            .eval(&self.forms[..]);
        let is_false = !r_u_up
            .map(|DecSignSet(ss, _)| ss.contains(SignSet::ZERO))
            .eval(&self.forms[..]);

        if is_true || is_false {
            let pixel = b.pixel_index();
            let pixel_width = b.pixel_width();
            let stat = if is_true { C_TRUE } else { C_FALSE };
            for y in pixel.y..(pixel.y + pixel_width).min(self.im.height) {
                for x in pixel.x..(pixel.x + pixel_width).min(self.im.width) {
                    *self.im.pixel_mut(PixelIndex { x, y }) = stat;
                }
            }
        } else {
            self.bs_to_subdivide.push_back(b);
        }
    }

    fn refine_subpixel(
        &mut self,
        b: ImageBlock,
        cache_eval_on_region: &mut EvalCache,
        cache_eval_on_point: &mut EvalCache,
    ) -> Result<(), GraphingError> {
        let pixel = b.pixel_index();
        if self.im.pixel(pixel) == C_TRUE {
            // This pixel has already been proven to be true.
            return Ok(());
        }

        let p_dn = self.block_to_region(pixel.to_block()).inner();
        if p_dn.is_empty() {
            return Err(GraphingError {
                kind: GraphingErrorKind::ReachedSubdivisionLimit,
            });
        }

        let u_up = self.block_to_region(b).subpixel_outer(b);
        let r_u_up = Self::eval_on_region(&mut self.rel, &u_up, Some(cache_eval_on_region));

        if r_u_up
            .map(|DecSignSet(ss, _)| ss == SignSet::ZERO)
            .eval(&self.forms[..])
        {
            // This pixel is proven to be true.
            *self.im.pixel_mut(pixel) = C_TRUE;
            return Ok(());
        }
        if !r_u_up
            .map(|DecSignSet(ss, _)| ss.contains(SignSet::ZERO))
            .eval(&self.forms[..])
        {
            // This subpixel is proven to be false.
            *self.im.pixel_mut(pixel) -= b.area();
            return Ok(());
        }

        let inter = u_up.intersection(&p_dn);
        if inter.is_empty() {
            *self.im.pixel_mut(pixel) -= b.area();
            return Ok(());
        }

        // Evaluate the relation for some sample points within the inner bounds of the subpixel
        // and try proving existence of a solution in two ways:
        //
        // a. Test if the relation is true for any of the sample points.
        //    This is useful especially for plotting inequalities such as "lcm(x, y) ≤ 1".
        //
        // b. Use the intermediate value theorem.
        //    A note about `locally_zero_mask` (for plotting conjunction):
        //    Suppose we are plotting "y = sin(x) && x ≥ 0".
        //    If the conjunct "x ≥ 0" is true throughout the subpixel
        //    and "y - sin(x)" evaluates to `POS` for a sample point and `NEG` for another,
        //    we can conclude that there is a point within the subpixel where the entire relation holds.
        //    Such observation would not be possible by merely converting the relation to
        //    "|y - sin(x)| + |x ≥ 0 ? 0 : 1| = 0".
        let dac_mask = r_u_up.map(|DecSignSet(_, d)| d >= Decoration::Dac);
        let locally_zero_mask =
            r_u_up.map(|DecSignSet(ss, d)| ss == SignSet::ZERO && d >= Decoration::Dac);

        let points = [
            Self::simple_eval_point(&inter),
            (inter.0.inf(), inter.1.inf()), // bottom left
            (inter.0.sup(), inter.1.inf()), // bottom right
            (inter.0.inf(), inter.1.sup()), // top left
            (inter.0.sup(), inter.1.sup()), // top right
        ];

        let mut neg_mask = r_u_up.map(|_| false);
        let mut pos_mask = neg_mask.clone();
        for point in &points {
            let r = Self::eval_on_point(&mut self.rel, point.0, point.1, Some(cache_eval_on_point));

            // `ss` is nonempty if the decoration is `Dac`, which is ensured by `dac_mask`.
            neg_mask |= r.map(|DecSignSet(ss, _)| (SignSet::NEG | SignSet::ZERO).contains(ss));
            pos_mask |= r.map(|DecSignSet(ss, _)| (SignSet::POS | SignSet::ZERO).contains(ss));

            if r.map(|DecSignSet(ss, _)| ss == SignSet::ZERO)
                .eval(&self.forms[..])
                || (&(&neg_mask & &pos_mask) & &dac_mask)
                    .solution_certainly_exists(&self.forms[..], &locally_zero_mask)
            {
                // Found a solution.
                *self.im.pixel_mut(pixel) = C_TRUE;
                return Ok(());
            }
        }

        if b.is_subdivisible() {
            self.bs_to_subdivide.push_back(b);
            Ok(())
        } else {
            Err(GraphingError {
                kind: GraphingErrorKind::ReachedSubdivisionLimit,
            })
        }
    }

    fn eval_on_point(
        rel: &mut DynRelation,
        x: f64,
        y: f64,
        cache: Option<&mut EvalCache>,
    ) -> EvalResult {
        rel.eval(Self::point_interval(x), Self::point_interval(y), cache)
    }

    fn eval_on_region(
        rel: &mut DynRelation,
        r: &Region,
        cache: Option<&mut EvalCache>,
    ) -> EvalResult {
        rel.eval(r.0, r.1, cache)
    }

    /// Returns a point within the given region whose coordinates have as many trailing zeros
    /// as they can in their significand bits.
    /// At such a point, arithmetic expressions are more likely to be evaluated to exact numbers.
    fn simple_eval_point(r: &Region) -> (f64, f64) {
        fn f(x: Interval) -> f64 {
            let a = x.inf();
            let b = x.sup();
            let a_bits = a.to_bits();
            let b_bits = b.to_bits();
            let diff = a_bits ^ b_bits;
            // The number of leading equal bits.
            let n = diff.leading_zeros();
            // Set all bits from the MSB through the first differing bit.
            let mask = !0u64 << (64 - n - 1);
            if a <= 0.0 {
                f64::from_bits(a_bits & mask)
            } else {
                f64::from_bits(b_bits & mask)
            }
        }

        (f(r.0), f(r.1))
    }

    /// Precondition: `!b.is_superpixel()`.
    fn block_to_region(&self, b: ImageBlock) -> InexactRegion {
        InexactRegion {
            l: Self::point_interval(b.x as f64).mul_add(self.sx, self.tx),
            r: Self::point_interval((b.x + b.width()) as f64).mul_add(self.sx, self.tx),
            b: Self::point_interval(b.y as f64).mul_add(self.sy, self.ty),
            t: Self::point_interval((b.y + b.height()) as f64).mul_add(self.sy, self.ty),
        }
    }

    fn block_to_region_clipped(&self, b: ImageBlock) -> InexactRegion {
        InexactRegion {
            l: Self::point_interval(b.x as f64).mul_add(self.sx, self.tx),
            r: Self::point_interval((b.x + b.width()).min(self.bx_end) as f64)
                .mul_add(self.sx, self.tx),
            b: Self::point_interval(b.y as f64).mul_add(self.sy, self.ty),
            t: Self::point_interval((b.y + b.height()).min(self.by_end) as f64)
                .mul_add(self.sy, self.ty),
        }
    }

    fn point_interval(x: f64) -> Interval {
        interval!(x, x).unwrap()
    }

    /// Precondition: `!b.is_superpixel()`.
    fn push_sub_blocks(&self, sub_bs: &mut Vec<ImageBlock>, b: ImageBlock) {
        let x0 = b.x;
        let y0 = b.y;
        let x1 = x0 + b.width() / 2;
        let y1 = y0 + b.height() / 2;
        match self.relation_type {
            RelationType::FunctionOfX => {
                let kx = b.kx - 1;
                let ky = b.ky;
                sub_bs.push(ImageBlock {
                    x: x0,
                    y: y0,
                    kx,
                    ky,
                });
                sub_bs.push(ImageBlock {
                    x: x1,
                    y: y0,
                    kx,
                    ky,
                });
            }
            RelationType::FunctionOfY => {
                let kx = b.kx;
                let ky = b.ky - 1;
                sub_bs.push(ImageBlock {
                    x: x0,
                    y: y0,
                    kx,
                    ky,
                });
                sub_bs.push(ImageBlock {
                    x: x0,
                    y: y1,
                    kx,
                    ky,
                });
            }
            _ => {
                let kx = b.kx - 1;
                let ky = b.ky - 1;
                sub_bs.push(ImageBlock {
                    x: x0,
                    y: y0,
                    kx,
                    ky,
                });
                sub_bs.push(ImageBlock {
                    x: x1,
                    y: y0,
                    kx,
                    ky,
                });
                sub_bs.push(ImageBlock {
                    x: x0,
                    y: y1,
                    kx,
                    ky,
                });
                sub_bs.push(ImageBlock {
                    x: x1,
                    y: y1,
                    kx,
                    ky,
                });
            }
        }
    }

    fn push_sub_blocks_clipped(&self, sub_bs: &mut Vec<ImageBlock>, b: ImageBlock) {
        let x0 = b.x;
        let y0 = b.y;
        let x1 = x0 + b.width() / 2;
        let y1 = y0 + b.height() / 2;
        let kx = b.kx - 1;
        let ky = b.ky - 1;
        sub_bs.push(ImageBlock {
            x: x0,
            y: y0,
            kx,
            ky,
        });
        if y1 < self.by_end {
            sub_bs.push(ImageBlock {
                x: x0,
                y: y1,
                kx,
                ky,
            });
        }
        if x1 < self.bx_end {
            sub_bs.push(ImageBlock {
                x: x1,
                y: y0,
                kx,
                ky,
            });
        }
        if x1 < self.bx_end && y1 < self.by_end {
            sub_bs.push(ImageBlock {
                x: x1,
                y: y1,
                kx,
                ky,
            });
        }
    }
}
