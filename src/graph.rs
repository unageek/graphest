use crate::{
    dyn_relation::{DynRelation, EvaluationCache, EvaluationCacheLevel},
    eval_result::EvalResult,
    interval_set::SignSet,
    rel::StaticRel,
};
use image::{imageops, Rgb, RgbImage};
use inari::{interval, Decoration, Interval};
use std::{
    error, fmt,
    mem::size_of,
    time::{Duration, Instant},
};

/// You can pick any value.
const MEM_LIMIT: usize = 1usize << 30; // 1GiB

/// The value is chosen so that `STAT_UNCERTAIN` fits in `u32`.
const MIN_K: i32 = -15;
/// An arbitrary value that satisfies `MAX_WIDTH * 2^(-MIN_K) < u32::MAX`.
const MAX_WIDTH: u32 = 1u32 << 15; // 32768

/// The width of the smallest subpixels.
const MIN_WIDTH: f64 = 1.0 / ((1u32 << -MIN_K) as f64);

const STAT_FALSE: u32 = 0;
const STAT_UNCERTAIN: u32 = 1u32 << (2 * -MIN_K);
const STAT_TRUE: u32 = !0u32;

/// Each pixel of an `Image` keeps track of the proof status as follows:
///
///   - `STAT_FALSE`: The relation has no solution within the pixel.
///   - `1..STAT_UNCERTAIN`: Uncertain, but the relation is known to
///     have no solution within some parts of the pixel.
///   - `STAT_UNCERTAIN`: Uncertain.
///   - `STAT_TRUE`: The relation has at least one solution within the pixel.
#[derive(Debug)]
struct Image {
    width: u32,
    height: u32,
    data: Vec<u32>,
}

impl Image {
    /// Creates a new `Image` with all pixels set to `STAT_UNCERTAIN`.
    fn new(width: u32, height: u32) -> Self {
        assert!(width <= MAX_WIDTH && height <= MAX_WIDTH);
        Self {
            width,
            height,
            data: vec![STAT_UNCERTAIN; height as usize * width as usize],
        }
    }

    /// Returns the index in `data` where the value of the pixel is stored.
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
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PixelIndex {
    x: u32,
    y: u32,
}

impl PixelIndex {
    fn to_block(&self) -> ImageBlock {
        ImageBlock {
            x: self.x << -MIN_K,
            y: self.y << -MIN_K,
            k: 0,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ImageBlock {
    /// The horizontal index of the first subpixel in the block.
    /// The index is represented in multiples of the smallest subpixel (= 2^MIN_K px).
    x: u32,
    /// The vertical index of the first subpixel.
    y: u32,
    k: i32,
}

impl ImageBlock {
    /// Returns the area of the block in multiples of `MIN_WIDTH^2`.
    /// Precondition: k ≤ 0.
    fn area(&self) -> u32 {
        1u32 << (2 * (self.k - MIN_K))
    }

    fn is_subdivisible(&self) -> bool {
        self.k > MIN_K
    }

    /// Returns the index of the pixel that contains the block.
    /// If the block spans multiple pixels (k > 0), the least index is returned.
    fn pixel_index(&self) -> PixelIndex {
        PixelIndex {
            x: self.x >> -MIN_K,
            y: self.y >> -MIN_K,
        }
    }

    /// Returns the pixel width (or height) of the block.
    /// Precondition: k ≥ 0.
    fn pixel_width(&self) -> u32 {
        1u32 << self.k
    }

    /// Returns the width (or height) of the block in multiples of `MIN_WIDTH`.
    fn width(&self) -> u32 {
        1u32 << (self.k - MIN_K)
    }
}

// Represents a rectangular region (subset of ℝ²).
#[derive(Debug, Clone)]
pub struct Region(Interval, Interval);

impl Region {
    /// Creates a new `Region` with the specified bounds.
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

#[derive(Debug)]
struct InexactRegion {
    l: Interval,
    r: Interval,
    b: Interval,
    t: Interval,
}

impl InexactRegion {
    /// Returns the inner bounds of the inexact region.
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

    /// Returns the outer bounds of the inexact region.
    fn outer(&self) -> Region {
        Region(
            interval!(self.l.inf(), self.r.sup()).unwrap(),
            interval!(self.b.inf(), self.t.sup()).unwrap(),
        )
    }

    fn subpixel_outer(&self, b: ImageBlock) -> Region {
        let modulus = b.width() - 1; // `b.width()` is a power of two, thus this works as a bit mask.
        let rem_bx = b.x & modulus; // `bx % modulus`
        let rem_by = b.y & modulus; // `by % modulus`
        let l = if rem_bx == 0 {
            self.l.inf()
        } else {
            self.l.mid()
        };
        let r = if rem_bx == modulus {
            self.r.sup()
        } else {
            self.r.mid()
        };
        let b = if rem_by == 0 {
            self.b.inf()
        } else {
            self.b.mid()
        };
        let t = if rem_by == modulus {
            self.t.sup()
        } else {
            self.t.mid()
        };
        Region(interval!(l, r).unwrap(), interval!(b, t).unwrap())
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
    pub evaluation_count: usize,
    pub time_elapsed: Duration,
}

#[derive(Debug)]
pub struct Graph {
    rel: DynRelation,
    rels: Vec<StaticRel>,
    im: Image,
    bx_end: u32,
    by_end: u32,
    bs: Vec<ImageBlock>,
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
        assert!(im_width > 0 && im_height > 0);
        let rels = rel.rels().clone();
        let mut g = Self {
            rel,
            rels,
            im: Image::new(im_width, im_height),
            bx_end: im_width << -MIN_K,
            by_end: im_height << -MIN_K,
            bs: Vec::new(),
            sx: region.width() / Self::point_interval(im_width as f64 / MIN_WIDTH),
            sy: region.height() / Self::point_interval(im_height as f64 / MIN_WIDTH),
            tx: Self::point_interval(region.0.inf()),
            ty: Self::point_interval(region.1.inf()),
            stats: GraphingStatistics {
                pixels: im_width as usize * im_height as usize,
                pixels_proven: 0,
                evaluation_count: 0,
                time_elapsed: Duration::new(0, 0),
            },
        };
        g.bs = g.get_initial_image_blocks();
        g
    }

    /// Performs the refinement step.
    ///
    /// Returns `Ok(true)`/`Ok(false)` if graphing is complete/incomplete after refinement.
    pub fn step(&mut self) -> Result<bool, GraphingError> {
        if self.bs.is_empty() {
            return Ok(true);
        }

        let now = Instant::now();
        self.bs = if self.bs[0].k >= 0 {
            self.refine_pixels()?
        } else {
            self.refine_subpixels()?
        };
        self.stats.time_elapsed += now.elapsed();
        Ok(self.bs.is_empty())
    }

    pub fn get_image(&self) -> RgbImage {
        let mut im = RgbImage::new(self.im.width, self.im.height);
        for (src, dst) in self.im.data.iter().zip(im.pixels_mut()) {
            *dst = match *src {
                STAT_TRUE => Rgb([0, 0, 0]),
                STAT_FALSE => Rgb([255, 255, 255]),
                _ => Rgb([64, 128, 192]),
            }
        }
        imageops::flip_vertical_in_place(&mut im);
        im
    }

    pub fn get_statistics(&self) -> GraphingStatistics {
        GraphingStatistics {
            pixels_proven: self
                .im
                .data
                .iter()
                .filter(|&s| *s == STAT_TRUE || *s == STAT_FALSE)
                .count(),
            evaluation_count: self.rel.evaluation_count(),
            ..self.stats
        }
    }

    fn get_initial_image_blocks(&self) -> Vec<ImageBlock> {
        let k = (self.im.width.max(self.im.height) as f64).log2() as i32;
        let bw = 2.0f64.powi(k);
        let nx = (self.im.width as f64 / bw).ceil() as u32;
        let ny = (self.im.height as f64 / bw).ceil() as u32;
        let mut blocks = Vec::<ImageBlock>::new();
        for by in 0..ny {
            for bx in 0..nx {
                blocks.push(ImageBlock {
                    x: bx << (k - MIN_K),
                    y: by << (k - MIN_K),
                    k,
                });
            }
        }
        blocks
    }

    fn refine_pixels(&mut self) -> Result<Vec<ImageBlock>, GraphingError> {
        let bs = &self.bs;
        let mut cache = EvaluationCache::new(EvaluationCacheLevel::PerAxis);
        let mut sub_blocks = Vec::<ImageBlock>::new();
        for b in bs.iter().copied() {
            let u_up = self.image_block_to_region_clipped(b).outer();
            let r_u_up = Self::eval_on_region(&mut self.rel, &u_up, Some(&mut cache));

            let is_true = r_u_up.map_reduce(&self.rels[..], &|ss, d| {
                d >= Decoration::Def && ss == SignSet::ZERO
            });
            let is_false = !r_u_up.map_reduce(&self.rels[..], &|ss, _| ss.contains(SignSet::ZERO));
            if is_true || is_false {
                let pixel = b.pixel_index();
                let pixel_width = b.pixel_width();
                let stat = if is_true { STAT_TRUE } else { STAT_FALSE };
                for y in pixel.y..(pixel.y + pixel_width).min(self.im.height) {
                    for x in pixel.x..(pixel.x + pixel_width).min(self.im.width) {
                        *self.im.pixel_mut(PixelIndex { x, y }) = stat;
                    }
                }
            } else {
                self.push_sub_blocks_clipped(&mut sub_blocks, b);
                if (bs.capacity() + sub_blocks.capacity()) * size_of::<ImageBlock>()
                    + cache.size_in_bytes()
                    > MEM_LIMIT
                {
                    return Err(GraphingError {
                        kind: GraphingErrorKind::ReachedMemLimit,
                    });
                }
            }
        }
        Ok(sub_blocks)
    }

    fn refine_subpixels(&mut self) -> Result<Vec<ImageBlock>, GraphingError> {
        let bs = &self.bs;
        let mut cache_per_axis = EvaluationCache::new(EvaluationCacheLevel::PerAxis);
        let mut cache_full = EvaluationCache::new(EvaluationCacheLevel::Full);
        let mut some_test_failed = false;
        let mut sub_blocks = Vec::<ImageBlock>::new();
        for b in bs.iter().copied() {
            let pixel = b.pixel_index();
            let stat = self.im.pixel(pixel);
            if stat == STAT_FALSE || stat == STAT_TRUE {
                continue;
            }

            let p_dn = self.image_block_to_region(pixel.to_block()).inner();
            if p_dn.is_empty() {
                some_test_failed = true;
                continue;
            }

            let u_up = self.image_block_to_region(b).subpixel_outer(b);
            let r_u_up = Self::eval_on_region(&mut self.rel, &u_up, Some(&mut cache_per_axis));

            if r_u_up.map_reduce(&self.rels[..], &|ss, _| ss == SignSet::ZERO) {
                // This pixel is proven to be true.
                *self.im.pixel_mut(pixel) = STAT_TRUE;
                continue;
            }
            if !r_u_up.map_reduce(&self.rels[..], &|ss, _| ss.contains(SignSet::ZERO)) {
                // This subpixel is proven to be false.
                *self.im.pixel_mut(pixel) -= b.area();
                continue;
            }

            let inter = u_up.intersection(&p_dn);
            if inter.is_empty() {
                *self.im.pixel_mut(pixel) -= b.area();
                continue;
            }

            // We could re-evaluate the relation on `inter` instead of `u_up`
            // to get a slightly better result, but the effect would be negligible.

            // To prove the existence of a solution by a change of sign...
            //   for conjunctions, both operands must be `Dac`.
            //   for disjunctions, at least one operand must be `Dac`.
            // There is little chance that an expression is evaluated
            // to zero on one of the probe points. In that case,
            // the expression is not required to be `Dac` on the entire
            // subpixel. We don't care such a chance.
            let dac_mask = r_u_up.map(&self.rels[..], &|_, d| d >= Decoration::Dac);
            if dac_mask.reduce(&self.rels[..]) {
                // Suppose we are plotting the graph of a conjunction such as
                // "y == sin(x) && x >= 0".
                // If the conjunct "x >= 0" holds everywhere in the subpixel,
                // and "y - sin(x)" evaluates to both `POS` and `NEG` at
                // different points in the region, we can tell that
                // there exists a point where the entire relation holds.
                // Such a test is not possible by merely converting
                // the relation to "|y - sin(x)| + |x >= 0 ? 0 : 1| == 0".
                let locally_zero_mask = r_u_up.map(&self.rels[..], &|ss, d| {
                    ss == SignSet::ZERO && d >= Decoration::Dac
                });

                let points = [
                    (inter.0.inf(), inter.1.inf()), // bottom left
                    (inter.0.sup(), inter.1.inf()), // bottom right
                    (inter.0.inf(), inter.1.sup()), // top left
                    (inter.0.sup(), inter.1.sup()), // top right
                ];

                let mut found_solution = false;
                let mut neg_mask = r_u_up.map(&self.rels[..], &|_, _| false);
                let mut pos_mask = neg_mask.clone();
                for point in &points {
                    let r =
                        Self::eval_on_point(&mut self.rel, point.0, point.1, Some(&mut cache_full));

                    // `ss` is not empty if the decoration is `Dac`, which is
                    // ensured by `dac_mask`.
                    neg_mask |= r.map(&self.rels[..], &|ss, _| {
                        ss == ss & (SignSet::NEG | SignSet::ZERO) // ss <= 0
                    });
                    pos_mask |= r.map(&self.rels[..], &|ss, _| {
                        ss == ss & (SignSet::POS | SignSet::ZERO) // ss >= 0
                    });

                    if (&(&neg_mask & &pos_mask) & &dac_mask)
                        .solution_certainly_exists(&self.rels[..], &locally_zero_mask)
                    {
                        found_solution = true;
                        break;
                    }
                }

                if found_solution {
                    *self.im.pixel_mut(pixel) = STAT_TRUE;
                    continue;
                }
            }

            if b.is_subdivisible() {
                Self::push_sub_blocks(&mut sub_blocks, b);
                if (bs.capacity() + sub_blocks.capacity()) * size_of::<ImageBlock>()
                    + cache_per_axis.size_in_bytes()
                    + cache_full.size_in_bytes()
                    > MEM_LIMIT
                {
                    return Err(GraphingError {
                        kind: GraphingErrorKind::ReachedMemLimit,
                    });
                }
            }
            some_test_failed = true;
        }

        if sub_blocks.is_empty() && some_test_failed {
            Err(GraphingError {
                kind: GraphingErrorKind::ReachedSubdivisionLimit,
            })
        } else {
            Ok(sub_blocks)
        }
    }

    fn eval_on_point(
        rel: &mut DynRelation,
        x: f64,
        y: f64,
        cache: Option<&mut EvaluationCache>,
    ) -> EvalResult {
        rel.evaluate(interval!(x, x).unwrap(), interval!(y, y).unwrap(), cache)
    }

    fn eval_on_region(
        rel: &mut DynRelation,
        r: &Region,
        cache: Option<&mut EvaluationCache>,
    ) -> EvalResult {
        rel.evaluate(r.0, r.1, cache)
    }

    fn image_block_to_region(&self, b: ImageBlock) -> InexactRegion {
        InexactRegion {
            l: Self::point_interval(b.x as f64).mul_add(self.sx, self.tx),
            r: Self::point_interval((b.x + b.width()) as f64).mul_add(self.sx, self.tx),
            b: Self::point_interval(b.y as f64).mul_add(self.sy, self.ty),
            t: Self::point_interval((b.y + b.width()) as f64).mul_add(self.sy, self.ty),
        }
    }

    fn image_block_to_region_clipped(&self, b: ImageBlock) -> InexactRegion {
        InexactRegion {
            l: Self::point_interval(b.x as f64).mul_add(self.sx, self.tx),
            r: Self::point_interval((b.x + b.width()).min(self.bx_end) as f64)
                .mul_add(self.sx, self.tx),
            b: Self::point_interval(b.y as f64).mul_add(self.sy, self.ty),
            t: Self::point_interval((b.y + b.width()).min(self.by_end) as f64)
                .mul_add(self.sy, self.ty),
        }
    }

    fn point_interval(x: f64) -> Interval {
        interval!(x, x).unwrap()
    }

    fn push_sub_blocks(blocks: &mut Vec<ImageBlock>, b: ImageBlock) {
        let x0 = b.x;
        let y0 = b.y;
        let x1 = x0 + b.width() / 2;
        let y1 = y0 + b.width() / 2;
        let k = b.k - 1;
        blocks.push(ImageBlock { x: x0, y: y0, k });
        blocks.push(ImageBlock { x: x1, y: y0, k });
        blocks.push(ImageBlock { x: x0, y: y1, k });
        blocks.push(ImageBlock { x: x1, y: y1, k });
    }

    fn push_sub_blocks_clipped(&self, blocks: &mut Vec<ImageBlock>, b: ImageBlock) {
        let x0 = b.x;
        let y0 = b.y;
        let x1 = x0 + b.width() / 2;
        let y1 = y0 + b.width() / 2;
        let k = b.k - 1;
        blocks.push(ImageBlock { x: x0, y: y0, k });
        if x1 < self.bx_end {
            blocks.push(ImageBlock { x: x1, y: y0, k });
        }
        if y1 < self.by_end {
            blocks.push(ImageBlock { x: x0, y: y1, k });
        }
        if x1 < self.bx_end && y1 < self.by_end {
            blocks.push(ImageBlock { x: x1, y: y1, k });
        }
    }
}
