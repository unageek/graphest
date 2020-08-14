use crate::interval_set::*;
use image::{imageops, Rgb, RgbImage};
use inari::{dec_interval, interval, DecoratedInterval, Decoration, Interval};
use std::{
    collections::HashMap,
    error, fmt, mem,
    ops::FnMut,
    time::{Duration, Instant},
};

const MEM_LIMIT: usize = 1usize << 30; // 1GiB
const MIN_K: i32 = -15;

const STAT_FALSE: u32 = 0;
const STAT_UNCERTAIN: u32 = 1u32 << (2 * -MIN_K);
const STAT_TRUE: u32 = !0u32;

// Each pixel of an `Image` keeps track of the proof status as follows.
//   STAT_FALSE: the relation has been proven to be false on the pixel.
//   1..STAT_UNCERTAIN: uncertain, but proven to be false on a part of the pixel.
//   STAT_UNCERTAIN: uncertain.
//   STAT_TRUE: proven to be true.
#[derive(Debug)]
struct Image {
    width: u32,
    height: u32,
    data: Vec<u32>,
}

impl Image {
    fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            data: vec![STAT_UNCERTAIN; height as usize * width as usize],
        }
    }

    fn pixel(&self, x: u32, y: u32) -> u32 {
        self.data[self.pixel_index(x, y)]
    }

    fn pixel_index(&self, x: u32, y: u32) -> usize {
        y as usize * self.width as usize + x as usize
    }

    fn pixel_mut(&mut self, x: u32, y: u32) -> &mut u32 {
        let i = self.pixel_index(x, y);
        &mut self.data[i]
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct ImageBlock(u32, u32);

// Represents a set of square regions of an image.
#[derive(Debug)]
struct ImageBlockSet {
    blocks: Vec<ImageBlock>,
    block_width: f64, // 2^k
    k: i32,           // TODO: Negate and rename to `level`?
}

// Represents a rectangular region (subset of ℝ²).
#[derive(Debug, Clone)]
pub struct Region(Interval, Interval);

impl Region {
    pub fn new(l: f64, r: f64, b: f64, t: f64) -> Self {
        // Regions constructed directly do not need to satisfy these conditions.
        assert!(l < r && b < t && l.is_finite() && r.is_finite() && b.is_finite() && t.is_finite());
        Self(interval!(l, r).unwrap(), interval!(b, t).unwrap())
    }

    fn height(&self) -> Interval {
        interval!(self.1.sup(), self.1.sup()).unwrap()
            - interval!(self.1.inf(), self.1.inf()).unwrap()
    }

    fn intersection(&self, rhs: &Self) -> Self {
        Self(self.0.intersection(rhs.0), self.1.intersection(rhs.1))
    }

    fn is_empty(&self) -> bool {
        self.0.is_empty() || self.1.is_empty()
    }

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

    fn outer(&self) -> Region {
        Region(
            interval!(self.l.inf(), self.r.sup()).unwrap(),
            interval!(self.b.inf(), self.t.sup()).unwrap(),
        )
    }

    fn subpixel_outer(&self, ix: u32, iy: u32, bx: u32, by: u32, nbx: u32) -> Region {
        let rem_x = bx - ix * nbx; // bx % nbx
        let rem_y = by - iy * nbx; // by % nbx
        let l = if rem_x == 0 {
            self.l.inf()
        } else {
            self.l.mid()
        };
        let r = if rem_x == nbx - 1 {
            self.r.sup()
        } else {
            self.r.mid()
        };
        let b = if rem_y == 0 {
            self.b.inf()
        } else {
            self.b.mid()
        };
        let t = if rem_y == nbx - 1 {
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

#[derive(Debug)]
pub struct Relation<T>
where
    T: FnMut(TupperIntervalSet, TupperIntervalSet) -> EvalResult,
{
    pub eval: T,
    pub prop: Proposition,
}

impl<T> Relation<T>
where
    T: FnMut(TupperIntervalSet, TupperIntervalSet) -> EvalResult,
{
    fn eval_on_point(&mut self, x: f64, y: f64) -> EvalResult {
        let x = dec_interval!(x, x).unwrap();
        let y = dec_interval!(y, y).unwrap();
        (self.eval)(TupperIntervalSet::from(x), TupperIntervalSet::from(y))
    }

    fn eval_on_region(&mut self, r: &Region) -> EvalResult {
        let x = DecoratedInterval::new(r.0);
        let y = DecoratedInterval::new(r.1);
        (self.eval)(TupperIntervalSet::from(x), TupperIntervalSet::from(y))
    }
}

#[derive(Clone, Debug)]
pub struct GraphingStatistics {
    pub pixels: usize,
    pub pixels_proven: usize,
    pub evaluations: usize,
    pub time_elapsed: Duration,
}

#[derive(Debug)]
pub struct Graph<T>
where
    T: FnMut(TupperIntervalSet, TupperIntervalSet) -> EvalResult,
{
    relation: Relation<T>,
    im: Image,
    bs: ImageBlockSet,
    // Affine transformation from pixel coordinates to real coordinates.
    sx: Interval,
    sy: Interval,
    tx: Interval,
    ty: Interval,
    stats: GraphingStatistics,
}

impl<T> Graph<T>
where
    T: FnMut(TupperIntervalSet, TupperIntervalSet) -> EvalResult,
{
    // TODO: Accept `InexactRegion` instead of `Region` for more exactness?
    pub fn new(relation: Relation<T>, region: Region, im_width: u32, im_height: u32) -> Self {
        assert!(im_width > 0 && im_height > 0);
        let mut g = Self {
            relation,
            im: Image::new(im_width, im_height),
            bs: ImageBlockSet {
                blocks: Vec::new(),
                block_width: 0.0,
                k: 0,
            },
            sx: region.width() / Self::point_interval(im_width as f64),
            sy: region.height() / Self::point_interval(im_height as f64),
            tx: Self::point_interval(region.0.inf()),
            ty: Self::point_interval(region.1.inf()),
            stats: GraphingStatistics {
                pixels: im_width as usize * im_height as usize,
                pixels_proven: 0,
                evaluations: 0,
                time_elapsed: Duration::new(0, 0),
            },
        };
        g.bs = g.get_initial_image_blocks();
        g
    }

    pub fn step(&mut self) -> Result<bool, GraphingError> {
        if self.bs.blocks.is_empty() {
            return Ok(true);
        }

        let mut evals = 0usize;
        let now = Instant::now();

        if self.bs.k >= 0 {
            self.bs = self.refine_pixels(&mut evals)?;
        } else {
            self.bs = self.refine_subpixels(&mut evals)?;
        }

        self.stats.evaluations += evals;
        self.stats.time_elapsed += now.elapsed();

        Ok(self.bs.blocks.is_empty())
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
            ..self.stats
        }
    }

    fn get_initial_image_blocks(&self) -> ImageBlockSet {
        let k = (self.im.width.max(self.im.height) as f64).log2() as i32;
        let bw = 2.0f64.powi(k);
        let nbc = (self.im.width as f64 / bw).ceil() as u32;
        let nbr = (self.im.height as f64 / bw).ceil() as u32;
        let mut blocks = Vec::<ImageBlock>::new();
        for by in 0..nbr {
            for bx in 0..nbc {
                blocks.push(ImageBlock(bx, by));
            }
        }
        ImageBlockSet {
            blocks,
            block_width: bw,
            k: k as i32,
        }
    }

    fn refine_pixels(&mut self, evals: &mut usize) -> Result<ImageBlockSet, GraphingError> {
        let bs = &self.bs;
        let bw = bs.block_width;
        let ibw = bw as u32;
        let sub_bw = bw / 2.0;
        let sub_nbc = (self.im.width as f64 / sub_bw).ceil() as u32;
        let sub_nbr = (self.im.height as f64 / sub_bw).ceil() as u32;
        let mut sub_blocks = Vec::<ImageBlock>::new();
        for ImageBlock(bx, by) in bs.blocks.iter().copied() {
            let u_up = self.image_block_to_region_clipped(bx, by, bw).outer();
            let r_u_up = self.relation.eval_on_region(&u_up);
            *evals += 1;

            let prop = &self.relation.prop;
            let is_true =
                r_u_up.map_reduce(prop, &|ss, d| d >= Decoration::Def && ss == SignSet::ZERO);
            let is_false = !r_u_up.map_reduce(prop, &|ss, _| ss.contains(SignSet::ZERO));
            if is_true || is_false {
                let ix = bx * ibw;
                let iy = by * ibw;
                let stat = if is_true { STAT_TRUE } else { STAT_FALSE };
                for iy in iy..(iy + ibw).min(self.im.height) {
                    for ix in ix..(ix + ibw).min(self.im.width) {
                        *self.im.pixel_mut(ix, iy) = stat;
                    }
                }
            } else {
                Self::push_sub_blocks_clipped(&mut sub_blocks, bx, by, sub_nbc, sub_nbr);
                if (bs.blocks.capacity() + sub_blocks.capacity()) * mem::size_of::<ImageBlock>()
                    > MEM_LIMIT
                {
                    return Err(GraphingError {
                        kind: GraphingErrorKind::ReachedMemLimit,
                    });
                }
            }
        }
        Ok(ImageBlockSet {
            blocks: sub_blocks,
            block_width: sub_bw,
            k: bs.k - 1,
        })
    }

    fn refine_subpixels(&mut self, evals: &mut usize) -> Result<ImageBlockSet, GraphingError> {
        let bs = &self.bs;
        let fbw = bs.block_width;
        let nbx = 1u32 << -bs.k; // Number of blocks in each row per pixel.
        let area = 1u32 << (2 * (bs.k - MIN_K));
        let mut cache = HashMap::<ImageBlock, EvalResult>::with_capacity(bs.blocks.len());
        let mut size_of_cached_payloads = 0usize;
        let mut some_test_failed = false;
        let mut sub_blocks = Vec::<ImageBlock>::new();
        for ImageBlock(bx, by) in bs.blocks.iter().copied() {
            let ix = bx >> -bs.k;
            let iy = by >> -bs.k;
            let stat = self.im.pixel(ix, iy);
            if stat == STAT_FALSE || stat == STAT_TRUE {
                continue;
            }

            let p_dn = self.image_block_to_region(ix, iy, 1.0).inner();
            if p_dn.is_empty() {
                some_test_failed = true;
                continue;
            }

            let u_up = self
                .image_block_to_region(bx, by, fbw)
                .subpixel_outer(ix, iy, bx, by, nbx);
            let r_u_up = self.relation.eval_on_region(&u_up);
            *evals += 1;

            if r_u_up.map_reduce(&self.relation.prop, &|ss, _| ss == SignSet::ZERO) {
                // This pixel is proven to be true.
                *self.im.pixel_mut(ix, iy) = STAT_TRUE;
                continue;
            }
            if !r_u_up.map_reduce(&self.relation.prop, &|ss, _| ss.contains(SignSet::ZERO)) {
                // This subpixel is proven to be false.
                *self.im.pixel_mut(ix, iy) -= area;
                continue;
            }

            let inter = u_up.intersection(&p_dn);
            if inter.is_empty() {
                *self.im.pixel_mut(ix, iy) -= area;
                continue;
            }

            // We could re-evaluate the relation on `inter` instead of `u_up`
            // to get a slightly better result, but the effect would be
            // negligible.

            // To prove the existence of a solution by a change of sign...
            //   for conjunctions, both operands must be `Dac`.
            //   for disjunctions, at least one operand must be `Dac`.
            // There is a little chance that an expression is evaluated
            // to zero on one of the probe points. In that case,
            // the expression is not required to be `Dac` on the entire
            // subpixel. We don't care such a chance.
            let dac_mask = r_u_up.map(&self.relation.prop, &|_, d| d >= Decoration::Dac);
            if dac_mask.reduce(&self.relation.prop) {
                // Suppose we are plotting the graph of a conjunction such as
                // "y == sin(x) && x >= 0".
                // If the conjunct "x >= 0" holds everywhere in the subpixel,
                // and "y - sin(x)" evaluates to both `POS` and `NEG` at
                // different points in the region, we can tell that
                // there exists a point where the entire relation holds.
                // Such a test is not possible by merely converting
                // the relation to "|y - sin(x)| + |x >= 0 ? 0 : 1| == 0".
                let locally_zero_mask = r_u_up.map(&self.relation.prop, &|ss, d| {
                    ss == SignSet::ZERO && d >= Decoration::Dac
                });

                // Use `(cx, cy)` instead of `(bx, by)` for cache indices
                // so that values at the pixel boundary may not be shared
                // among the adjacent pixels.
                //
                //                   Values are different at these two points.
                //                     vv
                //   +  ||  +   +   +  ||  +   +   +  ||  +
                //       \_____________/
                //             p_dn
                // The exact pixel boundary is somewhere between ||.
                //
                // We could keep the cache across levels by multiplying
                // `cy` and `cy` by `1u32 << (bs.k - MIN_K)`, but that
                // increased graphing time significantly.
                let cx = bx + ix;
                let cy = by + iy;

                let rel = &mut self.relation;
                let mut found_solution = false;
                let mut neg_mask = r_u_up.map(&rel.prop, &|_, _| false);
                let mut pos_mask = neg_mask.clone();
                for i in 0..4 {
                    let r = match i {
                        0 => cache.entry(ImageBlock(cx, cy)).or_insert_with(|| {
                            // bottom left
                            let res = rel.eval_on_point(inter.0.inf(), inter.1.inf());
                            *evals += 1;
                            size_of_cached_payloads += res.get_size_of_payload();
                            res
                        }),
                        1 => cache.entry(ImageBlock(cx + 1, cy)).or_insert_with(|| {
                            // bottom right
                            let res = rel.eval_on_point(inter.0.sup(), inter.1.inf());
                            *evals += 1;
                            size_of_cached_payloads += res.get_size_of_payload();
                            res
                        }),
                        2 => cache.entry(ImageBlock(cx, cy + 1)).or_insert_with(|| {
                            // top left
                            let res = rel.eval_on_point(inter.0.inf(), inter.1.sup());
                            *evals += 1;
                            size_of_cached_payloads += res.get_size_of_payload();
                            res
                        }),
                        3 => cache.entry(ImageBlock(cx + 1, cy + 1)).or_insert_with(|| {
                            // top right
                            let res = rel.eval_on_point(inter.0.sup(), inter.1.sup());
                            *evals += 1;
                            size_of_cached_payloads += res.get_size_of_payload();
                            res
                        }),
                        _ => unreachable!(),
                    };

                    neg_mask |= r.map(&rel.prop, &|ss, _| {
                        ss == SignSet::NEG || ss == SignSet::ZERO
                    });
                    pos_mask |= r.map(&rel.prop, &|ss, _| {
                        ss == SignSet::POS || ss == SignSet::ZERO
                    });

                    if (&(&neg_mask & &pos_mask) & &dac_mask)
                        .implies_solution(&rel.prop, &locally_zero_mask)
                    {
                        found_solution = true;
                        break;
                    }
                }

                if found_solution {
                    *self.im.pixel_mut(ix, iy) = STAT_TRUE;
                    continue;
                }
            }

            if bs.k > MIN_K {
                Self::push_sub_blocks(&mut sub_blocks, bx, by);
                if (bs.blocks.capacity() + sub_blocks.capacity()) * mem::size_of::<ImageBlock>()
                    + cache.capacity()
                        * (mem::size_of::<u64>()
                            + mem::size_of::<ImageBlock>()
                            + mem::size_of::<EvalResult>())
                    + size_of_cached_payloads
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
            Ok(ImageBlockSet {
                blocks: sub_blocks,
                block_width: fbw / 2.0,
                k: bs.k - 1,
            })
        }
    }

    fn image_block_to_region(&self, bx: u32, by: u32, bw: f64) -> InexactRegion {
        InexactRegion {
            l: Self::point_interval(bx as f64 * bw).mul_add(self.sx, self.tx),
            r: Self::point_interval((bx + 1) as f64 * bw).mul_add(self.sx, self.tx),
            b: Self::point_interval(by as f64 * bw).mul_add(self.sy, self.ty),
            t: Self::point_interval((by + 1) as f64 * bw).mul_add(self.sy, self.ty),
        }
    }

    fn image_block_to_region_clipped(&self, bx: u32, by: u32, bw: f64) -> InexactRegion {
        InexactRegion {
            l: Self::point_interval(bx as f64 * bw).mul_add(self.sx, self.tx),
            r: Self::point_interval(((bx + 1) as f64 * bw).min(self.im.width as f64))
                .mul_add(self.sx, self.tx),
            b: Self::point_interval(by as f64 * bw).mul_add(self.sy, self.ty),
            t: Self::point_interval(((by + 1) as f64 * bw).min(self.im.height as f64))
                .mul_add(self.sy, self.ty),
        }
    }

    fn point_interval(x: f64) -> Interval {
        interval!(x, x).unwrap()
    }

    fn push_sub_blocks(blocks: &mut Vec<ImageBlock>, bx: u32, by: u32) {
        let sub_bx = 2 * bx;
        let sub_by = 2 * by;
        blocks.push(ImageBlock(sub_bx, sub_by));
        blocks.push(ImageBlock(sub_bx + 1, sub_by));
        blocks.push(ImageBlock(sub_bx, sub_by + 1));
        blocks.push(ImageBlock(sub_bx + 1, sub_by + 1));
    }

    fn push_sub_blocks_clipped(
        blocks: &mut Vec<ImageBlock>,
        bx: u32,
        by: u32,
        sub_nbc: u32,
        sub_nbr: u32,
    ) {
        let sub_bx = 2 * bx;
        let sub_by = 2 * by;
        blocks.push(ImageBlock(sub_bx, sub_by));
        if sub_bx + 1 < sub_nbc {
            blocks.push(ImageBlock(sub_bx + 1, sub_by));
        }
        if sub_by + 1 < sub_nbr {
            blocks.push(ImageBlock(sub_bx, sub_by + 1));
        }
        if sub_bx + 1 < sub_nbc && sub_by + 1 < sub_nbr {
            blocks.push(ImageBlock(sub_bx + 1, sub_by + 1));
        }
    }
}
