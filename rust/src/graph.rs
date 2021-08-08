use image::{GrayAlphaImage, RgbImage};
use std::{error, fmt, time::Duration};

/// The graphing status of a pixel.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PixelState {
    /// There may be or may not be a solution in the pixel.
    Uncertain,
    /// Uncertain but we can't prove absence of solutions due to subdivision limit.
    UncertainNeverFalse,
    /// There are no solutions in the pixel.
    False,
    /// There is at least one solution in the pixel.
    True,
}

impl Default for PixelState {
    fn default() -> Self {
        PixelState::Uncertain
    }
}

/// The index of a [`Block`] in a [block-queue].
///
/// Indices returned by the methods of [block-queue] are [`usize`], but [`u32`] would be large enough.
///
/// [block-queue]: [`crate::block::BlockQueue`]
pub type QueuedBlockIndex = u32;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GraphingErrorKind {
    BlockIndexOverflow,
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
            GraphingErrorKind::BlockIndexOverflow => write!(f, "block index overflow"),
            GraphingErrorKind::ReachedMemLimit => write!(f, "reached the memory usage limit"),
            GraphingErrorKind::ReachedSubdivisionLimit => {
                write!(f, "reached the subdivision limit")
            }
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

/// A faithful graphing algorithm.
pub trait Graph {
    fn get_gray_alpha_image(&self, im: &mut GrayAlphaImage);

    fn get_image(&self, im: &mut RgbImage);

    fn get_statistics(&self) -> GraphingStatistics;

    /// Refines the graph for a given amount of time.
    ///
    /// Returns `Ok(true)`/`Ok(false)` if graphing is complete/incomplete after refinement.
    fn refine(&mut self, timeout: Duration) -> Result<bool, GraphingError>;
}

mod implicit;
mod parametric;

pub use crate::region::InexactRegion;
pub use implicit::Implicit;
pub use parametric::Parametric;
