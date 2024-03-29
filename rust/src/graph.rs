use crate::{image::Image, traits::BytesAllocated, Ternary};
use std::{error, fmt, time::Duration};

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

/// Statistical information about graphing.
#[derive(Clone, Debug)]
pub struct GraphingStatistics {
    /// The number of times the relation has been evaluated.
    pub eval_count: usize,
    /// The total number of pixels in the image.
    pub pixels: usize,
    /// The number of pixels that have been shown to be true or false.
    pub pixels_complete: usize,
    /// The total amount of time spent on refinement.
    pub time_elapsed: Duration,
}

/// An implementation of a faithful graphing algorithm.
pub trait Graph: BytesAllocated {
    /// Puts the image of the graph to `im`.
    ///
    /// The image is vertically flipped, i.e., the pixel (0, 0) is the bottom-left corner of the graph.
    fn get_image(&self, im: &mut Image<Ternary>);

    /// Returns the current statistics of graphing.
    fn get_statistics(&self) -> GraphingStatistics;

    /// Refines the graph for the specified amount of time.
    ///
    /// Returns `Ok(true)`/`Ok(false)` if graphing is complete/incomplete after refinement.
    fn refine(&mut self, duration: Duration) -> Result<bool, GraphingError>;
}

#[derive(Clone, Debug)]
pub struct Padding {
    pub bottom: u32,
    pub left: u32,
    pub right: u32,
    pub top: u32,
}

pub mod explicit;
pub mod implicit;
pub mod parametric;

mod common;
