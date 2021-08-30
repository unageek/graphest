use image::{ImageBuffer, Pixel};
use std::{
    error, fmt,
    ops::{Deref, DerefMut},
    time::Duration,
};

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

/// Statistical information of graphing.
#[derive(Clone, Debug)]
pub struct GraphingStatistics {
    /// The total numer of pixels.
    pub pixels: usize,
    /// The number of pixels that are proven to be either true or false.
    pub pixels_proven: usize,
    /// The number of times the relation has been evaluated.
    pub eval_count: usize,
    /// The total amount of time spent on evaluation.
    pub time_elapsed: Duration,
}

/// An implementation of a faithful graphing algorithm.
pub trait Graph {
    /// Puts the image of the graph to `im` with the specified colors.
    fn get_image<P, Container>(
        &self,
        im: &mut ImageBuffer<P, Container>,
        true_color: P,
        uncertain_color: P,
        false_color: P,
    ) where
        P: Pixel + 'static,
        Container: Deref<Target = [P::Subpixel]> + DerefMut;

    /// Returns the current statistics of graphing.
    fn get_statistics(&self) -> GraphingStatistics;

    /// Refines the graph for the specified amount of time.
    ///
    /// Returns `Ok(true)`/`Ok(false)` if graphing is complete/incomplete after refinement.
    fn refine(&mut self, duration: Duration) -> Result<bool, GraphingError>;

    /// Returns the amount of memory allocated by `self` in bytes.
    fn size_in_heap(&self) -> usize;
}

mod common;
mod explicit;
mod implicit;
mod parametric;

pub use crate::geom::Box2D;
pub use explicit::Explicit;
pub use implicit::Implicit;
pub use parametric::Parametric;
