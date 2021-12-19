pub trait BytesAllocated {
    /// Returns the approximate amount of memory allocated by `self` in bytes.
    fn bytes_allocated(&self) -> usize;
}
