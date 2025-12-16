use std::{
    collections::{HashMap, VecDeque},
    mem::size_of,
};

pub trait BytesAllocated {
    /// Returns the approximate amount of memory allocated by `self` in bytes.
    fn bytes_allocated(&self) -> usize;
}

impl<K, V> BytesAllocated for HashMap<K, V> {
    fn bytes_allocated(&self) -> usize {
        self.capacity() * (size_of::<u64>() + size_of::<K>() + size_of::<V>())
    }
}

impl<T> BytesAllocated for Option<T>
where
    T: BytesAllocated,
{
    fn bytes_allocated(&self) -> usize {
        match self {
            Some(value) => value.bytes_allocated(),
            None => 0,
        }
    }
}

impl<T> BytesAllocated for Vec<T> {
    fn bytes_allocated(&self) -> usize {
        self.capacity() * size_of::<T>()
    }
}

impl<T> BytesAllocated for VecDeque<T> {
    fn bytes_allocated(&self) -> usize {
        self.capacity() * size_of::<T>()
    }
}

impl<T, const N: usize> BytesAllocated for smallvec::SmallVec<[T; N]>
where
    [T; N]: smallvec::Array,
{
    fn bytes_allocated(&self) -> usize {
        if self.spilled() {
            self.capacity() * size_of::<T>()
        } else {
            0
        }
    }
}

pub trait Single: Iterator {
    fn single(self) -> Option<Self::Item>;
}

impl<I: Iterator> Single for I {
    fn single(mut self) -> Option<Self::Item> {
        match (self.next(), self.next()) {
            (Some(item), None) => Some(item),
            _ => None,
        }
    }
}
