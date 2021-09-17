use std::{
    mem::size_of,
    ops::{Index, IndexMut},
    slice::{Iter, IterMut},
};

/// The maximum limit of the width/height of an [`Image`] in pixels.
pub const MAX_IMAGE_WIDTH: u32 = 32768;

/// A two-dimensional image with a generic pixel type.
#[derive(Debug)]
pub struct Image<T: Clone + Copy + Default> {
    width: u32,
    height: u32,
    data: Vec<T>,
}

impl<T: Clone + Copy + Default> Image<T> {
    /// Creates a new [`Image`] with all pixels set to the default value of the type.
    pub fn new(width: u32, height: u32) -> Self {
        assert!(width > 0 && width <= MAX_IMAGE_WIDTH && height > 0 && height <= MAX_IMAGE_WIDTH);
        Self {
            width,
            height,
            data: vec![Default::default(); height as usize * width as usize],
        }
    }

    /// Returns the height of the image in pixels.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Returns an iterator over the references to the pixels of the image
    /// in the lexicographical order of `(y, x)`.
    pub fn pixels(&self) -> Iter<'_, T> {
        self.data.iter()
    }

    /// Returns an iterator over the mutable references to the pixels of the image
    /// in the lexicographical order of `(y, x)`.
    pub fn pixels_mut(&mut self) -> IterMut<'_, T> {
        self.data.iter_mut()
    }

    /// Returns the size allocated by the [`Image`] in bytes.
    pub fn size_in_heap(&self) -> usize {
        self.data.capacity() * size_of::<T>()
    }

    /// Returns the width of the image in pixels.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Returns the flattened index of the pixel.
    fn index(&self, p: PixelIndex) -> usize {
        p.y as usize * self.width as usize + p.x as usize
    }
}

impl<T: Clone + Copy + Default> Index<PixelIndex> for Image<T> {
    type Output = T;

    fn index(&self, index: PixelIndex) -> &Self::Output {
        &self.data[self.index(index)]
    }
}

impl<T: Clone + Copy + Default> IndexMut<PixelIndex> for Image<T> {
    fn index_mut(&mut self, index: PixelIndex) -> &mut Self::Output {
        let i = self.index(index);
        &mut self.data[i]
    }
}

/// The index of a pixel of an [`Image`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PixelIndex {
    /// The horizontal index of the pixel.
    pub x: u32,
    /// The vertical index of the pixel.
    pub y: u32,
}

impl PixelIndex {
    /// Creates a new [`PixelIndex`].
    pub fn new(x: u32, y: u32) -> Self {
        Self { x, y }
    }
}

/// A rectangular region of an [`Image`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PixelRange {
    begin: PixelIndex,
    end: PixelIndex,
}

impl PixelRange {
    pub const EMPTY: Self = Self {
        begin: PixelIndex { x: 0, y: 0 },
        end: PixelIndex { x: 0, y: 0 },
    };

    /// Creates a new [`PixelRange`] that spans pixels within
    /// `begin.x ≤ x < end.x` and `begin.y ≤ y < end.y`.
    pub fn new(begin: PixelIndex, end: PixelIndex) -> Self {
        assert!(begin.x <= end.x && begin.y <= end.y);
        if begin.x == end.x || begin.y == end.y {
            Self::EMPTY
        } else {
            Self { begin, end }
        }
    }

    /// Returns an iterator over the pixels in the region.
    pub fn iter(&self) -> PixelIter {
        self.into_iter()
    }
}

impl<'a> IntoIterator for &'a PixelRange {
    type Item = PixelIndex;
    type IntoIter = PixelIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        PixelIter {
            region: self,
            p: self.begin,
        }
    }
}

/// An iterator that iterates over the pixels of an [`Image`].
pub struct PixelIter<'a> {
    region: &'a PixelRange,
    p: PixelIndex,
}

impl<'a> Iterator for PixelIter<'a> {
    type Item = PixelIndex;

    fn next(&mut self) -> Option<Self::Item> {
        let p = self.p;
        if p.y == self.region.end.y {
            return None;
        }

        let mut x = p.x + 1;
        let mut y = p.y;
        if x == self.region.end.x {
            x = self.region.begin.x;
            y += 1;
        }
        self.p = PixelIndex::new(x, y);

        Some(p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image() {
        let mut im = Image::<i32>::new(34, 45);
        let p = PixelIndex::new(12, 23);

        assert_eq!(im.width(), 34);
        assert_eq!(im.height(), 45);

        assert_eq!(im[p], 0);
        im[p] = 123456;
        assert_eq!(im[p], 123456);
        assert_eq!(
            im.pixels().copied().nth((p.y * im.width() + p.x) as usize),
            Some(123456)
        );
    }

    #[test]
    fn pixel_range() {
        let r = PixelRange::new(PixelIndex::new(1, 2), PixelIndex::new(1, 2));
        let mut iter = r.iter();
        assert_eq!(iter.next(), None);

        let r = PixelRange::new(PixelIndex::new(1, 2), PixelIndex::new(4, 2));
        let mut iter = r.iter();
        assert_eq!(iter.next(), None);

        let r = PixelRange::new(PixelIndex::new(1, 2), PixelIndex::new(1, 8));
        let mut iter = r.iter();
        assert_eq!(iter.next(), None);

        let r = PixelRange::new(PixelIndex::new(1, 2), PixelIndex::new(4, 8));
        let mut iter = r.iter();
        for y in 2..8 {
            for x in 1..4 {
                assert_eq!(iter.next(), Some(PixelIndex::new(x, y)));
            }
        }
        assert_eq!(iter.next(), None);
    }
}
