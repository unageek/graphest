use std::{mem::size_of, slice::Iter};

/// The limit of the width/height of an [`Image`] in pixels.
const MAX_IMAGE_WIDTH: u32 = 32768;

/// A two-dimensional image.
#[derive(Debug)]
pub struct Image<T: Clone + Copy + Default> {
    width: u32,
    height: u32,
    data: Vec<T>,
}

impl<T: Clone + Copy + Default> Image<T> {
    /// Creates a new [`Image`] with all pixels set to the default values.
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

    /// Returns the value of the pixel.
    pub fn get(&self, p: PixelIndex) -> T {
        self.data[self.index(p)]
    }

    /// Returns a mutable reference to the value of the pixel.
    pub fn get_mut(&mut self, p: PixelIndex) -> &mut T {
        let i = self.index(p);
        &mut self.data[i]
    }

    /// Returns the iterator over the values in the lexicographical order of
    /// of (`PixelIndex.y`, `PixelIndex.x`).
    pub fn iter(&self) -> Iter<'_, T> {
        self.data.iter()
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image() {
        let mut im = Image::<i32>::new(34, 45);
        let p = PixelIndex::new(12, 23);

        assert_eq!(im.width(), 34);
        assert_eq!(im.height(), 45);

        assert_eq!(im.get(p), 0);
        *im.get_mut(p) = 123456;
        assert_eq!(im.get(p), 123456);
        assert_eq!(
            im.iter().copied().nth((p.y * im.width() + p.x) as usize),
            Some(123456)
        );
    }
}
