use graphest_fftw_sys::*;
use std::ffi::c_void;
use std::ops::{Index, IndexMut};
use std::slice::{from_raw_parts, from_raw_parts_mut};

// From https://www.fftw.org/fftw3_doc/Introduction.html
//
//   On the other hand, if you need a single transform of a given size,
//   the one-time cost of the planner becomes significant. For this case,
//   FFTW provides fast planners based on heuristics or on previously computed plans.
//
// The better choice would be to utilize FFTW wisdom.
const FFTW_ESTIMATE: u32 = 1 << 6;

pub struct FftImage {
    ptr: *mut f32,
    len: usize,
    stride: usize,
    plan_r2c: fftwf_plan,
    plan_c2r: fftwf_plan,
}

impl FftImage {
    pub fn new(width: u32, height: u32) -> Self {
        // For the stride, see
        //   https://www.fftw.org/fftw3_doc/Multi_002dDimensional-DFTs-of-Real-Data.html
        //   https://www.fftw.org/fftw3_doc/Real_002ddata-DFT-Array-Format.html
        let stride = 2 * (width as usize / 2 + 1);
        let len = height as usize * stride;
        let ptr = unsafe { fftwf_alloc_real(len as u64) };
        // Unless `FFTW_ESTIMATE` is set, which we currently do, the arrays are destroyed during planning.
        //   https://www.fftw.org/fftw3_doc/Planner-Flags.html
        let plan_r2c = unsafe {
            fftwf_plan_dft_r2c_2d(
                height as i32,
                width as i32,
                ptr,
                ptr as *mut [f32; 2],
                FFTW_ESTIMATE,
            )
        };
        let plan_c2r = unsafe {
            fftwf_plan_dft_c2r_2d(
                height as i32,
                width as i32,
                ptr as *mut [f32; 2],
                ptr,
                FFTW_ESTIMATE,
            )
        };
        unsafe { from_raw_parts_mut(ptr, len).fill(0.0) };
        Self {
            ptr,
            len,
            stride,
            plan_r2c,
            plan_c2r,
        }
    }

    pub fn complexes(&self) -> &[[f32; 2]] {
        unsafe { from_raw_parts(self.ptr as *const [f32; 2], self.len / 2) }
    }

    pub fn complexes_mut(&mut self) -> &mut [[f32; 2]] {
        unsafe { from_raw_parts_mut(self.ptr as *mut [f32; 2], self.len / 2) }
    }

    pub fn fft(&mut self) {
        unsafe { fftwf_execute(self.plan_r2c) };
    }

    pub fn ifft(&mut self) {
        unsafe { fftwf_execute(self.plan_c2r) };
    }
}

impl Drop for FftImage {
    fn drop(&mut self) {
        unsafe {
            fftwf_free(self.ptr as *mut c_void);
            fftwf_destroy_plan(self.plan_r2c);
            fftwf_destroy_plan(self.plan_c2r);
        };
    }
}

impl Index<usize> for FftImage {
    type Output = [f32];

    fn index(&self, index: usize) -> &Self::Output {
        let stride = self.stride;
        let slice = unsafe { from_raw_parts(self.ptr, self.len) };
        &slice[stride * index..stride * (index + 1)]
    }
}

impl IndexMut<usize> for FftImage {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        let stride = self.stride;
        let slice_mut = unsafe { from_raw_parts_mut(self.ptr, self.len) };
        &mut slice_mut[stride * index..stride * (index + 1)]
    }
}
