use core::slice;
use graphest_webp_sys::*;
use image::RgbaImage;
use std::{
    ffi::c_void,
    fs::File,
    io,
    io::{Read, Write},
    mem::MaybeUninit,
    path::Path,
};

const WEBP_ENCODER_ABI_VERSION: i32 = 0x0210;

pub fn load_webp<P>(path: P) -> Result<RgbaImage, io::Error>
where
    P: AsRef<Path>,
{
    let mut file = File::open(path)?;
    let mut buffer = vec![];
    file.read_to_end(&mut buffer)?;

    let mut width = 0;
    let mut height = 0;
    unsafe {
        WebPGetInfo(
            buffer.as_ptr(),
            buffer.len() as u64,
            &mut width,
            &mut height,
        )
    };

    let mut im = RgbaImage::new(width as u32, height as u32);
    unsafe {
        WebPDecodeRGBAInto(
            buffer.as_ptr(),
            buffer.len() as u64,
            im.as_mut_ptr(),
            im.len() as u64,
            width,
        )
    };

    Ok(im)
}

pub fn save_webp<P>(im: &RgbaImage, path: P) -> Result<(), io::Error>
where
    P: AsRef<Path>,
{
    let mut config = {
        let mut config = MaybeUninit::<WebPConfig>::uninit();
        assert_eq!(
            unsafe {
                WebPConfigInitInternal(
                    config.as_mut_ptr(),
                    WebPPreset_WEBP_PRESET_DEFAULT,
                    75.0,
                    WEBP_ENCODER_ABI_VERSION,
                )
            },
            1
        );
        let level = 6;
        assert_eq!(
            unsafe { WebPConfigLosslessPreset(config.as_mut_ptr(), level) },
            1
        );
        unsafe { config.assume_init() }
    };
    config.exact = 1;

    assert_eq!(unsafe { WebPValidateConfig(&config) }, 1);

    let mut picture = {
        let mut picture = MaybeUninit::<WebPPicture>::uninit();
        assert_eq!(
            unsafe { WebPPictureInitInternal(picture.as_mut_ptr(), WEBP_ENCODER_ABI_VERSION) },
            1
        );
        unsafe { picture.assume_init() }
    };
    picture.width = im.width() as i32;
    picture.height = im.height() as i32;
    picture.use_argb = 1;
    picture.argb = im.as_ptr() as *mut u32;
    picture.argb_stride = im.width() as i32;

    unsafe extern "C" fn write(
        data: *const u8,
        data_size: u64,
        picture: *const WebPPicture,
    ) -> i32 {
        let mut file = &*((*picture).custom_ptr as *mut File);
        file.write_all(slice::from_raw_parts(data, data_size as usize))
            .unwrap();
        1
    }

    let mut file = File::create(path)?;
    picture.writer = Some(write);
    picture.custom_ptr = &mut file as *mut _ as *mut c_void;

    assert_eq!(unsafe { WebPEncode(&config, &mut picture) }, 1);

    unsafe { WebPPictureFree(&mut picture) };

    Ok(())
}
