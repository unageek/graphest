use std::ffi::OsString;

use clap::{Arg, Command};
use image::{imageops, io::Reader as ImageReader, DynamicImage, GrayAlphaImage};

fn main() {
    let matches = Command::new("join-tiles")
        .about("Concatenates tiles of graphs.")
        .arg(
            Arg::new("output")
                .long("output")
                .allow_invalid_utf8(true)
                .forbid_empty_values(true)
                .value_name("file"),
        )
        .arg(
            Arg::new("prefix")
                .long("prefix")
                .allow_invalid_utf8(true)
                .takes_value(true)
                .forbid_empty_values(true),
        )
        .arg(
            Arg::new("size")
                .long("size")
                .number_of_values(2)
                .forbid_empty_values(true)
                .value_names(&["width", "height"]),
        )
        .arg(
            Arg::new("suffix")
                .long("suffix")
                .allow_invalid_utf8(true)
                .takes_value(true)
                .forbid_empty_values(true),
        )
        .arg(
            Arg::new("x-tiles")
                .long("x-tiles")
                .takes_value(true)
                .forbid_empty_values(true),
        )
        .arg(
            Arg::new("y-tiles")
                .long("y-tiles")
                .takes_value(true)
                .forbid_empty_values(true),
        )
        .get_matches();

    let output = matches.value_of_os("output").unwrap().to_owned();
    let prefix = matches.value_of_os("prefix").unwrap().to_owned();
    let size = {
        let s = matches.values_of_t_or_exit::<u32>("size");
        [s[0], s[1]]
    };
    let suffix = matches.value_of_os("suffix").unwrap().to_owned();
    let x_tiles = matches.value_of_t_or_exit::<u32>("x-tiles");
    let y_tiles = matches.value_of_t_or_exit::<u32>("y-tiles");

    let mut im = GrayAlphaImage::new(size[0], size[1]);
    let mut i = 0;
    for i_tile in 0..y_tiles {
        let mut j = 0;
        let mut last_tile_height = None;
        for j_tile in 0..x_tiles {
            let path = [
                prefix.clone(),
                OsString::from(format!("{}-{}", i_tile, j_tile)),
                suffix.clone(),
            ]
            .into_iter()
            .collect::<OsString>();
            let tile = ImageReader::open(&path)
                .unwrap_or_else(|_| panic!("failed to open the image '{:?}'", path))
                .decode()
                .unwrap_or_else(|_| panic!("failed to decode the image '{:?}'", path));
            let tile_width = tile.width();
            let tile_height = tile.height();
            assert!(last_tile_height.is_none() || last_tile_height == Some(tile_height));
            match tile {
                DynamicImage::ImageLumaA8(tile) => {
                    imageops::replace(&mut im, &tile, j as i64, i as i64);
                }
                _ => panic!("only LumaA8 images are supported"),
            }
            last_tile_height = Some(tile_height);
            j += tile_width;
        }
        assert_eq!(j, size[0]);
        i += last_tile_height.unwrap();
    }
    assert_eq!(i, size[1]);

    im.save(output).expect("failed to save the image");
}
