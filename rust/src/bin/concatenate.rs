use clap::{value_parser, Arg, Command};
use image::{imageops, ImageBuffer, ImageReader, LumaA};
use std::ffi::OsString;

type GrayAlpha16Image = ImageBuffer<LumaA<u16>, Vec<u16>>;

fn main() {
    let mut matches = Command::new("concatenate")
        .about("Concatenates tiles of graphs.")
        .arg(
            Arg::new("output")
                .long("output")
                .value_name("file")
                .value_parser(value_parser!(OsString)),
        )
        .arg(
            Arg::new("prefix")
                .long("prefix")
                .value_parser(value_parser!(OsString)),
        )
        .arg(
            Arg::new("size")
                .long("size")
                .num_args(2)
                .value_names(["width", "height"])
                .value_parser(value_parser!(u32)),
        )
        .arg(
            Arg::new("suffix")
                .long("suffix")
                .value_parser(value_parser!(OsString)),
        )
        .arg(
            Arg::new("x-tiles")
                .long("x-tiles")
                .value_parser(value_parser!(u32)),
        )
        .arg(
            Arg::new("y-tiles")
                .long("y-tiles")
                .value_parser(value_parser!(u32)),
        )
        .get_matches();

    let output = matches.remove_one::<OsString>("output").unwrap();
    let prefix = matches.remove_one::<OsString>("prefix").unwrap();
    let size = {
        let s = matches
            .remove_many::<u32>("size")
            .unwrap()
            .collect::<Vec<_>>();
        [s[0], s[1]]
    };
    let suffix = matches.remove_one::<OsString>("suffix").unwrap();
    let x_tiles = matches.remove_one::<u32>("x-tiles").unwrap();
    let y_tiles = matches.remove_one::<u32>("y-tiles").unwrap();

    let mut im = GrayAlpha16Image::new(size[0], size[1]);
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
                .unwrap_or_else(|_| panic!("failed to decode the image '{:?}'", path))
                .into_luma_alpha16();
            let tile_width = tile.width();
            let tile_height = tile.height();
            assert!(last_tile_height.is_none() || last_tile_height == Some(tile_height));
            imageops::replace(&mut im, &tile, j as i64, i as i64);
            last_tile_height = Some(tile_height);
            j += tile_width;
        }
        assert_eq!(j, size[0]);
        i += last_tile_height.unwrap();
    }
    assert_eq!(i, size[1]);

    im.save(output).expect("failed to save the image");
}
