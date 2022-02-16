use clap::{App, Arg};
use image::{imageops::overlay, io::Reader as ImageReader, DynamicImage, Rgb, Rgba, RgbaImage};

#[derive(Clone, Debug)]
struct Entry {
    color: Rgb<u8>,
    file: String,
}

fn parse_color(color: &str) -> Rgb<u8> {
    let chars = color.chars().collect::<Vec<_>>();
    assert_eq!(chars.len(), 7);
    assert_eq!(chars[0], '#');
    let r = 16 * chars[1].to_digit(16).unwrap() + chars[2].to_digit(16).unwrap();
    let g = 16 * chars[3].to_digit(16).unwrap() + chars[4].to_digit(16).unwrap();
    let b = 16 * chars[5].to_digit(16).unwrap() + chars[6].to_digit(16).unwrap();
    Rgb([r as u8, g as u8, b as u8])
}

fn sepia_tone(src: &DynamicImage, color: Rgb<u8>, dst: &mut RgbaImage) {
    match src {
        DynamicImage::ImageLumaA8(im) => {
            for (src, dst) in im.pixels().zip(dst.pixels_mut()) {
                *dst = Rgba([color[0], color[1], color[2], src[1]]);
            }
        }
        _ => panic!("only LumaA8 images are supported"),
    }
}

fn main() {
    let matches = App::new("compose")
        .about("Colorizes and alpha-composes graph-alpha graph images.")
        .arg(
            Arg::new("add")
                .long("add")
                .number_of_values(2)
                .multiple_occurrences(true)
                .forbid_empty_values(true)
                .value_names(&["file", "color"]),
        )
        .arg(
            Arg::new("output")
                .long("output")
                .allow_invalid_utf8(true)
                .forbid_empty_values(true)
                .value_name("file"),
        )
        .get_matches();

    let entries = matches
        .values_of("add")
        .unwrap()
        .collect::<Vec<_>>()
        .windows(2)
        .map(|e| Entry {
            color: parse_color(e[1]),
            file: e[0].into(),
        })
        .collect::<Vec<_>>();
    let output = matches.value_of_os("output").unwrap().to_owned();

    let mut sepia = None;
    let mut composed = None;
    for entry in entries {
        let im = ImageReader::open(&entry.file)
            .expect("failed to open the image '{file}'")
            .decode()
            .expect("failed to decode the image '{file}'");
        let sepia = sepia.get_or_insert_with(|| RgbaImage::new(im.width(), im.height()));
        let composed = composed.get_or_insert_with(|| {
            let mut composed = RgbaImage::new(im.width(), im.height());
            composed.fill(255);
            composed
        });
        assert_eq!(im.width(), sepia.width());
        assert_eq!(im.width(), sepia.height());
        sepia_tone(&im, entry.color, sepia);
        overlay(composed, sepia, 0, 0);
    }

    if let Some(composed) = composed {
        composed.save(output).expect("failed to save the image");
    }
}
