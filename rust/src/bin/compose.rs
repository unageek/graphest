use clap::{Arg, Command};
use image::{imageops, io::Reader as ImageReader, DynamicImage, Rgba, Rgba32FImage};

#[derive(Clone, Debug)]
struct Entry {
    color: Rgba<f32>,
    file: String,
}

fn colorize(im: &mut Rgba32FImage, color: Rgba<f32>) {
    for p in im.pixels_mut() {
        *p = Rgba([color[0], color[1], color[2], p[3] * color[3]]);
    }
}

/// Converts the color space of the image from linearized sRGB to sRGB.
fn linear_to_srgb(im: &mut Rgba32FImage) {
    fn linear_to_srgb(c: f32) -> f32 {
        if c <= 0.0031308 {
            12.92 * c
        } else {
            1.055 * c.powf(1.0 / 2.4) - 0.055
        }
    }

    for p in im.pixels_mut() {
        *p = Rgba([
            linear_to_srgb(p[0]),
            linear_to_srgb(p[1]),
            linear_to_srgb(p[2]),
            p[3],
        ]);
    }
}

/// Parses a hex color string and returns the result as a linearized sRGB value.
fn parse_color(color: &str) -> Rgba<f32> {
    fn srgb_to_linear(c: f32) -> f32 {
        if c <= 0.04045 {
            c / 12.92
        } else {
            ((c + 0.055) / 1.055).powf(2.4)
        }
    }

    let chars = color.chars().collect::<Vec<_>>();
    assert_eq!(chars.len(), 9);
    assert_eq!(chars[0], '#');
    let digits = chars[1..]
        .iter()
        .map(|c| c.to_digit(16).unwrap())
        .collect::<Vec<_>>();
    let r = (16 * digits[0] + digits[1]) as f32 / 255.0;
    let g = (16 * digits[2] + digits[3]) as f32 / 255.0;
    let b = (16 * digits[4] + digits[5]) as f32 / 255.0;
    let a = (16 * digits[6] + digits[7]) as f32 / 255.0;
    Rgba([srgb_to_linear(r), srgb_to_linear(g), srgb_to_linear(b), a])
}

fn main() {
    let matches = Command::new("compose")
        .about("Colorizes and alpha-composes gray-alpha images.")
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
        .arg(Arg::new("transparent").long("transparent"))
        .get_matches();

    let entries = matches
        .values_of("add")
        .unwrap()
        .collect::<Vec<_>>()
        .chunks_exact(2)
        .map(|e| Entry {
            color: parse_color(e[1]),
            file: e[0].into(),
        })
        .collect::<Vec<_>>();
    let output = matches.value_of_os("output").unwrap().to_owned();
    let transparent = matches.is_present("transparent");

    let mut composed = None;
    for entry in entries {
        let mut im = ImageReader::open(&entry.file)
            .unwrap_or_else(|_| panic!("failed to open the image '{}'", entry.file))
            .decode()
            .unwrap_or_else(|_| panic!("failed to decode the image '{}'", entry.file))
            .into_rgba32f();
        let composed = composed.get_or_insert_with(|| {
            let mut composed = Rgba32FImage::new(im.width(), im.height());
            composed.fill(if transparent { 0.0 } else { 1.0 });
            composed
        });

        assert_eq!(im.width(), composed.width());
        assert_eq!(im.height(), composed.height());

        colorize(&mut im, entry.color);
        imageops::overlay(composed, &im, 0, 0);
    }

    if let Some(mut composed) = composed {
        linear_to_srgb(&mut composed);
        DynamicImage::ImageRgba32F(composed)
            .to_rgba8()
            .save(output)
            .expect("failed to save the image");
    }
}
