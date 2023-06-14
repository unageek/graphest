use clap::{value_parser, Arg, ArgAction, Command};
use image::{imageops, io::Reader as ImageReader, DynamicImage, Rgba, Rgba32FImage};
use std::ffi::OsString;

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

/// Parses a hex color string.
///
/// Linearized sRGB value is returned if `correct_alpha` is `true`.
fn parse_color(color: &str, correct_alpha: bool) -> Rgba<f32> {
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
    if correct_alpha {
        Rgba([srgb_to_linear(r), srgb_to_linear(g), srgb_to_linear(b), a])
    } else {
        Rgba([r, g, b, a])
    }
}

fn main() {
    let mut matches = Command::new("compose")
        .about("Colorizes and alpha-composes gray-alpha images.")
        .arg(
            Arg::new("add")
                .long("add")
                .number_of_values(2)
                .value_names(["file", "color"])
                .action(ArgAction::Append),
        )
        .arg(
            Arg::new("background")
                .long("background")
                .default_value("#ffffffff"),
        )
        .arg(
            Arg::new("correct-alpha")
                .long("correct-alpha")
                .action(ArgAction::SetTrue),
        )
        .arg(
            Arg::new("output")
                .long("output")
                .value_name("file")
                .value_parser(value_parser!(OsString)),
        )
        .get_matches();

    let correct_alpha = matches.get_flag("correct-alpha");
    let background = parse_color(
        &matches.remove_one::<String>("background").unwrap(),
        correct_alpha,
    );
    let entries = matches
        .remove_many::<String>("add")
        .unwrap_or_default()
        .collect::<Vec<_>>()
        .chunks_exact(2)
        .map(|e| Entry {
            color: parse_color(&e[1], correct_alpha),
            file: e[0].clone(),
        })
        .collect::<Vec<_>>();
    let output = matches.remove_one::<OsString>("output").unwrap();

    let mut composed = None;
    for entry in entries {
        let mut im = ImageReader::open(&entry.file)
            .unwrap_or_else(|_| panic!("failed to open the image '{}'", entry.file))
            .decode()
            .unwrap_or_else(|_| panic!("failed to decode the image '{}'", entry.file))
            .into_rgba32f();
        let composed = composed.get_or_insert_with(|| {
            let mut composed = Rgba32FImage::new(im.width(), im.height());
            composed.fill(1.0);
            colorize(&mut composed, background);
            composed
        });

        assert_eq!(im.width(), composed.width());
        assert_eq!(im.height(), composed.height());

        colorize(&mut im, entry.color);
        imageops::overlay(composed, &im, 0, 0);
    }

    if let Some(mut composed) = composed {
        if correct_alpha {
            linear_to_srgb(&mut composed);
        }
        DynamicImage::ImageRgba32F(composed)
            .to_rgba8()
            .save(output)
            .expect("failed to save the image");
    }
}
