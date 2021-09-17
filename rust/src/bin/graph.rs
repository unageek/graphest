use clap::{App, Arg, ArgSettings};
use graphest::{
    Box2D, Constant, Explicit, Graph, GraphingStatistics, Image, Implicit, Parametric, PixelIndex,
    PixelRange, Relation, RelationType, Ternary,
};
use image::{GrayAlphaImage, LumaA, Rgb, RgbImage};
use inari::{const_interval, interval, Interval};
use itertools::Itertools;
use std::{ffi::OsString, time::Duration};

fn print_statistics_header() {
    println!(
        "  {:>14}  {:>24}  {:>28}",
        "Eval. Time (s)", "Area Proven (%)", "Number of Evaluations"
    );
    println!("  {:->14}  {:->24}  {:->28}", "", "", "");
}

fn print_statistics(cur: &GraphingStatistics, prev: &GraphingStatistics) {
    fn make_interval(x: f64) -> Interval {
        interval!(x, x).unwrap()
    }

    let i100 = const_interval!(100.0, 100.0);
    let ipx = make_interval(cur.pixels as f64);
    let area = i100 * make_interval(cur.pixels_proven as f64) / ipx;
    let delta_area = i100 * make_interval((cur.pixels_proven - prev.pixels_proven) as f64) / ipx;

    println!(
        "  {:>14.3}  {:>11}  (+ {:>7})  {:>13}  (+ {:>9})",
        cur.time_elapsed.as_secs_f64(),
        // Extract the lower bound and remove the minus sign in "-0.000".
        format!("{:7.3}", area)[1..8].replace('-', " "),
        format!("{:7.3}", delta_area)[1..8].replace('-', " "),
        cur.eval_count,
        (cur.eval_count - prev.eval_count)
    );
}

fn to_interval(s: &str) -> Interval {
    let ss = format!("[{}]", s);
    interval!(&ss).unwrap_or_else(|_| panic!("{} is not a valid number", s))
}

fn main() {
    let matches = App::new("graph")
        .about("Plots the graph of a mathematical relation to an image.")
        .arg(Arg::new("relation").index(1).about("Relation to plot."))
        .arg(
            Arg::new("bounds")
                .short('b')
                .long("bounds")
                .number_of_values(4)
                .allow_hyphen_values(true)
                .default_values(&["-10", "10", "-10", "10"])
                .value_names(&["xmin", "xmax", "ymin", "ymax"])
                .about("Bounds of the region to plot over."),
        )
        .arg(
            Arg::new("dilate")
                .long("dilate")
                .setting(ArgSettings::Hidden)
                .default_value("0"),
        )
        .arg(
            Arg::new("gray-alpha")
                .long("gray-alpha")
                .setting(ArgSettings::Hidden),
        )
        .arg(
            Arg::new("mem-limit")
                .long("mem-limit")
                .default_value("1024")
                .about("Approximate maximum amount of memory in MiB that the program can use."),
        )
        .arg(
            Arg::new("output")
                .short('o')
                .long("output")
                .default_value("graph.png")
                .forbid_empty_values(true)
                .value_name("file")
                .about("Path to the output image. It must end with '.png'."),
        )
        .arg(
            Arg::new("parse")
                .long("parse")
                .about("Only parse the relation and exit with 0 iff it is valid."),
        )
        .arg(
            Arg::new("size")
                .short('s')
                .long("size")
                .number_of_values(2)
                .default_values(&["1024", "1024"])
                .value_names(&["width", "height"])
                .about("Pixel dimensions of the output image."),
        )
        .arg(
            Arg::new("timeout")
                .long("timeout")
                .takes_value(true)
                .forbid_empty_values(true)
                .about("Maximum limit of evaluation time in milliseconds."),
        )
        .get_matches();

    let rel = matches.value_of_t_or_exit::<Relation>("relation");
    if matches.is_present("parse") {
        return;
    }

    let bounds = matches
        .values_of_lossy("bounds")
        .unwrap()
        .iter()
        .map(|s| to_interval(s))
        .collect::<Vec<_>>();
    let dilation = matches.value_of_t_or_exit::<u32>("dilate");
    let gray_alpha = matches.is_present("gray-alpha");
    let mem_limit = 1024 * 1024 * matches.value_of_t_or_exit::<usize>("mem-limit");
    let output = matches.value_of_os("output").unwrap().to_owned();
    let size = matches.values_of_t_or_exit::<u32>("size");
    let timeout = match matches.value_of_t::<u64>("timeout") {
        Ok(t) => Some(Duration::from_millis(t)),
        Err(e) if e.kind == clap::ErrorKind::ArgumentNotFound => None,
        Err(e) => e.exit(),
    };

    let dilation_kernel = {
        let diam = 2 * dilation + 1;
        let rad = dilation + 1;
        let mut ker = Image::<bool>::new(diam, diam);
        for ((y, x), p) in (0..diam).cartesian_product(0..diam).zip(ker.pixels_mut()) {
            *p = y < rad && x < rad;
        }
        ker
    };

    let padded_size = [size[0] + 2 * dilation, size[1] + 2 * dilation];

    let opts = PlotOptions {
        dilation_kernel,
        gray_alpha,
        output,
        padded_size,
        size: [size[0], size[1]],
        timeout,
    };
    let region = Box2D::new(bounds[0], bounds[1], bounds[2], bounds[3]);

    match rel.relation_type() {
        RelationType::Constant => plot(Constant::new(rel, padded_size[0], padded_size[1]), opts),
        RelationType::ExplicitFunctionOfX(_) | RelationType::ExplicitFunctionOfY(_) => plot(
            Explicit::new(
                rel,
                region,
                padded_size[0],
                padded_size[1],
                dilation,
                mem_limit,
            ),
            opts,
        ),
        RelationType::Parametric => plot(
            Parametric::new(
                rel,
                region,
                padded_size[0],
                padded_size[1],
                dilation,
                mem_limit,
            ),
            opts,
        ),
        _ => plot(
            Implicit::new(
                rel,
                region,
                padded_size[0],
                padded_size[1],
                dilation,
                mem_limit,
            ),
            opts,
        ),
    };
}

struct PlotOptions {
    dilation_kernel: Image<bool>,
    gray_alpha: bool,
    output: OsString,
    padded_size: [u32; 2],
    size: [u32; 2],
    timeout: Option<Duration>,
}

fn plot<G: Graph>(mut g: G, opts: PlotOptions) {
    let mut gray_alpha_im: Option<GrayAlphaImage> = None;
    let mut rgb_im: Option<RgbImage> = None;
    let mut raw_im = Image::<Ternary>::new(opts.padded_size[0], opts.padded_size[1]);
    if opts.gray_alpha {
        gray_alpha_im = Some(GrayAlphaImage::new(opts.size[0], opts.size[1]));
    } else {
        rgb_im = Some(RgbImage::new(opts.size[0], opts.size[1]));
    }

    let mut prev_stat = g.get_statistics();
    print_statistics_header();
    print_statistics(&prev_stat, &prev_stat);

    loop {
        let duration = match opts.timeout {
            Some(t) => t
                .saturating_sub(prev_stat.time_elapsed)
                .min(Duration::from_millis(1500)),
            _ => Duration::from_millis(1500),
        };
        if duration.is_zero() {
            eprintln!("Warning: reached the timeout");
            break;
        }
        let result = g.refine(duration);

        let stat = g.get_statistics();
        print_statistics(&stat, &prev_stat);
        prev_stat = stat;

        g.get_image(&mut raw_im);
        for (dy, dx) in (0..opts.size[1]).cartesian_product(0..opts.size[0]) {
            raw_im[PixelIndex::new(dx, dy)] = (0..opts.dilation_kernel.height())
                .cartesian_product(0..opts.dilation_kernel.width())
                .filter(|&(ky, kx)| opts.dilation_kernel[PixelIndex::new(kx, ky)])
                .map(|(ky, kx)| raw_im[PixelIndex::new(dx + kx, dy + ky)])
                .max()
                .unwrap_or(Ternary::False);
        }

        let src_pixels = PixelRange::new(
            PixelIndex::new(0, 0),
            PixelIndex::new(opts.size[0], opts.size[1]),
        );
        if let Some(im) = &mut gray_alpha_im {
            for (src, dst) in src_pixels
                .into_iter()
                .map(|p| raw_im[p])
                .zip(im.pixels_mut())
            {
                *dst = match src {
                    Ternary::True => LumaA([0, 255]),
                    Ternary::Uncertain => LumaA([0, 128]),
                    Ternary::False => LumaA([0, 0]),
                };
            }
            im.save(&opts.output).expect("saving image failed");
        } else if let Some(im) = &mut rgb_im {
            for (src, dst) in src_pixels
                .into_iter()
                .map(|p| raw_im[p])
                .zip(im.pixels_mut())
            {
                *dst = match src {
                    Ternary::True => Rgb([0, 0, 0]),
                    Ternary::Uncertain => Rgb([64, 128, 192]),
                    Ternary::False => Rgb([255, 255, 255]),
                };
            }
            im.save(&opts.output).expect("saving image failed");
        }

        match result {
            Ok(false) => continue,
            Ok(true) => break,
            Err(e) => {
                eprintln!("Warning: {}", e);
                break;
            }
        }
    }
}
