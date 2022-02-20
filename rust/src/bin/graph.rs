use clap::{App, Arg};
use graphest::{
    Box2D, Constant, Explicit, Graph, GraphingStatistics, Image, Implicit, Padding, Parametric,
    PixelIndex, PixelRange, Relation, RelationType, Ternary,
};
use image::{imageops, GrayAlphaImage, LumaA, Rgb, RgbImage};
use inari::{const_interval, interval, Interval};
use itertools::Itertools;
use std::{ffi::OsString, io::stdin, time::Duration};

// Returns the same matrix as Mathematica's `DiskMatrix[r]`.
fn disk_matrix(radius: f64) -> Image<bool> {
    // size = 2 ⌊r⌉ + 1 = 2 ⌊r + 1/2⌋ + 1.
    let size = 2 * (radius + 0.5).floor() as u32 + 1;
    let mut im = Image::new(size, size);
    let mid = size / 2;
    let max_d_sq = (radius + 0.5) * (radius + 0.5);
    for i in 0..size {
        let y = i as i32 - mid as i32;
        for j in 0..size {
            let x = j as i32 - mid as i32;
            let d_sq = (x * x + y * y) as f64;
            im[PixelIndex::new(j, i)] = d_sq <= max_d_sq;
        }
    }
    im
}

fn parse_binary_matrix(mat: &str) -> Image<bool> {
    let mat = mat
        .split(';')
        .map(|row| {
            row.split(',')
                .map(|c| match c {
                    "0" => false,
                    "1" => true,
                    _ => panic!("elements of dilation kernel must be either 0 or 1"),
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    assert!(
        !mat.is_empty() && !mat[0].is_empty(),
        "the matrix must not be empty"
    );
    assert!(
        mat[1..].iter().all(|r| r.len() == mat[0].len()),
        "the matrix rows must have the same length"
    );
    let height = u32::try_from(mat.len()).expect("the matrix has too many rows");
    let width = u32::try_from(mat[0].len()).expect("the matrix has too many columns");

    let mut im = Image::<bool>::new(width, height);
    for (p, el) in im.pixels_mut().zip(mat.into_iter().flatten()) {
        *p = el;
    }
    im
}

fn print_statistics_header() {
    println!(
        "  {:>14}  {:>24}  {:>28}",
        "Eval. Time (s)", "Area Proven (%)", "Number of Evaluations"
    );
    println!("  {:->14}  {:->24}  {:->28}", "", "", "");
}

fn print_statistics(cur: &GraphingStatistics, prev: &GraphingStatistics) {
    fn point_interval(x: f64) -> Interval {
        interval!(x, x).unwrap()
    }

    fn to_percent(x: Interval) -> Interval {
        const_interval!(100.0, 100.0) * x
    }

    let pixels = point_interval(cur.pixels as f64);
    let area = to_percent(point_interval(cur.pixels_complete as f64) / pixels);
    let delta_area =
        to_percent(point_interval((cur.pixels_complete - prev.pixels_complete) as f64) / pixels);

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
        .arg(Arg::new("relation").index(1).help("Relation to plot."))
        .arg(
            Arg::new("bounds")
                .short('b')
                .long("bounds")
                .number_of_values(4)
                .allow_hyphen_values(true)
                .default_values(&["-10", "10", "-10", "10"])
                .forbid_empty_values(true)
                .value_names(&["xmin", "xmax", "ymin", "ymax"])
                .help("Bounds of the region over which the relation is plotted."),
        )
        .arg(
            Arg::new("dilate")
                .long("dilate")
                .hide(true)
                .default_value("1")
                .forbid_empty_values(true),
        )
        .arg(Arg::new("dump-ast").long("dump-ast").hide(true))
        .arg(Arg::new("gray-alpha").long("gray-alpha").hide(true))
        .arg(
            Arg::new("mem-limit")
                .long("mem-limit")
                .default_value("1024")
                .forbid_empty_values(true)
                .value_name("mbytes")
                .help("Approximate maximum amount of memory in MiB that the program can use."),
        )
        .arg(
            Arg::new("output")
                .short('o')
                .long("output")
                .default_value("graph.png")
                .forbid_empty_values(true)
                .value_name("file")
                .help("Path to the output image. It must end with '.png'."),
        )
        .arg(
            Arg::new("output-once")
                .long("output-once")
                .help("Do not output intermediate images."),
        )
        .arg(
            Arg::new("pad-bottom")
                .long("pad-bottom")
                .hide(true)
                .default_value("0")
                .forbid_empty_values(true),
        )
        .arg(
            Arg::new("pad-left")
                .long("pad-left")
                .hide(true)
                .default_value("0")
                .forbid_empty_values(true),
        )
        .arg(
            Arg::new("pad-right")
                .long("pad-right")
                .hide(true)
                .default_value("0")
                .forbid_empty_values(true),
        )
        .arg(
            Arg::new("pad-top")
                .long("pad-top")
                .hide(true)
                .default_value("0")
                .forbid_empty_values(true),
        )
        .arg(
            Arg::new("parse")
                .long("parse")
                .help("Only validate the relation and exit without plotting."),
        )
        .arg(
            Arg::new("pause-per-iteration")
                .long("pause-per-iteration")
                .hide(true),
        )
        .arg(
            Arg::new("size")
                .short('s')
                .long("size")
                .number_of_values(2)
                .default_values(&["1024", "1024"])
                .forbid_empty_values(true)
                .value_names(&["width", "height"])
                .help("Dimensions of the output image in pixels."),
        )
        .arg(
            Arg::new("ssaa")
                .long("ssaa")
                .default_value("1")
                .forbid_empty_values(true)
                .value_name("scale")
                .next_line_help(true)
                .help(
                    "Anti-alias the graph by supersampling pixels by the given scale.\n\
                     Odd numbers ranging from 1 (no anti-aliasing) to 17 are accepted.",
                ),
        )
        .arg(
            Arg::new("timeout")
                .long("timeout")
                .takes_value(true)
                .forbid_empty_values(true)
                .value_name("msecs")
                .help("Maximum limit of evaluation time in milliseconds."),
        )
        .get_matches();

    let rel = matches.value_of_t_or_exit::<Relation>("relation");
    if matches.is_present("dump-ast") {
        println!("{}", rel.ast().dump_full());
    }
    if matches.is_present("parse") {
        return;
    }

    let bounds = matches
        .values_of("bounds")
        .unwrap()
        .map(to_interval)
        .collect::<Vec<_>>();
    let dilation = matches.value_of("dilate").unwrap();
    let gray_alpha = matches.is_present("gray-alpha");
    let mem_limit = 1024 * 1024 * matches.value_of_t_or_exit::<usize>("mem-limit");
    let output = matches.value_of_os("output").unwrap().to_owned();
    let output_once = matches.is_present("output-once");
    let output_padding = {
        Padding {
            bottom: matches.value_of_t_or_exit::<u32>("pad-bottom"),
            left: matches.value_of_t_or_exit::<u32>("pad-left"),
            right: matches.value_of_t_or_exit::<u32>("pad-right"),
            top: matches.value_of_t_or_exit::<u32>("pad-top"),
        }
    };
    let output_size = {
        let s = matches.values_of_t_or_exit::<u32>("size");
        [
            s[0] + output_padding.left + output_padding.right,
            s[1] + output_padding.bottom + output_padding.top,
        ]
    };
    let pause_per_iteration = matches.is_present("pause-per-iteration");
    let ssaa = matches.value_of_t_or_exit::<u32>("ssaa");
    let timeout = match matches.value_of_t::<u64>("timeout") {
        Ok(t) => Some(Duration::from_millis(t)),
        Err(e) if e.kind == clap::ErrorKind::ArgumentNotFound => None,
        Err(e) => e.exit(),
    };

    let dilation_kernel = if ssaa > 1 {
        if dilation != "1" {
            println!("`--dilate` and `--ssaa` cannot be used together.");
        }
        disk_matrix((ssaa / 2) as f64)
    } else {
        let k = parse_binary_matrix(dilation);
        assert!(
            k.width() == k.height() && k.width() % 2 == 1,
            "dilation kernel must be square and have odd dimensions"
        );
        k
    };

    let graph_padding = Padding {
        bottom: ssaa * output_padding.bottom + dilation_kernel.height() / 2,
        left: ssaa * output_padding.left + dilation_kernel.width() / 2,
        right: ssaa * output_padding.right + dilation_kernel.width() / 2,
        top: ssaa * output_padding.top + dilation_kernel.height() / 2,
    };
    let graph_size = [
        ssaa * output_size[0] + (dilation_kernel.width() - 1),
        ssaa * output_size[1] + (dilation_kernel.height() - 1),
    ];

    let opts = PlotOptions {
        dilation_kernel,
        graph_size,
        gray_alpha,
        output,
        output_once,
        output_size,
        pause_per_iteration,
        timeout,
    };
    let region = Box2D::new(bounds[0], bounds[1], bounds[2], bounds[3]);

    match rel.relation_type() {
        RelationType::Constant => plot(Constant::new(rel, graph_size[0], graph_size[1]), opts),
        RelationType::ExplicitFunctionOfX(_) | RelationType::ExplicitFunctionOfY(_) => plot(
            Explicit::new(
                rel,
                region,
                graph_size[0],
                graph_size[1],
                graph_padding,
                mem_limit,
            ),
            opts,
        ),
        RelationType::Parametric => plot(
            Parametric::new(
                rel,
                region,
                graph_size[0],
                graph_size[1],
                graph_padding,
                mem_limit,
            ),
            opts,
        ),
        _ => plot(
            Implicit::new(
                rel,
                region,
                graph_size[0],
                graph_size[1],
                graph_padding,
                mem_limit,
            ),
            opts,
        ),
    };
}

struct PlotOptions {
    dilation_kernel: Image<bool>,
    graph_size: [u32; 2],
    gray_alpha: bool,
    output: OsString,
    output_once: bool,
    output_size: [u32; 2],
    pause_per_iteration: bool,
    timeout: Option<Duration>,
}

fn plot<G: Graph>(mut graph: G, opts: PlotOptions) {
    let mut gray_alpha_im: Option<GrayAlphaImage> = None;
    let mut rgb_im: Option<RgbImage> = None;
    let mut raw_im = Image::<Ternary>::new(opts.graph_size[0], opts.graph_size[1]);
    if opts.gray_alpha {
        gray_alpha_im = Some(GrayAlphaImage::new(opts.graph_size[0], opts.graph_size[1]));
    } else {
        rgb_im = Some(RgbImage::new(opts.graph_size[0], opts.graph_size[1]));
    }

    let mut prev_stat = graph.get_statistics();
    print_statistics_header();
    print_statistics(&prev_stat, &prev_stat);

    let mut save_image = |graph: &G| {
        graph.get_image(&mut raw_im);

        // Dilation collects the values of pixels to top-left pixels.
        for (dy, dx) in (0..opts.graph_size[1] - (opts.dilation_kernel.height() - 1))
            .cartesian_product(0..opts.graph_size[0] - (opts.dilation_kernel.width() - 1))
        {
            raw_im[PixelIndex::new(dx, dy)] = (0..opts.dilation_kernel.height())
                .cartesian_product(0..opts.dilation_kernel.width())
                .filter(|&(ky, kx)| opts.dilation_kernel[PixelIndex::new(kx, ky)])
                .map(|(ky, kx)| raw_im[PixelIndex::new(dx + kx, dy + ky)])
                .max()
                .unwrap_or(Ternary::False);
        }

        let src_pixels = PixelRange::new(
            PixelIndex::new(0, 0),
            PixelIndex::new(opts.graph_size[0], opts.graph_size[1]),
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
            let im = imageops::crop(
                im,
                0,
                0,
                opts.graph_size[0] - (opts.dilation_kernel.width() - 1),
                opts.graph_size[1] - (opts.dilation_kernel.height() - 1),
            )
            .to_image();
            let im = imageops::resize(
                &im,
                opts.output_size[0],
                opts.output_size[1],
                imageops::FilterType::Triangle,
            );
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
            let im = imageops::crop(
                im,
                0,
                0,
                opts.graph_size[0] - (opts.dilation_kernel.width() - 1),
                opts.graph_size[1] - (opts.dilation_kernel.height() - 1),
            )
            .to_image();
            let im = imageops::resize(
                &im,
                opts.output_size[0],
                opts.output_size[1],
                imageops::FilterType::Triangle,
            );
            im.save(&opts.output).expect("saving image failed");
        }
    };

    loop {
        if opts.pause_per_iteration {
            // Await for a newline character.
            let mut input = String::new();
            stdin().read_line(&mut input).unwrap();
        }

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
        let result = graph.refine(duration);

        let stat = graph.get_statistics();
        print_statistics(&stat, &prev_stat);
        prev_stat = stat;

        if !opts.output_once {
            save_image(&graph);
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

    if opts.output_once {
        save_image(&graph);
    }
}

#[cfg(test)]
mod tests {
    use crate::*;

    #[test]
    fn test_disk_matrix() {
        // CopyToClipboard@StringRiffle[Map[ToString, DiskMatrix[r], {2}], ";", ","]
        assert_eq!(disk_matrix(0.0), parse_binary_matrix("1"));
        assert_eq!(disk_matrix(0.4), parse_binary_matrix("1"));
        assert_eq!(disk_matrix(0.5), parse_binary_matrix("0,1,0;1,1,1;0,1,0"));
        assert_eq!(disk_matrix(1.0), parse_binary_matrix("1,1,1;1,1,1;1,1,1"));
        assert_eq!(disk_matrix(1.4), parse_binary_matrix("1,1,1;1,1,1;1,1,1"));
        assert_eq!(
            disk_matrix(1.5),
            parse_binary_matrix(
                "0,0,1,0,0;\
                 0,1,1,1,0;\
                 1,1,1,1,1;\
                 0,1,1,1,0;\
                 0,0,1,0,0"
            )
        );
        assert_eq!(
            disk_matrix(2.0),
            parse_binary_matrix(
                "0,1,1,1,0;\
                 1,1,1,1,1;\
                 1,1,1,1,1;\
                 1,1,1,1,1;\
                 0,1,1,1,0"
            )
        );
        assert_eq!(
            disk_matrix(2.4),
            parse_binary_matrix(
                "1,1,1,1,1;\
                 1,1,1,1,1;\
                 1,1,1,1,1;\
                 1,1,1,1,1;\
                 1,1,1,1,1"
            )
        );
        assert_eq!(
            disk_matrix(2.5),
            parse_binary_matrix(
                "0,0,0,1,0,0,0;\
                 0,1,1,1,1,1,0;\
                 0,1,1,1,1,1,0;\
                 1,1,1,1,1,1,1;\
                 0,1,1,1,1,1,0;\
                 0,1,1,1,1,1,0;\
                 0,0,0,1,0,0,0"
            )
        );
        assert_eq!(
            disk_matrix(3.0),
            parse_binary_matrix(
                "0,0,1,1,1,0,0;\
                 0,1,1,1,1,1,0;\
                 1,1,1,1,1,1,1;\
                 1,1,1,1,1,1,1;\
                 1,1,1,1,1,1,1;\
                 0,1,1,1,1,1,0;\
                 0,0,1,1,1,0,0"
            )
        );
    }
}
