#![allow(clippy::float_cmp)]

mod ast;
mod dyn_relation;
mod eval_result;
mod graph;
mod interval_set;
mod interval_set_ops;
mod parse;
mod rel;
mod visit;

use crate::{
    dyn_relation::DynRelation,
    graph::{Graph, GraphingStatistics, Region},
};
use clap::{App, AppSettings, Arg};
use image::RgbImage;
use inari::{const_interval, interval, Interval};
use std::time::Duration;

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

fn main() {
    let matches = App::new("inari-graph")
        .setting(AppSettings::AllowLeadingHyphen)
        .about("Plots the graph of a relation over the x-y plane.")
        .arg(Arg::new("relation").index(1).about("Relation to plot."))
        .arg(
            Arg::new("bounds")
                .short('b')
                .number_of_values(4)
                .default_values(&["-10", "10", "-10", "10"])
                .value_names(&["xmin", "xmax", "ymin", "ymax"])
                .about("Bounds of the plot region."),
        )
        .arg(
            Arg::new("output")
                .short('o')
                .default_value("graph.png")
                .about("Output file, only .png is supported."),
        )
        .arg(
            Arg::new("size")
                .short('s')
                .number_of_values(2)
                .default_values(&["1024", "1024"])
                .value_names(&["width", "height"])
                .about("Dimensions of the plot."),
        )
        .get_matches();

    let rel = matches.value_of_t_or_exit::<DynRelation>("relation");
    let bounds = matches.values_of_t_or_exit::<f64>("bounds");
    let output = matches.value_of("output");
    let size = matches.values_of_t_or_exit::<u32>("size");

    let mut g = Graph::new(
        rel,
        Region::new(bounds[0], bounds[1], bounds[2], bounds[3]),
        size[0],
        size[1],
    );
    let mut im = RgbImage::new(size[0], size[1]);

    let mut prev_stat = g.get_statistics();
    print_statistics_header();
    print_statistics(&prev_stat, &prev_stat);

    loop {
        let result = g.refine(Duration::from_millis(1500));

        let stat = g.get_statistics();
        print_statistics(&stat, &prev_stat);
        prev_stat = stat;

        if let Some(output) = output {
            g.get_image(&mut im);
            im.save(output).expect("saving image failed");
        }

        match result {
            Ok(true) => break,
            Err(e) => {
                println!("Warning: {}", e);
                break;
            }
            _ => (),
        }
    }
}
