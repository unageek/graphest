mod ast;
mod dyn_relation;
mod graph;
mod interval_set;
mod parse;
mod rel;
mod visit;

use crate::{dyn_relation::*, graph::*};
use clap::{App, AppSettings, Arg};
use inari::{const_interval, interval, Interval};

fn print_statistics_header() {
    println!(
        "  {:^14}  {:^30}  {:^30}",
        "Eval. Time (s)", "Area Proven (%)", "# of Evaluations"
    );
    println!("  {:-^14}  {:-^30}  {:-^30}", "", "", "");
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
        "  {:>14.3}  {:>14}  {:>14}  {:>14}  {:>14}",
        cur.time_elapsed.as_secs_f64(),
        &format!("{:7.3}", area)[1..8],
        format!("(+{:>11})", &format!("{:7.3}", delta_area)[1..8]),
        cur.evaluations,
        format!("(+{:11})", cur.evaluations - prev.evaluations),
    );
}

fn main() {
    let matches = App::new("inari-graph")
        .setting(AppSettings::AllowLeadingHyphen)
        .about("Plots the graph of a relation over the x-y plane.")
        .arg(
            Arg::with_name("output")
                .index(1)
                .about("The path to save the output image (only .png is supported)."),
        )
        .arg(
            Arg::with_name("relation")
                .index(2)
                .about("The relation to plot."),
        )
        .arg(
            Arg::with_name("bounds")
                .short('b')
                .number_of_values(4)
                .default_values(&["-10", "10", "-10", "10"])
                .value_names(&["xmin", "xmax", "ymin", "ymax"])
                .about("Sets the bounds of the plot region."),
        )
        .arg(
            Arg::with_name("size")
                .short('s')
                .number_of_values(2)
                .default_values(&["1024", "1024"])
                .value_names(&["width", "height"])
                .about("Sets the dimensions of the output image."),
        )
        .get_matches();

    let output = matches.value_of("output");
    let rel = matches.value_of_t_or_exit::<DynRelation>("relation");
    let bounds = matches.values_of_t_or_exit::<f64>("bounds");
    let size = matches.values_of_t_or_exit::<u32>("size");

    let mut g = Graph::new(
        rel,
        Region::new(bounds[0], bounds[1], bounds[2], bounds[3]),
        size[0],
        size[1],
    );
    let mut prev_stat = g.get_statistics();

    print_statistics_header();

    loop {
        let result = g.step();

        let stat = g.get_statistics();
        print_statistics(&stat, &prev_stat);
        prev_stat = stat;

        if let Some(output) = output {
            let im = g.get_image();
            im.save(output).unwrap();
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
