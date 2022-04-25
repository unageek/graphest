#![cfg(all(not(debug_assertions), feature = "arb"))]

use std::{
    fs::{create_dir, read},
    path::PathBuf,
    process::{Command, Stdio},
};
use uuid::Uuid;

fn execute(cmd: &mut Command) -> bool {
    cmd.stdout(Stdio::null())
        .status()
        .unwrap_or_else(|_| panic!("failed to execute the command: {:?}", cmd))
        .success()
}

fn is_valid_id(id: &str) -> bool {
    &id[0..2] == "t_"
        && Uuid::parse_str(&id[2..]).map_or(false, |u| {
            u.get_variant() == uuid::Variant::RFC4122
                && u.get_version() == Some(uuid::Version::Random)
        })
}

pub fn test(id: &str, args: &[String]) {
    assert!(is_valid_id(id));

    let graph = PathBuf::from("../target/release/graph");
    let ref_dir = PathBuf::from("./tests/graph_tests/reference");
    let actual_dir = PathBuf::from("./tests/graph_tests/actual");
    let ref_img = ref_dir.join([id, ".png"].concat());
    let actual_img = actual_dir.join([id, ".png"].concat());

    if ref_img.exists() {
        let _ = create_dir(actual_dir);

        let mut cmd = Command::new(graph);
        cmd.args(args).arg("--output").arg(actual_img.clone());
        if !args.iter().any(|a| a == "--timeout") {
            cmd.args(["--timeout", "1000"]);
        }
        assert!(execute(&mut cmd));

        let ref_bytes = read(ref_img).unwrap();
        let actual_bytes = read(actual_img).unwrap();
        // Use `assert!` instead of `assert_eq!` to avoid the `Vec`s to be printed.
        assert!(ref_bytes == actual_bytes);
    } else {
        let mut cmd = Command::new(graph);
        cmd.args(args).arg("--output").arg(ref_img);
        if !args.iter().any(|a| a == "--timeout") {
            cmd.args(["--timeout", "1000"]);
        }
        assert!(execute(&mut cmd));
    }
}

#[macro_export]
macro_rules! t {
    ($id:ident, $($arg:expr),+, @bounds($xmin:expr, $xmax:expr, $ymin:expr, $ymax:expr) $(, @$opt:ident($($opt_arg:expr),+))* $(,)?) => {
        t!($id, $($arg),+, "--bounds", stringify!($xmin), stringify!($xmax), stringify!($ymin), stringify!($ymax) $(, @$opt($($opt_arg),+))*);
    };

    ($id:ident, $($arg:expr),+, @dilate($dilate:expr) $(, @$opt:ident($($opt_arg:expr),+))* $(,)?) => {
        t!($id, $($arg),+, "--dilate", $dilate $(, @$opt($($opt_arg),+))*);
    };

    ($id:ident, $($arg:expr),+, @pad_bottom($length:expr) $(, @$opt:ident($($opt_arg:expr),+))* $(,)?) => {
        t!($id, $($arg),+, "--pad-bottom", stringify!($length) $(, @$opt($($opt_arg),+))*);
    };

    ($id:ident, $($arg:expr),+, @pad_left($length:expr) $(, @$opt:ident($($opt_arg:expr),+))* $(,)?) => {
        t!($id, $($arg),+, "--pad-left", stringify!($length) $(, @$opt($($opt_arg),+))*);
    };

    ($id:ident, $($arg:expr),+, @pad_right($length:expr) $(, @$opt:ident($($opt_arg:expr),+))* $(,)?) => {
        t!($id, $($arg),+, "--pad-right", stringify!($length) $(, @$opt($($opt_arg),+))*);
    };

    ($id:ident, $($arg:expr),+, @pad_top($length:expr) $(, @$opt:ident($($opt_arg:expr),+))* $(,)?) => {
        t!($id, $($arg),+, "--pad-top", stringify!($length) $(, @$opt($($opt_arg),+))*);
    };

    ($id:ident, $($arg:expr),+, @pen_size($pen_size:expr) $(, @$opt:ident($($opt_arg:expr),+))* $(,)?) => {
        t!($id, $($arg),+, "--pen-size", stringify!($pen_size) $(, @$opt($($opt_arg),+))*);
    };

    ($id:ident, $($arg:expr),+, @size($width:expr, $height:expr) $(, @$opt:ident($($opt_arg:expr),+))* $(,)?) => {
        t!($id, $($arg),+, "--size", stringify!($width), stringify!($height) $(, @$opt($($opt_arg),+))*);
    };

    ($id:ident, $($arg:expr),+, @ssaa($ssaa:expr) $(, @$opt:ident($($opt_arg:expr),+))* $(,)?) => {
        t!($id, $($arg),+, "--ssaa", stringify!($ssaa) $(, @$opt($($opt_arg),+))*);
    };

    ($id:ident, $($arg:expr),+, @timeout($timeout:expr) $(, @$opt:ident($($opt_arg:expr),+))* $(,)?) => {
        t!($id, $($arg),+, "--timeout", stringify!($timeout) $(, @$opt($($opt_arg),+))*);
    };

    ($id:ident, $($arg:expr),+ $(,)?) => {
        #[test]
        fn $id() {
            let id = stringify!($id);
            let args = vec![$($arg.into()),+];
            $crate::test(id, &args);
        }
    };
}

mod graph_tests {
    mod constant;
    mod dilate;
    mod examples;
    mod explicit;
    mod functions;
    mod implicit;
    mod pad;
    mod parametric;
    mod pen_size;
    mod polar;
    mod ssaa;
}
