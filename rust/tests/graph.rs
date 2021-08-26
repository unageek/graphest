#![cfg(all(not(debug_assertions), feature = "arb"))]

use std::{
    fs::{create_dir, read},
    path::PathBuf,
    process::{Command, Stdio},
};

fn execute(cmd: &mut Command) -> bool {
    cmd.stdout(Stdio::null())
        .status()
        .unwrap_or_else(|_| panic!("failed to run the command: {:?}", cmd))
        .success()
}

pub fn test(id: &str, args: &[String]) {
    let graph = "./target/release/graph";
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
    ($id:ident, $($arg:expr),+) => {
        #[test]
        fn $id() {
            let id = stringify!($id);
            let args = vec![$($arg.into()),+];
            $crate::test(id, &args);
        }
    };
}

mod graph_tests {
    mod explicit;
    mod parametric;
}
