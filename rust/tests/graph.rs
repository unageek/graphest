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
    let ref_dir = PathBuf::from("./tests/reference");
    let actual_dir = PathBuf::from("./tests/actual");
    let ref_img = ref_dir.join([id, ".png"].concat());
    let actual_img = actual_dir.join([id, ".png"].concat());
    if ref_img.exists() {
        let _ = create_dir(actual_dir);
        assert!(execute(
            Command::new(graph)
                .args(args)
                .arg("-o")
                .arg(actual_img.clone()),
        ));
        let ref_bytes = read(ref_img).unwrap();
        let actual_bytes = read(actual_img).unwrap();
        // Use `assert!` instead of `assert_eq!` to avoid the `Vec`s to be printed.
        assert!(ref_bytes == actual_bytes);
    } else {
        assert!(execute(
            Command::new(graph).args(args).arg("-o").arg(ref_img),
        ));
    }
}

#[macro_export]
macro_rules! t {
    ($id:ident, $($arg:expr),+) => {
        #[cfg(all(not(debug_assertions), feature = "arb"))]
        #[test]
        fn $id() {
            let id = stringify!($id);
            let args = vec![$($arg.into()),+];
            crate::test(id, &args);
        }
    };
}

mod graph {
    mod explicit;
}
