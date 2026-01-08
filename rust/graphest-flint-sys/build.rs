use std::{
    env, fs, io,
    path::{Path, PathBuf},
    process::{Command, Output},
};

// https://gitlab.com/tspiteri/gmp-mpfr-sys/-/blob/master/build.rs

const FLINT_GIT_TAG: &str = "v3.4.0";
const FLINT_GIT_URL: &str = "https://github.com/flintlib/flint.git";

struct Environment {
    build_dir: PathBuf,
    cache_dir: Option<PathBuf>,
    gmp_dir: PathBuf,
    include_dir: PathBuf,
    lib_dir: PathBuf,
    makeflags: String,
    out_dir: PathBuf,
}

fn main() {
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let pkg_name = env::var("CARGO_PKG_NAME").unwrap();
    let pkg_version = env::var("CARGO_PKG_VERSION").unwrap();
    let env = Environment {
        build_dir: out_dir.join("build"),
        cache_dir: user_cache_dir().map(|c| c.join(pkg_name).join(pkg_version)),
        gmp_dir: PathBuf::from(env::var_os("DEP_GMP_OUT_DIR").unwrap()),
        include_dir: out_dir.join("include"),
        lib_dir: out_dir.join("lib"),
        makeflags: "-j".to_owned(),
        out_dir: out_dir.clone(),
    };
    fs::create_dir_all(&env.build_dir)
        .unwrap_or_else(|_| panic!("failed to create the directory: {:?}", env.build_dir));

    load_cache(&env);
    build_flint(&env);
    save_cache(&env);
    run_arb_bindgen(&env);
    write_link_info(&env);
}

fn load_cache(env: &Environment) {
    if let Some(c) = &env.cache_dir {
        let _ = copy_dir_all(c.join("include"), env.include_dir.clone());
        let _ = copy_dir_all(c.join("lib"), env.lib_dir.clone());
    }
}

fn build_flint(env: &Environment) {
    if env.lib_dir.join("libflint.a").exists() {
        return;
    }

    let build_dir = env.build_dir.join("flint-build");
    if !build_dir.exists() {
        execute_or_panic(Command::new("git").current_dir(&env.build_dir).args([
            "clone",
            "--branch",
            FLINT_GIT_TAG,
            "--depth",
            "1",
            FLINT_GIT_URL,
            build_dir.to_str().unwrap(),
        ]));
    }

    execute_or_panic(
        Command::new("sh")
            .current_dir(&build_dir)
            .arg("-c")
            .arg("./bootstrap.sh"),
    );
    execute_or_panic(
        Command::new("sh")
            .current_dir(&build_dir)
            .arg("-c")
            .arg(
                [
                    "./configure",
                    "--disable-shared",
                    "--enable-static",
                    &format!("--prefix={}", to_unix_path(&env.out_dir)),
                    &format!("--with-gmp={}", to_unix_path(&env.gmp_dir)),
                    &format!("--with-mpfr={}", to_unix_path(&env.gmp_dir)),
                    "ABI=64",
                ]
                .join(" "),
            )
            .env("CFLAGS", "-Wno-error"),
    );
    execute_or_panic(
        Command::new("make")
            .current_dir(&build_dir)
            .env("MAKEFLAGS", &env.makeflags),
    );
    execute_or_panic(
        Command::new("make")
            .current_dir(&build_dir)
            .arg("tests")
            .env("MAKEFLAGS", &env.makeflags),
    );
    execute_or_panic(
        Command::new("make")
            .current_dir(&build_dir)
            .arg("check")
            .env("FLINT_TEST_MULTIPLIER", "0.1")
            .env("MAKEFLAGS", &env.makeflags),
    );
    execute_or_panic(Command::new("make").current_dir(&build_dir).arg("install"));
}

fn run_arb_bindgen(env: &Environment) {
    let binding_file = env.out_dir.join("arb.rs");
    if binding_file.exists() {
        return;
    }

    let include_dir = env.gmp_dir.join("include").to_str().unwrap().to_owned();
    let mut clang_args = vec![
        "-DACB_INLINES_C",
        "-DARB_INLINES_C",
        "-DARF_INLINES_C",
        "-DMAG_INLINES_C",
        "-I",
        &include_dir,
    ];

    // https://github.com/rust-lang/rust-bindgen/issues/1760
    if cfg!(all(
        target_arch = "x86_64",
        target_os = "windows",
        target_env = "gnu"
    )) {
        clang_args.push("--target=x86_64-pc-windows-gnu");
    }

    bindgen::Builder::default()
        .header(
            env.include_dir
                .join("flint")
                .join("acb.h")
                .to_str()
                .unwrap(),
        )
        .header(
            env.include_dir
                .join("flint")
                .join("acb_elliptic.h")
                .to_str()
                .unwrap(),
        )
        .header(
            env.include_dir
                .join("flint")
                .join("arb.h")
                .to_str()
                .unwrap(),
        )
        .header(
            env.include_dir
                .join("flint")
                .join("arb_hypgeom.h")
                .to_str()
                .unwrap(),
        )
        .header(
            env.include_dir
                .join("flint")
                .join("arf.h")
                .to_str()
                .unwrap(),
        )
        .header(
            env.include_dir
                .join("flint")
                .join("mag.h")
                .to_str()
                .unwrap(),
        )
        .allowlist_function("(acb|arb|arf|mag)_.*")
        .clang_args(&clang_args)
        .generate()
        .expect("failed to generate bindings")
        .write_to_file(binding_file.clone())
        .unwrap_or_else(|_| {
            panic!(
                "failed to write to the file: {}",
                binding_file.to_string_lossy()
            )
        });
}

fn save_cache(env: &Environment) {
    if let Some(c) = &env.cache_dir {
        let _ = copy_dir_all(env.include_dir.clone(), c.join("include"));
        let _ = copy_dir_all(env.lib_dir.clone(), c.join("lib"));
    }
}

fn write_link_info(env: &Environment) {
    println!(
        "cargo:rustc-link-search=native={}",
        env.gmp_dir.join("lib").to_str().unwrap()
    );
    println!("cargo:rustc-link-lib=static=gmp");
    println!("cargo:rustc-link-lib=static=mpfr");
    println!(
        "cargo:rustc-link-search=native={}",
        env.lib_dir.to_str().unwrap()
    );
    println!("cargo:rustc-link-lib=static=flint");
}

/// Copies all files and directories in `from` into `to`, preserving the directory structure.
///
/// The directory `to` is created if it does not exist. Symlinks are ignored.
fn copy_dir_all<P: AsRef<Path>, Q: AsRef<Path>>(from: P, to: Q) -> io::Result<()> {
    fs::create_dir_all(&to)?;

    for entry in fs::read_dir(from)? {
        let entry = entry?;
        let from = entry.path();
        let to = to.as_ref().join(entry.file_name());
        let ty = entry.file_type()?;
        if ty.is_dir() {
            copy_dir_all(from, to)?;
        } else if ty.is_file() {
            fs::copy(from, to)?;
        }
    }

    Ok(())
}

fn execute_or_panic(cmd: &mut Command) -> Output {
    let output = cmd
        .output()
        .unwrap_or_else(|_| panic!("failed to execute the command: {:?}", cmd));

    if !output.status.success() {
        if let Some(code) = output.status.code() {
            panic!(
                "the process exited with code {}: {:?}\nstdout:\n{}\nstderr:\n{}",
                code,
                cmd,
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
        } else {
            panic!("the process is terminated by a signal: {:?}", cmd);
        }
    }

    output
}

fn to_unix_path(path: &Path) -> String {
    let s = path.to_str().unwrap();
    if cfg!(windows) {
        let output = execute_or_panic(Command::new("cygpath").arg(s));
        String::from_utf8_lossy(&output.stdout).trim().to_owned()
    } else {
        s.to_owned()
    }
}

fn user_cache_dir() -> Option<PathBuf> {
    let host = env::var("HOST").ok()?;

    if host.contains("darwin") {
        env::var_os("HOME")
            .filter(|s| !s.is_empty())
            .map(|s| PathBuf::from(s).join("Library").join("Caches"))
    } else if host.contains("linux") {
        env::var_os("XDG_CACHE_HOME")
            .filter(|s| !s.is_empty())
            .map(PathBuf::from)
            .or_else(|| {
                env::var_os("HOME")
                    .filter(|s| !s.is_empty())
                    .map(|s| PathBuf::from(s).join(".cache"))
            })
    } else if host.contains("windows") {
        env::var_os("LOCALAPPDATA")
            .filter(|s| !s.is_empty())
            .map(PathBuf::from)
    } else {
        None
    }
}
