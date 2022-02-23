use std::{
    env, fs, io,
    path::{Path, PathBuf},
    process::Command,
};

// https://gitlab.com/tspiteri/gmp-mpfr-sys/-/blob/master/build.rs

const ARB_GIT_TAG: &str = "2.22.1";
const ARB_GIT_URL: &str = "https://github.com/fredrik-johansson/arb.git";

const FLINT_GIT_TAG: &str = "v2.8.4";
const FLINT_GIT_URL: &str = "https://github.com/wbhart/flint2.git";

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
    build_arb(&env);
    save_cache(&env);
    run_bindgen(&env);
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

    let gmp_dir = env.build_dir.join("gmp");
    if !gmp_dir.exists() {
        symlink_dir_or_panic(&env.gmp_dir, &gmp_dir);
    }

    let build_dir = env.build_dir.join("flint-build");
    if !build_dir.exists() {
        execute_or_panic(Command::new("git").current_dir(&env.build_dir).args(&[
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
            .arg(
                [
                    "./configure",
                    "--disable-shared",
                    "--prefix=../..",     // `env.out_dir`
                    "--with-gmp=../gmp",  // `gmp_dir`
                    "--with-mpfr=../gmp", // `gmp_dir`
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
            .arg("check")
            .env("MAKEFLAGS", &env.makeflags),
    );
    execute_or_panic(Command::new("make").current_dir(&build_dir).arg("install"));
}

fn build_arb(env: &Environment) {
    if env.lib_dir.join("libarb.a").exists() {
        return;
    }

    let gmp_dir = env.build_dir.join("gmp");
    if !gmp_dir.exists() {
        symlink_dir_or_panic(&env.gmp_dir, &gmp_dir);
    }

    let build_dir = env.build_dir.join("arb-build");
    if !build_dir.exists() {
        execute_or_panic(Command::new("git").current_dir(&env.build_dir).args(&[
            "clone",
            "--branch",
            ARB_GIT_TAG,
            "--depth",
            "1",
            ARB_GIT_URL,
            build_dir.to_str().unwrap(),
        ]));
    }

    execute_or_panic(
        Command::new("sh")
            .current_dir(&build_dir)
            .arg("-c")
            .arg(
                [
                    "./configure",
                    "--disable-shared",
                    "--prefix=../..",     // `env.out_dir`
                    "--with-flint=../..", // `env.out_dir`
                    "--with-gmp=../gmp",  // `gmp_dir`
                    "--with-mpfr=../gmp", // `gmp_dir`
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
            .arg("check")
            .env("ARB_TEST_MULTIPLIER", "0.1")
            .env("MAKEFLAGS", &env.makeflags),
    );
    execute_or_panic(Command::new("make").current_dir(&build_dir).arg("install"));
}

fn run_bindgen(env: &Environment) {
    let binding_file = env.out_dir.join("arb.rs");
    if binding_file.exists() {
        return;
    }

    bindgen::Builder::default()
        .header(env.include_dir.join("acb.h").to_str().unwrap())
        .header(env.include_dir.join("acb_elliptic.h").to_str().unwrap())
        .header(env.include_dir.join("arb.h").to_str().unwrap())
        .header(env.include_dir.join("arb_hypgeom.h").to_str().unwrap())
        .header(env.include_dir.join("arf.h").to_str().unwrap())
        .header(env.include_dir.join("mag.h").to_str().unwrap())
        .allowlist_function("(acb|arb|arf|mag)_.*")
        .clang_args(&[
            "-DACB_INLINES_C",
            "-DARB_INLINES_C",
            "-DARF_INLINES_C",
            "-DMAG_INLINES_C",
            "-I",
            env.gmp_dir.join("include").to_str().unwrap(),
        ])
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
    println!("cargo:rustc-link-lib=static=arb");
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

fn execute_or_panic(cmd: &mut Command) {
    let status = cmd
        .status()
        .unwrap_or_else(|_| panic!("failed to execute the command: {:?}", cmd));

    if !status.success() {
        if let Some(code) = status.code() {
            panic!("the process exited with code {}: {:?}", code, cmd);
        } else {
            panic!("the process is terminated by a signal: {:?}", cmd);
        }
    }
}

#[cfg(unix)]
fn symlink_dir_or_panic(original: &Path, link: &Path) {
    std::os::unix::fs::symlink(original, link).unwrap_or_else(|_| {
        panic!("failed to create a symlink to {:?} at {:?}", original, link);
    });
}

#[cfg(windows)]
fn symlink_dir_or_panic(original: &Path, link: &Path) {
    if std::os::windows::fs::symlink_dir(original, link).is_ok() {
        return;
    }
    eprintln!("failed to create a symlink to {:?} at {:?}, copying instead", original, link);
    execute_or_panic(Command::new("cp").arg("-R").arg(original).arg(link));
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
