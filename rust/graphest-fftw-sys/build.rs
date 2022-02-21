use std::{
    env, fs, io,
    path::{Path, PathBuf},
    process::Command,
};

const FFTW_TAR_URL: &str = "https://www.fftw.org/fftw-3.3.10.tar.gz";

struct Environment {
    build_dir: PathBuf,
    cache_dir: Option<PathBuf>,
    include_dir: PathBuf,
    lib_dir: PathBuf,
    makeflags: String,
    out_dir: PathBuf,
    target_arch: String,
}

fn main() {
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let pkg_name = env::var("CARGO_PKG_NAME").unwrap();
    let pkg_version = env::var("CARGO_PKG_VERSION").unwrap();
    let env = Environment {
        build_dir: out_dir.join("build"),
        cache_dir: user_cache_dir().map(|c| c.join(pkg_name).join(pkg_version)),
        include_dir: out_dir.join("include"),
        lib_dir: out_dir.join("lib"),
        makeflags: "-j".to_owned(),
        out_dir: out_dir.clone(),
        target_arch: env::var("CARGO_CFG_TARGET_ARCH").unwrap(),
    };
    fs::create_dir_all(&env.build_dir)
        .unwrap_or_else(|_| panic!("failed to create the directory: {:?}", env.build_dir));

    load_cache(&env);
    build(&env);
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

fn build(env: &Environment) {
    if env.lib_dir.join("libfftw3f.a").exists() {
        return;
    }

    let build_dir = env.build_dir.join("fftw-build");
    if !build_dir.exists() {
        execute_or_panic(Command::new("wget").current_dir(&env.build_dir).args(&[
            "--output-document",
            "fftw.tar.gz",
            "--quiet",
            FFTW_TAR_URL,
        ]));
        execute_or_panic(Command::new("mkdir").args(&[build_dir.to_str().unwrap()]));
        execute_or_panic(Command::new("tar").current_dir(&env.build_dir).args(&[
            "xf",
            "fftw.tar.gz",
            "--directory",
            build_dir.to_str().unwrap(),
            "--strip-components=1",
        ]));
    }

    execute_or_panic(
        Command::new("sh").current_dir(&build_dir).arg("-c").arg(
            [
                "./configure",
                "--prefix",
                env.out_dir.to_str().unwrap(),
                "--disable-fortran",
                "--enable-float",
                if env.target_arch == "x86_64" {
                    "--enable-avx2"
                } else if env.target_arch == "aarch64" {
                    "--enable-neon"
                } else {
                    panic!("unsupported architecture: {}", env.target_arch)
                },
            ]
            .join(" "),
        ),
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

fn run_bindgen(env: &Environment) {
    let binding_file = env.out_dir.join("fftw.rs");
    if binding_file.exists() {
        return;
    }

    bindgen::Builder::default()
        .header(env.include_dir.join("fftw3.h").to_str().unwrap())
        .allowlist_function("fftwf_.*")
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
        env.lib_dir.to_str().unwrap()
    );
    println!("cargo:rustc-link-lib=static=fftw3f");
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
