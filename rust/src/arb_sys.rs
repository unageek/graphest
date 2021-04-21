#![allow(
    dead_code,
    deref_nullptr, // https://github.com/rust-lang/rust-bindgen/issues/1651
    non_camel_case_types,
    non_snake_case,
    non_upper_case_globals,
    clippy::upper_case_acronyms
)]

include!(concat!(env!("OUT_DIR"), "/arb_sys.rs"));
