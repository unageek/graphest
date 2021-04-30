#![allow(clippy::float_cmp)]
#![feature(once_cell)]

pub mod graph;
pub mod relation;

#[cfg(feature = "arb")]
mod arb;
#[cfg(feature = "arb")]
mod arb_interval_set_ops;
#[cfg(feature = "arb")]
mod arb_sys;
mod ast;
mod context;
mod eval_result;
mod image_block_queue;
mod interval_set;
mod interval_set_ops;
mod ops;
mod parse;
mod rational_ops;
mod visit;
