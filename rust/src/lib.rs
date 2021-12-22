#![allow(clippy::float_cmp)]
#![feature(box_patterns, box_syntax, default_free_fn, once_cell)]

pub use crate::{
    geom::Box2D,
    graph::{
        constant::Constant, explicit::Explicit, implicit::Implicit, parametric::Parametric, Graph,
        GraphingStatistics, Padding,
    },
    image::{Image, PixelIndex, PixelRange},
    relation::{Relation, RelationType},
    ternary::Ternary,
};

#[cfg(feature = "arb")]
mod arb;
#[cfg(feature = "arb")]
mod arb_interval_set_ops;
mod ast;
mod block;
mod context;
mod eval_cache;
mod eval_result;
mod geom;
mod graph;
mod image;
mod interval_set;
mod interval_set_ops;
mod ops;
mod parse;
mod rational_ops;
mod real;
mod region;
mod relation;
mod ternary;
mod traits;
mod vars;
mod visit;
