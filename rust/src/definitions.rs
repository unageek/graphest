use crate::{
    context::{Context, Def},
    parse::parse_expr,
};

#[derive(Clone, Debug)]
pub struct RawDefinition {
    pub name: String,
    pub body: String,
}

pub fn parse_definitions(raw_defs: &Vec<RawDefinition>) -> Context {
    let mut ctx = Context::new();

    for raw_def in raw_defs {
        let def = Def::Constant {
            body: parse_expr(&raw_def.body, &[Context::builtin(), &ctx]).unwrap(),
        };
        ctx = ctx.def(&raw_def.name, def);
    }

    ctx
}
