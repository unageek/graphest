use crate::{
    context::{Context, Def},
    parse::parse_expr,
};

#[derive(Clone, Debug)]
pub struct RawDefinition {
    pub name: String,
    pub body: String,
}

pub fn parse_definitions(defs: &Vec<RawDefinition>) -> Context {
    let mut ctx = Context::new();

    for def in defs {
        ctx = ctx.def(
            &def.name,
            Def::Constant {
                body: parse_expr(&def.body, &[Context::builtin()]).unwrap(),
            },
        );
    }

    ctx
}
