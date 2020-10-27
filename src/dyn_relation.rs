use crate::{
    eval_result::EvalResult,
    interval_set::TupperIntervalSet,
    parse::parse,
    rel::{StaticExpr, StaticExprKind, StaticRel, StaticRelKind},
    visit::*,
};
use std::str::FromStr;

#[derive(Clone, Debug)]
pub struct DynRelation {
    exprs: Vec<StaticExpr>,
    rels: Vec<StaticRel>,
    n_atom_rels: usize,
    ts: Vec<TupperIntervalSet>,
}

impl DynRelation {
    pub fn evaluate(&mut self, x: TupperIntervalSet, y: TupperIntervalSet) -> EvalResult {
        use StaticExprKind::*;
        for i in 0..self.exprs.len() {
            match &self.exprs[i].kind {
                Constant(_) => (),
                X => self.ts[i] = x.clone(),
                Y => self.ts[i] = y.clone(),
                _ => self.ts[i] = self.exprs[i].evaluate(&self.ts),
            }
        }
        EvalResult(
            self.rels
                .iter()
                .take(self.n_atom_rels)
                .map(|r| r.evaluate(&self.ts))
                .collect(),
        )
    }

    pub fn rels(&self) -> &Vec<StaticRel> {
        &self.rels
    }

    fn initialize(&mut self) {
        for i in 0..self.exprs.len() {
            if let StaticExprKind::Constant(_) = &self.exprs[i].kind {
                self.ts[i] = self.exprs[i].evaluate(&self.ts);
            }
        }
    }
}

impl FromStr for DynRelation {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, String> {
        let mut rel = parse(s)?;
        Transform.visit_rel_mut(&mut rel);
        FoldConstant.visit_rel_mut(&mut rel);
        Transform.visit_rel_mut(&mut rel);
        FoldConstant.visit_rel_mut(&mut rel);
        let mut v = AssignIdStage1::new();
        v.visit_rel(&rel);
        let mut v = AssignIdStage2::new(v);
        v.visit_rel(&rel);
        let mut v = CollectStatic::new(v);
        v.visit_rel(&rel);
        let (exprs, rels) = v.exprs_rels();
        let n_ts = exprs.len();
        let n_atom_rels = rels
            .iter()
            .filter(|r| matches!(r.kind, StaticRelKind::Atomic(_, _, _)))
            .count();
        let mut slf = Self {
            exprs,
            rels,
            n_atom_rels,
            ts: vec![TupperIntervalSet::empty(); n_ts],
        };
        slf.initialize();
        Ok(slf)
    }
}
