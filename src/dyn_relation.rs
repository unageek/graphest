use crate::{interval_set::*, parse::*, rel::*, visit::*};
use std::str::FromStr;

#[derive(Clone, Debug)]
pub struct DynRelation {
    exprs: Vec<StaticExpr>,
    rels: Vec<StaticRel>,
    ts: Vec<TupperIntervalSet>,
    es: Vec<EvalResult>,
}

impl DynRelation {
    pub fn evaluate(&mut self, x: TupperIntervalSet, y: TupperIntervalSet) -> EvalResult {
        for i in 0..self.exprs.len() {
            match &self.exprs[i].kind {
                StaticExprKind::Constant(_) => (),
                StaticExprKind::X => self.ts[i] = x.clone(),
                StaticExprKind::Y => self.ts[i] = y.clone(),
                _ => self.ts[i] = self.exprs[i].evaluate(&self.ts),
            }
        }
        for i in 0..self.rels.len() {
            self.es[i] = self.rels[i].evaluate(&self.ts, &self.es);
        }
        self.es.last().unwrap().clone()
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
        let mut v = AssignIdStage1::new();
        v.visit_rel(&rel);
        let mut v = AssignIdStage2::new(v);
        v.visit_rel(&rel);
        let mut v = CollectStatic::new(v);
        v.visit_rel(&rel);
        let (exprs, rels) = v.exprs_rels();
        let n_ts = exprs.len();
        let n_es = rels.len();
        let mut slf = Self {
            exprs,
            rels,
            ts: vec![TupperIntervalSet::empty(); n_ts],
            es: vec![EvalResult::default(); n_es],
        };
        slf.initialize();
        Ok(slf)
    }
}
