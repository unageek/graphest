use crate::{
    eval_result::{EvalExplicitResult, EvalParametricResult, EvalResult, RelationArgs},
    interval_set::TupperIntervalSet,
    traits::BytesAllocated,
    vars::VarSet,
};
use inari::Interval;
use std::{collections::HashMap, hash::Hash};

enum MultiKeyHashMap<K, V> {
    Zero(Option<V>),
    One(HashMap<[K; 1], V>),
    Two(HashMap<[K; 2], V>),
    Three(HashMap<[K; 3], V>),
    Four(HashMap<[K; 4], V>),
    Five(HashMap<[K; 5], V>),
}

impl<K, V> MultiKeyHashMap<K, V>
where
    K: Copy + Eq + Hash,
{
    fn new(n: usize) -> Self {
        match n {
            0 => MultiKeyHashMap::Zero(None),
            1 => MultiKeyHashMap::One(HashMap::new()),
            2 => MultiKeyHashMap::Two(HashMap::new()),
            3 => MultiKeyHashMap::Three(HashMap::new()),
            4 => MultiKeyHashMap::Four(HashMap::new()),
            5 => MultiKeyHashMap::Five(HashMap::new()),
            _ => panic!(),
        }
    }

    fn get(&self, k: &[K]) -> Option<&V> {
        match &self {
            MultiKeyHashMap::Zero(m) => m.into(),
            MultiKeyHashMap::One(m) => m.get(k),
            MultiKeyHashMap::Two(m) => m.get(k),
            MultiKeyHashMap::Three(m) => m.get(k),
            MultiKeyHashMap::Four(m) => m.get(k),
            MultiKeyHashMap::Five(m) => m.get(k),
        }
    }

    fn insert(&mut self, k: &[K], v: V) {
        match self {
            MultiKeyHashMap::Zero(m) => {
                *m = Some(v);
            }
            MultiKeyHashMap::One(m) => {
                m.insert(k.try_into().unwrap(), v);
            }
            MultiKeyHashMap::Two(m) => {
                m.insert(k.try_into().unwrap(), v);
            }
            MultiKeyHashMap::Three(m) => {
                m.insert(k.try_into().unwrap(), v);
            }
            MultiKeyHashMap::Four(m) => {
                m.insert(k.try_into().unwrap(), v);
            }
            MultiKeyHashMap::Five(m) => {
                m.insert(k.try_into().unwrap(), v);
            }
        }
    }
}

impl<K, V> BytesAllocated for MultiKeyHashMap<K, V> {
    fn bytes_allocated(&self) -> usize {
        match self {
            Self::Zero(_) => 0,
            Self::One(m) => m.bytes_allocated(),
            Self::Two(m) => m.bytes_allocated(),
            Self::Three(m) => m.bytes_allocated(),
            Self::Four(m) => m.bytes_allocated(),
            Self::Five(m) => m.bytes_allocated(),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum EvalCacheLevel {
    None,
    Univariate,
    Full,
}

/// A cache for memoizing evaluation of a relation.
pub struct EvalCacheGeneric<T: BytesAllocated> {
    cache_level: EvalCacheLevel,
    n_vars: usize,
    cx: Vec<HashMap<Interval, Vec<TupperIntervalSet>>>,
    c: MultiKeyHashMap<Interval, T>,
    bytes_allocated_by_values: usize,
}

impl<T: BytesAllocated> EvalCacheGeneric<T> {
    pub fn new(cache_level: EvalCacheLevel, vars: VarSet) -> Self {
        let n_vars = vars.len();
        Self {
            cache_level,
            n_vars,
            cx: vec![HashMap::new(); n_vars],
            c: MultiKeyHashMap::new(n_vars),
            bytes_allocated_by_values: 0,
        }
    }

    /// Clears the cache and releases the allocated memory.
    pub fn clear(&mut self) {
        self.cx = vec![HashMap::new(); self.n_vars];
        self.c = MultiKeyHashMap::new(self.n_vars);
        self.bytes_allocated_by_values = 0;
    }

    pub fn get(&self, args: &RelationArgs) -> Option<&T> {
        match self.cache_level {
            EvalCacheLevel::Full => self.c.get(args),
            _ => None,
        }
    }

    pub fn get_x(&self, index: usize, args: &RelationArgs) -> Option<&Vec<TupperIntervalSet>> {
        match self.cache_level {
            l if l >= EvalCacheLevel::Univariate => self.cx[index].get(&args[index]),
            _ => None,
        }
    }

    pub fn insert_with<F: FnOnce() -> T>(&mut self, args: &RelationArgs, f: F) {
        if self.cache_level == EvalCacheLevel::Full {
            let v = f();
            self.bytes_allocated_by_values += v.bytes_allocated();
            self.c.insert(args, v);
        }
    }

    pub fn insert_x_with<F: FnOnce() -> Vec<TupperIntervalSet>>(
        &mut self,
        index: usize,
        args: &RelationArgs,
        f: F,
    ) {
        if self.cache_level >= EvalCacheLevel::Univariate {
            let v = f();
            self.bytes_allocated_by_values +=
                v.bytes_allocated() + v.iter().map(|xs| xs.bytes_allocated()).sum::<usize>();
            self.cx[index].insert(args[index], v);
        }
    }
}

impl<T: BytesAllocated> BytesAllocated for EvalCacheGeneric<T> {
    fn bytes_allocated(&self) -> usize {
        self.cx.iter().map(|cx| cx.bytes_allocated()).sum::<usize>()
            + self.c.bytes_allocated()
            + self.bytes_allocated_by_values
    }
}

pub type EvalExplicitCache = EvalCacheGeneric<EvalExplicitResult>;
pub type EvalImplicitCache = EvalCacheGeneric<EvalResult>;
pub type EvalParametricCache = EvalCacheGeneric<EvalParametricResult>;
