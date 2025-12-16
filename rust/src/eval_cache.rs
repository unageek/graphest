use crate::{
    eval_result::{EvalArgs, EvalExplicitResult, EvalParametricResult, EvalResult},
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

    fn get_or_insert_with<F>(&mut self, k: &[K], f: F) -> &V
    where
        F: FnOnce() -> V,
    {
        match self {
            MultiKeyHashMap::Zero(m) => m.get_or_insert_with(f),
            MultiKeyHashMap::One(m) => m.entry(k.try_into().unwrap()).or_insert_with(f),
            MultiKeyHashMap::Two(m) => m.entry(k.try_into().unwrap()).or_insert_with(f),
            MultiKeyHashMap::Three(m) => m.entry(k.try_into().unwrap()).or_insert_with(f),
            MultiKeyHashMap::Four(m) => m.entry(k.try_into().unwrap()).or_insert_with(f),
            MultiKeyHashMap::Five(m) => m.entry(k.try_into().unwrap()).or_insert_with(f),
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

pub struct FullCache<T: BytesAllocated> {
    cache_level: EvalCacheLevel,
    n_vars: usize,
    c: MultiKeyHashMap<Interval, T>,
    last_value: Option<T>,
    bytes_allocated_by_values: usize,
}

impl<T: BytesAllocated> FullCache<T> {
    fn new(cache_level: EvalCacheLevel, vars: VarSet) -> Self {
        let n_vars = vars.len();
        Self {
            cache_level,
            n_vars,
            c: MultiKeyHashMap::new(n_vars),
            last_value: None,
            bytes_allocated_by_values: 0,
        }
    }

    fn clear(&mut self) {
        self.c = MultiKeyHashMap::new(self.n_vars);
        self.bytes_allocated_by_values = 0;
    }

    pub fn get_or_insert_with<F>(&mut self, args: &EvalArgs, f: F) -> &T
    where
        F: FnOnce() -> T,
    {
        if self.cache_level == EvalCacheLevel::Full {
            self.c.get_or_insert_with(args, || {
                let v = f();
                self.bytes_allocated_by_values += v.bytes_allocated();
                v
            })
        } else {
            self.last_value.insert(f())
        }
    }
}

impl<T: BytesAllocated> BytesAllocated for FullCache<T> {
    fn bytes_allocated(&self) -> usize {
        self.c.bytes_allocated() + self.bytes_allocated_by_values
    }
}

pub struct UnivariateCache {
    cache_level: EvalCacheLevel,
    n_vars: usize,
    cs: Vec<HashMap<Interval, Vec<TupperIntervalSet>>>,
    bytes_allocated_by_values: usize,
}

impl UnivariateCache {
    fn new(cache_level: EvalCacheLevel, vars: VarSet) -> Self {
        let n_vars = vars.len();
        Self {
            cache_level,
            n_vars,
            cs: vec![HashMap::new(); n_vars],
            bytes_allocated_by_values: 0,
        }
    }

    /// Clears the cache and releases the allocated memory.
    fn clear(&mut self) {
        self.cs = vec![HashMap::new(); self.n_vars];
        self.bytes_allocated_by_values = 0;
    }

    pub fn get(&self, index: usize, args: &EvalArgs) -> Option<&Vec<TupperIntervalSet>> {
        if self.cache_level >= EvalCacheLevel::Univariate {
            self.cs[index].get(&args[index])
        } else {
            None
        }
    }

    pub fn insert_with<F: FnOnce() -> Vec<TupperIntervalSet>>(
        &mut self,
        index: usize,
        args: &EvalArgs,
        f: F,
    ) {
        if self.cache_level >= EvalCacheLevel::Univariate {
            let v = f();
            self.bytes_allocated_by_values +=
                v.bytes_allocated() + v.iter().map(|xs| xs.bytes_allocated()).sum::<usize>();
            if let Some(old_v) = self.cs[index].insert(args[index], v) {
                self.bytes_allocated_by_values -= old_v.bytes_allocated()
                    + old_v.iter().map(|xs| xs.bytes_allocated()).sum::<usize>();
            }
        }
    }
}

impl BytesAllocated for UnivariateCache {
    fn bytes_allocated(&self) -> usize {
        self.cs.iter().map(|c| c.bytes_allocated()).sum::<usize>() + self.bytes_allocated_by_values
    }
}

/// A cache for memoizing evaluation of a relation.
pub struct EvalCache<T: BytesAllocated> {
    pub full: FullCache<T>,
    pub univariate: UnivariateCache,
}

impl<T: BytesAllocated> EvalCache<T> {
    pub fn new(cache_level: EvalCacheLevel, vars: VarSet) -> Self {
        Self {
            full: FullCache::new(cache_level, vars),
            univariate: UnivariateCache::new(cache_level, vars),
        }
    }

    /// Clears the cache and releases allocated memory.
    pub fn clear(&mut self) {
        self.full.clear();
        self.univariate.clear();
    }
}

impl<T: BytesAllocated> BytesAllocated for EvalCache<T> {
    fn bytes_allocated(&self) -> usize {
        self.full.bytes_allocated() + self.univariate.bytes_allocated()
    }
}

pub type EvalExplicitCache = EvalCache<EvalExplicitResult>;
pub type EvalImplicitCache = EvalCache<EvalResult>;
pub type EvalParametricCache = EvalCache<EvalParametricResult>;
