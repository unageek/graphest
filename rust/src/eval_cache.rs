use crate::{
    eval_result::{EvalArgs, EvalExplicitResult, EvalParametricResult, EvalResult},
    interval_set::TupperIntervalSet,
    ops::{OptionalValueStore, StaticTerm, StaticTermKind, StoreIndex},
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
    Six(HashMap<[K; 6], V>),
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
            6 => MultiKeyHashMap::Six(HashMap::new()),
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
            MultiKeyHashMap::Six(m) => m.entry(k.try_into().unwrap()).or_insert_with(f),
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
            Self::Six(m) => m.bytes_allocated(),
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

fn maximal_term_indices(vars: VarSet, terms: &[StaticTerm]) -> Vec<StoreIndex> {
    use StaticTermKind::*;

    let mut is_maximal = vec![false; terms.len()];
    for (i, term) in terms.iter().enumerate() {
        if term.vars != vars {
            continue;
        }

        let indices = match &term.kind {
            Unary(_, k) => vec![*k],
            Binary(_, k, l) => vec![*k, *l],
            Ternary(_, k, l, m) => vec![*k, *l, *m],
            Pown(k, _) => vec![*k],
            Rootn(k, _) => vec![*k],
            RankedMinMax(_, ks, l) => {
                let mut indices = ks.clone();
                indices.push(*l);
                indices
            }
            Constant(_) | Var(_, _) => continue,
        };

        for index in indices {
            is_maximal[index.get()] = false;
        }
        is_maximal[i] = true;
    }

    is_maximal
        .into_iter()
        .enumerate()
        .filter_map(|(i, maximal)| {
            if maximal {
                Some(StoreIndex::new(i))
            } else {
                None
            }
        })
        .collect::<Vec<_>>()
}

pub struct MaximalTermCache<const N: usize> {
    arg_indices: [usize; N],
    cache: HashMap<[Interval; N], Vec<Option<TupperIntervalSet>>>,
    term_indices: Vec<StoreIndex>,
    bytes_allocated_by_values: usize,
}

impl<const N: usize> MaximalTermCache<N> {
    fn new(arg_indices: [usize; N], term_indices: Vec<StoreIndex>) -> Self {
        Self {
            arg_indices,
            cache: HashMap::new(),
            term_indices,
            bytes_allocated_by_values: 0,
        }
    }

    /// Clears the cache and releases the allocated memory.
    fn clear(&mut self) {
        self.cache = HashMap::new();
        self.bytes_allocated_by_values = 0;
    }

    pub fn restore(&self, args: &EvalArgs, store: &mut OptionalValueStore<TupperIntervalSet>) {
        let mut key = [Interval::EMPTY; N];
        for (i, &arg_index) in self.arg_indices.iter().enumerate() {
            key[i] = args[arg_index];
        }
        if let Some(vs) = self.cache.get(&key) {
            for (i, v) in self.term_indices.iter().zip(vs.iter()) {
                if let Some(v) = v {
                    store.insert(*i, v.clone());
                }
            }
        }
    }

    pub fn store(&mut self, args: &EvalArgs, store: &OptionalValueStore<TupperIntervalSet>) {
        let mut key = [Interval::EMPTY; N];
        for (i, &arg_index) in self.arg_indices.iter().enumerate() {
            key[i] = args[arg_index];
        }
        if let Some(vs) = self.cache.get_mut(&key) {
            for (i, v) in self.term_indices.iter().zip(vs.iter_mut()) {
                if v.is_none() {
                    *v = store.get(*i).cloned();
                    self.bytes_allocated_by_values +=
                        v.iter().map(|xs| xs.bytes_allocated()).sum::<usize>();
                }
            }
        } else {
            let mut vs = vec![None; self.term_indices.len()];
            for (i, v) in self.term_indices.iter().zip(vs.iter_mut()) {
                *v = store.get(*i).cloned();
                self.bytes_allocated_by_values +=
                    v.iter().map(|xs| xs.bytes_allocated()).sum::<usize>();
            }
            self.bytes_allocated_by_values += vs.bytes_allocated();
            self.cache.insert(key, vs);
        }
    }
}

impl<const N: usize> BytesAllocated for MaximalTermCache<N> {
    fn bytes_allocated(&self) -> usize {
        self.cache.bytes_allocated() + self.bytes_allocated_by_values
    }
}

/// A cache for memoizing evaluation of a relation.
pub struct EvalCache<T: BytesAllocated> {
    pub full: FullCache<T>,
    pub univariate: Vec<MaximalTermCache<1>>,
    level: EvalCacheLevel,
    initialized: bool,
}

impl<T: BytesAllocated> EvalCache<T> {
    pub fn new(cache_level: EvalCacheLevel, vars: VarSet) -> Self {
        Self {
            full: FullCache::new(cache_level, vars),
            univariate: vec![],
            level: cache_level,
            initialized: false,
        }
    }

    pub fn setup(&mut self, terms: &[StaticTerm], vars_ordered: &[VarSet]) {
        if self.initialized || self.level < EvalCacheLevel::Univariate {
            return;
        }

        let mut univariate_vars = vec![];
        for term in terms {
            if term.vars.len() == 1 {
                univariate_vars.push(term.vars);
            }
        }
        univariate_vars.sort();
        univariate_vars.dedup();

        self.univariate = univariate_vars
            .into_iter()
            .filter_map(|vars| {
                let indices = maximal_term_indices(vars, terms);
                if indices.is_empty() {
                    return None;
                }
                Some(MaximalTermCache::new(
                    [vars_ordered.iter().position(|&v| v == vars).unwrap()],
                    indices,
                ))
            })
            .collect();

        self.initialized = true;
    }

    /// Clears the cache and releases allocated memory.
    pub fn clear(&mut self) {
        self.full.clear();
        for cache in &mut self.univariate {
            cache.clear();
        }
    }
}

impl<T: BytesAllocated> BytesAllocated for EvalCache<T> {
    fn bytes_allocated(&self) -> usize {
        self.full.bytes_allocated()
            + self
                .univariate
                .iter()
                .map(|c| c.bytes_allocated())
                .sum::<usize>()
    }
}

pub type EvalExplicitCache = EvalCache<EvalExplicitResult>;
pub type EvalImplicitCache = EvalCache<EvalResult>;
pub type EvalParametricCache = EvalCache<EvalParametricResult>;
