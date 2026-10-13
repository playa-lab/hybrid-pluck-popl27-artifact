use std::collections::{BTreeSet, HashMap, HashSet};

use ordered_float::NotNan;
use rsdd::builder::bdd::RobddBuilder;
use rsdd::builder::cache::AllIteTable;
use rsdd::builder::BottomUpBuilder;
use rsdd::repr::{BddPtr, DDNNFPtr, PartialVariableOrder, VarLabel, VarOrder, WmcParams};
use rsdd::util::semirings::RealSemiring;

use crate::inference::conjugate_pairs::exponential::ExponentialPrior;
use crate::inference::conjugate_pairs::poisson::PoissonPrior;
use crate::inference::conjugate_pairs::{
    Assignment, BetaFlip, ContVarName, DirichletFlip, GammaDrawFamily, GammaFlip, GaussianFlip,
    PriorRegistry, RegistryInsert,
};
use crate::inference::spn::evidence::EvidenceLeaf;
use crate::inference::spn::Spn;
use crate::language::values::PluckVal;
use crate::utils::epsilon::RealEps;
use crate::utils::math::logsumexp;

pub type World = (PluckVal, BddPtr<'static>);

pub type GuardedWorlds = (Vec<World>, BddPtr<'static>);

#[derive(Clone)]
pub struct LazyKCConfig {
    pub max_depth: Option<usize>,
    pub time_limit: Option<f64>,
    pub use_strict_order: bool,
    pub show_bdd_size: bool,
}

impl Default for LazyKCConfig {
    fn default() -> Self {
        LazyKCConfig {
            max_depth: None,
            time_limit: None,
            use_strict_order: true,
            show_bdd_size: false,
        }
    }
}

#[derive(Default)]
pub struct LazyKCStats {
    pub num_compile_calls: usize,
    pub num_cache_hits: usize,
    pub num_cache_misses: usize,
    pub depth_reached: usize,
    pub num_thunk_first_eval: usize,
    pub num_thunk_widen: usize,
    pub num_tu_eval: usize,
    pub num_false_pc_exits: usize,
}

fn zobrist(depth: usize, val: i32) -> u64 {
    let mut h = (depth as u64).wrapping_mul(0x9E3779B97F4A7C15);
    h ^= (val as u64).wrapping_mul(0x517CC1B727220A95);
    h ^= h >> 33;
    h = h.wrapping_mul(0xFF51AFD7ED558CCD);
    h ^= h >> 33;
    h = h.wrapping_mul(0xC4CEB9FE1A85EC53);
    h ^= h >> 33;
    h
}

pub struct Callstack {
    stack: Vec<i32>,
    hash: u64,
}

impl Default for Callstack {
    fn default() -> Self {
        Self::new()
    }
}

impl Callstack {
    pub fn new() -> Self {
        Callstack {
            stack: Vec::new(),
            hash: 0,
        }
    }

    pub fn push(&mut self, val: i32) {
        let depth = self.stack.len();
        self.hash ^= zobrist(depth, val);
        self.stack.push(val);
    }

    pub fn pop(&mut self) -> i32 {
        let val = self.stack.pop().expect("pop from empty callstack");
        self.hash ^= zobrist(self.stack.len(), val);
        val
    }

    pub fn hash(&self) -> u64 {
        self.hash
    }
    pub fn as_slice(&self) -> &[i32] {
        &self.stack
    }
    pub fn to_vec(&self) -> Vec<i32> {
        self.stack.clone()
    }
    pub fn len(&self) -> usize {
        self.stack.len()
    }
    pub fn is_empty(&self) -> bool {
        self.stack.is_empty()
    }

    pub fn hash_with_suffix(&self, extra: &[i32]) -> u64 {
        let mut h = self.hash;
        for (i, &val) in extra.iter().enumerate() {
            h ^= zobrist(self.stack.len() + i, val);
        }
        h
    }

    pub fn from_vec(v: Vec<i32>) -> Self {
        let mut hash = 0u64;
        for (depth, &val) in v.iter().enumerate() {
            hash ^= zobrist(depth, val);
        }
        Callstack { stack: v, hash }
    }
}

pub type CallstackKey = (u64, u64);

pub type SortedCallstackKey = (Vec<i32>, u64);

pub enum FlipWeight {
    Constant(f64),
    Beta(BetaFlip),
    Dirichlet(DirichletFlip),
    Gamma(GammaFlip),
    Gaussian(GaussianFlip),
}

impl FlipWeight {
    pub fn weights(&self, registry: &PriorRegistry) -> (Spn<EvidenceLeaf>, Spn<EvidenceLeaf>) {
        match self {
            FlipWeight::Constant(p) => (Spn::scalar((1.0 - p).ln()), Spn::scalar(p.ln())),
            FlipWeight::Beta(f) => f.weights(),
            FlipWeight::Dirichlet(f) => f.weights(),
            FlipWeight::Gamma(f) => f.weights(),
            FlipWeight::Gaussian(f) => f.weights(registry),
        }
    }

    pub fn discriminator(&self) -> u64 {
        match self {
            FlipWeight::Constant(p) => p.to_bits(),
            FlipWeight::Beta(f) => f.discriminator(),
            FlipWeight::Dirichlet(f) => f.discriminator(),
            FlipWeight::Gamma(f) => f.discriminator(),
            FlipWeight::Gaussian(f) => f.discriminator(),
        }
    }

    pub fn is_pinned_observation(&self) -> bool {
        match self {
            FlipWeight::Constant(_) => false,
            FlipWeight::Beta(f) => f.is_pinned_observation(),
            FlipWeight::Dirichlet(f) => f.is_pinned_observation(),
            FlipWeight::Gamma(f) => f.is_pinned_observation(),
            FlipWeight::Gaussian(f) => f.is_pinned_observation(),
        }
    }

    pub fn sample_probability(&self, a: &Assignment) -> Option<f64> {
        match self {
            FlipWeight::Constant(p) => Some(*p),
            FlipWeight::Beta(f) => f.sample_probability(a),
            FlipWeight::Dirichlet(f) => f.sample_probability(a),
            FlipWeight::Gamma(f) => f.sample_probability(a),
            FlipWeight::Gaussian(f) => f.sample_probability(a),
        }
    }
}

type GammaDrawEntry = (GammaDrawFamily, (ContVarName, NotNan<f64>));

pub struct LazyKCState {
    pub builder: &'static mut RobddBuilder<'static, AllIteTable<BddPtr<'static>>>,
    pub wmc_params: WmcParams<Spn<EvidenceLeaf>>,
    pub bdd_true: BddPtr<'static>,
    pub bdd_false: BddPtr<'static>,
    pub callstack: Callstack,
    pub sorted_callstacks: Vec<SortedCallstackKey>,
    pub var_of_callstack: HashMap<CallstackKey, (Vec<i32>, BddPtr<'static>)>,
    pub sorted_var_labels: Vec<usize>,
    pub depth: usize,
    pub stats: LazyKCStats,
    pub cfg: LazyKCConfig,
    #[cfg(not(target_arch = "wasm32"))]
    pub start_time: Option<std::time::Instant>,
    #[cfg(target_arch = "wasm32")]
    pub start_time: Option<()>,
    pub hit_limit: bool,
    pub prior_registry: PriorRegistry,
    pub continuous_var_counter: u64,
    pub continuous_of_callstack: HashMap<u64, (Vec<i32>, ContVarName)>,
    pub continuous_blocks_of_callstack: HashMap<u64, (Vec<i32>, Vec<ContVarName>)>,
    pub observation_vars: HashMap<u64, (Vec<i32>, BddPtr<'static>)>,
    pub observation_var_labels: HashSet<usize>,
    pub gamma_draws: HashMap<ContVarName, GammaDrawEntry>,
}

impl LazyKCState {
    pub fn new(cfg: LazyKCConfig) -> Self {
        let order = VarOrder::linear_order(0);
        let builder = Box::new(RobddBuilder::<AllIteTable<BddPtr>>::new(order));
        let builder: &'static mut RobddBuilder<'static, AllIteTable<BddPtr<'static>>> =
            Box::leak(builder);

        let bdd_true = BddPtr::true_ptr();
        let bdd_false = BddPtr::false_ptr();

        let wmc_params = WmcParams::new(HashMap::new());

        LazyKCState {
            builder,
            wmc_params,
            bdd_true,
            bdd_false,
            callstack: Callstack::new(),
            sorted_callstacks: Vec::new(),
            var_of_callstack: HashMap::new(),
            sorted_var_labels: Vec::new(),
            depth: 0,
            stats: LazyKCStats::default(),
            cfg,
            start_time: None,
            hit_limit: false,
            prior_registry: PriorRegistry::new(),
            continuous_var_counter: 0,
            continuous_of_callstack: HashMap::new(),
            continuous_blocks_of_callstack: HashMap::new(),
            observation_vars: HashMap::new(),
            observation_var_labels: HashSet::new(),
            gamma_draws: HashMap::new(),
        }
    }

    pub fn current_address(&mut self, p: f64) -> BddPtr<'static> {
        self.current_address_weighted(FlipWeight::Constant(p))
    }

    pub fn register_prior<P: RegistryInsert>(&mut self, prior: P) {
        prior.insert_into(&mut self.prior_registry);
    }

    pub fn with_callstack_index<R>(&mut self, idx: i32, f: impl FnOnce(&mut Self) -> R) -> R {
        self.callstack.push(idx);
        let r = f(self);
        self.callstack.pop();
        r
    }

    pub fn current_address_weighted(&mut self, weight: FlipWeight) -> BddPtr<'static> {
        let ch = self.callstack.hash();
        let discriminator = weight.discriminator();
        let key: CallstackKey = (ch, discriminator);

        if let Some((stored_cs, bdd)) = self.var_of_callstack.get(&key) {
            debug_assert_eq!(
                stored_cs.as_slice(),
                self.callstack.as_slice(),
                "Zobrist hash collision in var_of_callstack! key={:?}",
                key
            );
            return *bdd;
        }

        let cs_vec = self.callstack.to_vec();
        let sort_key: SortedCallstackKey = (cs_vec.clone(), discriminator);
        let pos = self
            .sorted_callstacks
            .binary_search_by(|probe| probe.0.cmp(&sort_key.0).then(probe.1.cmp(&sort_key.1)))
            .unwrap_or_else(|pos| pos);

        let (var_label, new_var) = self.bdd().new_var_at_position(pos, true);

        let (low_weight, high_weight) = weight.weights(&self.prior_registry);
        self.wmc_params
            .set_weight(var_label, low_weight, high_weight);

        self.sorted_callstacks.insert(pos, sort_key);
        self.sorted_var_labels
            .insert(pos, var_label.value() as usize);
        self.var_of_callstack.insert(key, (cs_vec.clone(), new_var));

        if weight.is_pinned_observation() {
            self.observation_vars.insert(ch, (cs_vec, new_var));
            self.observation_var_labels
                .insert(var_label.value() as usize);
        }

        new_var
    }

    pub fn continuous_var_hash(callstack: &Callstack, params: &[f64]) -> u64 {
        let bits: Vec<i32> = params
            .iter()
            .flat_map(|&p| {
                let bits = p.to_bits();
                [bits as i32, (bits >> 32) as i32]
            })
            .collect();
        callstack.hash_with_suffix(&bits)
    }

    fn continuous_var_key_vec(callstack: &Callstack, params: &[f64]) -> Vec<i32> {
        let mut key = callstack.to_vec();
        for &p in params {
            let bits = p.to_bits() as i64;
            key.push(bits as i32);
            key.push((bits >> 32) as i32);
        }
        key
    }

    pub fn current_continuous_name(&mut self, params: &[f64]) -> ContVarName {
        let hash_key = Self::continuous_var_hash(&self.callstack, params);
        if let Some((stored_key, existing)) = self.continuous_of_callstack.get(&hash_key) {
            let full_key = Self::continuous_var_key_vec(&self.callstack, params);
            debug_assert_eq!(
                stored_key, &full_key,
                "Zobrist hash collision in continuous_of_callstack!"
            );
            return *existing;
        }
        let name = self.continuous_var_counter;
        self.continuous_var_counter += 1;
        let full_key = Self::continuous_var_key_vec(&self.callstack, params);
        self.continuous_of_callstack
            .insert(hash_key, (full_key, name));
        name
    }

    pub fn current_continuous_names(&mut self, params: &[f64], n: usize) -> Vec<ContVarName> {
        let hash_key = Self::continuous_var_hash(&self.callstack, params);
        if let Some((stored_key, existing)) = self.continuous_blocks_of_callstack.get(&hash_key) {
            let full_key = Self::continuous_var_key_vec(&self.callstack, params);
            debug_assert_eq!(
                stored_key, &full_key,
                "Zobrist hash collision in continuous_blocks_of_callstack!"
            );
            debug_assert_eq!(
                existing.len(),
                n,
                "continuous_blocks_of_callstack: re-entry with mismatched block size"
            );
            return existing.clone();
        }
        let names: Vec<ContVarName> = (0..n)
            .map(|_| {
                let name = self.continuous_var_counter;
                self.continuous_var_counter += 1;
                name
            })
            .collect();
        let full_key = Self::continuous_var_key_vec(&self.callstack, params);
        self.continuous_blocks_of_callstack
            .insert(hash_key, (full_key, names.clone()));
        names
    }

    pub fn wmc_spn(&self, bdd: BddPtr<'static>) -> Spn<EvidenceLeaf> {
        bdd.unsmoothed_wmc(&self.wmc_params)
    }

    pub fn wmc(&self, bdd: BddPtr<'static>) -> f64 {
        let spn_result = self.wmc_spn(bdd);
        spn_result
            .posterior(&self.prior_registry, &BTreeSet::new())
            .log_coeff()
            .to_f64()
    }

    pub fn wmc_real_eps(&self, bdd: BddPtr<'static>) -> RealEps {
        let spn_result = self.wmc_spn(bdd);
        spn_result
            .posterior(&self.prior_registry, &BTreeSet::new())
            .log_coeff()
    }

    pub fn wmc_params_real(&self) -> WmcParams<RealSemiring> {
        let mut params = WmcParams::<RealSemiring>::new(HashMap::new());
        for &label_val in &self.sorted_var_labels {
            let label = VarLabel::new(label_val as u64);
            let (low_spn, high_spn) = self.wmc_params.var_weight(label);
            let low_f64 = low_spn
                .posterior(&self.prior_registry, &BTreeSet::new())
                .log_coeff()
                .to_f64();
            let high_f64 = high_spn
                .posterior(&self.prior_registry, &BTreeSet::new())
                .log_coeff()
                .to_f64();
            params.set_weight(label, RealSemiring(low_f64), RealSemiring(high_f64));
        }
        params
    }

    pub fn bdd(&self) -> &'static RobddBuilder<'static, AllIteTable<BddPtr<'static>>> {
        unsafe { &*(self.builder as *const _) }
    }

    pub(crate) fn implies(
        &self,
        constraint: BddPtr<'static>,
        var_bdd: BddPtr<'static>,
    ) -> Option<bool> {
        let builder = self.bdd();
        let neg_constraint = builder.negate(constraint);
        if builder.or(neg_constraint, var_bdd).is_true() {
            return Some(true);
        }
        let neg_var = builder.negate(var_bdd);
        if builder.or(neg_constraint, neg_var).is_true() {
            return Some(false);
        }
        None
    }

    pub fn constraint_forces(
        &self,
        constraint: BddPtr<'static>,
        callstack_hash: u64,
        key: u64,
    ) -> Option<bool> {
        let (_, var_bdd) = self.var_of_callstack.get(&(callstack_hash, key))?;
        self.implies(constraint, *var_bdd)
    }

    pub fn constraint_forces_observation(
        &self,
        constraint: BddPtr<'static>,
        callstack_hash: u64,
    ) -> Option<bool> {
        let (_, var_bdd) = self.observation_vars.get(&callstack_hash)?;
        self.implies(constraint, *var_bdd)
    }

    pub fn sample_continuous_posterior<R: rand::Rng>(
        &self,
        evidence_spn: &Spn<EvidenceLeaf>,
        rng: &mut R,
    ) -> Assignment {
        let evidence_scope: BTreeSet<ContVarName> = evidence_spn.scope().iter().copied().collect();
        let touched = evidence_spn
            .posterior(&self.prior_registry, &evidence_scope)
            .sample(rng);

        let full_scope: BTreeSet<ContVarName> = self.prior_registry.scope().copied().collect();
        let missing: BTreeSet<ContVarName> =
            full_scope.difference(&touched.scope()).copied().collect();
        let untouched = self.prior_registry.slice(&missing).sample(rng);

        touched.merge(&untouched)
    }

    pub fn realize_missing_gamma_draws<R: rand::Rng>(
        &self,
        labels: impl IntoIterator<Item = usize>,
        assignment: &mut Assignment,
        rng: &mut R,
    ) {
        let mut covered = assignment.scope();
        for label in labels {
            let (low, high) = self.wmc_params.var_weight(VarLabel::new(label as u64));
            for var in low.scope().iter().chain(high.scope().iter()) {
                if covered.contains(var) {
                    continue;
                }
                let Some((family, (rate, scale))) = self.gamma_draws.get(var).copied() else {
                    continue;
                };
                let lambda = assignment.gamma_value(rate) * scale;
                let sampled = match family {
                    GammaDrawFamily::Poisson => {
                        PoissonPrior::unconstrained(std::iter::once(*var)).sample(lambda, rng)
                    }
                    GammaDrawFamily::Exponential => {
                        ExponentialPrior::unconstrained(std::iter::once(*var)).sample(lambda, rng)
                    }
                };
                let single =
                    Assignment::from_sorted(vec![], vec![], sampled.into_iter().collect(), vec![]);
                *assignment = assignment.merge(&single);
                covered.insert(*var);
            }
        }
    }

    pub fn cover_support_gamma_draws<R: rand::Rng>(
        &self,
        bdd: BddPtr<'static>,
        assignment: &mut Assignment,
        rng: &mut R,
    ) {
        let mut support: BTreeSet<usize> = BTreeSet::new();
        support_vars(bdd, &mut support);
        self.realize_missing_gamma_draws(support, assignment, rng);
    }

    pub fn sample_discrete_given<R: rand::Rng>(
        &self,
        evidence_bdd: BddPtr<'static>,
        assignment: &Assignment,
        rng: &mut R,
    ) -> BddPtr<'static> {
        let mut support: BTreeSet<usize> = BTreeSet::new();
        support_vars(evidence_bdd, &mut support);
        let mut log_weights: HashMap<usize, (f64, f64)> = HashMap::new();
        for &label_val in &support {
            let label = VarLabel::new(label_val as u64);
            let (low_spn, high_spn) = self.wmc_params.var_weight(label);
            let log_low = low_spn.log_likelihood(assignment);
            let log_high = high_spn.log_likelihood(assignment);
            log_weights.insert(label_val, (log_low, log_high));
        }

        let mut log_cache: HashMap<u64, f64> = HashMap::new();
        bottomup_log_marginal(evidence_bdd, &log_weights, &mut log_cache);

        let default_log_weight = (-std::f64::consts::LN_2, -std::f64::consts::LN_2);

        let builder = self.bdd();
        let mut result = BddPtr::true_ptr();
        let mut current = evidence_bdd;
        let mut aborted = false;
        loop {
            match current {
                BddPtr::PtrTrue => break,
                BddPtr::PtrFalse => {
                    eprintln!("sample_discrete_given: reached False during sampling");
                    aborted = true;
                    break;
                }
                _ => {
                    let (l, h) = if current.is_neg() {
                        (current.low_raw().neg(), current.high_raw().neg())
                    } else {
                        (current.low_raw(), current.high_raw())
                    };
                    let var = current.var().unwrap();

                    let choose_high = if l.is_false() {
                        true
                    } else if h.is_false() {
                        false
                    } else {
                        let (log_w_lo, log_w_hi) = log_weights
                            .get(&(var.value() as usize))
                            .copied()
                            .unwrap_or(default_log_weight);
                        let log_m_lo = *log_cache.get(&bdd_key(l)).unwrap_or(&f64::NEG_INFINITY);
                        let log_m_hi = *log_cache.get(&bdd_key(h)).unwrap_or(&f64::NEG_INFINITY);
                        let log_prob_low = log_w_lo + log_m_lo;
                        let log_prob_high = log_w_hi + log_m_hi;
                        let log_total = logsumexp(log_prob_low, log_prob_high);
                        if log_total == f64::NEG_INFINITY {
                            eprintln!("sample_discrete_given: zero total at var {:?}", var);
                            aborted = true;
                            break;
                        }
                        let u: f64 = rng.gen();
                        u.ln() < log_prob_high - log_total
                    };

                    let lit = builder.var(var, choose_high);
                    result = builder.and(result, lit);
                    current = if choose_high { h } else { l };
                }
            }
        }
        if aborted {
            BddPtr::false_ptr()
        } else {
            result
        }
    }

    pub fn var_sample_prob(&self, label: usize, assignment: &Assignment) -> f64 {
        let (low_spn, high_spn) = self.wmc_params.var_weight(VarLabel::new(label as u64));
        let ll = low_spn.log_likelihood(assignment);
        let lh = high_spn.log_likelihood(assignment);
        debug_assert!(
            ll != f64::NEG_INFINITY || lh != f64::NEG_INFINITY,
            "var_sample_prob: both edge weights are zero for var {}",
            label
        );
        1.0 / (1.0 + (ll - lh).exp())
    }

    pub fn check_time_limit(&self) -> bool {
        #[cfg(not(target_arch = "wasm32"))]
        {
            if let (Some(start), Some(limit)) = (self.start_time, self.cfg.time_limit) {
                return start.elapsed().as_secs_f64() > limit;
            }
        }
        false
    }
}

pub fn support_vars(bdd: BddPtr<'static>, out: &mut BTreeSet<usize>) {
    fn walk(ptr: BddPtr<'static>, out: &mut BTreeSet<usize>, visited: &mut HashSet<u64>) {
        match ptr {
            BddPtr::PtrTrue | BddPtr::PtrFalse => {}
            BddPtr::Reg(node) | BddPtr::Compl(node) => {
                if !visited.insert(node as *const _ as u64) {
                    return;
                }
                out.insert(ptr.var().unwrap().value() as usize);
                walk(ptr.low_raw(), out, visited);
                walk(ptr.high_raw(), out, visited);
            }
        }
    }
    let mut visited = HashSet::new();
    walk(bdd, out, &mut visited);
}

fn bottomup_log_marginal(
    ptr: BddPtr<'static>,
    log_weights: &HashMap<usize, (f64, f64)>,
    cache: &mut HashMap<u64, f64>,
) -> f64 {
    let key = bdd_key(ptr);
    if let Some(&val) = cache.get(&key) {
        return val;
    }
    let result = match ptr {
        BddPtr::PtrTrue => 0.0,
        BddPtr::PtrFalse => f64::NEG_INFINITY,
        _ => {
            let (l, h) = if ptr.is_neg() {
                (ptr.low_raw().neg(), ptr.high_raw().neg())
            } else {
                (ptr.low_raw(), ptr.high_raw())
            };
            let log_low_m = bottomup_log_marginal(l, log_weights, cache);
            let log_high_m = bottomup_log_marginal(h, log_weights, cache);
            let var = ptr.var().unwrap();
            let (log_w_lo, log_w_hi) = log_weights
                .get(&(var.value() as usize))
                .copied()
                .unwrap_or((-std::f64::consts::LN_2, -std::f64::consts::LN_2));
            logsumexp(log_w_lo + log_low_m, log_w_hi + log_high_m)
        }
    };
    cache.insert(key, result);
    result
}

fn bdd_key(ptr: BddPtr<'static>) -> u64 {
    match ptr {
        BddPtr::PtrTrue => 0,
        BddPtr::PtrFalse => 1,
        BddPtr::Reg(node) => (node as *const _ as u64) << 1,
        BddPtr::Compl(node) => ((node as *const _ as u64) << 1) | 1,
    }
}
