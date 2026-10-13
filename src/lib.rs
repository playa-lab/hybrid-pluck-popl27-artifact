
pub mod inference;
pub mod language;
pub mod utils;

#[cfg(target_arch = "wasm32")]
pub mod wasm;

use itertools::Itertools;
use nalgebra as na;
use rsdd::repr::BddPtr;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use inference::conjugate_pairs::{
    ContVarName, EvidenceLeaf, ExponentialObs, GammaSuffStat, GaussianAffineExpr, GaussianPrior,
    JointGammaPrior, PoissonObs, PriorRegistry,
};
use inference::lazy_kc::ctx::CompilerCtx;
use inference::lazy_kc::state::{LazyKCConfig, LazyKCState};
use inference::sampling::posterior_sample;
use inference::spn::Spn;
use language::define::DefinitionRegistry;
use language::pexpr::PExpr;
use language::toplevel::{parse_toplevel, TopLevelForm};
use language::types::{StringInterner, TypeRegistry};
use language::values::{empty_env, env_cons, PluckVal, PluckValue};
use utils::epsilon::RealEps;
use utils::formatting::{format_value, NamingScheme};
use utils::intervals::IntervalOrEq;
use utils::stats::{KcStats, StageTimes};
use utils::union_find::UnionFind;

use crate::language::values::EnvInner;


fn build_resolver(file_registry: &HashMap<String, String>) -> impl Fn(&str) -> Option<String> + '_ {
    |path: &str| {
        if let Some(content) = file_registry.get(path) {
            return Some(content.clone());
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            if let Ok(content) = std::fs::read_to_string(path) {
                return Some(content);
            }
        }
        None
    }
}

pub struct PluckContext {
    pub interner: StringInterner,
    pub types: TypeRegistry,
    pub defs: DefinitionRegistry,
    pub file_registry: HashMap<String, String>,
    syms: Syms,
    parse_time: Duration,
}

struct Syms {
    true_sym: u32,
    false_sym: u32,
    marginal_sym: u32,
    posterior_sym: u32,
    posterior_samples_sym: u32,
    gibbs_sym: u32,
    o_sym: u32,
    s_sym: u32,
    nil_sym: u32,
    cons_sym: u32,
    with_prior_sample_sym: u32,
    with_constant_sym: u32,
}

impl PluckContext {
    pub fn new() -> Self {
        let mut interner = StringInterner::new();
        let mut types = TypeRegistry::new();
        types.register_builtins(&mut interner);

        let syms = Syms {
            true_sym: interner.intern("True"),
            false_sym: interner.intern("False"),
            marginal_sym: interner.intern("Marginal"),
            posterior_sym: interner.intern("Posterior"),
            posterior_samples_sym: interner.intern("PosteriorSamples"),
            gibbs_sym: interner.intern("Gibbs"),
            o_sym: interner.intern("O"),
            s_sym: interner.intern("S"),
            nil_sym: interner.intern("Nil"),
            cons_sym: interner.intern("Cons"),
            with_prior_sample_sym: interner.intern("WithPriorSample"),
            with_constant_sym: interner.intern("WithConstant"),
        };

        let mut ctx = PluckContext {
            interner,
            types,
            defs: DefinitionRegistry::new(),
            file_registry: HashMap::new(),
            syms,
            parse_time: Duration::ZERO,
        };
        ctx.load_stdlib();
        ctx
    }

    fn load_stdlib(&mut self) {
        let stdlib_source = include_str!("../stdlib.pluck");
        let resolver = build_resolver(&self.file_registry);
        parse_toplevel(
            stdlib_source,
            &mut self.interner,
            &mut self.types,
            &mut self.defs,
            &resolver,
        );
    }

    pub fn register_file(&mut self, name: String, content: String) {
        self.file_registry.insert(name, content);
    }

    pub fn run(&mut self, source: &str) -> Vec<QueryResult> {
        self.run_with_progress(source, None)
    }

    pub fn run_with_progress(
        &mut self,
        source: &str,
        progress: Option<&GibbsProgress>,
    ) -> Vec<QueryResult> {
        let forms = {
            let resolver = build_resolver(&self.file_registry);
            let parse_start = std::time::Instant::now();
            let forms = parse_toplevel(
                source,
                &mut self.interner,
                &mut self.types,
                &mut self.defs,
                &resolver,
            );
            self.parse_time = parse_start.elapsed();
            forms
        };

        let mut results = Vec::new();
        for form in forms {
            if let TopLevelForm::Query { expr, display_str } = form {
                results.push(self.process_query(&expr, &display_str, progress));
            }
        }
        results
    }

    fn get_query_val(&self, expr: &PExpr, state: &mut LazyKCState) -> Option<Rc<PluckValue>> {
        let env = empty_env();
        let mut ctx = CompilerCtx {
            types: &self.types,
            defs: &self.defs,
            interner: &self.interner,
            true_sym: self.syms.true_sym,
            false_sym: self.syms.false_sym,
            state,
        };
        let mut mode = inference::lazy_kc::compile::CompileMode::KC {
            path_condition: ctx.state.bdd_true,
        };
        let (worlds, _) = ctx.traced_compile(expr, &env, &mut mode, 0);

        if worlds.len() != 1 {
            eprintln!(
                    "Query must evaluate to a single (Marginal …) / (Posterior …) / (PosteriorSamples …) value, got {} worlds",
                    worlds.len()
                );
            return None;
        }

        Some(worlds.into_iter().next().unwrap().0)
    }

    fn process_query(
        &mut self,
        expr: &PExpr,
        display_str: &str,
        progress: Option<&GibbsProgress>,
    ) -> QueryResult {
        inference::spn::evidence::clear_intern_table();
        inference::spn::evidence::clear_mul_cache();
        let mut state = LazyKCState::new(LazyKCConfig::default());
        #[cfg(not(target_arch = "wasm32"))]
        {
            state.start_time = Some(std::time::Instant::now());
        }

        let sym_start = std::time::Instant::now();
        let query_val = match self.get_query_val(expr, &mut state) {
            Some(val) => val,
            None => {
                return self.empty_distribution(display_str);
            }
        };
        let sym_pre = sym_start.elapsed();

        let kind = match query_val.as_ref() {
            PluckValue::Value { constructor, args } if *constructor == self.syms.marginal_sym => {
                self.process_marginal_query(args, &mut state, sym_pre)
            }
            PluckValue::Value { constructor, args } if *constructor == self.syms.posterior_sym => {
                self.process_posterior_query(args, &mut state, sym_pre)
            }
            PluckValue::Value { constructor, args }
                if *constructor == self.syms.posterior_samples_sym =>
            {
                self.process_posterior_samples_query(args, &mut state)
            }
            PluckValue::Value { constructor, args } if *constructor == self.syms.gibbs_sym => {
                self.process_gibbs_query(args, &mut state, progress)
            }
            _ => {
                let val_str = format_value(&query_val, &self.interner, None);
                eprintln!(
                    "Expected Marginal / Posterior / PosteriorSamples query, got: {}",
                    val_str
                );
                ResultKind::Distribution { items: Vec::new() }
            }
        };

        QueryResult {
            query: display_str.to_string(),
            kind,
        }
    }

    fn process_marginal_query(
        &self,
        args: &[PluckVal],
        state: &mut LazyKCState,
        sym_pre: Duration,
    ) -> ResultKind {
        if args.is_empty() {
            return ResultKind::Distribution { items: Vec::new() };
        }
        let thunk = args[0].clone();
        let compile_start = std::time::Instant::now();
        let full = {
            let mut ctx = CompilerCtx {
                types: &self.types,
                defs: &self.defs,
                interner: &self.interner,
                true_sym: self.syms.true_sym,
                false_sym: self.syms.false_sym,
                state,
            };
            let mut mode = inference::lazy_kc::compile::CompileMode::KC {
                path_condition: ctx.state.bdd_true,
            };
            let (thunk_worlds, _) = ctx.evaluate_thunk(&thunk, &mut mode);
            ctx.infer_full_distribution(thunk_worlds)
        };
        let symbolic = sym_pre + compile_start.elapsed();
        self.run_distribution(full, state, symbolic)
    }

    fn build_given_expr_and_env(&mut self, args: &[PluckVal]) -> Option<(PExpr, Rc<EnvInner>)> {
        let given_sym = self.interner.intern("given");
        self.defs.lookup(given_sym)?;
        let a_sym = self.interner.intern("__posterior_a");
        let b_sym = self.interner.intern("__posterior_b");
        let env = env_cons(
            a_sym,
            args[0].clone(),
            env_cons(b_sym, args[1].clone(), empty_env()),
        );
        Some((
            PExpr::App {
                func: Box::new(PExpr::App {
                    func: Box::new(PExpr::Defined { name: given_sym }),
                    arg: Box::new(PExpr::Var { name: b_sym }),
                }),
                arg: Box::new(PExpr::Var { name: a_sym }),
            },
            env,
        ))
    }

    fn process_posterior_query(
        &mut self,
        args: &[PluckVal],
        state: &mut LazyKCState,
        sym_pre: Duration,
    ) -> ResultKind {
        if args.len() < 2 {
            return ResultKind::Distribution { items: Vec::new() };
        }
        let (given_expr, env) = match self.build_given_expr_and_env(args) {
            None => {
                eprintln!("'given' not defined in stdlib — cannot process Posterior query");
                return ResultKind::Distribution { items: Vec::new() };
            }
            Some((expr, env)) => (expr, env),
        };
        let compile_start = std::time::Instant::now();
        let full = {
            let mut ctx = CompilerCtx {
                types: &self.types,
                defs: &self.defs,
                interner: &self.interner,
                true_sym: self.syms.true_sym,
                false_sym: self.syms.false_sym,
                state,
            };
            let mut mode = inference::lazy_kc::compile::CompileMode::KC {
                path_condition: ctx.state.bdd_true,
            };
            let (ret_worlds, _) = ctx.traced_compile(&given_expr, &env, &mut mode, 0);
            ctx.infer_full_distribution(ret_worlds)
        };
        let symbolic = sym_pre + compile_start.elapsed();
        self.run_distribution(full, state, symbolic)
    }

    fn process_posterior_samples_query(
        &self,
        args: &[PluckVal],
        state: &mut LazyKCState,
    ) -> ResultKind {
        if args.len() < 3 {
            return ResultKind::Distribution { items: Vec::new() };
        }
        self.run_samples(args, state)
    }

    fn process_gibbs_query(
        &mut self,
        args: &[PluckVal],
        state: &mut LazyKCState,
        progress: Option<&GibbsProgress>,
    ) -> ResultKind {
        if args.len() < 5 {
            eprintln!(
                "Gibbs requires 5 args (query, cond, num_samples, blocks_list, initialization); got {}",
                args.len()
            );
            return ResultKind::Distribution { items: Vec::new() };
        }
        self.run_gibbs(args, state, progress)
    }

    fn empty_distribution(&self, display_str: &str) -> QueryResult {
        QueryResult {
            query: display_str.to_string(),
            kind: ResultKind::Distribution { items: Vec::new() },
        }
    }

    fn run_distribution(
        &self,
        worlds: Vec<(PluckVal, BddPtr<'static>)>,
        state: &LazyKCState,
        symbolic: Duration,
    ) -> ResultKind {
        let items = distribution_items(
            &worlds,
            state,
            &self.interner,
            &self.types,
            symbolic,
            self.parse_time,
        );
        ResultKind::Distribution { items }
    }

    fn run_samples(&self, args: &[PluckVal], state: &mut LazyKCState) -> ResultKind {
        let mut ctx = CompilerCtx {
            types: &self.types,
            defs: &self.defs,
            interner: &self.interner,
            true_sym: self.syms.true_sym,
            false_sym: self.syms.false_sym,
            state,
        };
        let samples = posterior_sample(args, &mut ctx, self.syms.o_sym, self.syms.s_sym);
        let values = samples
            .iter()
            .map(|s| format_value(s, &self.interner, None))
            .collect();
        ResultKind::Samples { values }
    }

    fn run_gibbs(
        &mut self,
        args: &[PluckVal],
        state: &mut LazyKCState,
        progress: Option<&GibbsProgress>,
    ) -> ResultKind {
        let symbols =
            inference::gibbs::GibbsCtx::generate_gibbs_symbols(&self.syms, &mut self.interner);
        let mut ctx = CompilerCtx {
            types: &self.types,
            defs: &self.defs,
            interner: &self.interner,
            true_sym: self.syms.true_sym,
            false_sym: self.syms.false_sym,
            state,
        };
        let outcome = match inference::gibbs::GibbsCtx::new(args, &mut ctx, symbols) {
            Ok(runner) => {
                let runner = match progress {
                    Some(p) => {
                        p.total.store(runner.num_samples(), Ordering::Relaxed);
                        p.current.store(0, Ordering::Relaxed);
                        runner.with_progress(p.current.clone())
                    }
                    None => runner,
                };
                runner.run(&mut ctx)
            }
            Err(msg) => {
                eprintln!("Gibbs: {msg}");
                inference::gibbs::GibbsOutcome::Stuck {
                    samples: Vec::new(),
                }
            }
        };
        let samples = match outcome {
            inference::gibbs::GibbsOutcome::Complete { samples } => samples,
            inference::gibbs::GibbsOutcome::Stuck { samples } => {
                eprintln!(
                    "Gibbs: chain stalled after {} samples (cond ∧ pin unsatisfiable)",
                    samples.len()
                );
                samples
            }
        };
        let values = samples
            .iter()
            .map(|s| format_value(s, &self.interner, None))
            .collect();
        ResultKind::Samples { values }
    }
}

impl Default for PluckContext {
    fn default() -> Self {
        Self::new()
    }
}


#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct QueryResult {
    pub query: String,
    pub kind: ResultKind,
}

#[derive(Debug, Clone)]
pub struct GibbsProgress {
    pub current: Arc<AtomicUsize>,
    pub total: Arc<AtomicUsize>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "result_kind", rename_all = "snake_case")]
pub enum ResultKind {
    Distribution { items: Vec<MixtureItem> },
    Samples { values: Vec<String> },
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MixtureItem {
    pub display: String,
    pub log_probability: f64,
    pub posteriors: PosteriorPacket,
}

pub type PosteriorPacket = PriorRegistry<String>;


pub type Path = Vec<(u32, usize)>;

#[derive(Default, Debug)]
pub struct ResultPositions {
    gauss_path_to_expr: BTreeMap<Path, GaussianAffineExpr>,
    beta_path_to_var: BTreeMap<Path, ContVarName>,
    dir_path_to_var: BTreeMap<Path, ContVarName>,
    gamma_path_to_var: BTreeMap<Path, (ContVarName, ordered_float::NotNan<f64>)>,
}

fn collect_positions(val: &PluckVal, path: &mut Path, positions: &mut ResultPositions) {
    use language::values::SymbolicPositionKind;
    if let Some(kind) = val.position_kind() {
        match kind {
            SymbolicPositionKind::Beta(name) => {
                positions.beta_path_to_var.insert(path.clone(), name);
            }
            SymbolicPositionKind::Dirichlet(name) => {
                positions.dir_path_to_var.insert(path.clone(), name);
            }
            SymbolicPositionKind::Gamma { name, scale } => {
                let s = ordered_float::NotNan::new(scale).expect("gamma scale must be finite");
                positions.gamma_path_to_var.insert(path.clone(), (name, s));
            }
            SymbolicPositionKind::Gaussian(expr) => {
                positions.gauss_path_to_expr.insert(path.clone(), expr);
            }
        }
        return;
    }
    if let PluckValue::Value { constructor, args } = val.as_ref() {
        for (i, arg) in args.iter().enumerate() {
            path.push((*constructor, i));
            collect_positions(arg, path, positions);
            path.pop();
        }
    }
    if let PluckValue::FloatMatrix { entries, .. } = val.as_ref() {
        for (i, e) in entries.iter().enumerate() {
            path.push((u32::MAX, i));
            collect_positions(e, path, positions);
            path.pop();
        }
    }
}

fn collect_variable_positions(worlds: &[(PluckVal, BddPtr<'static>)]) -> Vec<ResultPositions> {
    let mut positions_by_world = Vec::new();
    for (val, _) in worlds {
        let mut positions = ResultPositions::default();
        let mut path = Vec::new();
        collect_positions(val, &mut path, &mut positions);
        positions_by_world.push(positions);
    }
    positions_by_world
}

fn fill_in_missing_variables_with_prior(
    priors: &PriorRegistry,
    partial_posterior: &PriorRegistry,
    scope: &BTreeSet<ContVarName>,
) -> PriorRegistry {
    let posterior_variables: BTreeSet<ContVarName> = partial_posterior.scope().copied().collect();
    let unchanged_variables: BTreeSet<ContVarName> = scope
        .iter()
        .filter(|n| !posterior_variables.contains(n))
        .copied()
        .collect();
    partial_posterior.merge(&priors.slice(&unchanged_variables))
}

fn distribution_items(
    worlds: &[(PluckVal, BddPtr<'static>)],
    state: &LazyKCState,
    interner: &StringInterner,
    types: &TypeRegistry,
    symbolic: Duration,
    parse: Duration,
) -> Vec<MixtureItem> {
    let position_data = collect_variable_positions(worlds);
    let priors = &state.prior_registry;

    let mut stats = KcStats::new();
    stats.set_symbolic(symbolic);
    stats.set_parse(parse);

    let mut world_likelihoods: HashMap<(String, PosteriorPacket), RealEps> = HashMap::new();
    let mut order: Vec<(String, PosteriorPacket)> = Vec::new();

    for ((val, bdd), positions) in worlds.iter().zip(position_data.iter()) {
        let naming_t0 = std::time::Instant::now();
        let naming = NamingScheme::derive(positions, interner, types);
        let mut gamma_query_scales: BTreeMap<ContVarName, BTreeSet<ordered_float::NotNan<f64>>> =
            BTreeMap::new();
        for (var, scale) in positions.gamma_path_to_var.values() {
            gamma_query_scales.entry(*var).or_default().insert(*scale);
        }
        let naming_setup = naming_t0.elapsed();
        let scope = scope_of(val);
        let t0 = std::time::Instant::now();
        let evidence = state.wmc_spn(*bdd);
        let evidence = gamma_draw_refs(val).into_iter().fold(evidence, |spn, r| {
            spn * Spn::leaf(EvidenceLeaf::Gamma(r.vacuous_stat()))
        });
        let t1 = std::time::Instant::now();
        let posterior = evidence.posterior(priors, &scope);
        let t2 = std::time::Instant::now();
        let comps = posterior.components();
        let t3 = std::time::Instant::now();

        let mut n_components = 0usize;
        for (weight, partial_factored_component) in comps {
            if weight.is_zero() {
                continue;
            }
            n_components += 1;
            let full_factored_component =
                fill_in_missing_variables_with_prior(priors, &partial_factored_component, &scope);
            let packet = rename_to_affine_posterior_packet(
                &full_factored_component,
                &naming,
                &gamma_query_scales,
            );
            let display = format_value(val, interner, Some(&naming));
            let key = (display, packet);
            if !world_likelihoods.contains_key(&key) {
                order.push(key.clone());
            }
            let log_likelihood = world_likelihoods.entry(key).or_insert(RealEps::zero());
            *log_likelihood = *log_likelihood + weight;
        }
        let t4 = std::time::Instant::now();
        stats.record_world(
            *bdd,
            &evidence,
            StageTimes {
                wmc: t1 - t0,
                posterior: t2 - t1,
                flatten: t3 - t2,
                display: naming_setup + (t4 - t3),
            },
            n_components,
        );
    }

    stats.report(state);

    let z: RealEps = world_likelihoods
        .values()
        .copied()
        .fold(RealEps::zero(), |a, b| a + b);

    if z.is_zero() {
        return Vec::new();
    }

    let mut items: Vec<MixtureItem> = order
        .into_iter()
        .filter_map(|key| {
            let w = *world_likelihoods
                .get(&key)
                .unwrap_or_else(|| panic!("Could not find {:?}", key));
            let normed = w / z;
            if normed.is_zero() || normed.power != 0 {
                return None;
            }
            let (display, posteriors) = key;
            Some(MixtureItem {
                display,
                log_probability: normed.log_coeff,
                posteriors,
            })
        })
        .collect();

    items.sort_by(|a, b| {
        b.log_probability
            .partial_cmp(&a.log_probability)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.display.cmp(&b.display))
            .then_with(|| a.posteriors.to_string().cmp(&b.posteriors.to_string()))
    });

    items
}

fn scope_of(val: &PluckVal) -> BTreeSet<ContVarName> {
    let mut scope = BTreeSet::new();
    collect_scope(val, &mut scope);
    scope
}

fn collect_scope(val: &PluckVal, scope: &mut BTreeSet<ContVarName>) {
    val.scope_into(scope);
    if let PluckValue::Value { args, .. } = val.as_ref() {
        for arg in args {
            collect_scope(arg, scope);
        }
    }
    if let PluckValue::FloatMatrix { entries, .. } = val.as_ref() {
        for e in entries {
            collect_scope(e, scope);
        }
    }
}

#[derive(PartialEq, Eq, PartialOrd, Ord)]
enum GammaDrawRef {
    Poisson(ContVarName, ContVarName, ordered_float::NotNan<f64>),
    Exponential(ContVarName, ContVarName, ordered_float::NotNan<f64>),
}

impl GammaDrawRef {
    fn vacuous_stat(&self) -> GammaSuffStat {
        match self {
            GammaDrawRef::Poisson(gamma, draw, scale) => GammaSuffStat::poisson(
                *gamma,
                PoissonObs::constraint(*draw, IntervalOrEq::geq(0)).with_scale(*scale),
            ),
            GammaDrawRef::Exponential(gamma, draw, scale) => GammaSuffStat::exponential(
                *gamma,
                ExponentialObs::geq(*draw, 0.0).with_scale(*scale),
            ),
        }
    }
}

fn gamma_draw_refs(val: &PluckVal) -> BTreeSet<GammaDrawRef> {
    let mut out = BTreeSet::new();
    collect_gamma_draw_refs(val, &mut out);
    out
}

fn collect_gamma_draw_refs(val: &PluckVal, out: &mut BTreeSet<GammaDrawRef>) {
    match val.as_ref() {
        PluckValue::Poisson { name, gamma, scale } => {
            let s = ordered_float::NotNan::new(*scale).expect("gamma scale must be finite");
            out.insert(GammaDrawRef::Poisson(*gamma, *name, s));
        }
        PluckValue::Exponential { name, gamma, scale } => {
            let s = ordered_float::NotNan::new(*scale).expect("gamma scale must be finite");
            out.insert(GammaDrawRef::Exponential(*gamma, *name, s));
        }
        PluckValue::Value { args, .. } => {
            for arg in args {
                collect_gamma_draw_refs(arg, out);
            }
        }
        PluckValue::FloatMatrix { entries, .. } => {
            for e in entries {
                collect_gamma_draw_refs(e, out);
            }
        }
        _ => {}
    }
}


fn rename_to_affine_posterior_packet(
    registry: &PriorRegistry,
    naming: &NamingScheme,
    gamma_query_scales: &BTreeMap<ContVarName, BTreeSet<ordered_float::NotNan<f64>>>,
) -> PosteriorPacket {
    let beta = registry
        .beta
        .iter()
        .map(|b| {
            let name = b.name();
            let display = naming.var_names.get(&name).unwrap_or_else(|| {
                panic!(
                    "rename_to_affine_posterior_packet: per-world naming \
                     missing Beta variable {name}; this should have been \
                     populated by collect_positions for any var in this \
                     world's value tree"
                )
            });
            b.clone().with_display_name(display.clone())
        })
        .sorted_by_key(|b| b.name().clone())
        .collect();
    let dirichlet = registry
        .dirichlet
        .iter()
        .map(|d| {
            let display = naming.var_names.get(&d.name()).unwrap_or_else(|| {
                panic!(
                    "rename_to_affine_posterior_packet: per-world naming \
                     missing Dirichlet variable {}; this should have been \
                     populated by collect_positions for any var in this \
                     world's value tree",
                    d.name()
                )
            });
            d.clone().with_display_name(display.clone())
        })
        .sorted_by_key(|d| d.name().clone())
        .collect();
    let gamma = registry
        .gamma
        .iter()
        .flat_map(|g| {
            let rate_var = g.name();
            let bare_name = |name: ContVarName| {
                naming
                    .var_names
                    .get(&name)
                    .cloned()
                    .unwrap_or_else(|| format!("rate_{name}"))
            };
            let has_draws =
                !g.poisson.constraints().is_empty() || !g.exponential.constraints().is_empty();
            let mut scales: BTreeSet<ordered_float::NotNan<f64>> = gamma_query_scales
                .get(&rate_var)
                .cloned()
                .unwrap_or_default();
            if has_draws || scales.is_empty() {
                scales.insert(ordered_float::NotNan::new(1.0).unwrap());
            }
            scales.into_iter().map(move |s| {
                if s.into_inner() == 1.0 {
                    g.with_display_names(bare_name)
                } else {
                    let label = language::values::mk_gamma_scaled(rate_var, s.into_inner())
                        .render_symbolic(Some(naming))
                        .expect("scaled gamma renders to a name");
                    JointGammaPrior::from_gamma(g.gamma.clone().scaled(s.into_inner()))
                        .with_display_names(|_| label.clone())
                }
            })
        })
        .sorted_by_key(|g| g.name().clone())
        .collect();
    let gaussian = build_affine_gaussian_blocks(registry, naming);
    PriorRegistry {
        beta,
        dirichlet,
        gamma,
        gaussian,
    }
}

fn build_affine_gaussian_blocks(
    registry: &PriorRegistry,
    naming: &NamingScheme,
) -> Vec<GaussianPrior<String>> {
    let unique_exprs = dedupe_and_order_exprs(naming);
    let touched = compute_touched_blocks(&unique_exprs, registry);
    let components = group_exprs_by_shared_blocks(&touched);
    components
        .into_iter()
        .map(|comp| build_block_for_component(&comp, &unique_exprs, registry, naming))
        .collect()
}

fn dedupe_and_order_exprs(naming: &NamingScheme) -> Vec<&GaussianAffineExpr> {
    let mut exprs: Vec<&GaussianAffineExpr> = naming.expr_names.keys().collect();
    exprs.sort_by(|a, b| {
        let a_keys: Vec<_> = a.coefficients.keys().collect();
        let b_keys: Vec<_> = b.coefficients.keys().collect();
        a_keys
            .cmp(&b_keys)
            .then_with(|| {
                let a_vals: Vec<_> = a.coefficients.values().collect();
                let b_vals: Vec<_> = b.coefficients.values().collect();
                a_vals.cmp(&b_vals)
            })
            .then_with(|| a.constant.cmp(&b.constant))
    });
    exprs
}

fn compute_touched_blocks(
    unique_exprs: &[&GaussianAffineExpr],
    registry: &PriorRegistry,
) -> Vec<BTreeSet<usize>> {
    unique_exprs
        .iter()
        .map(|expr| {
            let touched: BTreeSet<usize> = (0..registry.gaussian.len())
                .filter(|&bi| {
                    expr.scope()
                        .any(|v| registry.gaussian[bi].var_order.contains(v))
                })
                .collect();
            assert!(
                expr.scope().all(|v| {
                    touched
                        .iter()
                        .any(|&bi| registry.gaussian[bi].var_order.contains(v))
                }),
                "queried affine expression touches a variable not in any registry Gaussian block",
            );
            touched
        })
        .collect()
}

fn group_exprs_by_shared_blocks(touched: &[BTreeSet<usize>]) -> Vec<Vec<usize>> {
    let n = touched.len();
    let mut uf = UnionFind::new(n);
    for i in 0..n {
        for j in (i + 1)..n {
            if !touched[i].is_disjoint(&touched[j]) {
                uf.union(i, j);
            }
        }
    }
    let mut components = uf.components();
    for c in components.iter_mut() {
        c.sort();
    }
    components.sort_by_key(|c| {
        c.iter()
            .flat_map(|&i| touched[i].iter().copied())
            .min()
            .unwrap_or(usize::MAX)
    });
    components
}

fn build_block_for_component(
    component: &[usize],
    unique_exprs: &[&GaussianAffineExpr],
    registry: &PriorRegistry,
    naming: &NamingScheme,
) -> GaussianPrior<String> {
    let scope: BTreeSet<ContVarName> = component
        .iter()
        .flat_map(|&i| unique_exprs[i].scope().copied())
        .collect();
    let block = registry.gaussian_prior_for(&scope);

    let k = component.len();
    let mut mean = na::DVector::<f64>::zeros(k);
    let mut cov = na::DMatrix::<f64>::zeros(k, k);
    let mut var_order: Vec<String> = Vec::with_capacity(k);

    let coeffs_f64: Vec<BTreeMap<ContVarName, f64>> = component
        .iter()
        .map(|&i| {
            unique_exprs[i]
                .coefficients
                .iter()
                .map(|(name, val)| (*name, val.into_inner()))
                .collect()
        })
        .collect();

    for (row, &expr_idx) in component.iter().enumerate() {
        let expr = unique_exprs[expr_idx];
        let constant = expr.constant.into_inner();
        let (m, v) = block.affine_moments(&coeffs_f64[row], constant);
        mean[row] = m;
        cov[(row, row)] = v;
        for col in 0..row {
            let c = block.affine_covariance(&coeffs_f64[row], &coeffs_f64[col]);
            cov[(row, col)] = c;
            cov[(col, row)] = c;
        }
        let display = naming.expr_names.get(expr).unwrap_or_else(|| {
            panic!(
                "build_block_for_component: per-world naming missing \
                 Gaussian affine expression; should have been populated \
                 by collect_positions for any expr in this world's value tree"
            )
        });
        var_order.push(display.clone());
    }

    GaussianPrior {
        var_order,
        mean,
        cov,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ordered_float::NotNan;

    fn affine(coeffs: &[(ContVarName, f64)], constant: f64) -> GaussianAffineExpr {
        GaussianAffineExpr {
            coefficients: coeffs
                .iter()
                .map(|&(k, v)| (k, NotNan::new(v).unwrap()))
                .collect(),
            constant: NotNan::new(constant).unwrap(),
        }
    }

    fn gauss_block(var_order: Vec<ContVarName>, mean: &[f64], cov: &[&[f64]]) -> GaussianPrior {
        let n = var_order.len();
        let m = na::DVector::from_iterator(n, mean.iter().copied());
        let c =
            na::DMatrix::from_row_iterator(n, n, cov.iter().flat_map(|row| row.iter().copied()));
        GaussianPrior {
            var_order,
            mean: m,
            cov: c,
        }
    }

    fn naming_with(
        vars: &[(ContVarName, &str)],
        exprs: &[(GaussianAffineExpr, &str)],
    ) -> NamingScheme {
        NamingScheme {
            var_names: vars.iter().map(|(n, s)| (*n, s.to_string())).collect(),
            expr_names: exprs
                .iter()
                .map(|(e, s)| (e.clone(), s.to_string()))
                .collect(),
        }
    }

    #[test]
    fn identity_affine_single_var() {
        let expr = affine(&[(1, 1.0)], 0.0);
        let registry = PriorRegistry::from_sorted(
            vec![],
            vec![],
            vec![],
            vec![gauss_block(vec![1], &[1.5], &[&[2.0]])],
        );
        let naming = naming_with(&[], &[(expr.clone(), "x")]);

        let packet = rename_to_affine_posterior_packet(&registry, &naming, &BTreeMap::new());
        assert_eq!(packet.gaussian.len(), 1);
        let g = &packet.gaussian[0];
        assert_eq!(g.var_order, vec!["x".to_string()]);
        assert!((g.mean[0] - 1.5).abs() < 1e-12);
        assert!((g.cov[(0, 0)] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn two_independent_exprs_two_blocks() {
        let ex_x = affine(&[(1, 1.0)], 0.0);
        let ex_y = affine(&[(2, 1.0)], 0.0);
        let registry = PriorRegistry::from_sorted(
            vec![],
            vec![],
            vec![],
            vec![
                gauss_block(vec![1], &[1.0], &[&[1.0]]),
                gauss_block(vec![2], &[2.0], &[&[3.0]]),
            ],
        );
        let naming = naming_with(&[], &[(ex_x.clone(), "x"), (ex_y.clone(), "y")]);

        let packet = rename_to_affine_posterior_packet(&registry, &naming, &BTreeMap::new());
        assert_eq!(packet.gaussian.len(), 2);
        assert_eq!(packet.gaussian[0].var_order, vec!["x".to_string()]);
        assert!((packet.gaussian[0].mean[0] - 1.0).abs() < 1e-12);
        assert!((packet.gaussian[0].cov[(0, 0)] - 1.0).abs() < 1e-12);
        assert_eq!(packet.gaussian[1].var_order, vec!["y".to_string()]);
        assert!((packet.gaussian[1].mean[0] - 2.0).abs() < 1e-12);
        assert!((packet.gaussian[1].cov[(0, 0)] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn expr_spans_two_independent_blocks() {
        let ex = affine(&[(1, 1.0), (2, 1.0)], 0.0);
        let registry = PriorRegistry::from_sorted(
            vec![],
            vec![],
            vec![],
            vec![
                gauss_block(vec![1], &[5.0], &[&[2.0]]),
                gauss_block(vec![2], &[7.0], &[&[3.0]]),
            ],
        );
        let naming = naming_with(&[], &[(ex.clone(), "z")]);

        let packet = rename_to_affine_posterior_packet(&registry, &naming, &BTreeMap::new());
        assert_eq!(packet.gaussian.len(), 1);
        let g = &packet.gaussian[0];
        assert_eq!(g.var_order, vec!["z".to_string()]);
        assert!((g.mean[0] - 12.0).abs() < 1e-12);
        assert!((g.cov[(0, 0)] - 5.0).abs() < 1e-12);
    }

    #[test]
    fn dedupe_same_expr_at_two_paths() {
        let expr = affine(&[(1, 1.0)], 0.0);
        let registry = PriorRegistry::from_sorted(
            vec![],
            vec![],
            vec![],
            vec![gauss_block(vec![1], &[0.0], &[&[1.0]])],
        );
        let naming = naming_with(&[], &[(expr.clone(), "x")]);

        let packet = rename_to_affine_posterior_packet(&registry, &naming, &BTreeMap::new());
        assert_eq!(packet.gaussian.len(), 1);
        assert_eq!(packet.gaussian[0].var_order.len(), 1);
        assert_eq!(packet.gaussian[0].var_order[0], "x");
    }
}
