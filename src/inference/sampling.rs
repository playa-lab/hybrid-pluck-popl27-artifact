
use std::collections::{BTreeSet, HashMap};
use std::rc::Rc;

use rand::Rng;
use rsdd::builder::BottomUpBuilder;
use rsdd::repr::{BddPtr, DDNNFPtr, VarLabel};

use super::conjugate_pairs::exponential::ExponentialPrior;
use super::conjugate_pairs::poisson::PoissonPrior;
use super::conjugate_pairs::{Assignment, ContVarName, PriorRegistry};
use super::lazy_kc::compile::CompileMode;
use super::lazy_kc::ctx::CompilerCtx;
use super::lazy_kc::state::{support_vars, CallstackKey, LazyKCState, World};
use crate::inference::lazy_kc::state::GuardedWorlds;
use crate::language::types::Symbol;
use crate::language::values::*;

pub use super::conjugate_pairs::gamma::GammaDrawFamily;

impl GammaDrawFamily {
    pub fn mk_value(self, gamma: ContVarName, draw: ContVarName, scale: f64) -> PluckVal {
        match self {
            GammaDrawFamily::Poisson => mk_poisson(gamma, draw, scale),
            GammaDrawFamily::Exponential => mk_exponential(gamma, draw, scale),
        }
    }
}

pub struct SampleState {
    pub constraint: BddPtr<'static>,
    pub assignment: Assignment,
    pub trace: HashMap<CallstackKey, bool>,
    pub memo: HashMap<usize, (PluckVal, PluckVal)>,
    pub lazy: bool,
    pub true_sym: Symbol,
    pub false_sym: Symbol,
}

impl SampleState {
    pub fn ensure_continuous(
        &mut self,
        names: impl IntoIterator<Item = ContVarName>,
        registry: &PriorRegistry,
    ) {
        let scope = self.assignment.scope();
        let missing: BTreeSet<ContVarName> =
            names.into_iter().filter(|n| !scope.contains(n)).collect();
        if missing.is_empty() {
            return;
        }
        let drawn = registry.slice(&missing).sample(&mut rand::thread_rng());
        self.assignment = self.assignment.merge(&drawn);
    }

    pub fn ensure_gamma_draw(
        &mut self,
        rate: ContVarName,
        draw: ContVarName,
        family: GammaDrawFamily,
        scale: f64,
    ) {
        if self.assignment.scope().contains(&draw) {
            return;
        }
        let lambda = self.assignment.gamma_value(rate)
            * ordered_float::NotNan::new(scale).expect("gamma scale must be finite");
        let rng = &mut rand::thread_rng();
        let sampled = match family {
            GammaDrawFamily::Poisson => {
                PoissonPrior::unconstrained(std::iter::once(draw)).sample(lambda, rng)
            }
            GammaDrawFamily::Exponential => {
                ExponentialPrior::unconstrained(std::iter::once(draw)).sample(lambda, rng)
            }
        };
        let single = Assignment::from_sorted(vec![], vec![], sampled.into_iter().collect(), vec![]);
        self.assignment = self.assignment.merge(&single);
    }

    pub fn check_cache(&self, val: &PluckVal) -> Option<GuardedWorlds> {
        if let Some((_, result)) = self.memo.get(&(Rc::as_ptr(val) as usize)) {
            Some((
                vec![(result.clone(), BddPtr::true_ptr())],
                BddPtr::true_ptr(),
            ))
        } else {
            None
        }
    }

    pub fn set_cache(&mut self, val: &PluckVal, worlds: &GuardedWorlds) {
        if let Some((world_val, _)) = worlds.0.first() {
            self.memo
                .insert(Rc::as_ptr(val) as usize, (val.clone(), world_val.clone()));
        }
    }
}

pub fn posterior_sample(
    args: &[PluckVal],
    ctx: &mut CompilerCtx,
    o_sym: Symbol,
    s_sym: Symbol,
) -> Vec<PluckVal> {
    assert!(
        args.len() >= 3,
        "PosteriorSamples requires 3 args: query, evidence, num_samples"
    );
    let query_thunk = &args[0];
    let evidence_thunk = &args[1];
    let n_thunk = &args[2];

    let evidence_results = {
        let mut kc_mode = CompileMode::KC {
            path_condition: ctx.state.bdd_true,
        };
        ctx.evaluate_thunk(evidence_thunk, &mut kc_mode).0
    };

    let n_samples = force_n_samples(n_thunk, o_sym, s_sym, ctx);

    let true_sym = ctx.true_sym;
    let evidence_bdd = evidence_results
        .iter()
        .find_map(|(result, bdd)| match result.as_ref() {
            PluckValue::Value { constructor, args }
                if *constructor == true_sym && args.is_empty() && !bdd.is_false() =>
            {
                Some(*bdd)
            }
            _ => None,
        });
    let evidence_bdd = match evidence_bdd {
        Some(bdd) => bdd,
        None => {
            eprintln!("PosteriorSamples: evidence has zero probability; cannot sample.");
            return Vec::new();
        }
    };

    let evidence_spn = ctx.state.wmc_spn(evidence_bdd);
    let mut samples = Vec::new();
    let mut rng = rand::thread_rng();
    for _ in 0..n_samples {
        let mut assignment = ctx
            .state
            .sample_continuous_posterior(&evidence_spn, &mut rng);
        ctx.state
            .cover_support_gamma_draws(evidence_bdd, &mut assignment, &mut rng);
        let sampled_bdd = ctx
            .state
            .sample_discrete_given(evidence_bdd, &assignment, &mut rng);
        if sampled_bdd.is_false() {
            eprintln!("PosteriorSamples: sampled BDD is False; skipping sample.");
            continue;
        }

        let mut state = SampleState {
            constraint: sampled_bdd,
            assignment,
            trace: HashMap::new(),
            memo: HashMap::new(),
            lazy: true,
            true_sym: ctx.true_sym,
            false_sym: ctx.false_sym,
        };
        let forced = force_value(query_thunk, &mut state, ctx);
        samples.push(forced);
    }

    samples
}

pub fn resolve_flip(
    p: f64,
    key: CallstackKey,
    state: &mut SampleState,
    kc_state: &LazyKCState,
) -> bool {
    if let Some(&hit) = state.trace.get(&key) {
        return hit;
    }
    if let Some(forced) = kc_state.constraint_forces(state.constraint, key.0, key.1) {
        state.trace.insert(key, forced);
        return forced;
    }
    let outcome = rand::thread_rng().gen::<f64>() < p;
    state.trace.insert(key, outcome);
    let (_, var_bdd) = kc_state
        .var_of_callstack
        .get(&key)
        .expect("resolve_flip: flip variable not registered before resolution");
    let builder = kc_state.bdd();
    let lit = if outcome {
        *var_bdd
    } else {
        builder.negate(*var_bdd)
    };
    state.constraint = builder.and(state.constraint, lit);
    outcome
}

pub fn resolve_observation(state: &mut SampleState, kc_state: &LazyKCState) -> bool {
    let hash = kc_state.callstack.hash();
    if let Some(forced) = kc_state.constraint_forces_observation(state.constraint, hash) {
        return forced;
    }
    if let Some((_, var_bdd)) = kc_state.observation_vars.get(&hash) {
        let builder = kc_state.bdd();
        state.constraint = builder.and(state.constraint, builder.negate(*var_bdd));
    }
    false
}

fn extend_constraint_over(
    bdds: &[BddPtr<'static>],
    state: &mut SampleState,
    kc_state: &LazyKCState,
) {
    let mut support: BTreeSet<usize> = BTreeSet::new();
    for bdd in bdds {
        support_vars(*bdd, &mut support);
    }
    kc_state.realize_missing_gamma_draws(
        support.iter().copied(),
        &mut state.assignment,
        &mut rand::thread_rng(),
    );
    let builder = kc_state.bdd();
    for label in support {
        let var_bdd = builder.var(VarLabel::new(label as u64), true);
        if kc_state.implies(state.constraint, var_bdd).is_some() {
            continue;
        }
        let polarity = if kc_state.observation_var_labels.contains(&label) {
            false
        } else {
            let p = kc_state.var_sample_prob(label, &state.assignment);
            rand::thread_rng().gen::<f64>() < p
        };
        let lit = if polarity {
            var_bdd
        } else {
            builder.negate(var_bdd)
        };
        state.constraint = builder.and(state.constraint, lit);
    }
}

pub(crate) fn select_world(
    worlds: Vec<World>,
    state: &mut SampleState,
    kc_state: &LazyKCState,
) -> PluckVal {
    let builder = kc_state.bdd();
    let mut consistent: Vec<World> = worlds
        .into_iter()
        .filter(|(_, guard)| !builder.and(state.constraint, *guard).is_false())
        .collect();
    if consistent.len() > 1 {
        let guards: Vec<BddPtr<'static>> = consistent.iter().map(|(_, g)| *g).collect();
        extend_constraint_over(&guards, state, kc_state);
        consistent.retain(|(_, guard)| !builder.and(state.constraint, *guard).is_false());
    }
    match consistent.len() {
        1 => consistent.pop().unwrap().0,
        0 => panic!(
            "Sample mode: no world consistent with the sampled constraint \
             (program error along the sampled path)"
        ),
        n => panic!(
            "select_world: {} worlds remain consistent after pinning every \
             guard variable — world guards are not disjoint",
            n
        ),
    }
}

pub fn force_value(val: &PluckVal, state: &mut SampleState, ctx: &mut CompilerCtx) -> PluckVal {
    match val.as_ref() {
        PluckValue::Thunk(_) | PluckValue::ThunkUnion(_) => {
            let evaluated = {
                let mut mode = CompileMode::Sample(state);
                let (worlds, _) = ctx.evaluate_thunk(val, &mut mode);
                if worlds.is_empty() {
                    if ctx.state.hit_limit {
                        panic!("Sample mode: inference limit (max_depth / time_limit) hit");
                    }
                    panic!(
                        "Sample mode: thunk evaluation produced no worlds \
                         (program error along the sampled path)"
                    );
                }
                assert_eq!(
                    worlds.len(),
                    1,
                    "Sample mode: thunk evaluation returned {} worlds; \
                     expected exactly one",
                    worlds.len()
                );
                worlds.into_iter().next().unwrap().0
            };
            force_value(&evaluated, state, ctx)
        }
        PluckValue::Value { constructor, args } => {
            let forced_args: Vec<PluckVal> = args
                .iter()
                .map(|arg| force_value(arg, state, ctx))
                .collect();
            mk_value(*constructor, forced_args)
        }
        PluckValue::FloatMatrix { entries, shape } => {
            let forced_entries: Vec<PluckVal> =
                entries.iter().map(|e| force_value(e, state, ctx)).collect();
            crate::language::values::mk_float_matrix(forced_entries, shape.clone())
        }
        PluckValue::IntDist { bits } => {
            assert!(
                bits.iter().all(|b| b.is_true() || b.is_false()),
                "force_value: IntDist with symbolic bits in Sample mode"
            );
            val.clone()
        }
        v if v.is_symbolic() => {
            match v {
                PluckValue::Poisson { name, gamma, scale } => {
                    state.ensure_gamma_draw(*gamma, *name, GammaDrawFamily::Poisson, *scale)
                }
                PluckValue::Exponential { name, gamma, scale } => {
                    state.ensure_gamma_draw(*gamma, *name, GammaDrawFamily::Exponential, *scale)
                }
                _ => {}
            }
            v.force_symbolic(&state.assignment, state.true_sym)
                .expect("is_symbolic / force_symbolic disagreement")
        }
        _ => val.clone(),
    }
}

pub fn force_n_samples(
    n_thunk: &Rc<PluckValue>,
    o_sym: Symbol,
    s_sym: Symbol,
    ctx: &mut CompilerCtx,
) -> usize {
    let mut state = SampleState {
        constraint: ctx.state.bdd_true,
        assignment: Assignment::from_sorted(vec![], vec![], vec![], vec![]),
        trace: HashMap::new(),
        memo: HashMap::new(),
        lazy: false,
        true_sym: ctx.true_sym,
        false_sym: ctx.false_sym,
    };
    let n_val = force_value(n_thunk, &mut state, ctx);
    extract_nat(&n_val, o_sym, s_sym).unwrap_or_else(|| {
        panic!(
            "PosteriorSamples: third argument must be an integer, got {:?}",
            n_val
        )
    }) as usize
}
