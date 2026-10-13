
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use rand::rngs::ThreadRng;
use rsdd::builder::BottomUpBuilder;
use rsdd::repr::{BddPtr, DDNNFPtr};

use super::conjugate_pairs::Assignment;
use super::lazy_kc::compile::CompileMode;
use super::lazy_kc::ctx::CompilerCtx;
use super::sampling::{force_value, SampleState};
use crate::inference::sampling::force_n_samples;
use crate::language::pexpr::{CaseOfGuard, PExpr};
use crate::language::types::{StringInterner, Symbol};
use crate::language::values::{empty_env, env_cons, EnvInner, PluckVal, PluckValue};
use crate::{inference, Syms};

const N_INIT_RETRIES: usize = 10;

pub enum GibbsOutcome {
    Complete { samples: Vec<PluckVal> },
    Stuck { samples: Vec<PluckVal> },
}

pub struct GibbsSymbols {
    o_sym: Symbol,
    s_sym: Symbol,
    nil_sym: Symbol,
    cons_sym: Symbol,
    with_prior_sample_sym: Symbol,
    with_constant_sym: Symbol,
    block_sym: Symbol,
    value_sym: Symbol,
    evidence_sym: Symbol,
}

pub struct GibbsCtx {
    query_thunk: PluckVal,
    evidence_thunk: PluckVal,
    num_samples: usize,
    blocks: Vec<PluckVal>,
    num_blocks: usize,
    out_samples: Vec<PluckVal>,
    prev_block_value: PluckVal,
    rng: ThreadRng,
    progress: Option<Arc<AtomicUsize>>,
    symbols: GibbsSymbols,
}

impl GibbsCtx {
    pub(crate) fn generate_gibbs_symbols(
        symbols: &Syms,
        interner: &mut StringInterner,
    ) -> GibbsSymbols {
        let block_sym = interner.intern("__gibbs_block");
        let value_sym = interner.intern("__gibbs_value");
        let evidence_sym = interner.intern("__gibbs_evidence");

        GibbsSymbols {
            o_sym: symbols.o_sym,
            s_sym: symbols.s_sym,
            nil_sym: symbols.nil_sym,
            cons_sym: symbols.cons_sym,
            with_prior_sample_sym: symbols.with_prior_sample_sym,
            with_constant_sym: symbols.with_constant_sym,
            block_sym,
            value_sym,
            evidence_sym,
        }
    }

    pub fn new(
        args: &[PluckVal],
        ctx: &mut CompilerCtx,
        symbols: GibbsSymbols,
    ) -> Result<Self, String> {
        assert!(
            args.len() >= 5,
            "Gibbs requires 5 args: query, cond, num_samples, blocks_list, initialization"
        );
        let query_thunk = args[0].clone();
        let evidence_thunk = args[1].clone();
        let n_thunk = &args[2];
        let blocks_thunk = &args[3];
        let init_arg = &args[4];

        let num_samples = force_n_samples(n_thunk, symbols.o_sym, symbols.s_sym, ctx);

        let blocks = match force_cons_list(blocks_thunk, ctx, symbols.nil_sym, symbols.cons_sym) {
            Ok(b) if b.is_empty() => {
                return Err("requires at least one block expression".to_string());
            }
            Ok(b) => b,
            Err(msg) => {
                return Err(msg);
            }
        };
        let num_blocks = blocks.len();

        let mut rng = rand::thread_rng();
        let prev_block_value = seed_initial_block(
            init_arg,
            &symbols,
            &blocks[0],
            &evidence_thunk,
            &mut rng,
            ctx,
        )?;

        Ok(GibbsCtx {
            query_thunk,
            evidence_thunk,
            num_samples,
            blocks,
            num_blocks,
            out_samples: Vec::with_capacity(num_samples),
            prev_block_value,
            rng,
            progress: None,
            symbols,
        })
    }

    pub fn with_progress(mut self, counter: Arc<AtomicUsize>) -> Self {
        self.progress = Some(counter);
        self
    }

    pub fn num_samples(&self) -> usize {
        self.num_samples
    }

    pub fn run(mut self, ctx: &mut CompilerCtx) -> GibbsOutcome {
        let mut cur = 0usize;
        for _ in 0..self.num_samples {
            let next = (cur + 1) % self.num_blocks;

            let cur_thunk = self.blocks[cur].clone();
            let prev_val = self.prev_block_value.clone();

            let conditioning_bdd = build_conditioning_bdd(
                &cur_thunk,
                &prev_val,
                &self.evidence_thunk,
                &self.symbols,
                ctx,
            );
            if conditioning_bdd.is_false() {
                return GibbsOutcome::Stuck {
                    samples: self.out_samples,
                };
            }

            let (sampled_bdd, assignment) = match sample_world(conditioning_bdd, ctx, &mut self.rng)
            {
                Some(pair) => pair,
                None => {
                    return GibbsOutcome::Stuck {
                        samples: self.out_samples,
                    };
                }
            };

            let next_thunk = self.blocks[next].clone();
            let (block_val, query_val) = self.force_step(&next_thunk, sampled_bdd, assignment, ctx);

            self.out_samples.push(query_val);
            if let Some(p) = &self.progress {
                p.store(self.out_samples.len(), Ordering::Relaxed);
            }
            self.prev_block_value = block_val;
            cur = next;
        }

        GibbsOutcome::Complete {
            samples: self.out_samples,
        }
    }

    fn force_step(
        &self,
        block_thunk: &PluckVal,
        sampled_bdd: BddPtr<'static>,
        assignment: Assignment,
        ctx: &mut CompilerCtx,
    ) -> (PluckVal, PluckVal) {
        let mut state = SampleState {
            constraint: sampled_bdd,
            assignment,
            trace: HashMap::new(),
            memo: HashMap::new(),
            lazy: true,
            true_sym: ctx.true_sym,
            false_sym: ctx.false_sym,
        };
        let block_val = force_value(block_thunk, &mut state, ctx);
        let query_val = force_value(&self.query_thunk, &mut state, ctx);
        (block_val, query_val)
    }
}

fn build_conditioning_expr(
    block_thunk: &PluckVal,
    block_value: &PluckVal,
    evidence_thunk: &PluckVal,
    symbols: &GibbsSymbols,
    ctx: &mut CompilerCtx,
) -> (PExpr, Rc<EnvInner>) {
    let mut env = empty_env();
    env = env_cons(symbols.block_sym, block_thunk.clone(), env);
    env = env_cons(symbols.value_sym, block_value.clone(), env);
    env = env_cons(symbols.evidence_sym, evidence_thunk.clone(), env);

    let pin_expr = PExpr::Pin {
        expr: Box::new(PExpr::Var {
            name: symbols.block_sym,
        }),
        val: Box::new(PExpr::Var {
            name: symbols.value_sym,
        }),
    };

    let evidence_expr = PExpr::CaseOf {
        guards: vec![CaseOfGuard {
            constructor: ctx.true_sym,
            args: vec![],
        }],
        scrutinee: Box::new(PExpr::Var {
            name: symbols.evidence_sym,
        }),
        branches: vec![PExpr::Construct {
            constructor: ctx.true_sym,
            args: vec![],
        }],
    };

    (
        PExpr::CaseOf {
            guards: vec![CaseOfGuard {
                constructor: ctx.true_sym,
                args: vec![],
            }],
            scrutinee: Box::new(pin_expr),
            branches: vec![evidence_expr],
        },
        env,
    )
}

fn build_conditioning_bdd(
    block_thunk: &PluckVal,
    block_value: &PluckVal,
    evidence_thunk: &PluckVal,
    symbols: &GibbsSymbols,
    ctx: &mut CompilerCtx,
) -> BddPtr<'static> {
    let (conditioning_expr, env) =
        build_conditioning_expr(block_thunk, block_value, evidence_thunk, symbols, ctx);
    let mut mode = inference::lazy_kc::compile::CompileMode::KC {
        path_condition: ctx.state.bdd_true,
    };
    let (worlds, _) = ctx.traced_compile(&conditioning_expr, &env, &mut mode, 0);
    or_true_worlds(worlds, ctx)
}

fn or_true_worlds(
    worlds: Vec<(PluckVal, BddPtr<'static>)>,
    ctx: &mut CompilerCtx,
) -> BddPtr<'static> {
    let builder = ctx.state.bdd();
    let mut acc = BddPtr::false_ptr();
    for (result, bdd) in worlds {
        assert!(
            if let PluckValue::Value { constructor, args } = result.as_ref() {
                *constructor == ctx.true_sym && args.is_empty()
            } else {
                false
            },
            "Found non-true result value in condition_expr result"
        );
        acc = builder.or(acc, bdd);
    }
    acc
}

fn sample_world(
    evidence_bdd: BddPtr<'static>,
    ctx: &mut CompilerCtx,
    rng: &mut ThreadRng,
) -> Option<(BddPtr<'static>, Assignment)> {
    let evidence_spn = ctx.state.wmc_spn(evidence_bdd);
    let mut assignment = ctx.state.sample_continuous_posterior(&evidence_spn, rng);
    ctx.state
        .cover_support_gamma_draws(evidence_bdd, &mut assignment, rng);
    let candidate = ctx
        .state
        .sample_discrete_given(evidence_bdd, &assignment, rng);
    if !candidate.is_false() {
        return Some((candidate, assignment));
    }
    None
}

fn force_one(
    thunk: &PluckVal,
    constraint: BddPtr<'static>,
    assignment: Assignment,
    ctx: &mut CompilerCtx,
) -> PluckVal {
    let mut state = SampleState {
        constraint,
        assignment,
        trace: HashMap::new(),
        memo: HashMap::new(),
        lazy: true,
        true_sym: ctx.true_sym,
        false_sym: ctx.false_sym,
    };
    force_value(thunk, &mut state, ctx)
}

fn sample_expr_from_prior(expr: &PluckVal, rng: &mut ThreadRng, ctx: &mut CompilerCtx) -> PluckVal {
    {
        let mut kc_mode = CompileMode::KC {
            path_condition: ctx.state.bdd_true,
        };
        let _ = ctx.evaluate_thunk(expr, &mut kc_mode);
    }
    let bdd_true = ctx.state.bdd_true;
    let prior_spn = ctx.state.wmc_spn(bdd_true);
    let assignment = ctx.state.sample_continuous_posterior(&prior_spn, rng);
    force_one(expr, bdd_true, assignment, ctx)
}

fn seed_initial_block(
    init_arg: &PluckVal,
    symbols: &GibbsSymbols,
    block_thunk: &PluckVal,
    evidence_thunk: &PluckVal,
    rng: &mut ThreadRng,
    ctx: &mut CompilerCtx,
) -> Result<PluckVal, String> {
    let (ctor, inner) = force_constructor(init_arg, ctx)?;
    if ctor == symbols.with_constant_sym && inner.len() == 1 {
        seed_via_constant(&inner[0], block_thunk, evidence_thunk, symbols, rng, ctx)
    } else if ctor == symbols.with_prior_sample_sym && inner.len() == 1 {
        seed_via_prior_sample(&inner[0], block_thunk, evidence_thunk, symbols, rng, ctx)
    } else {
        Err("initialization must be (WithPriorSample expr) or (WithConstant expr).".to_string())
    }
}

fn seed_via_constant(
    init_expr: &PluckVal,
    block_expr: &PluckVal,
    evidence_thunk: &PluckVal,
    symbols: &GibbsSymbols,
    rng: &mut ThreadRng,
    ctx: &mut CompilerCtx,
) -> Result<PluckVal, String> {
    let const_val = sample_expr_from_prior(init_expr, rng, ctx);
    let condition_bdd =
        build_conditioning_bdd(block_expr, &const_val, evidence_thunk, symbols, ctx);
    if condition_bdd.is_false() {
        Err("WithConstant value is inconsistent with the evidence \
                    (zero probability)."
            .to_string())
    } else {
        Ok(const_val)
    }
}

fn seed_via_prior_sample(
    init_expr: &PluckVal,
    block_thunk: &PluckVal,
    evidence_thunk: &PluckVal,
    symbols: &GibbsSymbols,
    rng: &mut ThreadRng,
    ctx: &mut CompilerCtx,
) -> Result<PluckVal, String> {
    for _ in 0..N_INIT_RETRIES {
        let init_val = sample_expr_from_prior(init_expr, rng, ctx);
        let condition_bdd =
            build_conditioning_bdd(init_expr, &init_val, evidence_thunk, symbols, ctx);
        if condition_bdd.is_false() {
            continue;
        }
        if let Some((sampled_bdd, assignment)) = sample_world(condition_bdd, ctx, rng) {
            return Ok(force_one(block_thunk, sampled_bdd, assignment, ctx));
        }
    }
    Err(format!(
        "WithPriorSample drew {} off-support init values; the initialization \
         expression has no overlap with the conditioning evidence. Choose an \
         init expression with full support under the evidence (e.g. `x` when \
         the constraint is `x + y = 1`).",
        N_INIT_RETRIES
    ))
}

fn force_constructor(
    init_arg: &PluckVal,
    ctx: &mut CompilerCtx,
) -> Result<(Symbol, Vec<PluckVal>), String> {
    let mut kc_mode = CompileMode::KC {
        path_condition: ctx.state.bdd_true,
    };
    let (worlds, _) = ctx.evaluate_thunk(init_arg, &mut kc_mode);
    let live = worlds.into_iter().find(|(_, bdd)| !bdd.is_false());
    match live {
        Some((val, _)) => match val.as_ref() {
            PluckValue::Value { constructor, args } => Ok((*constructor, args.clone())),
            _ => Err("initialization must be a constructor value \
                      (WithPriorSample or WithConstant)."
                .to_string()),
        },
        None => Err("initialization evaluated to no live worlds.".to_string()),
    }
}

fn force_cons_list(
    list_thunk: &PluckVal,
    ctx: &mut CompilerCtx,
    nil_sym: Symbol,
    cons_sym: Symbol,
) -> Result<Vec<PluckVal>, String> {
    let mut out: Vec<PluckVal> = Vec::new();
    let mut current = list_thunk.clone();
    loop {
        let mut mode = CompileMode::KC {
            path_condition: ctx.state.bdd_true,
        };
        let (worlds, _) = ctx.evaluate_thunk(&current, &mut mode);
        let live = worlds.into_iter().find(|(_, bdd)| !bdd.is_false());
        let val = match live {
            Some((v, _)) => v,
            None => {
                return Err("blocks list evaluated to no live worlds".to_string());
            }
        };
        match val.as_ref() {
            PluckValue::Value { constructor, args } if *constructor == nil_sym => {
                return Ok(out);
            }
            PluckValue::Value { constructor, args } if *constructor == cons_sym => {
                if args.len() != 2 {
                    return Err(format!(
                        "Cons in blocks list has {} args, expected 2",
                        args.len()
                    ));
                }
                out.push(args[0].clone());
                current = args[1].clone();
            }
            _ => {
                return Err(
                    "blocks list must be a Cons/Nil chain, got constructor not Nil/Cons".into(),
                );
            }
        }
    }
}
