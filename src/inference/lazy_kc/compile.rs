use std::rc::Rc;

use rsdd::builder::BottomUpBuilder;
use rsdd::repr::{BddPtr, DDNNFPtr};

use super::ctx::CompilerCtx;
use super::monad::*;
use super::state::{FlipWeight, GuardedWorlds, LazyKCState, World};
use crate::inference::conjugate_pairs::{
    BetaFlip, BetaPrior, DirichletFlip, DirichletPrior, GammaFlip, GammaPrior, GaussianFlip,
    GaussianPrior,
};
use crate::inference::int_dists::{
    enumerate_int_dist, int_dist_add, int_dist_eq, int_dist_lt, int_dist_sub, IntDist,
};
use crate::inference::sampling::{GammaDrawFamily, SampleState};
use crate::language::pexpr::{CaseOfGuard, NativeVal, PExpr};
use crate::language::types::Symbol;
use crate::language::values::*;
use crate::utils::formatting::format_value;
use crate::utils::intervals::{Interval, IntervalOrEq};
use ordered_float::NotNan;

pub enum CompileMode<'a> {
    KC { path_condition: BddPtr<'static> },
    Sample(&'a mut SampleState),
}

impl<'a> CompileMode<'a> {
    pub fn path_condition(&self) -> BddPtr<'static> {
        match self {
            CompileMode::KC { path_condition } => *path_condition,
            CompileMode::Sample(_) => BddPtr::true_ptr(),
        }
    }
}

#[derive(Clone, Copy)]
enum FloatOp {
    Add,
    Sub,
    Mul,
    Div,
}

#[derive(Clone, Copy)]
enum RealCmp {
    Lt,
    Geq,
}

#[derive(Clone, Copy)]
enum NativeCmp {
    Leq,
    Geq,
}

impl NativeCmp {
    fn mirror(self) -> Self {
        match self {
            NativeCmp::Leq => NativeCmp::Geq,
            NativeCmp::Geq => NativeCmp::Leq,
        }
    }
}

enum CatKind {
    Dirichlet { name: u64, categories: usize },
    Constants(Vec<f64>),
}

impl<'a> CompilerCtx<'a> {
    fn bool_val(&self, b: bool) -> PluckVal {
        mk_value(if b { self.true_sym } else { self.false_sym }, vec![])
    }

    fn constant_flip_shortcut(&self, p: f64) -> Option<PluckVal> {
        if (p - 0.0).abs() < 1e-15 {
            return Some(self.bool_val(false));
        }
        if (p - 1.0).abs() < 1e-15 {
            return Some(self.bool_val(true));
        }
        None
    }

    pub fn traced_compile(
        &mut self,
        expr: &PExpr,
        env: &Env,
        mode: &mut CompileMode<'_>,
        strict_order_index: i32,
    ) -> GuardedWorlds {
        if mode.path_condition().is_false() {
            self.state.stats.num_false_pc_exits += 1;
            return false_path_condition_worlds();
        }

        if self.state.hit_limit {
            return inference_error_worlds();
        }

        if let Some(max_depth) = self.state.cfg.max_depth {
            if self.state.depth > max_depth {
                self.state.hit_limit = true;
                return inference_error_worlds();
            }
        }

        if self.state.check_time_limit() {
            self.state.hit_limit = true;
            return inference_error_worlds();
        }

        self.state.depth += 1;
        self.state.stats.num_compile_calls += 1;
        self.state.callstack.push(strict_order_index);

        let result = self.compile(expr, env, mode);

        self.state.callstack.pop();
        self.state.depth -= 1;

        if self.state.hit_limit {
            return inference_error_worlds();
        }
        result
    }

    fn compile(&mut self, expr: &PExpr, env: &Env, mode: &mut CompileMode<'_>) -> GuardedWorlds {
        let pc = mode.path_condition();
        match expr {
            PExpr::Abs { var, body } => self.compile_abs(*var, body, env, pc),
            PExpr::Var { name } => self.compile_var(*name, env, mode),
            PExpr::Defined { name } => self.compile_defined(*name, mode),
            PExpr::App { func, arg } => self.compile_app(func, arg, env, mode),
            PExpr::Construct { constructor, args } => {
                self.compile_construct(*constructor, args, env, pc)
            }
            PExpr::CaseOf {
                guards,
                scrutinee,
                branches,
            } => self.compile_case_of(scrutinee, guards, branches, env, mode),
            PExpr::YComb { func } => self.compile_ycomb(func, env, mode),
            PExpr::Flip { prob } => self.compile_flip(prob, env, mode),
            PExpr::ConstNative { val } => self.compile_const_native(val, pc),
            PExpr::NativeEq { a, b } => self.compile_native_eq(a, b, env, mode),
            PExpr::NativeLeq { a, b } => self.compile_native_cmp(NativeCmp::Leq, a, b, env, mode),
            PExpr::NativeGeq { a, b } => self.compile_native_cmp(NativeCmp::Geq, a, b, env, mode),
            PExpr::GetArgs { expr: inner } => self.compile_get_args(inner, env, mode),
            PExpr::GetConstructor { expr: inner } => self.compile_get_constructor(inner, env, mode),
            PExpr::Print { expr: inner } => self.compile_print(inner, env, mode),
            PExpr::Error { .. } => program_error_worlds(),

            PExpr::FAdd { a, b } => {
                self.compile_variable_aware_binop(FloatOp::Add, a, b, env, mode)
            }
            PExpr::FSub { a, b } => {
                self.compile_variable_aware_binop(FloatOp::Sub, a, b, env, mode)
            }
            PExpr::FMul { a, b } => {
                self.compile_variable_aware_binop(FloatOp::Mul, a, b, env, mode)
            }
            PExpr::FDiv { a, b } => {
                self.compile_variable_aware_binop(FloatOp::Div, a, b, env, mode)
            }

            PExpr::MkInt { bitwidth, value } => self.compile_mk_int(bitwidth, value, env, mode),
            PExpr::UniformInt { bitwidth } => self.compile_uniform_int(bitwidth, env, mode),
            PExpr::UniformIntRange { bitwidth, lo, hi } => {
                self.compile_uniform_int_range(bitwidth, lo, hi, env, mode)
            }
            PExpr::IntDistEq { a, b } => self.compile_int_dist_eq(a, b, env, mode),
            PExpr::IntDistAdd { a, b } => self.compile_int_dist_add(a, b, env, mode),
            PExpr::IntDistSub { a, b } => self.compile_int_dist_sub(a, b, env, mode),
            PExpr::IntDistLt { a, b } => self.compile_int_dist_lt(a, b, env, mode),

            PExpr::GSymbol { name } => pure_monad(mk_native_symbol(*name), pc),
            PExpr::GVarSymbol { type_name } => pure_monad(mk_native_symbol(*type_name), pc),
            PExpr::GetConfig => pure_monad(mk_native_bool(false), pc),

            PExpr::Beta { a, b } => self.compile_beta(a, b, env, mode),
            PExpr::ProbEq { prob, val } => self.compile_prob_close(prob, val, env, mode),
            PExpr::Gaussian { mu, sigma } => self.compile_gaussian(mu, sigma, env, mode),
            PExpr::RealEq { expr, val } => self.compile_close(expr, val, env, mode),
            PExpr::RealLt { expr, val } => self.compile_real_cmp(RealCmp::Lt, expr, val, env, mode),
            PExpr::RealGeq { expr, val } => {
                self.compile_real_cmp(RealCmp::Geq, expr, val, env, mode)
            }
            PExpr::Dirichlet { alphas } => self.compile_dirichlet(alphas, env, mode),
            PExpr::DirichletEq { expr: dexpr, val } => {
                self.compile_dirichlet_eq(dexpr, val, env, mode)
            }
            PExpr::Index { v, idx } => self.compile_index(v, idx, env, mode),
            PExpr::Categorical { probs } => self.compile_categorical(probs, env, mode),
            PExpr::FloatMatrix { entries, shape } => {
                self.compile_float_matrix(entries, shape, env, mode)
            }
            PExpr::MatMul { a, b } => self.compile_matmul(a, b, env, mode),
            PExpr::Sum { expr: inner } => self.compile_sum(inner, env, mode),
            PExpr::Pin { expr, val } => self.compile_pin(expr, val, env, mode),
            PExpr::Gamma { shape, rate } => self.compile_gamma(shape, rate, env, mode),
            PExpr::Poisson { rate } => {
                self.compile_gamma_consumer(rate, env, mode, GammaDrawFamily::Poisson)
            }
            PExpr::Exponential { rate } => {
                self.compile_gamma_consumer(rate, env, mode, GammaDrawFamily::Exponential)
            }
        }
    }

    fn compile_abs(
        &self,
        var: Symbol,
        body: &PExpr,
        env: &Env,
        pc: BddPtr<'static>,
    ) -> GuardedWorlds {
        let closure = mk_closure(var, body.clone().into(), env.clone());
        pure_monad(closure, pc)
    }

    fn compile_var(
        &mut self,
        name: Symbol,
        env: &Env,
        mode: &mut CompileMode<'_>,
    ) -> GuardedWorlds {
        match env_lookup(env, name) {
            Some(val) => match val.as_ref() {
                PluckValue::Thunk(_) | PluckValue::ThunkUnion(_) => self.evaluate_thunk(&val, mode),
                _ => pure_monad(val, mode.path_condition()),
            },
            None => {
                eprintln!("variable not found: {:?}", name);
                program_error_worlds()
            }
        }
    }

    fn compile_defined(&mut self, name: Symbol, mode: &mut CompileMode<'_>) -> GuardedWorlds {
        match self.defs.lookup(name) {
            Some(def) => {
                let expr = def.expr.clone();
                let env = empty_env();
                self.traced_compile(&expr, &env, mode, 0)
            }
            None => {
                eprintln!("undefined: {:?}", name);
                program_error_worlds()
            }
        }
    }

    fn compile_construct(
        &mut self,
        constructor: Symbol,
        args: &[PExpr],
        env: &Env,
        pc: BddPtr<'static>,
    ) -> GuardedWorlds {
        let thunked_args: Vec<PluckVal> = args
            .iter()
            .enumerate()
            .map(|(i, arg)| make_thunk(arg, env.clone(), i as i32, self.state))
            .collect();

        let val = mk_value(constructor, thunked_args);
        pure_monad(val, pc)
    }

    fn compile_app(
        &mut self,
        func: &PExpr,
        arg: &PExpr,
        env: &Env,
        mode: &mut CompileMode<'_>,
    ) -> GuardedWorlds {
        let pc = mode.path_condition();
        let arg_thunk = make_thunk(arg, env.clone(), 1, self.state);
        let func_worlds = self.traced_compile(func, env, mode, 0);

        bind_compile(
            func_worlds,
            pc,
            self,
            mode,
            |func_val, _pc, ctx, mode| match func_val.as_ref() {
                PluckValue::Closure(closure_data) => {
                    let mut new_env = env_cons(
                        closure_data.var,
                        arg_thunk.clone(),
                        closure_data.env.clone(),
                    );
                    if let Some(rec_name) = closure_data.rec_name {
                        if let Some(self_ref) = closure_data.self_ref.borrow().as_ref() {
                            new_env = env_cons(rec_name, self_ref.clone(), new_env);
                        }
                    }
                    ctx.traced_compile(&closure_data.body, &new_env, mode, 2)
                }
                _ => {
                    eprintln!("tried to apply a non-function value");
                    program_error_worlds()
                }
            },
        )
    }

    fn compile_case_of(
        &mut self,
        scrutinee: &PExpr,
        guards: &[CaseOfGuard],
        branches: &[PExpr],
        env: &Env,
        mode: &mut CompileMode<'_>,
    ) -> GuardedWorlds {
        let pc = mode.path_condition();
        let in_sample_mode = matches!(mode, CompileMode::Sample(_));
        let scrutinee_worlds = self.traced_compile(scrutinee, env, mode, 0);

        bind_compile(
            scrutinee_worlds,
            pc,
            self,
            mode,
            |scrutinee_val, _pc, ctx, mode| {
                match scrutinee_val.as_ref() {
                    PluckValue::Value { constructor, args } => {
                        for (i, guard) in guards.iter().enumerate() {
                            if guard.constructor == *constructor {
                                let mut new_env = env.clone();
                                for (j, arg_name) in guard.args.iter().enumerate() {
                                    if j < args.len() {
                                        new_env = env_cons(*arg_name, args[j].clone(), new_env);
                                    }
                                }
                                return ctx.traced_compile(
                                    &branches[i],
                                    &new_env,
                                    mode,
                                    (i + 1) as i32,
                                );
                            }
                        }
                        if in_sample_mode {
                            eprintln!("incomplete case/match encountered in sampling mode")
                        }
                        program_error_worlds()
                    }
                    _ => {
                        eprintln!("case/match on non-value: {:?}", scrutinee_val);
                        program_error_worlds()
                    }
                }
            },
        )
    }

    fn compile_ycomb(
        &mut self,
        func: &PExpr,
        env: &Env,
        mode: &mut CompileMode<'_>,
    ) -> GuardedWorlds {
        let pc = mode.path_condition();
        let func_worlds = self.traced_compile(func, env, mode, 0);

        bind_compile(
            func_worlds,
            pc,
            self,
            mode,
            |func_val, pc, _ctx, _mode| match func_val.as_ref() {
                PluckValue::Closure(outer_closure) => {
                    let rec_name = outer_closure.var;
                    match outer_closure.body.as_ref() {
                        PExpr::Abs {
                            var: arg_name,
                            body,
                        } => {
                            let closure = mk_y_closure(
                                body.clone().into(),
                                outer_closure.env.clone(),
                                rec_name,
                                *arg_name,
                            );
                            pure_monad(closure, pc)
                        }
                        _ => {
                            eprintln!("Y combinator expects a function returning a function");
                            program_error_worlds()
                        }
                    }
                }
                _ => {
                    eprintln!("Y combinator applied to non-function");
                    program_error_worlds()
                }
            },
        )
    }

    fn compile_flip(
        &mut self,
        prob: &PExpr,
        env: &Env,
        mode: &mut CompileMode<'_>,
    ) -> GuardedWorlds {
        if matches!(mode, CompileMode::Sample(_)) {
            return self.sample_compile_flip(prob, env, mode);
        }

        let pc = mode.path_condition();
        let prob_worlds = self.traced_compile(prob, env, mode, 0);

        bind_compile(prob_worlds, pc, self, mode, |prob_val, pc, ctx, _mode| {
            let flip_weight = flip_weight_from(prob_val.as_ref());

            if let FlipWeight::Constant(p) = &flip_weight {
                if let Some(v) = ctx.constant_flip_shortcut(*p) {
                    return pure_monad(v, pc);
                }
            }

            let flip_var = ctx
                .state
                .with_callstack_index(1, |s| s.current_address_weighted(flip_weight));

            let true_val = ctx.bool_val(true);
            let false_val = ctx.bool_val(false);
            if_then_else_monad(true_val, false_val, flip_var, pc, ctx)
        })
    }

    fn compile_const_native(&self, val: &NativeVal, pc: BddPtr<'static>) -> GuardedWorlds {
        let pluck_val = match val {
            NativeVal::Float(f) => mk_native_float(*f),
            NativeVal::Int(i) => mk_native_int(*i),
            NativeVal::Symbol(s) => mk_native_symbol(*s),
            NativeVal::Bool(b) => mk_native_bool(*b),
        };
        pure_monad(pluck_val, pc)
    }

    fn compile_native_eq(
        &mut self,
        a: &PExpr,
        b: &PExpr,
        env: &Env,
        mode: &mut CompileMode<'_>,
    ) -> GuardedWorlds {
        let pc = mode.path_condition();
        let a_worlds = self.traced_compile(a, env, mode, 0);

        bind_compile(a_worlds, pc, self, mode, |a_val, pc, ctx, mode| {
            let b_worlds = ctx.traced_compile(b, env, mode, 1);
            bind_compile(b_worlds, pc, ctx, mode, |b_val, pc, ctx, mode| {
                let poisson = match (a_val.as_ref(), b_val.as_ref()) {
                    (PluckValue::Poisson { name, gamma, scale }, _) => {
                        Some((*name, *gamma, *scale, b_val.clone()))
                    }
                    (_, PluckValue::Poisson { name, gamma, scale }) => {
                        Some((*name, *gamma, *scale, a_val.clone()))
                    }
                    _ => None,
                };
                if let Some((draw, gamma, scale, other)) = poisson {
                    let (k, k_used) = ctx.force_int(&other, mode).unwrap_or_else(|| {
                        panic!(
                            "native_eq: a Poisson draw must be compared to a \
                             native int or nat literal, got {:?}",
                            other
                        )
                    });
                    if k < 0 {
                        return pure_monad_used(ctx.bool_val(false), k_used);
                    }
                    let flip = GammaFlip::PoissonEvent {
                        gamma,
                        draw,
                        event: IntervalOrEq::eq(k as u64),
                        scale,
                    };
                    if matches!(mode, CompileMode::Sample(_)) {
                        return ctx.sample_gamma_flip_outcome(
                            flip,
                            GammaDrawFamily::Poisson,
                            gamma,
                            draw,
                            scale,
                            mode,
                        );
                    }
                    let (worlds, used) = ctx.gamma_flip_worlds(flip, pc);
                    return (worlds, ctx.state.bdd().and(used, k_used));
                }
                let eq = match (a_val.as_ref(), b_val.as_ref()) {
                    (PluckValue::Native(a), PluckValue::Native(b)) => a == b,
                    _ => false,
                };
                pure_monad(ctx.bool_val(eq), pc)
            })
        })
    }

    fn compile_native_cmp(
        &mut self,
        cmp: NativeCmp,
        a: &PExpr,
        b: &PExpr,
        env: &Env,
        mode: &mut CompileMode<'_>,
    ) -> GuardedWorlds {
        let pc = mode.path_condition();
        let a_worlds = self.traced_compile(a, env, mode, 0);

        bind_compile(a_worlds, pc, self, mode, move |a_val, pc, ctx, mode| {
            let b_worlds = ctx.traced_compile(b, env, mode, 1);
            bind_compile(b_worlds, pc, ctx, mode, move |b_val, pc, ctx, mode| {
                let op_name = match cmp {
                    NativeCmp::Leq => "native_leq",
                    NativeCmp::Geq => "native_geq",
                };
                let poisson = match (a_val.as_ref(), b_val.as_ref()) {
                    (PluckValue::Poisson { name, gamma, scale }, _) => {
                        Some((*name, *gamma, *scale, b_val.clone(), cmp))
                    }
                    (_, PluckValue::Poisson { name, gamma, scale }) => {
                        Some((*name, *gamma, *scale, a_val.clone(), cmp.mirror()))
                    }
                    _ => None,
                };
                if let Some((draw, gamma, scale, other, cmp)) = poisson {
                    let (n, n_used) = ctx.force_int(&other, mode).unwrap_or_else(|| {
                        panic!(
                            "{op_name}: a Poisson draw must be compared to a \
                             native int or nat literal, got {:?}",
                            other
                        )
                    });
                    match cmp {
                        NativeCmp::Leq if n < 0 => {
                            return pure_monad_used(ctx.bool_val(false), n_used)
                        }
                        NativeCmp::Geq if n <= 0 => {
                            return pure_monad_used(ctx.bool_val(true), n_used)
                        }
                        _ => {}
                    }
                    let event = match cmp {
                        NativeCmp::Leq => IntervalOrEq::lt(n as u64 + 1),
                        NativeCmp::Geq => IntervalOrEq::geq(n as u64),
                    };
                    let flip = GammaFlip::PoissonEvent {
                        gamma,
                        draw,
                        event,
                        scale,
                    };
                    if matches!(mode, CompileMode::Sample(_)) {
                        return ctx.sample_gamma_flip_outcome(
                            flip,
                            GammaDrawFamily::Poisson,
                            gamma,
                            draw,
                            scale,
                            mode,
                        );
                    }
                    let (worlds, used) = ctx.gamma_flip_worlds(flip, pc);
                    return (worlds, ctx.state.bdd().and(used, n_used));
                }
                let ka = ctx.force_int(&a_val, mode);
                let kb = ctx.force_int(&b_val, mode);
                match (ka, kb) {
                    (Some((x, x_used)), Some((y, y_used))) => {
                        let holds = match cmp {
                            NativeCmp::Leq => x <= y,
                            NativeCmp::Geq => x >= y,
                        };
                        let used = ctx.state.bdd().and(x_used, y_used);
                        pure_monad_used(ctx.bool_val(holds), used)
                    }
                    _ => panic!(
                        "{op_name}: expected native ints / nat literals or a \
                         Poisson draw, got {:?} and {:?}",
                        a_val, b_val
                    ),
                }
            })
        })
    }

    fn force_int(
        &mut self,
        val: &PluckVal,
        mode: &mut CompileMode<'_>,
    ) -> Option<(i64, BddPtr<'static>)> {
        let (o_sym, s_sym) = self
            .types
            .find_nat_constructors()
            .expect("force_int: nat type not registered");
        let pc = mode.path_condition();
        let mut used = BddPtr::true_ptr();
        let mut count: i64 = 0;
        let mut current = val.clone();
        loop {
            match current.as_ref() {
                PluckValue::Native(NativeVal::Int(i)) => return Some((i + count, used)),
                PluckValue::Value { constructor, args }
                    if *constructor == o_sym && args.is_empty() =>
                {
                    return Some((count, used))
                }
                PluckValue::Value { constructor, args }
                    if *constructor == s_sym && args.len() == 1 =>
                {
                    count += 1;
                    let (mut worlds, w_used) = self.evaluate_thunk(&args[0], mode);
                    let builder = self.state.bdd();
                    worlds.retain(|w| !builder.and(w.1, pc).is_false());
                    assert!(
                        !worlds.is_empty(),
                        "force_int: nat spine produced no live worlds under \
                         the current path condition (program error along this path)"
                    );
                    assert_eq!(
                        worlds.len(),
                        1,
                        "force_int: stochastic nat spine — integer comparands \
                         must be deterministic"
                    );
                    let (v, guard) = worlds.pop().unwrap();
                    used = builder.and(builder.and(used, w_used), guard);
                    current = v;
                }
                _ => return None,
            }
        }
    }

    fn gamma_flip_worlds(&mut self, flip: GammaFlip, pc: BddPtr<'static>) -> GuardedWorlds {
        let obs_var = self
            .state
            .with_callstack_index(2, |s| s.current_address_weighted(FlipWeight::Gamma(flip)));
        let true_val = self.bool_val(true);
        let false_val = self.bool_val(false);
        if_then_else_monad(true_val, false_val, obs_var, pc, self)
    }

    fn sample_gamma_flip_outcome(
        &mut self,
        flip: GammaFlip,
        family: GammaDrawFamily,
        rate: u64,
        draw: u64,
        scale: f64,
        mode: &mut CompileMode<'_>,
    ) -> GuardedWorlds {
        let CompileMode::Sample(sample_state) = mode else {
            unreachable!("sample_gamma_flip_outcome called outside Sample mode")
        };
        sample_state.ensure_gamma_draw(rate, draw, family, scale);
        let weight = FlipWeight::Gamma(flip);
        let p = weight
            .sample_probability(&sample_state.assignment)
            .expect("gamma interval flips have an indicator probability");
        let discriminator = weight.discriminator();
        let key: super::state::CallstackKey = self.state.with_callstack_index(2, |s| {
            let k = (s.callstack.hash(), discriminator);
            s.current_address_weighted(weight);
            k
        });
        let outcome = crate::inference::sampling::resolve_flip(p, key, sample_state, &*self.state);
        let val = self.bool_val(outcome);
        pure_monad(val, BddPtr::true_ptr())
    }

    fn compile_get_args(
        &mut self,
        inner: &PExpr,
        env: &Env,
        mode: &mut CompileMode<'_>,
    ) -> GuardedWorlds {
        let pc = mode.path_condition();
        let worlds = self.traced_compile(inner, env, mode, 0);
        let (nil_s, cons_s) = self
            .types
            .find_list_constructors()
            .expect("list type not registered");

        bind_compile(
            worlds,
            pc,
            self,
            mode,
            move |val, pc, _ctx, _mode| match val.as_ref() {
                PluckValue::Value { args, .. } => {
                    let mut list = mk_value(nil_s, vec![]);
                    for arg in args.iter().rev() {
                        list = mk_value(cons_s, vec![arg.clone(), list]);
                    }
                    pure_monad(list, pc)
                }
                _ => pure_monad(val, pc),
            },
        )
    }

    fn compile_get_constructor(
        &mut self,
        inner: &PExpr,
        env: &Env,
        mode: &mut CompileMode<'_>,
    ) -> GuardedWorlds {
        let pc = mode.path_condition();
        let worlds = self.traced_compile(inner, env, mode, 0);

        bind_compile(worlds, pc, self, mode, |val, pc, _ctx, _mode| {
            match val.as_ref() {
                PluckValue::Value { constructor, .. } => {
                    let sym_val = mk_native_symbol(*constructor);
                    pure_monad(sym_val, pc)
                }
                _ => pure_monad(val, pc),
            }
        })
    }

    /// `(print e)` evaluates to `e`, and writes the values `e` can take on the current
    /// path to stderr, separated by `|`. Parts not yet evaluated print as `<thunk>`.
    fn compile_print(
        &mut self,
        inner: &PExpr,
        env: &Env,
        mode: &mut CompileMode<'_>,
    ) -> GuardedWorlds {
        let worlds = self.traced_compile(inner, env, mode, 0);
        let mut shown: Vec<String> = Vec::new();
        for (val, _) in &worlds.0 {
            let s = format_value(val, self.interner, None);
            if !shown.contains(&s) {
                shown.push(s);
            }
        }
        if shown.is_empty() {
            eprintln!("print: no value");
        } else {
            eprintln!("print: {}", shown.join(" | "));
        }
        worlds
    }

    fn compile_mk_int(
        &mut self,
        bitwidth: &PExpr,
        value: &PExpr,
        env: &Env,
        mode: &mut CompileMode<'_>,
    ) -> GuardedWorlds {
        let pc = mode.path_condition();
        let bw_worlds = self.traced_compile(bitwidth, env, mode, 0);

        bind_compile(bw_worlds, pc, self, mode, |bw_val, pc, ctx, mode| {
            let val_worlds = ctx.traced_compile(value, env, mode, 1);
            bind_compile(val_worlds, pc, ctx, mode, |v_val, pc, _ctx, _mode| {
                let width = as_int_or_panic(bw_val.as_ref(), "mk_int bitwidth") as usize;
                let v = as_int_or_panic(v_val.as_ref(), "mk_int value") as u64;

                let bits: Vec<BddPtr<'static>> = (0..width)
                    .map(|i| {
                        if (v >> i) & 1 == 1 {
                            BddPtr::true_ptr()
                        } else {
                            BddPtr::false_ptr()
                        }
                    })
                    .collect();

                pure_monad(mk_int_dist(bits), pc)
            })
        })
    }

    fn compile_uniform_int(
        &mut self,
        bitwidth: &PExpr,
        env: &Env,
        mode: &mut CompileMode<'_>,
    ) -> GuardedWorlds {
        if matches!(mode, CompileMode::Sample(_)) {
            return self.sample_compile_uniform_int(bitwidth, env, mode);
        }

        let pc = mode.path_condition();
        let bw_worlds = self.traced_compile(bitwidth, env, mode, 0);

        bind_compile(bw_worlds, pc, self, mode, |bw_val, pc, ctx, _mode| {
            let width = as_int_or_panic(bw_val.as_ref(), "uniform_int bitwidth") as usize;

            let bits: Vec<BddPtr<'static>> = (0..width)
                .map(|i| {
                    ctx.state
                        .with_callstack_index(i as i32, |s| s.current_address(0.5))
                })
                .collect();

            pure_monad(mk_int_dist(bits), pc)
        })
    }

    fn sample_compile_uniform_int(
        &mut self,
        bitwidth: &PExpr,
        env: &Env,
        mode: &mut CompileMode<'_>,
    ) -> GuardedWorlds {
        let bw_val = self.sample_compile_one(bitwidth, env, mode, 0, "uniform_int: bitwidth");
        let width = as_int_or_panic(bw_val.as_ref(), "uniform_int bitwidth") as usize;

        let CompileMode::Sample(sample_state) = mode else {
            unreachable!()
        };
        let mut bits: Vec<BddPtr<'static>> = Vec::with_capacity(width);
        for i in 0..width {
            let key = self.state.with_callstack_index(i as i32, |s| {
                let k = (s.callstack.hash(), 0.5_f64.to_bits());
                s.current_address(0.5);
                k
            });
            let bit =
                crate::inference::sampling::resolve_flip(0.5, key, sample_state, &*self.state);
            bits.push(if bit {
                BddPtr::true_ptr()
            } else {
                BddPtr::false_ptr()
            });
        }
        pure_monad(mk_int_dist(bits), BddPtr::true_ptr())
    }

    fn compile_uniform_int_range(
        &mut self,
        bitwidth: &PExpr,
        lo: &PExpr,
        hi: &PExpr,
        env: &Env,
        mode: &mut CompileMode<'_>,
    ) -> GuardedWorlds {
        if matches!(mode, CompileMode::Sample(_)) {
            return self.sample_compile_uniform_int_range(bitwidth, lo, hi, env, mode);
        }

        let pc = mode.path_condition();
        let bw_worlds = self.traced_compile(bitwidth, env, mode, 0);

        bind_compile(bw_worlds, pc, self, mode, |bw_val, pc, ctx, mode| {
            let lo_worlds = ctx.traced_compile(lo, env, mode, 1);
            bind_compile(lo_worlds, pc, ctx, mode, |lo_val, pc, ctx, mode| {
                let hi_worlds = ctx.traced_compile(hi, env, mode, 2);
                bind_compile(hi_worlds, pc, ctx, mode, |hi_val, pc, ctx, _mode| {
                    let width =
                        as_int_or_panic(bw_val.as_ref(), "uniform_int_range bitwidth") as usize;
                    let start = as_int_or_panic(lo_val.as_ref(), "uniform_int_range lo");
                    let stop = as_int_or_panic(hi_val.as_ref(), "uniform_int_range hi");

                    assert!(stop >= start, "uniform_int_range: hi must be >= lo");

                    let mut bits: Vec<BddPtr<'static>> = vec![BddPtr::false_ptr(); width];

                    encode_range(
                        start,
                        stop,
                        BddPtr::true_ptr(),
                        &mut bits,
                        width,
                        ctx.state,
                        0,
                    );

                    pure_monad(mk_int_dist(bits), pc)
                })
            })
        })
    }

    fn sample_compile_uniform_int_range(
        &mut self,
        bitwidth: &PExpr,
        lo: &PExpr,
        hi: &PExpr,
        env: &Env,
        mode: &mut CompileMode<'_>,
    ) -> GuardedWorlds {
        let bw_val = self.sample_compile_one(bitwidth, env, mode, 0, "uniform_int_range: bitwidth");
        let lo_val = self.sample_compile_one(lo, env, mode, 1, "uniform_int_range: lo");
        let hi_val = self.sample_compile_one(hi, env, mode, 2, "uniform_int_range: hi");

        let width = as_int_or_panic(bw_val.as_ref(), "uniform_int_range bitwidth") as usize;
        let start = as_int_or_panic(lo_val.as_ref(), "uniform_int_range lo");
        let stop = as_int_or_panic(hi_val.as_ref(), "uniform_int_range hi");
        assert!(stop >= start, "uniform_int_range: hi must be >= lo");

        let CompileMode::Sample(sample_state) = mode else {
            unreachable!()
        };
        let mut lo_cur = start;
        let mut hi_cur = stop;
        let mut depth_idx: i32 = 0;
        while lo_cur != hi_cur {
            let mid = lo_cur + (hi_cur - lo_cur) / 2;
            let lower_size = (mid - lo_cur + 1) as f64;
            let upper_size = (hi_cur - mid) as f64;
            let p = lower_size / (lower_size + upper_size);

            let key = self.state.with_callstack_index(depth_idx, |s| {
                let k = (s.callstack.hash(), p.to_bits());
                s.current_address(p);
                k
            });
            let take_lower =
                crate::inference::sampling::resolve_flip(p, key, sample_state, &*self.state);
            if take_lower {
                hi_cur = mid;
            } else {
                lo_cur = mid + 1;
            }
            depth_idx += 1;
        }

        let bits: Vec<BddPtr<'static>> = (0..width)
            .map(|i| {
                if ((lo_cur as u64) >> i) & 1 == 1 {
                    BddPtr::true_ptr()
                } else {
                    BddPtr::false_ptr()
                }
            })
            .collect();
        pure_monad(mk_int_dist(bits), BddPtr::true_ptr())
    }

    fn compile_int_dist_eq(
        &mut self,
        a: &PExpr,
        b: &PExpr,
        env: &Env,
        mode: &mut CompileMode<'_>,
    ) -> GuardedWorlds {
        let pc = mode.path_condition();
        let a_worlds = self.traced_compile(a, env, mode, 0);

        bind_compile(a_worlds, pc, self, mode, |a_val, pc, ctx, mode| {
            let b_worlds = ctx.traced_compile(b, env, mode, 1);
            bind_compile(b_worlds, pc, ctx, mode, |b_val, pc, ctx, _mode| {
                let a_bits = as_int_dist_bits(a_val.as_ref(), "int_dist_eq: first argument");
                let b_bits = as_int_dist_bits(b_val.as_ref(), "int_dist_eq: second argument");

                let a_int = IntDist::new(a_bits.to_vec());
                let b_int = IntDist::new(b_bits.to_vec());

                let eq_bdd = int_dist_eq(&a_int, &b_int, ctx.state.bdd());

                let true_val = ctx.bool_val(true);
                let false_val = ctx.bool_val(false);
                if_then_else_monad(true_val, false_val, eq_bdd, pc, ctx)
            })
        })
    }

    fn compile_int_dist_add(
        &mut self,
        a: &PExpr,
        b: &PExpr,
        env: &Env,
        mode: &mut CompileMode<'_>,
    ) -> GuardedWorlds {
        let pc = mode.path_condition();
        let a_worlds = self.traced_compile(a, env, mode, 0);

        bind_compile(a_worlds, pc, self, mode, |a_val, pc, ctx, mode| {
            let b_worlds = ctx.traced_compile(b, env, mode, 1);
            bind_compile(b_worlds, pc, ctx, mode, |b_val, pc, ctx, _mode| {
                let a_bits = as_int_dist_bits(a_val.as_ref(), "int_add: first argument");
                let b_bits = as_int_dist_bits(b_val.as_ref(), "int_add: second argument");

                let a_int = IntDist::new(a_bits.to_vec());
                let b_int = IntDist::new(b_bits.to_vec());
                let result = int_dist_add(&a_int, &b_int, ctx.state.bdd());
                pure_monad(mk_int_dist(result.bits), pc)
            })
        })
    }

    fn compile_int_dist_sub(
        &mut self,
        a: &PExpr,
        b: &PExpr,
        env: &Env,
        mode: &mut CompileMode<'_>,
    ) -> GuardedWorlds {
        let pc = mode.path_condition();
        let a_worlds = self.traced_compile(a, env, mode, 0);

        bind_compile(a_worlds, pc, self, mode, |a_val, pc, ctx, mode| {
            let b_worlds = ctx.traced_compile(b, env, mode, 1);
            bind_compile(b_worlds, pc, ctx, mode, |b_val, pc, ctx, _mode| {
                let a_bits = as_int_dist_bits(a_val.as_ref(), "int_sub: first argument");
                let b_bits = as_int_dist_bits(b_val.as_ref(), "int_sub: second argument");

                let a_int = IntDist::new(a_bits.to_vec());
                let b_int = IntDist::new(b_bits.to_vec());
                let result = int_dist_sub(&a_int, &b_int, ctx.state.bdd());
                pure_monad(mk_int_dist(result.bits), pc)
            })
        })
    }

    fn compile_int_dist_lt(
        &mut self,
        a: &PExpr,
        b: &PExpr,
        env: &Env,
        mode: &mut CompileMode<'_>,
    ) -> GuardedWorlds {
        let pc = mode.path_condition();
        let a_worlds = self.traced_compile(a, env, mode, 0);

        bind_compile(a_worlds, pc, self, mode, |a_val, pc, ctx, mode| {
            let b_worlds = ctx.traced_compile(b, env, mode, 1);
            bind_compile(b_worlds, pc, ctx, mode, |b_val, pc, ctx, _mode| {
                let a_bits = as_int_dist_bits(a_val.as_ref(), "int_lt: first argument");
                let b_bits = as_int_dist_bits(b_val.as_ref(), "int_lt: second argument");

                let a_int = IntDist::new(a_bits.to_vec());
                let b_int = IntDist::new(b_bits.to_vec());
                let lt_bdd = int_dist_lt(&a_int, &b_int, ctx.state.bdd());

                let true_val = ctx.bool_val(true);
                let false_val = ctx.bool_val(false);
                if_then_else_monad(true_val, false_val, lt_bdd, pc, ctx)
            })
        })
    }

    fn compile_beta(
        &mut self,
        a: &PExpr,
        b: &PExpr,
        env: &Env,
        mode: &mut CompileMode<'_>,
    ) -> GuardedWorlds {
        let pc = mode.path_condition();
        let a_worlds = self.traced_compile(a, env, mode, 0);

        bind_compile(a_worlds, pc, self, mode, |a_val, pc, ctx, mode| {
            let b_worlds = ctx.traced_compile(b, env, mode, 1);
            bind_compile(b_worlds, pc, ctx, mode, |b_val, pc, ctx, mode| {
                let a_f = as_float_or_panic(
                    a_val.as_ref(),
                    "beta: first argument must be a number (use 1.0 not 1)",
                );
                let b_f =
                    as_float_or_panic(b_val.as_ref(), "beta: second argument must be a number");

                let name = ctx.state.current_continuous_name(&[a_f, b_f]);
                ctx.state.register_prior(BetaPrior::Beta {
                    name,
                    a: a_f,
                    b: b_f,
                });
                if let CompileMode::Sample(state) = mode {
                    state.ensure_continuous([name], &ctx.state.prior_registry);
                }
                pure_monad(mk_probability(name), pc)
            })
        })
    }

    fn compile_prob_close(
        &mut self,
        prob: &PExpr,
        val: &PExpr,
        env: &Env,
        mode: &mut CompileMode<'_>,
    ) -> GuardedWorlds {
        if matches!(mode, CompileMode::Sample(_)) {
            return self.sample_compile_prob_close(prob, val, env, mode);
        }

        let pc = mode.path_condition();
        let prob_worlds = self.traced_compile(prob, env, mode, 0);

        bind_compile(prob_worlds, pc, self, mode, |prob_val, pc, ctx, mode| {
            let val_worlds = ctx.traced_compile(val, env, mode, 1);
            bind_compile(val_worlds, pc, ctx, mode, |val_val, pc, ctx, _mode| {
                let r = as_float_or_panic(
                    val_val.as_ref(),
                    "prob_close: second argument must be a number",
                );
                match prob_val.as_ref() {
                    PluckValue::Probability(name) => {
                        let pin_var = ctx.state.with_callstack_index(2, |s| {
                            s.current_address_weighted(FlipWeight::Beta(BetaFlip::ProbPin(
                                *name, r,
                            )))
                        });
                        let true_val = ctx.bool_val(true);
                        let false_val = ctx.bool_val(false);
                        if_then_else_monad(true_val, false_val, pin_var, pc, ctx)
                    }
                    PluckValue::Native(NativeVal::Float(c)) => {
                        let result = ctx.bool_val((*c - r).abs() < 1e-15);
                        pure_monad(result, pc)
                    }
                    _ => panic!("prob_close: first argument must be a Probability or float"),
                }
            })
        })
    }

    fn compile_gaussian(
        &mut self,
        mu: &PExpr,
        sigma: &PExpr,
        env: &Env,
        mode: &mut CompileMode<'_>,
    ) -> GuardedWorlds {
        let pc = mode.path_condition();
        let mu_worlds = self.traced_compile(mu, env, mode, 0);

        bind_compile(mu_worlds, pc, self, mode, |mu_val, pc, ctx, mode| {
            let sigma_worlds = ctx.traced_compile(sigma, env, mode, 1);
            bind_compile(sigma_worlds, pc, ctx, mode, |sigma_val, pc, ctx, mode| {
                if matches!(
                    mu_val.as_ref(),
                    PluckValue::Native(NativeVal::Float(_)) | PluckValue::Native(NativeVal::Int(_))
                ) && matches!(
                    sigma_val.as_ref(),
                    PluckValue::Native(NativeVal::Float(_)) | PluckValue::Native(NativeVal::Int(_))
                ) {
                    let mu_f = as_float_or_panic(mu_val.as_ref(), "gaussian: mu must be a number");
                    let sigma_f =
                        as_float_or_panic(sigma_val.as_ref(), "gaussian: sigma must be a number");
                    let name = ctx.state.current_continuous_name(&[mu_f, sigma_f]);
                    ctx.state
                        .register_prior(GaussianPrior::from_single(name, mu_f, sigma_f));
                    if let CompileMode::Sample(state) = mode {
                        state.ensure_continuous([name], &ctx.state.prior_registry);
                    }
                    let mut coefficients = std::collections::BTreeMap::new();
                    coefficients.insert(name, ordered_float::NotNan::new(1.0).unwrap());
                    return pure_monad(mk_gaussian_expr(0.0, coefficients), pc);
                }

                let (mu_entries, mu_shape) = match mu_val.as_ref() {
                    PluckValue::FloatMatrix { entries, shape } => (entries.clone(), shape.clone()),
                    _ => panic!(
                        "gaussian: mu must be a scalar or a 1D FloatMatrix, got {:?}",
                        mu_val
                    ),
                };
                let (cov_entries, cov_shape) = match sigma_val.as_ref() {
                    PluckValue::FloatMatrix { entries, shape } => (entries.clone(), shape.clone()),
                    _ => panic!(
                        "gaussian: sigma must be a scalar or a 2D FloatMatrix, got {:?}",
                        sigma_val
                    ),
                };
                assert_eq!(
                    mu_shape.len(),
                    1,
                    "gaussian: mu matrix must be 1D, got shape {:?}",
                    mu_shape
                );
                let n = mu_shape[0];
                assert_eq!(
                    cov_shape,
                    vec![n, n],
                    "gaussian: cov must have shape [{}, {}], got {:?}",
                    n,
                    n,
                    cov_shape
                );

                let builder = ctx.state.bdd();
                let mut used = BddPtr::true_ptr();
                let mut mu_vec = nalgebra::DVector::<f64>::zeros(n);
                for (i, e) in mu_entries.iter().enumerate() {
                    let (c, m, u) = force_entry_to_affine(ctx, e, pc);
                    used = builder.and(used, u);
                    assert!(
                        m.is_empty(),
                        "gaussian: mu[{}] depends on a symbolic Gaussian; mu must be deterministic",
                        i
                    );
                    mu_vec[i] = c;
                }
                let mut cov_mat = nalgebra::DMatrix::<f64>::zeros(n, n);
                for r in 0..n {
                    for c_idx in 0..n {
                        let (cval, m, u) =
                            force_entry_to_affine(ctx, &cov_entries[r * n + c_idx], pc);
                        used = builder.and(used, u);
                        assert!(
                            m.is_empty(),
                            "gaussian: cov[{},{}] depends on a symbolic Gaussian; cov must be deterministic",
                            r,
                            c_idx
                        );
                        cov_mat[(r, c_idx)] = cval;
                    }
                }
                for r in 0..n {
                    for c_idx in (r + 1)..n {
                        let asym = (cov_mat[(r, c_idx)] - cov_mat[(c_idx, r)]).abs();
                        assert!(
                            asym < 1e-9,
                            "gaussian: cov is not symmetric at [{},{}] (|diff|={})",
                            r,
                            c_idx,
                            asym
                        );
                    }
                }
                assert!(
                    cov_mat.clone().cholesky().is_some(),
                    "gaussian: cov is not positive semi-definite"
                );

                let mut key_params: Vec<f64> = Vec::with_capacity(n + n * n);
                key_params.extend(mu_vec.iter().copied());
                for r in 0..n {
                    for c in 0..n {
                        key_params.push(cov_mat[(r, c)]);
                    }
                }
                let names = ctx.state.current_continuous_names(&key_params, n);

                ctx.state.register_prior(GaussianPrior::from_multivariate(
                    names.clone(),
                    mu_vec,
                    cov_mat,
                ));
                if let CompileMode::Sample(state) = mode {
                    state.ensure_continuous(names.iter().copied(), &ctx.state.prior_registry);
                }

                let one = ordered_float::NotNan::new(1.0).unwrap();
                let entries: Vec<PluckVal> = names
                    .iter()
                    .map(|&nm| {
                        let mut coeffs = std::collections::BTreeMap::new();
                        coeffs.insert(nm, one);
                        mk_gaussian_expr(0.0, coeffs)
                    })
                    .collect();
                pure_monad_used(mk_float_matrix(entries, vec![n]), used)
            })
        })
    }

    fn compile_close(
        &mut self,
        expr: &PExpr,
        val: &PExpr,
        env: &Env,
        mode: &mut CompileMode<'_>,
    ) -> GuardedWorlds {
        if matches!(mode, CompileMode::Sample(_)) {
            return self.sample_compile_close(expr, val, env, mode);
        }

        let pc = mode.path_condition();
        let expr_worlds = self.traced_compile(expr, env, mode, 0);

        bind_compile(expr_worlds, pc, self, mode, |expr_val, pc, ctx, mode| {
            let val_worlds = ctx.traced_compile(val, env, mode, 1);
            bind_compile(val_worlds, pc, ctx, mode, |val_v, pc, ctx, _mode| {
                let observed = as_float_or_panic(val_v.as_ref(), "close: val must be a number");
                match expr_val.as_ref() {
                    PluckValue::GaussianExpr {
                        constant,
                        coefficients,
                    } => {
                        if coefficients.is_empty() {
                            let result = ctx.bool_val((*constant - observed).abs() < 1e-15);
                            pure_monad(result, pc)
                        } else {
                            let coeffs_f64 = coefficients
                                .iter()
                                .map(|(k, v)| (*k, v.into_inner()))
                                .collect();
                            let weight = FlipWeight::Gaussian(GaussianFlip::Obs {
                                coefficients: coeffs_f64,
                                constant: *constant,
                                observed,
                            });
                            let obs_var = ctx
                                .state
                                .with_callstack_index(2, |s| s.current_address_weighted(weight));
                            let true_val = ctx.bool_val(true);
                            let false_val = ctx.bool_val(false);
                            if_then_else_monad(true_val, false_val, obs_var, pc, ctx)
                        }
                    }
                    PluckValue::Native(NativeVal::Float(f)) => {
                        let result = ctx.bool_val((*f - observed).abs() < 1e-15);
                        pure_monad(result, pc)
                    }
                    PluckValue::Gamma { name, scale } => {
                        if observed <= 0.0 {
                            pure_monad(ctx.bool_val(false), pc)
                        } else {
                            ctx.gamma_flip_worlds(
                                GammaFlip::RatePin {
                                    name: *name,
                                    value: observed,
                                    scale: *scale,
                                },
                                pc,
                            )
                        }
                    }
                    PluckValue::Exponential { name, gamma, scale } => {
                        if observed < 0.0 {
                            pure_monad(ctx.bool_val(false), pc)
                        } else {
                            ctx.gamma_flip_worlds(
                                GammaFlip::ExpEq {
                                    gamma: *gamma,
                                    draw: *name,
                                    value: observed,
                                    scale: *scale,
                                },
                                pc,
                            )
                        }
                    }
                    PluckValue::Poisson { .. } => {
                        panic!("close: Poisson draws are integer-valued; use native_eq")
                    }
                    _ => panic!(
                        "close: expr must be a GaussianExpr, Gamma, Exponential \
                         or float, got {:?}",
                        expr_val
                    ),
                }
            })
        })
    }

    fn compile_real_cmp(
        &mut self,
        cmp: RealCmp,
        expr: &PExpr,
        val: &PExpr,
        env: &Env,
        mode: &mut CompileMode<'_>,
    ) -> GuardedWorlds {
        let pc = mode.path_condition();
        let expr_worlds = self.traced_compile(expr, env, mode, 0);

        bind_compile(
            expr_worlds,
            pc,
            self,
            mode,
            move |expr_val, pc, ctx, mode| {
                let val_worlds = ctx.traced_compile(val, env, mode, 1);
                bind_compile(val_worlds, pc, ctx, mode, move |val_v, pc, ctx, mode| {
                    let op_name = match cmp {
                        RealCmp::Lt => "real_lt",
                        RealCmp::Geq => "real_geq",
                    };
                    let bound = as_float_or_panic(
                        val_v.as_ref(),
                        &format!("{op_name}: bound must be a number"),
                    );
                    match expr_val.as_ref() {
                        PluckValue::Native(NativeVal::Float(f)) => {
                            let holds = match cmp {
                                RealCmp::Lt => *f < bound,
                                RealCmp::Geq => *f >= bound,
                            };
                            pure_monad(ctx.bool_val(holds), pc)
                        }
                        PluckValue::Exponential { name, gamma, scale } => {
                            if bound <= 0.0 {
                                let holds = matches!(cmp, RealCmp::Geq);
                                return pure_monad(ctx.bool_val(holds), pc);
                            }
                            let b = NotNan::new(bound).unwrap();
                            let interval = match cmp {
                                RealCmp::Lt => Interval::lt(b),
                                RealCmp::Geq => Interval::geq(b),
                            };
                            let flip = GammaFlip::ExpInterval {
                                gamma: *gamma,
                                draw: *name,
                                interval,
                                scale: *scale,
                            };
                            if matches!(mode, CompileMode::Sample(_)) {
                                return ctx.sample_gamma_flip_outcome(
                                    flip,
                                    GammaDrawFamily::Exponential,
                                    *gamma,
                                    *name,
                                    *scale,
                                    mode,
                                );
                            }
                            ctx.gamma_flip_worlds(flip, pc)
                        }
                        v => panic!(
                            "{op_name}: range observations are only supported on \
                         Exponential draws (or plain floats), got {:?}",
                            v
                        ),
                    }
                })
            },
        )
    }

    pub fn compile_pin(
        &mut self,
        expr: &PExpr,
        value: &PExpr,
        env: &Env,
        mode: &mut CompileMode<'_>,
    ) -> GuardedWorlds {
        debug_assert!(
            matches!(mode, CompileMode::KC { .. }),
            "compile_pin is conditioning machinery and must run in KC mode \
             (a Sample-mode bind would collapse the pinned expression's \
             worlds to one)"
        );
        let pc = mode.path_condition();
        let expr_worlds = self.traced_compile(expr, env, mode, 0);
        let result = bind_compile(expr_worlds, pc, self, mode, |sym_val, pc, ctx, _mode| {
            let mut inner_mode = CompileMode::KC { path_condition: pc };
            let val_worlds = ctx.traced_compile(value, env, &mut inner_mode, 1);
            bind_compile(
                val_worlds,
                pc,
                ctx,
                &mut inner_mode,
                |concrete_val, pc, ctx, _mode| ctx.pin_value(&sym_val, &concrete_val, pc),
            )
        });
        let true_sym = self.true_sym;
        debug_assert!(
            result.0.iter().all(|(v, _)| matches!(
                v.as_ref(),
                PluckValue::Value { constructor, args }
                    if *constructor == true_sym && args.is_empty()
            )),
            "compile_pin returned a non-True world; pin_value must yield \
             True-valued worlds or none"
        );
        result
    }

    pub fn pin_value(
        &mut self,
        sym: &PluckVal,
        concrete: &PluckVal,
        pc: BddPtr<'static>,
    ) -> GuardedWorlds {
        match (sym.as_ref(), concrete.as_ref()) {
            (
                PluckValue::Value {
                    constructor: cs,
                    args: args_s,
                },
                PluckValue::Value {
                    constructor: cc,
                    args: args_c,
                },
            ) => {
                if cs != cc || args_s.len() != args_c.len() {
                    return program_error_worlds();
                }
                let pairs: Vec<(PluckVal, PluckVal)> =
                    args_s.iter().cloned().zip(args_c.iter().cloned()).collect();
                self.pin_args_and(pairs, pc)
            }

            (PluckValue::IntDist { bits: bs }, PluckValue::IntDist { bits: bc }) => {
                if bs.len() != bc.len() {
                    return program_error_worlds();
                }
                let builder = self.state.bdd();
                let mut eq_bdd = BddPtr::true_ptr();
                for (b_s, b_c) in bs.iter().zip(bc.iter()) {
                    let bit_iff = builder.iff(*b_s, *b_c);
                    eq_bdd = builder.and(eq_bdd, bit_iff);
                }
                self.pin_holds_when(eq_bdd)
            }

            (PluckValue::Native(a), PluckValue::Native(b)) => {
                if a == b {
                    pure_monad(self.bool_val(true), pc)
                } else {
                    program_error_worlds()
                }
            }

            (PluckValue::Probability(name), PluckValue::Native(NativeVal::Float(r))) => {
                let pin_var = self.state.with_callstack_index(2, |s| {
                    s.current_address_weighted(FlipWeight::Beta(BetaFlip::ProbPin(*name, *r)))
                });
                self.pin_holds_when(pin_var)
            }

            (PluckValue::Gamma { name, scale }, PluckValue::Native(NativeVal::Float(r))) => {
                if *r <= 0.0 {
                    return program_error_worlds();
                }
                let pin_var = self.state.with_callstack_index(2, |s| {
                    s.current_address_weighted(FlipWeight::Gamma(GammaFlip::RatePin {
                        name: *name,
                        value: *r,
                        scale: *scale,
                    }))
                });
                self.pin_holds_when(pin_var)
            }

            (
                PluckValue::Exponential { name, gamma, scale },
                PluckValue::Native(NativeVal::Float(r)),
            ) => {
                if *r < 0.0 {
                    return program_error_worlds();
                }
                let pin_var = self.state.with_callstack_index(2, |s| {
                    s.current_address_weighted(FlipWeight::Gamma(GammaFlip::ExpEq {
                        gamma: *gamma,
                        draw: *name,
                        value: *r,
                        scale: *scale,
                    }))
                });
                self.pin_holds_when(pin_var)
            }

            (PluckValue::Poisson { name, gamma, scale }, PluckValue::Native(NativeVal::Int(k))) => {
                if *k < 0 {
                    return program_error_worlds();
                }
                let pin_var = self.state.with_callstack_index(2, |s| {
                    s.current_address_weighted(FlipWeight::Gamma(GammaFlip::PoissonEvent {
                        gamma: *gamma,
                        draw: *name,
                        event: IntervalOrEq::eq(*k as u64),
                        scale: *scale,
                    }))
                });
                self.pin_holds_when(pin_var)
            }

            (
                PluckValue::GaussianExpr {
                    constant,
                    coefficients,
                },
                PluckValue::Native(NativeVal::Float(r)),
            ) => {
                if coefficients.is_empty() {
                    if (*constant - *r).abs() < 1e-15 {
                        pure_monad(self.bool_val(true), pc)
                    } else {
                        program_error_worlds()
                    }
                } else {
                    let coeffs_f64 = coefficients
                        .iter()
                        .map(|(k, v)| (*k, v.into_inner()))
                        .collect();
                    let weight = FlipWeight::Gaussian(GaussianFlip::Obs {
                        coefficients: coeffs_f64,
                        constant: *constant,
                        observed: *r,
                    });
                    let obs_var = self
                        .state
                        .with_callstack_index(2, |s| s.current_address_weighted(weight));
                    self.pin_holds_when(obs_var)
                }
            }

            (
                PluckValue::DirichletVector { name, categories },
                PluckValue::Value { .. } | PluckValue::FloatMatrix { .. },
            ) => {
                let observed = extract_dirichlet_value(concrete);
                if observed.len() != *categories {
                    return program_error_worlds();
                }
                let weight = FlipWeight::Dirichlet(DirichletFlip::VecPin {
                    name: *name,
                    value: observed,
                });
                let obs_var = self
                    .state
                    .with_callstack_index(2, |s| s.current_address_weighted(weight));
                self.pin_holds_when(obs_var)
            }

            (PluckValue::DirichletProbability { .. }, PluckValue::Native(NativeVal::Float(_))) => {
                panic!(
                    "compile_pin: cannot pin a single Dirichlet component \
                     (e.g. (vector_index weights i)); block on the parent \
                     vector instead so correlations between components are \
                     preserved."
                );
            }

            (
                PluckValue::FloatMatrix {
                    entries: es,
                    shape: ss,
                },
                PluckValue::FloatMatrix {
                    entries: ec,
                    shape: sc,
                },
            ) => {
                if ss != sc || es.len() != ec.len() {
                    return program_error_worlds();
                }
                let pairs: Vec<(PluckVal, PluckVal)> =
                    es.iter().cloned().zip(ec.iter().cloned()).collect();
                self.pin_args_and(pairs, pc)
            }

            _ => program_error_worlds(),
        }
    }

    fn pin_holds_when(&self, g: BddPtr<'static>) -> GuardedWorlds {
        if g.is_false() {
            program_error_worlds()
        } else {
            (vec![(self.bool_val(true), g)], BddPtr::true_ptr())
        }
    }

    fn pin_args_and(
        &mut self,
        pairs: Vec<(PluckVal, PluckVal)>,
        pc: BddPtr<'static>,
    ) -> GuardedWorlds {
        let mut acc = pure_monad(self.bool_val(true), pc);
        for (sym_arg, concrete_arg) in pairs {
            let mut outer_mode = CompileMode::KC { path_condition: pc };
            acc = bind_compile(acc, pc, self, &mut outer_mode, move |_, pc, ctx, _mode| {
                let mut inner_mode = CompileMode::KC { path_condition: pc };
                let forced = ctx.evaluate_thunk(&sym_arg, &mut inner_mode);
                let concrete_inner = concrete_arg.clone();
                bind_compile(
                    forced,
                    pc,
                    ctx,
                    &mut inner_mode,
                    move |forced_sym, pc, ctx, _mode| {
                        ctx.pin_value(&forced_sym, &concrete_inner, pc)
                    },
                )
            });
        }
        acc
    }

    fn compile_dirichlet_eq(
        &mut self,
        dexpr: &PExpr,
        val: &PExpr,
        env: &Env,
        mode: &mut CompileMode<'_>,
    ) -> GuardedWorlds {
        if matches!(mode, CompileMode::Sample(_)) {
            return self.sample_compile_dirichlet_eq(dexpr, val, env, mode);
        }

        let pc = mode.path_condition();
        let expr_worlds = self.traced_compile(dexpr, env, mode, 0);

        bind_compile(expr_worlds, pc, self, mode, |expr_val, pc, ctx, mode| {
            let val_worlds = ctx.traced_compile(val, env, mode, 1);
            bind_compile(val_worlds, pc, ctx, mode, |val_v, pc, ctx, _mode| {
                let observed = extract_dirichlet_value(val_v.as_ref());
                match expr_val.as_ref() {
                    PluckValue::DirichletVector { name, categories } => {
                        assert_eq!(
                            observed.len(),
                            *categories,
                            "dirichlet_eq: value length {} does not match Dirichlet's {} categories",
                            observed.len(),
                            *categories
                        );
                        let weight = FlipWeight::Dirichlet(DirichletFlip::VecPin {
                            name: *name,
                            value: observed,
                        });
                        let obs_var = ctx
                            .state
                            .with_callstack_index(2, |s| s.current_address_weighted(weight));
                        let true_val = ctx.bool_val(true);
                        let false_val = ctx.bool_val(false);
                        if_then_else_monad(true_val, false_val, obs_var, pc, ctx)
                    }
                    PluckValue::FloatMatrix { .. } => {
                        let lhs = extract_dirichlet_value(expr_val.as_ref());
                        let matches = lhs.len() == observed.len()
                            && lhs.iter().zip(observed.iter()).all(|(a, b)| a == b);
                        let result = ctx.bool_val(matches);
                        pure_monad(result, pc)
                    }
                    _ => panic!(
                        "dirichlet_eq: expr must be a DirichletVector or 1-D FloatMatrix, got {:?}",
                        expr_val
                    ),
                }
            })
        })
    }

    fn sample_compile_dirichlet_eq(
        &mut self,
        dexpr: &PExpr,
        val: &PExpr,
        env: &Env,
        mode: &mut CompileMode<'_>,
    ) -> GuardedWorlds {
        let expr_val = self.sample_compile_one(dexpr, env, mode, 0, "DirichletEq: expr");
        let val_v = self.sample_compile_one(val, env, mode, 1, "DirichletEq: val");

        let observed = extract_dirichlet_value(val_v.as_ref());
        match expr_val.as_ref() {
            PluckValue::DirichletVector { name, categories } => {
                assert_eq!(
                    observed.len(),
                    *categories,
                    "dirichlet_eq: value length {} does not match Dirichlet's {} categories",
                    observed.len(),
                    *categories
                );
                let weight = FlipWeight::Dirichlet(DirichletFlip::VecPin {
                    name: *name,
                    value: observed,
                });
                let CompileMode::Sample(sample_state) = mode else {
                    unreachable!()
                };
                let outcome = self.state.with_callstack_index(2, |s| {
                    s.current_address_weighted(weight);
                    crate::inference::sampling::resolve_observation(sample_state, s)
                });
                pure_monad(self.bool_val(outcome), BddPtr::true_ptr())
            }
            PluckValue::FloatMatrix { .. } => {
                let lhs = extract_dirichlet_value(expr_val.as_ref());
                let matches = lhs.len() == observed.len()
                    && lhs.iter().zip(observed.iter()).all(|(a, b)| a == b);
                pure_monad(self.bool_val(matches), BddPtr::true_ptr())
            }
            _ => panic!(
                "dirichlet_eq: expr must be a DirichletVector or 1-D FloatMatrix, got {:?}",
                expr_val
            ),
        }
    }

    fn compile_dirichlet(
        &mut self,
        alphas: &[PExpr],
        env: &Env,
        mode: &mut CompileMode<'_>,
    ) -> GuardedWorlds {
        self.compile_dirichlet_step(alphas, env, mode, 0, vec![])
    }

    fn compile_dirichlet_step(
        &mut self,
        remaining: &[PExpr],
        env: &Env,
        mode: &mut CompileMode<'_>,
        idx: i32,
        acc: Vec<f64>,
    ) -> GuardedWorlds {
        let pc = mode.path_condition();
        if remaining.is_empty() {
            let name = self.state.current_continuous_name(&acc);
            self.state.register_prior(DirichletPrior::Dirichlet {
                name,
                alphas: acc.clone(),
            });
            if let CompileMode::Sample(state) = mode {
                state.ensure_continuous([name], &self.state.prior_registry);
            }
            let dirichlet = mk_dirichlet(name, acc.len());
            return pure_monad(dirichlet, pc);
        }

        let expr = &remaining[0];
        let worlds = self.traced_compile(expr, env, mode, idx);

        bind_compile(worlds, pc, self, mode, move |val, _new_pc, ctx, mode| {
            let f = as_float_or_panic(
                val.as_ref(),
                &format!("dirichlet: argument {} must be a number", idx),
            );
            let mut new_acc = acc.clone();
            new_acc.push(f);
            ctx.compile_dirichlet_step(&remaining[1..], env, mode, idx + 1, new_acc)
        })
    }

    fn compile_index(
        &mut self,
        v: &PExpr,
        idx: &[PExpr],
        env: &Env,
        mode: &mut CompileMode<'_>,
    ) -> GuardedWorlds {
        assert!(
            !idx.is_empty(),
            "vector_index requires at least one index argument"
        );
        let pc = mode.path_condition();
        let v_worlds = self.traced_compile(v, env, mode, 0);
        let idx_owned: Vec<PExpr> = idx.to_vec();
        bind_compile(v_worlds, pc, self, mode, move |v_val, _pc, ctx, mode| {
            ctx.compile_index_resolve(v_val, &idx_owned, env, mode, Vec::new())
        })
    }

    fn compile_index_resolve(
        &mut self,
        v_val: PluckVal,
        remaining: &[PExpr],
        env: &Env,
        mode: &mut CompileMode<'_>,
        acc: Vec<i64>,
    ) -> GuardedWorlds {
        let pc = mode.path_condition();
        if remaining.is_empty() {
            return match v_val.as_ref() {
                PluckValue::DirichletVector { name, categories } => {
                    self.compile_index_dirichlet(*name, *categories, &acc, pc)
                }
                PluckValue::FloatMatrix { entries, shape } => {
                    self.compile_index_float_matrix(entries, shape, &acc, pc)
                }
                _ => panic!(
                    "vector_index: first argument must be a DirichletVector or FloatMatrix, got {:?}",
                    v_val
                ),
            };
        }
        let worlds = self.traced_compile(&remaining[0], env, mode, 0);
        let rest: Vec<PExpr> = remaining[1..].to_vec();
        bind_compile(
            worlds,
            pc,
            self,
            mode,
            move |idx_val, _new_pc, ctx, mode| {
                let i = as_int_or_panic(idx_val.as_ref(), "vector_index: index");
                let mut new_acc = acc.clone();
                new_acc.push(i);
                ctx.compile_index_resolve(v_val.clone(), &rest, env, mode, new_acc)
            },
        )
    }

    fn compile_index_dirichlet(
        &self,
        name: u64,
        categories: usize,
        idx_vals: &[i64],
        pc: BddPtr<'static>,
    ) -> GuardedWorlds {
        assert_eq!(
            idx_vals.len(),
            1,
            "vector_index on DirichletVector takes exactly one index, got {}",
            idx_vals.len()
        );
        let i = idx_vals[0];
        assert!(
            i >= 0 && (i as usize) < categories,
            "vector_index: index {} out of bounds for DirichletVector of {} categories",
            i,
            categories
        );
        pure_monad(mk_dirichlet_probability(name, i as usize, categories), pc)
    }

    fn compile_index_float_matrix(
        &self,
        entries: &[PluckVal],
        shape: &[usize],
        idx_vals: &[i64],
        pc: BddPtr<'static>,
    ) -> GuardedWorlds {
        let bounds_check = |dim_idx: usize, val: i64| {
            let dim = shape[dim_idx];
            assert!(
                val >= 0 && (val as usize) < dim,
                "vector_index: index {} out of bounds for axis {} (size {})",
                val,
                dim_idx,
                dim
            );
        };

        let result = match (shape.len(), idx_vals.len()) {
            (1, 1) => {
                bounds_check(0, idx_vals[0]);
                entries[idx_vals[0] as usize].clone()
            }
            (2, 1) => {
                bounds_check(0, idx_vals[0]);
                let cols = shape[1];
                let row_start = idx_vals[0] as usize * cols;
                let row: Vec<PluckVal> = entries[row_start..row_start + cols].to_vec();
                mk_float_matrix(row, vec![cols])
            }
            (2, 2) => {
                bounds_check(0, idx_vals[0]);
                bounds_check(1, idx_vals[1]);
                let cols = shape[1];
                entries[idx_vals[0] as usize * cols + idx_vals[1] as usize].clone()
            }
            (r, k) => panic!(
                "vector_index: unsupported (rank, n_indices) = ({}, {}) for FloatMatrix of shape {:?}",
                r, k, shape
            ),
        };
        pure_monad(result, pc)
    }

    fn compile_categorical(
        &mut self,
        probs: &[PExpr],
        env: &Env,
        mode: &mut CompileMode<'_>,
    ) -> GuardedWorlds {
        if matches!(mode, CompileMode::Sample(_)) {
            return self.sample_compile_categorical(probs, env, mode);
        }
        let pc = mode.path_condition();
        let total = probs.len();
        self.compile_categorical_step(probs, env, pc, 0, total, vec![], None)
    }

    fn sample_compile_categorical(
        &mut self,
        probs: &[PExpr],
        env: &Env,
        mode: &mut CompileMode<'_>,
    ) -> GuardedWorlds {
        let mut float_acc: Vec<f64> = Vec::new();
        let mut vector: Option<(u64, usize)> = None;
        for (idx, prob) in probs.iter().enumerate() {
            let val = self.sample_compile_one(prob, env, mode, idx as i32, "Categorical: prob");
            match val.as_ref() {
                PluckValue::Native(NativeVal::Float(f)) => float_acc.push(*f),
                PluckValue::DirichletVector { name, categories } => {
                    if probs.len() != 1 {
                        panic!(
                            "Categorical: DirichletVector must be the single argument, got {} total args",
                            probs.len()
                        );
                    }
                    vector = Some((*name, *categories));
                }
                _ => panic!(
                    "Categorical: argument {} must be a float or a DirichletVector, got {:?}",
                    idx, val
                ),
            }
        }

        let kind = if let Some((name, categories)) = vector {
            if !float_acc.is_empty() {
                eprintln!("Categorical: cannot mix DirichletVector with constant probabilities");
                return program_error_worlds();
            }
            CatKind::Dirichlet { name, categories }
        } else {
            let sum_diff = (float_acc.iter().sum::<f64>() - 1.0).abs();
            if sum_diff > 1e-9 {
                eprintln!("Categorical: constant probabilities do not sum to one");
                return program_error_worlds();
            }
            CatKind::Constants(float_acc)
        };

        let n = match &kind {
            CatKind::Dirichlet { categories, .. } => *categories,
            CatKind::Constants(fs) => fs.len(),
        };
        if n == 0 {
            return program_error_worlds();
        }
        if n == 1 {
            return pure_monad(mk_native_int(0), BddPtr::true_ptr());
        }

        let betas: Vec<f64> = match &kind {
            CatKind::Constants(fs) => stick_betas(fs),
            CatKind::Dirichlet { name, .. } => {
                let CompileMode::Sample(sample_state) = mode else {
                    unreachable!()
                };
                let theta = sample_state.assignment.dirichlet_value(*name);
                let mut suffix = vec![0.0_f64; n + 1];
                for i in (0..n).rev() {
                    suffix[i] = suffix[i + 1] + theta[i].into_inner();
                }
                (0..n - 1)
                    .map(|i| {
                        let r_i = suffix[i];
                        if r_i <= 0.0 {
                            0.0
                        } else {
                            theta[i].into_inner() / r_i
                        }
                    })
                    .collect()
            }
        };

        #[allow(clippy::needless_range_loop)]
        for i in 0..n - 1 {
            let weight = match &kind {
                CatKind::Dirichlet { name, .. } => {
                    FlipWeight::Dirichlet(DirichletFlip::StickElement {
                        name: *name,
                        index: i,
                    })
                }
                CatKind::Constants(_) => FlipWeight::Constant(betas[i]),
            };
            let discriminator = weight.discriminator();
            let key = self.state.with_callstack_index(i as i32, |s| {
                let k = (s.callstack.hash(), discriminator);
                s.current_address_weighted(weight);
                k
            });
            let CompileMode::Sample(sample_state) = mode else {
                unreachable!()
            };
            let hit =
                crate::inference::sampling::resolve_flip(betas[i], key, sample_state, &*self.state);
            if hit {
                return pure_monad(mk_native_int(i as i64), BddPtr::true_ptr());
            }
        }

        pure_monad(mk_native_int((n - 1) as i64), BddPtr::true_ptr())
    }

    #[allow(clippy::too_many_arguments)]
    fn compile_categorical_step(
        &mut self,
        remaining: &[PExpr],
        env: &Env,
        pc: BddPtr<'static>,
        idx: i32,
        total_args: usize,
        float_acc: Vec<f64>,
        vector: Option<(u64, usize)>,
    ) -> GuardedWorlds {
        if remaining.is_empty() {
            let kind = if let Some((name, categories)) = vector {
                if !float_acc.is_empty() {
                    eprintln!(
                        "Categorical: cannot mix DirichletVector with constant probabilities"
                    );
                    return program_error_worlds();
                }
                CatKind::Dirichlet { name, categories }
            } else {
                let sum_diff = (float_acc.iter().sum::<f64>() - 1.0).abs();
                if sum_diff > 1e-9 {
                    eprintln!("Categorical: constant probabilities do not sum to one");
                    return program_error_worlds();
                }
                CatKind::Constants(float_acc)
            };
            return self.finalize_categorical(pc, kind);
        }

        let expr = &remaining[0];
        let mut inner_mode = CompileMode::KC { path_condition: pc };
        let worlds = self.traced_compile(expr, env, &mut inner_mode, idx);

        bind_compile(
            worlds,
            pc,
            self,
            &mut inner_mode,
            move |val, new_pc, ctx, _mode| {
                let mut new_float_acc = float_acc.clone();
                let mut new_vector = vector;

                match val.as_ref() {
                    PluckValue::Native(NativeVal::Float(f)) => new_float_acc.push(*f),
                    PluckValue::DirichletVector { name, categories } => {
                        if total_args != 1 {
                            panic!(
                            "Categorical: DirichletVector must be the single argument, got {} total args",
                            total_args
                        );
                        }
                        new_vector = Some((*name, *categories));
                    }
                    _ => panic!(
                        "Categorical: argument {} must be a float or a DirichletVector, got {:?}",
                        idx, val
                    ),
                };

                ctx.compile_categorical_step(
                    &remaining[1..],
                    env,
                    new_pc,
                    idx + 1,
                    total_args,
                    new_float_acc,
                    new_vector,
                )
            },
        )
    }

    fn finalize_categorical(&mut self, pc: BddPtr<'static>, kind: CatKind) -> GuardedWorlds {
        let n = match &kind {
            CatKind::Dirichlet { categories, .. } => *categories,
            CatKind::Constants(fs) => fs.len(),
        };
        if n == 0 {
            return program_error_worlds();
        }

        let vals: Vec<PluckVal> = (0..n).map(|i| mk_native_int(i as i64)).collect();

        if n == 1 {
            return categorical_monad(vals, vec![BddPtr::true_ptr()], pc, self);
        }

        let betas: Option<Vec<f64>> = match &kind {
            CatKind::Constants(fs) => Some(stick_betas(fs)),
            CatKind::Dirichlet { .. } => None,
        };

        let vars = allocate_one_hot_vars(
            n - 1,
            |i| match &kind {
                CatKind::Dirichlet { name, .. } => {
                    FlipWeight::Dirichlet(DirichletFlip::StickElement {
                        name: *name,
                        index: i,
                    })
                }
                CatKind::Constants(_) => FlipWeight::Constant(betas.as_ref().unwrap()[i]),
            },
            self.state,
        );

        let conditions = build_stick_breaking_guards(&vars, self.state);

        categorical_monad(vals, conditions, pc, self)
    }

    fn compile_variable_aware_binop(
        &mut self,
        op: FloatOp,
        a: &PExpr,
        b: &PExpr,
        env: &Env,
        mode: &mut CompileMode<'_>,
    ) -> GuardedWorlds {
        let pc = mode.path_condition();
        let a_worlds = self.traced_compile(a, env, mode, 0);

        bind_compile(a_worlds, pc, self, mode, move |a_val, pc, ctx, mode| {
            let b_worlds = ctx.traced_compile(b, env, mode, 1);
            bind_compile(b_worlds, pc, ctx, mode, move |b_val, pc, ctx, _mode| {
                match try_scale_gamma(op, &a_val, &b_val) {
                    Some(scaled) => pure_monad(scaled, pc),
                    None => {
                        let (result, used) = apply_float_op(&op, &a_val, &b_val, ctx, pc);
                        pure_monad_used(result, used)
                    }
                }
            })
        })
    }

    fn compile_gamma(
        &mut self,
        shape: &PExpr,
        rate: &PExpr,
        env: &Env,
        mode: &mut CompileMode<'_>,
    ) -> GuardedWorlds {
        let pc = mode.path_condition();
        let shape_worlds = self.traced_compile(shape, env, mode, 0);

        bind_compile(shape_worlds, pc, self, mode, |shape_val, pc, ctx, mode| {
            let rate_worlds = ctx.traced_compile(rate, env, mode, 1);
            bind_compile(rate_worlds, pc, ctx, mode, |rate_val, pc, ctx, mode| {
                let shape_f = as_float_or_panic(
                    shape_val.as_ref(),
                    "Gamma: first argument must be a number (use 1.0 not 1)",
                );
                let rate_f = as_float_or_panic(
                    rate_val.as_ref(),
                    "Gamma: second argument must be a number (use 1.0 not 1)",
                );
                assert!(
                    shape_f > 0.0 && rate_f > 0.0,
                    "Gamma: shape and rate must be positive, got Gamma({shape_f}, {rate_f})"
                );

                let name = ctx.state.current_continuous_name(&[shape_f, rate_f]);
                ctx.state.register_prior(GammaPrior::Gamma {
                    name,
                    shape: shape_f,
                    rate: rate_f,
                });
                if let CompileMode::Sample(state) = mode {
                    state.ensure_continuous([name], &ctx.state.prior_registry);
                }
                pure_monad(mk_gamma(name), pc)
            })
        })
    }

    fn compile_gamma_consumer(
        &mut self,
        rate: &PExpr,
        env: &Env,
        mode: &mut CompileMode<'_>,
        family: GammaDrawFamily,
    ) -> GuardedWorlds {
        let pc = mode.path_condition();

        let rate_worlds = self.traced_compile(rate, env, mode, 0);
        bind_compile(
            rate_worlds,
            pc,
            self,
            mode,
            move |rate_val, pc, ctx, mode| {
                let (gamma_name, scale) = match rate_val.as_ref() {
                    PluckValue::Gamma { name, scale } => (*name, *scale),
                    value => {
                        let value_f = as_float_or_panic(
                            value,
                            "Exponential: rate parameter must be Gamma or a number (use 1.0 not 1)",
                        );
                        assert!(
                            value_f > 0.0,
                            "Poisson/Exponential: rate must be positive, got {value_f}"
                        );

                        let name = ctx.state.current_continuous_name(&[value_f]);
                        ctx.state.register_prior(GammaPrior::Pinned {
                            name,
                            value: value_f,
                        });
                        if let CompileMode::Sample(state) = mode {
                            state.ensure_continuous([name], &ctx.state.prior_registry);
                        }

                        (name, 1.0)
                    }
                };
                let scale_nn = NotNan::new(scale).expect("gamma scale must be finite");
                let draw_name = ctx
                    .state
                    .current_continuous_name(&[gamma_name as f64, scale]);
                ctx.state
                    .gamma_draws
                    .insert(draw_name, (family, (gamma_name, scale_nn)));
                if let CompileMode::Sample(state) = mode {
                    state.ensure_gamma_draw(gamma_name, draw_name, family, scale);
                }
                pure_monad(family.mk_value(gamma_name, draw_name, scale), pc)
            },
        )
    }


    fn compile_float_matrix(
        &mut self,
        entries: &[Vec<PExpr>],
        shape: &[usize],
        env: &Env,
        mode: &mut CompileMode<'_>,
    ) -> GuardedWorlds {
        let pc = mode.path_condition();
        let mut flat: Vec<PluckVal> = Vec::with_capacity(shape.iter().product());
        for row in entries {
            for cell in row {
                let v: PluckVal = match cell {
                    PExpr::ConstNative {
                        val: NativeVal::Float(f),
                    } => mk_native_float(*f),
                    PExpr::ConstNative {
                        val: NativeVal::Int(i),
                    } => mk_native_float(*i as f64),
                    other => make_thunk(other, env.clone(), flat.len() as i32, self.state),
                };
                flat.push(v);
            }
        }
        pure_monad(mk_float_matrix(flat, shape.to_vec()), pc)
    }

    fn compile_sum(
        &mut self,
        expr: &PExpr,
        env: &Env,
        mode: &mut CompileMode<'_>,
    ) -> GuardedWorlds {
        let pc = mode.path_condition();
        let m_worlds = self.traced_compile(expr, env, mode, 0);
        bind_compile(m_worlds, pc, self, mode, |m_val, pc, ctx, _mode| {
            let entries = match m_val.as_ref() {
                PluckValue::FloatMatrix { entries, .. } => entries.clone(),
                _ => panic!("sum: argument must be a FloatMatrix, got {:?}", m_val),
            };
            let builder = ctx.state.bdd();
            let mut used = BddPtr::true_ptr();
            let mut acc_const = 0.0_f64;
            let mut acc_coeffs: std::collections::BTreeMap<u64, ordered_float::NotNan<f64>> =
                std::collections::BTreeMap::new();
            for cell in &entries {
                if let Some(f) = entry_constant(cell.as_ref()) {
                    if f == 0.0 {
                        continue;
                    }
                    acc_const += f;
                    continue;
                }
                let (c, m, u) = force_entry_to_affine(ctx, cell, pc);
                used = builder.and(used, u);
                acc_const += c;
                acc_coeffs = merge_coefficients(&acc_coeffs, &m, |a, b| a + b);
            }
            let out = if acc_coeffs.is_empty() {
                mk_native_float(acc_const)
            } else {
                mk_gaussian_expr(acc_const, acc_coeffs)
            };
            pure_monad_used(out, used)
        })
    }

    fn compile_matmul(
        &mut self,
        a: &PExpr,
        b: &PExpr,
        env: &Env,
        mode: &mut CompileMode<'_>,
    ) -> GuardedWorlds {
        let pc = mode.path_condition();
        let a_worlds = self.traced_compile(a, env, mode, 0);
        bind_compile(a_worlds, pc, self, mode, |a_val, pc, ctx, mode| {
            let b_worlds = ctx.traced_compile(b, env, mode, 1);
            bind_compile(b_worlds, pc, ctx, mode, |b_val, pc, ctx, _mode| {
                let (a_entries, a_shape) = expect_float_matrix(&a_val, "@: left operand");
                let (b_entries, b_shape) = expect_float_matrix(&b_val, "@: right operand");
                let (a_rows, a_cols, a_was_1d) = lift_for_matmul(a_shape, true);
                let (b_rows, b_cols, b_was_1d) = lift_for_matmul(b_shape, false);
                assert_eq!(
                    a_cols, b_rows,
                    "@: inner dimensions disagree: lhs cols = {}, rhs rows = {}",
                    a_cols, b_rows
                );

                let builder = ctx.state.bdd();
                let mut used = BddPtr::true_ptr();
                let mut out_entries: Vec<PluckVal> = Vec::with_capacity(a_rows * b_cols);
                for i in 0..a_rows {
                    for j in 0..b_cols {
                        let mut acc_const = 0.0_f64;
                        let mut acc_coeffs: std::collections::BTreeMap<
                            u64,
                            ordered_float::NotNan<f64>,
                        > = std::collections::BTreeMap::new();
                        for k in 0..a_cols {
                            let (c, m, u) = cell_op_affine(
                                &FloatOp::Mul,
                                &a_entries[i * a_cols + k],
                                &b_entries[k * b_cols + j],
                                ctx,
                                pc,
                            );
                            used = builder.and(used, u);
                            acc_const += c;
                            if !m.is_empty() {
                                acc_coeffs = merge_coefficients(&acc_coeffs, &m, |a, b| a + b);
                            }
                        }
                        out_entries.push(pack_affine(acc_const, acc_coeffs));
                    }
                }

                let result = squeeze_matmul_result(out_entries, a_rows, b_cols, a_was_1d, b_was_1d);
                pure_monad_used(result, used)
            })
        })
    }


    fn sample_compile_one(
        &mut self,
        expr: &PExpr,
        env: &Env,
        mode: &mut CompileMode<'_>,
        strict_order_index: i32,
        label: &'static str,
    ) -> PluckVal {
        let (worlds, _) = self.traced_compile(expr, env, mode, strict_order_index);
        if worlds.is_empty() {
            if self.state.hit_limit {
                panic!(
                    "Sample-mode {}: inference limit (max_depth / time_limit) hit",
                    label
                );
            }
            panic!(
                "Sample-mode {}: produced no worlds \
                 (program error along the sampled path)",
                label
            );
        }
        assert_eq!(
            worlds.len(),
            1,
            "Sample-mode {}: expected exactly one world, got {}",
            label,
            worlds.len()
        );
        worlds.into_iter().next().unwrap().0
    }

    fn sample_compile_flip(
        &mut self,
        prob: &PExpr,
        env: &Env,
        mode: &mut CompileMode<'_>,
    ) -> GuardedWorlds {
        let prob_val = self.sample_compile_one(prob, env, mode, 0, "Flip: prob");
        let flip_weight = flip_weight_from(prob_val.as_ref());

        if let FlipWeight::Constant(p) = &flip_weight {
            if let Some(v) = self.constant_flip_shortcut(*p) {
                return pure_monad(v, BddPtr::true_ptr());
            }
        }

        let CompileMode::Sample(sample_state) = mode else {
            unreachable!();
        };
        let p = flip_weight
            .sample_probability(&sample_state.assignment)
            .expect("sample_compile_flip: flip weight has no scalar probability");
        let discriminator = flip_weight.discriminator();
        let key: super::state::CallstackKey = self.state.with_callstack_index(1, |s| {
            let k = (s.callstack.hash(), discriminator);
            s.current_address_weighted(flip_weight);
            k
        });
        let outcome = crate::inference::sampling::resolve_flip(p, key, sample_state, &*self.state);
        let val = self.bool_val(outcome);

        pure_monad(val, BddPtr::true_ptr())
    }

    fn sample_compile_prob_close(
        &mut self,
        prob: &PExpr,
        val: &PExpr,
        env: &Env,
        mode: &mut CompileMode<'_>,
    ) -> GuardedWorlds {
        let prob_val = self.sample_compile_one(prob, env, mode, 0, "ProbEq: prob");
        let val_val = self.sample_compile_one(val, env, mode, 1, "ProbEq: val");

        let r = as_float_or_panic(
            val_val.as_ref(),
            "prob_close: second argument must be a number",
        );
        match prob_val.as_ref() {
            PluckValue::Probability(name) => {
                let CompileMode::Sample(sample_state) = mode else {
                    unreachable!()
                };
                let outcome = self.state.with_callstack_index(2, |s| {
                    s.current_address_weighted(FlipWeight::Beta(BetaFlip::ProbPin(*name, r)));
                    crate::inference::sampling::resolve_observation(sample_state, s)
                });
                pure_monad(self.bool_val(outcome), BddPtr::true_ptr())
            }
            PluckValue::Native(NativeVal::Float(c)) => {
                let result = self.bool_val((*c - r).abs() < 1e-15);
                pure_monad(result, BddPtr::true_ptr())
            }
            _ => panic!("prob_close: first argument must be a Probability or float"),
        }
    }

    fn sample_compile_close(
        &mut self,
        expr: &PExpr,
        val: &PExpr,
        env: &Env,
        mode: &mut CompileMode<'_>,
    ) -> GuardedWorlds {
        let expr_val = self.sample_compile_one(expr, env, mode, 0, "RealEq: expr");
        let val_v = self.sample_compile_one(val, env, mode, 1, "RealEq: val");

        let observed = as_float_or_panic(val_v.as_ref(), "close: val must be a number");
        match expr_val.as_ref() {
            PluckValue::GaussianExpr {
                constant,
                coefficients,
            } => {
                if coefficients.is_empty() {
                    let result = self.bool_val((*constant - observed).abs() < 1e-15);
                    return pure_monad(result, BddPtr::true_ptr());
                }
                let coeffs_f64 = coefficients
                    .iter()
                    .map(|(k, v)| (*k, v.into_inner()))
                    .collect();
                let weight = FlipWeight::Gaussian(GaussianFlip::Obs {
                    coefficients: coeffs_f64,
                    constant: *constant,
                    observed,
                });
                let CompileMode::Sample(sample_state) = mode else {
                    unreachable!()
                };
                let outcome = self.state.with_callstack_index(2, |s| {
                    s.current_address_weighted(weight);
                    crate::inference::sampling::resolve_observation(sample_state, s)
                });
                pure_monad(self.bool_val(outcome), BddPtr::true_ptr())
            }
            PluckValue::Native(NativeVal::Float(f)) => {
                let result = self.bool_val((*f - observed).abs() < 1e-15);
                pure_monad(result, BddPtr::true_ptr())
            }
            PluckValue::Gamma { name, scale } => {
                if observed <= 0.0 {
                    return pure_monad(self.bool_val(false), BddPtr::true_ptr());
                }
                let weight = FlipWeight::Gamma(GammaFlip::RatePin {
                    name: *name,
                    value: observed,
                    scale: *scale,
                });
                let CompileMode::Sample(sample_state) = mode else {
                    unreachable!()
                };
                let outcome = self.state.with_callstack_index(2, |s| {
                    s.current_address_weighted(weight);
                    crate::inference::sampling::resolve_observation(sample_state, s)
                });
                pure_monad(self.bool_val(outcome), BddPtr::true_ptr())
            }
            PluckValue::Exponential { name, gamma, scale } => {
                if observed < 0.0 {
                    return pure_monad(self.bool_val(false), BddPtr::true_ptr());
                }
                let weight = FlipWeight::Gamma(GammaFlip::ExpEq {
                    gamma: *gamma,
                    draw: *name,
                    value: observed,
                    scale: *scale,
                });
                let CompileMode::Sample(sample_state) = mode else {
                    unreachable!()
                };
                let outcome = self.state.with_callstack_index(2, |s| {
                    s.current_address_weighted(weight);
                    crate::inference::sampling::resolve_observation(sample_state, s)
                });
                pure_monad(self.bool_val(outcome), BddPtr::true_ptr())
            }
            PluckValue::Poisson { .. } => {
                panic!("close: Poisson draws are integer-valued; use native_eq")
            }
            _ => panic!(
                "close: expr must be a GaussianExpr, Gamma, Exponential \
                 or float, got {:?}",
                expr_val
            ),
        }
    }


    pub fn evaluate_thunk(&mut self, val: &PluckVal, mode: &mut CompileMode<'_>) -> GuardedWorlds {
        let path_condition = mode.path_condition();
        if path_condition.is_false() {
            return false_path_condition_worlds();
        }

        if let (CompileMode::Sample(state), PluckValue::Thunk(_)) = (&*mode, val.as_ref()) {
            if let Some(worlds) = state.check_cache(val) {
                return worlds;
            }
        }

        let worlds = match val.as_ref() {
            PluckValue::Thunk(thunk_data) => self.evaluate_single_thunk(thunk_data, mode),
            PluckValue::ThunkUnion(union_data) => self.evaluate_thunk_union(union_data, mode),
            _ => return pure_monad(val.clone(), path_condition),
        };

        if let CompileMode::Sample(state) = &mut *mode {
            state.set_cache(val, &worlds);
        }
        worlds
    }

    fn evaluate_single_thunk_no_cache(
        &mut self,
        thunk_data: &ThunkData,
        mode: &mut CompileMode<'_>,
    ) -> GuardedWorlds {
        let old_callstack = std::mem::replace(
            &mut self.state.callstack,
            super::state::Callstack::from_vec(thunk_data.callstack.clone()),
        );

        let result = self.traced_compile(
            &thunk_data.expr,
            &thunk_data.env,
            mode,
            thunk_data.strict_order_index,
        );

        self.state.callstack = old_callstack;
        result
    }

    fn evaluate_single_thunk(
        &mut self,
        thunk_data: &ThunkData,
        mode: &mut CompileMode<'_>,
    ) -> GuardedWorlds {
        if matches!(mode, CompileMode::Sample(_)) {
            return self.evaluate_single_thunk_no_cache(thunk_data, mode);
        }

        let path_condition = mode.path_condition();
        {
            let cache = thunk_data.cache.borrow();
            if let Some((cached_worlds, cached_used_info)) = cache.first() {
                let builder = self.state.bdd();
                let neg_pc = builder.negate(path_condition);
                let implies = builder.or(neg_pc, *cached_used_info);
                if implies.is_true() {
                    self.state.stats.num_cache_hits += 1;
                    return (cached_worlds.clone(), *cached_used_info);
                }
            }
        }

        self.state.stats.num_cache_misses += 1;

        let is_empty = thunk_data.cache.borrow().is_empty();
        if is_empty {
            self.state.stats.num_thunk_first_eval += 1;
            let result = self.evaluate_single_thunk_no_cache(thunk_data, mode);
            thunk_data.cache.borrow_mut().push(result.clone());
            return result;
        }

        self.state.stats.num_thunk_widen += 1;
        let (cached_worlds, cache_guard) = {
            let cache = thunk_data.cache.borrow();
            cache[0].clone()
        };

        let builder = self.state.bdd();
        let neg_cache_guard = builder.negate(cache_guard);
        let inner_path_condition = builder.and(path_condition, neg_cache_guard);

        if inner_path_condition.is_false() {
            self.state.stats.num_cache_hits += 1;
            return (cached_worlds, cache_guard);
        }

        let mut inner_mode = CompileMode::KC {
            path_condition: inner_path_condition,
        };
        let (miss_worlds, miss_used) =
            self.evaluate_single_thunk_no_cache(thunk_data, &mut inner_mode);

        let nested_worlds = vec![
            ((cached_worlds, cache_guard), cache_guard),
            ((miss_worlds, miss_used), neg_cache_guard),
        ];
        let result = join_monad(nested_worlds, BddPtr::true_ptr(), self);

        thunk_data.cache.borrow_mut()[0] = result.clone();

        result
    }

    fn evaluate_thunk_union_no_cache(
        &mut self,
        union_data: &ThunkUnionData,
        mode: &mut CompileMode<'_>,
    ) -> GuardedWorlds {
        let path_condition = mode.path_condition();
        let nested_worlds_list: Vec<(PluckVal, BddPtr<'static>)> = union_data.thunks.clone();
        let pre_worlds: GuardedWorlds = (nested_worlds_list, BddPtr::true_ptr());

        bind_monad(
            pre_worlds,
            path_condition,
            self,
            |thunk_val, _pc, ctx, mode| {
                ctx.evaluate_thunk(&thunk_val, mode)
            },
        )
    }

    fn evaluate_thunk_union(
        &mut self,
        union_data: &ThunkUnionData,
        mode: &mut CompileMode<'_>,
    ) -> GuardedWorlds {
        self.state.stats.num_tu_eval += 1;
        if matches!(mode, CompileMode::Sample(_)) {
            let chosen = {
                let CompileMode::Sample(state) = &mut *mode else {
                    unreachable!()
                };
                crate::inference::sampling::select_world(
                    union_data.thunks.clone(),
                    state,
                    &*self.state,
                )
            };
            return self.evaluate_thunk(&chosen, mode);
        }
        self.evaluate_thunk_union_no_cache(union_data, mode)
    }

    pub fn infer_full_distribution(&mut self, initial_results: Vec<World>) -> Vec<World> {
        let mut queue: Vec<World> = initial_results;
        let mut resolved: Vec<World> = Vec::new();

        while let Some((current_val, current_bdd)) = queue.pop() {
            if current_bdd.is_false() {
                continue;
            }

            if let Some(thunk_path) = find_first_thunk(&current_val) {
                let thunk = get_value_at_path(&current_val, &thunk_path);

                let mut inner_mode = CompileMode::KC {
                    path_condition: current_bdd,
                };
                let (sub_results, _used_info) = self.evaluate_thunk(&thunk, &mut inner_mode);

                for (sub_val, sub_bdd) in sub_results {
                    let builder = self.state.bdd();
                    let new_bdd = builder.and(current_bdd, sub_bdd);
                    if !new_bdd.is_false() {
                        let new_val = replace_at_path(&current_val, &thunk_path, &sub_val);
                        queue.push((new_val, new_bdd));
                    }
                }
                continue;
            }

            if let Some(int_dist_path) = find_first_non_const_int_dist(&current_val) {
                let int_dist_val = get_value_at_path(&current_val, &int_dist_path);
                if let PluckValue::IntDist { bits } = int_dist_val.as_ref() {
                    let int_dist = IntDist::new(bits.clone());
                    let expanded = enumerate_int_dist(&int_dist, current_bdd, self.state.bdd());
                    for (int_value, world_bdd) in expanded {
                        if !world_bdd.is_false() {
                            let const_bits: Vec<BddPtr<'static>> = (0..bits.len())
                                .map(|i| {
                                    if (int_value >> i) & 1 == 1 {
                                        BddPtr::true_ptr()
                                    } else {
                                        BddPtr::false_ptr()
                                    }
                                })
                                .collect();
                            let const_int_dist = mk_int_dist(const_bits);
                            let new_val =
                                replace_at_path(&current_val, &int_dist_path, &const_int_dist);
                            queue.push((new_val, world_bdd));
                        }
                    }
                    continue;
                }
            }

            resolved.push((current_val, current_bdd));
        }

        resolved.reverse();
        resolved
    }
}

fn encode_range(
    lo_val: i64,
    hi_val: i64,
    guard: BddPtr<'static>,
    bits: &mut Vec<BddPtr<'static>>,
    width: usize,
    state: &mut LazyKCState,
    depth_idx: i32,
) {
    if guard.is_false() {
        return;
    }
    if lo_val == hi_val {
        let builder = state.bdd();
        for (i, bit) in bits.iter_mut().enumerate().take(width) {
            let bit_set = ((lo_val as u64) >> i) & 1 == 1;
            let bit_bdd = if bit_set {
                BddPtr::true_ptr()
            } else {
                BddPtr::false_ptr()
            };
            let contribution = builder.and(bit_bdd, guard);
            *bit = builder.or(*bit, contribution);
        }
        return;
    }
    let mid = lo_val + (hi_val - lo_val) / 2;
    let lower_size = (mid - lo_val + 1) as f64;
    let upper_size = (hi_val - mid) as f64;
    let p = lower_size / (lower_size + upper_size);

    let addr = state.with_callstack_index(depth_idx, |s| s.current_address(p));

    let builder = state.bdd();
    let lo_guard = builder.and(guard, addr);
    let neg_addr = builder.negate(addr);
    let hi_guard = builder.and(guard, neg_addr);

    encode_range(lo_val, mid, lo_guard, bits, width, state, depth_idx + 1);
    encode_range(mid + 1, hi_val, hi_guard, bits, width, state, depth_idx + 1);
}

fn stick_betas(fs: &[f64]) -> Vec<f64> {
    let n = fs.len();
    let mut remaining = 1.0;
    let mut out = Vec::with_capacity(n.saturating_sub(1));
    for &f in fs.iter().take(n.saturating_sub(1)) {
        if remaining <= 0.0 {
            out.push(0.0);
        } else {
            out.push((f / remaining).clamp(0.0, 1.0));
        }
        remaining -= f;
    }
    out
}

fn allocate_one_hot_vars<F>(
    n: usize,
    mut weight_fn: F,
    state: &mut LazyKCState,
) -> Vec<BddPtr<'static>>
where
    F: FnMut(usize) -> FlipWeight,
{
    (0..n)
        .map(|i| state.with_callstack_index(i as i32, |s| s.current_address_weighted(weight_fn(i))))
        .collect()
}

fn build_stick_breaking_guards(
    vars: &[BddPtr<'static>],
    state: &LazyKCState,
) -> Vec<BddPtr<'static>> {
    let builder = state.bdd();
    let n_sticks = vars.len();
    let n_categories = n_sticks + 1;

    let mut guards = Vec::with_capacity(n_categories);
    let mut prefix = BddPtr::true_ptr();
    for &var in vars.iter() {
        guards.push(builder.and(prefix, var));
        prefix = builder.and(prefix, builder.negate(var));
    }
    guards.push(prefix);
    guards
}

fn make_thunk(expr: &PExpr, env: Env, strict_order_index: i32, state: &LazyKCState) -> PluckVal {
    if let PExpr::Var { name } = expr {
        if let Some(val) = env_lookup(&env, *name) {
            if matches!(val.as_ref(), PluckValue::Thunk(_)) {
                return val;
            }
        }
    }
    mk_thunk(
        Rc::new(expr.clone()),
        env,
        strict_order_index,
        state.callstack.to_vec(),
    )
}

fn force_entry_to_affine(
    ctx: &mut CompilerCtx,
    entry: &PluckVal,
    pc: BddPtr<'static>,
) -> (
    f64,
    std::collections::BTreeMap<u64, ordered_float::NotNan<f64>>,
    BddPtr<'static>,
) {
    let (resolved, used): (PluckVal, BddPtr<'static>) = match entry.as_ref() {
        PluckValue::Thunk(_) | PluckValue::ThunkUnion(_) => {
            let mut mode = CompileMode::KC { path_condition: pc };
            let (mut worlds, used) = ctx.evaluate_thunk(entry, &mut mode);
            let builder = ctx.state.bdd();
            worlds.retain(|w| !builder.and(w.1, pc).is_false());
            assert!(
                !worlds.is_empty(),
                "matrix entry produced no live worlds under the current path \
                 condition (program error along this path)"
            );
            assert_eq!(
                worlds.len(),
                1,
                "matrix entry produced {} live worlds when forced; multi-world matrix entries are not supported",
                worlds.len()
            );
            let (v, guard) = worlds.pop().unwrap();
            (v, builder.and(used, guard))
        }
        _ => (entry.clone(), BddPtr::true_ptr()),
    };
    match resolved.as_ref() {
        PluckValue::Native(NativeVal::Float(f)) => (*f, std::collections::BTreeMap::new(), used),
        PluckValue::Native(NativeVal::Int(i)) => {
            (*i as f64, std::collections::BTreeMap::new(), used)
        }
        PluckValue::GaussianExpr {
            constant,
            coefficients,
        } => (*constant, coefficients.clone(), used),
        _ => panic!(
            "matrix entry forced to non-numeric / non-Gaussian value: {:?}",
            resolved
        ),
    }
}

fn affine_binop(
    op: &FloatOp,
    (ca, ma): &(
        f64,
        std::collections::BTreeMap<u64, ordered_float::NotNan<f64>>,
    ),
    (cb, mb): &(
        f64,
        std::collections::BTreeMap<u64, ordered_float::NotNan<f64>>,
    ),
) -> (
    f64,
    std::collections::BTreeMap<u64, ordered_float::NotNan<f64>>,
) {
    match op {
        FloatOp::Add => (ca + cb, merge_coefficients(ma, mb, |x, y| x + y)),
        FloatOp::Sub => (
            ca - cb,
            merge_coefficients(ma, &scale_coefficients(mb, -1.0), |x, y| x + y),
        ),
        FloatOp::Mul => {
            if !ma.is_empty() && !mb.is_empty() {
                panic!("*.: cannot multiply two non-trivial Gaussian terms (would be nonlinear)");
            }
            if ma.is_empty() {
                (ca * cb, scale_coefficients(mb, *ca))
            } else {
                (ca * cb, scale_coefficients(ma, *cb))
            }
        }
        FloatOp::Div => {
            if !mb.is_empty() {
                panic!("/.: divisor must be a constant (cannot divide by a Gaussian term)");
            }
            assert!(*cb != 0.0, "/.: division by zero");
            (ca / cb, scale_coefficients(ma, 1.0 / cb))
        }
    }
}

fn pack_affine(c: f64, m: std::collections::BTreeMap<u64, ordered_float::NotNan<f64>>) -> PluckVal {
    if m.is_empty() {
        mk_native_float(c)
    } else {
        mk_gaussian_expr(c, m)
    }
}

fn try_scale_gamma(op: FloatOp, a: &PluckVal, b: &PluckVal) -> Option<PluckVal> {
    let is_draw = |v: &PluckVal| {
        matches!(
            v.as_ref(),
            PluckValue::Exponential { .. } | PluckValue::Poisson { .. }
        )
    };
    if is_draw(a) || is_draw(b) {
        panic!(
            "float arithmetic on an Exponential/Poisson draw is not supported; \
             scale the gamma rate instead, e.g. (exponential (*. g 5.0))"
        );
    }

    let gamma_parts = |v: &PluckVal| match v.as_ref() {
        PluckValue::Gamma { name, scale } => Some((*name, *scale)),
        _ => None,
    };
    let native_const = |v: &PluckVal| match v.as_ref() {
        PluckValue::Native(NativeVal::Float(f)) => Some(*f),
        _ => None,
    };

    if gamma_parts(a).is_none() && gamma_parts(b).is_none() {
        return None;
    }

    match op {
        FloatOp::Mul => {
            let (name, scale, c) = match (gamma_parts(a), native_const(b)) {
                (Some((n, s)), Some(c)) => (n, s, c),
                _ => match (native_const(a), gamma_parts(b)) {
                    (Some(c), Some((n, s))) => (n, s, c),
                    _ => panic!(
                        "a Gamma variable can only be scaled by a numeric constant \
                         (use a float literal, e.g. (*. g 5.0))"
                    ),
                },
            };
            let new_scale = scale * c;
            assert!(
                new_scale > 0.0,
                "scaling a Gamma must yield a positive scale, got {new_scale} \
                 (scale {scale} * {c})"
            );
            Some(mk_gamma_scaled(name, new_scale))
        }
        FloatOp::Div => match (gamma_parts(a), native_const(b)) {
            (Some((name, scale)), Some(c)) => {
                let new_scale = scale / c;
                assert!(
                    new_scale > 0.0,
                    "dividing a Gamma must yield a positive scale, got {new_scale} \
                     (scale {scale} / {c})"
                );
                Some(mk_gamma_scaled(name, new_scale))
            }
            _ => panic!(
                "a Gamma variable can only be divided by a numeric constant \
                 (constant / Gamma is not supported)"
            ),
        },
        FloatOp::Add | FloatOp::Sub => panic!(
            "a Gamma variable supports only scaling (`*.` / `/.`); \
             addition/subtraction of a Gamma is not supported"
        ),
    }
}

fn apply_float_op(
    op: &FloatOp,
    a: &PluckVal,
    b: &PluckVal,
    ctx: &mut CompilerCtx,
    pc: BddPtr<'static>,
) -> (PluckVal, BddPtr<'static>) {
    let a_is_mat = matches!(a.as_ref(), PluckValue::FloatMatrix { .. });
    let b_is_mat = matches!(b.as_ref(), PluckValue::FloatMatrix { .. });

    let builder = ctx.state.bdd();
    let mut used = BddPtr::true_ptr();
    let mut fold_cell = |ctx: &mut CompilerCtx, x: &PluckVal, y: &PluckVal| {
        let (v, u) = cell_op(op, x, y, ctx, pc);
        used = builder.and(used, u);
        v
    };

    let val = match (a_is_mat, b_is_mat) {
        (false, false) => fold_cell(ctx, a, b),
        (true, true) => {
            let (ae, ash) = match a.as_ref() {
                PluckValue::FloatMatrix { entries, shape } => (entries.clone(), shape.clone()),
                _ => unreachable!(),
            };
            let (be, bsh) = match b.as_ref() {
                PluckValue::FloatMatrix { entries, shape } => (entries.clone(), shape.clone()),
                _ => unreachable!(),
            };
            assert_eq!(
                ash, bsh,
                "elementwise float op: shape mismatch {:?} vs {:?}",
                ash, bsh
            );
            let out: Vec<PluckVal> = ae
                .iter()
                .zip(be.iter())
                .map(|(ca, cb)| fold_cell(ctx, ca, cb))
                .collect();
            mk_float_matrix(out, ash)
        }
        (true, false) => {
            let (ae, ash) = match a.as_ref() {
                PluckValue::FloatMatrix { entries, shape } => (entries.clone(), shape.clone()),
                _ => unreachable!(),
            };
            let out: Vec<PluckVal> = ae.iter().map(|ca| fold_cell(ctx, ca, b)).collect();
            mk_float_matrix(out, ash)
        }
        (false, true) => {
            let (be, bsh) = match b.as_ref() {
                PluckValue::FloatMatrix { entries, shape } => (entries.clone(), shape.clone()),
                _ => unreachable!(),
            };
            let out: Vec<PluckVal> = be.iter().map(|cb| fold_cell(ctx, a, cb)).collect();
            mk_float_matrix(out, bsh)
        }
    };
    (val, used)
}

fn cell_op_affine(
    op: &FloatOp,
    a: &PluckVal,
    b: &PluckVal,
    ctx: &mut CompilerCtx,
    pc: BddPtr<'static>,
) -> (
    f64,
    std::collections::BTreeMap<u64, ordered_float::NotNan<f64>>,
    BddPtr<'static>,
) {
    let a_known = entry_constant(a.as_ref());
    let b_known = entry_constant(b.as_ref());
    let empty = std::collections::BTreeMap::new();
    let true_ptr = BddPtr::true_ptr();
    match (op, a_known, b_known) {
        (FloatOp::Add, Some(0.0), _) => return force_entry_to_affine(ctx, b, pc),
        (FloatOp::Add, _, Some(0.0)) => return force_entry_to_affine(ctx, a, pc),
        (FloatOp::Sub, _, Some(0.0)) => return force_entry_to_affine(ctx, a, pc),
        (FloatOp::Mul, Some(0.0), _) | (FloatOp::Mul, _, Some(0.0)) => {
            return (0.0, empty, true_ptr)
        }
        (FloatOp::Mul, Some(1.0), _) => return force_entry_to_affine(ctx, b, pc),
        (FloatOp::Mul, _, Some(1.0)) => return force_entry_to_affine(ctx, a, pc),
        (FloatOp::Div, Some(0.0), _) => return (0.0, empty, true_ptr),
        (FloatOp::Div, _, Some(1.0)) => return force_entry_to_affine(ctx, a, pc),
        _ => {}
    }
    let (ca, ma, ua) = force_entry_to_affine(ctx, a, pc);
    let (cb, mb, ub) = force_entry_to_affine(ctx, b, pc);
    let (c, m) = affine_binop(op, &(ca, ma), &(cb, mb));
    (c, m, ctx.state.bdd().and(ua, ub))
}

fn cell_op(
    op: &FloatOp,
    a: &PluckVal,
    b: &PluckVal,
    ctx: &mut CompilerCtx,
    pc: BddPtr<'static>,
) -> (PluckVal, BddPtr<'static>) {
    let (c, m, used) = cell_op_affine(op, a, b, ctx, pc);
    (pack_affine(c, m), used)
}

fn expect_float_matrix<'a>(val: &'a PluckVal, context: &str) -> (&'a [PluckVal], &'a [usize]) {
    match val.as_ref() {
        PluckValue::FloatMatrix { entries, shape } => (entries, shape),
        _ => panic!("{}: expected a FloatMatrix, got {:?}", context, val),
    }
}

fn lift_for_matmul(shape: &[usize], is_left: bool) -> (usize, usize, bool) {
    match shape {
        [n] => {
            if is_left {
                (1, *n, true)
            } else {
                (*n, 1, true)
            }
        }
        [r, c] => (*r, *c, false),
        _ => panic!("@: rank > 2 not supported, got shape {:?}", shape),
    }
}

fn squeeze_matmul_result(
    out_entries: Vec<PluckVal>,
    a_rows: usize,
    b_cols: usize,
    a_was_1d: bool,
    b_was_1d: bool,
) -> PluckVal {
    match (a_was_1d, b_was_1d) {
        (true, true) => {
            debug_assert_eq!(out_entries.len(), 1);
            out_entries.into_iter().next().unwrap()
        }
        (true, false) => mk_float_matrix(out_entries, vec![b_cols]),
        (false, true) => mk_float_matrix(out_entries, vec![a_rows]),
        (false, false) => mk_float_matrix(out_entries, vec![a_rows, b_cols]),
    }
}

fn merge_coefficients(
    a: &std::collections::BTreeMap<u64, ordered_float::NotNan<f64>>,
    b: &std::collections::BTreeMap<u64, ordered_float::NotNan<f64>>,
    op: fn(f64, f64) -> f64,
) -> std::collections::BTreeMap<u64, ordered_float::NotNan<f64>> {
    let mut result = a.clone();
    for (&k, v) in b {
        let entry = result
            .entry(k)
            .or_insert_with(|| ordered_float::NotNan::new(0.0).unwrap());
        let combined = op(entry.into_inner(), v.into_inner());
        *entry = ordered_float::NotNan::new(combined).expect("merge_coefficients produced NaN");
        if entry.into_inner().abs() < 1e-30 {
            result.remove(&k);
        }
    }
    result
}

fn scale_coefficients(
    m: &std::collections::BTreeMap<u64, ordered_float::NotNan<f64>>,
    s: f64,
) -> std::collections::BTreeMap<u64, ordered_float::NotNan<f64>> {
    m.iter()
        .map(|(&k, v)| {
            (
                k,
                ordered_float::NotNan::new(v.into_inner() * s).expect("scale_coefficients NaN"),
            )
        })
        .collect()
}

fn as_float_or_panic(val: &PluckValue, context: &str) -> f64 {
    match val {
        PluckValue::Native(NativeVal::Float(f)) => *f,
        PluckValue::Native(NativeVal::Int(i)) => *i as f64,
        _ => panic!("{}, got {:?}", context, val),
    }
}

fn extract_dirichlet_value(val: &PluckValue) -> Box<[ordered_float::NotNan<f64>]> {
    let push = |f: f64, out: &mut Vec<ordered_float::NotNan<f64>>| {
        out.push(
            ordered_float::NotNan::new(f).expect("dirichlet_eq: probability component is NaN"),
        );
    };
    let mut out: Vec<ordered_float::NotNan<f64>> = Vec::new();
    match val {
        PluckValue::FloatMatrix { entries, shape } => {
            assert!(
                shape.len() == 1,
                "dirichlet_eq: value must be a 1-D FloatMatrix, got shape {:?}",
                shape
            );
            for e in entries {
                match e.as_ref() {
                    PluckValue::Native(NativeVal::Float(f)) => push(*f, &mut out),
                    PluckValue::Native(NativeVal::Int(i)) => push(*i as f64, &mut out),
                    _ => panic!(
                        "dirichlet_eq: FloatMatrix entry must be a Native float, got {:?}",
                        e
                    ),
                }
            }
        }
        PluckValue::Value { args, .. } => {
            for a in args {
                match a.as_ref() {
                    PluckValue::Native(NativeVal::Float(f)) => push(*f, &mut out),
                    PluckValue::Native(NativeVal::Int(i)) => push(*i as f64, &mut out),
                    _ => panic!(
                        "dirichlet_eq: constructor arg must be a Native float, got {:?}",
                        a
                    ),
                }
            }
        }
        _ => panic!(
            "dirichlet_eq: value must be a FloatMatrix or constructor of floats, got {:?}",
            val
        ),
    }
    out.into_boxed_slice()
}

fn as_int_or_panic(val: &PluckValue, context: &str) -> i64 {
    match val {
        PluckValue::Native(NativeVal::Int(i)) => *i,
        _ => panic!("{} must be an int, got {:?}", context, val),
    }
}

fn as_int_dist_bits<'a>(val: &'a PluckValue, context: &str) -> &'a [BddPtr<'static>] {
    match val {
        PluckValue::IntDist { bits } => bits,
        _ => panic!("{} must be an IntDist, got {:?}", context, val),
    }
}

fn flip_weight_from(prob_val: &PluckValue) -> FlipWeight {
    prob_val.flip_weight().unwrap_or_else(|| {
        panic!(
            "flip probability must be a float or Probability, got {:?}",
            prob_val
        )
    })
}

fn find_first_thunk(val: &PluckVal) -> Option<Vec<usize>> {
    fn find_inner(val: &PluckVal, path: &mut Vec<usize>) -> bool {
        match val.as_ref() {
            PluckValue::Thunk(_) | PluckValue::ThunkUnion(_) => true,
            PluckValue::Value { args, .. } => {
                for (i, arg) in args.iter().enumerate() {
                    path.push(i);
                    if find_inner(arg, path) {
                        return true;
                    }
                    path.pop();
                }
                false
            }
            _ => false,
        }
    }
    let mut path = Vec::new();
    if find_inner(val, &mut path) {
        Some(path)
    } else {
        None
    }
}

fn find_first_non_const_int_dist(val: &PluckVal) -> Option<Vec<usize>> {
    fn find_inner(val: &PluckVal, path: &mut Vec<usize>) -> bool {
        match val.as_ref() {
            PluckValue::IntDist { bits } => !bits.iter().all(|b| b.is_true() || b.is_false()),
            PluckValue::Value { args, .. } => {
                for (i, arg) in args.iter().enumerate() {
                    path.push(i);
                    if find_inner(arg, path) {
                        return true;
                    }
                    path.pop();
                }
                false
            }
            _ => false,
        }
    }
    let mut path = Vec::new();
    if find_inner(val, &mut path) {
        Some(path)
    } else {
        None
    }
}

fn get_value_at_path(val: &PluckVal, path: &[usize]) -> PluckVal {
    if path.is_empty() {
        return val.clone();
    }
    match val.as_ref() {
        PluckValue::Value { args, .. } => get_value_at_path(&args[path[0]], &path[1..]),
        _ => val.clone(),
    }
}

fn replace_at_path(val: &PluckVal, path: &[usize], new_val: &PluckVal) -> PluckVal {
    if path.is_empty() {
        return new_val.clone();
    }
    match val.as_ref() {
        PluckValue::Value { constructor, args } => {
            let mut new_args = args.clone();
            if path.len() == 1 {
                new_args[path[0]] = new_val.clone();
            } else {
                new_args[path[0]] = replace_at_path(&args[path[0]], &path[1..], new_val);
            }
            mk_value(*constructor, new_args)
        }
        _ => val.clone(),
    }
}

#[cfg(test)]
mod try_scale_gamma_tests {
    use super::*;

    fn assert_scaled_gamma(v: &PluckVal, expect_name: u64, expect_scale: f64) {
        match v.as_ref() {
            PluckValue::Gamma { name, scale } => {
                assert_eq!(*name, expect_name);
                assert_eq!(*scale, expect_scale);
            }
            other => panic!("expected scaled Gamma, got {other:?}"),
        }
    }

    #[test]
    fn mul_by_constant_scales_either_order() {
        let g = mk_gamma(7);
        let r = try_scale_gamma(FloatOp::Mul, &g, &mk_native_float(5.0)).unwrap();
        assert_scaled_gamma(&r, 7, 5.0);
        let r = try_scale_gamma(FloatOp::Mul, &mk_native_float(5.0), &g).unwrap();
        assert_scaled_gamma(&r, 7, 5.0);
    }

    #[test]
    fn scale_composes() {
        let g2 = mk_gamma_scaled(7, 2.0);
        let r = try_scale_gamma(FloatOp::Mul, &g2, &mk_native_float(3.0)).unwrap();
        assert_scaled_gamma(&r, 7, 6.0);
    }

    #[test]
    fn div_by_constant_scales() {
        let g = mk_gamma(7);
        let r = try_scale_gamma(FloatOp::Div, &g, &mk_native_float(2.0)).unwrap();
        assert_scaled_gamma(&r, 7, 0.5);
    }

    #[test]
    fn no_gamma_falls_through() {
        let r = try_scale_gamma(FloatOp::Mul, &mk_native_float(2.0), &mk_native_float(3.0));
        assert!(r.is_none());
    }

    #[test]
    #[should_panic(expected = "addition/subtraction of a Gamma")]
    fn add_of_gamma_panics() {
        try_scale_gamma(FloatOp::Add, &mk_gamma(7), &mk_native_float(1.0));
    }

    #[test]
    #[should_panic(expected = "can only be scaled by a numeric constant")]
    fn gamma_times_gamma_panics() {
        try_scale_gamma(FloatOp::Mul, &mk_gamma(7), &mk_gamma(8));
    }

    #[test]
    #[should_panic(expected = "Exponential/Poisson draw")]
    fn scaling_a_draw_panics() {
        let draw = mk_exponential(7, 99, 1.0);
        try_scale_gamma(FloatOp::Mul, &draw, &mk_native_float(5.0));
    }

    #[test]
    #[should_panic(expected = "positive scale")]
    fn scaling_by_zero_panics() {
        try_scale_gamma(FloatOp::Mul, &mk_gamma(7), &mk_native_float(0.0));
    }

    #[test]
    #[should_panic(expected = "constant / Gamma is not supported")]
    fn constant_over_gamma_panics() {
        try_scale_gamma(FloatOp::Div, &mk_native_float(5.0), &mk_gamma(7));
    }
}
