use std::rc::Rc;

use rsdd::builder::bdd::RobddBuilder;
use rsdd::builder::cache::AllIteTable;
use rsdd::builder::BottomUpBuilder;
use rsdd::repr::{BddPtr, DDNNFPtr};

use super::compile::CompileMode;
use super::ctx::CompilerCtx;
use super::state::{GuardedWorlds, World};
use crate::language::types::Symbol;
use crate::language::values::*;

pub fn pure_monad(val: PluckVal, _path_condition: BddPtr<'static>) -> GuardedWorlds {
    (vec![(val, BddPtr::true_ptr())], BddPtr::true_ptr())
}

pub fn pure_monad_used(val: PluckVal, used: BddPtr<'static>) -> GuardedWorlds {
    (vec![(val, BddPtr::true_ptr())], used)
}

pub fn program_error_worlds() -> GuardedWorlds {
    (Vec::new(), BddPtr::true_ptr())
}

pub fn inference_error_worlds() -> GuardedWorlds {
    (Vec::new(), BddPtr::true_ptr())
}

pub fn false_path_condition_worlds() -> GuardedWorlds {
    (Vec::new(), BddPtr::false_ptr())
}

pub fn condition_worlds_on_pc(
    worlds: GuardedWorlds,
    pc: BddPtr<'static>,
    builder: &'static RobddBuilder<'static, AllIteTable<BddPtr<'static>>>,
) -> GuardedWorlds {
    if pc.is_true() {
        return worlds;
    }
    let (world_list, used_info) = worlds;
    let new_worlds: Vec<World> = world_list
        .into_iter()
        .map(|(val, guard)| {
            let new_val = condition_value_int_dists(&val, pc, builder);
            (new_val, guard)
        })
        .collect();
    let new_used_info = builder.and(used_info, pc);
    (new_worlds, new_used_info)
}

fn condition_value_int_dists(
    val: &PluckVal,
    pc: BddPtr<'static>,
    builder: &'static RobddBuilder<'static, AllIteTable<BddPtr<'static>>>,
) -> PluckVal {
    match val.as_ref() {
        PluckValue::IntDist { bits } => {
            let conditioned_bits: Vec<BddPtr<'static>> =
                bits.iter().map(|b| builder.and(*b, pc)).collect();
            mk_int_dist(conditioned_bits)
        }
        PluckValue::Value { constructor, args } => {
            let new_args: Vec<PluckVal> = args
                .iter()
                .map(|a| condition_value_int_dists(a, pc, builder))
                .collect();
            if new_args
                .iter()
                .zip(args.iter())
                .all(|(n, o)| Rc::ptr_eq(n, o))
            {
                val.clone()
            } else {
                mk_value(*constructor, new_args)
            }
        }
        _ => val.clone(),
    }
}

pub fn bind_monad<F>(
    pre_worlds: GuardedWorlds,
    path_condition: BddPtr<'static>,
    ctx: &mut CompilerCtx,
    mut cont: F,
) -> GuardedWorlds
where
    F: FnMut(PluckVal, BddPtr<'static>, &mut CompilerCtx, &mut CompileMode<'_>) -> GuardedWorlds,
{
    let (pre_world_list, pre_used_info) = pre_worlds;

    let mut nested_worlds: Vec<(GuardedWorlds, BddPtr<'static>)> = Vec::new();

    for (pre_val, pre_guard) in pre_world_list {
        if ctx.state.hit_limit {
            return inference_error_worlds();
        }

        let builder = ctx.state.bdd();
        let inner_path_condition = builder.and(path_condition, pre_guard);

        if inner_path_condition.is_false() {
            ctx.state.stats.num_false_pc_exits += 1;
            nested_worlds.push((false_path_condition_worlds(), pre_guard));
            continue;
        }

        let mut inner_mode = CompileMode::KC {
            path_condition: inner_path_condition,
        };
        let post_worlds = cont(pre_val, inner_path_condition, ctx, &mut inner_mode);
        nested_worlds.push((post_worlds, pre_guard));
    }

    join_monad(nested_worlds, pre_used_info, ctx)
}

pub fn bind_compile<F>(
    pre_worlds: GuardedWorlds,
    path_condition: BddPtr<'static>,
    ctx: &mut CompilerCtx,
    mode: &mut CompileMode<'_>,
    mut cont: F,
) -> GuardedWorlds
where
    F: FnMut(PluckVal, BddPtr<'static>, &mut CompilerCtx, &mut CompileMode<'_>) -> GuardedWorlds,
{
    if matches!(mode, CompileMode::Sample(_)) {
        let (worlds, _) = pre_worlds;
        if worlds.is_empty() {
            if ctx.state.hit_limit {
                return inference_error_worlds();
            }
            panic!(
                "Sample mode: expression produced no worlds \
                 (program error along the sampled path)"
            );
        }
        assert_eq!(
            worlds.len(),
            1,
            "Sample mode: bind received {} pre-worlds; every Sample-mode \
             producer must yield exactly one world",
            worlds.len()
        );
        let val = worlds.into_iter().next().unwrap().0;
        return cont(val, BddPtr::true_ptr(), ctx, mode);
    }
    bind_monad(pre_worlds, path_condition, ctx, cont)
}

fn mk_thunk_union_flat(
    thunks: Vec<(PluckVal, BddPtr<'static>)>,
    builder: &'static rsdd::builder::bdd::RobddBuilder<
        'static,
        rsdd::builder::cache::AllIteTable<BddPtr<'static>>,
    >,
) -> PluckVal {
    let mut flat: Vec<(PluckVal, BddPtr<'static>)> = Vec::new();

    for (val, outer_guard) in thunks {
        if let PluckValue::ThunkUnion(union_data) = val.as_ref() {
            for (inner_val, inner_guard) in &union_data.thunks {
                let combined_guard = builder.and(outer_guard, *inner_guard);
                if combined_guard.is_false() {
                    continue;
                }
                let mut found = false;
                for (existing_val, existing_guard) in flat.iter_mut() {
                    if Rc::ptr_eq(existing_val, inner_val) {
                        *existing_guard = builder.or(*existing_guard, combined_guard);
                        found = true;
                        break;
                    }
                }
                if !found {
                    flat.push((inner_val.clone(), combined_guard));
                }
            }
        } else {
            let mut found = false;
            for (existing_val, existing_guard) in flat.iter_mut() {
                if Rc::ptr_eq(existing_val, &val) {
                    *existing_guard = builder.or(*existing_guard, outer_guard);
                    found = true;
                    break;
                }
            }
            if !found {
                flat.push((val, outer_guard));
            }
        }
    }

    if flat.len() == 1 {
        flat.into_iter().next().unwrap().0
    } else {
        mk_thunk_union(flat)
    }
}

pub fn join_monad(
    nested_worlds: Vec<(GuardedWorlds, BddPtr<'static>)>,
    pre_used_info: BddPtr<'static>,
    ctx: &mut CompilerCtx,
) -> GuardedWorlds {
    let builder = ctx.state.bdd();

    let mut used_information = pre_used_info;
    for ((_, used_info), pre_guard) in &nested_worlds {
        let neg_guard = builder.negate(*pre_guard);
        let implies = builder.or(neg_guard, *used_info);
        used_information = builder.and(used_information, implies);
    }

    let mut join_results: Vec<World> = Vec::new();
    let mut results_for_constructor: Vec<(Symbol, Vec<(PluckVal, BddPtr<'static>)>)> = Vec::new();

    for ((post_worlds, _), pre_guard) in nested_worlds {
        for (post_val, post_guard) in post_worlds {
            let pre_and_post = builder.and(post_guard, pre_guard);
            if pre_and_post.is_false() {
                continue;
            }

            match post_val.as_ref() {
                PluckValue::Value { constructor, .. } => {
                    let ctor = *constructor;
                    if let Some(group) =
                        results_for_constructor.iter_mut().find(|(c, _)| *c == ctor)
                    {
                        group.1.push((post_val, pre_and_post));
                    } else {
                        results_for_constructor.push((ctor, vec![(post_val, pre_and_post)]));
                    }
                }
                PluckValue::Closure(_) | PluckValue::Native(_) => {
                    let mut found = false;
                    for (existing_val, existing_guard) in join_results.iter_mut() {
                        if std::rc::Rc::ptr_eq(existing_val, &post_val)
                            || values_equal(existing_val, &post_val)
                        {
                            *existing_guard = builder.or(*existing_guard, pre_and_post);
                            found = true;
                            break;
                        }
                    }
                    if !found {
                        join_results.push((post_val, pre_and_post));
                    }
                }
                PluckValue::IntDist { bits } => {
                    let mut found = false;
                    for (existing_val, existing_guard) in join_results.iter_mut() {
                        if let PluckValue::IntDist {
                            bits: existing_bits,
                        } = existing_val.as_ref()
                        {
                            if existing_bits.len() == bits.len() {
                                let mut new_bits = Vec::with_capacity(bits.len());
                                for i in 0..bits.len() {
                                    let new_contrib = builder.and(bits[i], pre_and_post);
                                    new_bits.push(builder.or(existing_bits[i], new_contrib));
                                }
                                let new_guard = builder.or(*existing_guard, pre_and_post);
                                *existing_val = mk_int_dist(new_bits);
                                *existing_guard = new_guard;
                                found = true;
                                break;
                            }
                        }
                    }
                    if !found {
                        let guarded_bits: Vec<BddPtr<'static>> =
                            bits.iter().map(|b| builder.and(*b, pre_and_post)).collect();
                        join_results.push((mk_int_dist(guarded_bits), pre_and_post));
                    }
                }
                PluckValue::Thunk(_) | PluckValue::ThunkUnion(_) => {
                    join_results.push((post_val, pre_and_post));
                }
                PluckValue::FloatMatrix { .. } => {
                    let mut found = false;
                    for (existing_val, existing_guard) in join_results.iter_mut() {
                        if std::rc::Rc::ptr_eq(existing_val, &post_val) {
                            *existing_guard = builder.or(*existing_guard, pre_and_post);
                            found = true;
                            break;
                        }
                    }
                    if !found {
                        join_results.push((post_val, pre_and_post));
                    }
                }
                v if v.is_symbolic() => {
                    let mut found = false;
                    for (existing_val, existing_guard) in join_results.iter_mut() {
                        if values_equal(existing_val, &post_val) {
                            *existing_guard = builder.or(*existing_guard, pre_and_post);
                            found = true;
                            break;
                        }
                    }
                    if !found {
                        join_results.push((post_val, pre_and_post));
                    }
                }
                _ => unreachable!("join_monad: unhandled PluckValue variant"),
            }
        }
    }

    for (ctor, entries) in results_for_constructor {
        if entries.len() == 1 {
            let (val, guard) = entries.into_iter().next().unwrap();
            join_results.push((val, guard));
            continue;
        }

        let mut deduped: Vec<(PluckVal, BddPtr<'static>)> = Vec::new();
        for (val, guard) in &entries {
            let mut found = false;
            for (existing_val, existing_guard) in deduped.iter_mut() {
                if std::rc::Rc::ptr_eq(existing_val, val) || values_equal(existing_val, val) {
                    *existing_guard = builder.or(*existing_guard, *guard);
                    found = true;
                    break;
                }
            }
            if !found {
                deduped.push((val.clone(), *guard));
            }
        }

        if deduped.len() == 1 {
            let (val, guard) = deduped.into_iter().next().unwrap();
            join_results.push((val, guard));
            continue;
        }

        let mut overall_guard = BddPtr::false_ptr();
        let num_args = match deduped[0].0.as_ref() {
            PluckValue::Value { args, .. } => args.len(),
            _ => unreachable!(),
        };

        let mut thunks_of_arg: Vec<Vec<(PluckVal, BddPtr<'static>)>> = vec![Vec::new(); num_args];

        for (post_val, pre_and_post) in &deduped {
            overall_guard = builder.or(overall_guard, *pre_and_post);
            if let PluckValue::Value { args, .. } = post_val.as_ref() {
                for (i, arg) in args.iter().enumerate() {
                    thunks_of_arg[i].push((arg.clone(), *pre_and_post));
                }
            }
        }

        let merged_args: Vec<PluckVal> = thunks_of_arg
            .into_iter()
            .map(|thunks| {
                if thunks.len() == 1 {
                    thunks.into_iter().next().unwrap().0
                } else {
                    mk_thunk_union_flat(thunks, builder)
                }
            })
            .collect();

        join_results.push((mk_value(ctor, merged_args), overall_guard));
    }

    (join_results, used_information)
}

pub fn if_then_else_monad(
    val_true: PluckVal,
    val_false: PluckVal,
    condition: BddPtr<'static>,
    _path_condition: BddPtr<'static>,
    ctx: &mut CompilerCtx,
) -> GuardedWorlds {
    let builder = ctx.state.bdd();
    let neg_condition = builder.negate(condition);

    let mut worlds = Vec::new();
    if !condition.is_false() {
        worlds.push((val_true, condition));
    }
    if !neg_condition.is_false() {
        worlds.push((val_false, neg_condition));
    }
    (worlds, BddPtr::true_ptr())
}

pub fn categorical_monad(
    vals: Vec<PluckVal>,
    conditions: Vec<BddPtr<'static>>,
    _path_condition: BddPtr<'static>,
    _ctx: &mut CompilerCtx,
) -> GuardedWorlds {
    let mut worlds = Vec::new();
    for (val, condition) in vals.iter().zip(conditions) {
        if !condition.is_false() {
            worlds.push((val.clone(), condition));
        }
    }
    (worlds, BddPtr::true_ptr())
}

fn values_equal(a: &PluckVal, b: &PluckVal) -> bool {
    match (a.as_ref(), b.as_ref()) {
        (PluckValue::Native(a), PluckValue::Native(b)) => a == b,
        (
            PluckValue::Value {
                constructor: ca,
                args: aa,
            },
            PluckValue::Value {
                constructor: cb,
                args: ab,
            },
        ) => {
            ca == cb
                && aa.len() == ab.len()
                && aa.iter().zip(ab).all(|(x, y)| std::rc::Rc::ptr_eq(x, y))
        }
        (PluckValue::IntDist { bits: ba }, PluckValue::IntDist { bits: bb }) => {
            ba.len() == bb.len() && ba.iter().zip(bb).all(|(x, y)| *x == *y)
        }
        _ => a.symbolic_eq(b),
    }
}
