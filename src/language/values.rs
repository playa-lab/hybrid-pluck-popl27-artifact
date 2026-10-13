use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::rc::Rc;

use rsdd::repr::BddPtr;

use super::pexpr::{NativeVal, PExpr};
use super::types::Symbol;
use crate::inference::conjugate_pairs::{
    Assignment, BetaFlip, ContVarName, DirichletFlip, GaussianAffineExpr,
};
use crate::inference::lazy_kc::state::FlipWeight;
use crate::utils::formatting::{subscript_str, NamingScheme};

pub type PluckVal = Rc<PluckValue>;

pub type ThunkCache = RefCell<Vec<(Vec<(PluckVal, BddPtr<'static>)>, BddPtr<'static>)>>;

#[derive(Debug, Clone)]
pub enum PluckValue {
    Value {
        constructor: Symbol,
        args: Vec<PluckVal>,
    },
    Native(NativeVal),
    Closure(ClosureData),
    IntDist { bits: Vec<BddPtr<'static>> },
    Thunk(ThunkData),
    ThunkUnion(ThunkUnionData),
    Probability(u64),
    DirichletVector { name: u64, categories: usize },
    DirichletProbability {
        name: u64,
        index: usize,
        categories: usize,
    },
    GaussianExpr {
        constant: f64,
        coefficients: BTreeMap<u64, ordered_float::NotNan<f64>>,
    },
    Gamma { name: u64, scale: f64 },
    Exponential { name: u64, gamma: u64, scale: f64 },
    Poisson { name: u64, gamma: u64, scale: f64 },
    FloatMatrix {
        entries: Vec<PluckVal>,
        shape: Vec<usize>,
    },
}

#[derive(Clone)]
pub struct ThunkData {
    pub expr: Rc<PExpr>,
    pub env: Env,
    pub strict_order_index: i32,
    pub callstack: Vec<i32>,
    pub cache: ThunkCache,
}

impl fmt::Debug for ThunkData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Thunk(callstack={:?}, cache_len={})",
            self.callstack,
            self.cache.borrow().len()
        )
    }
}

#[derive(Clone)]
pub struct ThunkUnionData {
    pub thunks: Vec<(PluckVal, BddPtr<'static>)>,
    pub cache: ThunkCache,
}

impl fmt::Debug for ThunkUnionData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ThunkUnion(len={})", self.thunks.len())
    }
}

#[derive(Clone)]
pub struct ClosureData {
    pub var: Symbol,
    pub body: Rc<PExpr>,
    pub env: Env,
    pub self_ref: RefCell<Option<PluckVal>>,
    pub rec_name: Option<Symbol>,
    pub arg_name: Option<Symbol>,
}

impl fmt::Debug for ClosureData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Closure(var={:?}, body=..., env_len={})",
            self.var,
            env_len(&self.env)
        )
    }
}

pub type Env = Rc<EnvInner>;

#[derive(Debug, Clone)]
pub enum EnvInner {
    Nil,
    Cons {
        name: Symbol,
        val: PluckVal,
        tail: Env,
    },
}

pub fn empty_env() -> Env {
    Rc::new(EnvInner::Nil)
}

pub fn env_cons(name: Symbol, val: PluckVal, tail: Env) -> Env {
    Rc::new(EnvInner::Cons { name, val, tail })
}

pub fn env_lookup(env: &Env, name: Symbol) -> Option<PluckVal> {
    let mut current = env;
    loop {
        match current.as_ref() {
            EnvInner::Nil => return None,
            EnvInner::Cons { name: n, val, tail } => {
                if *n == name {
                    return Some(val.clone());
                }
                current = tail;
            }
        }
    }
}

pub fn env_len(env: &Env) -> usize {
    let mut count = 0;
    let mut current = env;
    loop {
        match current.as_ref() {
            EnvInner::Nil => return count,
            EnvInner::Cons { tail, .. } => {
                count += 1;
                current = tail;
            }
        }
    }
}

pub fn extract_nat(val: &PluckVal, o_sym: Symbol, s_sym: Symbol) -> Option<u64> {
    if let PluckValue::Native(NativeVal::Int(i)) = val.as_ref() {
        return Some(*i as u64);
    }
    let mut count: u64 = 0;
    let mut current = val;
    loop {
        match current.as_ref() {
            PluckValue::Value { constructor, args } if *constructor == o_sym && args.is_empty() => {
                return Some(count);
            }
            PluckValue::Value { constructor, args } if *constructor == s_sym && args.len() == 1 => {
                count += 1;
                current = &args[0];
            }
            _ => return None,
        }
    }
}


pub fn mk_value(constructor: Symbol, args: Vec<PluckVal>) -> PluckVal {
    Rc::new(PluckValue::Value { constructor, args })
}

pub fn mk_native_float(v: f64) -> PluckVal {
    Rc::new(PluckValue::Native(NativeVal::Float(v)))
}

pub fn mk_native_int(v: i64) -> PluckVal {
    Rc::new(PluckValue::Native(NativeVal::Int(v)))
}

pub fn mk_native_symbol(s: Symbol) -> PluckVal {
    Rc::new(PluckValue::Native(NativeVal::Symbol(s)))
}

pub fn mk_native_bool(b: bool) -> PluckVal {
    Rc::new(PluckValue::Native(NativeVal::Bool(b)))
}

pub fn mk_closure(var: Symbol, body: Rc<PExpr>, env: Env) -> PluckVal {
    Rc::new(PluckValue::Closure(ClosureData {
        var,
        body,
        env,
        self_ref: RefCell::new(None),
        rec_name: None,
        arg_name: None,
    }))
}

pub fn mk_int_dist(bits: Vec<BddPtr<'static>>) -> PluckVal {
    Rc::new(PluckValue::IntDist { bits })
}

pub fn mk_y_closure(body: Rc<PExpr>, env: Env, rec_name: Symbol, arg_name: Symbol) -> PluckVal {
    let closure = Rc::new(PluckValue::Closure(ClosureData {
        var: arg_name,
        body,
        env,
        self_ref: RefCell::new(None),
        rec_name: Some(rec_name),
        arg_name: Some(arg_name),
    }));
    if let PluckValue::Closure(ref data) = *closure {
        *data.self_ref.borrow_mut() = Some(closure.clone());
    }
    closure
}

pub fn mk_thunk(
    expr: Rc<PExpr>,
    env: Env,
    strict_order_index: i32,
    callstack: Vec<i32>,
) -> PluckVal {
    Rc::new(PluckValue::Thunk(ThunkData {
        expr,
        env,
        strict_order_index,
        callstack,
        cache: RefCell::new(Vec::new()),
    }))
}

pub fn mk_thunk_union(thunks: Vec<(PluckVal, BddPtr<'static>)>) -> PluckVal {
    Rc::new(PluckValue::ThunkUnion(ThunkUnionData {
        thunks,
        cache: RefCell::new(Vec::new()),
    }))
}

pub fn mk_probability(beta_name: u64) -> PluckVal {
    Rc::new(PluckValue::Probability(beta_name))
}

pub fn mk_gamma(gamma_name: u64) -> PluckVal {
    Rc::new(PluckValue::Gamma {
        name: gamma_name,
        scale: 1.0,
    })
}

pub fn mk_gamma_scaled(gamma_name: u64, scale: f64) -> PluckVal {
    debug_assert!(scale > 0.0, "gamma scale must be positive, got {scale}");
    Rc::new(PluckValue::Gamma {
        name: gamma_name,
        scale,
    })
}

pub fn mk_exponential(gamma_name: u64, exponential_name: u64, scale: f64) -> PluckVal {
    debug_assert!(
        scale > 0.0,
        "exponential scale must be positive, got {scale}"
    );
    Rc::new(PluckValue::Exponential {
        gamma: gamma_name,
        name: exponential_name,
        scale,
    })
}

pub fn mk_poisson(gamma_name: u64, poisson_name: u64, scale: f64) -> PluckVal {
    debug_assert!(scale > 0.0, "poisson scale must be positive, got {scale}");
    Rc::new(PluckValue::Poisson {
        gamma: gamma_name,
        name: poisson_name,
        scale,
    })
}

pub fn mk_dirichlet_probability(dirichlet_name: u64, index: usize, categories: usize) -> PluckVal {
    Rc::new(PluckValue::DirichletProbability {
        name: dirichlet_name,
        index,
        categories,
    })
}

pub fn mk_dirichlet(dirichlet_name: u64, categories: usize) -> PluckVal {
    Rc::new(PluckValue::DirichletVector {
        name: dirichlet_name,
        categories,
    })
}

pub fn mk_gaussian_expr(
    constant: f64,
    coefficients: BTreeMap<u64, ordered_float::NotNan<f64>>,
) -> PluckVal {
    Rc::new(PluckValue::GaussianExpr {
        constant,
        coefficients,
    })
}

pub fn mk_float_matrix(entries: Vec<PluckVal>, shape: Vec<usize>) -> PluckVal {
    let expected: usize = shape.iter().product();
    debug_assert_eq!(
        entries.len(),
        expected,
        "FloatMatrix entry count {} does not match shape {:?} (= {})",
        entries.len(),
        shape,
        expected
    );
    Rc::new(PluckValue::FloatMatrix { entries, shape })
}

pub fn flat_index(shape: &[usize], idx: &[usize]) -> usize {
    assert_eq!(
        shape.len(),
        idx.len(),
        "flat_index: rank mismatch (shape rank {}, got {} indices)",
        shape.len(),
        idx.len()
    );
    let mut out = 0usize;
    for (i, dim) in idx.iter().zip(shape.iter()) {
        assert!(
            *i < *dim,
            "flat_index: out-of-bounds index {} for dimension of size {}",
            i,
            dim
        );
        out = out * dim + i;
    }
    out
}

pub fn entry_constant(v: &PluckValue) -> Option<f64> {
    match v {
        PluckValue::Native(NativeVal::Float(f)) => Some(*f),
        _ => None,
    }
}

pub fn is_thunk(val: &PluckValue) -> bool {
    matches!(val, PluckValue::Thunk(_) | PluckValue::ThunkUnion(_))
}


#[derive(Debug)]
pub enum SymbolicPositionKind {
    Beta(ContVarName),
    Dirichlet(ContVarName),
    Gamma {
        name: ContVarName,
        scale: f64,
    },
    Gaussian(GaussianAffineExpr),
}

impl PluckValue {
    pub fn is_symbolic(&self) -> bool {
        matches!(
            self,
            PluckValue::Probability(_)
                | PluckValue::DirichletProbability { .. }
                | PluckValue::DirichletVector { .. }
                | PluckValue::GaussianExpr { .. }
                | PluckValue::Gamma { .. }
                | PluckValue::Exponential { .. }
                | PluckValue::Poisson { .. }
        )
    }

    pub fn symbolic_eq(&self, other: &Self) -> bool {
        match (self, other) {
            (PluckValue::Probability(a), PluckValue::Probability(b)) => a == b,
            (
                PluckValue::GaussianExpr {
                    constant: ca,
                    coefficients: ma,
                },
                PluckValue::GaussianExpr {
                    constant: cb,
                    coefficients: mb,
                },
            ) => ca.to_bits() == cb.to_bits() && ma == mb,
            (
                PluckValue::Gamma { name: a, scale: sa },
                PluckValue::Gamma { name: b, scale: sb },
            ) => a == b && sa.to_bits() == sb.to_bits(),
            (
                PluckValue::Exponential {
                    name: na,
                    gamma: ga,
                    scale: sa,
                },
                PluckValue::Exponential {
                    name: nb,
                    gamma: gb,
                    scale: sb,
                },
            ) => na == nb && ga == gb && sa.to_bits() == sb.to_bits(),
            (
                PluckValue::Poisson {
                    name: na,
                    gamma: ga,
                    scale: sa,
                },
                PluckValue::Poisson {
                    name: nb,
                    gamma: gb,
                    scale: sb,
                },
            ) => na == nb && ga == gb && sa.to_bits() == sb.to_bits(),
            _ => false,
        }
    }

    pub fn force_symbolic(&self, a: &Assignment, true_sym: Symbol) -> Option<PluckVal> {
        match self {
            PluckValue::Probability(name) => {
                let p = a.beta_value(*name).into_inner();
                Some(mk_native_float(p))
            }
            PluckValue::DirichletProbability { name, index, .. } => {
                let p = a.dirichlet_value(*name)[*index].into_inner();
                Some(mk_native_float(p))
            }
            PluckValue::DirichletVector { name, categories } => {
                let probs = a.dirichlet_value(*name);
                assert_eq!(probs.len(), *categories);
                let args: Vec<PluckVal> = probs
                    .iter()
                    .map(|p| mk_native_float(p.into_inner()))
                    .collect();
                Some(mk_value(true_sym, args))
            }
            PluckValue::GaussianExpr {
                constant,
                coefficients,
            } => {
                let mut total = *constant;
                for (name, coef) in coefficients {
                    let g_val = a
                        .gaussian
                        .iter()
                        .find(|(n, _)| *n == *name)
                        .map(|(_, v)| v.into_inner())
                        .expect("force_symbolic: GaussianExpr references unassigned variable");
                    total += coef.into_inner() * g_val;
                }
                Some(mk_native_float(total))
            }
            PluckValue::Gamma { name, scale } => {
                Some(mk_native_float(scale * a.gamma_value(*name).into_inner()))
            }
            PluckValue::Exponential { name, .. } => {
                Some(mk_native_float(a.gamma_value(*name).into_inner()))
            }
            PluckValue::Poisson { name, .. } => {
                let v = a.gamma_value(*name).into_inner();
                debug_assert_eq!(
                    v.fract(),
                    0.0,
                    "force_symbolic: non-integer Poisson draw {v}"
                );
                Some(mk_native_int(v as i64))
            }
            _ => None,
        }
    }

    pub fn flip_weight(&self) -> Option<FlipWeight> {
        match self {
            PluckValue::Native(NativeVal::Float(f)) => Some(FlipWeight::Constant(*f)),
            PluckValue::Probability(name) => Some(FlipWeight::Beta(BetaFlip::Var(*name))),
            PluckValue::DirichletProbability {
                name,
                index,
                categories,
            } => Some(FlipWeight::Dirichlet(DirichletFlip::VarElement {
                name: *name,
                index: *index,
                categories: *categories,
            })),
            _ => None,
        }
    }

    pub fn render_symbolic(&self, naming: Option<&NamingScheme>) -> Option<String> {
        match self {
            PluckValue::Probability(name) => {
                Some(match naming.and_then(|n| n.var_names.get(name)) {
                    Some(n) => n.clone(),
                    None => format!("Beta(#{}) [prior]", name),
                })
            }
            PluckValue::DirichletProbability { name, index, .. } => {
                Some(match naming.and_then(|n| n.var_names.get(name)) {
                    Some(n) => format!("{}{}", n, subscript_str(*index)),
                    None => format!("Dirichlet(#{})[{}] [prior]", name, index),
                })
            }
            PluckValue::DirichletVector { name, .. } => {
                Some(match naming.and_then(|n| n.var_names.get(name)) {
                    Some(n) => n.clone(),
                    None => format!("Dirichlet(#{}) [prior]", name),
                })
            }
            PluckValue::GaussianExpr {
                constant,
                coefficients,
            } => {
                if coefficients.is_empty() {
                    return Some(format!("{}", constant));
                }
                if let Some(n) = naming.and_then(|n| {
                    n.expr_names
                        .get(&GaussianAffineExpr::new(coefficients, constant))
                }) {
                    return Some(n.clone());
                }
                let mut parts = Vec::new();
                if *constant != 0.0 {
                    parts.push(format!("{}", constant));
                }
                for (name, coeff) in coefficients {
                    if coeff.into_inner() == 1.0 {
                        parts.push(format!("G#{}", name));
                    } else {
                        parts.push(format!("{}*G#{}", coeff, name));
                    }
                }
                Some(format!("GaussianExpr({})", parts.join(" + ")))
            }
            PluckValue::Gamma { name, scale } => {
                let base = match naming.and_then(|n| n.var_names.get(name)) {
                    Some(n) => n.clone(),
                    None => format!("Gamma(#{}) [prior]", name),
                };
                Some(if *scale == 1.0 {
                    base
                } else {
                    format!("{} * {}", scale, base)
                })
            }
            PluckValue::Exponential { name, .. } => {
                Some(match naming.and_then(|n| n.var_names.get(name)) {
                    Some(n) => n.clone(),
                    None => format!("Exponential(#{}) [prior]", name),
                })
            }
            PluckValue::Poisson { name, .. } => {
                Some(match naming.and_then(|n| n.var_names.get(name)) {
                    Some(n) => n.clone(),
                    None => format!("Poisson(#{}) [prior]", name),
                })
            }
            _ => None,
        }
    }

    pub fn scope_into(&self, out: &mut BTreeSet<ContVarName>) {
        match self {
            PluckValue::Probability(name) => {
                out.insert(*name);
            }
            PluckValue::GaussianExpr { coefficients, .. } => {
                for &n in coefficients.keys() {
                    out.insert(n);
                }
            }
            PluckValue::DirichletProbability { name, .. }
            | PluckValue::DirichletVector { name, .. } => {
                out.insert(*name);
            }
            PluckValue::Gamma { name, .. } => {
                out.insert(*name);
            }
            PluckValue::Exponential { name, gamma, .. }
            | PluckValue::Poisson { name, gamma, .. } => {
                out.insert(*name);
                out.insert(*gamma);
            }
            _ => {}
        }
    }

    pub fn position_kind(&self) -> Option<SymbolicPositionKind> {
        match self {
            PluckValue::Probability(name) => Some(SymbolicPositionKind::Beta(*name)),
            PluckValue::DirichletProbability { name, .. }
            | PluckValue::DirichletVector { name, .. } => {
                Some(SymbolicPositionKind::Dirichlet(*name))
            }
            PluckValue::GaussianExpr {
                coefficients,
                constant,
            } if !coefficients.is_empty() => Some(SymbolicPositionKind::Gaussian(
                GaussianAffineExpr::new(coefficients, constant),
            )),
            PluckValue::Gamma { name, scale } => Some(SymbolicPositionKind::Gamma {
                name: *name,
                scale: *scale,
            }),
            PluckValue::Exponential { name, .. } | PluckValue::Poisson { name, .. } => {
                Some(SymbolicPositionKind::Gamma {
                    name: *name,
                    scale: 1.0,
                })
            }
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ordered_float::NotNan;

    fn nn(v: f64) -> NotNan<f64> {
        NotNan::new(v).unwrap()
    }

    #[test]
    fn symbolic_eq_distinguishes_gamma_scale() {
        let bare = mk_gamma(7);
        let scaled5 = mk_gamma_scaled(7, 5.0);
        let scaled5_again = mk_gamma_scaled(7, 5.0);
        let scaled3 = mk_gamma_scaled(7, 3.0);

        assert!(bare.symbolic_eq(&mk_gamma(7)));
        assert!(scaled5.symbolic_eq(&scaled5_again));
        assert!(!bare.symbolic_eq(&scaled5));
        assert!(!scaled5.symbolic_eq(&scaled3));
    }

    #[test]
    fn force_symbolic_scales_gamma_but_not_draws() {
        let a = Assignment::from_sorted(vec![], vec![], vec![(7, nn(2.0))], vec![]);

        let scaled = mk_gamma_scaled(7, 5.0);
        match scaled.force_symbolic(&a, 0).unwrap().as_ref() {
            PluckValue::Native(NativeVal::Float(f)) => assert_eq!(*f, 10.0),
            other => panic!("expected Native(Float), got {other:?}"),
        }

        match mk_gamma(7).force_symbolic(&a, 0).unwrap().as_ref() {
            PluckValue::Native(NativeVal::Float(f)) => assert_eq!(*f, 2.0),
            other => panic!("expected Native(Float), got {other:?}"),
        }

        let draw = mk_exponential(7, 99, 5.0);
        let a_draw = Assignment::from_sorted(vec![], vec![], vec![(99, nn(2.0))], vec![]);
        match draw.force_symbolic(&a_draw, 0).unwrap().as_ref() {
            PluckValue::Native(NativeVal::Float(f)) => assert_eq!(*f, 2.0),
            other => panic!("expected Native(Float), got {other:?}"),
        }
    }
}
