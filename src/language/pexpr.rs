use std::fmt;

use super::types::Symbol;

#[derive(Debug, Clone)]
pub struct CaseOfGuard {
    pub constructor: Symbol,
    pub args: Vec<Symbol>,
}

#[derive(Debug, Clone)]
pub enum PExpr {
    App {
        func: Box<PExpr>,
        arg: Box<PExpr>,
    },
    Abs {
        var: Symbol,
        body: Box<PExpr>,
    },
    Var {
        name: Symbol,
    },
    Defined {
        name: Symbol,
    },
    Construct {
        constructor: Symbol,
        args: Vec<PExpr>,
    },
    CaseOf {
        guards: Vec<CaseOfGuard>,
        scrutinee: Box<PExpr>,
        branches: Vec<PExpr>,
    },
    Flip {
        prob: Box<PExpr>,
    },
    YComb {
        func: Box<PExpr>,
    },
    ConstNative {
        val: NativeVal,
    },

    NativeEq {
        a: Box<PExpr>,
        b: Box<PExpr>,
    },
    MkInt {
        bitwidth: Box<PExpr>,
        value: Box<PExpr>,
    },
    UniformInt {
        bitwidth: Box<PExpr>,
    },
    UniformIntRange {
        bitwidth: Box<PExpr>,
        lo: Box<PExpr>,
        hi: Box<PExpr>,
    },
    IntDistEq {
        a: Box<PExpr>,
        b: Box<PExpr>,
    },
    IntDistAdd {
        a: Box<PExpr>,
        b: Box<PExpr>,
    },
    IntDistSub {
        a: Box<PExpr>,
        b: Box<PExpr>,
    },
    IntDistLt {
        a: Box<PExpr>,
        b: Box<PExpr>,
    },
    GetArgs {
        expr: Box<PExpr>,
    },
    GetConstructor {
        expr: Box<PExpr>,
    },
    Print {
        expr: Box<PExpr>,
    },
    Error {
        msg: Box<PExpr>,
    },
    Index {
        v: Box<PExpr>,
        idx: Vec<PExpr>,
    },

    FDiv {
        a: Box<PExpr>,
        b: Box<PExpr>,
    },
    FMul {
        a: Box<PExpr>,
        b: Box<PExpr>,
    },
    FAdd {
        a: Box<PExpr>,
        b: Box<PExpr>,
    },
    FSub {
        a: Box<PExpr>,
        b: Box<PExpr>,
    },

    GSymbol {
        name: Symbol,
    },
    GVarSymbol {
        type_name: Symbol,
    },

    GetConfig,

    Beta {
        a: Box<PExpr>,
        b: Box<PExpr>,
    },
    ProbEq {
        prob: Box<PExpr>,
        val: Box<PExpr>,
    },

    Gaussian {
        mu: Box<PExpr>,
        sigma: Box<PExpr>,
    },
    RealEq {
        expr: Box<PExpr>,
        val: Box<PExpr>,
    },
    RealLt {
        expr: Box<PExpr>,
        val: Box<PExpr>,
    },
    RealGeq {
        expr: Box<PExpr>,
        val: Box<PExpr>,
    },

    Dirichlet {
        alphas: Vec<PExpr>,
    },
    DirichletEq {
        expr: Box<PExpr>,
        val: Box<PExpr>,
    },

    Categorical {
        probs: Vec<PExpr>,
    },

    NativeLeq {
        a: Box<PExpr>,
        b: Box<PExpr>,
    },
    NativeGeq {
        a: Box<PExpr>,
        b: Box<PExpr>,
    },

    Poisson {
        rate: Box<PExpr>,
    },

    Exponential {
        rate: Box<PExpr>,
    },

    Gamma {
        shape: Box<PExpr>,
        rate: Box<PExpr>,
    },

    FloatMatrix {
        entries: Vec<Vec<PExpr>>,
        shape: Vec<usize>,
    },

    MatMul {
        a: Box<PExpr>,
        b: Box<PExpr>,
    },

    Sum {
        expr: Box<PExpr>,
    },

    Pin {
        expr: Box<PExpr>,
        val: Box<PExpr>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum NativeVal {
    Float(f64),
    Int(i64),
    Symbol(Symbol),
    Bool(bool),
}

impl fmt::Display for NativeVal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NativeVal::Float(v) => write!(f, "{}", v),
            NativeVal::Int(v) => write!(f, "{}", v),
            NativeVal::Symbol(s) => write!(f, "'{}", s),
            NativeVal::Bool(b) => write!(f, "{}", b),
        }
    }
}
