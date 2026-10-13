
use std::collections::{HashMap, HashSet};
use std::rc::Rc;
use std::time::Duration;

use rsdd::repr::{BddPtr, DDNNFPtr as _};

use crate::inference::lazy_kc::state::LazyKCState;
use crate::inference::spn::evidence::intern_table_size;
use crate::inference::spn::{Spn, SpnKind, SpnLeaf};

#[derive(Clone, Copy, Default)]
pub struct StageTimes {
    pub wmc: Duration,
    pub posterior: Duration,
    pub flatten: Duration,
    pub display: Duration,
}

#[derive(Default)]
pub struct KcStats {
    enabled: bool,
    worlds: usize,
    bdd_sum: usize,
    bdd_max: usize,
    spn_tree_sum: usize,
    spn_tree_max: usize,
    spn_leaf_sum: usize,
    sum_arity_max: usize,
    prod_arity_max: usize,
    spn_dag_max: usize,
    spn_dag_all: HashSet<*const ()>,
    components: usize,
    t_parse: f64,
    t_symbolic: f64,
    t_wmc: f64,
    t_post: f64,
    t_flatten: f64,
    t_display: f64,
}

impl KcStats {
    pub fn new() -> Self {
        KcStats {
            enabled: std::env::var_os("PLUCK_STATS").is_some(),
            ..Default::default()
        }
    }

    pub fn enabled(&self) -> bool {
        self.enabled
    }

    pub fn set_parse(&mut self, d: Duration) {
        if !self.enabled {
            return;
        }
        self.t_parse += d.as_secs_f64();
    }

    pub fn set_symbolic(&mut self, d: Duration) {
        if !self.enabled {
            return;
        }
        self.t_symbolic += d.as_secs_f64();
    }

    pub fn record_world<L: SpnLeaf>(
        &mut self,
        bdd: BddPtr<'static>,
        evidence: &Spn<L>,
        times: StageTimes,
        components: usize,
    ) {
        if !self.enabled {
            return;
        }
        self.worlds += 1;

        let gn = bdd.count_nodes();
        self.bdd_sum += gn;
        self.bdd_max = self.bdd_max.max(gn);

        let tree = spn_tree_size(evidence);
        self.spn_tree_sum += tree.nodes;
        self.spn_tree_max = self.spn_tree_max.max(tree.nodes);
        self.spn_leaf_sum += tree.leaves;
        self.sum_arity_max = self.sum_arity_max.max(tree.sums);
        self.prod_arity_max = self.prod_arity_max.max(tree.products);

        let mut world_dag = HashSet::new();
        collect_dag_nodes(evidence, &mut world_dag);
        collect_dag_nodes(evidence, &mut self.spn_dag_all);
        self.spn_dag_max = self.spn_dag_max.max(world_dag.len());

        self.components += components;
        self.t_wmc += times.wmc.as_secs_f64();
        self.t_post += times.posterior.as_secs_f64();
        self.t_flatten += times.flatten.as_secs_f64();
        self.t_display += times.display.as_secs_f64();
    }

    pub fn report(&self, state: &LazyKCState) {
        if !self.enabled {
            return;
        }
        let nw = self.worlds.max(1);
        eprintln!("=== PLUCK_STATS ===");
        eprintln!("worlds (value,BDD-guard pairs)        : {}", self.worlds);
        eprintln!(
            "continuous latents (Beta/Gaussian/…) : {}",
            state.continuous_var_counter
        );
        eprintln!(
            "BDD vars allocated (discrete bits)    : {}",
            state.bdd().num_vars()
        );
        eprintln!(
            "BDD recursive calls (compile work)    : {}",
            state.bdd().stats().num_recursive_calls
        );
        eprintln!(
            "BDD guard nodes  : sum={}  max={}  mean={}",
            self.bdd_sum,
            self.bdd_max,
            self.bdd_sum / nw
        );
        eprintln!(
            "SPN nodes (TREE) : sum={}  max={}  mean={}",
            self.spn_tree_sum,
            self.spn_tree_max,
            self.spn_tree_sum / nw
        );
        eprintln!(
            "SPN nodes (DAG)  : per-world-max={}  all-worlds-unique={}  intern-table={}",
            self.spn_dag_max,
            self.spn_dag_all.len(),
            intern_table_size()
        );
        eprintln!(
            "SPN leaf nodes   : sum={}  (max sum-arity={}, max prod-arity={})",
            self.spn_leaf_sum, self.sum_arity_max, self.prod_arity_max
        );
        eprintln!(
            "posterior mixture components (total)  : {}",
            self.components
        );
        let total = self.t_parse
            + self.t_symbolic
            + self.t_wmc
            + self.t_post
            + self.t_flatten
            + self.t_display;
        eprintln!(
            "time (s): parse={:.4}  symbolic-exec={:.4}  wmc={:.4}  posterior={:.4}  flatten={:.4}  display/naming={:.4}  total={:.4}",
            self.t_parse, self.t_symbolic, self.t_wmc, self.t_post, self.t_flatten, self.t_display, total
        );
        eprintln!("===================");
        eprintln!("inference_seconds: {:.8}", total);
    }
}

#[derive(Default, Clone, Copy)]
struct TreeSize {
    nodes: usize,
    leaves: usize,
    sums: usize,
    products: usize,
}

fn spn_tree_size<L: SpnLeaf>(spn: &Spn<L>) -> TreeSize {
    fn go<L: SpnLeaf>(spn: &Spn<L>, memo: &mut HashMap<*const (), TreeSize>) -> TreeSize {
        let ptr = spn.inner_ptr() as *const ();
        if let Some(&v) = memo.get(&ptr) {
            return v;
        }
        let v = match spn.kind() {
            SpnKind::Scalar => TreeSize {
                nodes: 1,
                leaves: 0,
                sums: 0,
                products: 0,
            },
            SpnKind::Leaf(_) => TreeSize {
                nodes: 1,
                leaves: 1,
                sums: 0,
                products: 0,
            },
            SpnKind::Sum(cs) | SpnKind::Product(cs) => {
                let is_sum = matches!(spn.kind(), SpnKind::Sum(_));
                let mut acc = TreeSize {
                    nodes: 1,
                    leaves: 0,
                    sums: is_sum as usize,
                    products: !is_sum as usize,
                };
                for c in cs {
                    let child = go(c, memo);
                    acc.nodes = acc.nodes.saturating_add(child.nodes);
                    acc.leaves = acc.leaves.saturating_add(child.leaves);
                    acc.sums = acc.sums.saturating_add(child.sums);
                    acc.products = acc.products.saturating_add(child.products);
                }
                acc
            }
        };
        memo.insert(ptr, v);
        v
    }
    go(spn, &mut HashMap::new())
}

fn collect_dag_nodes<L: SpnLeaf>(spn: &Spn<L>, seen: &mut HashSet<*const ()>) {
    let ptr = Rc::as_ptr(&spn.inner) as *const ();
    if !seen.insert(ptr) {
        return;
    }
    if let SpnKind::Sum(cs) | SpnKind::Product(cs) = spn.kind() {
        for c in cs {
            collect_dag_nodes(c, seen);
        }
    }
}
