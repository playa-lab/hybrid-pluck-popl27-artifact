
use std::{
    collections::BTreeSet,
    fmt::{Display, Formatter, Result, Write},
    hash::Hash,
};

use indenter::indented;
use itertools::{chain, Itertools};
use nalgebra as na;

use ordered_float::NotNan;

use super::Assignment;
use crate::inference::conjugate_pairs::{
    BetaPrior, ContVarName, DirichletPrior, GammaPrior, GaussianPrior, JointGammaPrior, Prior,
};
use crate::utils::sorted_vec_by_name::{
    add_to_sorted_by_name, lookup_sorted_by_name, slice_sorted_by_name,
};

#[derive(Debug, Clone, Hash, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PriorRegistry<N: Display + Hash + PartialEq + Eq = ContVarName> {
    pub beta: Vec<BetaPrior<N>>,
    pub dirichlet: Vec<DirichletPrior<N>>,
    #[serde(
        default = "Vec::new",
        bound(
            serialize = "N: serde::Serialize + Ord",
            deserialize = "N: serde::Deserialize<'de> + Ord"
        )
    )]
    pub gamma: Vec<JointGammaPrior<N>>,
    pub gaussian: Vec<GaussianPrior<N>>,
}

impl<N: Display + Hash + PartialEq + Eq + Clone> Display for PriorRegistry<N> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        let mut indented_f = indented(f).with_str("    ");

        for g in &self.gaussian {
            writeln!(indented_f, "{}", g)?;
        }
        for b in &self.beta {
            writeln!(indented_f, "{}", b)?;
        }
        for d in &self.dirichlet {
            writeln!(indented_f, "{}", d)?;
        }
        for g in &self.gamma {
            writeln!(indented_f, "{}", g)?;
        }

        Ok(())
    }
}

impl Default for PriorRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl PriorRegistry {
    pub fn new() -> Self {
        Self {
            beta: Vec::new(),
            dirichlet: Vec::new(),
            gamma: Vec::new(),
            gaussian: Vec::new(),
        }
    }

    pub fn scope(&self) -> impl Iterator<Item = &ContVarName> + '_ {
        chain![
            self.beta.iter().flat_map(|b| b.scope()),
            self.dirichlet.iter().flat_map(|b| b.scope()),
            self.gamma.iter().flat_map(|b| b.scope()),
            self.gaussian.iter().flat_map(|b| b.scope()),
        ]
    }

    pub fn from_sorted(
        beta: Vec<BetaPrior>,
        dirichlet: Vec<DirichletPrior>,
        gamma: Vec<JointGammaPrior>,
        gaussian: Vec<GaussianPrior>,
    ) -> Self {
        debug_assert!(
            beta.windows(2).all(|w| w[0].name() < w[1].name()),
            "PriorRegistry::beta must be strictly increasing by name"
        );
        debug_assert!(
            dirichlet.windows(2).all(|w| w[0].name() < w[1].name()),
            "PriorRegistry::dirichlet must be strictly increasing by name"
        );
        debug_assert!(
            gamma.windows(2).all(|w| w[0].name() < w[1].name()),
            "PriorRegistry::gamma must be strictly increasing by name"
        );
        for b in &gaussian {
            debug_assert!(
                !b.var_order.is_empty(),
                "PriorRegistry::gaussian: empty block"
            );
            debug_assert!(
                b.var_order.windows(2).all(|w| w[0] < w[1]),
                "PriorRegistry::gaussian: var_order must be strictly increasing"
            );
        }
        debug_assert!(
            gaussian
                .windows(2)
                .all(|w| w[0].var_order[0] < w[1].var_order[0]),
            "PriorRegistry::gaussian: blocks must be sorted by var_order[0]"
        );
        for i in 0..gaussian.len() {
            for j in (i + 1)..gaussian.len() {
                let a: BTreeSet<ContVarName> = gaussian[i].var_order.iter().copied().collect();
                let b: BTreeSet<ContVarName> = gaussian[j].var_order.iter().copied().collect();
                debug_assert!(
                    a.is_disjoint(&b),
                    "PriorRegistry::gaussian: blocks {:?} and {:?} share variables",
                    gaussian[i].var_order,
                    gaussian[j].var_order
                );
            }
        }
        Self {
            beta,
            dirichlet,
            gamma,
            gaussian,
        }
    }


    pub fn add_beta(&mut self, prior: BetaPrior) {
        add_to_sorted_by_name(
            &mut self.beta,
            prior,
            BetaPrior::name,
            "PriorRegistry::add_beta",
        );
    }

    pub fn add_dirichlet(&mut self, prior: DirichletPrior) {
        add_to_sorted_by_name(
            &mut self.dirichlet,
            prior,
            DirichletPrior::name,
            "PriorRegistry::add_dirichlet",
        );
    }

    pub fn add_gamma(&mut self, prior: GammaPrior) {
        add_to_sorted_by_name(
            &mut self.gamma,
            JointGammaPrior::from_gamma(prior),
            JointGammaPrior::name,
            "PriorRegistry::add_gamma",
        );
    }

    pub fn add_gaussian(&mut self, prior: GaussianPrior) {
        if let Some(existing) = self
            .gaussian
            .iter()
            .find(|g| g.var_order == prior.var_order)
        {
            debug_assert!(
                existing.mean == prior.mean && existing.cov == prior.cov,
                "PriorRegistry::add_gaussian: re-registering scope {:?} with different params",
                prior.var_order
            );
            return;
        }
        debug_assert!(
            self.gaussian
                .iter()
                .flat_map(|g| &g.var_order)
                .all(|name| { !prior.var_order.contains(name) }),
            "PriorRegistry::add_gaussian: re-registering variable names {:?} with non-identical scope",
            prior.var_order
        );
        debug_assert!(
            self.gaussian.iter().map(|g| { g.var_order[0] }).is_sorted(),
            "PriorRegistry::add_gaussian current self.gaussian vec is not sorted"
        );
        debug_assert!(
            self.gaussian
                .iter()
                .flat_map(|g| &g.var_order)
                .max()
                .is_none_or(|n| n < prior.var_order.iter().min().unwrap()),
            "PriorRegistry::add_gaussian new name is smaller than existing name"
        );
        self.gaussian.push(prior);
    }

    pub fn beta_prior(&self, name: ContVarName) -> &BetaPrior {
        lookup_sorted_by_name(
            &self.beta,
            name,
            BetaPrior::name,
            "PriorRegistry::beta_prior",
        )
    }

    pub fn dirichlet_prior(&self, name: ContVarName) -> &DirichletPrior {
        lookup_sorted_by_name(
            &self.dirichlet,
            name,
            DirichletPrior::name,
            "PriorRegistry::dirichlet_prior",
        )
    }

    pub fn gamma_prior(&self, name: ContVarName) -> &JointGammaPrior {
        lookup_sorted_by_name(
            &self.gamma,
            name,
            JointGammaPrior::name,
            "PriorRegistry::gamma_prior",
        )
    }

    pub fn gaussian_block_closure<'a>(
        &self,
        vars: impl IntoIterator<Item = &'a ContVarName>,
    ) -> BTreeSet<ContVarName> {
        let vars: BTreeSet<ContVarName> = vars.into_iter().copied().collect();
        let mut closure = BTreeSet::new();
        for b in &self.gaussian {
            if b.var_order.iter().any(|v| vars.contains(v)) {
                closure.extend(b.var_order.iter().copied());
            }
        }
        closure
    }

    pub fn gaussian_prior_for(&self, scope: &BTreeSet<ContVarName>) -> GaussianPrior {
        assert!(
            !scope.is_empty(),
            "PriorRegistry::gaussian_prior_for: empty scope"
        );

        let hits: Vec<&GaussianPrior> = self
            .gaussian
            .iter()
            .filter(|b| b.var_order.iter().any(|v| scope.contains(v)))
            .collect();

        let covered: BTreeSet<ContVarName> = hits
            .iter()
            .flat_map(|b| b.var_order.iter().copied())
            .collect();
        for v in scope {
            assert!(
                covered.contains(v),
                "PriorRegistry::gaussian_prior_for: no Gaussian block contains variable {}",
                v
            );
        }

        if hits.len() == 1 {
            return hits[0].slice(scope);
        }

        let var_order: Vec<ContVarName> = scope.iter().copied().collect();
        let n = var_order.len();
        let idx: std::collections::BTreeMap<ContVarName, usize> =
            var_order.iter().enumerate().map(|(i, v)| (*v, i)).collect();
        let mut mean = na::DVector::<f64>::zeros(n);
        let mut cov = na::DMatrix::<f64>::zeros(n, n);
        for b in &hits {
            let sliced = b.slice(scope);
            for (bi, &v) in sliced.var_order.iter().enumerate() {
                let gi = idx[&v];
                mean[gi] = sliced.mean[bi];
                for (bj, &w) in sliced.var_order.iter().enumerate() {
                    let gj = idx[&w];
                    cov[(gi, gj)] = sliced.cov[(bi, bj)];
                }
            }
        }
        GaussianPrior {
            var_order,
            mean,
            cov,
        }
    }

    pub fn merge(&self, other: &PriorRegistry) -> PriorRegistry {
        debug_assert!(!self.scope().any(|n| other.scope().contains(n)));
        let beta = self
            .beta
            .iter()
            .merge_by(&other.beta, |b1, b2| b1.name() <= b2.name())
            .cloned()
            .collect();
        let dirichlet = self
            .dirichlet
            .iter()
            .merge_by(&other.dirichlet, |d1, d2| d1.name() <= d2.name())
            .cloned()
            .collect();
        let gamma = self
            .gamma
            .iter()
            .merge_by(&other.gamma, |g1, g2| g1.name() <= g2.name())
            .cloned()
            .collect();
        let gaussian = self
            .gaussian
            .iter()
            .merge_by(&other.gaussian, |g1, g2| g1.var_order[0] <= g2.var_order[0])
            .cloned()
            .collect();

        PriorRegistry {
            beta,
            dirichlet,
            gamma,
            gaussian,
        }
    }

    pub fn slice(&self, scope: &BTreeSet<ContVarName>) -> PriorRegistry {
        let beta = slice_sorted_by_name(&self.beta, scope, BetaPrior::name);
        let dirichlet = slice_sorted_by_name(&self.dirichlet, scope, DirichletPrior::name);
        let gamma = slice_sorted_by_name(&self.gamma, scope, JointGammaPrior::name);
        let gaussian: Vec<GaussianPrior> = self
            .gaussian
            .iter()
            .filter_map(|b| {
                let sliced = b.slice(scope);
                if sliced.var_order.is_empty() {
                    None
                } else {
                    Some(sliced)
                }
            })
            .collect();
        PriorRegistry {
            beta,
            dirichlet,
            gamma,
            gaussian,
        }
    }

    pub fn sample<R: rand::Rng>(&self, rng: &mut R) -> Assignment {
        let beta: Vec<(ContVarName, NotNan<f64>)> = self
            .beta
            .iter()
            .map(|p| (p.name(), p.sample(rng)))
            .collect();
        let dirichlet: Vec<(ContVarName, Box<[NotNan<f64>]>)> = self
            .dirichlet
            .iter()
            .map(|p| (p.name(), p.sample(rng)))
            .collect();
        let mut gamma: Vec<(ContVarName, NotNan<f64>)> = Vec::new();
        for block in &self.gamma {
            gamma.extend(block.sample(rng));
        }
        gamma.sort_by_key(|(n, _)| *n);
        let mut gaussian: Vec<(ContVarName, NotNan<f64>)> = Vec::new();
        for block in &self.gaussian {
            let values = block.sample(rng);
            for (i, &name) in block.var_order.iter().enumerate() {
                gaussian.push((name, values[i]));
            }
        }
        gaussian.sort_by_key(|(n, _)| *n);
        Assignment::from_sorted(beta, dirichlet, gamma, gaussian)
    }
}

pub trait RegistryInsert {
    fn insert_into(self, reg: &mut PriorRegistry);
}

impl RegistryInsert for BetaPrior {
    fn insert_into(self, reg: &mut PriorRegistry) {
        reg.add_beta(self);
    }
}

impl RegistryInsert for DirichletPrior {
    fn insert_into(self, reg: &mut PriorRegistry) {
        reg.add_dirichlet(self);
    }
}

impl RegistryInsert for GammaPrior {
    fn insert_into(self, reg: &mut PriorRegistry) {
        reg.add_gamma(self);
    }
}

impl RegistryInsert for GaussianPrior {
    fn insert_into(self, reg: &mut PriorRegistry) {
        reg.add_gaussian(self);
    }
}

#[derive(Debug, Clone, Default)]
pub struct RegistryBuilder {
    beta: Vec<BetaPrior>,
    dirichlet: Vec<DirichletPrior>,
    gamma: Vec<JointGammaPrior>,
    gaussian: Vec<GaussianPrior>,
}

impl RegistryBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push_beta(&mut self, p: BetaPrior) {
        self.beta.push(p);
    }
    pub fn push_dirichlet(&mut self, p: DirichletPrior) {
        self.dirichlet.push(p);
    }
    pub fn push_gamma(&mut self, p: JointGammaPrior) {
        self.gamma.push(p);
    }
    pub fn push_gaussian(&mut self, p: GaussianPrior) {
        self.gaussian.push(p);
    }

    pub fn extend_from(&mut self, other: &RegistryBuilder) {
        self.beta.extend(other.beta.iter().cloned());
        self.dirichlet.extend(other.dirichlet.iter().cloned());
        self.gamma.extend(other.gamma.iter().cloned());
        self.gaussian.extend(other.gaussian.iter().cloned());
    }

    pub fn finalize(mut self) -> PriorRegistry {
        self.beta.sort_by_key(|p| p.name());
        self.dirichlet.sort_by_key(|p| p.name());
        self.gamma.sort_by_key(|p| p.name());
        self.gaussian
            .sort_by(|a, b| a.var_order[0].cmp(&b.var_order[0]));
        PriorRegistry::from_sorted(self.beta, self.dirichlet, self.gamma, self.gaussian)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra as na;

    fn beta(name: ContVarName, a: f64, b: f64) -> BetaPrior {
        BetaPrior::Beta { name, a, b }
    }

    fn dirichlet(name: ContVarName, alphas: Vec<f64>) -> DirichletPrior {
        DirichletPrior::Dirichlet { name, alphas }
    }

    fn gaussian_block(var_order: Vec<ContVarName>) -> GaussianPrior {
        let n = var_order.len();
        GaussianPrior {
            var_order,
            mean: na::DVector::zeros(n),
            cov: na::DMatrix::identity(n, n),
        }
    }

    #[test]
    fn beta_lookup_hits_and_misses() {
        let reg = PriorRegistry::from_sorted(
            vec![beta(1, 1.0, 1.0), beta(3, 2.0, 2.0)],
            vec![],
            vec![],
            vec![],
        );
        assert_eq!(reg.beta_prior(1).name(), 1);
        assert_eq!(reg.beta_prior(3).name(), 3);
    }

    #[test]
    #[should_panic(expected = "variable 2 not found")]
    fn beta_missing_panics() {
        let reg = PriorRegistry::from_sorted(vec![beta(1, 1.0, 1.0)], vec![], vec![], vec![]);
        let _ = reg.beta_prior(2);
    }

    #[test]
    fn dirichlet_lookup() {
        let reg =
            PriorRegistry::from_sorted(vec![], vec![dirichlet(5, vec![1.0, 1.0])], vec![], vec![]);
        assert_eq!(reg.dirichlet_prior(5).name(), 5);
    }

    #[test]
    fn gaussian_prior_for_returns_sliced_block() {
        let reg = PriorRegistry::from_sorted(
            vec![],
            vec![],
            vec![],
            vec![gaussian_block(vec![1, 2, 3]), gaussian_block(vec![7, 8])],
        );
        let scope: BTreeSet<ContVarName> = [2u64, 3].iter().copied().collect();
        let p = reg.gaussian_prior_for(&scope);
        assert_eq!(p.var_order, vec![2, 3]);
    }

    #[test]
    #[should_panic(expected = "no Gaussian block contains variable")]
    fn gaussian_prior_for_unknown_variable_panics() {
        let reg =
            PriorRegistry::from_sorted(vec![], vec![], vec![], vec![gaussian_block(vec![1, 2])]);
        let scope: BTreeSet<ContVarName> = [9u64].iter().copied().collect();
        let _ = reg.gaussian_prior_for(&scope);
    }

    #[test]
    fn gaussian_prior_for_cross_block_scope_assembles_block_diagonal() {
        let mut b1 = gaussian_block(vec![1, 2]);
        b1.mean = na::DVector::from_vec(vec![10.0, 20.0]);
        b1.cov = na::DMatrix::from_row_slice(2, 2, &[4.0, 0.5, 0.5, 9.0]);
        let mut b2 = gaussian_block(vec![3, 4]);
        b2.mean = na::DVector::from_vec(vec![30.0, 40.0]);
        b2.cov = na::DMatrix::from_row_slice(2, 2, &[1.0, 0.2, 0.2, 2.0]);
        let reg = PriorRegistry::from_sorted(vec![], vec![], vec![], vec![b1, b2]);

        let scope: BTreeSet<ContVarName> = [2u64, 3].iter().copied().collect();
        let joint = reg.gaussian_prior_for(&scope);
        assert_eq!(joint.var_order, vec![2, 3]);
        assert_eq!(joint.mean[0], 20.0);
        assert_eq!(joint.mean[1], 30.0);
        assert_eq!(joint.cov[(0, 0)], 9.0);
        assert_eq!(joint.cov[(1, 1)], 1.0);
        assert_eq!(joint.cov[(0, 1)], 0.0);
        assert_eq!(joint.cov[(1, 0)], 0.0);
    }

    #[test]
    fn gaussian_prior_for_legacy_independent_priors_full_scope() {
        let b1 = {
            let mut g = gaussian_block(vec![0]);
            g.mean = na::DVector::from_vec(vec![0.0]);
            g.cov = na::DMatrix::from_row_slice(1, 1, &[1.0]);
            g
        };
        let b2 = {
            let mut g = gaussian_block(vec![1]);
            g.mean = na::DVector::from_vec(vec![0.0]);
            g.cov = na::DMatrix::from_row_slice(1, 1, &[1.0]);
            g
        };
        let reg = PriorRegistry::from_sorted(vec![], vec![], vec![], vec![b1, b2]);
        let scope: BTreeSet<ContVarName> = [0u64, 1].iter().copied().collect();
        let joint = reg.gaussian_prior_for(&scope);
        assert_eq!(joint.var_order, vec![0, 1]);
        assert_eq!(joint.cov[(0, 0)], 1.0);
        assert_eq!(joint.cov[(1, 1)], 1.0);
        assert_eq!(joint.cov[(0, 1)], 0.0);
        assert_eq!(joint.cov[(1, 0)], 0.0);
    }

    #[test]
    fn slice_filters_all_three_families() {
        let reg = PriorRegistry::from_sorted(
            vec![beta(1, 1.0, 1.0), beta(2, 1.0, 1.0)],
            vec![dirichlet(5, vec![1.0, 1.0])],
            vec![],
            vec![gaussian_block(vec![10, 11])],
        );
        let scope: BTreeSet<ContVarName> = [1u64, 5, 11].iter().copied().collect();
        let sliced = reg.slice(&scope);
        assert_eq!(sliced.beta.len(), 1);
        assert_eq!(sliced.beta[0].name(), 1);
        assert_eq!(sliced.dirichlet.len(), 1);
        assert_eq!(sliced.dirichlet[0].name(), 5);
        assert_eq!(sliced.gaussian.len(), 1);
        assert_eq!(sliced.gaussian[0].var_order, vec![11]);
    }

    #[test]
    fn slice_drops_disjoint_gaussian_block() {
        let reg = PriorRegistry::from_sorted(
            vec![],
            vec![],
            vec![],
            vec![gaussian_block(vec![1, 2]), gaussian_block(vec![3, 4])],
        );
        let scope: BTreeSet<ContVarName> = [3u64, 4].iter().copied().collect();
        let sliced = reg.slice(&scope);
        assert_eq!(sliced.gaussian.len(), 1);
        assert_eq!(sliced.gaussian[0].var_order, vec![3, 4]);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "strictly increasing")]
    fn from_sorted_rejects_unsorted_beta() {
        let _ = PriorRegistry::from_sorted(
            vec![beta(3, 1.0, 1.0), beta(1, 1.0, 1.0)],
            vec![],
            vec![],
            vec![],
        );
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "share variables")]
    fn from_sorted_rejects_overlapping_gaussian_blocks() {
        let _ = PriorRegistry::from_sorted(
            vec![],
            vec![],
            vec![],
            vec![gaussian_block(vec![1, 2]), gaussian_block(vec![2, 3])],
        );
    }

    #[test]
    #[should_panic]
    fn add_beta_inserts_in_sorted_order_and_dedups() {
        let mut reg = PriorRegistry::new();
        reg.add_beta(BetaPrior::Beta {
            name: 3,
            a: 2.0,
            b: 3.0,
        });
        reg.add_beta(BetaPrior::Beta {
            name: 1,
            a: 1.0,
            b: 1.0,
        });
        reg.add_beta(BetaPrior::Beta {
            name: 3,
            a: 2.0,
            b: 3.0,
        });
        assert_eq!(reg.beta.len(), 2);
        assert_eq!(reg.beta[0].name(), 1);
        assert_eq!(reg.beta[1].name(), 3);
    }

    #[test]
    #[should_panic]
    fn add_dirichlet_inserts_in_sorted_order_and_dedups() {
        let mut reg = PriorRegistry::new();
        reg.add_dirichlet(DirichletPrior::Dirichlet {
            name: 5,
            alphas: vec![1.0, 1.0],
        });
        reg.add_dirichlet(DirichletPrior::Dirichlet {
            name: 2,
            alphas: vec![1.0, 2.0],
        });
        reg.add_dirichlet(DirichletPrior::Dirichlet {
            name: 5,
            alphas: vec![1.0, 1.0],
        });
        assert_eq!(reg.dirichlet.len(), 2);
        assert_eq!(reg.dirichlet[0].name(), 2);
        assert_eq!(reg.dirichlet[1].name(), 5);
    }
}
