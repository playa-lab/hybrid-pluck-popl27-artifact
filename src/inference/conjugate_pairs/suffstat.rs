use std::collections::BTreeSet;
use std::fmt::Display;
use std::hash::Hash;

use crate::inference::conjugate_pairs::{Assignment, AssignmentBuilder, PriorRegistry};
use crate::utils::epsilon::RealEps;

pub type ContVarName = u64;

pub const PRIOR_QUANTIZATION_LEVEL: f64 = 1e6;

pub trait Prior: Display + Hash {
    type Realization;

    fn scope(&self) -> impl Iterator<Item = &ContVarName> + '_;

    fn sample<R: rand::Rng>(&self, rng: &mut R) -> Self::Realization;

    fn push_into(&self, sampled: Self::Realization, out: &mut AssignmentBuilder);
}

pub trait SuffStat: PartialEq + Clone + Hash {
    type ConjugatePrior: Prior;

    fn scope(&self) -> impl Iterator<Item = &ContVarName> + '_;

    fn merge(&self, other: &Self) -> (Self, f64);

    fn log_likelihood(&self, value: &<Self::ConjugatePrior as Prior>::Realization) -> f64;

    fn posterior(
        &self,
        prior: &Self::ConjugatePrior,
        query_vars: &BTreeSet<ContVarName>,
    ) -> (Option<Self::ConjugatePrior>, RealEps);

    fn read_realization(&self, a: &Assignment) -> <Self::ConjugatePrior as Prior>::Realization;

    fn lookup_prior_in(&self, reg: &PriorRegistry) -> Self::ConjugatePrior;

    fn log_likelihood_in(&self, a: &Assignment) -> f64 {
        self.log_likelihood(&self.read_realization(a))
    }

    fn posterior_in(
        &self,
        reg: &PriorRegistry,
        query_vars: &BTreeSet<ContVarName>,
    ) -> (Option<Self::ConjugatePrior>, RealEps) {
        self.posterior(&self.lookup_prior_in(reg), query_vars)
    }
}
