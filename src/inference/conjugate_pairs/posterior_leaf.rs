
use std::collections::BTreeSet;
use std::fmt::Display;
use std::hash::Hash;

use crate::inference::conjugate_pairs::{
    BetaPrior, ContVarName, DirichletPrior, GaussianPrior, JointGammaPrior, Prior,
};
use crate::inference::spn::coeff::SpnLeaf;
use crate::utils::epsilon::RealEps;

#[derive(Debug, Clone)]
pub enum PosteriorLeaf<N: Display + Hash + PartialEq + Eq = ContVarName> {
    Beta(BetaPrior<N>),
    Dirichlet(DirichletPrior<N>),
    Gamma(JointGammaPrior<N>),
    Gaussian(GaussianPrior<N>),
}

impl SpnLeaf for PosteriorLeaf<ContVarName> {
    type Coeff = RealEps;

    fn scope(&self) -> BTreeSet<ContVarName> {
        crate::for_each_posterior_family!(self, |p| p.scope().copied().collect())
    }

}
