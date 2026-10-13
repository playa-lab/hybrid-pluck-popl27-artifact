
use crate::inference::conjugate_pairs::PriorRegistry;
use crate::utils::epsilon::RealEps;

pub use crate::inference::conjugate_pairs::PosteriorLeaf;

use super::node::{Spn, SpnKind};

pub type PosteriorSpn = Spn<PosteriorLeaf>;
pub type PosteriorKind = SpnKind<PosteriorLeaf>;

pub fn posterior_leaf(coeff: RealEps, l: PosteriorLeaf) -> PosteriorSpn {
    Spn::leaf(l).coeff_mul(coeff)
}

impl Spn<PosteriorLeaf> {
    pub fn components(&self) -> impl Iterator<Item = (RealEps, PriorRegistry)> + 'static {
        use crate::inference::conjugate_pairs::RegistryBuilder;

        fn enumerate(node: &PosteriorSpn) -> Vec<(RealEps, RegistryBuilder)> {
            match node.kind() {
                SpnKind::Scalar => vec![(node.log_coeff(), RegistryBuilder::new())],
                SpnKind::Leaf(l) => {
                    let mut acc = RegistryBuilder::new();
                    match l {
                        PosteriorLeaf::Beta(p) => acc.push_beta(p.clone()),
                        PosteriorLeaf::Dirichlet(p) => acc.push_dirichlet(p.clone()),
                        PosteriorLeaf::Gamma(p) => acc.push_gamma(p.clone()),
                        PosteriorLeaf::Gaussian(p) => acc.push_gaussian(p.clone()),
                    }
                    vec![(node.log_coeff(), acc)]
                }
                SpnKind::Sum(children) => {
                    let mut out = Vec::new();
                    for c in children {
                        for (w, acc) in enumerate(c) {
                            out.push((node.log_coeff() * w, acc));
                        }
                    }
                    out
                }
                SpnKind::Product(children) => {
                    let mut acc: Vec<(RealEps, RegistryBuilder)> =
                        vec![(node.log_coeff(), RegistryBuilder::new())];
                    for c in children {
                        let cc = enumerate(c);
                        let mut next: Vec<(RealEps, RegistryBuilder)> =
                            Vec::with_capacity(acc.len() * cc.len());
                        for (w1, b1) in &acc {
                            for (w2, b2) in &cc {
                                let mut merged = b1.clone();
                                merged.extend_from(b2);
                                next.push((*w1 * *w2, merged));
                            }
                        }
                        acc = next;
                    }
                    acc
                }
            }
        }

        enumerate(self)
            .into_iter()
            .map(|(w, builder)| (w, builder.finalize()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::conjugate_pairs::{BetaPrior, ContVarName};
    use std::collections::BTreeSet;

    fn beta_leaf_at(name: ContVarName, a: f64, b: f64, coeff: RealEps) -> PosteriorSpn {
        posterior_leaf(coeff, PosteriorLeaf::Beta(BetaPrior::Beta { name, a, b }))
    }

    fn beta_leaf(name: ContVarName, a: f64, b: f64) -> PosteriorSpn {
        beta_leaf_at(name, a, b, RealEps::from_log(0.0, 0))
    }

    fn re(log_coeff: f64, power: u32) -> RealEps {
        RealEps::from_log(log_coeff, power)
    }

    #[test]
    fn zero_and_one_sentinels() {
        let z = PosteriorSpn::zero();
        assert!(z.is_zero());
        assert!(z.is_scalar());
        assert!(z.scope().is_empty());

        let o = PosteriorSpn::one();
        assert!(o.is_one());
        assert!(o.is_scalar());
        assert!(o.scope().is_empty());
    }

    #[test]
    fn leaf_carries_prior_scope() {
        let leaf = beta_leaf(7, 1.0, 1.0);
        assert_eq!(
            leaf.scope(),
            &std::iter::once(7u64).collect::<BTreeSet<_>>()
        );
        assert!(!leaf.is_zero());
        assert!(!leaf.is_one());
    }


    #[test]
    fn sum_empty_is_zero() {
        let r = PosteriorSpn::sum(vec![]);
        assert!(r.is_zero());
    }

    #[test]
    fn sum_single_branch_collapses_to_child() {
        let leaf = beta_leaf_at(1, 1.0, 1.0, re(0.7, 0));
        let r = PosteriorSpn::sum(vec![leaf]);
        match r.kind() {
            PosteriorKind::Leaf(PosteriorLeaf::Beta(BetaPrior::Beta { name, .. })) => {
                assert_eq!(*name, 1);
            }
            other => panic!("expected leaf, got {:?}", other),
        }
        assert_eq!(r.log_coeff().power, 0);
        assert!((r.log_coeff().log_coeff - 0.7).abs() < 1e-12);
    }

    #[test]
    fn sum_drops_higher_power_branches_after_normalisation() {
        let l_low = beta_leaf_at(1, 1.0, 1.0, re(0.0, 0));
        let l_high = beta_leaf_at(1, 2.0, 2.0, re(1.0, 1));
        let r = PosteriorSpn::sum(vec![l_low, l_high]);
        match r.kind() {
            PosteriorKind::Leaf(PosteriorLeaf::Beta(BetaPrior::Beta { a, .. })) => {
                assert_eq!(*a, 1.0);
            }
            other => panic!("expected leaf, got {:?}", other),
        }
        assert_eq!(r.log_coeff().power, 0);
        assert!(r.log_coeff().log_coeff.abs() < 1e-12);
    }

    #[test]
    fn sum_two_branches_same_power_normalises() {
        let l1 = beta_leaf_at(1, 1.0, 1.0, re(0.5, 0));
        let l2 = beta_leaf_at(1, 2.0, 2.0, re(1.5, 0));
        let r = PosteriorSpn::sum(vec![l1, l2]);
        let expected_z = crate::utils::math::logsumexp(0.5, 1.5);
        assert_eq!(r.log_coeff().power, 0);
        assert!((r.log_coeff().log_coeff - expected_z).abs() < 1e-12);
        match r.kind() {
            PosteriorKind::Sum(children) => {
                assert_eq!(children.len(), 2);
                let lse = children
                    .iter()
                    .fold(RealEps::zero(), |acc, c| acc + c.log_coeff());
                assert_eq!(lse.power, 0);
                assert!(lse.log_coeff.abs() < 1e-12, "lse={}", lse.log_coeff);
            }
            other => panic!("expected sum, got {:?}", other),
        }
    }

    #[test]
    fn sum_drops_zero_children() {
        let l1 = beta_leaf_at(1, 1.0, 1.0, re(0.0, 0));
        let l_zero = PosteriorSpn::zero();
        let r = PosteriorSpn::sum(vec![l1, l_zero]);
        match r.kind() {
            PosteriorKind::Leaf(PosteriorLeaf::Beta(BetaPrior::Beta { a, .. })) => {
                assert_eq!(*a, 1.0);
            }
            other => panic!("expected leaf, got {:?}", other),
        }
        assert_eq!(r.log_coeff().power, 0);
        assert!(r.log_coeff().log_coeff.abs() < 1e-12);
    }


    #[test]
    fn product_empty_is_one() {
        let r = PosteriorSpn::product(vec![]);
        assert!(r.is_one());
    }

    #[test]
    fn product_skips_one_children() {
        let l = beta_leaf(1, 1.0, 1.0);
        let r = PosteriorSpn::product(vec![Spn::one(), l, Spn::one()]);
        match r.kind() {
            PosteriorKind::Leaf(_) => {}
            other => panic!("expected leaf after dropping ones, got {:?}", other),
        }
    }

    #[test]
    fn product_zero_short_circuits() {
        let l = beta_leaf(1, 1.0, 1.0);
        let r = PosteriorSpn::product(vec![l, Spn::zero()]);
        assert!(r.is_zero());
    }

    #[test]
    fn product_disjoint_children_keep_log_coeff() {
        let l1 = beta_leaf(1, 1.0, 1.0);
        let l2 = beta_leaf(2, 1.0, 1.0);
        let r = PosteriorSpn::product(vec![l1, l2]);
        match r.kind() {
            PosteriorKind::Product(children) => {
                assert_eq!(children.len(), 2);
                let scope: BTreeSet<ContVarName> = [1u64, 2].iter().copied().collect();
                assert_eq!(r.scope(), &scope);
            }
            other => panic!("expected product, got {:?}", other),
        }
    }


    #[test]
    fn components_leaf_yields_single_item() {
        let l = beta_leaf_at(5, 2.0, 3.0, re(0.4, 0));
        let comps: Vec<_> = l.components().collect();
        assert_eq!(comps.len(), 1);
        let (w, reg) = &comps[0];
        assert!((w.log_coeff - 0.4).abs() < 1e-12);
        assert_eq!(reg.beta.len(), 1);
        assert_eq!(reg.beta[0].name(), 5);
    }

    #[test]
    fn components_product_of_leaves_yields_single_item() {
        let p = PosteriorSpn::product(vec![beta_leaf(1, 1.0, 1.0), beta_leaf(2, 2.0, 2.0)]);
        let comps: Vec<_> = p.components().collect();
        assert_eq!(comps.len(), 1);
        assert_eq!(comps[0].1.beta.len(), 2);
    }

    #[test]
    fn components_sum_yields_one_per_branch() {
        let l1 = beta_leaf_at(1, 1.0, 1.0, re(0.0, 0));
        let l2 = beta_leaf_at(2, 2.0, 2.0, re(0.0, 0));
        let s = PosteriorSpn::sum(vec![l1, l2]);
        let comps: Vec<_> = s.components().collect();
        assert_eq!(comps.len(), 2);
        for (_w, reg) in &comps {
            assert_eq!(reg.beta.len(), 1);
        }
    }

    #[test]
    fn components_weights_logsumexp_to_root_coeff() {
        let l1 = beta_leaf_at(1, 1.0, 1.0, re(0.3, 0));
        let l2 = beta_leaf_at(2, 2.0, 2.0, re(0.8, 0));
        let s = PosteriorSpn::sum(vec![l1, l2]);
        let root = s.log_coeff();
        let comps: Vec<_> = s.components().collect();
        let total = comps.iter().fold(RealEps::zero(), |acc, (w, _)| acc + *w);
        assert_eq!(total.power, root.power);
        assert!((total.log_coeff - root.log_coeff).abs() < 1e-12);
    }

    #[test]
    fn components_product_of_sums_is_cartesian_product() {
        let a = PosteriorSpn::sum(vec![
            beta_leaf_at(1, 1.0, 1.0, re(0.0, 0)),
            beta_leaf_at(2, 1.0, 1.0, re(0.0, 0)),
        ]);
        let b = PosteriorSpn::sum(vec![
            beta_leaf_at(3, 1.0, 1.0, re(0.0, 0)),
            beta_leaf_at(4, 1.0, 1.0, re(0.0, 0)),
        ]);
        let p = PosteriorSpn::product(vec![a, b]);
        let comps: Vec<_> = p.components().collect();
        assert_eq!(comps.len(), 4);
        for (_w, reg) in &comps {
            assert_eq!(reg.beta.len(), 2);
        }
    }
}
