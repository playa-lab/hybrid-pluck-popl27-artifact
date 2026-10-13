
use ordered_float::NotNan;
use rsdd::util::semirings::Semiring;

use crate::inference::conjugate_pairs::{Assignment, ContVarName, PriorRegistry, SuffStat};
use crate::inference::spn::coeff::{SpnCoeff, SpnLeaf};
use crate::inference::spn::{posterior_leaf, PosteriorSpn};
use crate::utils::epsilon::RealEps;

pub use crate::inference::conjugate_pairs::evidence_leaf::{clear_intern_table, intern_table_size};
pub use crate::inference::conjugate_pairs::EvidenceLeaf;

use super::node::{Spn, SpnInner, SpnKind};
use std::cell::RefCell;
use std::collections::{BTreeSet, HashMap};
use std::fmt;
use std::ops;


impl Spn<EvidenceLeaf> {
    pub fn log_likelihood(&self, assignment: &Assignment) -> f64 {
        let mut cache: HashMap<*const SpnInner<EvidenceLeaf>, f64> = HashMap::new();
        log_likelihood_memo(self, assignment, &mut cache)
    }

    pub fn posterior(
        &self,
        priors: &PriorRegistry,
        query_vars: &BTreeSet<ContVarName>,
    ) -> super::posterior::PosteriorSpn {
        let ctx = MemoCtx::new(query_vars);
        posterior_memo(self, priors, &ctx)
    }
}

pub(crate) struct MemoCtx<'a> {
    pub(crate) query_vars: &'a BTreeSet<ContVarName>,
    pub(crate) post:
        RefCell<HashMap<*const SpnInner<EvidenceLeaf>, super::posterior::PosteriorSpn>>,
}

impl<'a> MemoCtx<'a> {
    pub(crate) fn new(query_vars: &'a BTreeSet<ContVarName>) -> Self {
        Self {
            query_vars,
            post: RefCell::new(HashMap::new()),
        }
    }
}

fn leaf_log_likelihood(lk: &EvidenceLeaf, a: &Assignment) -> f64 {
    crate::for_each_evidence_family!(lk, |s| s.log_likelihood_in(a))
}

fn leaf_posterior(
    lk: &EvidenceLeaf,
    p: &PriorRegistry,
    query_vars: &BTreeSet<ContVarName>,
) -> super::posterior::PosteriorSpn {
    let (maybe_posterior, log_z) = crate::for_each_evidence_family!(lk, |s| {
        let (mp, log_z) = s.posterior_in(p, query_vars);
        (mp.map(Into::into), log_z)
    });
    if let Some(posterior) = maybe_posterior {
        posterior_leaf(log_z, posterior)
    } else {
        PosteriorSpn::from_coeff(log_z)
    }
}

pub(crate) fn posterior_memo(
    spn: &Spn<EvidenceLeaf>,
    priors: &PriorRegistry,
    ctx: &MemoCtx,
) -> super::posterior::PosteriorSpn {
    if spn.is_zero() {
        return PosteriorSpn::zero();
    }
    if spn.is_scalar() {
        return PosteriorSpn::from_coeff(RealEps::from_log(spn.log_coeff().into_inner(), 0));
    }

    let key = spn.inner_ptr();
    if let Some(hit) = ctx.post.borrow().get(&key) {
        return hit.scalar_mul(spn.log_coeff().into_inner());
    }

    let v: PosteriorSpn = match spn.kind() {
        SpnKind::Scalar => PosteriorSpn::one(),
        SpnKind::Leaf(lk) => leaf_posterior(lk, priors, ctx.query_vars),
        SpnKind::Product(children) => {
            let kids: Vec<PosteriorSpn> = children
                .iter()
                .map(|c| {
                    let sub = priors.slice(c.scope());
                    posterior_memo(c, &sub, ctx)
                })
                .collect();
            PosteriorSpn::product(kids)
        }
        SpnKind::Sum(children) => {
            let posterior_children: Vec<PosteriorSpn> = children
                .iter()
                .map(|c| posterior_memo(c, priors, ctx))
                .filter(|p| !p.is_zero())
                .collect();
            PosteriorSpn::sum(posterior_children)
        }
    };

    ctx.post.borrow_mut().insert(key, v.clone());
    v.scalar_mul(spn.log_coeff().into_inner())
}

fn log_likelihood_memo(
    spn: &Spn<EvidenceLeaf>,
    assignment: &Assignment,
    cache: &mut HashMap<*const SpnInner<EvidenceLeaf>, f64>,
) -> f64 {
    if spn.is_zero() {
        return f64::NEG_INFINITY;
    }
    let key = spn.inner_ptr();
    let inner_ll = if let Some(&hit) = cache.get(&key) {
        hit
    } else {
        let v = match spn.kind() {
            SpnKind::Scalar => 0.0,
            SpnKind::Leaf(lk) => leaf_log_likelihood(lk, assignment),
            SpnKind::Product(children) => {
                let mut total = 0.0;
                for c in children {
                    let cll = log_likelihood_memo(c, assignment, cache);
                    if cll == f64::NEG_INFINITY {
                        total = f64::NEG_INFINITY;
                        break;
                    }
                    total += cll;
                }
                total
            }
            SpnKind::Sum(children) => crate::utils::math::logsumexp_many(
                children
                    .iter()
                    .map(|c| log_likelihood_memo(c, assignment, cache)),
            ),
        };
        cache.insert(key, v);
        v
    };
    inner_ll + spn.log_coeff().into_inner()
}


thread_local! {
    static MUL_CACHE: RefCell<HashMap<(usize, usize), Spn<EvidenceLeaf>>> =
        RefCell::new(HashMap::new());
}

pub fn clear_mul_cache() {
    MUL_CACHE.with(|c| c.borrow_mut().clear());
}

fn spn_mul(a: &Spn<EvidenceLeaf>, b: &Spn<EvidenceLeaf>) -> Spn<EvidenceLeaf> {
    if a.is_zero() || b.is_zero() {
        return Spn::zero();
    }

    if a.is_one() {
        return b.clone();
    }
    if b.is_one() {
        return a.clone();
    }

    if a.is_scalar() {
        return b.scalar_mul(a.log_coeff.into_inner());
    }
    if b.is_scalar() {
        return a.scalar_mul(b.log_coeff.into_inner());
    }

    if a.scope().is_disjoint(b.scope()) {
        return Spn::product(vec![a.clone(), b.clone()]);
    }

    mul_overlapping(a, b)
}

fn mul_overlapping(a: &Spn<EvidenceLeaf>, b: &Spn<EvidenceLeaf>) -> Spn<EvidenceLeaf> {
    let combined_log_coeff = a.log_coeff + b.log_coeff;
    let a_norm = Spn {
        log_coeff: NotNan::new(0.0).unwrap(),
        inner: a.inner.clone(),
    };
    let b_norm = Spn {
        log_coeff: NotNan::new(0.0).unwrap(),
        inner: b.inner.clone(),
    };

    let pa = a_norm.inner_ptr() as usize;
    let pb = b_norm.inner_ptr() as usize;
    let key = if pa <= pb { (pa, pb) } else { (pb, pa) };
    if let Some(hit) = MUL_CACHE.with(|c| c.borrow().get(&key).cloned()) {
        return hit.scalar_mul(combined_log_coeff.into_inner());
    }

    let result = match (a_norm.kind(), b_norm.kind()) {
        (SpnKind::Leaf(l1), SpnKind::Leaf(l2)) => mul_leaf_into_leaf(l1, l2),

        (SpnKind::Sum(children), _) => {
            let results: Vec<Spn<EvidenceLeaf>> =
                children.iter().map(|c| spn_mul(c, &b_norm)).collect();
            Spn::sum(results)
        }
        (_, SpnKind::Sum(children)) => {
            let results: Vec<Spn<EvidenceLeaf>> =
                children.iter().map(|c| spn_mul(&a_norm, c)).collect();
            Spn::sum(results)
        }

        (SpnKind::Product(_), SpnKind::Product(b_children)) => {
            let mut acc = a_norm.clone();
            for b_child in b_children {
                acc = spn_mul(&acc, b_child);
                if acc.is_zero() {
                    return Spn::zero();
                }
            }
            acc
        }

        (SpnKind::Product(p_children), SpnKind::Leaf(_)) => {
            mul_leaf_into_product(p_children, &b_norm)
        }
        (SpnKind::Leaf(_), SpnKind::Product(p_children)) => {
            mul_leaf_into_product(p_children, &a_norm)
        }

        (SpnKind::Scalar, _) | (_, SpnKind::Scalar) => {
            unreachable!("scalar operands are short-circuited by spn_mul")
        }
    };

    MUL_CACHE.with(|c| c.borrow_mut().insert(key, result.clone()));
    result.scalar_mul(combined_log_coeff.into_inner())
}

fn mul_leaf_into_leaf(l1: &EvidenceLeaf, l2: &EvidenceLeaf) -> Spn<EvidenceLeaf> {
    let (merged, log_factor) = l1.try_merge(l2).expect(
        "leaf*leaf with overlapping scope must agree on EvidenceLeaf: \
         each variable has a fixed distribution type",
    );
    if log_factor == f64::NEG_INFINITY {
        Spn::zero()
    } else {
        Spn::leaf(merged).scalar_mul(log_factor)
    }
}

fn mul_leaf_into_product(
    product_children: &[Spn<EvidenceLeaf>],
    leaf: &Spn<EvidenceLeaf>,
) -> Spn<EvidenceLeaf> {
    debug_assert!(matches!(leaf.kind(), SpnKind::Leaf(_)));
    debug_assert!(leaf.log_coeff.into_inner() == 0.0);

    let leaf_scope = leaf.scope();
    let mut acc = leaf.clone();
    let mut kept: Vec<Spn<EvidenceLeaf>> = Vec::with_capacity(product_children.len() + 1);

    for child in product_children {
        if child.scope().is_disjoint(leaf_scope) {
            kept.push(child.clone());
        } else {
            acc = spn_mul(&acc, child);
            if acc.is_zero() {
                return Spn::zero();
            }
        }
    }

    kept.push(acc);
    Spn::product(kept)
}


impl ops::Add<Spn<EvidenceLeaf>> for Spn<EvidenceLeaf> {
    type Output = Spn<EvidenceLeaf>;
    fn add(self, rhs: Spn<EvidenceLeaf>) -> Spn<EvidenceLeaf> {
        if self.is_scalar() && rhs.is_scalar() {
            return Spn::from_coeff(self.log_coeff.logsumexp(rhs.log_coeff));
        }
        Spn::sum(vec![self, rhs])
    }
}

impl ops::Mul<Spn<EvidenceLeaf>> for Spn<EvidenceLeaf> {
    type Output = Spn<EvidenceLeaf>;
    fn mul(self, rhs: Spn<EvidenceLeaf>) -> Spn<EvidenceLeaf> {
        spn_mul(&self, &rhs)
    }
}

impl Semiring for Spn<EvidenceLeaf> {
    fn zero() -> Self {
        Spn::zero()
    }
    fn one() -> Self {
        Spn::one()
    }
}

impl<L: SpnLeaf> fmt::Display for Spn<L> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SPN")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::conjugate_pairs::beta::BetaPrior;
    use crate::inference::conjugate_pairs::gaussian::AffineConstraint;
    use crate::inference::conjugate_pairs::{BetaSuffStat, GaussianObs, SuffStat};
    use crate::utils::math::logsumexp;
    use std::collections::BTreeMap;
    use std::rc::Rc;

    fn beta_leaf(coeff: f64, stat: BetaSuffStat) -> Spn<EvidenceLeaf> {
        Spn::leaf(EvidenceLeaf::Beta(stat)).scalar_mul(coeff)
    }

    fn gaussian_leaf(coeff: f64, obs: GaussianObs) -> Spn<EvidenceLeaf> {
        Spn::leaf(EvidenceLeaf::Gaussian(obs)).scalar_mul(coeff)
    }

    fn gaussian_eq(coefs: &[(u64, f64)], v: f64) -> AffineConstraint {
        AffineConstraint {
            coefficients: coefs
                .iter()
                .map(|&(k, c)| (k, NotNan::new(c).unwrap()))
                .collect(),
            value: NotNan::new(v).unwrap(),
        }
    }

    fn collect_leaf_vars(s: &Spn<EvidenceLeaf>) -> BTreeSet<ContVarName> {
        s.scope().clone()
    }

    #[test]
    fn one_is_identity() {
        let leaf = beta_leaf(0.0, BetaSuffStat::counts(1, 2, 3));
        let one = Spn::<EvidenceLeaf>::one();
        let p1 = leaf.clone() * one.clone();
        assert!(Rc::ptr_eq(&p1.inner, &leaf.inner));
        let p2 = one * leaf.clone();
        assert!(Rc::ptr_eq(&p2.inner, &leaf.inner));
    }

    #[test]
    fn scalar_times_leaf_is_scaled_leaf() {
        let leaf = beta_leaf(0.0, BetaSuffStat::counts(1, 2, 3));
        let scalar = Spn::scalar(0.5_f64.ln());
        let prod = leaf.clone() * scalar;
        assert!(Rc::ptr_eq(&prod.inner, &leaf.inner));
        assert!((prod.log_coeff().into_inner() - 0.5_f64.ln()).abs() < 1e-12);
    }

    #[test]
    fn zero_absorbs_everything() {
        let leaf = beta_leaf(0.0, BetaSuffStat::counts(1, 2, 3));
        let z = Spn::zero();
        assert!((z.clone() * leaf.clone()).is_zero());
        assert!((leaf * z).is_zero());
    }

    #[test]
    fn product_of_two_beta_leaves_then_mul_into_third() {
        let p = Spn::product(vec![
            beta_leaf(0.0, BetaSuffStat::counts(1, 1, 0)),
            beta_leaf(0.0, BetaSuffStat::counts(2, 1, 0)),
        ]);
        let q = p * beta_leaf(0.0, BetaSuffStat::counts(3, 1, 0));
        assert_eq!(
            collect_leaf_vars(&q),
            [1u64, 2, 3].iter().copied().collect()
        );
        match q.kind() {
            SpnKind::Product(children) => assert_eq!(children.len(), 3),
            other => panic!("expected Product, got {:?}", other),
        }
    }

    #[test]
    fn product_of_betas_times_leaf_on_existing_var() {
        let p = Spn::product(vec![
            beta_leaf(0.0, BetaSuffStat::counts(1, 2, 1)),
            beta_leaf(0.0, BetaSuffStat::counts(2, 3, 0)),
        ]);
        let q = p * beta_leaf(0.0, BetaSuffStat::counts(1, 1, 1));
        match q.kind() {
            SpnKind::Product(children) => {
                assert_eq!(children.len(), 2);
                let mut saw_merged_var1 = false;
                let mut saw_var2 = false;
                for c in children {
                    match c.kind() {
                        SpnKind::Leaf(EvidenceLeaf::Beta(BetaSuffStat::Counts {
                            name,
                            h,
                            t,
                            ..
                        })) => {
                            if *name == 1 {
                                assert_eq!(*h, 3);
                                assert_eq!(*t, 2);
                                saw_merged_var1 = true;
                            } else if *name == 2 {
                                assert_eq!(*h, 3);
                                assert_eq!(*t, 0);
                                saw_var2 = true;
                            }
                        }
                        _ => panic!("unexpected child kind"),
                    }
                }
                assert!(saw_merged_var1 && saw_var2);
            }
            _ => panic!("expected Product"),
        }
    }

    #[test]
    fn gaussian_leaf_overlapping_multiple_product_children() {
        let g1 = gaussian_leaf(
            0.0,
            GaussianObs::from_observation(
                &BTreeMap::from([(1u64, 1.0)]),
                0.5,
                BTreeSet::from([1u64]),
            ),
        );
        let g2 = gaussian_leaf(
            0.0,
            GaussianObs::from_observation(
                &BTreeMap::from([(2u64, 1.0)]),
                1.5,
                BTreeSet::from([2u64]),
            ),
        );
        let b3 = beta_leaf(0.0, BetaSuffStat::counts(3, 1, 0));
        let p = Spn::product(vec![g1, g2, b3]);

        let coupling = gaussian_leaf(
            0.0,
            GaussianObs::new(
                vec![gaussian_eq(&[(1, 1.0), (2, 1.0)], 2.0)],
                vec![],
                BTreeSet::from([1u64, 2]),
            ),
        );

        let q = p * coupling;

        match q.kind() {
            SpnKind::Product(children) => {
                assert_eq!(children.len(), 2);
                let mut saw_merged_gaussian = false;
                let mut saw_beta = false;
                for c in children {
                    match c.kind() {
                        SpnKind::Leaf(EvidenceLeaf::Gaussian(g)) => {
                            let scope: BTreeSet<ContVarName> = g.scope().copied().collect();
                            assert_eq!(scope, [1u64, 2].iter().copied().collect::<BTreeSet<_>>());
                            saw_merged_gaussian = true;
                        }
                        SpnKind::Leaf(EvidenceLeaf::Beta(BetaSuffStat::Counts {
                            name: 3, ..
                        })) => saw_beta = true,
                        _ => panic!("unexpected child kind"),
                    }
                }
                assert!(
                    saw_merged_gaussian && saw_beta,
                    "expected merged Gaussian + beta child"
                );
            }
            _ => panic!("expected Product"),
        }
    }

    #[test]
    fn contradictory_beta_merge_yields_zero() {
        let a = beta_leaf(0.0, BetaSuffStat::close(1, 0.4));
        let b = beta_leaf(0.0, BetaSuffStat::close(1, 0.5));
        let q = a * b;
        assert!(q.is_zero());
    }

    #[test]
    fn hashconsing_dedupes_structurally_equal_leaves() {
        let a = beta_leaf(0.0, BetaSuffStat::counts(1, 2, 3));
        let b = beta_leaf(0.0, BetaSuffStat::counts(1, 2, 3));
        assert!(Rc::ptr_eq(&a.inner, &b.inner));
    }


    fn nn(x: f64) -> NotNan<f64> {
        NotNan::new(x).unwrap()
    }

    fn beta_assignment(entries: &[(ContVarName, f64)]) -> Assignment {
        let mut beta: Vec<(ContVarName, NotNan<f64>)> =
            entries.iter().map(|&(n, v)| (n, nn(v))).collect();
        beta.sort_by_key(|(n, _)| *n);
        Assignment::from_sorted(beta, vec![], vec![], vec![])
    }

    #[test]
    fn log_likelihood_single_beta_leaf() {
        let leaf = beta_leaf(0.0, BetaSuffStat::counts(1, 2, 1));
        let a = beta_assignment(&[(1, 0.4)]);
        let ll = leaf.log_likelihood(&a);
        let expected = 2.0 * 0.4_f64.ln() + 0.6_f64.ln();
        assert!((ll - expected).abs() < 1e-12);
    }

    #[test]
    fn log_likelihood_product_sums_logs() {
        let p = Spn::product(vec![
            beta_leaf(0.0, BetaSuffStat::counts(1, 2, 1)),
            beta_leaf(0.0, BetaSuffStat::counts(2, 1, 2)),
        ]);
        let a = beta_assignment(&[(1, 0.3), (2, 0.5)]);
        let ll = p.log_likelihood(&a);
        let e1 = 2.0 * 0.3_f64.ln() + 0.7_f64.ln();
        let e2 = 0.5_f64.ln() + 2.0 * 0.5_f64.ln();
        assert!((ll - (e1 + e2)).abs() < 1e-12);
    }

    #[test]
    fn log_likelihood_sum_is_logsumexp_of_branches() {
        let l1 = beta_leaf(0.0, BetaSuffStat::counts(1, 2, 1));
        let l2 = beta_leaf(0.0, BetaSuffStat::counts(1, 1, 2));
        let s = Spn::sum(vec![l1.clone(), l2.clone()]);
        let a = beta_assignment(&[(1, 0.4)]);
        let ll = s.log_likelihood(&a);
        let ll1 = l1.log_likelihood(&a);
        let ll2 = l2.log_likelihood(&a);
        let expected = logsumexp(ll1, ll2);
        assert!(
            (ll - expected).abs() < 1e-12,
            "ll={} expected={}",
            ll,
            expected
        );
    }

    #[test]
    fn log_likelihood_outer_log_coeff_adds() {
        let leaf = beta_leaf(0.0, BetaSuffStat::counts(1, 2, 1));
        let scaled = leaf.scalar_mul(0.7);
        let a = beta_assignment(&[(1, 0.4)]);
        let base = leaf.log_likelihood(&a);
        let lifted = scaled.log_likelihood(&a);
        assert!((lifted - (base + 0.7)).abs() < 1e-12);
    }

    #[test]
    fn log_likelihood_zero_short_circuits() {
        let z = Spn::<EvidenceLeaf>::zero();
        let a = beta_assignment(&[(1, 0.4)]);
        assert_eq!(z.log_likelihood(&a), f64::NEG_INFINITY);
    }

    #[test]
    fn log_likelihood_pure_scalar_is_log_coeff() {
        let s = Spn::<EvidenceLeaf>::scalar(1.3);
        let a = Assignment::from_sorted(vec![], vec![], vec![], vec![]);
        assert!((s.log_likelihood(&a) - 1.3).abs() < 1e-12);
    }

    #[test]
    fn log_likelihood_realeq_branch_returns_neg_infinity_on_mismatch() {
        let leaf = beta_leaf(0.0, BetaSuffStat::close(1, 0.4));
        let a_match = beta_assignment(&[(1, 0.4)]);
        let a_miss = beta_assignment(&[(1, 0.5)]);
        assert_eq!(leaf.log_likelihood(&a_match), 0.0);
        assert_eq!(leaf.log_likelihood(&a_miss), f64::NEG_INFINITY);
    }


    fn beta_prior(name: ContVarName, a: f64, b: f64) -> BetaPrior {
        BetaPrior::Beta { name, a, b }
    }

    fn registry_of_betas(entries: &[(ContVarName, f64, f64)]) -> PriorRegistry {
        let mut betas: Vec<_> = entries
            .iter()
            .map(|&(n, a, b)| beta_prior(n, a, b))
            .collect();
        betas.sort_by_key(BetaPrior::name);
        PriorRegistry::from_sorted(betas, vec![], vec![], vec![])
    }

    #[test]
    fn log_marginal_single_leaf_matches_suffstat() {
        let leaf = beta_leaf(0.0, BetaSuffStat::counts(1, 2, 3));
        let reg = registry_of_betas(&[(1, 1.0, 1.0)]);
        let direct = match leaf.kind() {
            SpnKind::Leaf(EvidenceLeaf::Beta(s)) => {
                s.posterior(reg.beta_prior(1), &BTreeSet::new())
            }
            _ => unreachable!(),
        };
        let via_spn = leaf.posterior(&reg, &BTreeSet::new());
        assert_eq!(direct.1.power, via_spn.log_coeff.power);
        assert!((direct.1.log_coeff - via_spn.log_coeff.log_coeff).abs() < 1e-12);
    }

    #[test]
    fn log_marginal_product_multiplies_independent_factors() {
        let p = Spn::product(vec![
            beta_leaf(0.0, BetaSuffStat::counts(1, 2, 1)),
            beta_leaf(0.0, BetaSuffStat::counts(2, 1, 3)),
        ]);
        let reg = registry_of_betas(&[(1, 1.0, 1.0), (2, 1.0, 1.0)]);
        let total = p.posterior(&reg, &BTreeSet::new()).log_coeff;
        let m1 = match beta_leaf(0.0, BetaSuffStat::counts(1, 2, 1)).kind() {
            SpnKind::Leaf(EvidenceLeaf::Beta(s)) => {
                s.posterior(reg.beta_prior(1), &BTreeSet::new()).1
            }
            _ => unreachable!(),
        };
        let m2 = match beta_leaf(0.0, BetaSuffStat::counts(2, 1, 3)).kind() {
            SpnKind::Leaf(EvidenceLeaf::Beta(s)) => {
                s.posterior(reg.beta_prior(2), &BTreeSet::new()).1
            }
            _ => unreachable!(),
        };
        let expected = m1 * m2;
        assert_eq!(total.power, expected.power);
        assert!((total.log_coeff - expected.log_coeff).abs() < 1e-12);
    }

    #[test]
    fn log_marginal_sum_adds_branches() {
        let l1 = beta_leaf(0.0, BetaSuffStat::counts(1, 2, 1));
        let l2 = beta_leaf(0.0, BetaSuffStat::counts(1, 1, 2));
        let s = Spn::sum(vec![l1.clone(), l2.clone()]);
        let reg = registry_of_betas(&[(1, 1.0, 1.0)]);
        let total = s.posterior(&reg, &BTreeSet::new()).log_coeff;
        let m1 = l1.posterior(&reg, &BTreeSet::new()).log_coeff;
        let m2 = l2.posterior(&reg, &BTreeSet::new()).log_coeff;
        let expected = m1 + m2;
        assert_eq!(total.power, expected.power);
        assert!((total.log_coeff - expected.log_coeff).abs() < 1e-12);
    }

    #[test]
    fn log_marginal_mixed_power_sum_drops_higher_power() {
        let counts = beta_leaf(0.0, BetaSuffStat::counts(1, 2, 1));
        let pinned = beta_leaf(0.0, BetaSuffStat::close(1, 0.4));
        let s = Spn::sum(vec![counts.clone(), pinned]);
        let reg = registry_of_betas(&[(1, 1.0, 1.0)]);
        let total = s.posterior(&reg, &BTreeSet::new()).log_coeff;
        let counts_only = counts.posterior(&reg, &BTreeSet::new()).log_coeff;
        assert_eq!(total.power, 0);
        assert!((total.log_coeff - counts_only.log_coeff).abs() < 1e-12);
    }

    #[test]
    fn log_marginal_zero_short_circuits() {
        let z = Spn::<EvidenceLeaf>::zero();
        let reg = registry_of_betas(&[]);
        assert!(z.posterior(&reg, &BTreeSet::new()).is_zero());
    }

    #[test]
    fn log_marginal_outer_log_coeff_scales() {
        let leaf = beta_leaf(0.0, BetaSuffStat::counts(1, 2, 1));
        let scaled = leaf.scalar_mul(0.5);
        let reg = registry_of_betas(&[(1, 1.0, 1.0)]);
        let base = leaf.posterior(&reg, &BTreeSet::new());
        let lifted = scaled.posterior(&reg, &BTreeSet::new());
        assert_eq!(base.log_coeff.power, lifted.log_coeff.power);
        assert!((lifted.log_coeff.log_coeff - (base.log_coeff.log_coeff + 0.5)).abs() < 1e-12);
    }


    #[test]
    fn sum_factors_log_z_into_outer_coeff() {
        let l1 = beta_leaf(0.0, BetaSuffStat::counts(1, 2, 1)).scalar_mul(2.0);
        let l2 = beta_leaf(0.0, BetaSuffStat::counts(1, 1, 2)).scalar_mul(3.0);
        let s = Spn::sum(vec![l1, l2]);
        let expected_z = logsumexp(2.0, 3.0);
        assert!(
            (s.log_coeff().into_inner() - expected_z).abs() < 1e-12,
            "outer log_coeff should equal log_Z={} got {}",
            expected_z,
            s.log_coeff().into_inner()
        );
        match s.kind() {
            SpnKind::Sum(children) => {
                let lse = children.iter().fold(f64::NEG_INFINITY, |acc, c| {
                    logsumexp(acc, c.log_coeff().into_inner())
                });
                assert!(
                    lse.abs() < 1e-12,
                    "normalised child weights should logsumexp to 0, got {}",
                    lse
                );
            }
            other => panic!("expected Sum, got {:?}", other),
        }
    }


    use crate::inference::spn::posterior::{PosteriorKind, PosteriorLeaf};

    #[test]
    fn posterior_single_beta_leaf_full_query() {
        let leaf = beta_leaf(0.0, BetaSuffStat::counts(1, 2, 1));
        let reg = registry_of_betas(&[(1, 1.0, 1.0)]);
        let q: BTreeSet<ContVarName> = [1u64].iter().copied().collect();
        let post = leaf.posterior(&reg, &q);
        match post.kind() {
            PosteriorKind::Leaf(PosteriorLeaf::Beta(
                crate::inference::conjugate_pairs::beta::BetaPrior::Beta { name, a, b },
            )) => {
                assert_eq!(*name, 1);
                assert!((*a - 3.0).abs() < 1e-12);
                assert!((*b - 2.0).abs() < 1e-12);
            }
            other => panic!("expected Beta leaf, got {:?}", other),
        }
    }

    #[test]
    fn posterior_empty_query_returns_marginal_scalar() {
        let leaf = beta_leaf(0.0, BetaSuffStat::counts(1, 2, 1));
        let reg = registry_of_betas(&[(1, 1.0, 1.0)]);
        let q: BTreeSet<ContVarName> = BTreeSet::new();
        let post = leaf.posterior(&reg, &q);
        assert!(matches!(post.kind(), PosteriorKind::Scalar));
        assert!(post.scope().is_empty());
        let expected_lml = leaf.posterior(&reg, &BTreeSet::new()).log_coeff;
        assert_eq!(post.log_coeff().power, expected_lml.power);
        assert!((post.log_coeff().log_coeff - expected_lml.log_coeff).abs() < 1e-12);
    }

    #[test]
    fn posterior_disjoint_query_returns_marginal_scalar() {
        let leaf = beta_leaf(0.0, BetaSuffStat::counts(1, 2, 1));
        let reg = registry_of_betas(&[(1, 1.0, 1.0)]);
        let q: BTreeSet<ContVarName> = [99u64].iter().copied().collect();
        let post = leaf.posterior(&reg, &q);
        assert!(matches!(post.kind(), PosteriorKind::Scalar));
        let expected_lml = leaf.posterior(&reg, &BTreeSet::new()).log_coeff;
        assert_eq!(post.log_coeff().power, expected_lml.power);
        assert!((post.log_coeff().log_coeff - expected_lml.log_coeff).abs() < 1e-12);
    }

    #[test]
    fn posterior_zero_returns_zero() {
        let z = Spn::<EvidenceLeaf>::zero();
        let reg = registry_of_betas(&[]);
        let q: BTreeSet<ContVarName> = BTreeSet::new();
        assert!(z.posterior(&reg, &q).is_zero());
    }

    #[test]
    fn posterior_pure_scalar_returns_scalar() {
        let s = Spn::<EvidenceLeaf>::scalar(0.7);
        let reg = registry_of_betas(&[]);
        let q: BTreeSet<ContVarName> = BTreeSet::new();
        let post = s.posterior(&reg, &q);
        assert!(matches!(post.kind(), PosteriorKind::Scalar));
        assert_eq!(post.log_coeff().power, 0);
        assert!((post.log_coeff().log_coeff - 0.7).abs() < 1e-12);
    }

    #[test]
    fn posterior_product_partial_query_drops_unqueried_factor() {
        let p = Spn::product(vec![
            beta_leaf(0.0, BetaSuffStat::counts(1, 2, 1)),
            beta_leaf(0.0, BetaSuffStat::counts(2, 1, 3)),
        ]);
        let reg = registry_of_betas(&[(1, 1.0, 1.0), (2, 1.0, 1.0)]);
        let q: BTreeSet<ContVarName> = [1u64].iter().copied().collect();
        let post = p.posterior(&reg, &q);
        match post.kind() {
            PosteriorKind::Leaf(PosteriorLeaf::Beta(
                crate::inference::conjugate_pairs::beta::BetaPrior::Beta { name, a, b },
            )) => {
                assert_eq!(*name, 1);
                assert!((*a - 3.0).abs() < 1e-12);
                assert!((*b - 2.0).abs() < 1e-12);
            }
            other => panic!("expected Beta leaf, got {:?}", other),
        }
        let leaf1 = beta_leaf(0.0, BetaSuffStat::counts(1, 2, 1));
        let leaf2 = beta_leaf(0.0, BetaSuffStat::counts(2, 1, 3));
        let lml1 = leaf1.posterior(&reg, &BTreeSet::new()).log_coeff;
        let lml2 = leaf2.posterior(&reg, &BTreeSet::new()).log_coeff;
        let expected = lml1 * lml2;
        assert_eq!(post.log_coeff().power, expected.power);
        assert!((post.log_coeff().log_coeff - expected.log_coeff).abs() < 1e-12);
    }

    #[test]
    fn posterior_sum_full_query_pushes_log_z_to_outer() {
        let l1 = beta_leaf(0.0, BetaSuffStat::counts(1, 2, 1));
        let l2 = beta_leaf(0.0, BetaSuffStat::counts(1, 1, 2));
        let s = Spn::sum(vec![l1.clone(), l2.clone()]);
        let reg = registry_of_betas(&[(1, 1.0, 1.0)]);
        let q: BTreeSet<ContVarName> = [1u64].iter().copied().collect();
        let post = s.posterior(&reg, &q);

        let lml1 = l1.posterior(&reg, &BTreeSet::new()).log_coeff;
        let lml2 = l2.posterior(&reg, &BTreeSet::new()).log_coeff;
        let expected_z = lml1 + lml2;

        assert_eq!(post.log_coeff().power, expected_z.power);
        assert!((post.log_coeff().log_coeff - expected_z.log_coeff).abs() < 1e-12);

        match post.kind() {
            PosteriorKind::Sum(children) => {
                assert_eq!(children.len(), 2);
                let lse = children
                    .iter()
                    .fold(RealEps::zero(), |acc, c| acc + c.log_coeff());
                assert_eq!(lse.power, 0);
                assert!(lse.log_coeff.abs() < 1e-12, "logsumexp = {}", lse.log_coeff);
            }
            other => panic!("expected Sum, got {:?}", other),
        }
    }

    #[test]
    fn posterior_gaussian_full_query_matches_suffstat() {
        use crate::inference::conjugate_pairs::gaussian::{GaussianObs, GaussianPrior};
        use nalgebra as na;

        let prior_block = GaussianPrior {
            var_order: vec![1, 2],
            mean: na::DVector::from_vec(vec![0.0, 0.0]),
            cov: na::DMatrix::identity(2, 2),
        };
        let reg = PriorRegistry::from_sorted(vec![], vec![], vec![], vec![prior_block.clone()]);
        let obs = GaussianObs::new(
            vec![gaussian_eq(&[(1, 1.0), (2, 1.0)], 2.0)],
            vec![],
            BTreeSet::from([1u64, 2]),
        );
        let spn = gaussian_leaf(0.0, obs.clone());

        let q: BTreeSet<ContVarName> = [1u64, 2].iter().copied().collect();
        let post = spn.posterior(&reg, &q);

        let direct = obs.posterior(&prior_block, &q).0.unwrap();
        match post.kind() {
            PosteriorKind::Leaf(PosteriorLeaf::Gaussian(g)) => {
                assert_eq!(g.var_order, direct.var_order);
                for i in 0..g.mean.len() {
                    assert!((g.mean[i] - direct.mean[i]).abs() < 1e-10);
                }
                for i in 0..g.var_order.len() {
                    for j in 0..g.var_order.len() {
                        assert!((g.cov[(i, j)] - direct.cov[(i, j)]).abs() < 1e-10);
                    }
                }
            }
            other => panic!("expected Gaussian leaf, got {:?}", other),
        }
    }

    #[test]
    fn posterior_gaussian_partial_query_slices_correctly() {
        use crate::inference::conjugate_pairs::gaussian::{GaussianObs, GaussianPrior};
        use nalgebra as na;

        let prior_block = GaussianPrior {
            var_order: vec![1, 2],
            mean: na::DVector::from_vec(vec![0.0, 0.0]),
            cov: na::DMatrix::identity(2, 2),
        };
        let reg = PriorRegistry::from_sorted(vec![], vec![], vec![], vec![prior_block.clone()]);
        let obs = GaussianObs::new(
            vec![gaussian_eq(&[(1, 1.0), (2, 1.0)], 2.0)],
            vec![],
            BTreeSet::from([1u64, 2]),
        );
        let spn = gaussian_leaf(0.0, obs.clone());

        let q: BTreeSet<ContVarName> = [1u64].iter().copied().collect();
        let post = spn.posterior(&reg, &q);

        let direct = obs.posterior(&prior_block, &q).0.unwrap();
        match post.kind() {
            PosteriorKind::Leaf(PosteriorLeaf::Gaussian(g)) => {
                assert_eq!(g.var_order, vec![1]);
                assert_eq!(g.var_order, direct.var_order);
                assert!((g.mean[0] - direct.mean[0]).abs() < 1e-10);
                assert!((g.cov[(0, 0)] - direct.cov[(0, 0)]).abs() < 1e-10);
            }
            other => panic!("expected Gaussian leaf, got {:?}", other),
        }
    }

    #[test]
    fn block_closed_disjoint_obs_merge_and_match_det1_rewrite() {
        use crate::inference::conjugate_pairs::gaussian::{GaussianObs, GaussianPrior};
        use nalgebra as na;

        let prior_block = GaussianPrior {
            var_order: vec![1, 2],
            mean: na::DVector::from_vec(vec![0.0, 0.0]),
            cov: na::DMatrix::from_row_slice(2, 2, &[1.0, 0.5, 0.5, 1.0]),
        };
        let reg = PriorRegistry::from_sorted(vec![], vec![], vec![], vec![prior_block]);
        let closure = reg.gaussian_block_closure([1u64].iter());
        assert_eq!(closure, [1u64, 2].iter().copied().collect::<BTreeSet<_>>());

        let single_obs = |var: u64, v: f64| {
            gaussian_leaf(
                0.0,
                GaussianObs::from_observation(&BTreeMap::from([(var, 1.0)]), v, closure.clone()),
            )
        };

        let disjoint_written = single_obs(1, 1.0) * single_obs(2, 2.0);
        assert!(
            matches!(disjoint_written.kind(), SpnKind::Leaf(_)),
            "block-closed same-block leaves must merge, not product"
        );

        let coupled = gaussian_leaf(
            0.0,
            GaussianObs::new(
                vec![gaussian_eq(&[(1, 1.0), (2, 1.0)], 3.0)],
                vec![],
                closure.clone(),
            ),
        );
        let overlap_written = single_obs(1, 1.0) * coupled;

        let q: BTreeSet<ContVarName> = BTreeSet::new();
        let z_disjoint = disjoint_written.posterior(&reg, &q).log_coeff();
        let z_overlap = overlap_written.posterior(&reg, &q).log_coeff();
        assert_eq!(z_disjoint.power, z_overlap.power);
        assert!((z_disjoint.log_coeff - z_overlap.log_coeff).abs() < 1e-10);

        let expected = -(2.0 * std::f64::consts::PI).ln() - 0.5 * 0.75_f64.ln() - 2.0;
        assert!(
            (z_disjoint.log_coeff - expected).abs() < 1e-10,
            "Z = {}, expected joint {}",
            z_disjoint.log_coeff,
            expected
        );
    }

    #[test]
    fn posterior_outer_log_coeff_propagates() {
        let leaf = beta_leaf(0.0, BetaSuffStat::counts(1, 2, 1));
        let scaled = leaf.scalar_mul(0.5);
        let reg = registry_of_betas(&[(1, 1.0, 1.0)]);
        let q: BTreeSet<ContVarName> = [1u64].iter().copied().collect();
        let p_base = leaf.posterior(&reg, &q);
        let p_lifted = scaled.posterior(&reg, &q);
        assert_eq!(p_base.log_coeff().power, p_lifted.log_coeff().power);
        assert!(
            (p_lifted.log_coeff().log_coeff - (p_base.log_coeff().log_coeff + 0.5)).abs() < 1e-12
        );
    }

    #[test]
    fn log_likelihood_gaussian_equality_satisfied() {
        let g = gaussian_leaf(
            0.0,
            crate::inference::conjugate_pairs::gaussian::GaussianObs::new(
                vec![gaussian_eq(&[(1, 1.0), (2, 1.0)], 3.0)],
                vec![],
                BTreeSet::from([1u64, 2]),
            ),
        );
        let a = Assignment::from_sorted(vec![], vec![], vec![], vec![(1, nn(1.0)), (2, nn(2.0))]);
        assert_eq!(g.log_likelihood(&a), 0.0);

        let a_off =
            Assignment::from_sorted(vec![], vec![], vec![], vec![(1, nn(1.0)), (2, nn(2.5))]);
        assert_eq!(g.log_likelihood(&a_off), f64::NEG_INFINITY);
    }
}
