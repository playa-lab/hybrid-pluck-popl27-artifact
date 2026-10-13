use std::collections::BTreeMap;
use std::fmt::{Display, Formatter, Result};

use ordered_float::NotNan;

use super::gamma_mixture::{GammaMixture, GammaTerm, SignedWeight};
use crate::inference::conjugate_pairs::{ContVarName, Scaled};
use crate::utils::epsilon::RealEps;
use crate::utils::intervals::IntervalOrEq;
use crate::utils::math::{ln_factorial, logsumexp_many};

type ScaledConstraint = Scaled<IntervalOrEq<u64>>;

#[derive(Debug, Clone, Hash, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(bound(
    serialize = "N: serde::Serialize + Ord",
    deserialize = "N: serde::Deserialize<'de> + Ord"
))]
pub struct PoissonPrior<N>(BTreeMap<N, ScaledConstraint>);

impl PoissonPrior<ContVarName> {
    pub fn unconstrained(vars: impl Iterator<Item = ContVarName>) -> Self {
        PoissonPrior(
            vars.map(|v| (v, Scaled::unit(IntervalOrEq::geq(0))))
                .collect(),
        )
    }

    pub fn conditioned_on(obs: &PoissonObs) -> Self {
        PoissonPrior(obs.0.clone())
    }

    pub fn restricted_to(&self, vars: &std::collections::BTreeSet<ContVarName>) -> Self {
        PoissonPrior(
            self.0
                .iter()
                .filter(|(n, _)| vars.contains(n))
                .map(|(n, c)| (*n, c.clone()))
                .collect(),
        )
    }

    pub fn sample<R: rand::Rng>(
        &self,
        lambda: NotNan<f64>,
        rng: &mut R,
    ) -> BTreeMap<ContVarName, NotNan<f64>> {
        self.0
            .iter()
            .map(|(var, scaled)| {
                let rate = lambda * scaled.scale;
                let k = sample_truncated(&scaled.constraint, rate, rng);
                (*var, NotNan::new(k as f64).unwrap())
            })
            .collect()
    }

    pub fn map_names<M: Ord>(&self, f: impl Fn(ContVarName) -> M) -> PoissonPrior<M> {
        PoissonPrior(self.0.iter().map(|(n, c)| (f(*n), c.clone())).collect())
    }

    pub fn scope(&self) -> impl Iterator<Item = &ContVarName> + '_ {
        self.0.keys()
    }
}

impl<N> PoissonPrior<N> {
    pub fn constraints(&self) -> &BTreeMap<N, ScaledConstraint> {
        &self.0
    }
}

impl<N: Display> PoissonPrior<N> {
    pub fn fmt(&self, lambda: N, f: &mut Formatter<'_>) -> Result {
        for (name, scaled) in &self.0 {
            let rate = if scaled.scale.into_inner() == 1.0 {
                format!("{}", lambda)
            } else {
                format!("{}·{}", scaled.scale, lambda)
            };
            match &scaled.constraint {
                IntervalOrEq::Eq(k) => {
                    writeln!(f, "{} ~ TruncatedPoisson({}, {{{}}})", name, rate, k)?
                }
                IntervalOrEq::Interval(interval) => match (interval.geq, interval.lt) {
                    (0, None) => writeln!(f, "{} ~ Poisson({})", name, rate)?,
                    (geq, Some(lt)) => writeln!(
                        f,
                        "{} ~ TruncatedPoisson({}, [{}, {}))",
                        name, rate, geq, lt
                    )?,
                    (geq, None) => {
                        writeln!(f, "{} ~ TruncatedPoisson({}, [{}, ∞))", name, rate, geq)?
                    }
                },
            }
        }
        Ok(())
    }
}

fn sample_truncated<R: rand::Rng>(
    constraint: &IntervalOrEq<u64>,
    lambda: NotNan<f64>,
    rng: &mut R,
) -> u64 {
    let interval = match constraint {
        IntervalOrEq::Eq(k) => return *k,
        IntervalOrEq::Interval(interval) => interval,
    };
    debug_assert!(!interval.is_empty(), "sample_truncated: empty interval");
    let lam = lambda.into_inner();
    if lam <= 0.0 {
        debug_assert!(
            interval.contains(0),
            "sample_truncated: interval has zero probability under Poisson(0)"
        );
        return 0;
    }
    let target = rng.gen::<f64>() * log_prob(constraint, &lambda).exp();
    let mut k = interval.geq;
    let mut pmf = (k as f64 * lam.ln() - lam - ln_factorial(k)).exp();
    let mut cum = pmf;
    while cum < target && interval.lt.is_none_or(|lt| k + 1 < lt) {
        if pmf <= 0.0 {
            break;
        }
        k += 1;
        pmf *= lam / k as f64;
        cum += pmf;
    }
    k
}

pub fn log_prob(constraint: &IntervalOrEq<u64>, value: &NotNan<f64>) -> f64 {
    if constraint.is_empty() {
        return f64::NEG_INFINITY;
    }
    if value.into_inner() <= 0.0 {
        return if constraint.contains(0) {
            0.0
        } else {
            f64::NEG_INFINITY
        };
    }
    let log_pmf = |k: u64| k as f64 * value.ln() - value.into_inner() - ln_factorial(k);
    match constraint {
        IntervalOrEq::Eq(k) => log_pmf(*k),
        IntervalOrEq::Interval(interval) => match interval.lt {
            Some(lt) => logsumexp_many((interval.geq..lt).map(log_pmf)),
            None if interval.geq == 0 => 0.0,
            None => {
                let log_cdf = logsumexp_many((0..interval.geq).map(log_pmf));
                let p_cdf = log_cdf.exp();
                if p_cdf >= 1.0 {
                    f64::NEG_INFINITY
                } else {
                    (-p_cdf).ln_1p()
                }
            }
        },
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct PoissonObs(pub BTreeMap<ContVarName, ScaledConstraint>);

impl PoissonObs {
    fn single(var: ContVarName, constraint: IntervalOrEq<u64>) -> Self {
        PoissonObs(BTreeMap::from([(var, Scaled::unit(constraint))]))
    }

    pub fn with_scale(mut self, scale: NotNan<f64>) -> Self {
        for v in self.0.values_mut() {
            v.scale = scale;
        }
        self
    }

    pub fn geq(var: ContVarName, b: u64) -> Self {
        Self::single(var, IntervalOrEq::geq(b))
    }

    pub fn leq(var: ContVarName, b: u64) -> Self {
        Self::single(var, IntervalOrEq::lt(b + 1))
    }

    pub fn eq(var: ContVarName, k: u64) -> Self {
        Self::single(var, IntervalOrEq::eq(k))
    }

    pub fn constraint(var: ContVarName, constraint: IntervalOrEq<u64>) -> Self {
        Self::single(var, constraint)
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn scope(&self) -> impl Iterator<Item = &ContVarName> + '_ {
        self.0.keys()
    }

    pub fn merge(&self, other: &Self) -> (Self, f64) {
        let mut map = self.0.clone();
        let mut log_factor = 0.0;
        for (name, c) in &other.0 {
            let merged = match map.get(name) {
                Some(existing) => {
                    debug_assert_eq!(
                        existing.scale, c.scale,
                        "PoissonObs::merge: conflicting scales on draw {name}"
                    );
                    let inner = existing.constraint.merge(&c.constraint).unwrap_or_else(|| {
                        log_factor = f64::NEG_INFINITY;
                        existing.constraint.clone()
                    });
                    Scaled::new(inner, existing.scale)
                }
                None => c.clone(),
            };
            map.insert(*name, merged);
        }
        (PoissonObs(map), log_factor)
    }

    pub fn marginal_log_likelihood_contribution(&self, lambda: &NotNan<f64>) -> RealEps {
        RealEps::from_log(
            self.0
                .values()
                .map(|scaled| log_prob(&scaled.constraint, &(*lambda * scaled.scale)))
                .sum(),
            0,
        )
    }

    pub fn log_likelihood(&self, values: &BTreeMap<ContVarName, NotNan<f64>>) -> f64 {
        for (var, scaled) in &self.0 {
            let v = values
                .get(var)
                .expect("PoissonObs::log_likelihood: missing draw value");
            debug_assert_eq!(
                v.fract(),
                0.0,
                "PoissonObs::log_likelihood: non-integer draw value {v}"
            );
            if !scaled.constraint.contains(v.into_inner() as u64) {
                return f64::NEG_INFINITY;
            }
        }
        0.0
    }

    pub fn mixture_representation(&self) -> GammaMixture {
        use crate::utils::math::ln_factorial;
        let pmf_term = |k: u64, sign: i8| {
            (
                GammaTerm {
                    n: NotNan::new(k as f64).unwrap(),
                    c: NotNan::new(1.).unwrap(),
                },
                SignedWeight {
                    log_w: -ln_factorial(k),
                    sign,
                },
            )
        };
        self.0.values().fold(GammaMixture::one(), |acc, scaled| {
            let new_terms = match &scaled.constraint {
                IntervalOrEq::Eq(k) => std::iter::once(pmf_term(*k, 1)).collect(),
                IntervalOrEq::Interval(interval) => match interval.lt {
                    Some(lt) => (interval.geq..lt).map(|k| pmf_term(k, 1)).collect(),
                    None => {
                        std::iter::once((
                            GammaTerm {
                                n: NotNan::new(0.0).unwrap(),
                                c: NotNan::new(0.0).unwrap(),
                            },
                            SignedWeight {
                                log_w: 0.0,
                                sign: 1,
                            },
                        ))
                        .chain((0..interval.geq).map(|k| pmf_term(k, -1)))
                        .collect()
                    }
                },
            };
            acc * GammaMixture(new_terms).scale_rate(scaled.scale.into_inner())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::intervals::Interval;

    #[test]
    fn merge_same_draw_intersects() {
        let a = PoissonObs::leq(1, 2);
        let b = PoissonObs::geq(1, 2);
        let (m, f) = a.merge(&b);
        assert_eq!(f, 0.0);
        assert_eq!(
            m.0.get(&1),
            Some(&Scaled::unit(IntervalOrEq::Interval(Interval {
                geq: 2,
                lt: Some(3)
            })))
        );
    }

    #[test]
    fn merge_distinct_draws_unions() {
        let a = PoissonObs::leq(1, 2);
        let b = PoissonObs::geq(2, 2);
        let (m, f) = a.merge(&b);
        assert_eq!(f, 0.0);
        assert_eq!(m.0.len(), 2);
    }

    #[test]
    fn merge_contradiction_is_neg_infinity() {
        let a = PoissonObs::leq(1, 1);
        let b = PoissonObs::geq(1, 3);
        let (_, f) = a.merge(&b);
        assert_eq!(f, f64::NEG_INFINITY);
    }

    #[test]
    fn merge_eq_with_excluding_interval_contradicts() {
        let a = PoissonObs::eq(1, 3);
        let b = PoissonObs::leq(1, 2);
        let (_, f) = a.merge(&b);
        assert_eq!(f, f64::NEG_INFINITY);
    }

    #[test]
    fn signed_terms_of_empty_is_identity() {
        let st = PoissonObs::default().mixture_representation();
        assert_eq!(st.0.len(), 1);
        let (term, w) = st.0.iter().next().unwrap();
        assert_eq!(
            (term.n.into_inner(), term.c.into_inner(), w.log_w, w.sign),
            (0.0, 0.0, 0.0, 1)
        );
    }

    #[test]
    fn log_prob_point_matches_pmf() {
        let lambda: f64 = 1.7;
        let lp = log_prob(&IntervalOrEq::eq(2), &NotNan::new(lambda).unwrap());
        let expected = 2.0 * lambda.ln() - lambda - 2.0_f64.ln();
        assert!((lp - expected).abs() < 1e-12);
    }

    #[test]
    fn log_prob_complement_pair_sums_to_one() {
        let lambda = 2.3;
        let p_lt = log_prob(&IntervalOrEq::lt(4), &NotNan::new(lambda).unwrap()).exp();
        let p_geq = log_prob(&IntervalOrEq::geq(4), &NotNan::new(lambda).unwrap()).exp();
        assert!((p_lt + p_geq - 1.0).abs() < 1e-12);
    }

    #[test]
    fn log_prob_full_interval_is_zero() {
        assert_eq!(
            log_prob(&IntervalOrEq::geq(0), &NotNan::new(3.0).unwrap()),
            0.0
        );
    }

    #[test]
    fn log_prob_empty_interval_is_neg_infinity() {
        let empty = IntervalOrEq::Interval(Interval {
            geq: 3,
            lt: Some(1),
        });
        assert_eq!(
            log_prob(&empty, &NotNan::new(1.0).unwrap()),
            f64::NEG_INFINITY
        );
    }


    #[test]
    fn log_likelihood_is_containment_indicator() {
        let obs = PoissonObs::geq(1, 2);
        let inside = BTreeMap::from([(1u64, NotNan::new(3.0).unwrap())]);
        assert_eq!(obs.log_likelihood(&inside), 0.0);
        let outside = BTreeMap::from([(1u64, NotNan::new(1.0).unwrap())]);
        assert_eq!(obs.log_likelihood(&outside), f64::NEG_INFINITY);
    }

    #[test]
    fn log_likelihood_point_constraint_is_equality_indicator() {
        let obs = PoissonObs::eq(1, 5);
        let hit = BTreeMap::from([(1u64, NotNan::new(5.0).unwrap())]);
        assert_eq!(obs.log_likelihood(&hit), 0.0);
        let miss = BTreeMap::from([(1u64, NotNan::new(4.0).unwrap())]);
        assert_eq!(obs.log_likelihood(&miss), f64::NEG_INFINITY);
    }


    #[test]
    fn constructors_preserve_keys_and_constraints() {
        let (obs, _) = PoissonObs::geq(1, 2).merge(&PoissonObs::eq(2, 5));
        let prior = PoissonPrior::unconstrained(obs.scope().copied());
        assert_eq!(prior.scope().copied().collect::<Vec<_>>(), vec![1, 2]);
        assert_eq!(prior.0.get(&1), Some(&Scaled::unit(IntervalOrEq::geq(0))));
        assert_eq!(prior.0.get(&2), Some(&Scaled::unit(IntervalOrEq::geq(0))));
        let cond = PoissonPrior::conditioned_on(&obs);
        assert_eq!(cond.0.get(&1), Some(&Scaled::unit(IntervalOrEq::geq(2))));
        assert_eq!(cond.0.get(&2), Some(&Scaled::unit(IntervalOrEq::eq(5))));
    }


    use rand::SeedableRng;

    fn rng() -> rand::rngs::StdRng {
        rand::rngs::StdRng::seed_from_u64(42)
    }

    #[test]
    fn sample_truncated_respects_constraints() {
        let mut r = rng();
        let lam = NotNan::new(2.3).unwrap();
        for _ in 0..200 {
            assert!(sample_truncated(&IntervalOrEq::geq(2), lam, &mut r) >= 2);
            assert!(sample_truncated(&IntervalOrEq::lt(2), lam, &mut r) <= 1);
            assert_eq!(sample_truncated(&IntervalOrEq::eq(5), lam, &mut r), 5);
        }
    }

    #[test]
    fn sample_truncated_unconstrained_empirical_mean() {
        let lam = 2.3;
        let mut r = rng();
        let n = 20_000;
        let mean = (0..n)
            .map(|_| {
                sample_truncated(&IntervalOrEq::geq(0), NotNan::new(lam).unwrap(), &mut r) as f64
            })
            .sum::<f64>()
            / n as f64;
        assert!((mean - lam).abs() < 0.05, "empirical mean {:.4}", mean);
    }

    #[test]
    fn sample_truncated_zero_rate_is_point_mass_at_zero() {
        let mut r = rng();
        assert_eq!(
            sample_truncated(&IntervalOrEq::lt(4), NotNan::new(0.0).unwrap(), &mut r),
            0
        );
    }
}
