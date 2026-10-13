use std::collections::{BTreeMap, BTreeSet};
use std::fmt::{Display, Formatter, Result};

use ordered_float::NotNan;

use crate::inference::conjugate_pairs::gamma_mixture::{GammaMixture, GammaTerm, SignedWeight};
use crate::inference::conjugate_pairs::{ContVarName, Scaled};
use crate::utils::epsilon::RealEps;
use crate::utils::intervals::{Interval, IntervalOrEq};

type ScaledConstraint = Scaled<IntervalOrEq<NotNan<f64>>>;

#[derive(Debug, Clone, Hash, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(bound(
    serialize = "N: serde::Serialize + Ord",
    deserialize = "N: serde::Deserialize<'de> + Ord"
))]
pub struct ExponentialPrior<N>(BTreeMap<N, ScaledConstraint>);

impl ExponentialPrior<ContVarName> {
    pub fn unconstrained(vars: impl Iterator<Item = ContVarName>) -> Self {
        ExponentialPrior(
            vars.map(|v| {
                (
                    v,
                    Scaled::unit(IntervalOrEq::geq(NotNan::new(0.0).unwrap())),
                )
            })
            .collect(),
        )
    }

    pub fn conditioned_on(obs: &ExponentialObs) -> Self {
        ExponentialPrior(obs.ranges().clone())
    }

    pub fn restricted_to(&self, vars: &std::collections::BTreeSet<ContVarName>) -> Self {
        ExponentialPrior(
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
                let x = sample_truncated(&scaled.constraint, rate, rng);
                (*var, NotNan::new(x).expect("truncated Exp sample is NaN"))
            })
            .collect()
    }

    pub fn map_names<M: Ord>(&self, f: impl Fn(ContVarName) -> M) -> ExponentialPrior<M> {
        ExponentialPrior(self.0.iter().map(|(n, c)| (f(*n), c.clone())).collect())
    }

    pub fn scope(&self) -> impl Iterator<Item = &ContVarName> + '_ {
        self.0.keys()
    }
}

fn sample_truncated<R: rand::Rng>(
    constraint: &IntervalOrEq<NotNan<f64>>,
    lambda: NotNan<f64>,
    rng: &mut R,
) -> f64 {
    match constraint {
        IntervalOrEq::Eq(v) => v.into_inner(),
        IntervalOrEq::Interval(interval) => {
            debug_assert!(!interval.is_empty(), "sample_truncated: empty interval");
            let lam = lambda.into_inner();
            debug_assert!(lam > 0.0, "sample_truncated: Exp(λ) requires λ > 0");
            let s = interval.geq.into_inner();
            let u: f64 = rng.gen();
            match interval.lt {
                None => s - (1.0 - u).ln() / lam,
                Some(t) => {
                    let span_mass = 1.0 - (-lam * (t.into_inner() - s)).exp();
                    s - (1.0 - u * span_mass).ln() / lam
                }
            }
        }
    }
}

impl<N> ExponentialPrior<N> {
    pub fn constraints(&self) -> &BTreeMap<N, ScaledConstraint> {
        &self.0
    }
}

impl<N: Display> ExponentialPrior<N> {
    pub fn fmt(&self, lambda: N, f: &mut Formatter<'_>) -> Result {
        for (name, scaled) in &self.0 {
            let rate = if scaled.scale.into_inner() == 1.0 {
                format!("{}", lambda)
            } else {
                format!("{}·{}", scaled.scale, lambda)
            };
            match &scaled.constraint {
                IntervalOrEq::Eq(v) => writeln!(f, "{} = {:.4}", name, v)?,
                IntervalOrEq::Interval(interval) => {
                    let s = interval.geq.into_inner();
                    match interval.lt {
                        None if s == 0.0 => writeln!(f, "{} ~ Exponential({})", name, rate)?,
                        Some(t) => writeln!(
                            f,
                            "{} ~ TruncatedExponential({}, [{}, {}))",
                            name, rate, s, t
                        )?,
                        None => {
                            writeln!(f, "{} ~ TruncatedExponential({}, [{}, ∞))", name, rate, s)?
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

impl IntervalOrEq<NotNan<f64>> {
    pub fn marginal_log_likelihood_contribution(&self, lambda: &NotNan<f64>) -> RealEps {
        if lambda.into_inner() <= 0.0 {
            return RealEps::zero();
        }
        match self {
            IntervalOrEq::Interval(interval) => {
                if interval.is_empty() {
                    return RealEps::zero();
                }
                let s = interval.geq.into_inner();
                match interval.lt {
                    Some(t) => RealEps::from_log(
                        -lambda * s + (-(-lambda * (t.into_inner() - s)).exp()).ln_1p(),
                        0,
                    ),
                    None => RealEps::from_log(-lambda * s, 0),
                }
            }
            IntervalOrEq::Eq(v) => {
                RealEps::from_log(lambda.ln() - lambda.into_inner() * v.into_inner(), 1)
            }
        }
    }

    fn mixture_representation(&self) -> GammaMixture {
        let survival = |c: NotNan<f64>, sign: i8| {
            (
                GammaTerm {
                    n: NotNan::new(0.).unwrap(),
                    c,
                },
                SignedWeight { log_w: 0.0, sign },
            )
        };
        match self {
            Self::Interval(interval) => {
                let terms = std::iter::once(survival(interval.geq, 1))
                    .chain(interval.lt.iter().map(|t| survival(*t, -1)))
                    .collect();
                GammaMixture(terms)
            }
            Self::Eq(v) => {
                let terms = [(
                    GammaTerm {
                        n: NotNan::new(1.).unwrap(),
                        c: *v,
                    },
                    SignedWeight {
                        log_w: 0.0,
                        sign: 1,
                    },
                )]
                .into_iter()
                .collect();
                GammaMixture(terms)
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct ExponentialObs {
    ranges: BTreeMap<ContVarName, ScaledConstraint>,
    excluded: BTreeMap<ContVarName, BTreeSet<NotNan<f64>>>,
}

impl ExponentialObs {
    fn single(var: ContVarName, range: IntervalOrEq<NotNan<f64>>) -> Self {
        ExponentialObs {
            ranges: BTreeMap::from([(var, Scaled::unit(range))]),
            excluded: BTreeMap::new(),
        }
    }

    pub fn with_scale(mut self, scale: NotNan<f64>) -> Self {
        for v in self.ranges.values_mut() {
            v.scale = scale;
        }
        self
    }

    pub fn geq(var: ContVarName, b: f64) -> Self {
        debug_assert!(b >= 0.0, "ExponentialObs::geq: negative bound {b}");
        Self::single(var, IntervalOrEq::geq(NotNan::new(b).unwrap()))
    }

    pub fn lt(var: ContVarName, b: f64) -> Self {
        debug_assert!(b >= 0.0, "ExponentialObs::lt: negative bound {b}");
        Self::single(var, IntervalOrEq::lt(NotNan::new(b).unwrap()))
    }

    pub fn close(var: ContVarName, v: f64) -> Self {
        debug_assert!(v >= 0.0, "ExponentialObs::close: negative value {v}");
        Self::single(var, IntervalOrEq::eq(NotNan::new(v).unwrap()))
    }

    pub fn interval(var: ContVarName, interval: Interval<NotNan<f64>>) -> Self {
        Self::single(var, IntervalOrEq::Interval(interval))
    }

    pub fn not_eq(var: ContVarName, v: f64) -> Self {
        ExponentialObs {
            ranges: BTreeMap::from([(
                var,
                Scaled::unit(IntervalOrEq::geq(NotNan::new(0.0).unwrap())),
            )]),
            excluded: BTreeMap::from([(var, BTreeSet::from([NotNan::new(v).unwrap()]))]),
        }
    }

    pub fn ranges(&self) -> &BTreeMap<ContVarName, ScaledConstraint> {
        &self.ranges
    }

    pub fn scope(&self) -> impl Iterator<Item = &ContVarName> + '_ {
        self.ranges.keys()
    }

    pub fn is_empty(&self) -> bool {
        self.ranges.is_empty()
    }

    pub fn n_exact(&self) -> usize {
        self.ranges
            .values()
            .filter(|scaled| matches!(scaled.constraint, IntervalOrEq::Eq(_)))
            .count()
    }

    pub fn marginal_log_likelihood_contribution(&self, value: &NotNan<f64>) -> RealEps {
        self.ranges
            .values()
            .map(|scaled| {
                let rate = *value * scaled.scale;
                scaled
                    .constraint
                    .marginal_log_likelihood_contribution(&rate)
            })
            .fold(RealEps::from_log(0.0, 0), |acc, logp| acc * logp)
    }

    pub fn log_likelihood(&self, values: &BTreeMap<ContVarName, NotNan<f64>>) -> f64 {
        for (var, scaled) in &self.ranges {
            let v = values
                .get(var)
                .expect("ExponentialObs::log_likelihood: missing draw value");
            if !scaled.constraint.contains(*v)
                || self.excluded.get(var).is_some_and(|set| set.contains(v))
            {
                return f64::NEG_INFINITY;
            }
        }
        0.0
    }

    pub fn merge(&self, other: &Self) -> (Self, f64) {
        let mut ranges = self.ranges.clone();
        let mut log_factor = 0.0;
        for (name, r) in &other.ranges {
            let merged = match ranges.get(name) {
                Some(existing) => {
                    debug_assert_eq!(
                        existing.scale, r.scale,
                        "ExponentialObs::merge: conflicting scales on draw {name}"
                    );
                    let inner = existing.constraint.merge(&r.constraint).unwrap_or_else(|| {
                        log_factor = f64::NEG_INFINITY;
                        existing.constraint.clone()
                    });
                    Scaled::new(inner, existing.scale)
                }
                None => r.clone(),
            };
            ranges.insert(*name, merged);
        }
        let mut excluded = self.excluded.clone();
        for (name, set) in &other.excluded {
            excluded.entry(*name).or_default().extend(set.iter());
        }
        for (name, set) in &excluded {
            if let Some(scaled) = ranges.get(name) {
                if let IntervalOrEq::Eq(v) = &scaled.constraint {
                    if set.contains(v) {
                        log_factor = f64::NEG_INFINITY;
                    }
                }
            }
        }
        (ExponentialObs { ranges, excluded }, log_factor)
    }

    pub fn mixture_representation(&self) -> GammaMixture {
        self.ranges
            .values()
            .fold(GammaMixture::one(), |acc, scaled| {
                acc * scaled
                    .constraint
                    .mixture_representation()
                    .scale_rate(scaled.scale.into_inner())
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nn(x: f64) -> NotNan<f64> {
        NotNan::new(x).unwrap()
    }

    #[test]
    fn marginal_interval_matches_closed_form() {
        let lambda = 0.8;
        let m = IntervalOrEq::lt(nn(3.0))
            .merge(&IntervalOrEq::geq(nn(1.0)))
            .unwrap();
        let p = m.marginal_log_likelihood_contribution(&nn(lambda)).to_f64();
        let expected = (-lambda).exp() - (-3.0 * lambda).exp();
        assert!((p - expected).abs() < 1e-12);
    }

    #[test]
    fn marginal_complement_pair_sums_to_one() {
        let lambda = 1.3;
        let p_lt = IntervalOrEq::lt(nn(2.0))
            .marginal_log_likelihood_contribution(&nn(lambda))
            .to_f64();
        let p_geq = IntervalOrEq::geq(nn(2.0))
            .marginal_log_likelihood_contribution(&nn(lambda))
            .to_f64();
        assert!((p_lt + p_geq - 1.0).abs() < 1e-12);
    }

    #[test]
    fn density_matches_pdf() {
        let lambda = 1.3;
        let d = IntervalOrEq::eq(nn(0.7)).marginal_log_likelihood_contribution(&nn(lambda));
        let expected = lambda.ln() - lambda * 0.7;
        assert_eq!(d.power, 1);
        assert!((d.log_coeff - expected).abs() < 1e-12);
    }


    #[test]
    fn log_likelihood_is_constraint_indicator() {
        let (obs, _) = ExponentialObs::geq(1, 0.5).merge(&ExponentialObs::close(2, 0.7));
        let ok = BTreeMap::from([(1u64, nn(0.9)), (2u64, nn(0.7))]);
        assert_eq!(obs.log_likelihood(&ok), 0.0);
        let bad_interval = BTreeMap::from([(1u64, nn(0.2)), (2u64, nn(0.7))]);
        assert_eq!(obs.log_likelihood(&bad_interval), f64::NEG_INFINITY);
        let bad_eq = BTreeMap::from([(1u64, nn(0.9)), (2u64, nn(0.71))]);
        assert_eq!(obs.log_likelihood(&bad_eq), f64::NEG_INFINITY);
    }


    #[test]
    fn not_eq_then_eq_same_value_contradicts() {
        let (_, f) = ExponentialObs::not_eq(1, 0.5).merge(&ExponentialObs::close(1, 0.5));
        assert_eq!(f, f64::NEG_INFINITY);
        let (_, f) = ExponentialObs::close(1, 0.5).merge(&ExponentialObs::not_eq(1, 0.5));
        assert_eq!(f, f64::NEG_INFINITY);
    }

    #[test]
    fn not_eq_then_eq_different_value_keeps_eq() {
        let (m, f) = ExponentialObs::not_eq(1, 0.5).merge(&ExponentialObs::close(1, 0.7));
        assert_eq!(f, 0.0);
        assert_eq!(
            m.ranges().get(&1),
            Some(&Scaled::unit(IntervalOrEq::eq(nn(0.7))))
        );
    }

    #[test]
    fn not_eq_has_probability_one_and_keeps_draw_in_scope() {
        let obs = ExponentialObs::not_eq(1, 0.5);
        assert_eq!(obs.scope().copied().collect::<Vec<_>>(), vec![1]);
        let p = obs.marginal_log_likelihood_contribution(&nn(1.3)).to_f64();
        assert!((p - 1.0).abs() < 1e-12);
    }

    #[test]
    fn log_likelihood_rejects_excluded_value() {
        let obs = ExponentialObs::not_eq(1, 0.5);
        let at_excluded = BTreeMap::from([(1u64, nn(0.5))]);
        assert_eq!(obs.log_likelihood(&at_excluded), f64::NEG_INFINITY);
        let elsewhere = BTreeMap::from([(1u64, nn(0.9))]);
        assert_eq!(obs.log_likelihood(&elsewhere), 0.0);
    }


    #[test]
    fn constructors_preserve_keys_and_constraints() {
        let (obs, _) = ExponentialObs::geq(1, 0.5).merge(&ExponentialObs::close(2, 0.7));
        let prior = ExponentialPrior::unconstrained(obs.scope().copied());
        assert_eq!(
            prior.0.get(&1),
            Some(&Scaled::unit(IntervalOrEq::geq(nn(0.0))))
        );
        assert_eq!(
            prior.0.get(&2),
            Some(&Scaled::unit(IntervalOrEq::geq(nn(0.0))))
        );
        let cond = ExponentialPrior::conditioned_on(&obs);
        assert_eq!(
            cond.0.get(&1),
            Some(&Scaled::unit(IntervalOrEq::geq(nn(0.5))))
        );
        assert_eq!(
            cond.0.get(&2),
            Some(&Scaled::unit(IntervalOrEq::eq(nn(0.7))))
        );
    }


    use rand::SeedableRng;

    fn rng() -> rand::rngs::StdRng {
        rand::rngs::StdRng::seed_from_u64(42)
    }

    #[test]
    fn sample_truncated_respects_constraints() {
        let mut r = rng();
        let lam = nn(1.3);
        let bounded = IntervalOrEq::geq(nn(0.5))
            .merge(&IntervalOrEq::lt(nn(2.0)))
            .unwrap();
        for _ in 0..200 {
            assert!(sample_truncated(&IntervalOrEq::geq(nn(0.5)), lam, &mut r) >= 0.5);
            let x = sample_truncated(&bounded, lam, &mut r);
            assert!((0.5..2.0).contains(&x), "sample {x} outside [0.5, 2)");
            assert_eq!(
                sample_truncated(&IntervalOrEq::eq(nn(0.7)), lam, &mut r),
                0.7
            );
        }
    }

    #[test]
    fn sample_truncated_unconstrained_empirical_mean() {
        let lam = 1.3;
        let mut r = rng();
        let n = 20_000;
        let mean = (0..n)
            .map(|_| sample_truncated(&IntervalOrEq::geq(nn(0.0)), nn(lam), &mut r))
            .sum::<f64>()
            / n as f64;
        assert!(
            (mean - 1.0 / lam).abs() < 0.03,
            "empirical mean {:.4}",
            mean
        );
    }
}
