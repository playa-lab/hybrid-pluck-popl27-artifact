use std::{
    collections::BTreeMap,
    fmt::{self, Display, Formatter},
    hash::{Hash, Hasher},
    ops::{Add, Mul},
};

use itertools::Itertools;
use ordered_float::NotNan;

use super::suffstat::PRIOR_QUANTIZATION_LEVEL;
use crate::utils::math::{ln_gamma, log_gamma_pdf, quant, signed_logsumexp};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(
    into = "Vec<(GammaTerm, SignedWeight)>",
    from = "Vec<(GammaTerm, SignedWeight)>"
)]
pub struct GammaMixture(pub BTreeMap<GammaTerm, SignedWeight>);

impl GammaMixture {
    pub fn one() -> Self {
        GammaMixture(BTreeMap::from([(
            GammaTerm {
                n: NotNan::new(0.0).unwrap(),
                c: NotNan::new(0.0).unwrap(),
            },
            SignedWeight {
                log_w: 0.0,
                sign: 1,
            },
        )]))
    }

    pub fn gamma(shape: f64, rate: f64) -> Self {
        GammaMixture(BTreeMap::from([(
            GammaTerm {
                n: NotNan::new(shape - 1.0).unwrap(),
                c: NotNan::new(rate).unwrap(),
            },
            SignedWeight {
                log_w: 0.0,
                sign: 1,
            },
        )]))
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub(super) fn components(&self) -> impl Iterator<Item = (f64, f64, f64, i8)> + '_ {
        self.0.iter().map(|(t, w)| {
            let shape = t.n + 1.0;
            let rate = t.c.into_inner();
            let log_weight = w.log_w + ln_gamma(shape) - (shape) * rate.ln();
            (shape, rate, log_weight, w.sign)
        })
    }

    pub fn normalize(&mut self) {
        let (log_norm, _) =
            signed_logsumexp(self.components().map(|(_, _, log_w, sign)| (log_w, sign)));
        for w in self.0.values_mut() {
            w.log_w -= log_norm;
        }
    }

    pub fn scale_rate(self, s: f64) -> GammaMixture {
        debug_assert!(s > 0.0, "scale_rate: scale must be positive, got {s}");
        let ln_s = s.ln();
        let scaled = self.0.into_iter().map(|(t, w)| {
            let n = t.n.into_inner();
            (
                GammaTerm {
                    n: t.n,
                    c: NotNan::new(t.c.into_inner() * s).unwrap(),
                },
                SignedWeight {
                    log_w: w.log_w + n * ln_s,
                    sign: w.sign,
                },
            )
        });
        GammaMixture(scaled.collect())
    }

    pub fn as_single_gamma(&self) -> Option<(f64, f64)> {
        let mut it = self.0.iter();
        let (t, w) = it.next()?;
        if it.next().is_none() && w.sign > 0 {
            Some((t.n + 1.0, t.c.into_inner()))
        } else {
            None
        }
    }

    pub fn sample<R: rand::Rng>(&self, rng: &mut R) -> f64 {
        use crate::utils::sampling::{pick_categorical_log, sample_gamma};
        let comps: Vec<(f64, f64, f64, i8)> = self.components().collect();
        let positive: Vec<(f64, f64, f64, i8)> =
            comps.iter().copied().filter(|c| c.3 > 0).collect();
        assert!(
            !positive.is_empty(),
            "GammaMixture::sample: mixture has no positive components"
        );
        let log_weights: Vec<f64> = positive.iter().map(|c| c.2).collect();
        let all_positive = positive.len() == comps.len();
        const MAX_REJECTS: usize = 100_000;
        for _ in 0..MAX_REJECTS {
            let (shape, rate, _, _) = positive[pick_categorical_log(rng, &log_weights)];
            let x = sample_gamma(rng, shape) / rate;
            if all_positive {
                return x;
            }
            let log_at = |c: &(f64, f64, f64, i8)| c.2 + log_gamma_pdf(x, c.0, c.1);
            let (log_f, sign) = signed_logsumexp(comps.iter().map(|c| (log_at(c), c.3)));
            if sign <= 0 {
                continue;
            }
            let (log_envelope, _) = signed_logsumexp(positive.iter().map(|c| (log_at(c), 1)));
            if rng.gen::<f64>() < (log_f - log_envelope).exp() {
                return x;
            }
        }
        panic!("GammaMixture::sample: no acceptance after {MAX_REJECTS} proposals");
    }
}

impl Display for GammaMixture {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let body = self
            .components()
            .map(|(shape, rate, log_w, sign)| {
                format!(
                    "{:+.4}·Gamma({:.4}, {:.4})",
                    sign as f64 * log_w.exp(),
                    shape,
                    rate
                )
            })
            .join(" ");
        write!(f, "{}", body)
    }
}

impl Hash for GammaMixture {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for (t, w) in &self.0 {
            t.n.hash(state);
            quant(t.c.into_inner(), PRIOR_QUANTIZATION_LEVEL).hash(state);
            quant(w.log_w, PRIOR_QUANTIZATION_LEVEL).hash(state);
            w.sign.hash(state);
        }
    }
}

impl PartialEq for GammaMixture {
    fn eq(&self, other: &Self) -> bool {
        self.0.len() == other.0.len()
            && self
                .0
                .iter()
                .zip(other.0.iter())
                .all(|((t1, w1), (t2, w2))| {
                    t1.n == t2.n
                        && quant(t1.c.into_inner(), PRIOR_QUANTIZATION_LEVEL)
                            == quant(t2.c.into_inner(), PRIOR_QUANTIZATION_LEVEL)
                        && quant(w1.log_w, PRIOR_QUANTIZATION_LEVEL)
                            == quant(w2.log_w, PRIOR_QUANTIZATION_LEVEL)
                        && w1.sign == w2.sign
                })
    }
}

impl Eq for GammaMixture {}

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
pub struct GammaTerm {
    pub(super) n: NotNan<f64>,
    pub(super) c: NotNan<f64>,
}

impl GammaTerm {
    pub fn n(&self) -> f64 {
        self.n.into_inner()
    }
    pub fn c(&self) -> f64 {
        self.c.into_inner()
    }
}

impl Add for GammaTerm {
    type Output = GammaTerm;
    fn add(self, rhs: Self) -> Self::Output {
        GammaTerm {
            n: self.n + rhs.n,
            c: self.c + rhs.c,
        }
    }
}

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct SignedWeight {
    pub(super) log_w: f64,
    pub(super) sign: i8,
}

impl SignedWeight {
    pub fn log_w(&self) -> f64 {
        self.log_w
    }
    pub fn sign(&self) -> i8 {
        self.sign
    }
}

impl Mul for SignedWeight {
    type Output = SignedWeight;
    fn mul(self, rhs: Self) -> Self::Output {
        SignedWeight {
            log_w: self.log_w + rhs.log_w,
            sign: self.sign * rhs.sign,
        }
    }
}

impl Add for SignedWeight {
    type Output = SignedWeight;
    fn add(self, rhs: Self) -> Self::Output {
        let (log_w, sign) = signed_logsumexp([(self.log_w, self.sign), (rhs.log_w, rhs.sign)]);
        SignedWeight { log_w, sign }
    }
}

impl Mul for GammaMixture {
    type Output = GammaMixture;
    fn mul(self, rhs: Self) -> Self::Output {
        let mut new: GammaMixture = GammaMixture(BTreeMap::new());
        for (t_term, t_weight) in self.0 {
            for (s_term, s_weight) in &rhs.0 {
                let term = t_term + *s_term;
                let weight = match new.0.get(&term) {
                    Some(&existing) => t_weight * *s_weight + existing,
                    None => t_weight * *s_weight,
                };
                new.0.insert(term, weight);
            }
        }
        new
    }
}

impl From<GammaMixture> for Vec<(GammaTerm, SignedWeight)> {
    fn from(m: GammaMixture) -> Self {
        m.0.into_iter().collect()
    }
}

impl From<Vec<(GammaTerm, SignedWeight)>> for GammaMixture {
    fn from(terms: Vec<(GammaTerm, SignedWeight)>) -> Self {
        GammaMixture(terms.into_iter().collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn term(n: f64, c: f64, log_w: f64, sign: i8) -> (GammaTerm, SignedWeight) {
        (
            GammaTerm {
                n: NotNan::new(n).unwrap(),
                c: NotNan::new(c).unwrap(),
            },
            SignedWeight { log_w, sign },
        )
    }

    #[test]
    fn scale_rate_maps_term_correctly() {
        let s = 5.0;
        let m = GammaMixture(BTreeMap::from([term(3.0, 2.0, 0.5, 1)]));
        let scaled = m.scale_rate(s);
        let (t, w) = scaled.0.iter().next().unwrap();
        assert_eq!(t.n.into_inner(), 3.0);
        assert!((t.c.into_inner() - 2.0 * s).abs() < 1e-12);
        assert!((w.log_w - (0.5 + 3.0 * s.ln())).abs() < 1e-12);
        assert_eq!(w.sign, 1);
    }

    #[test]
    fn scale_rate_leaves_constant_term_invariant() {
        let scaled = GammaMixture::one().scale_rate(7.0);
        let (t, w) = scaled.0.iter().next().unwrap();
        assert_eq!(t.n.into_inner(), 0.0);
        assert_eq!(t.c.into_inner(), 0.0);
        assert_eq!(w.log_w, 0.0);
    }

    #[test]
    fn scale_rate_by_one_is_identity() {
        let m = GammaMixture(BTreeMap::from([term(2.0, 1.5, 0.3, -1)]));
        assert_eq!(m.clone().scale_rate(1.0), m);
    }
}
