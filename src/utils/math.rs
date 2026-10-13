use std::collections::BTreeMap;

use ordered_float::NotNan;

pub fn logsumexp(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        return b;
    }
    if b == f64::NEG_INFINITY {
        return a;
    }
    let max = a.max(b);
    max + ((a - max).exp() + (b - max).exp()).ln()
}

pub fn logsumexp_many(it: impl IntoIterator<Item = f64>) -> f64 {
    let v: Vec<f64> = it.into_iter().filter(|x| *x != f64::NEG_INFINITY).collect();
    if v.is_empty() {
        return f64::NEG_INFINITY;
    }
    let max = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    max + v.iter().map(|x| (x - max).exp()).sum::<f64>().ln()
}

pub fn log_dirichlet_ratio(alphas: &[f64], counts: &BTreeMap<usize, (u64, u64)>) -> f64 {
    let n = alphas.len();
    let mut suffix_sum = vec![0.0; n + 1];
    for i in (0..n).rev() {
        suffix_sum[i] = suffix_sum[i + 1] + alphas[i];
    }

    let mut total = 0.0;
    for (&i, &(h, t)) in counts {
        let alpha_i = alphas[i];
        let rest = suffix_sum[i + 1];
        total += log_beta_ratio(alpha_i + h as f64, rest + t as f64, alpha_i, rest);
    }
    total
}

pub fn signed_logsumexp(it: impl IntoIterator<Item = (f64, i8)>) -> (f64, i8) {
    let v: Vec<(f64, i8)> = it
        .into_iter()
        .filter(|(l, s)| *l != f64::NEG_INFINITY && *s != 0)
        .collect();
    if v.is_empty() {
        return (f64::NEG_INFINITY, 0);
    }
    let max = v.iter().map(|(l, _)| *l).fold(f64::NEG_INFINITY, f64::max);
    let sum: f64 = v.iter().map(|(l, s)| (*s as f64) * (l - max).exp()).sum();
    if sum == 0.0 {
        (f64::NEG_INFINITY, 0)
    } else {
        (max + sum.abs().ln(), if sum > 0.0 { 1 } else { -1 })
    }
}

pub fn ln_factorial(k: u64) -> f64 {
    ln_gamma(k as f64 + 1.0)
}

pub fn log_gamma_pdf(x: f64, shape: f64, rate: f64) -> f64 {
    if x <= 0.0 {
        return f64::NEG_INFINITY;
    }
    shape * rate.ln() + (shape - 1.0) * x.ln() - rate * x - ln_gamma(shape)
}

pub fn log_gamma_marginal(shape: f64, rate: f64, n: f64, c: f64) -> f64 {
    let a_post = shape + n;
    let rate_post = rate + c;
    ln_gamma(a_post) - a_post * rate_post.ln() - (ln_gamma(shape) - shape * rate.ln())
}

pub fn log_beta_ratio(a1: f64, b1: f64, a2: f64, b2: f64) -> f64 {
    ln_gamma(a1) + ln_gamma(b1)
        - ln_gamma(a1 + b1)
        - (ln_gamma(a2) + ln_gamma(b2) - ln_gamma(a2 + b2))
}

pub fn log_beta_pdf(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 || x >= 1.0 {
        return f64::NEG_INFINITY;
    }
    (a - 1.0) * x.ln() + (b - 1.0) * (1.0 - x).ln() - (ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b))
}

pub fn log_dirichlet_pdf(value: &[NotNan<f64>], alphas: &[f64]) -> f64 {
    debug_assert_eq!(
        value.len(),
        alphas.len(),
        "log_dirichlet_pdf: shape mismatch ({} vs {})",
        value.len(),
        alphas.len()
    );
    if value.len() != alphas.len() {
        return f64::NEG_INFINITY;
    }
    let alpha_sum: f64 = alphas.iter().sum();
    let mut log_pdf = ln_gamma(alpha_sum);
    for &a in alphas {
        log_pdf -= ln_gamma(a);
    }
    for (theta, &a) in value.iter().zip(alphas.iter()) {
        let exp = a - 1.0;
        if exp == 0.0 {
            continue;
        }
        let t = theta.into_inner();
        if t <= 0.0 {
            return if exp > 0.0 {
                f64::NEG_INFINITY
            } else {
                f64::INFINITY
            };
        }
        log_pdf += exp * t.ln();
    }
    log_pdf
}

pub fn ln_gamma(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::INFINITY;
    }
    lanczos_ln_gamma(x)
}

pub fn lanczos_ln_gamma(x: f64) -> f64 {
    const G: f64 = 7.0;
    const COEFFS: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.5203681218851,
        -1259.1392167224028,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507343278686905,
        -0.13857109526572012,
        9.984_369_578_019_572e-6,
        1.5056327351493116e-7,
    ];

    if x < 0.5 {
        let pi = std::f64::consts::PI;
        return (pi / (pi * x).sin()).ln() - lanczos_ln_gamma(1.0 - x);
    }

    let x = x - 1.0;
    let mut ag = COEFFS[0];
    for (i, &c) in COEFFS.iter().enumerate().skip(1) {
        ag += c / (x + i as f64);
    }

    let tmp = x + G + 0.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * tmp.ln() - tmp + ag.ln()
}

pub fn quant(x: f64, level: f64) -> i64 {
    (x * level).round() as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn logsumexp_many_empty_is_neg_infinity() {
        assert_eq!(logsumexp_many(std::iter::empty::<f64>()), f64::NEG_INFINITY);
    }

    #[test]
    fn logsumexp_many_only_neg_infinity_is_neg_infinity() {
        assert_eq!(
            logsumexp_many(vec![f64::NEG_INFINITY, f64::NEG_INFINITY]),
            f64::NEG_INFINITY
        );
    }

    #[test]
    fn logsumexp_many_single_element_passes_through() {
        let v = logsumexp_many(vec![1.5]);
        assert!((v - 1.5).abs() < 1e-12);
    }

    #[test]
    fn logsumexp_many_matches_pairwise_binary() {
        let a = 0.5_f64;
        let b = 1.2_f64;
        let lhs = logsumexp_many(vec![a, b]);
        let rhs = logsumexp(a, b);
        assert!((lhs - rhs).abs() < 1e-12);
    }

    #[test]
    fn logsumexp_many_skips_neg_infinity_entries() {
        let a = 0.5_f64;
        let b = 1.2_f64;
        let lhs = logsumexp_many(vec![a, f64::NEG_INFINITY, b]);
        let rhs = logsumexp(a, b);
        assert!((lhs - rhs).abs() < 1e-12);
    }

    fn nn(xs: &[f64]) -> Vec<NotNan<f64>> {
        xs.iter().map(|x| NotNan::new(*x).unwrap()).collect()
    }

    #[test]
    fn log_dirichlet_pdf_uniform_is_log_gamma_n() {
        let alphas = vec![1.0, 1.0, 1.0];
        let theta = nn(&[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]);
        let v = log_dirichlet_pdf(&theta, &alphas);
        let expected = ln_gamma(3.0);
        assert!(
            (v - expected).abs() < 1e-12,
            "got {} expected {}",
            v,
            expected
        );
    }

    #[test]
    fn log_dirichlet_pdf_matches_beta_for_two_categories() {
        let alphas = vec![2.0, 3.0];
        for &x in &[0.25_f64, 0.5, 0.7] {
            let theta = nn(&[x, 1.0 - x]);
            let dir = log_dirichlet_pdf(&theta, &alphas);
            let beta = log_beta_pdf(x, 2.0, 3.0);
            assert!(
                (dir - beta).abs() < 1e-10,
                "x={} dir={} beta={}",
                x,
                dir,
                beta
            );
        }
    }

    #[test]
    fn log_dirichlet_pdf_zero_with_high_alpha_is_neg_inf() {
        let alphas = vec![2.0, 2.0];
        let theta = nn(&[0.0, 1.0]);
        assert_eq!(log_dirichlet_pdf(&theta, &alphas), f64::NEG_INFINITY);
    }

    #[test]
    fn signed_logsumexp_all_positive_matches_logsumexp_many() {
        let terms = vec![(0.5_f64, 1i8), (1.2, 1), (-0.3, 1)];
        let (l, s) = signed_logsumexp(terms.clone());
        let expected = logsumexp_many(terms.into_iter().map(|(l, _)| l));
        assert_eq!(s, 1);
        assert!((l - expected).abs() < 1e-12);
    }

    #[test]
    fn signed_logsumexp_subtraction() {
        let (l, s) = signed_logsumexp(vec![(0.0, 1), (-1.0, -1)]);
        assert_eq!(s, 1);
        assert!((l.exp() - (1.0 - (-1.0_f64).exp())).abs() < 1e-12);
    }

    #[test]
    fn signed_logsumexp_negative_total() {
        let (l, s) = signed_logsumexp(vec![(-1.0, 1), (0.0, -1)]);
        assert_eq!(s, -1);
        assert!((l.exp() - (1.0 - (-1.0_f64).exp())).abs() < 1e-12);
    }

    #[test]
    fn signed_logsumexp_exact_cancellation_is_zero() {
        let (l, s) = signed_logsumexp(vec![(0.7, 1), (0.7, -1)]);
        assert_eq!(s, 0);
        assert_eq!(l, f64::NEG_INFINITY);
    }

    #[test]
    fn signed_logsumexp_empty_is_zero() {
        let (l, s) = signed_logsumexp(std::iter::empty());
        assert_eq!(s, 0);
        assert_eq!(l, f64::NEG_INFINITY);
    }

    #[test]
    fn ln_factorial_small_values() {
        assert!((ln_factorial(0) - 0.0).abs() < 1e-12);
        assert!((ln_factorial(1) - 0.0).abs() < 1e-12);
        assert!((ln_factorial(5) - 120.0_f64.ln()).abs() < 1e-9);
    }

    #[test]
    fn log_gamma_pdf_integrates_to_one() {
        let (shape, rate) = (2.5, 1.5);
        let n = 200_000;
        let hi = 40.0;
        let dx = hi / n as f64;
        let total: f64 = (1..n)
            .map(|i| log_gamma_pdf(i as f64 * dx, shape, rate).exp() * dx)
            .sum();
        assert!((total - 1.0).abs() < 1e-6, "integral = {}", total);
    }

    #[test]
    fn log_gamma_pdf_out_of_support() {
        assert_eq!(log_gamma_pdf(0.0, 2.0, 1.0), f64::NEG_INFINITY);
        assert_eq!(log_gamma_pdf(-1.0, 2.0, 1.0), f64::NEG_INFINITY);
    }

    #[test]
    fn log_gamma_marginal_identity_term_is_zero() {
        assert!((log_gamma_marginal(2.0, 3.0, 0.0, 0.0)).abs() < 1e-12);
    }

    #[test]
    fn log_gamma_marginal_matches_brute_force() {
        let (a, beta, n, c) = (2.0, 1.5, 3.0, 2.0);
        let steps = 200_000;
        let hi = 60.0;
        let dx = hi / steps as f64;
        let total: f64 = (1..steps)
            .map(|i| {
                let x = i as f64 * dx;
                (n * x.ln() - c * x + log_gamma_pdf(x, a, beta)).exp() * dx
            })
            .sum();
        let expected = log_gamma_marginal(a, beta, n, c);
        assert!(
            (total.ln() - expected).abs() < 1e-5,
            "brute force {} vs closed form {}",
            total.ln(),
            expected
        );
    }

    #[test]
    fn log_dirichlet_pdf_shape_mismatch_returns_neg_inf() {
        let alphas = vec![1.0, 1.0];
        let theta = nn(&[0.5, 0.3, 0.2]);
        let result = std::panic::catch_unwind(|| log_dirichlet_pdf(&theta, &alphas));
        if let Ok(v) = result {
            assert_eq!(v, f64::NEG_INFINITY);
        }
    }
}
