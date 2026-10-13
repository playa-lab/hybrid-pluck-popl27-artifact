
use pluck::inference::conjugate_pairs::GammaPrior;
use pluck::{PluckContext, ResultKind};

fn run_on_big_stack(source: &'static str) -> Vec<pluck::QueryResult> {
    let handle = std::thread::Builder::new()
        .stack_size(16 * 1024 * 1024)
        .spawn(move || {
            let mut ctx = PluckContext::new();
            ctx.run(source)
        })
        .expect("spawn");
    match handle.join() {
        Ok(results) => results,
        Err(payload) => std::panic::resume_unwind(payload),
    }
}


#[test]
fn scaled_exponential_rate_compiles_and_runs() {
    let results = run_on_big_stack(
        "(query (let ((r (gamma 2.0 1.0))) \
           (Posterior r (close (exponential (*. r 5.0)) 0.3))))",
    );
    assert_eq!(results.len(), 1);
    assert!(matches!(results[0].kind, ResultKind::Distribution { .. }));
}

#[test]
fn scaled_poisson_rate_compiles_and_runs() {
    let results = run_on_big_stack(
        "(query (let ((r (gamma 2.0 1.0))) \
           (Posterior r (native_eq (poisson (*. r 1.5)) @2))))",
    );
    assert_eq!(results.len(), 1);
    assert!(matches!(results[0].kind, ResultKind::Distribution { .. }));
}

#[test]
fn dividing_gamma_rate_compiles_and_runs() {
    let results = run_on_big_stack(
        "(query (let ((r (gamma 2.0 1.0))) \
           (Posterior r (close (exponential (/. r 2.0)) 0.3))))",
    );
    assert_eq!(results.len(), 1);
    assert!(matches!(results[0].kind, ResultKind::Distribution { .. }));
}

#[test]
#[should_panic(expected = "addition/subtraction of a Gamma")]
fn adding_to_a_gamma_panics() {
    run_on_big_stack(
        "(query (let ((r (gamma 2.0 1.0))) \
           (Posterior r (close (exponential (+. r 1.0)) 0.3))))",
    );
}

#[test]
#[should_panic(expected = "can only be scaled by a numeric constant")]
fn multiplying_two_gammas_panics() {
    run_on_big_stack(
        "(query (let ((r (gamma 2.0 1.0))) \
           (Posterior r (close (exponential (*. r r)) 0.3))))",
    );
}

#[test]
#[should_panic(expected = "Exponential/Poisson draw")]
fn scaling_a_draw_panics() {
    run_on_big_stack(
        "(query (let ((r (gamma 2.0 1.0))) \
           (Posterior r (close (*. (exponential r) 5.0) 0.3))))",
    );
}

#[test]
#[should_panic(expected = "positive scale")]
fn scaling_a_gamma_by_zero_panics() {
    run_on_big_stack(
        "(query (let ((r (gamma 2.0 1.0))) \
           (Posterior r (close (exponential (*. r 0.0)) 0.3))))",
    );
}


fn single_gamma_posterior(source: &'static str) -> (f64, f64) {
    let results = run_on_big_stack(source);
    assert_eq!(results.len(), 1, "expected one query");
    let items = match &results[0].kind {
        ResultKind::Distribution { items } => items,
        other => panic!("expected Distribution, got {other:?}"),
    };
    assert_eq!(items.len(), 1, "expected a single posterior world");
    let gammas = &items[0].posteriors.gamma;
    assert_eq!(gammas.len(), 1, "expected exactly one gamma in the packet");
    match &gammas[0].gamma {
        GammaPrior::Gamma { shape, rate, .. } => (*shape, *rate),
        other => panic!("expected a Gamma posterior, got {other}"),
    }
}

#[test]
fn scaled_exponential_single_use_posterior() {
    let (shape, rate) = single_gamma_posterior(
        "(query (let ((g (gamma 2.0 1.0))) \
           (Posterior g (close (exponential (*. g 3.0)) 0.5))))",
    );
    assert!((shape - 3.0).abs() < 1e-9, "shape {shape}");
    assert!((rate - 2.5).abs() < 1e-9, "rate {rate}");
}

#[test]
fn scaled_exponential_reparameterization_equivalence() {
    let (shape, rate) = single_gamma_posterior(
        "(query (let ((g (gamma 2.0 1.0))) \
           (Posterior g (close (exponential (*. g 4.0)) 0.25))))",
    );
    assert!((shape - 3.0).abs() < 1e-9, "shape {shape}");
    assert!((rate - 2.0).abs() < 1e-9, "rate {rate}");
}

#[test]
fn pooled_scaled_exponentials_share_one_posterior() {
    let (shape, rate) = single_gamma_posterior(
        "(query (let ((g (gamma 2.0 1.0))) \
           (Posterior g (and (close (exponential (*. g 2.0)) 0.5) \
                             (close (exponential (*. g 4.0)) 0.25)))))",
    );
    assert!((shape - 4.0).abs() < 1e-9, "shape {shape}");
    assert!((rate - 3.0).abs() < 1e-9, "rate {rate}");
}

#[test]
fn scaled_exponential_draw_is_sampled_at_scaled_rate() {
    let results = run_on_big_stack(
        "(query x (let ((g (gamma 3.0 2.0))) \
           (PosteriorSamples (exponential (*. g 4.0)) (True) 6000)))",
    );
    let values = match &results[0].kind {
        ResultKind::Samples { values } => values,
        other => panic!("expected Samples, got {other:?}"),
    };
    let xs: Vec<f64> = values
        .iter()
        .filter_map(|s| s.parse::<f64>().ok())
        .collect();
    assert!(
        xs.len() > 5000,
        "expected mostly numeric samples, got {}",
        xs.len()
    );
    let mean = xs.iter().sum::<f64>() / xs.len() as f64;
    assert!(
        (mean - 0.25).abs() < 0.05,
        "empirical mean {mean:.4}, expected ≈ 0.25 (drawn at 4·g, not g)"
    );
}

#[test]
fn pooled_scaled_poisson_share_one_posterior() {
    let (shape, rate) = single_gamma_posterior(
        "(query (let ((g (gamma 2.0 1.0))) \
           (Posterior g (and (native_eq (poisson (*. g 2.0)) @1) \
                             (native_eq (poisson (*. g 3.0)) @4)))))",
    );
    assert!((shape - 7.0).abs() < 1e-9, "shape {shape}");
    assert!((rate - 6.0).abs() < 1e-9, "rate {rate}");
}


fn pinned_gamma_value(source: &'static str) -> f64 {
    let results = run_on_big_stack(source);
    let items = match &results[0].kind {
        ResultKind::Distribution { items } => items,
        other => panic!("expected Distribution, got {other:?}"),
    };
    assert_eq!(items.len(), 1, "expected a single posterior world");
    let gammas = &items[0].posteriors.gamma;
    assert_eq!(gammas.len(), 1, "expected exactly one gamma in the packet");
    match &gammas[0].gamma {
        GammaPrior::Pinned { value, .. } => *value,
        other => panic!("expected a Pinned posterior, got {other}"),
    }
}

fn posterior_prob(source: &'static str, target: &str) -> f64 {
    let results = run_on_big_stack(source);
    let items = match &results[0].kind {
        ResultKind::Distribution { items } => items,
        other => panic!("expected Distribution, got {other:?}"),
    };
    items
        .iter()
        .filter(|i| i.display == target)
        .map(|i| i.log_probability.exp())
        .sum()
}

#[test]
fn scaled_rate_pin_pins_the_quotient() {
    let scaled = pinned_gamma_value(
        "(query g (let ((g (gamma 2.0 1.0))) (Posterior g (close (*. g 4.0) 2.0))))",
    );
    let direct =
        pinned_gamma_value("(query g (let ((g (gamma 2.0 1.0))) (Posterior g (close g 0.5))))");
    assert!((scaled - 0.5).abs() < 1e-9, "scaled pin value {scaled}");
    assert!((direct - 0.5).abs() < 1e-9, "direct pin value {direct}");
}


#[test]
fn querying_scaled_gamma_reports_scaled_distribution() {
    let (shape, rate) = single_gamma_posterior(
        "(query scaled-g (let ((g (gamma 2.0 1.0))) \
           (Posterior (*. g 2.0) (close (exponential g) 0.5))))",
    );
    assert!((shape - 3.0).abs() < 1e-9, "shape {shape}");
    assert!((rate - 0.75).abs() < 1e-9, "rate {rate}");

    let (bshape, brate) = single_gamma_posterior(
        "(query g (let ((g (gamma 2.0 1.0))) \
           (Posterior g (close (exponential g) 0.5))))",
    );
    assert!((bshape - 3.0).abs() < 1e-9, "bare shape {bshape}");
    assert!((brate - 1.5).abs() < 1e-9, "bare rate {brate}");
}

#[test]
fn scaled_gamma_query_label_shows_the_scale() {
    let results = run_on_big_stack(
        "(query scaled-g (let ((g (gamma 2.0 1.0))) \
           (Posterior (*. g 2.0) (close (exponential g) 0.5))))",
    );
    let items = match &results[0].kind {
        ResultKind::Distribution { items } => items,
        other => panic!("expected Distribution, got {other:?}"),
    };
    assert_eq!(items.len(), 1);
    let display = &items[0].display;
    assert!(
        display.contains("2") && display.contains('*'),
        "display {display:?} should show the scale factor `2 * …`"
    );
}


fn gamma_entries(source: &'static str) -> Vec<GammaPrior<String>> {
    let results = run_on_big_stack(source);
    let items = match &results[0].kind {
        ResultKind::Distribution { items } => items,
        other => panic!("expected Distribution, got {other:?}"),
    };
    assert_eq!(items.len(), 1, "expected a single posterior world");
    items[0]
        .posteriors
        .gamma
        .iter()
        .map(|jg| jg.gamma.clone())
        .collect()
}

#[test]
fn pair_of_two_scales_reports_two_entries() {
    let entries = gamma_entries(
        "(query both (let ((g (gamma 2.0 1.0))) \
           (Posterior (Pair (*. g 2.0) (*. g 3.0)) (close (exponential g) 0.5))))",
    );
    assert_eq!(
        entries.len(),
        2,
        "expected two scaled entries, got {entries:?}"
    );
    let mut by_rate: Vec<(String, f64, f64)> = entries
        .iter()
        .map(|g| match g {
            GammaPrior::Gamma { name, shape, rate } => (name.clone(), *shape, *rate),
            other => panic!("expected Gamma, got {other}"),
        })
        .collect();
    by_rate.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
    let (n2, sh2, r2) = &by_rate[0];
    let (n3, sh3, r3) = &by_rate[1];
    assert!(
        (sh2 - 3.0).abs() < 1e-9 && (r2 - 0.75).abs() < 1e-9,
        "2·g entry {by_rate:?}"
    );
    assert!(
        (sh3 - 3.0).abs() < 1e-9 && (r3 - 0.5).abs() < 1e-9,
        "3·g entry {by_rate:?}"
    );
    assert!(n2.contains("2 *"), "2·g label {n2:?}");
    assert!(n3.contains("3 *"), "3·g label {n3:?}");
}

#[test]
fn scaled_rate_with_queried_draw_keeps_conditioning_rate_unscaled() {
    let entries = gamma_entries(
        "(query both (let ((g (gamma 2.0 1.0))) \
           (Posterior (Pair (*. g 2.0) (exponential g)) (True))))",
    );
    assert_eq!(
        entries.len(),
        2,
        "expected bare g + scaled 2·g, got {entries:?}"
    );
    let rates: Vec<f64> = entries
        .iter()
        .map(|g| match g {
            GammaPrior::Gamma { shape, rate, .. } => {
                assert!((shape - 2.0).abs() < 1e-9, "shape {shape}");
                *rate
            }
            other => panic!("expected Gamma, got {other}"),
        })
        .collect();
    assert!(
        rates.iter().any(|r| (r - 1.0).abs() < 1e-9),
        "expected an UNSCALED g (rate 1.0) for the draw to condition on: {rates:?}"
    );
    assert!(
        rates.iter().any(|r| (r - 0.5).abs() < 1e-9),
        "expected a scaled 2·g (rate 0.5): {rates:?}"
    );
}

#[test]
fn scaled_rate_pin_jacobian_reweights_competing_branches() {
    let p_true = posterior_prob(
        "(query c (let ((c (flip 0.5)) (g (gamma 2.0 1.0))) \
           (Posterior c (if c (close (*. g 4.0) 2.0) (close g 0.5)))))",
        "True",
    );
    assert!(
        (p_true - 0.2).abs() < 1e-9,
        "P(True) = {p_true}, expected 0.2 (scaled branch downweighted by 1/4)"
    );
}
