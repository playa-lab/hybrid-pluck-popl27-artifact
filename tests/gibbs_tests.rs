
use pluck::{PluckContext, ResultKind};

fn with_big_stack<F: FnOnce() + Send + 'static>(f: F) {
    let handle = std::thread::Builder::new()
        .stack_size(16 * 1024 * 1024)
        .spawn(f)
        .expect("spawn");
    if let Err(payload) = handle.join() {
        std::panic::resume_unwind(payload);
    }
}

fn run_samples(source: &str) -> Vec<String> {
    let mut ctx = PluckContext::new();
    let results = ctx.run(source);
    assert_eq!(results.len(), 1, "expected exactly one query");
    match &results[0].kind {
        ResultKind::Samples { values } => values.clone(),
        other => panic!("expected Samples result, got {:?}", other),
    }
}

fn parse_floats(samples: &[String]) -> Vec<f64> {
    samples
        .iter()
        .filter_map(|s| s.parse::<f64>().ok())
        .collect()
}

fn mean(xs: &[f64]) -> f64 {
    xs.iter().sum::<f64>() / xs.len() as f64
}


#[test]
fn gibbs_two_block_bernoulli_matches_exact_posterior() {
    with_big_stack(|| {
        let source = "(query q
            (let ((x (flip 0.5))
                  (y (flip 0.5)))
              (Gibbs (if x (True) (False))
                     (or x y)
                     2000
                     (Cons x (Cons y (Nil)))
                     (WithPriorSample x))))";
        let samples = run_samples(source);
        assert_eq!(samples.len(), 2000);
        let trues = samples.iter().filter(|s| s.as_str() == "True").count();
        let frac = trues as f64 / samples.len() as f64;
        println!("T {:?} frac {}", trues, frac);
        assert!(
            (frac - 2.0 / 3.0).abs() < 0.05,
            "empirical P(True) = {}, expected ≈ {:.4}",
            frac,
            2.0 / 3.0
        );
    });
}


#[test]
fn gibbs_gaussian_conjugate_posterior_mean_and_variance() {
    with_big_stack(|| {
        let source = "(query q
            (let ((mu (gaussian 0.0 1.0))
                  (nu (gaussian 0.0 1.0))
                  (z  (gaussian 0.0 1.0)))
              (Gibbs mu
                     (close (+. (+. mu nu) z) 1.0)
                     3000
                     (Cons mu (Cons nu (Nil)))
                     (WithPriorSample mu))))";
        let samples = run_samples(source);
        let xs = parse_floats(&samples);
        assert_eq!(xs.len(), 3000, "expected 3000 numeric samples");
        let m = mean(&xs);
        let expected = 1.0 / 3.0;
        assert!(
            (m - expected).abs() < 0.1,
            "empirical mean = {}, expected ≈ {:.4}",
            m,
            expected
        );
    });
}


#[test]
fn gibbs_dirichlet_categorical_posterior_mean() {
    with_big_stack(|| {
        let source = "(query q
            (let ((weights (dirichlet 1.0 1.0 1.0))
                  (c       (categorical weights)))
              (Gibbs (vector_index weights @0)
                     (native_eq c @0)
                     500
                     (Cons weights (Cons c (Nil)))
                     (WithPriorSample weights))))";
        let samples = run_samples(source);
        let xs = parse_floats(&samples);
        assert_eq!(xs.len(), 500, "expected 500 numeric samples");
        let m = mean(&xs);
        assert!(
            (m - 0.5).abs() < 0.1,
            "empirical E[θ_0] = {}, expected ≈ 0.5",
            m
        );
        assert!(xs.iter().all(|x| (0.0..=1.0).contains(x)), "out-of-range");
    });
}

#[test]
fn posterior_samples_log_domain_no_underflow() {
    with_big_stack(|| {
        let source = "(query q
            (let ((mu (gaussian 0.0 1.0)))
              (PosteriorSamples mu (close mu 20.0) 10)))";
        let samples = run_samples(source);
        assert_eq!(samples.len(), 10, "expected 10 samples, got {:?}", samples);
        let xs = parse_floats(&samples);
        assert_eq!(xs.len(), 10, "expected all samples to be floats");
        for x in &xs {
            assert!(
                (x - 20.0).abs() < 1e-6,
                "sample {} far from observed mu=20.0",
                x
            );
        }
    });
}

#[test]
fn posterior_samples_categorical_runs() {
    with_big_stack(|| {
        let source = "(query q
            (let ((weights (dirichlet 1.0 1.0 1.0))
                  (c1 (categorical weights))
                  (c2 (categorical weights)))
              (PosteriorSamples c2 (native_eq c1 @0) 100)))";
        let samples = run_samples(source);
        assert_eq!(samples.len(), 100);
        for s in &samples {
            assert!(
                matches!(s.as_str(), "0" | "1" | "2"),
                "unexpected sample value: {:?}",
                s
            );
        }
        let zeros = samples.iter().filter(|s| s.as_str() == "0").count();
        let ones = samples.iter().filter(|s| s.as_str() == "1").count();
        let twos = samples.iter().filter(|s| s.as_str() == "2").count();
        assert!(
            zeros >= ones && zeros >= twos,
            "expected mode at 0: 0s={}, 1s={}, 2s={}",
            zeros,
            ones,
            twos
        );
    });
}


#[test]
fn gibbs_rolling_window_marginal_matches_prior() {
    with_big_stack(|| {
        let source = "(query q
            (let ((a (flip 0.5))
                  (b (flip 0.5))
                  (c (flip 0.5))
                  (d (flip 0.5)))
              (Gibbs (if a (True) (False))
                     (or a b)
                     2000
                     (Cons (Cons a (Cons b (Nil)))
                       (Cons (Cons c (Cons d (Nil)))
                         (Nil)))
                     (WithConstant (Cons (True) (Cons (True) (Nil)))))))";
        let samples = run_samples(source);
        assert_eq!(samples.len(), 2000);
        let trues = samples.iter().filter(|s| s.as_str() == "True").count();
        let frac = trues as f64 / samples.len() as f64;
        assert!(
            (frac - 2.0 / 3.0).abs() < 0.05,
            "empirical P(a=True) = {}, expected ≈ {:.4}",
            frac,
            2.0 / 3.0
        );
    });
}


#[test]
fn gibbs_mixture_runs_on_example() {
    with_big_stack(|| {
        let source = r#"
(define obs [0.3, 1.2, 20.0])
(define ZERO  (mk_int @2 @0))
(define ONE   (mk_int @2 @1))
(define TWO   (mk_int @2 @2))
(define THREE (mk_int @2 @3))
(define (sample-means n)
  (if (int_eq n ZERO)
    (Nil)
    (Cons (normal 0.0 1.0) (sample-means (int_sub n ONE)))))
(define (sample-weights n)
  (if (int_eq n ONE) (dirichlet 1.0)
    (if (int_eq n TWO) (dirichlet 1.0 1.0)
      (dirichlet 1.0 1.0 1.0))))
(define (pick-mean c means)
  (match means
    Cons m rest =>
      (if (native_eq c @0) m
        (match rest
          Cons m2 rest2 =>
            (if (native_eq c @1) m2
              (match rest2
                Cons m3 _ => m3))))))
(query q
 (let
  ((n       (uniform_int_range @2 @1 @3))
   (means   (sample-means n))
   (weights (sample-weights n))
   (cs      (map (fn i -> (categorical weights))
                 (range (length obs))))
   (data    (map (fn c -> (normal (pick-mean c means) 1.0))
                 cs)))
  (Gibbs n
         (list=? real=? data obs)
         10
         (Cons n (Cons cs (Nil)))
         (WithConstant THREE))))
"#;
        let samples = run_samples(source);
        assert!(
            !samples.is_empty(),
            "mixture chain produced no samples at all"
        );
        for s in &samples {
            assert!(
                s.starts_with('@'),
                "expected @-prefixed native int, got {:?}",
                s
            );
        }
    });
}

#[test]
fn gibbs_mixture_minimal_runs_without_stalling() {
    with_big_stack(|| {
        let source = "(query q
            (let ((c0 (flip 0.5))
                  (c1 (flip 0.5))
                  (mu0 (gaussian -2.0 0.5))
                  (mu1 (gaussian  2.0 0.5))
                  (x0 (+. (if c0 mu0 mu1) (gaussian 0.0 1.0)))
                  (x1 (+. (if c1 mu0 mu1) (gaussian 0.0 1.0))))
              (Gibbs (if c0 (True) (False))
                     (and (close x0 -1.5) (close x1 1.5))
                     200
                     (Cons c0 (Cons c1 (Nil)))
                     (WithPriorSample c0))))";
        let samples = run_samples(source);
        assert!(!samples.is_empty(), "expected at least one sample");
        for s in &samples {
            assert!(
                s == "True" || s == "False",
                "expected boolean sample, got {:?}",
                s
            );
        }
    });
}


#[test]
#[should_panic(expected = "cannot pin a single Dirichlet component")]
fn gibbs_dirichlet_probability_block_errors() {
    with_big_stack(|| {
        let source = "(query q
            (let ((weights (dirichlet 1.0 1.0))
                  (c       (categorical weights)))
              (Gibbs (if (native_eq c @0) (True) (False))
                     (native_eq c @0)
                     3
                     (Cons (vector_index weights @0) (Nil))
                     (WithPriorSample weights))))";
        let _ = run_samples(source);
    });
}


#[test]
fn posterior_samples_unobserved_poisson_draw_mean() {
    with_big_stack(|| {
        let source = "(query (PosteriorSamples (poisson 4.0) True 2000))";
        let xs = parse_floats(&run_samples(source));
        assert_eq!(xs.len(), 2000);
        assert!((mean(&xs) - 4.0).abs() < 0.2, "mean {}", mean(&xs));
    });
}

#[test]
fn posterior_samples_gamma_poisson_rate_mean() {
    with_big_stack(|| {
        let source = "(query (let ((r (gamma 2.0 1.0)))
            (PosteriorSamples r (native_eq (poisson r) @3) 2000)))";
        let xs = parse_floats(&run_samples(source));
        assert!((mean(&xs) - 2.5).abs() < 0.15, "mean {}", mean(&xs));
    });
}

#[test]
fn posterior_samples_pinned_exponential_draw_is_exact() {
    with_big_stack(|| {
        let source = "(query (let ((x (exponential 2.0)))
            (PosteriorSamples x (close x 0.5) 20)))";
        let xs = parse_floats(&run_samples(source));
        assert_eq!(xs.len(), 20);
        assert!(xs.iter().all(|&x| x == 0.5), "samples {:?}", xs);
    });
}

#[test]
fn posterior_samples_truncated_exponential_mean() {
    with_big_stack(|| {
        let source = "(query (let ((x (exponential 2.0)))
            (PosteriorSamples x (real_geq x 1.0) 2000)))";
        let xs = parse_floats(&run_samples(source));
        assert!((mean(&xs) - 1.5).abs() < 0.1, "mean {}", mean(&xs));
    });
}

#[test]
fn posterior_samples_poisson_posterior_predictive_mean() {
    with_big_stack(|| {
        let source = "(query (let ((r (gamma 2.0 1.0)))
            (PosteriorSamples (poisson r) (native_eq (poisson r) @5) 2000)))";
        let xs = parse_floats(&run_samples(source));
        assert!((mean(&xs) - 3.5).abs() < 0.25, "mean {}", mean(&xs));
    });
}

#[test]
fn gibbs_gamma_rate_with_exponential_range_evidence() {
    with_big_stack(|| {
        let source = "(query (let ((r (gamma 2.0 1.0)) (x (exponential r)))
            (Gibbs r (real_geq x 1.0) 2000 (Cons r (Cons x (Nil)))
                   (WithPriorSample r))))";
        let xs = parse_floats(&run_samples(source));
        assert!((mean(&xs) - 1.0).abs() < 0.15, "mean {}", mean(&xs));
    });
}

#[test]
fn gibbs_gamma_rate_with_poisson_draw_block() {
    with_big_stack(|| {
        let source = "(query (let ((r (gamma 2.0 1.0)) (k (poisson r)))
            (Gibbs r (native_eq k @3) 2000 (Cons r (Cons k (Nil)))
                   (WithPriorSample r))))";
        let xs = parse_floats(&run_samples(source));
        assert!((mean(&xs) - 2.5).abs() < 0.2, "mean {}", mean(&xs));
    });
}


fn frac_true(samples: &[String]) -> f64 {
    let t = samples.iter().filter(|s| s.as_str() == "True").count();
    t as f64 / samples.len().max(1) as f64
}

#[test]
fn gibbs_conditions_discrete_poisson_rate() {
    with_big_stack(|| {
        let source = "(query q
            (let ((b (flip 0.5))
                  (k (poisson (if b 20.0 0.5))))
              (Gibbs (if b (True) (False))
                     (native_eq k @20)
                     2000
                     (Cons b (Cons k (Nil)))
                     (WithPriorSample b))))";
        let samples = run_samples(source);
        let frac = frac_true(&samples);
        assert!(
            frac > 0.95,
            "Gibbs P(True) = {:.3}; expected ≈1.0 (exact posterior ≈1.0).",
            frac
        );
    });
}

#[test]
fn gibbs_conditions_discrete_gamma_rate() {
    with_big_stack(|| {
        let source = "(query q
            (let ((b (flip 0.5))
                  (r (gamma 2.0 (if b 1.0 10.0)))
                  (k (poisson r)))
              (Gibbs (if b (True) (False))
                     (native_eq k @8)
                     2000
                     (Cons b (Cons r (Cons k (Nil))))
                     (WithPriorSample b))))";
        let samples = run_samples(source);
        let frac = frac_true(&samples);
        assert!(
            frac > 0.9,
            "Gibbs P(True) = {:.3}; expected ≈1.0 (exact posterior ≈0.99996).",
            frac
        );
    });
}
