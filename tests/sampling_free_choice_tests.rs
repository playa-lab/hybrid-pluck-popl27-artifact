
use pluck::{PluckContext, ResultKind};

const N_SAMPLES: usize = 2000;
const TOL: f64 = 0.05;

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

fn frac(samples: &[String], target: &str) -> f64 {
    samples.iter().filter(|s| s.as_str() == target).count() as f64 / samples.len() as f64
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

fn assert_frac(samples: &[String], target: &str, expected: f64, label: &str) {
    let f = frac(samples, target);
    assert!(
        (f - expected).abs() < TOL,
        "{}: empirical P({}) = {:.4}, expected ≈ {:.4} (±{})",
        label,
        target,
        f,
        expected,
        TOL
    );
}


#[test]
fn free_flip_in_case_branch_follows_prior() {
    with_big_stack(|| {
        let source = "(query free-flip-in-branch
            (let ((obs (flip 0.5)))
              (PosteriorSamples (if obs (flip 0.2) (False)) obs 2000)))";
        let samples = run_samples(source);
        assert_eq!(samples.len(), N_SAMPLES);
        assert_frac(&samples, "True", 0.2, "free flip in case branch");
    });
}


#[test]
fn free_flip_in_function_body_follows_prior() {
    with_big_stack(|| {
        let source = "(define (draw u) (flip 0.2))
            (query free-flip-in-function
              (let ((obs (flip 0.5)))
                (PosteriorSamples (draw obs) obs 2000)))";
        let samples = run_samples(source);
        assert_eq!(samples.len(), N_SAMPLES);
        assert_frac(&samples, "True", 0.2, "free flip in function body");
    });
}


#[test]
fn free_categorical_in_branch_follows_prior() {
    with_big_stack(|| {
        let source = "(query free-categorical-in-branch
            (let ((obs (flip 0.5)))
              (PosteriorSamples (if obs (categorical 0.2 0.3 0.5) @9) obs 2000)))";
        let samples = run_samples(source);
        assert_eq!(samples.len(), N_SAMPLES);
        assert_frac(&samples, "0", 0.2, "free categorical");
        assert_frac(&samples, "1", 0.3, "free categorical");
        assert_frac(&samples, "2", 0.5, "free categorical");
    });
}


#[test]
fn free_flip_at_query_root_follows_prior() {
    with_big_stack(|| {
        let source = "(query free-flip-at-query-root
            (let ((obs (flip 0.5)))
              (PosteriorSamples (flip 0.2) obs 2000)))";
        let samples = run_samples(source);
        assert_eq!(samples.len(), N_SAMPLES);
        assert_frac(&samples, "True", 0.2, "control flip at root");
    });
}


#[test]
fn path_skipped_evidence_var_follows_prior() {
    with_big_stack(|| {
        let source = "(query path-skipped-evidence-var-follows-prior
            (let ((x (flip 0.5)) (y (flip 0.5)))
              (PosteriorSamples (if x y (False)) (or x y) 2000)))";
        let samples = run_samples(source);
        assert_eq!(samples.len(), N_SAMPLES);
        assert_frac(&samples, "True", 1.0 / 3.0, "path-skipped evidence var");
    });
}


#[test]
fn free_uniform_int_range_uniform_and_in_range() {
    with_big_stack(|| {
        let source = "(query free-uniform-int-range
            (let ((obs (flip 0.5)))
              (PosteriorSamples (uniform_int_range @2 @1 @3) obs 2000)))";
        let samples = run_samples(source);
        assert_eq!(samples.len(), N_SAMPLES);
        let zeros = samples.iter().filter(|s| s.as_str() == "@0").count();
        assert_eq!(
            zeros, 0,
            "{} out-of-range @0 samples (range is [1, 3])",
            zeros
        );
        assert_frac(&samples, "@1", 1.0 / 3.0, "uniform_int_range");
        assert_frac(&samples, "@2", 1.0 / 3.0, "uniform_int_range");
        assert_frac(&samples, "@3", 1.0 / 3.0, "uniform_int_range");
    });
}


#[test]
fn gibbs_free_flip_in_branch_follows_prior() {
    with_big_stack(|| {
        let source = "(query gibbs-free-flip-in-branch
            (let ((x (flip 0.5)) (y (flip 0.5)))
              (Gibbs (if x (flip 0.2) (False))
                     (or x y)
                     2000
                     (Cons x (Cons y (Nil)))
                     (WithPriorSample x))))";
        let samples = run_samples(source);
        assert_eq!(samples.len(), N_SAMPLES);
        assert_frac(
            &samples,
            "True",
            (2.0 / 3.0) * 0.2,
            "Gibbs free flip in branch",
        );
    });
}


#[test]
fn geometric_peano_follows_prior() {
    with_big_stack(|| {
        let source = "(define (geomp) (if (flip 0.5) (O) (S (geomp))))
            (query geom-peano (PosteriorSamples (geomp) (True) 2000))";
        let samples = run_samples(source);
        assert_eq!(samples.len(), N_SAMPLES);
        assert_frac(&samples, "0", 0.5, "geom-peano");
        assert_frac(&samples, "1", 0.25, "geom-peano");
        assert_frac(&samples, "2", 0.125, "geom-peano");
    });
}

#[test]
fn geometric_native_terminates_and_follows_prior() {
    with_big_stack(|| {
        let source = "(define (geomn)
              (if (flip 0.5) (mk_int @8 @0) (int_add (mk_int @8 @1) (geomn))))
            (query geom-native (PosteriorSamples (geomn) (True) 2000))";
        let samples = run_samples(source);
        assert_eq!(samples.len(), N_SAMPLES);
        assert_frac(&samples, "@0", 0.5, "geom-native");
        assert_frac(&samples, "@1", 0.25, "geom-native");
        assert_frac(&samples, "@2", 0.125, "geom-native");
    });
}


#[test]
fn within_sample_flip_consistency() {
    with_big_stack(|| {
        let source = "(query flip-consistency
            (let ((obs (flip 0.5)) (z (flip 0.3)))
              (PosteriorSamples (Pair z (if obs z (True))) obs 2000)))";
        let samples = run_samples(source);
        assert_eq!(samples.len(), N_SAMPLES);
        let agree_true = frac(&samples, "(True, True)");
        let agree_false = frac(&samples, "(False, False)");
        assert!(
            (agree_true + agree_false - 1.0).abs() < 1e-9,
            "flip consistency: {} mismatched pairs",
            samples
                .iter()
                .filter(|s| s.as_str() != "(True, True)" && s.as_str() != "(False, False)")
                .count()
        );
        assert!(
            (agree_true - 0.3).abs() < TOL,
            "flip consistency: empirical P(True pair) = {:.4}, expected ≈ 0.3",
            agree_true
        );
    });
}

#[test]
fn within_sample_int_dist_consistency() {
    with_big_stack(|| {
        let source = "(query int-consistency
            (let ((n (uniform_int_range @2 @1 @3)))
              (PosteriorSamples (Pair n n) (True) 2000)))";
        let samples = run_samples(source);
        assert_eq!(samples.len(), N_SAMPLES);
        for target in ["(@1, @1)", "(@2, @2)", "(@3, @3)"] {
            assert_frac(&samples, target, 1.0 / 3.0, "IntDist consistency");
        }
        let mismatched = samples
            .iter()
            .filter(|s| !["(@1, @1)", "(@2, @2)", "(@3, @3)"].contains(&s.as_str()))
            .count();
        assert_eq!(mismatched, 0, "IntDist consistency: mismatched pairs");
    });
}


#[test]
fn separate_applications_draw_independent_flips() {
    with_big_stack(|| {
        let source = "(define (draw u) (flip 0.5))
            (query independent-applications
              (let ((obs (flip 0.5)))
                (PosteriorSamples (Pair (draw obs) (draw obs)) obs 2000)))";
        let samples = run_samples(source);
        assert_eq!(samples.len(), N_SAMPLES);
        for target in [
            "(True, True)",
            "(True, False)",
            "(False, True)",
            "(False, False)",
        ] {
            assert_frac(&samples, target, 0.25, "independent applications");
        }
    });
}


#[test]
fn thunk_union_selection_follows_mixture() {
    with_big_stack(|| {
        let source = "(query tu-prob
            (match (if (flip 0.3) (Pair 0.2 (True)) (Pair 0.8 (True)))
              Pair p c => (PosteriorSamples (flip p) c 2000)))";
        let samples = run_samples(source);
        assert_eq!(samples.len(), N_SAMPLES);
        assert_frac(&samples, "True", 0.62, "ThunkUnion mixture");
    });
}

#[test]
fn shared_thunk_reevaluates_forward() {
    with_big_stack(|| {
        let source = "(query smuggle-let
            (let ((n (if (flip 0.5) (mk_int @2 @1) (mk_int @2 @2))))
              (PosteriorSamples (int_eq n (mk_int @2 @1)) (True) 2000)))";
        let samples = run_samples(source);
        assert_eq!(samples.len(), N_SAMPLES);
        assert_frac(&samples, "True", 0.5, "shared thunk forward re-eval");
    });
}

#[test]
fn recursive_query_target_memoizes() {
    with_big_stack(|| {
        let source = "(define EOS2 (mk_int @8 @36))
            (define (nxt prev)
              (if (int_dist_eq prev (mk_int @8 @65))
                (discrete ((mk_int @8 @66) 0.5) (EOS2 0.5))
                (discrete ((mk_int @8 @65) 0.5) (EOS2 0.5))))
            (define (gen prev)
              (let ((c (nxt prev)))
                (if (int_dist_eq c EOS2)
                  (Nil)
                  (Cons c (gen c)))))
            (query g (PosteriorSamples (gen (mk_int @8 @65)) (True) 2000))";
        let samples = run_samples(source);
        assert_eq!(samples.len(), N_SAMPLES);
        assert_frac(&samples, "[]", 0.5, "recursive query target: P(empty)");
        assert_frac(&samples, "\"B\"", 0.25, "recursive query target: P(\"B\")");
    });
}

#[test]
fn thunk_union_int_dist_selection() {
    with_big_stack(|| {
        let source = "(query tu-int
            (match (if (flip 0.5) (Pair (mk_int @2 @1) (True)) (Pair (mk_int @2 @2) (True)))
              Pair n c => (PosteriorSamples (int_eq n (mk_int @2 @1)) (True) 2000)))";
        let samples = run_samples(source);
        assert_eq!(samples.len(), N_SAMPLES);
        assert_frac(&samples, "True", 0.5, "ThunkUnion IntDist selection");
    });
}


#[test]
fn mid_sample_continuous_registration_dirichlet() {
    with_big_stack(|| {
        let source = "(define (draw-cat) (categorical (dirichlet 1.0 1.0)))
            (query cat-dirichlet
              (let ((obs (flip 0.5)))
                (PosteriorSamples (draw-cat) obs 2000)))";
        let samples = run_samples(source);
        assert_eq!(samples.len(), N_SAMPLES);
        assert_frac(&samples, "0", 0.5, "mid-sample dirichlet registration");
        assert_frac(&samples, "1", 0.5, "mid-sample dirichlet registration");
    });
}

#[test]
fn mid_sample_continuous_registration_gaussian() {
    with_big_stack(|| {
        let source = "(define (draw-g u) (normal 0.0 1.0))
            (query g-body
              (let ((obs (flip 0.5)))
                (PosteriorSamples (draw-g obs) obs 2000)))";
        let samples = run_samples(source);
        assert_eq!(samples.len(), N_SAMPLES);
        let vals = parse_floats(&samples);
        assert_eq!(vals.len(), N_SAMPLES, "all samples should be floats");
        let m = mean(&vals);
        assert!(
            m.abs() < 0.1,
            "mid-sample gaussian registration: sample mean = {:.4}, expected ≈ 0",
            m
        );
    });
}


#[test]
fn posterior_samples_free_choice_routes_gamma_draw() {
    with_big_stack(|| {
        let source = "(query free-choice-gamma-draw
            (let ((lam (gamma 2.0 1.0))
                  (w (flip 0.5)))
              (PosteriorSamples lam
                (if w
                    (real_lt (exponential lam) 1.0)
                    (real_lt (exponential lam) 2.0))
                2000)))";
        let samples = run_samples(source);
        let xs = parse_floats(&samples);
        assert_eq!(xs.len(), N_SAMPLES, "all samples should be floats");
        let m = mean(&xs);
        assert!(
            (m - 2.2429).abs() < 0.15,
            "free-choice gamma draw: empirical E[lam] = {:.4}, expected ≈ 2.2429",
            m
        );
    });
}

#[test]
fn gibbs_free_choice_routes_gamma_draw() {
    with_big_stack(|| {
        let source = "(query gibbs-free-choice-gamma-draw
            (let ((lam (gamma 2.0 1.0))
                  (w (flip 0.5))
                  (z (flip 0.5)))
              (Gibbs lam
                     (and z
                          (if w
                              (real_lt (exponential lam) 1.0)
                              (real_lt (exponential lam) 2.0)))
                     2000
                     (Cons z (Nil))
                     (WithConstant (True)))))";
        let samples = run_samples(source);
        let xs = parse_floats(&samples);
        assert_eq!(xs.len(), N_SAMPLES, "all samples should be floats");
        let m = mean(&xs);
        assert!(
            (m - 2.2429).abs() < 0.15,
            "Gibbs free-choice gamma draw: empirical E[lam] = {:.4}, expected ≈ 2.2429",
            m
        );
    });
}

#[test]
fn corner_case_shared_draw_uses_routing_posterior_weights() {
    with_big_stack(|| {
        let source = "(query corner-shared-draw-weights
            (let ((w (flip 0.9))
                  (z (flip 0.5))
                  (d (exponential (if w 1.0 10.0))))
              (PosteriorSamples d (or z (real_lt d 1.0)) 6000)))";
        let samples = run_samples(source);
        let xs = parse_floats(&samples);
        assert_eq!(xs.len(), 6000, "all samples should be floats");
        let m = mean(&xs);
        assert!(
            (m - 0.6938).abs() < 0.05,
            "corner-case shared draw: empirical E[d] = {:.4}, expected ≈ 0.6938 \
             (uniform-rate fill gives ≈ 0.473)",
            m
        );
    });
}

#[test]
fn corner_case_shared_poisson_draw_uses_routing_posterior_weights() {
    with_big_stack(|| {
        let source = "(query corner-shared-poisson-weights
            (let ((w (flip 0.9))
                  (z (flip 0.5))
                  (k (poisson (if w 1.0 10.0))))
              (PosteriorSamples k (or z (native_eq k @0)) 6000)))";
        let samples = run_samples(source);
        let xs = parse_floats(&samples);
        assert_eq!(xs.len(), 6000, "all samples should be floats");
        let m = mean(&xs);
        assert!(
            (m - 1.427).abs() < 0.2,
            "corner-case shared poisson draw: empirical E[k] = {:.4}, expected ≈ 1.427 \
             (uniform-rate fill gives ≈ 4.0)",
            m
        );
    });
}


#[test]
#[should_panic(expected = "program error along the sampled path")]
fn error_mass_along_sampled_path_panics() {
    with_big_stack(|| {
        let source = "(query err-branch
            (let ((obs (flip 0.5)))
              (PosteriorSamples (if obs (match (flip 0.3) True => (True)) (False)) obs 100)))";
        let _ = run_samples(source);
    });
}
