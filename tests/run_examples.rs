
use std::fs;
use std::panic::{self, AssertUnwindSafe};
use std::path::{Path, PathBuf};

use pluck::{PluckContext, ResultKind};

fn examples_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples")
}

fn example_files() -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = fs::read_dir(examples_dir())
        .expect("read examples/ dir")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("pluck"))
        .collect();
    files.sort();
    files
}

fn register_shared(ctx: &mut PluckContext) {
    let shared = examples_dir().join("shared");
    for entry in fs::read_dir(&shared).expect("read examples/shared/ dir") {
        let path = entry.expect("shared dir entry").path();
        if path.extension().and_then(|s| s.to_str()) != Some("pluck") {
            continue;
        }
        let name = path.file_name().and_then(|s| s.to_str()).unwrap();
        let content = fs::read_to_string(&path).expect("read shared include");
        ctx.register_file(format!("examples/shared/{name}"), content);
    }
}

fn variant_family(stem: &str) -> Option<(&str, u64)> {
    let digits_start = stem
        .rfind(|c: char| !c.is_ascii_digit())
        .map(|i| i + 1)
        .unwrap_or(0);
    if digits_start == stem.len() {
        return None;
    }
    let (prefix, digits) = stem.split_at(digits_start);
    let bytes = prefix.as_bytes();
    let n = bytes.len();
    if n >= 2 && bytes[n - 1].is_ascii_alphabetic() && bytes[n - 2] == b'_' {
        digits.parse::<u64>().ok().map(|size| (prefix, size))
    } else {
        None
    }
}

const PREFER_LARGEST: &[&str] = &["king_ei_d"];

fn select_programs(files: &[PathBuf]) -> Vec<PathBuf> {
    use std::collections::HashMap;

    let mut singletons: Vec<PathBuf> = Vec::new();
    let mut chosen: HashMap<String, (u64, PathBuf)> = HashMap::new();

    for path in files {
        let stem = path.file_stem().and_then(|s| s.to_str()).unwrap();
        match variant_family(stem) {
            None => singletons.push(path.clone()),
            Some((family, size)) => {
                let prefer_largest = PREFER_LARGEST.contains(&family);
                let entry = chosen
                    .entry(family.to_string())
                    .or_insert_with(|| (size, path.clone()));
                let better = if prefer_largest {
                    size > entry.0
                } else {
                    size < entry.0
                };
                if better {
                    *entry = (size, path.clone());
                }
            }
        }
    }

    let mut selected = singletons;
    selected.extend(chosen.into_values().map(|(_, path)| path));
    selected.sort();
    selected
}

fn result_is_nonempty(kind: &ResultKind) -> bool {
    match kind {
        ResultKind::Distribution { items } => !items.is_empty(),
        ResultKind::Samples { values } => !values.is_empty(),
    }
}

fn check_example(path: &Path) -> Result<(), String> {
    let source = fs::read_to_string(path).map_err(|e| format!("failed to read: {e}"))?;

    let outcome = panic::catch_unwind(AssertUnwindSafe(|| {
        let mut ctx = PluckContext::new();
        register_shared(&mut ctx);
        ctx.run(&source)
    }));

    let results = match outcome {
        Ok(results) => results,
        Err(payload) => {
            let msg = payload
                .downcast_ref::<&str>()
                .map(|s| s.to_string())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "<non-string panic>".to_string());
            return Err(format!("panicked: {msg}"));
        }
    };

    if results.is_empty() {
        return Err("produced no query results".to_string());
    }
    for r in &results {
        if !result_is_nonempty(&r.kind) {
            return Err(format!("query `{}` produced an empty result", r.query));
        }
    }
    Ok(())
}

#[test]
fn examples_run_correctly() {
    std::thread::Builder::new()
        .stack_size(1024 * 1024 * 1024)
        .spawn(run_all)
        .expect("spawn worker thread")
        .join()
        .expect("worker thread panicked");
}

fn run_all() {
    let files = example_files();
    assert!(!files.is_empty(), "no .pluck files found in examples/");
    let selected = select_programs(&files);

    let prev_hook = panic::take_hook();
    panic::set_hook(Box::new(|_| {}));

    let mut failures: Vec<String> = Vec::new();
    for path in &selected {
        let name = path.file_name().and_then(|s| s.to_str()).unwrap();
        let start = std::time::Instant::now();
        let result = check_example(path);
        eprintln!("  {name} ({:?})", start.elapsed());
        if let Err(reason) = result {
            failures.push(format!("  {name}: {reason}"));
        }
    }

    panic::set_hook(prev_hook);

    if !failures.is_empty() {
        panic!(
            "{} of {} selected example(s) failed to run correctly:\n{}",
            failures.len(),
            selected.len(),
            failures.join("\n")
        );
    }
}
