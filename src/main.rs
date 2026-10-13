
use std::collections::BTreeMap;
use std::fmt::{Display, Write as _};
use std::io::Write as _;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use indenter::indented;
use pluck::utils::math::logsumexp;
use pluck::{GibbsProgress, MixtureItem, PluckContext, QueryResult, ResultKind};

const CLI_SHOW_TOP: usize = 10;
const CLI_TAIL_LOG_THRESHOLD: f64 = -13.815510557964274;

fn main() {
    let mut ctx = PluckContext::new();

    let args: Vec<String> = std::env::args().collect();
    let filename = args.get(1).unwrap_or_else(|| {
        eprintln!("Usage: pluck <filename.pluck> [--json]");
        std::process::exit(1);
    });
    let source = std::fs::read_to_string(filename).unwrap_or_else(|e| {
        eprintln!("Error reading {}: {}", filename, e);
        std::process::exit(1);
    });

    let progress = GibbsProgress {
        current: Arc::new(AtomicUsize::new(0)),
        total: Arc::new(AtomicUsize::new(0)),
    };
    let done = Arc::new(AtomicBool::new(false));
    let monitor = std::thread::spawn(build_gibbs_logger(
        progress.current.clone(),
        progress.total.clone(),
        done.clone(),
    ));

    let results = ctx.run_with_progress(&source, Some(&progress));
    done.store(true, Ordering::Relaxed);
    let _ = monitor.join();

    if args.iter().any(|a| a == "--json") {
        println!("{}", serde_json::to_string(&results).unwrap());
        return;
    }
    for result in &results {
        print_query_result(result);
    }
}

fn build_gibbs_logger(
    counter: Arc<AtomicUsize>,
    num_samples: Arc<AtomicUsize>,
    done: Arc<AtomicBool>,
) -> impl Fn() {
    move || {
        let mut drew = false;
        while !done.load(Ordering::Relaxed) {
            let t = num_samples.load(Ordering::Relaxed);
            if t > 0 {
                let c = counter.load(Ordering::Relaxed).min(t);
                eprint!("\r  gibbs: {c}/{t}        ");
                let _ = std::io::stderr().flush();
                drew = true;
            }
            std::thread::sleep(Duration::from_millis(100));
        }
        if drew {
            eprint!("\r{:40}\r", "");
            let _ = std::io::stderr().flush();
        }
    }
}


fn print_query_result(result: &QueryResult) {
    println!("{}:", result.query);
    match &result.kind {
        ResultKind::Samples { values } => {
            for v in values {
                println!("  {}", v);
            }
        }
        ResultKind::Distribution { items } => print_distribution(items),
    }
    println!();
}

fn print_distribution(items: &[MixtureItem]) {
    let groups = group_by_display(items);

    let mut shown = 0usize;
    let mut hidden_count = 0usize;
    let mut hidden_log_total = f64::NEG_INFINITY;

    for group in &groups {
        if group.log_probability == f64::NEG_INFINITY {
            continue;
        }
        if shown >= CLI_SHOW_TOP && group.log_probability < CLI_TAIL_LOG_THRESHOLD {
            hidden_count += 1;
            hidden_log_total = logsumexp(hidden_log_total, group.log_probability);
            continue;
        }
        shown += 1;
        print_group(group);
    }

    if hidden_count > 0 {
        println!(
            "  ... and {} more components (combined prob: {})",
            hidden_count,
            format_prob(hidden_log_total),
        );
    }
}


struct Group<'a> {
    display: &'a str,
    log_probability: f64,
    items: Vec<&'a MixtureItem>,
}

fn group_by_display(items: &[MixtureItem]) -> Vec<Group<'_>> {
    let mut by_display: BTreeMap<&str, Vec<&MixtureItem>> = BTreeMap::new();
    for item in items {
        by_display.entry(&item.display).or_default().push(item);
    }
    let mut groups: Vec<Group<'_>> = by_display
        .into_iter()
        .map(|(display, items)| {
            let log_probability = items
                .iter()
                .map(|i| i.log_probability)
                .fold(f64::NEG_INFINITY, logsumexp);
            Group {
                display,
                log_probability,
                items,
            }
        })
        .collect();
    groups.sort_by(|a, b| {
        b.log_probability
            .partial_cmp(&a.log_probability)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    groups
}


fn print_group(group: &Group<'_>) {
    println!(
        "  {}  {}",
        group.display,
        format_prob(group.log_probability),
    );

    if group.items.len() <= 1 {
        if let Some(item) = group.items.first() {
            print_indented(&item.posteriors, "    ");
        }
        return;
    }

    let n = group.items.len();
    for (i, item) in group.items.iter().enumerate() {
        let is_last = i + 1 == n;
        let connector = if is_last { '└' } else { '├' };
        let prefix = if is_last { "        " } else { "    │   " };
        let rel_weight = (item.log_probability - group.log_probability).exp();
        let wp = pluck::utils::display_precision::weight_precision();
        println!("    {} {:.wp$}%", connector, rel_weight * 100.0, wp = wp);
        print_indented(&item.posteriors, prefix);
    }
}

fn print_indented(value: &impl Display, prefix: &'static str) {
    let mut buf = String::new();
    write!(indented(&mut buf).with_str(prefix), "{}", value).unwrap();
    print!("{}", buf);
}

fn format_prob(log_p: f64) -> String {
    let p = log_p.exp();
    if p > 0.0 && p.is_finite() {
        format!("{}", p)
    } else {
        let log10 = log_p / std::f64::consts::LN_10;
        let exponent = log10.floor() as i64;
        let mantissa = 10_f64.powf(log10 - exponent as f64);
        format!("{:.2}e{}", mantissa, exponent)
    }
}
