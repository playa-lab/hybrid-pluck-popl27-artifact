
use std::sync::OnceLock;

fn precision_override() -> Option<usize> {
    static P: OnceLock<Option<usize>> = OnceLock::new();
    *P.get_or_init(|| {
        std::env::var("PLUCK_DISPLAY_PRECISION")
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok())
    })
}

pub fn param_precision() -> usize {
    precision_override().unwrap_or(4)
}

pub fn weight_precision() -> usize {
    precision_override().unwrap_or(2)
}
