
use std::collections::BTreeSet;

pub(crate) fn add_to_sorted_by_name<P, K, F>(vec: &mut Vec<P>, item: P, key_of: F, label: &str)
where
    K: Ord + Copy + std::fmt::Display,
    P: PartialEq,
    F: Fn(&P) -> K,
{
    let new_key = key_of(&item);
    if let Some(existing) = vec.iter().find(|p| key_of(p) == new_key) {
        debug_assert!(
            existing == &item,
            "{}: re-registering name {} with different params",
            label,
            new_key
        );
        return;
    }
    debug_assert!(
        vec.iter().map(&key_of).is_sorted(),
        "{}: existing vec is not sorted by name",
        label
    );
    debug_assert!(
        vec.iter().map(&key_of).max().is_none_or(|m| m < new_key),
        "{}: new name {} is not larger than every existing name",
        label,
        new_key
    );
    vec.push(item);
}

pub(crate) fn lookup_sorted_by_name<'a, P, K, F>(
    vec: &'a [P],
    name: K,
    key_of: F,
    label: &str,
) -> &'a P
where
    K: Ord + Copy + std::fmt::Display,
    F: Fn(&P) -> K,
{
    match vec.binary_search_by_key(&name, &key_of) {
        Ok(i) => &vec[i],
        Err(_) => panic!("{}: variable {} not found", label, name),
    }
}

pub(crate) fn slice_sorted_by_name<P, K, F>(vec: &[P], scope: &BTreeSet<K>, key_of: F) -> Vec<P>
where
    K: Ord + Copy,
    P: Clone,
    F: Fn(&P) -> K,
{
    vec.iter()
        .filter(|p| scope.contains(&key_of(p)))
        .cloned()
        .collect()
}
