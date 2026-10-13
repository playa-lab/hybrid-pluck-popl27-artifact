use num_traits::Num;

#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct Interval<T: Ord + Copy + Num> {
    pub geq: T,
    pub lt: Option<T>,
}

impl<T: Ord + Copy + Num> Interval<T> {
    pub fn lt(b: T) -> Self {
        Interval {
            geq: T::zero(),
            lt: Some(b),
        }
    }

    pub fn geq(b: T) -> Self {
        Interval { geq: b, lt: None }
    }

    pub fn is_empty(&self) -> bool {
        self.lt.is_some_and(|lt| lt <= self.geq)
    }

    pub fn contains(&self, v: T) -> bool {
        v >= self.geq && self.lt.is_none_or(|lt| v < lt)
    }

    pub fn intersect(&self, other: &Self) -> Interval<T> {
        let lt = match (self.lt, other.lt) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (Some(a), None) | (None, Some(a)) => Some(a),
            (None, None) => None,
        };
        Interval {
            geq: self.geq.max(other.geq),
            lt,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum IntervalOrEq<T: Ord + Copy + Num> {
    Interval(Interval<T>),
    Eq(T),
}

impl<T: Ord + Copy + Num> IntervalOrEq<T> {
    pub fn eq(v: T) -> Self {
        IntervalOrEq::Eq(v)
    }

    pub fn lt(b: T) -> Self {
        IntervalOrEq::Interval(Interval::lt(b))
    }

    pub fn geq(b: T) -> Self {
        IntervalOrEq::Interval(Interval::geq(b))
    }

    pub fn is_empty(&self) -> bool {
        match self {
            IntervalOrEq::Interval(interval) => interval.is_empty(),
            IntervalOrEq::Eq(_) => false,
        }
    }

    pub fn contains(&self, v: T) -> bool {
        match self {
            IntervalOrEq::Interval(interval) => interval.contains(v),
            IntervalOrEq::Eq(e) => v == *e,
        }
    }

    pub fn merge(&self, other: &Self) -> Option<Self> {
        match (self, other) {
            (IntervalOrEq::Interval(i1), IntervalOrEq::Interval(i2)) => {
                let merged = i1.intersect(i2);
                (!merged.is_empty()).then_some(IntervalOrEq::Interval(merged))
            }
            (IntervalOrEq::Eq(v), IntervalOrEq::Interval(i))
            | (IntervalOrEq::Interval(i), IntervalOrEq::Eq(v)) => {
                i.contains(*v).then_some(IntervalOrEq::Eq(*v))
            }
            (IntervalOrEq::Eq(v1), IntervalOrEq::Eq(v2)) => {
                (v1 == v2).then_some(IntervalOrEq::Eq(*v1))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intersect_narrows_to_singleton() {
        let r = Interval::lt(3).intersect(&Interval::geq(2));
        assert_eq!(
            r,
            Interval {
                geq: 2,
                lt: Some(3)
            }
        );
        assert!(!r.is_empty());
        assert!(r.contains(2));
        assert!(!r.contains(3));
    }

    #[test]
    fn intersect_empty_detected() {
        let r = Interval::lt(3).intersect(&Interval::geq(3));
        assert!(r.is_empty());
    }

    #[test]
    fn intersect_unbounded_pair_stays_unbounded() {
        let r = Interval::geq(2).intersect(&Interval::geq(5));
        assert_eq!(r, Interval::geq(5));
    }

    #[test]
    fn boundary_belongs_to_exactly_one_side() {
        let below = Interval::lt(3);
        let above = Interval::geq(3);
        assert!(!below.contains(3));
        assert!(above.contains(3));
    }

    #[test]
    fn merge_eq_inside_and_outside_interval() {
        let m = IntervalOrEq::eq(2).merge(&IntervalOrEq::geq(1));
        assert_eq!(m, Some(IntervalOrEq::eq(2)));
        assert_eq!(IntervalOrEq::eq(2).merge(&IntervalOrEq::lt(2)), None);
    }

    #[test]
    fn merge_eq_eq_match_and_mismatch() {
        assert_eq!(
            IntervalOrEq::eq(5).merge(&IntervalOrEq::eq(5)),
            Some(IntervalOrEq::eq(5))
        );
        assert_eq!(IntervalOrEq::eq(5).merge(&IntervalOrEq::eq(6)), None);
    }

    #[test]
    fn merge_empty_intersection_is_none() {
        assert_eq!(IntervalOrEq::lt(2).merge(&IntervalOrEq::geq(3)), None);
    }
}
