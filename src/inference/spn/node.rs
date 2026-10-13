
use itertools::Itertools;

use super::coeff::{SpnCoeff, SpnLeaf};

use crate::inference::conjugate_pairs::suffstat::ContVarName;
use std::collections::{BTreeSet, HashMap};

use std::hash::{Hash, Hasher};

use std::rc::Rc;

#[derive(Debug, Clone)]
pub enum SpnKind<L: SpnLeaf> {
    Scalar,
    Leaf(L),
    Sum(Vec<Spn<L>>),
    Product(Vec<Spn<L>>),
}

#[derive(Debug, Clone)]
pub struct Spn<L: SpnLeaf> {
    pub(crate) log_coeff: L::Coeff,
    pub(crate) inner: Rc<SpnInner<L>>,
}

#[derive(Debug, Clone)]
pub struct SpnInner<L: SpnLeaf> {
    kind: SpnKind<L>,
    scope: BTreeSet<ContVarName>,
}


impl<L> PartialEq for SpnInner<L>
where
    L: SpnLeaf + PartialEq,
    L::Coeff: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.kind == other.kind
    }
}

impl<L> Eq for SpnInner<L>
where
    L: SpnLeaf + Eq,
    L::Coeff: Eq,
{
}

impl<L> Hash for SpnInner<L>
where
    L: SpnLeaf + Hash,
    L::Coeff: Hash,
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.kind.hash(state);
    }
}

impl<L> PartialEq for SpnKind<L>
where
    L: SpnLeaf + PartialEq,
    L::Coeff: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (SpnKind::Scalar, SpnKind::Scalar) => true,
            (SpnKind::Leaf(a), SpnKind::Leaf(b)) => a == b,
            (SpnKind::Sum(c1), SpnKind::Sum(c2)) | (SpnKind::Product(c1), SpnKind::Product(c2)) => {
                c1.len() == c2.len() && c1.iter().zip(c2).all(|(a, b)| spn_structural_eq(a, b))
            }
            _ => false,
        }
    }
}

impl<L> Eq for SpnKind<L>
where
    L: SpnLeaf + Eq,
    L::Coeff: Eq,
{
}

impl<L> Hash for SpnKind<L>
where
    L: SpnLeaf + Hash,
    L::Coeff: Hash,
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            SpnKind::Scalar => {}
            SpnKind::Leaf(lk) => lk.hash(state),
            SpnKind::Sum(children) | SpnKind::Product(children) => {
                children.len().hash(state);
                for c in children {
                    hash_spn_child(c, state);
                }
            }
        }
    }
}

fn spn_structural_eq<L: SpnLeaf>(a: &Spn<L>, b: &Spn<L>) -> bool
where
    L::Coeff: PartialEq,
{
    a.log_coeff == b.log_coeff && Rc::ptr_eq(&a.inner, &b.inner)
}

fn hash_spn_child<L: SpnLeaf, H: Hasher>(spn: &Spn<L>, state: &mut H)
where
    L::Coeff: Hash,
{
    spn.log_coeff.hash(state);
    (Rc::as_ptr(&spn.inner) as usize).hash(state);
}


impl<L: SpnLeaf> Spn<L> {
    fn new(log_coeff: L::Coeff, kind: SpnKind<L>, scope: BTreeSet<ContVarName>) -> Self {
        Spn {
            log_coeff,
            inner: L::intern(SpnInner { kind, scope }),
        }
    }

    pub fn scalar(log_coeff: f64) -> Self {
        Spn::new(
            L::Coeff::one().scale_log(log_coeff),
            SpnKind::Scalar,
            BTreeSet::new(),
        )
    }

    pub fn from_coeff(c: L::Coeff) -> Self {
        Spn::new(c, SpnKind::Scalar, BTreeSet::new())
    }

    pub fn zero() -> Self {
        Spn::new(L::Coeff::zero(), SpnKind::Scalar, BTreeSet::new())
    }

    pub fn one() -> Self {
        Spn::new(L::Coeff::one(), SpnKind::Scalar, BTreeSet::new())
    }

    pub fn leaf(l: L) -> Self {
        let scope = l.scope();
        debug_assert!(!scope.is_empty());
        Spn::new(L::Coeff::one(), SpnKind::Leaf(l), scope)
    }

    pub fn is_zero(&self) -> bool {
        self.log_coeff.is_zero()
    }

    pub(crate) fn is_one(&self) -> bool {
        self.log_coeff.is_one() && matches!(self.kind(), SpnKind::Scalar)
    }

    pub(crate) fn is_scalar(&self) -> bool {
        matches!(self.kind(), SpnKind::Scalar)
    }

    pub fn log_coeff(&self) -> L::Coeff {
        self.log_coeff
    }

    pub fn kind(&self) -> &SpnKind<L> {
        &self.inner.kind
    }

    pub fn scope(&self) -> &BTreeSet<ContVarName> {
        &self.inner.scope
    }

    pub fn inner_ptr(&self) -> *const SpnInner<L> {
        Rc::as_ptr(&self.inner)
    }

    pub(crate) fn scalar_mul(&self, log_s: f64) -> Self {
        Spn {
            log_coeff: self.log_coeff.scale_log(log_s),
            inner: self.inner.clone(),
        }
    }

    pub(crate) fn coeff_mul(&self, factor: L::Coeff) -> Self {
        Spn {
            log_coeff: self.log_coeff.product(factor),
            inner: self.inner.clone(),
        }
    }
}


impl<L: SpnLeaf> Spn<L> {
    pub fn sum(children: Vec<Spn<L>>) -> Self {
        let mut flat: Vec<Spn<L>> = Vec::new();
        for child in children {
            if child.is_zero() {
                continue;
            }
            match child.kind() {
                SpnKind::Sum(grandchildren) => {
                    let outer = child.log_coeff;
                    for gc in grandchildren {
                        let scaled = gc.coeff_mul(outer);
                        if !scaled.is_zero() {
                            flat.push(scaled);
                        }
                    }
                }
                _ => flat.push(child),
            }
        }

        let mut deduped: Vec<Spn<L>> = Vec::new();
        let mut ptr_to_idx: HashMap<*const SpnInner<L>, usize> = HashMap::new();
        for child in flat {
            let ptr = Rc::as_ptr(&child.inner);
            if let Some(&idx) = ptr_to_idx.get(&ptr) {
                deduped[idx].log_coeff = deduped[idx].log_coeff.logsumexp(child.log_coeff);
            } else {
                ptr_to_idx.insert(ptr, deduped.len());
                deduped.push(child);
            }
        }
        deduped.retain(|c| !c.is_zero());

        if deduped.is_empty() {
            return Self::zero();
        }
        if deduped.len() == 1 {
            return deduped.pop().unwrap();
        }

        let log_z = deduped
            .iter()
            .fold(L::Coeff::zero(), |acc, c| acc.logsumexp(c.log_coeff));
        if log_z.is_zero() {
            return Self::zero();
        }
        for c in deduped.iter_mut() {
            c.log_coeff = c.log_coeff.divide(log_z);
        }
        deduped.retain(|c| !c.is_zero());
        if deduped.is_empty() {
            return Self::zero();
        }
        if deduped.len() == 1 {
            let mut only = deduped.pop().unwrap();
            only.log_coeff = log_z.product(only.log_coeff);
            return only;
        }

        deduped.sort_unstable_by_key(|c| Rc::as_ptr(&c.inner) as usize);

        let scope = deduped.iter().fold(BTreeSet::new(), |mut acc, c| {
            acc.extend(c.scope());
            acc
        });
        Spn::new(log_z, SpnKind::Sum(deduped), scope)
    }

    pub fn product(children: Vec<Spn<L>>) -> Self {
        let mut flat: Vec<Spn<L>> = Vec::new();
        let mut scope: BTreeSet<ContVarName> = BTreeSet::new();
        let mut accum: L::Coeff = L::Coeff::one();

        fn pull_out<L: SpnLeaf>(mut child: Spn<L>, flat: &mut Vec<Spn<L>>, accum: &mut L::Coeff) {
            *accum = accum.product(child.log_coeff);
            child.log_coeff = L::Coeff::one();
            if !child.is_scalar() {
                flat.push(child);
            }
        }

        for child in children {
            if child.is_zero() {
                return Self::zero();
            }
            debug_assert!(scope.intersection(child.scope()).collect_vec().is_empty());
            scope.extend(child.scope().iter().copied());
            match child.kind() {
                SpnKind::Product(grandchildren) => {
                    accum = accum.product(child.log_coeff);
                    for gc in grandchildren {
                        pull_out::<L>(gc.clone(), &mut flat, &mut accum);
                    }
                }
                _ => pull_out::<L>(child, &mut flat, &mut accum),
            }
        }

        if flat.is_empty() {
            return Self::from_coeff(accum);
        }
        if flat.len() == 1 {
            return flat.pop().unwrap().coeff_mul(accum);
        }

        flat.sort_unstable_by_key(|c| Rc::as_ptr(&c.inner) as usize);
        Spn::new(accum, SpnKind::Product(flat), scope)
    }
}
