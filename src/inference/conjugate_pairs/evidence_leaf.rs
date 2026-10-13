
use std::cell::RefCell;
use std::collections::{BTreeSet, HashSet};
use std::rc::Rc;

use ordered_float::NotNan;

use crate::inference::conjugate_pairs::{
    BetaSuffStat, ContVarName, DirichletSuffStat, GammaSuffStat, GaussianObs, SuffStat,
};
use crate::inference::spn::coeff::SpnLeaf;
use crate::inference::spn::node::SpnInner;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum EvidenceLeaf {
    Beta(BetaSuffStat),
    Dirichlet(DirichletSuffStat),
    Gamma(GammaSuffStat),
    Gaussian(GaussianObs),
}


thread_local! {
    static INTERN_TABLE: RefCell<HashSet<Rc<SpnInner<EvidenceLeaf>>>> = RefCell::new(HashSet::new());
}

pub fn clear_intern_table() {
    INTERN_TABLE.with(|table| table.borrow_mut().clear());
}

pub fn intern_table_size() -> usize {
    INTERN_TABLE.with(|table| table.borrow().len())
}

impl SpnLeaf for EvidenceLeaf {
    type Coeff = NotNan<f64>;

    fn scope(&self) -> BTreeSet<ContVarName> {
        crate::for_each_evidence_family!(self, |s| s.scope().copied().collect())
    }

    fn intern(inner: SpnInner<Self>) -> Rc<SpnInner<Self>> {
        INTERN_TABLE.with(|table| {
            let mut table = table.borrow_mut();
            if let Some(existing) = table.get(&inner) {
                existing.clone()
            } else {
                let rc = Rc::new(inner);
                table.insert(rc.clone());
                rc
            }
        })
    }
}

impl EvidenceLeaf {
    pub fn try_merge(&self, other: &EvidenceLeaf) -> Option<(EvidenceLeaf, f64)> {
        match (self, other) {
            (EvidenceLeaf::Beta(a), EvidenceLeaf::Beta(b)) => {
                let (m, f) = a.merge(b);
                Some((EvidenceLeaf::Beta(m), f))
            }
            (EvidenceLeaf::Dirichlet(a), EvidenceLeaf::Dirichlet(b)) => {
                let (m, f) = a.merge(b);
                Some((EvidenceLeaf::Dirichlet(m), f))
            }
            (EvidenceLeaf::Gamma(a), EvidenceLeaf::Gamma(b)) => {
                let (m, f) = a.merge(b);
                Some((EvidenceLeaf::Gamma(m), f))
            }
            (EvidenceLeaf::Gaussian(a), EvidenceLeaf::Gaussian(b)) => {
                let (m, f) = a.merge(b);
                Some((EvidenceLeaf::Gaussian(m), f))
            }
            _ => None,
        }
    }
}
