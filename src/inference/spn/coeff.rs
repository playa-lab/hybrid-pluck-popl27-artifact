
use std::collections::BTreeSet;
use std::fmt::Debug;
use std::rc::Rc;

use ordered_float::NotNan;

use crate::utils::epsilon::RealEps;
use crate::utils::math::logsumexp;

use crate::inference::conjugate_pairs::ContVarName;

pub trait SpnCoeff: Copy + Debug {
    fn zero() -> Self;

    fn one() -> Self;

    fn is_zero(self) -> bool;
    fn is_one(self) -> bool;

    fn logsumexp(self, rhs: Self) -> Self;

    fn product(self, rhs: Self) -> Self;

    fn scale_log(self, log_s: f64) -> Self;

    fn divide(self, rhs: Self) -> Self;
}

impl SpnCoeff for NotNan<f64> {
    fn zero() -> Self {
        NotNan::new(f64::NEG_INFINITY).unwrap()
    }
    fn one() -> Self {
        NotNan::new(0.0).unwrap()
    }
    fn is_zero(self) -> bool {
        self.into_inner() == f64::NEG_INFINITY
    }
    fn is_one(self) -> bool {
        self.into_inner().abs() < 1e-15
    }
    fn logsumexp(self, rhs: Self) -> Self {
        NotNan::new(logsumexp(self.into_inner(), rhs.into_inner())).unwrap()
    }
    fn product(self, rhs: Self) -> Self {
        self + rhs
    }
    fn scale_log(self, log_s: f64) -> Self {
        self + NotNan::new(log_s).unwrap()
    }
    fn divide(self, rhs: Self) -> Self {
        NotNan::new(self.into_inner() - rhs.into_inner()).unwrap()
    }
}

impl SpnCoeff for RealEps {
    fn zero() -> Self {
        RealEps::zero()
    }
    fn one() -> Self {
        RealEps::from_log(0.0, 0)
    }
    fn is_zero(self) -> bool {
        RealEps::is_zero(&self)
    }
    fn is_one(self) -> bool {
        self.power == 0 && self.log_coeff.abs() < 1e-15
    }
    fn logsumexp(self, rhs: Self) -> Self {
        self + rhs
    }
    fn product(self, rhs: Self) -> Self {
        self * rhs
    }
    fn scale_log(self, log_s: f64) -> Self {
        RealEps::scale_log(&self, log_s)
    }
    fn divide(self, rhs: Self) -> Self {
        if self.power > rhs.power {
            Self::zero()
        } else {
            self / rhs
        }
    }
}

pub trait SpnLeaf: Clone + Debug + Sized {
    type Coeff: SpnCoeff;

    fn scope(&self) -> BTreeSet<ContVarName>;

    fn intern(inner: super::node::SpnInner<Self>) -> Rc<super::node::SpnInner<Self>> {
        Rc::new(inner)
    }
}
