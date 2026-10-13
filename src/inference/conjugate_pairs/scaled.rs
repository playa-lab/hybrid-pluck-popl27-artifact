use ordered_float::NotNan;

pub type ContVarName = u64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct Scaled<C> {
    pub constraint: C,
    pub scale: NotNan<f64>,
}

impl<C> Scaled<C> {
    pub fn new(constraint: C, scale: NotNan<f64>) -> Self {
        Scaled { constraint, scale }
    }

    pub fn unit(constraint: C) -> Self {
        Scaled {
            constraint,
            scale: NotNan::new(1.0).unwrap(),
        }
    }

    pub fn map<D>(self, f: impl FnOnce(C) -> D) -> Scaled<D> {
        Scaled {
            constraint: f(self.constraint),
            scale: self.scale,
        }
    }
}
