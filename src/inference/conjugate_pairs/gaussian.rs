use super::suffstat::{ContVarName, Prior, SuffStat, PRIOR_QUANTIZATION_LEVEL};
use crate::utils::{epsilon::RealEps, math::quant};

use itertools::Itertools;
use nalgebra as na;
use ordered_float::NotNan;
use std::{
    collections::{BTreeMap, BTreeSet},
    fmt::{Display, Formatter, Result},
    hash::Hash,
};

const RANK_REL_TOL: f64 = 1e-9;
const RANGE_TOL: f64 = 1e-8;
const EXCLUSION_VAR_TOL: f64 = 1e-10;
const EXCLUSION_VAL_TOL: f64 = 1e-8;
const SAMPLE_EQ_REL_TOL: f64 = 1e-6;

const AXIS_REL_TOL: f64 = 1e-9;
const VOL_REL_TOL: f64 = 1e-9;
const VERT_FEAS_TOL: f64 = 1e-8;
const DET_TOL: f64 = 1e-10;

fn sample_eq_rel_tol() -> f64 {
    static TOL: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
    *TOL.get_or_init(|| {
        std::env::var("PLUCK_EQ_TOL")
            .ok()
            .and_then(|s| s.parse::<f64>().ok())
            .filter(|t| t.is_finite() && *t > 0.0)
            .unwrap_or(SAMPLE_EQ_REL_TOL)
    })
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GaussianAffineExpr {
    pub constant: NotNan<f64>,
    pub coefficients: BTreeMap<ContVarName, NotNan<f64>>,
}

impl GaussianAffineExpr {
    pub fn scope(&self) -> impl Iterator<Item = &ContVarName> + '_ {
        self.coefficients.keys()
    }

    pub fn new(coefficients: &BTreeMap<ContVarName, NotNan<f64>>, constant: &f64) -> Self {
        GaussianAffineExpr {
            coefficients: coefficients.clone(),
            constant: NotNan::new(*constant).unwrap(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AffineConstraint {
    pub coefficients: BTreeMap<ContVarName, NotNan<f64>>,
    pub value: NotNan<f64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GaussianObs {
    pub(crate) equalities: Vec<AffineConstraint>,
    pub(crate) excluded: Vec<AffineConstraint>,
    pub(crate) block_scope: BTreeSet<ContVarName>,
}

impl GaussianObs {
    pub fn new(
        equalities: Vec<AffineConstraint>,
        excluded: Vec<AffineConstraint>,
        block_scope: BTreeSet<ContVarName>,
    ) -> GaussianObs {
        GaussianObs {
            equalities,
            excluded,
            block_scope,
        }
        .checked()
    }

    fn checked(self) -> GaussianObs {
        debug_assert!(
            !self.equalities.is_empty() || !self.excluded.is_empty(),
            "GaussianObs: empty observations are not a valid leaf"
        );
        debug_assert!(
            !self.block_scope.is_empty(),
            "GaussianObs: block_scope is empty; it must cover every \
             constraint variable (it is the leaf's scope)"
        );
        debug_assert!(
            self.equalities
                .iter()
                .chain(self.excluded.iter())
                .flat_map(|c| c.coefficients.keys())
                .all(|v| self.block_scope.contains(v)),
            "GaussianObs: block_scope must cover every constraint variable \
             (it is the leaf's scope)"
        );
        self
    }

    pub fn from_observation(
        coefficients: &BTreeMap<ContVarName, f64>,
        residual: f64,
        block_scope: BTreeSet<ContVarName>,
    ) -> GaussianObs {
        GaussianObs::new(
            vec![AffineConstraint {
                coefficients: coefficients
                    .iter()
                    .map(|(k, v)| (*k, NotNan::new(*v).unwrap()))
                    .collect(),
                value: NotNan::new(residual).unwrap(),
            }],
            Vec::new(),
            block_scope,
        )
    }

    pub fn from_exclusion(
        coefficients: BTreeMap<ContVarName, f64>,
        value: f64,
        block_scope: BTreeSet<ContVarName>,
    ) -> GaussianObs {
        GaussianObs::new(
            Vec::new(),
            vec![AffineConstraint {
                coefficients: coefficients
                    .into_iter()
                    .map(|(k, v)| (k, NotNan::new(v).unwrap()))
                    .collect(),
                value: NotNan::new(value).unwrap(),
            }],
            block_scope,
        )
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GaussianPrior<N = ContVarName> {
    pub var_order: Vec<N>,
    pub mean: na::DVector<f64>,
    pub cov: na::DMatrix<f64>,
}

impl<N: Display + Clone> Display for GaussianPrior<N> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        if self.var_order.is_empty() {
            Result::Ok(())
        } else if self.var_order.len() == 1 {
            let std = self.cov[(0, 0)].sqrt();
            write!(
                f,
                "{} ~ N({:.p$}, {:.p$})",
                self.var_order[0],
                self.mean[0],
                std,
                p = crate::utils::display_precision::param_precision(),
            )
        } else {
            let names = self.var_order.iter().cloned().join(", ");
            let p = crate::utils::display_precision::param_precision();
            let means: Vec<String> = self
                .mean
                .iter()
                .map(|m| format!("{:.p$}", m, p = p))
                .collect();
            writeln!(f, "{} ~ Joint Gaussian:", names)?;
            writeln!(f, "  mean = [{}]", means.join(", "))?;
            for j in 0..self.cov.nrows() {
                let row: Vec<String> = (0..self.cov.ncols())
                    .map(|i| format!("{:.p$}", self.cov[(j, i)], p = p))
                    .collect();
                let prefix = if j == 0 { "  cov  = " } else { "         " };
                writeln!(f, "{}[{}]", prefix, row.join(", "))?;
            }
            Result::Ok(())
        }
    }
}

impl GaussianPrior<ContVarName> {
    pub fn from_single(name: ContVarName, mean: f64, std: f64) -> Self {
        GaussianPrior {
            var_order: vec![name],
            mean: na::dvector![mean],
            cov: na::dmatrix![std.powf(2.0);],
        }
    }

    pub fn from_multivariate(
        names: Vec<ContVarName>,
        mean: na::DVector<f64>,
        cov: na::DMatrix<f64>,
    ) -> Self {
        assert_eq!(
            names.len(),
            mean.len(),
            "GaussianPrior::from_multivariate: names ({}) and mean ({}) length mismatch",
            names.len(),
            mean.len()
        );
        assert_eq!(
            cov.nrows(),
            names.len(),
            "GaussianPrior::from_multivariate: cov has {} rows, expected {}",
            cov.nrows(),
            names.len()
        );
        assert_eq!(
            cov.ncols(),
            names.len(),
            "GaussianPrior::from_multivariate: cov has {} cols, expected {}",
            cov.ncols(),
            names.len()
        );
        GaussianPrior {
            var_order: names,
            mean,
            cov,
        }
    }

    fn coeff_vec(&self, coefficients: &BTreeMap<ContVarName, f64>) -> na::DVector<f64> {
        let mut c = na::DVector::<f64>::zeros(self.var_order.len());
        for (i, name) in self.var_order.iter().enumerate() {
            if let Some(&v) = coefficients.get(name) {
                c[i] = v;
            }
        }
        c
    }

    pub fn affine_moments(
        &self,
        coefficients: &BTreeMap<ContVarName, f64>,
        constant: f64,
    ) -> (f64, f64) {
        let c = self.coeff_vec(coefficients);
        let mean = c.dot(&self.mean) + constant;
        let sigma_c = &self.cov * &c;
        let variance = c.dot(&sigma_c);
        (mean, variance)
    }

    pub fn affine_covariance(
        &self,
        coefficients_i: &BTreeMap<ContVarName, f64>,
        coefficients_j: &BTreeMap<ContVarName, f64>,
    ) -> f64 {
        let ci = self.coeff_vec(coefficients_i);
        let cj = self.coeff_vec(coefficients_j);
        let sigma_cj = &self.cov * &cj;
        ci.dot(&sigma_cj)
    }

    pub fn slice(&self, names: &BTreeSet<ContVarName>) -> GaussianPrior {
        let (indices, new_var_order): (Vec<usize>, Vec<ContVarName>) = self
            .var_order
            .iter()
            .enumerate()
            .filter_map(|(i, v)| {
                if names.contains(v) {
                    Some((i, *v))
                } else {
                    None
                }
            })
            .unzip();
        let new_mean = self.mean.select_rows(&indices);
        let new_cov = self.cov.select_rows(&indices).select_columns(&indices);
        GaussianPrior {
            var_order: new_var_order,
            mean: new_mean,
            cov: new_cov,
        }
    }
}

impl<N: Hash> Hash for GaussianPrior<N> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        for n in &self.var_order {
            n.hash(state);
        }
        for m in self.mean.iter() {
            quant(*m, PRIOR_QUANTIZATION_LEVEL).hash(state);
        }
        for v in self.cov.iter() {
            quant(*v, PRIOR_QUANTIZATION_LEVEL).hash(state);
        }
    }
}

impl<N: PartialEq> PartialEq for GaussianPrior<N> {
    fn eq(&self, other: &Self) -> bool {
        if self.var_order != other.var_order || self.mean.nrows() != other.mean.nrows() {
            return false;
        }
        for (m1, m2) in self.mean.iter().zip(other.mean.iter()) {
            if quant(*m1, PRIOR_QUANTIZATION_LEVEL) != quant(*m2, PRIOR_QUANTIZATION_LEVEL) {
                return false;
            }
        }
        if self.cov.shape() != other.cov.shape() {
            return false;
        }
        for (v1, v2) in self.cov.iter().zip(other.cov.iter()) {
            if quant(*v1, PRIOR_QUANTIZATION_LEVEL) != quant(*v2, PRIOR_QUANTIZATION_LEVEL) {
                return false;
            }
        }
        true
    }
}

impl<N: Eq> Eq for GaussianPrior<N> {}

impl Prior for GaussianPrior<ContVarName> {
    type Realization = Vec<NotNan<f64>>;

    fn scope(&self) -> impl Iterator<Item = &ContVarName> + '_ {
        self.var_order.iter()
    }

    fn sample<R: rand::Rng>(&self, rng: &mut R) -> Vec<NotNan<f64>> {
        let n = self.var_order.len();
        if n == 0 {
            return Vec::new();
        }
        let eig = na::SymmetricEigen::new(self.cov.clone());
        let mut sqrt_d = na::DVector::<f64>::zeros(n);
        for i in 0..n {
            let lambda = eig.eigenvalues[i];
            sqrt_d[i] = if lambda > 0.0 { lambda.sqrt() } else { 0.0 };
        }
        let z = na::DVector::from_fn(n, |_, _| crate::utils::sampling::sample_normal(rng));
        let scaled =
            na::DVector::from_iterator(n, sqrt_d.iter().zip(z.iter()).map(|(s, zi)| s * zi));
        let x = &self.mean + &eig.eigenvectors * scaled;
        x.iter()
            .map(|v| NotNan::new(*v).expect("Gaussian sample is NaN"))
            .collect()
    }

    fn push_into(&self, sampled: Vec<NotNan<f64>>, out: &mut super::AssignmentBuilder) {
        for (i, &name) in self.var_order.iter().enumerate() {
            out.push_gaussian(name, sampled[i]);
        }
    }
}

impl From<GaussianPrior> for super::PosteriorLeaf {
    fn from(p: GaussianPrior) -> Self {
        super::PosteriorLeaf::Gaussian(p)
    }
}


use crate::inference::conjugate_pairs::EvidenceLeaf;
use crate::inference::spn::node::Spn;

pub enum GaussianFlip {
    Obs {
        coefficients: BTreeMap<u64, f64>,
        constant: f64,
        observed: f64,
    },
}

impl GaussianFlip {
    pub fn weights(
        &self,
        registry: &super::PriorRegistry,
    ) -> (Spn<EvidenceLeaf>, Spn<EvidenceLeaf>) {
        let GaussianFlip::Obs {
            coefficients,
            constant,
            observed,
        } = self;
        let residual = *observed - *constant;
        let block_scope = registry.gaussian_block_closure(coefficients.keys());
        let g_high = GaussianObs::from_observation(coefficients, residual, block_scope.clone());
        let g_low = GaussianObs::from_exclusion(coefficients.clone(), residual, block_scope);
        (
            Spn::leaf(EvidenceLeaf::Gaussian(g_low)),
            Spn::leaf(EvidenceLeaf::Gaussian(g_high)),
        )
    }

    pub fn discriminator(&self) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let GaussianFlip::Obs {
            coefficients,
            constant,
            observed,
        } = self;
        let mut hasher = DefaultHasher::new();
        for (name, coeff) in coefficients {
            name.hash(&mut hasher);
            coeff.to_bits().hash(&mut hasher);
        }
        constant.to_bits().hash(&mut hasher);
        observed.to_bits().hash(&mut hasher);
        hasher.finish().wrapping_add(0xA1B2C3D4E5F60718)
    }

    pub fn is_pinned_observation(&self) -> bool {
        true
    }

    pub fn sample_probability(&self, _a: &super::Assignment) -> Option<f64> {
        None
    }
}

fn spd_inverse_logdet(mat: &na::DMatrix<f64>) -> (na::DMatrix<f64>, f64) {
    if let Some(ch) = na::Cholesky::new(mat.clone()) {
        let log_det = 2.0 * ch.l().diagonal().iter().map(|d| d.ln()).sum::<f64>();
        (ch.inverse(), log_det)
    } else {
        let eig = na::SymmetricEigen::new(mat.clone());
        let max_eig = eig
            .eigenvalues
            .iter()
            .fold(0.0_f64, |acc, &v| acc.max(v.abs()));
        let tol = 1e-12 * max_eig.max(1.0);
        let mut d_plus = na::DMatrix::<f64>::zeros(mat.nrows(), mat.ncols());
        let mut log_det = 0.0;
        for i in 0..mat.nrows() {
            let lambda = eig.eigenvalues[i];
            if lambda.abs() > tol {
                d_plus[(i, i)] = 1.0 / lambda;
                log_det += lambda.abs().ln();
            }
        }
        (
            &eig.eigenvectors * d_plus * eig.eigenvectors.transpose(),
            log_det,
        )
    }
}

fn condition_and_project(
    prior: &GaussianPrior,
    obs: &GaussianObs,
    kept_order: &[ContVarName],
) -> (Option<GaussianPrior>, RealEps) {
    debug_assert!(
        obs.scope().all(|v| prior.var_order.contains(v)),
        "GaussianObs references a variable not in the paired GaussianPrior"
    );
    debug_assert!(
        kept_order.iter().all(|v| prior.var_order.contains(v)),
        "kept_order has a variable not in prior.var_order"
    );

    let idx_map: BTreeMap<ContVarName, usize> = prior
        .var_order
        .iter()
        .enumerate()
        .map(|(i, v)| (*v, i))
        .collect();
    let mu = &prior.mean;
    let sigma = &prior.cov;
    let n = prior.var_order.len();
    let m = obs.equalities.len();
    let idx_kept: Vec<usize> = kept_order.iter().map(|v| idx_map[v]).collect();

    let mu_kept = mu.select_rows(&idx_kept);
    let sigma_kept_rows = sigma.select_rows(&idx_kept);
    let sigma_kk = sigma_kept_rows.select_columns(&idx_kept);

    let mut a = na::DMatrix::<f64>::zeros(m, n);
    let mut b = na::DVector::<f64>::zeros(m);
    for (
        i,
        AffineConstraint {
            coefficients,
            value,
        },
    ) in obs.equalities.iter().enumerate()
    {
        for (name, coef) in coefficients.iter() {
            a[(i, idx_map[name])] = coef.into_inner();
        }
        b[i] = value.into_inner();
    }
    let r = &b - &a * mu;
    let sigma_at = sigma * a.transpose();
    let s_full = &a * &sigma_at;

    let mut chosen: Vec<usize> = Vec::new();
    for i in 0..m {
        let s_ii = s_full[(i, i)];
        let proj_var = if chosen.is_empty() {
            s_ii
        } else {
            let s_t = s_full.select_rows(&chosen).select_columns(&chosen);
            let w =
                na::DVector::from_iterator(chosen.len(), chosen.iter().map(|&j| s_full[(j, i)]));
            let (s_t_inv, _) = spd_inverse_logdet(&s_t);
            s_ii - w.dot(&(&s_t_inv * &w))
        };
        if proj_var > RANK_REL_TOL * s_ii {
            chosen.push(i);
        }
    }
    let k = chosen.len();
    let s_t = s_full.select_rows(&chosen).select_columns(&chosen);
    let (s_t_inv, log_det_s_t) = spd_inverse_logdet(&s_t);
    let r_t = r.select_rows(&chosen);

    let mut k_rows: Vec<na::DVector<f64>> = Vec::with_capacity(m);
    for pos in 0..k {
        let mut e = na::DVector::<f64>::zeros(k);
        e[pos] = 1.0;
        k_rows.push(e);
    }
    for i in 0..m {
        if chosen.contains(&i) {
            continue;
        }
        let w = na::DVector::from_iterator(chosen.len(), chosen.iter().map(|&j| s_full[(j, i)]));
        let lambda = &s_t_inv * &w;
        let entailed = lambda.dot(&r_t);
        if (r[i] - entailed).abs() > RANGE_TOL * (1.0 + r[i].abs().max(entailed.abs())) {
            return (None, RealEps::zero());
        }
        if lambda.amax() > 1e-12 {
            k_rows.push(lambda);
        }
    }

    let mut cuts: Vec<na::DVector<f64>> = Vec::new();
    for ex in &obs.excluded {
        let mut c = na::DVector::<f64>::zeros(n);
        for (name, coef) in ex.coefficients.iter() {
            c[idx_map[name]] = coef.into_inner();
        }
        let v = sigma * &c;
        let (var, gamma, m_val) = if k == 0 {
            (c.dot(&v), na::DVector::<f64>::zeros(0), c.dot(mu))
        } else {
            let av = &a * &v;
            let w = av.select_rows(&chosen);
            let gamma = &s_t_inv * &w;
            let var = c.dot(&v) - w.dot(&gamma);
            let m_val = c.dot(mu) + gamma.dot(&r_t);
            (var, gamma, m_val)
        };
        if var > EXCLUSION_VAR_TOL {
            continue;
        }
        if (m_val - ex.value.into_inner()).abs() >= EXCLUSION_VAL_TOL {
            continue;
        }
        if k == 0 || gamma.amax() <= 1e-12 {
            return (None, RealEps::zero());
        }
        cuts.push(gamma);
    }

    let vol = cross_section_volume(&k_rows, &cuts, k);
    if vol <= 0.0 {
        return (None, RealEps::zero());
    }

    let z = if k == 0 {
        RealEps::from_log(0.0, 0)
    } else {
        let quad = r_t.dot(&(&s_t_inv * &r_t));
        let log_f = -0.5 * (k as f64 * (2.0 * std::f64::consts::PI).ln() + log_det_s_t + quad);
        RealEps::from_log(log_f + vol.ln(), k as u32)
    };

    if kept_order.is_empty() {
        return (None, z);
    }

    let (mu_post_k, cov_post_k) = if k == 0 {
        (mu_kept, sigma_kk)
    } else {
        let t = a.select_rows(&chosen);
        let sigma_t_kept = &sigma_kept_rows * t.transpose();
        let k_mat = &sigma_t_kept * &s_t_inv;
        let mu_post_k = &mu_kept + &k_mat * &r_t;
        let cov_unsym = &sigma_kk - &k_mat * &sigma_t_kept.transpose();
        let cov_post_k = 0.5 * (&cov_unsym + cov_unsym.transpose());
        (mu_post_k, cov_post_k)
    };

    (
        Some(GaussianPrior {
            var_order: kept_order.to_vec(),
            mean: mu_post_k,
            cov: cov_post_k,
        }),
        z,
    )
}


fn cross_section_volume(k_rows: &[na::DVector<f64>], cuts: &[na::DVector<f64>], k: usize) -> f64 {
    if k == 0 {
        return if cuts.is_empty() { 1.0 } else { 0.0 };
    }

    let axis = |v: &na::DVector<f64>| -> Option<usize> {
        let max = v.amax();
        let mut nz = None;
        for i in 0..k {
            if v[i].abs() > AXIS_REL_TOL * max {
                if nz.is_some() {
                    return None;
                }
                nz = Some(i);
            }
        }
        nz
    };

    let axes_k: Option<Vec<usize>> = k_rows.iter().map(axis).collect();
    let axes_c: Option<Vec<usize>> = cuts.iter().map(axis).collect();
    if let (Some(axes_k), Some(axes_c)) = (axes_k, axes_c) {
        let mut r = vec![f64::INFINITY; k];
        let mut u = vec![0.0f64; k];
        for (v, &ax) in k_rows.iter().zip(&axes_k) {
            r[ax] = r[ax].min(0.5 / v[ax].abs());
        }
        for (v, &ax) in cuts.iter().zip(&axes_c) {
            u[ax] = u[ax].max(0.5 / v[ax].abs());
        }
        let mut vol = 1.0;
        for i in 0..k {
            debug_assert!(r[i].is_finite(), "basis rows must cover every direction");
            let len = r[i] - u[i];
            if len <= VOL_REL_TOL * r[i] {
                return 0.0;
            }
            vol *= 2.0 * len;
        }
        return vol;
    }

    let vol_k = hpoly_volume(&k_rows.iter().collect::<Vec<_>>(), k);
    if vol_k <= 0.0 {
        return 0.0;
    }
    assert!(
        cuts.len() <= 20,
        "Gaussian leaf: inclusion–exclusion over {} mixed-direction cuts is intractable",
        cuts.len()
    );
    let mut total = 0.0;
    for mask in 0..(1usize << cuts.len()) {
        let mut rows: Vec<&na::DVector<f64>> = k_rows.iter().collect();
        for (j, cut) in cuts.iter().enumerate() {
            if mask & (1 << j) != 0 {
                rows.push(cut);
            }
        }
        let v = hpoly_volume(&rows, k);
        if mask.count_ones() % 2 == 0 {
            total += v;
        } else {
            total -= v;
        }
    }
    if total <= VOL_REL_TOL * vol_k {
        0.0
    } else {
        total
    }
}

fn hpoly_volume(rows: &[&na::DVector<f64>], k: usize) -> f64 {
    if k == 1 {
        let r = rows
            .iter()
            .filter(|v| v[0].abs() > 0.0)
            .map(|v| 0.5 / v[0].abs())
            .fold(f64::INFINITY, f64::min);
        return 2.0 * r;
    }

    let round9 = |x: f64| (x * 1e9).round() / 1e9;
    let mut normals: Vec<na::DVector<f64>> = Vec::new();
    for v in rows {
        for cand in [(*v).clone(), -(*v).clone()] {
            if !normals
                .iter()
                .any(|n| (0..k).all(|i| round9(n[i]) == round9(cand[i])))
            {
                normals.push(cand);
            }
        }
    }

    let mut verts: Vec<na::DVector<f64>> = Vec::new();
    let mut seen: BTreeSet<Vec<NotNan<f64>>> = BTreeSet::new();
    let mut combo = vec![0usize; k];
    enumerate_combinations(normals.len(), k, &mut combo, 0, 0, &mut |combo| {
        let mat = na::DMatrix::<f64>::from_fn(k, k, |i, j| normals[combo[i]][j]);
        let scale = mat.amax().max(1.0);
        let lu = mat.clone().lu();
        let det = lu.determinant();
        if det.abs() < DET_TOL * scale.powi(k as i32) {
            return;
        }
        let rhs = na::DVector::from_element(k, 0.5);
        if let Some(x) = lu.solve(&rhs) {
            if normals.iter().all(|nrm| nrm.dot(&x) <= 0.5 + VERT_FEAS_TOL) {
                let key: Vec<NotNan<f64>> = (0..k)
                    .map(|i| NotNan::new(round9(x[i])).expect("vertex is NaN"))
                    .collect();
                if seen.insert(key) {
                    verts.push(x);
                }
            }
        }
    });
    if verts.len() < k + 1 {
        return 0.0;
    }

    match k {
        2 => {
            let mut vs: Vec<(f64, f64)> = verts.iter().map(|v| (v[0], v[1])).collect();
            vs.sort_by(|p, q| {
                p.1.atan2(p.0)
                    .partial_cmp(&q.1.atan2(q.0))
                    .expect("vertex angle is NaN")
            });
            let mut area2 = 0.0;
            for i in 0..vs.len() {
                let (x0, y0) = vs[i];
                let (x1, y1) = vs[(i + 1) % vs.len()];
                area2 += x0 * y1 - x1 * y0;
            }
            area2.abs() / 2.0
        }
        3 => {
            let cross3 = |p: &na::DVector<f64>, q: &na::DVector<f64>| {
                na::DVector::from_vec(vec![
                    p[1] * q[2] - p[2] * q[1],
                    p[2] * q[0] - p[0] * q[2],
                    p[0] * q[1] - p[1] * q[0],
                ])
            };
            let mut vol = 0.0;
            for nrm in &normals {
                let face: Vec<&na::DVector<f64>> = verts
                    .iter()
                    .filter(|v| nrm.dot(v) >= 0.5 - VERT_FEAS_TOL)
                    .collect();
                if face.len() < 3 {
                    continue;
                }
                let seed_axis = (0..3)
                    .min_by(|&i, &j| {
                        nrm[i]
                            .abs()
                            .partial_cmp(&nrm[j].abs())
                            .expect("normal is NaN")
                    })
                    .unwrap();
                let mut seed = na::DVector::<f64>::zeros(3);
                seed[seed_axis] = 1.0;
                let u1 = cross3(nrm, &seed).normalize();
                let u2 = cross3(nrm, &u1).normalize();
                let centroid_2d = face
                    .iter()
                    .fold((0.0, 0.0), |acc, v| (acc.0 + u1.dot(v), acc.1 + u2.dot(v)));
                let cnt = face.len() as f64;
                let (cx, cy) = (centroid_2d.0 / cnt, centroid_2d.1 / cnt);
                let mut ordered: Vec<(f64, &na::DVector<f64>)> = face
                    .iter()
                    .map(|v| ((u2.dot(v) - cy).atan2(u1.dot(v) - cx), *v))
                    .collect();
                ordered.sort_by(|p, q| p.0.partial_cmp(&q.0).expect("facet angle is NaN"));
                let w0 = ordered[0].1;
                for i in 1..ordered.len() - 1 {
                    let wi = ordered[i].1;
                    let wj = ordered[i + 1].1;
                    vol += w0.dot(&cross3(wi, wj)).abs() / 6.0;
                }
            }
            vol
        }
        _ => panic!(
            "Gaussian leaf: exact window volume for {} independent directions with \
             mixed-direction redundant tests or cuts is unsupported (k ≤ 3 only)",
            k
        ),
    }
}

fn enumerate_combinations(
    n: usize,
    k: usize,
    combo: &mut Vec<usize>,
    depth: usize,
    start: usize,
    f: &mut impl FnMut(&[usize]),
) {
    if depth == k {
        f(combo);
        return;
    }
    for i in start..n {
        combo[depth] = i;
        enumerate_combinations(n, k, combo, depth + 1, i + 1, f);
    }
}

impl SuffStat for GaussianObs {
    type ConjugatePrior = GaussianPrior;

    fn merge(&self, other: &GaussianObs) -> (GaussianObs, f64) {
        debug_assert!(
            {
                let s: BTreeSet<ContVarName> = self.scope().copied().collect();
                other.scope().any(|v| s.contains(v))
            },
            "GaussianObs::merge requires overlapping scopes"
        );
        let merged_equalities: Vec<AffineConstraint> = self
            .equalities
            .iter()
            .merge(other.equalities.iter())
            .dedup()
            .cloned()
            .collect();
        let merged_excluded: Vec<AffineConstraint> = self
            .excluded
            .iter()
            .merge(other.excluded.iter())
            .dedup()
            .cloned()
            .collect();
        let merged_scope = self
            .block_scope
            .union(&other.block_scope)
            .copied()
            .collect();
        (
            GaussianObs::new(merged_equalities, merged_excluded, merged_scope),
            0.0,
        )
    }

    fn scope(&self) -> impl Iterator<Item = &ContVarName> + '_ {
        self.block_scope.iter()
    }

    fn log_likelihood(&self, value: &Vec<NotNan<f64>>) -> f64 {
        let scope: BTreeSet<ContVarName> = self.scope().copied().collect();
        debug_assert_eq!(
            value.len(),
            scope.len(),
            "GaussianObs::log_likelihood: realisation has wrong length"
        );
        let idx_map: BTreeMap<ContVarName, usize> =
            scope.iter().enumerate().map(|(i, v)| (*v, i)).collect();

        let eval = |c: &AffineConstraint| -> (f64, f64) {
            let mut a_t_x = 0.0;
            let mut scale = c.value.into_inner().abs();
            for (name, coef) in c.coefficients.iter() {
                let i = idx_map[name];
                let term = coef.into_inner() * value[i].into_inner();
                a_t_x += term;
                scale = scale.max(term.abs());
            }
            (
                a_t_x - c.value.into_inner(),
                sample_eq_rel_tol() * scale.max(1.0),
            )
        };

        for c in &self.equalities {
            let (residual, tol) = eval(c);
            if residual.abs() > tol {
                return f64::NEG_INFINITY;
            }
        }
        for c in &self.excluded {
            let (residual, tol) = eval(c);
            if residual.abs() <= tol {
                return f64::NEG_INFINITY;
            }
        }
        0.0
    }

    fn posterior(
        &self,
        prior: &GaussianPrior,
        query_vars: &BTreeSet<ContVarName>,
    ) -> (Option<Self::ConjugatePrior>, RealEps) {
        let kept_order: Vec<ContVarName> = prior
            .var_order
            .iter()
            .filter(|v| query_vars.contains(v))
            .copied()
            .collect();
        condition_and_project(prior, self, &kept_order)
    }

    fn read_realization(&self, a: &super::Assignment) -> Vec<NotNan<f64>> {
        a.gaussian_vector(&self.scope().copied().collect())
    }

    fn lookup_prior_in(&self, reg: &super::PriorRegistry) -> GaussianPrior {
        reg.gaussian_prior_for(&self.scope().copied().collect())
    }

    fn posterior_in(
        &self,
        reg: &super::PriorRegistry,
        query_vars: &BTreeSet<ContVarName>,
    ) -> (Option<Self::ConjugatePrior>, RealEps) {
        let obs_scope: BTreeSet<ContVarName> = self.scope().copied().collect();
        let mut joint: BTreeSet<ContVarName> = obs_scope.clone();
        for v in query_vars {
            let shares_block = reg.gaussian.iter().any(|b| {
                let names: BTreeSet<ContVarName> = b.var_order.iter().copied().collect();
                names.contains(v) && names.intersection(&obs_scope).next().is_some()
            });
            if shares_block {
                joint.insert(*v);
            }
        }
        let prior = reg.gaussian_prior_for(&joint);
        self.posterior(&prior, query_vars)
    }
}

#[cfg(test)]
mod tests {
    use super::*;


    fn mkmap(pairs: &[(u64, f64)]) -> BTreeMap<ContVarName, f64> {
        pairs.iter().map(|&(k, v)| (k, v)).collect()
    }

    fn nn(x: f64) -> NotNan<f64> {
        NotNan::new(x).unwrap()
    }

    fn mkprior(var_order: Vec<u64>, mean: &[f64], cov: &[&[f64]]) -> GaussianPrior {
        let n = var_order.len();
        assert_eq!(mean.len(), n);
        assert_eq!(cov.len(), n);
        for row in cov.iter() {
            assert_eq!(row.len(), n);
        }
        let m = na::DVector::from_iterator(n, mean.iter().copied());
        let cm =
            na::DMatrix::from_row_iterator(n, n, cov.iter().flat_map(|row| row.iter().copied()));
        GaussianPrior {
            var_order,
            mean: m,
            cov: cm,
        }
    }

    fn assert_prior_close(a: &GaussianPrior, b: &GaussianPrior, tol: f64) {
        assert_eq!(a.var_order, b.var_order, "var_order mismatch");
        assert_eq!(a.mean.len(), b.mean.len(), "mean dim mismatch");
        for i in 0..a.mean.len() {
            let d = (a.mean[i] - b.mean[i]).abs();
            assert!(
                d < tol,
                "mean[{}] differs: {} vs {} (|diff|={})",
                i,
                a.mean[i],
                b.mean[i],
                d
            );
        }
        assert_eq!(a.cov.shape(), b.cov.shape(), "cov shape mismatch");
        let (r, c) = a.cov.shape();
        for i in 0..r {
            for j in 0..c {
                let d = (a.cov[(i, j)] - b.cov[(i, j)]).abs();
                assert!(
                    d < tol,
                    "cov[{},{}] differs: {} vs {} (|diff|={})",
                    i,
                    j,
                    a.cov[(i, j)],
                    b.cov[(i, j)],
                    d
                );
            }
        }
    }

    fn full_query(prior: &GaussianPrior) -> BTreeSet<ContVarName> {
        prior.var_order.iter().copied().collect()
    }

    fn mk_constraint(coefs: &[(u64, f64)], value: f64) -> AffineConstraint {
        AffineConstraint {
            coefficients: coefs
                .iter()
                .map(|&(k, v)| (k, NotNan::new(v).unwrap()))
                .collect(),
            value: NotNan::new(value).unwrap(),
        }
    }

    fn set(names: &[u64]) -> BTreeSet<ContVarName> {
        names.iter().copied().collect()
    }

    fn obs(pairs: &[(u64, f64)], v: f64) -> GaussianObs {
        let closure = pairs.iter().map(|&(k, _)| k).collect();
        GaussianObs::from_observation(&mkmap(pairs), v, closure)
    }

    fn excl(pairs: &[(u64, f64)], v: f64) -> GaussianObs {
        let closure = pairs.iter().map(|&(k, _)| k).collect();
        GaussianObs::from_exclusion(mkmap(pairs), v, closure)
    }

    #[test]
    fn slice_preserves_order() {
        let prior = mkprior(
            vec![1, 2, 3],
            &[0.5, 1.5, 2.5],
            &[&[1.0, 0.0, 0.0], &[0.0, 2.0, 0.0], &[0.0, 0.0, 3.0]],
        );
        let sliced = prior.slice(&[2, 3].iter().copied().collect());
        let expected = mkprior(vec![2, 3], &[1.5, 2.5], &[&[2.0, 0.0], &[0.0, 3.0]]);
        assert_prior_close(&sliced, &expected, 1e-12);
    }

    #[test]
    fn slice_skips_missing_names() {
        let prior = mkprior(
            vec![1, 2, 3],
            &[0.0, 0.0, 0.0],
            &[&[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0], &[0.0, 0.0, 1.0]],
        );
        let sliced = prior.slice(&[4u64, 5].iter().copied().collect());
        assert!(sliced.var_order.is_empty());
        assert_eq!(sliced.mean.len(), 0);
        assert_eq!(sliced.cov.shape(), (0, 0));
    }

    #[test]
    fn slice_reorders_to_prior_order() {
        let prior = mkprior(
            vec![1, 2, 3],
            &[10.0, 20.0, 30.0],
            &[&[1.0, 0.0, 0.0], &[0.0, 2.0, 0.0], &[0.0, 0.0, 3.0]],
        );
        let sliced = prior.slice(&[3u64, 1].iter().copied().collect());
        assert_eq!(sliced.var_order, vec![1, 3]);
        let expected = mkprior(vec![1, 3], &[10.0, 30.0], &[&[1.0, 0.0], &[0.0, 3.0]]);
        assert_prior_close(&sliced, &expected, 1e-12);
    }


    #[test]
    fn from_observation_single() {
        let obs = obs(&[(1, 2.0), (2, -1.0)], 5.0);
        assert_eq!(obs.equalities.len(), 1);
        assert_eq!(obs.equalities[0].value, nn(5.0));
        assert!(obs.excluded.is_empty());
        assert_eq!(obs.equalities[0].coefficients.get(&1), Some(&nn(2.0)));
        assert_eq!(obs.equalities[0].coefficients.get(&2), Some(&nn(-1.0)));
    }

    #[test]
    fn from_exclusion_single() {
        let obs = excl(&[(1, 3.0)], 7.0);
        assert!(obs.equalities.is_empty());
        assert_eq!(obs.excluded.len(), 1);
        assert_eq!(obs.excluded[0].value.into_inner(), 7.0);
        assert_eq!(
            obs.excluded[0].coefficients.get(&1).unwrap().into_inner(),
            3.0
        );
    }

    #[test]
    fn merge_dedups_exclusions() {
        let a = excl(&[(1, 1.0)], 0.0);
        let b = excl(&[(1, 1.0)], 0.0);
        let merged = a.merge(&b).0;
        assert_eq!(merged.excluded.len(), 1);
    }

    #[test]
    fn merge_sorts_exclusions() {
        let a = excl(&[(1, 1.0)], 2.0);
        let b = excl(&[(1, 1.0)], 1.0);
        let merged = a.merge(&b).0;
        assert_eq!(merged.excluded.len(), 2);
        assert_eq!(merged.excluded[0].value.into_inner(), 1.0);
        assert_eq!(merged.excluded[1].value.into_inner(), 2.0);
    }


    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "block_scope must cover")]
    fn new_rejects_closure_missing_constraint_var() {
        let _ = GaussianObs::new(
            vec![
                mk_constraint(&[(1, 1.0)], 1.0),
                mk_constraint(&[(2, 1.0)], 2.0),
            ],
            vec![],
            set(&[1]),
        );
    }

    #[test]
    fn new_accepts_two_constraint_block() {
        let obs = GaussianObs::new(
            vec![
                mk_constraint(&[(1, 1.0), (2, 1.0)], 3.0),
                mk_constraint(&[(2, 1.0), (3, 1.0)], 5.0),
            ],
            vec![],
            set(&[1, 2, 3]),
        );
        assert_eq!(obs.equalities.len(), 2);
    }

    #[test]
    fn from_observation_multi_var() {
        let _ = obs(&[(1, 1.0), (2, 1.0), (3, 1.0)], 6.0);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "empty observations")]
    fn new_rejects_empty_observations() {
        let _ = GaussianObs::new(vec![], vec![], set(&[1]));
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "overlapping scopes")]
    fn merge_disjoint_panics_in_debug() {
        let a = obs(&[(1, 1.0)], 1.0);
        let b = obs(&[(2, 1.0)], 2.0);
        let _ = a.merge(&b);
    }

    #[test]
    fn merge_overlap_unions_scopes() {
        let a = obs(&[(1, 1.0), (2, 1.0)], 3.0);
        let b = obs(&[(2, 1.0), (3, 1.0)], 5.0);
        let merged = a.merge(&b).0;
        assert_eq!(merged.equalities.len(), 2);
        assert_eq!(
            merged.scope().copied().collect::<BTreeSet<_>>(),
            set(&[1, 2, 3])
        );
    }


    #[test]
    fn posterior_1d_point_observation() {
        let prior = mkprior(vec![1], &[0.0], &[&[1.0]]);
        let obs = obs(&[(1, 1.0)], 3.0);
        let post = obs.posterior(&prior, &full_query(&prior)).0.unwrap();
        let expected = mkprior(vec![1], &[3.0], &[&[0.0]]);
        assert_prior_close(&post, &expected, 1e-10);
    }

    #[test]
    fn posterior_2d_sum_constraint() {
        let prior = mkprior(vec![1, 2], &[0.0, 0.0], &[&[1.0, 0.0], &[0.0, 1.0]]);
        let obs = obs(&[(1, 1.0), (2, 1.0)], 2.0);
        let post = obs.posterior(&prior, &full_query(&prior)).0.unwrap();
        let expected = mkprior(vec![1, 2], &[1.0, 1.0], &[&[0.5, -0.5], &[-0.5, 0.5]]);
        assert_prior_close(&post, &expected, 1e-10);
    }

    #[test]
    fn posterior_rank_deficient_redundant_constraint() {
        let prior = mkprior(vec![1, 2], &[0.0, 0.0], &[&[1.0, 0.0], &[0.0, 1.0]]);
        let single = obs(&[(1, 1.0), (2, 1.0)], 2.0);
        let doubled = single.clone().merge(&single).0;
        let post_single = single.posterior(&prior, &full_query(&prior)).0.unwrap();
        let post_doubled = doubled.posterior(&prior, &full_query(&prior)).0.unwrap();
        assert_prior_close(&post_single, &post_doubled, 1e-8);
    }

    #[test]
    fn posterior_rank_deficient_combined_constraints() {
        let prior = mkprior(vec![1, 2], &[0.0, 0.0], &[&[1.0, 0.0], &[0.0, 1.0]]);
        let obs1 = obs(&[(1, 1.0), (2, 1.0)], 2.0);
        let obs2 = obs(&[(1, 2.0), (2, 2.0)], 4.0);
        let merged = obs1.clone().merge(&obs2).0;
        let post_single = obs1.posterior(&prior, &full_query(&prior)).0.unwrap();
        let post_merged = merged.posterior(&prior, &full_query(&prior)).0.unwrap();
        assert_prior_close(&post_single, &post_merged, 1e-8);
    }


    #[test]
    fn loglik_1d_matches_normal_pdf() {
        let prior = mkprior(vec![1], &[0.0], &[&[1.0]]);
        let obs = obs(&[(1, 1.0)], 1.0);
        let ll = obs.posterior(&prior, &BTreeSet::new()).1;
        let expected = -0.5 * ((2.0 * std::f64::consts::PI).ln() + 1.0);
        assert!((ll.log_coeff - expected).abs() < 1e-12);
        assert_eq!(ll.power, 1);
    }

    #[test]
    fn loglik_rank_deficient_consistent() {
        let prior = mkprior(vec![1, 2], &[0.0, 0.0], &[&[1.0, 0.0], &[0.0, 1.0]]);
        let obs = GaussianObs::new(
            vec![
                mk_constraint(&[(1, 1.0), (2, 1.0)], 1.0),
                mk_constraint(&[(1, 2.0), (2, 2.0)], 2.0),
            ],
            vec![],
            set(&[1, 2]),
        );
        let ll = obs.posterior(&prior, &BTreeSet::new()).1;
        let expected = -0.5 * ((2.0 * std::f64::consts::PI).ln() + 8.0_f64.ln() + 0.5);
        assert_eq!(ll.power, 1);
        assert!(
            (ll.log_coeff - expected).abs() < 1e-8,
            "ll={} expected={}",
            ll.log_coeff,
            expected
        );
    }

    #[test]
    fn loglik_inconsistent_returns_neg_infinity() {
        let prior = mkprior(vec![1, 2], &[0.0, 0.0], &[&[1.0, 0.0], &[0.0, 1.0]]);
        let obs = GaussianObs::new(
            vec![
                mk_constraint(&[(1, 1.0), (2, 1.0)], 1.0),
                mk_constraint(&[(1, 1.0), (2, 1.0)], 2.0),
            ],
            vec![],
            set(&[1, 2]),
        );
        let ll = obs.posterior(&prior, &BTreeSet::new()).1;
        assert!(ll.is_zero(), "expected log-likelihood = -inf (linear 0)");
    }


    #[test]
    fn exclusion_satisfied_automatically() {
        let prior = mkprior(vec![1], &[0.0], &[&[1.0]]);
        let obs = excl(&[(1, 1.0)], 3.0);
        let post = obs.posterior(&prior, &full_query(&prior)).0.unwrap();
        assert_prior_close(&post, &prior, 1e-12);
    }

    #[test]
    fn exclusion_collides_with_equality() {
        let prior = mkprior(vec![1], &[0.0], &[&[1.0]]);
        let obs = GaussianObs::new(
            vec![mk_constraint(&[(1, 1.0)], 3.0)],
            vec![mk_constraint(&[(1, 1.0)], 3.0)],
            set(&[1]),
        );
        let (p, log_z) = obs.posterior(&prior, &full_query(&prior));
        assert_eq!(log_z.log_coeff, f64::NEG_INFINITY);
        assert!(p.is_none());
    }

    #[test]
    fn exclusion_orthogonal_to_constraint() {
        let prior = mkprior(vec![1, 2], &[0.0, 0.0], &[&[1.0, 0.0], &[0.0, 1.0]]);
        let obs = GaussianObs::new(
            vec![mk_constraint(&[(1, 1.0), (2, 1.0)], 0.0)],
            vec![mk_constraint(&[(1, 1.0), (2, -1.0)], 0.0)],
            set(&[1, 2]),
        );
        let post = obs.posterior(&prior, &full_query(&prior)).0.unwrap();
        let expected = mkprior(vec![1, 2], &[0.0, 0.0], &[&[0.5, -0.5], &[-0.5, 0.5]]);
        assert_prior_close(&post, &expected, 1e-10);
    }

    #[test]
    fn exclusion_in_row_space_of_constraint_is_a_cut() {
        let prior = mkprior(vec![1, 2], &[0.0, 0.0], &[&[1.0, 0.0], &[0.0, 1.0]]);
        let obs = GaussianObs::new(
            vec![mk_constraint(&[(1, 1.0), (2, 1.0)], 1.0)],
            vec![mk_constraint(&[(1, 2.0), (2, 2.0)], 2.0)],
            set(&[1, 2]),
        );
        let (p, z) = obs.posterior(&prior, &full_query(&prior));
        let expected =
            -0.5 * ((2.0 * std::f64::consts::PI).ln() + 2.0_f64.ln() + 0.5) + 0.5_f64.ln();
        assert_eq!(z.power, 1);
        assert!(
            (z.log_coeff - expected).abs() < 1e-10,
            "z={} expected={}",
            z.log_coeff,
            expected
        );
        let expected_post = mkprior(vec![1, 2], &[0.5, 0.5], &[&[0.5, -0.5], &[-0.5, 0.5]]);
        assert_prior_close(&p.unwrap(), &expected_post, 1e-10);
    }

    #[test]
    fn exclusion_wider_than_slab_is_zero() {
        let prior = mkprior(vec![1, 2], &[0.0, 0.0], &[&[1.0, 0.0], &[0.0, 1.0]]);
        let obs = GaussianObs::new(
            vec![mk_constraint(&[(1, 2.0), (2, 2.0)], 2.0)],
            vec![mk_constraint(&[(1, 1.0), (2, 1.0)], 1.0)],
            set(&[1, 2]),
        );
        let (p, log_z) = obs.posterior(&prior, &full_query(&prior));
        assert!(log_z.is_zero());
        assert!(p.is_none());
    }


    fn phi(x: f64) -> f64 {
        (-x * x / 2.0).exp() / (2.0 * std::f64::consts::PI).sqrt()
    }

    fn coeff_of(z: &RealEps) -> f64 {
        z.log_coeff.exp()
    }

    #[test]
    fn demo_annulus_not_zero() {
        let prior = mkprior(vec![1], &[0.0], &[&[1.0]]);
        let obs = GaussianObs::new(
            vec![mk_constraint(&[(1, 1.0)], 5.0)],
            vec![mk_constraint(&[(1, 2.0)], 10.0)],
            set(&[1]),
        );
        let (p, z) = obs.posterior(&prior, &full_query(&prior));
        assert_eq!(z.power, 1);
        assert!((coeff_of(&z) - phi(5.0) / 2.0).abs() < 1e-12 * phi(5.0));
        let post = p.unwrap();
        assert!((post.mean[0] - 5.0).abs() < 1e-10);
        assert!(post.cov[(0, 0)].abs() < 1e-10);
    }

    #[test]
    fn demo_disjunction_count_is_order_independent() {
        let prior = mkprior(vec![1], &[0.0], &[&[1.0]]);
        let slab = obs(&[(1, 1.0)], 5.0);
        let narrow = obs(&[(1, 2.0)], 10.0);
        let annulus = GaussianObs::new(
            vec![mk_constraint(&[(1, 1.0)], 5.0)],
            vec![mk_constraint(&[(1, 2.0)], 10.0)],
            set(&[1]),
        );
        let q = BTreeSet::new();
        let total_1 = slab.posterior(&prior, &q).1;
        let total_2 = narrow.posterior(&prior, &q).1 + annulus.posterior(&prior, &q).1;
        assert_eq!(total_1.power, total_2.power);
        assert!(
            (total_1.log_coeff - total_2.log_coeff).abs() < 1e-10,
            "order f<f': {}   order f'<f: {}",
            total_1.log_coeff,
            total_2.log_coeff
        );
        assert!((coeff_of(&total_1) - phi(5.0)).abs() < 1e-12 * phi(5.0));
    }

    #[test]
    fn demo_hexagon_mixed_direction_redundancy() {
        let prior = mkprior(vec![1, 2], &[0.0, 0.0], &[&[1.0, 0.0], &[0.0, 1.0]]);
        let obs = GaussianObs::new(
            vec![
                mk_constraint(&[(1, 1.0)], 0.0),
                mk_constraint(&[(2, 1.0)], 0.0),
                mk_constraint(&[(1, 1.0), (2, 1.0)], 0.0),
            ],
            vec![],
            set(&[1, 2]),
        );
        let (p, z) = obs.posterior(&prior, &full_query(&prior));
        assert_eq!(z.power, 2);
        let expected = phi(0.0) * phi(0.0) * 0.75;
        assert!(
            (coeff_of(&z) - expected).abs() < 1e-10 * expected,
            "coeff={} expected={}",
            coeff_of(&z),
            expected
        );
        let post = p.unwrap();
        assert!(post.mean.amax() < 1e-10);
        assert!(post.cov.amax() < 1e-10);
    }

    #[test]
    fn demo_residual_promoted_to_cut() {
        let prior = mkprior(vec![1, 2], &[0.0, 0.0], &[&[1.0, 0.0], &[0.0, 1.0]]);
        let before = GaussianObs::new(
            vec![mk_constraint(&[(1, 1.0)], 5.0)],
            vec![mk_constraint(&[(1, 1.0), (2, 1.0)], 9.0)],
            set(&[1, 2]),
        );
        let z_before = before.posterior(&prior, &BTreeSet::new()).1;
        assert_eq!(z_before.power, 1);
        assert!((coeff_of(&z_before) - phi(5.0)).abs() < 1e-12 * phi(5.0));

        let (after, _) = before.merge(&obs(&[(2, 1.0)], 4.0));
        let (p, z_after) = after.posterior(&prior, &full_query(&prior));
        assert_eq!(z_after.power, 2);
        let expected = phi(5.0) * phi(4.0) * 0.25;
        assert!(
            (coeff_of(&z_after) - expected).abs() < 1e-10 * expected,
            "coeff={} expected={}",
            coeff_of(&z_after),
            expected
        );
        let post = p.unwrap();
        assert!((post.mean[0] - 5.0).abs() < 1e-10);
        assert!((post.mean[1] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn demo_bare_co_weight_observes_nothing() {
        let prior = mkprior(vec![1], &[0.0], &[&[1.0]]);
        let obs = excl(&[(1, 1.0)], 5.0);
        let (p, z) = obs.posterior(&prior, &full_query(&prior));
        assert_eq!(z.power, 0);
        assert!(z.log_coeff.abs() < 1e-12);
        assert_prior_close(&p.unwrap(), &prior, 1e-12);
    }

    #[test]
    fn deterministic_consistent_row_contributes_nothing() {
        let prior = mkprior(vec![1], &[3.0], &[&[0.0]]);
        let hit = obs(&[(1, 1.0)], 3.0);
        let (p, z) = hit.posterior(&prior, &full_query(&prior));
        assert_eq!(z.power, 0);
        assert!(z.log_coeff.abs() < 1e-12);
        assert_prior_close(&p.unwrap(), &prior, 1e-12);

        let miss = obs(&[(1, 1.0)], 4.0);
        let (p, z) = miss.posterior(&prior, &full_query(&prior));
        assert!(z.is_zero());
        assert!(p.is_none());
    }


    #[test]
    fn posterior_empty_query_returns_none() {
        let prior = mkprior(vec![1, 2], &[0.0, 0.0], &[&[1.0, 0.0], &[0.0, 1.0]]);
        let obs = obs(&[(1, 1.0), (2, 1.0)], 2.0);
        let q: BTreeSet<ContVarName> = BTreeSet::new();
        assert!(obs.posterior(&prior, &q).0.is_none());
    }

    #[test]
    fn posterior_disjoint_query_returns_none() {
        let prior = mkprior(vec![1, 2], &[0.0, 0.0], &[&[1.0, 0.0], &[0.0, 1.0]]);
        let obs = obs(&[(1, 1.0), (2, 1.0)], 2.0);
        let q: BTreeSet<ContVarName> = [99u64].iter().copied().collect();
        assert!(obs.posterior(&prior, &q).0.is_none());
    }

    #[test]
    fn posterior_partial_query_equals_full_then_slice() {
        let prior = mkprior(
            vec![1, 2, 3],
            &[0.0, 0.0, 0.0],
            &[&[1.0, 0.5, 0.0], &[0.5, 1.0, 0.3], &[0.0, 0.3, 1.0]],
        );
        let obs = obs(&[(1, 1.0), (2, 1.0), (3, 1.0)], 3.0);
        let q12: BTreeSet<ContVarName> = [1u64, 2].iter().copied().collect();

        let partial = obs.posterior(&prior, &q12).0.unwrap();
        let full = obs.posterior(&prior, &full_query(&prior)).0.unwrap();
        let sliced = full.slice(&q12);
        assert_prior_close(&partial, &sliced, 1e-10);
    }

    #[test]
    fn posterior_partial_query_two_var_block() {
        let prior = mkprior(vec![1, 2], &[0.0, 0.0], &[&[1.0, 0.0], &[0.0, 1.0]]);
        let obs = obs(&[(1, 1.0), (2, 1.0)], 2.0);
        let q1: BTreeSet<ContVarName> = [1u64].iter().copied().collect();
        let partial = obs.posterior(&prior, &q1).0.unwrap();
        let full = obs.posterior(&prior, &full_query(&prior)).0.unwrap();
        let sliced = full.slice(&q1);
        assert_prior_close(&partial, &sliced, 1e-10);
    }


    fn vec_nn(values: &[f64]) -> Vec<NotNan<f64>> {
        values.iter().map(|v| nn(*v)).collect()
    }

    #[test]
    fn log_likelihood_equality_satisfied() {
        let obs = GaussianObs::new(
            vec![mk_constraint(&[(1, 1.0), (2, 1.0)], 3.0)],
            vec![],
            set(&[1, 2]),
        );
        let ll = obs.log_likelihood(&vec_nn(&[1.0, 2.0]));
        assert_eq!(ll, 0.0);
    }

    #[test]
    fn log_likelihood_equality_violated() {
        let obs = GaussianObs::new(
            vec![mk_constraint(&[(1, 1.0), (2, 1.0)], 3.0)],
            vec![],
            set(&[1, 2]),
        );
        let ll = obs.log_likelihood(&vec_nn(&[1.0, 2.5]));
        assert_eq!(ll, f64::NEG_INFINITY);
    }

    #[test]
    fn log_likelihood_exclusion_collides() {
        let obs = excl(&[(1, 1.0)], 1.0);
        let ll = obs.log_likelihood(&vec_nn(&[1.0]));
        assert_eq!(ll, f64::NEG_INFINITY);
    }

    #[test]
    fn log_likelihood_exclusion_satisfied() {
        let obs = excl(&[(1, 1.0)], 1.0);
        let ll = obs.log_likelihood(&vec_nn(&[2.0]));
        assert_eq!(ll, 0.0);
    }


    #[test]
    fn affine_moments_unit_coefficient_is_marginal() {
        let prior = mkprior(vec![3], &[1.5], &[&[2.0]]);
        let (m, v) = prior.affine_moments(&mkmap(&[(3, 1.0)]), 0.0);
        assert!((m - 1.5).abs() < 1e-12);
        assert!((v - 2.0).abs() < 1e-12);
    }

    #[test]
    fn affine_moments_constant_offset() {
        let prior = mkprior(vec![1], &[0.5], &[&[1.0]]);
        let (m, v) = prior.affine_moments(&mkmap(&[(1, 1.0)]), 10.0);
        assert!((m - 10.5).abs() < 1e-12);
        assert!((v - 1.0).abs() < 1e-12);
    }

    #[test]
    fn affine_moments_sum_with_correlation() {
        let prior = mkprior(vec![1, 2], &[1.0, 2.0], &[&[1.0, 0.3], &[0.3, 1.0]]);
        let (m, v) = prior.affine_moments(&mkmap(&[(1, 1.0), (2, 1.0)]), 0.0);
        assert!((m - 3.0).abs() < 1e-12);
        assert!((v - 2.6).abs() < 1e-12);
    }

    #[test]
    fn affine_moments_difference_subtracts_covariance() {
        let prior = mkprior(vec![1, 2], &[1.0, 2.0], &[&[1.0, 0.3], &[0.3, 1.0]]);
        let (m, v) = prior.affine_moments(&mkmap(&[(1, 1.0), (2, -1.0)]), 0.0);
        assert!((m + 1.0).abs() < 1e-12);
        assert!((v - 1.4).abs() < 1e-12);
    }

    #[test]
    fn affine_moments_ignores_unknown_vars() {
        let prior = mkprior(vec![1], &[1.0], &[&[1.0]]);
        let (m, v) = prior.affine_moments(&mkmap(&[(1, 2.0), (99, 5.0)]), 0.0);
        assert!((m - 2.0).abs() < 1e-12);
        assert!((v - 4.0).abs() < 1e-12);
    }

    #[test]
    fn affine_covariance_basic() {
        let prior = mkprior(vec![1, 2], &[0.0, 0.0], &[&[1.0, 0.3], &[0.3, 1.0]]);
        let cov = prior.affine_covariance(&mkmap(&[(1, 1.0)]), &mkmap(&[(2, 1.0)]));
        assert!((cov - 0.3).abs() < 1e-12);
    }

    #[test]
    fn affine_covariance_consistent_with_variance() {
        let prior = mkprior(vec![1, 2], &[1.0, 2.0], &[&[2.0, 0.5], &[0.5, 3.0]]);
        let coefs = mkmap(&[(1, 1.5), (2, -0.5)]);
        let (_m, v) = prior.affine_moments(&coefs, 0.0);
        let cov = prior.affine_covariance(&coefs, &coefs);
        assert!((cov - v).abs() < 1e-12);
    }


    use rand::SeedableRng;

    #[test]
    fn sample_gaussian_single_var_matches_prior_mean() {
        let prior = GaussianPrior {
            var_order: vec![5],
            mean: na::DVector::from_vec(vec![3.0]),
            cov: na::DMatrix::from_row_slice(1, 1, &[1.0]),
        };
        let mut r = rand::rngs::StdRng::seed_from_u64(42);
        let n = 2000;
        let mut sum = 0.0;
        for _ in 0..n {
            sum += prior.sample(&mut r)[0].into_inner();
        }
        let mean = sum / n as f64;
        assert!((mean - 3.0).abs() < 0.1, "empirical mean {:.4}", mean);
    }

    #[test]
    fn sample_gaussian_zero_variance_returns_mean() {
        let prior = GaussianPrior {
            var_order: vec![1, 2],
            mean: na::DVector::from_vec(vec![10.0, -5.0]),
            cov: na::DMatrix::from_row_slice(2, 2, &[0.0, 0.0, 0.0, 0.0]),
        };
        let mut r = rand::rngs::StdRng::seed_from_u64(7);
        for _ in 0..5 {
            let xs = prior.sample(&mut r);
            assert_eq!(xs[0].into_inner(), 10.0);
            assert_eq!(xs[1].into_inner(), -5.0);
        }
    }
}
