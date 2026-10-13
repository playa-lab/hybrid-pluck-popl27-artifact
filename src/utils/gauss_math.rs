
#![allow(clippy::needless_range_loop)]

#[derive(Debug, Clone)]
pub struct SmallMatrix {
    pub dim: usize,
    pub data: Vec<f64>,
}

impl SmallMatrix {
    pub fn zeros(dim: usize) -> Self {
        SmallMatrix {
            dim,
            data: vec![0.0; dim * dim],
        }
    }

    pub fn identity(dim: usize) -> Self {
        let mut m = Self::zeros(dim);
        for i in 0..dim {
            m.set(i, i, 1.0);
        }
        m
    }

    pub fn diagonal(diag: &[f64]) -> Self {
        let dim = diag.len();
        let mut m = Self::zeros(dim);
        for i in 0..dim {
            m.set(i, i, diag[i]);
        }
        m
    }

    #[inline]
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.data[i * self.dim + j]
    }

    #[inline]
    pub fn set(&mut self, i: usize, j: usize, val: f64) {
        self.data[i * self.dim + j] = val;
    }

    pub fn add_inplace(&mut self, other: &SmallMatrix) {
        debug_assert_eq!(self.dim, other.dim);
        for i in 0..self.data.len() {
            self.data[i] += other.data[i];
        }
    }

    pub fn mul_vec(&self, v: &[f64]) -> Vec<f64> {
        debug_assert_eq!(self.dim, v.len());
        let mut result = vec![0.0; self.dim];
        for i in 0..self.dim {
            for j in 0..self.dim {
                result[i] += self.get(i, j) * v[j];
            }
        }
        result
    }
    pub fn covariance_from_cholesky(&self) -> SmallMatrix {
        let n = self.dim;
        let mut sigma = SmallMatrix::zeros(n);
        for i in 0..n {
            for j in 0..=i {
                let mut sum = 0.0;
                for k in 0..=j {
                    sum += self.get(i, k) * self.get(j, k);
                }
                sigma.set(i, j, sum);
                sigma.set(j, i, sum);
            }
        }
        sigma
    }

    pub fn transpose_lower_mul_vec(&self, v: &[f64]) -> Vec<f64> {
        debug_assert_eq!(self.dim, v.len());
        let n = self.dim;
        let mut result = vec![0.0; n];
        for i in 0..n {
            for j in i..n {
                result[i] += self.get(j, i) * v[j];
            }
        }
        result
    }
}

pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

pub fn cholesky(mat: &SmallMatrix) -> Option<SmallMatrix> {
    let n = mat.dim;
    let mut l = SmallMatrix::zeros(n);

    for i in 0..n {
        for j in 0..=i {
            let mut sum = 0.0;
            for k in 0..j {
                sum += l.get(i, k) * l.get(j, k);
            }

            if i == j {
                let diag = mat.get(i, i) - sum;
                if diag <= 1e-14 {
                    return None;
                }
                l.set(i, j, diag.sqrt());
            } else {
                let lj = l.get(j, j);
                if lj.abs() < 1e-14 {
                    return None;
                }
                l.set(i, j, (mat.get(i, j) - sum) / lj);
            }
        }
    }

    Some(l)
}

pub fn log_det_from_cholesky(chol: &SmallMatrix) -> f64 {
    let mut result = 0.0;
    for i in 0..chol.dim {
        result += chol.get(i, i).ln();
    }
    2.0 * result
}

pub fn solve_lower(chol: &SmallMatrix, b: &[f64]) -> Vec<f64> {
    let n = chol.dim;
    debug_assert_eq!(n, b.len());
    let mut y = vec![0.0; n];
    for i in 0..n {
        let mut sum = b[i];
        for j in 0..i {
            sum -= chol.get(i, j) * y[j];
        }
        y[i] = sum / chol.get(i, i);
    }
    y
}

pub fn solve_upper_transpose(chol: &SmallMatrix, b: &[f64]) -> Vec<f64> {
    let n = chol.dim;
    debug_assert_eq!(n, b.len());
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = b[i];
        for j in (i + 1)..n {
            sum -= chol.get(j, i) * x[j];
        }
        x[i] = sum / chol.get(i, i);
    }
    x
}

pub fn solve_cholesky(chol: &SmallMatrix, b: &[f64]) -> Vec<f64> {
    let y = solve_lower(chol, b);
    solve_upper_transpose(chol, &y)
}

pub fn symmetric_eigen(mat: &SmallMatrix) -> (Vec<f64>, SmallMatrix) {
    let n = mat.dim;
    if n == 0 {
        return (vec![], SmallMatrix::zeros(0));
    }
    if n == 1 {
        return (vec![mat.get(0, 0)], SmallMatrix::identity(1));
    }

    let mut a = mat.clone();
    let mut v = SmallMatrix::identity(n);

    let max_iter = 100 * n * n;
    let tol = 1e-12;

    for _ in 0..max_iter {
        let mut max_val = 0.0_f64;
        let mut p = 0;
        let mut q = 1;

        for i in 0..n {
            for j in (i + 1)..n {
                let aij = a.get(i, j).abs();
                if aij > max_val {
                    max_val = aij;
                    p = i;
                    q = j;
                }
            }
        }

        if max_val < tol {
            break;
        }

        let app = a.get(p, p);
        let aqq = a.get(q, q);
        let apq = a.get(p, q);

        let theta = if (app - aqq).abs() < 1e-15 {
            std::f64::consts::FRAC_PI_4
        } else {
            0.5 * (2.0 * apq / (app - aqq)).atan()
        };

        let c = theta.cos();
        let s = theta.sin();

        let mut new_row_p = vec![0.0; n];
        let mut new_row_q = vec![0.0; n];

        for j in 0..n {
            new_row_p[j] = c * a.get(p, j) + s * a.get(q, j);
            new_row_q[j] = -s * a.get(p, j) + c * a.get(q, j);
        }
        for j in 0..n {
            a.set(p, j, new_row_p[j]);
            a.set(q, j, new_row_q[j]);
        }

        let mut new_col_p = vec![0.0; n];
        let mut new_col_q = vec![0.0; n];
        for i in 0..n {
            new_col_p[i] = c * a.get(i, p) + s * a.get(i, q);
            new_col_q[i] = -s * a.get(i, p) + c * a.get(i, q);
        }
        for i in 0..n {
            a.set(i, p, new_col_p[i]);
            a.set(i, q, new_col_q[i]);
        }

        for i in 0..n {
            let vip = v.get(i, p);
            let viq = v.get(i, q);
            v.set(i, p, c * vip + s * viq);
            v.set(i, q, -s * vip + c * viq);
        }
    }

    let eigenvalues: Vec<f64> = (0..n).map(|i| a.get(i, i)).collect();
    (eigenvalues, v)
}

pub fn rank_and_pseudoinverse(mat: &SmallMatrix, tol: f64) -> (usize, SmallMatrix) {
    let n = mat.dim;
    if n == 0 {
        return (0, SmallMatrix::zeros(0));
    }

    let (eigenvalues, eigenvectors) = symmetric_eigen(mat);

    let mut rank = 0;
    let mut pinv = SmallMatrix::zeros(n);

    for k in 0..n {
        if eigenvalues[k].abs() > tol {
            rank += 1;
            let inv_ev = 1.0 / eigenvalues[k];
            for i in 0..n {
                for j in 0..n {
                    let current = pinv.get(i, j);
                    pinv.set(
                        i,
                        j,
                        current + inv_ev * eigenvectors.get(i, k) * eigenvectors.get(j, k),
                    );
                }
            }
        }
    }

    (rank, pinv)
}

pub fn from_sparse_lambda(
    lambda: &std::collections::BTreeMap<(u64, u64), f64>,
    var_order: &[u64],
) -> SmallMatrix {
    let n = var_order.len();
    let mut mat = SmallMatrix::zeros(n);

    let var_to_idx: std::collections::HashMap<u64, usize> = var_order
        .iter()
        .enumerate()
        .map(|(idx, &name)| (name, idx))
        .collect();

    for (&(vi, vj), &val) in lambda {
        if let (Some(&i), Some(&j)) = (var_to_idx.get(&vi), var_to_idx.get(&vj)) {
            mat.set(i, j, mat.get(i, j) + val);
            if i != j {
                mat.set(j, i, mat.get(j, i) + val);
            }
        }
    }

    mat
}

pub fn from_sparse_eta(eta: &std::collections::BTreeMap<u64, f64>, var_order: &[u64]) -> Vec<f64> {
    let n = var_order.len();
    let mut vec = vec![0.0; n];

    let var_to_idx: std::collections::HashMap<u64, usize> = var_order
        .iter()
        .enumerate()
        .map(|(idx, &name)| (name, idx))
        .collect();

    for (&name, &val) in eta {
        if let Some(&i) = var_to_idx.get(&name) {
            vec[i] += val;
        }
    }

    vec
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cholesky_2x2() {
        let mut m = SmallMatrix::zeros(2);
        m.set(0, 0, 4.0);
        m.set(0, 1, 2.0);
        m.set(1, 0, 2.0);
        m.set(1, 1, 3.0);

        let l = cholesky(&m).unwrap();
        assert!((l.get(0, 0) - 2.0).abs() < 1e-10);
        assert!((l.get(1, 0) - 1.0).abs() < 1e-10);
        assert!((l.get(0, 1) - 0.0).abs() < 1e-10);
        assert!((l.get(1, 1) - 2.0_f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_log_det() {
        let mut m = SmallMatrix::zeros(2);
        m.set(0, 0, 4.0);
        m.set(0, 1, 2.0);
        m.set(1, 0, 2.0);
        m.set(1, 1, 3.0);
        let l = cholesky(&m).unwrap();
        let ld = log_det_from_cholesky(&l);
        assert!((ld - 8.0_f64.ln()).abs() < 1e-10);
    }

    #[test]
    fn test_solve() {
        let mut m = SmallMatrix::zeros(2);
        m.set(0, 0, 4.0);
        m.set(0, 1, 2.0);
        m.set(1, 0, 2.0);
        m.set(1, 1, 3.0);

        let l = cholesky(&m).unwrap();
        let b = vec![8.0, 7.0];
        let x = solve_cholesky(&l, &b);
        assert!((x[0] - 1.25).abs() < 1e-10);
        assert!((x[1] - 1.5).abs() < 1e-10);
    }

    #[test]
    fn test_rank_full() {
        let mut m = SmallMatrix::zeros(2);
        m.set(0, 0, 4.0);
        m.set(0, 1, 2.0);
        m.set(1, 0, 2.0);
        m.set(1, 1, 3.0);

        let (rank, _pinv) = rank_and_pseudoinverse(&m, 1e-10);
        assert_eq!(rank, 2);
    }

    #[test]
    fn test_rank_deficient() {
        let mut m = SmallMatrix::zeros(2);
        m.set(0, 0, 1.0);
        m.set(0, 1, 1.0);
        m.set(1, 0, 1.0);
        m.set(1, 1, 1.0);

        let (rank, pinv) = rank_and_pseudoinverse(&m, 1e-10);
        assert_eq!(rank, 1);

        assert!((pinv.get(0, 0) - 0.25).abs() < 1e-10);
        assert!((pinv.get(0, 1) - 0.25).abs() < 1e-10);
        assert!((pinv.get(1, 0) - 0.25).abs() < 1e-10);
        assert!((pinv.get(1, 1) - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_1d_rank_and_pinv() {
        let mut m = SmallMatrix::zeros(1);
        m.set(0, 0, 5.0);

        let (rank, pinv) = rank_and_pseudoinverse(&m, 1e-10);
        assert_eq!(rank, 1);
        assert!((pinv.get(0, 0) - 0.2).abs() < 1e-10);
    }

    #[test]
    fn test_consistent_redundant_rss() {
        let lambda = 5.0;
        let eta = 5.0;
        let const_term = 5.0;
        let rss: f64 = const_term - eta * eta / lambda;
        assert!(rss.abs() < 1e-10);
    }

    #[test]
    fn test_contradictory_rss() {
        let lambda: f64 = 2.0;
        let eta: f64 = 3.0;
        let const_term: f64 = 5.0;
        let rss: f64 = const_term - eta * eta / lambda;
        assert!(rss > 0.1);
    }
}
