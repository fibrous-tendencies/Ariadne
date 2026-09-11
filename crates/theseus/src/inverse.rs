//! Inverse FDM solvers: find force densities q from a target geometry.
//!
//! Direct unconstrained particulars use Gram, the saddle (Moore--Penrose /
//! Tikhonov), or sparse QR. Iterative unconstrained uses LSQR. A finite box
//! (signs and/or bounds) uses Clarabel (Direct) or spectral projected
//! gradient (Iterative). `L2 = false` wraps any inner in IRLS.

use crate::nullspace::{
    apply_pseudoinverse, solve_lsqr, solve_saddle_pseudoinverse, EquilibriumSystem,
    EquilibriumUnknown,
};
use crate::sparse::SparseColMatOwned;
use crate::types::{Factorization, FactorizationStrategy, Problem, TheseusError};
use dyn_stack::{GlobalPodBuffer, PodStack};
use faer_core::{Conj, Mat, Parallelism};
use faer_sparse::qr::{factorize_symbolic_qr, QrSymbolicParams, SymbolicQr};
use ndarray::Array2;

/// InvFDM particular-solution backend selected by the Grasshopper menu.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParticularMethod {
    /// Gram normal equations `(MᵀM + λI)q = Mᵀp`. λ = 0 is unregularized.
    Gram = 0,
    /// Augmented saddle / Moore–Penrose / Tikhonov.
    Augmented = 1,
    /// Tall full-column-rank sparse QR least squares (comparison path).
    SparseQr = 2,
    /// Clarabel quadratic programming, including unconstrained Direct solves.
    Clarabel = 3,
}

/// Direct factorization versus iterative matvec linear algebra.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinearAlgebra {
    Direct = 0,
    Iterative = 1,
}

/// Which residual the inverse solve minimises.
///
/// The FDM residual at a frozen target is the geometric error pre-conditioned
/// by the Laplacian: `r(q) = E(x*)q - p = D(q)(x* - x(q))`, so
/// `x(q) - x* = -D(q)^-1 r(q)`.  `Force` minimises `‖r‖` and therefore treats
/// every nodal force error alike; the geometric variants minimise `‖D^-1 r‖`,
/// which is the distance the forward solve actually lands from the target.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum InverseMetric {
    /// Minimise `‖Mx - p‖`. Historical behaviour.
    #[default]
    Force = 0,
    /// Minimise the geometric error with the Jacobian frozen at the target,
    /// `E(x*)`. One weighted least-squares solve per outer iteration.
    Geometry = 1,
    /// Minimise the geometric error with the Jacobian re-assembled at the
    /// current form-found geometry `x(q_k)`. True Gauss--Newton; costs no
    /// forward solve because `x(q_k) = x* - D(q_k)^-1 r(q_k)`.
    GeometryNewton = 2,
}

impl InverseMetric {
    /// True when the solve is weighted by the Laplacian compliance.
    pub fn is_geometric(self) -> bool {
        !matches!(self, Self::Force)
    }
}

impl TryFrom<i32> for InverseMetric {
    type Error = TheseusError;

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Force),
            1 => Ok(Self::Geometry),
            2 => Ok(Self::GeometryNewton),
            other => Err(TheseusError::Solver(format!(
                "unknown InvFDM metric {other} \
                 (expected 0=Force, 1=Geometry, 2=GeometryNewton)"
            ))),
        }
    }
}

impl TryFrom<i32> for LinearAlgebra {
    type Error = TheseusError;

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Direct),
            1 => Ok(Self::Iterative),
            other => Err(TheseusError::Solver(format!(
                "unknown InvFDM linear algebra {other} (expected 0=Direct, 1=Iterative)"
            ))),
        }
    }
}

/// Per-edge box on the inverse unknown (q or t).
#[derive(Debug, Clone)]
pub struct BoxBounds {
    pub lower: Vec<f64>,
    pub upper: Vec<f64>,
}

impl BoxBounds {
    pub fn unconstrained(n: usize) -> Self {
        Self {
            lower: vec![f64::NEG_INFINITY; n],
            upper: vec![f64::INFINITY; n],
        }
    }

    pub fn has_finite(&self) -> bool {
        self.lower.iter().any(|v| v.is_finite()) || self.upper.iter().any(|v| v.is_finite())
    }
}

/// Options for [`solve_inverse_fdm`].
#[derive(Debug, Clone)]
pub struct InverseFdmOptions {
    pub regularization: f64,
    pub use_l2: bool,
    pub max_l1_iter: usize,
    pub particular_method: ParticularMethod,
    pub linear_algebra: LinearAlgebra,
    pub enforce_zero_rx: bool,
    pub enforce_zero_ry: bool,
    pub enforce_zero_rz: bool,
    pub solve_for_q: bool,
    pub signs: Vec<i32>,
    pub lower: Vec<f64>,
    pub upper: Vec<f64>,
    pub max_iter: usize,
    pub tol: f64,
    /// Residual metric. `Force` reproduces the historical solve exactly.
    pub metric: InverseMetric,
    /// Reference force densities that seed the geometric outer loop. Empty
    /// derives `q_e = 1 / L_e` from the target edge lengths. Ignored by
    /// `InverseMetric::Force`.
    pub q_ref: Vec<f64>,
    /// Outer iteration budget for the geometric metrics.
    pub max_outer: usize,
}

impl InverseFdmOptions {
    pub fn direct_unconstrained(
        regularization: f64,
        use_l2: bool,
        max_l1_iter: usize,
        particular_method: ParticularMethod,
        enforce_zero_rx: bool,
        enforce_zero_ry: bool,
        enforce_zero_rz: bool,
        solve_for_q: bool,
    ) -> Self {
        Self {
            regularization,
            use_l2,
            max_l1_iter,
            particular_method,
            linear_algebra: LinearAlgebra::Direct,
            enforce_zero_rx,
            enforce_zero_ry,
            enforce_zero_rz,
            solve_for_q,
            signs: Vec::new(),
            lower: Vec::new(),
            upper: Vec::new(),
            max_iter: 500,
            tol: 1e-6,
            metric: InverseMetric::Force,
            q_ref: Vec::new(),
            max_outer: DEFAULT_MAX_OUTER,
        }
    }
}

/// Default outer iteration budget for the geometric metrics.
pub const DEFAULT_MAX_OUTER: usize = 8;

/// Result of an inverse-FDM particular (force densities at the target).
#[derive(Debug, Clone)]
pub struct InverseFdmResult {
    pub q: Vec<f64>,
    pub iterations: usize,
    pub converged: bool,
    /// `‖x(q) − x*‖`, the distance the forward solve lands from the target.
    ///
    /// Exact for geometry-independent loads via `e = −D(q)^-1 r(q)`. Reported
    /// for every metric, so a `Force` solve can be compared against a
    /// geometric one on the same scale. NaN when the Laplacian at the returned
    /// q is singular and the error could not be evaluated.
    pub geometric_error: f64,
}

/// Result from the box-constrained spectral projected-gradient solver.
pub struct SpgBoxResult {
    pub q: Vec<f64>,
    pub iterations: usize,
    pub converged: bool,
}

impl TryFrom<i32> for ParticularMethod {
    type Error = TheseusError;

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Gram),
            1 => Ok(Self::Augmented),
            2 => Ok(Self::SparseQr),
            3 => Ok(Self::Clarabel),
            other => Err(TheseusError::Solver(format!(
                "unknown InvFDM particular method {other} \
                 (expected 0=Gram, 1=Augmented, 2=SparseQr, 3=Clarabel)"
            ))),
        }
    }
}

/// Factorise with LDL and solve a single RHS using ephemeral workspace.
fn ldl_solve(g: &SparseColMatOwned, rhs: &[f64]) -> Result<Vec<f64>, TheseusError> {
    let mut factor_stack = dyn_stack::GlobalPodBuffer::new(dyn_stack::StackReq::empty());
    let mut solve_stack = dyn_stack::GlobalPodBuffer::new(dyn_stack::StackReq::empty());
    let mut workspace = vec![0.0; rhs.len().max(1)];
    let fac = Factorization::new(g, FactorizationStrategy::LDL, &mut factor_stack)?;
    fac.solve(rhs, &mut workspace, &mut solve_stack)
}

/// Gram LDL solve. At λ = 0 a singular MᵀM is reported as such.
fn gram_ldl_solve(
    g: &SparseColMatOwned,
    rhs: &[f64],
    regularization: f64,
) -> Result<Vec<f64>, TheseusError> {
    match ldl_solve(g, rhs) {
        Ok(sol) => Ok(sol),
        Err(_) if regularization == 0.0 => Err(TheseusError::Solver(
            "Gram at λ=0: MᵀM is singular".into(),
        )),
        Err(error) => Err(error),
    }
}

/// Convert axial forces F back to force densities: `q[k] = F[k] / L[k]`.
fn forces_to_q(f: &[f64], lengths: &[f64]) -> Vec<f64> {
    f.iter()
        .zip(lengths.iter())
        .map(|(&fi, &li)| fi / li)
        .collect()
}

// ─────────────────────────────────────────────────────────────
//  Augmented (saddle-point) system builder
// ─────────────────────────────────────────────────────────────

/// Build the augmented (saddle-point) system that avoids forming M^T M.
///
/// Instead of the normal equations `(M^T M + λI) q = M^T p`, assembles:
///
///   [ I    M  ] [r]   [rhs_top]
///   [ M^T  -λI] [q] = [   0   ]
///
/// The augmented matrix is symmetric indefinite with size (m+n)×(m+n)
/// where m = M.rows() and n = M.cols().  Non-zeros are O(nnz(M))
/// rather than O(nnz(M^T M)), which avoids the fill-in explosion
/// from the Gram product at large scales (>50k edges).
///
/// Returns `(K, rhs)` where K is CSC and `rhs = [rhs_top; 0]`.
fn build_augmented_system(
    m_mat: &SparseColMatOwned,
    rhs_top: &[f64],
    regularization: f64,
) -> (SparseColMatOwned, Vec<f64>) {
    let m = m_mat.nrows;
    let n = m_mat.ncols;
    let total = m + n;

    let mut triplets: Vec<(u32, u32, f64)> = Vec::new();

    // Top-left: I  (m × m)
    for i in 0..m {
        triplets.push((i as u32, i as u32, 1.0));
    }

    // Upper-right: M  and  lower-left: M^T  (symmetric pair)
    for col in 0..n {
        let start = m_mat.col_ptrs[col] as usize;
        let end_ = m_mat.col_ptrs[col + 1] as usize;
        for nz in start..end_ {
            let row = m_mat.row_indices[nz] as usize;
            let val = m_mat.values[nz];
            triplets.push((row as u32, (m + col) as u32, val));
            triplets.push(((m + col) as u32, row as u32, val));
        }
    }

    // Bottom-right: -λI  (n × n)
    for j in 0..n {
        triplets.push(((m + j) as u32, (m + j) as u32, -regularization));
    }

    let k_mat =
        SparseColMatOwned::from_triplets(total, total, &triplets).expect("build_augmented_system");

    // RHS = [rhs_top; 0]
    let mut rhs = vec![0.0; total];
    rhs[..m].copy_from_slice(rhs_top);

    (k_mat, rhs)
}

// ─────────────────────────────────────────────────────────────
//  Pseudoinverse  (Tikhonov-regularised sparse normal equations)
// ─────────────────────────────────────────────────────────────

/// Find force densities via pseudoinverse of the equilibrium system.
///
/// Solves  `(M^T M + λI) q = M^T p`  using sparse LDL factorisation.
/// When `solve_for_q` is false, solves for axial forces F instead and
/// recovers q = F / L from the target edge lengths.
pub fn solve_pseudoinverse(
    problem: &Problem,
    target_free_xyz: &Array2<f64>,
    regularization: f64,
    enforce_zero_rx: bool,
    enforce_zero_ry: bool,
    enforce_zero_rz: bool,
    solve_for_q: bool,
) -> Result<Vec<f64>, TheseusError> {
    let ne = problem.topology.num_edges;
    let system = EquilibriumSystem::assemble(
        problem,
        target_free_xyz,
        if solve_for_q {
            EquilibriumUnknown::ForceDensity
        } else {
            EquilibriumUnknown::Force
        },
        enforce_zero_rx,
        enforce_zero_ry,
        enforce_zero_rz,
    )?;
    let m_mat = system.a;
    let p = system.p;

    // G = M^T M  (ne × ne, sparse)
    let m_t = m_mat.transpose();
    let mut g = SparseColMatOwned::sparse_times_sparse(&m_t, &m_mat)
        .map_err(|e| TheseusError::Solver(e))?;

    // Add regularisation: G += λ I
    if regularization > 0.0 {
        g.add_diagonal(regularization);
    }

    // h = M^T p  (sparse-dense matvec)
    let h = m_t.matvec(&p);

    // Factorise G and solve (LDL for normal equations)
    let sol = gram_ldl_solve(&g, &h, regularization)?;

    let q = if solve_for_q {
        sol
    } else {
        forces_to_q(&sol, &system.lengths)
    };

    // Validate solution
    for (i, &v) in q.iter().enumerate() {
        if !v.is_finite() {
            return Err(TheseusError::Solver(format!(
                "pseudoinverse produced non-finite q at edge {i}; \
                 increase regularisation or check target geometry",
            )));
        }
    }

    if q.len() != ne {
        return Err(TheseusError::Shape(format!(
            "pseudoinverse returned {} q values, expected {ne}",
            q.len()
        )));
    }

    Ok(q)
}

// ─────────────────────────────────────────────────────────────
//  Pseudoinverse L2  (augmented saddle-point system)
// ─────────────────────────────────────────────────────────────

/// Find force densities via pseudoinverse using the augmented saddle-point system.
///
/// Mathematically equivalent to `solve_pseudoinverse` but avoids forming M^T M.
/// Instead factorises the larger but much sparser augmented system:
///
///   [ I    M  ] [r]   [p]
///   [ M^T  -λI] [q] = [0]
///
/// Asymptotically faster for large meshes (>50k edges) where the M^T M
/// fill-in explosion dominates runtime. At `λ = 0`, Direct MP is an LDL-only
/// solve and reports a singular saddle instead of silently falling back.
/// When `solve_for_q` is false, solves for axial forces F instead and
/// recovers q = F / L from the target edge lengths.
pub fn solve_pseudoinverse_augmented(
    problem: &Problem,
    target_free_xyz: &Array2<f64>,
    regularization: f64,
    enforce_zero_rx: bool,
    enforce_zero_ry: bool,
    enforce_zero_rz: bool,
    solve_for_q: bool,
) -> Result<Vec<f64>, TheseusError> {
    if regularization < 0.0 {
        return Err(TheseusError::Solver(
            "regularization must be non-negative".into(),
        ));
    }

    let ne = problem.topology.num_edges;
    let system = EquilibriumSystem::assemble(
        problem,
        target_free_xyz,
        if solve_for_q {
            EquilibriumUnknown::ForceDensity
        } else {
            EquilibriumUnknown::Force
        },
        enforce_zero_rx,
        enforce_zero_ry,
        enforce_zero_rz,
    )?;
    let m_mat = system.a;
    let p = system.p;
    let m_rows = p.len();

    let raw = if regularization == 0.0 {
        solve_saddle_pseudoinverse(&m_mat, &p, 0.0, 1e-11, 0)?
    } else {
        let (k_mat, rhs) = build_augmented_system(&m_mat, &p, regularization);
        let sol = ldl_solve(&k_mat, &rhs)?;
        sol[m_rows..m_rows + ne].to_vec()
    };
    let q = if solve_for_q {
        raw
    } else {
        forces_to_q(&raw, &system.lengths)
    };

    for (i, &v) in q.iter().enumerate() {
        if !v.is_finite() {
            return Err(TheseusError::Solver(format!(
                "pseudoinverse (augmented) produced non-finite q at edge {i}; \
                 increase regularisation or check target geometry",
            )));
        }
    }

    if q.len() != ne {
        return Err(TheseusError::Shape(format!(
            "pseudoinverse (augmented) returned {} q values, expected {ne}",
            q.len()
        )));
    }

    Ok(q)
}

// ─────────────────────────────────────────────────────────────
//  Pseudoinverse L1  (IRLS — iteratively reweighted least squares)
// ─────────────────────────────────────────────────────────────

/// Find force densities via L1-minimisation of the equilibrium residual.
///
/// Minimises `‖Mq − p‖₁` (sum of absolute residuals) using IRLS:
/// each iteration solves a weighted least-squares problem
/// `(M^T W M + λI) q = M^T W p` where `W = diag(1/max(|r_i|, ε))`.
///
/// Warm-starts from the L2 pseudoinverse solution for fast convergence.
/// When `solve_for_q` is false, solves for axial forces F instead and
/// recovers q = F / L from the target edge lengths.
pub fn solve_pseudoinverse_l1(
    problem: &Problem,
    target_free_xyz: &Array2<f64>,
    regularization: f64,
    max_iter: usize,
    enforce_zero_rx: bool,
    enforce_zero_ry: bool,
    enforce_zero_rz: bool,
    solve_for_q: bool,
) -> Result<Vec<f64>, TheseusError> {
    let ne = problem.topology.num_edges;
    let system = EquilibriumSystem::assemble(
        problem,
        target_free_xyz,
        if solve_for_q {
            EquilibriumUnknown::ForceDensity
        } else {
            EquilibriumUnknown::Force
        },
        enforce_zero_rx,
        enforce_zero_ry,
        enforce_zero_rz,
    )?;
    let m_mat = system.a;
    let p = system.p;

    let m_t = m_mat.transpose();
    let m_rows = p.len();

    // Warm-start: L2 solution  (M^T M + λI) q = M^T p
    let mut g_l2 = SparseColMatOwned::sparse_times_sparse(&m_t, &m_mat)
        .map_err(|e| TheseusError::Solver(e))?;
    if regularization > 0.0 {
        g_l2.add_diagonal(regularization);
    }
    let h_l2 = m_t.matvec(&p);
    let mut sol = gram_ldl_solve(&g_l2, &h_l2, regularization)?;

    const ABS_EPS: f64 = 1e-12;
    let mut prev_l1 = f64::MAX;

    for _ in 0..max_iter {
        // r = M*sol − p
        let mut r = m_mat.matvec(&sol);
        for (ri, &pi) in r.iter_mut().zip(p.iter()) {
            *ri -= pi;
        }

        // Adaptive epsilon: proportional to the largest residual so that
        // the weight ratio stays bounded (~1e4:1), preventing the weighted
        // Gram matrix from becoming ill-conditioned.
        let max_abs_r = r.iter().map(|ri| ri.abs()).fold(0.0_f64, f64::max);
        let eps_iter = (1e-4 * max_abs_r).max(ABS_EPS);

        // L1 objective and IRLS weights  (w_i = 1/max(|r_i|, eps))
        let mut l1_obj = 0.0;
        let mut sqrt_w = vec![0.0; m_rows];
        let mut wp = vec![0.0; m_rows];
        for i in 0..m_rows {
            let abs_r = r[i].abs();
            l1_obj += abs_r;
            let w_i = 1.0 / abs_r.max(eps_iter);
            sqrt_w[i] = w_i.sqrt();
            wp[i] = w_i * p[i];
        }

        // Normalize weights so max(w) = 1, keeping M^T W M entries O(1).
        // Divide regularisation by the same factor to preserve the solution.
        let w_max = sqrt_w.iter().map(|s| s * s).fold(0.0_f64, f64::max);
        let effective_reg = if w_max > 0.0 {
            let inv_sqrt_wmax = 1.0 / w_max.sqrt();
            for i in 0..m_rows {
                sqrt_w[i] *= inv_sqrt_wmax;
                wp[i] /= w_max;
            }
            regularization / w_max
        } else {
            regularization
        };

        // Convergence: relative change in L1 objective
        if prev_l1 < f64::MAX {
            let rel_change = (prev_l1 - l1_obj).abs() / (prev_l1 + ABS_EPS);
            if rel_change < 1e-8 {
                break;
            }
        }
        prev_l1 = l1_obj;

        // Weighted normal equations: G = M_w^T M_w + λ'I,  h = M^T W' p
        let m_w = row_scaled_copy(&m_mat, &sqrt_w);
        let m_w_t = m_w.transpose();
        let mut g = SparseColMatOwned::sparse_times_sparse(&m_w_t, &m_w)
            .map_err(|e| TheseusError::Solver(e))?;
        if effective_reg > 0.0 {
            g.add_diagonal(effective_reg);
        }
        let h = m_t.matvec(&wp);

        sol = gram_ldl_solve(&g, &h, effective_reg)?;
    }

    let q = if solve_for_q {
        sol
    } else {
        forces_to_q(&sol, &system.lengths)
    };

    // Validate solution
    for (i, &v) in q.iter().enumerate() {
        if !v.is_finite() {
            return Err(TheseusError::Solver(format!(
                "pseudoinverse L1 produced non-finite q at edge {i}; \
                 increase regularisation or check target geometry",
            )));
        }
    }
    if q.len() != ne {
        return Err(TheseusError::Shape(format!(
            "pseudoinverse L1 returned {} q values, expected {ne}",
            q.len()
        )));
    }

    Ok(q)
}

// ─────────────────────────────────────────────────────────────
//  Pseudoinverse L1  (augmented saddle-point IRLS)
// ─────────────────────────────────────────────────────────────

/// Find force densities via L1-minimisation using the augmented saddle-point system.
///
/// Equivalent to `solve_pseudoinverse_l1` but each IRLS iteration factorises
/// the augmented system instead of forming M_w^T M_w. Avoids fill-in explosion
/// at large scales; zero regularization uses the Moore--Penrose LSQR fallback.
/// When `solve_for_q` is false, solves for axial forces F instead and
/// recovers q = F / L from the target edge lengths.
pub fn solve_pseudoinverse_l1_augmented(
    problem: &Problem,
    target_free_xyz: &Array2<f64>,
    regularization: f64,
    max_iter: usize,
    enforce_zero_rx: bool,
    enforce_zero_ry: bool,
    enforce_zero_rz: bool,
    solve_for_q: bool,
) -> Result<Vec<f64>, TheseusError> {
    if regularization < 0.0 {
        return Err(TheseusError::Solver(
            "regularization must be non-negative".into(),
        ));
    }

    let ne = problem.topology.num_edges;
    let system = EquilibriumSystem::assemble(
        problem,
        target_free_xyz,
        if solve_for_q {
            EquilibriumUnknown::ForceDensity
        } else {
            EquilibriumUnknown::Force
        },
        enforce_zero_rx,
        enforce_zero_ry,
        enforce_zero_rz,
    )?;
    let m_mat = system.a;
    let p = system.p;

    let m_rows = p.len();

    // Warm-start: L2 solution via augmented system
    let mut sol = if regularization == 0.0 {
        apply_pseudoinverse(&m_mat, &p, 1e-11, 0)?
    } else {
        let (k_l2, rhs_l2) = build_augmented_system(&m_mat, &p, regularization);
        let sol_l2 = ldl_solve(&k_l2, &rhs_l2)?;
        sol_l2[m_rows..m_rows + ne].to_vec()
    };

    const ABS_EPS: f64 = 1e-12;
    let mut prev_l1 = f64::MAX;

    for _ in 0..max_iter {
        // r = M*sol − p
        let mut r = m_mat.matvec(&sol);
        for (ri, &pi) in r.iter_mut().zip(p.iter()) {
            *ri -= pi;
        }

        let max_abs_r = r.iter().map(|ri| ri.abs()).fold(0.0_f64, f64::max);
        let eps_iter = (1e-4 * max_abs_r).max(ABS_EPS);

        // L1 objective and IRLS weights
        let mut l1_obj = 0.0;
        let mut sqrt_w = vec![0.0; m_rows];
        for i in 0..m_rows {
            let abs_r = r[i].abs();
            l1_obj += abs_r;
            let w_i = 1.0 / abs_r.max(eps_iter);
            sqrt_w[i] = w_i.sqrt();
        }

        // Normalize weights so max(w) = 1
        let w_max = sqrt_w.iter().map(|s| s * s).fold(0.0_f64, f64::max);
        let effective_reg = if w_max > 0.0 {
            let inv_sqrt_wmax = 1.0 / w_max.sqrt();
            for sw in sqrt_w.iter_mut() {
                *sw *= inv_sqrt_wmax;
            }
            regularization / w_max
        } else {
            regularization
        };

        // Convergence: relative change in L1 objective
        if prev_l1 < f64::MAX {
            let rel_change = (prev_l1 - l1_obj).abs() / (prev_l1 + ABS_EPS);
            if rel_change < 1e-8 {
                break;
            }
        }
        prev_l1 = l1_obj;

        // Build M_w = diag(sqrt_w) * M, then augmented system with
        // RHS top = sqrt_w ⊙ p  (so that M_w^T * rhs_top = M^T W p).
        let m_w = row_scaled_copy(&m_mat, &sqrt_w);
        let rhs_top: Vec<f64> = (0..m_rows).map(|i| sqrt_w[i] * p[i]).collect();

        let (k_mat, rhs) = build_augmented_system(&m_w, &rhs_top, effective_reg);

        if effective_reg == 0.0 {
            sol = apply_pseudoinverse(&m_w, &rhs_top, 1e-11, 0)?;
        } else {
            let iter_sol = ldl_solve(&k_mat, &rhs)?;
            sol = iter_sol[m_rows..m_rows + ne].to_vec();
        }
    }

    let q = if solve_for_q {
        sol
    } else {
        forces_to_q(&sol, &system.lengths)
    };

    for (i, &v) in q.iter().enumerate() {
        if !v.is_finite() {
            return Err(TheseusError::Solver(format!(
                "pseudoinverse L1 (augmented) produced non-finite q at edge {i}; \
                 increase regularisation or check target geometry",
            )));
        }
    }
    if q.len() != ne {
        return Err(TheseusError::Shape(format!(
            "pseudoinverse L1 (augmented) returned {} q values, expected {ne}",
            q.len()
        )));
    }

    Ok(q)
}

// ─────────────────────────────────────────────────────────────
//  Pseudoinverse dispatcher
// ─────────────────────────────────────────────────────────────

/// Sparse QR least-squares particular for tall full-column-rank systems.
///
/// Uses COLAMD symbolic ordering and validates rank from sparse `R` pivots.
/// Rank-deficient or wide systems return a precise unsupported error.
pub fn solve_pseudoinverse_qr(
    problem: &Problem,
    target_free_xyz: &Array2<f64>,
    enforce_zero_rx: bool,
    enforce_zero_ry: bool,
    enforce_zero_rz: bool,
    solve_for_q: bool,
) -> Result<Vec<f64>, TheseusError> {
    let ne = problem.topology.num_edges;
    let system = EquilibriumSystem::assemble(
        problem,
        target_free_xyz,
        if solve_for_q {
            EquilibriumUnknown::ForceDensity
        } else {
            EquilibriumUnknown::Force
        },
        enforce_zero_rx,
        enforce_zero_ry,
        enforce_zero_rz,
    )?;
    let a = &system.a;
    if a.nrows < a.ncols {
        return Err(TheseusError::Solver(format!(
            "sparse QR particular requires rows >= columns ({} < {})",
            a.nrows, a.ncols
        )));
    }

    let mut symbolic = None;
    let sol = solve_qr_on(a, &system.p, true, &mut symbolic)?;
    let q = if solve_for_q {
        sol
    } else {
        forces_to_q(&sol, &system.lengths)
    };

    for (i, &v) in q.iter().enumerate() {
        if !v.is_finite() {
            return Err(TheseusError::Solver(format!(
                "pseudoinverse (sparse QR) produced non-finite q at edge {i}"
            )));
        }
    }
    if q.len() != ne {
        return Err(TheseusError::Shape(format!(
            "pseudoinverse (sparse QR) returned {} q values, expected {ne}",
            q.len()
        )));
    }
    Ok(q)
}

/// Dispatch to an inverse-FDM particular. Unconstrained Direct keeps the
/// historical Gram / saddle / QR arguments.
pub fn solve_pseudoinverse_dispatch(
    problem: &Problem,
    target_free_xyz: &Array2<f64>,
    regularization: f64,
    use_l2: bool,
    max_l1_iter: usize,
    particular_method: ParticularMethod,
    enforce_zero_rx: bool,
    enforce_zero_ry: bool,
    enforce_zero_rz: bool,
    solve_for_q: bool,
) -> Result<Vec<f64>, TheseusError> {
    solve_inverse_fdm(
        problem,
        target_free_xyz,
        InverseFdmOptions::direct_unconstrained(
            regularization,
            use_l2,
            max_l1_iter,
            particular_method,
            enforce_zero_rx,
            enforce_zero_ry,
            enforce_zero_rz,
            solve_for_q,
        ),
    )
    .map(|result| result.q)
}

include!("inverse_extra.rs");
