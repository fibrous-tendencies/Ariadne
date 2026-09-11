//! Equilibrium assembly, Moore--Penrose actions, and null-space analysis.
//!
//! The production path keeps the equilibrium matrix sparse.  It applies the
//! right and left Moore--Penrose projectors with LSQR, then orthonormalises a
//! small projected probe panel with `faer-svd`.  Densifying the equilibrium
//! matrix is confined to [`analyze_dense_svd`], the small-fixture referee.

use crate::sparse::SparseColMatOwned;
use crate::types::{Factorization, FactorizationStrategy, Problem, TheseusError};
use dyn_stack::{GlobalPodBuffer, PodStack};
use faer_core::{Conj, Mat, Parallelism};
use faer_sparse::qr::{factorize_symbolic_qr, QrSymbolicParams};
use faer_svd::{compute_svd, compute_svd_req, ComputeVectors, SvdParams};
use ndarray::Array2;

const ZERO_LENGTH_TOL: f64 = 1e-14;
const DEFAULT_TOL: f64 = 1e-10;

/// Whether the equilibrium unknown is member force `t` or force density `q`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EquilibriumUnknown {
    /// Unit member directions: `A t = p`.
    Force,
    /// Member coordinate differences: `E q = p`.
    ForceDensity,
}

/// Sparse rectangular equilibrium system shared by inverse FDM and rigidity.
#[derive(Debug, Clone)]
pub struct EquilibriumSystem {
    pub a: SparseColMatOwned,
    pub p: Vec<f64>,
    pub lengths: Vec<f64>,
    pub n_eq: usize,
    pub n_edges: usize,
    /// Target positions of free nodes, in equilibrium-row order.
    pub free_positions: Array2<f64>,
    /// Number of ordinary free-node rows before optional reaction rows.
    pub n_free: usize,
}

impl EquilibriumSystem {
    /// Assemble force-form `A` (unit member directions).
    pub fn force(
        problem: &Problem,
        target_free_xyz: &Array2<f64>,
        enforce_zero_rx: bool,
        enforce_zero_ry: bool,
        enforce_zero_rz: bool,
    ) -> Result<Self, TheseusError> {
        Self::assemble(
            problem,
            target_free_xyz,
            EquilibriumUnknown::Force,
            enforce_zero_rx,
            enforce_zero_ry,
            enforce_zero_rz,
        )
    }

    /// Assemble either force-form `A` or force-density-form `E`.
    pub fn assemble(
        problem: &Problem,
        target_free_xyz: &Array2<f64>,
        unknown: EquilibriumUnknown,
        enforce_zero_rx: bool,
        enforce_zero_ry: bool,
        enforce_zero_rz: bool,
    ) -> Result<Self, TheseusError> {
        let topo = &problem.topology;
        let n_free = topo.free_node_indices.len();
        if target_free_xyz.dim() != (n_free, 3) {
            return Err(TheseusError::Shape(format!(
                "target XYZ is {:?}, expected ({n_free}, 3)",
                target_free_xyz.dim()
            )));
        }

        let mut positions = Array2::<f64>::zeros((topo.num_nodes, 3));
        for (i, &node) in topo.free_node_indices.iter().enumerate() {
            for d in 0..3 {
                positions[[node, d]] = target_free_xyz[[i, d]];
            }
        }
        for (i, &node) in topo.fixed_node_indices.iter().enumerate() {
            for d in 0..3 {
                positions[[node, d]] = problem.fixed_node_positions[[i, d]];
            }
        }

        let mut directions = Array2::<f64>::zeros((topo.num_edges, 3));
        for col in 0..topo.num_nodes {
            for nz in
                topo.incidence.col_ptrs[col] as usize..topo.incidence.col_ptrs[col + 1] as usize
            {
                let edge = topo.incidence.row_indices[nz] as usize;
                let sign = topo.incidence.values[nz];
                for d in 0..3 {
                    directions[[edge, d]] += sign * positions[[col, d]];
                }
            }
        }

        let mut lengths = vec![0.0; topo.num_edges];
        for edge in 0..topo.num_edges {
            let length = (0..3)
                .map(|d| directions[[edge, d]].powi(2))
                .sum::<f64>()
                .sqrt();
            if length < ZERO_LENGTH_TOL {
                return Err(TheseusError::Solver(format!(
                    "target edge {edge} has near-zero length ({length:.2e})"
                )));
            }
            lengths[edge] = length;
            if unknown == EquilibriumUnknown::Force {
                for d in 0..3 {
                    directions[[edge, d]] /= length;
                }
            }
        }

        let mut reaction_dims = Vec::new();
        if enforce_zero_rx && !topo.fixed_node_indices.is_empty() {
            reaction_dims.push(0);
        }
        if enforce_zero_ry && !topo.fixed_node_indices.is_empty() {
            reaction_dims.push(1);
        }
        if enforce_zero_rz && !topo.fixed_node_indices.is_empty() {
            reaction_dims.push(2);
        }
        let n_eq = 3 * n_free + reaction_dims.len() * topo.fixed_node_indices.len();
        let cn_t = topo.free_incidence.transpose();
        let mut triplets = Vec::with_capacity(3 * cn_t.nnz());
        for d in 0..3 {
            let row_offset = d * n_free;
            for edge in 0..topo.num_edges {
                let scale = directions[[edge, d]];
                for nz in cn_t.col_ptrs[edge] as usize..cn_t.col_ptrs[edge + 1] as usize {
                    triplets.push((
                        (row_offset + cn_t.row_indices[nz] as usize) as u32,
                        edge as u32,
                        cn_t.values[nz] * scale,
                    ));
                }
            }
        }

        let mut row_offset = 3 * n_free;
        for &d in &reaction_dims {
            for fixed in 0..topo.fixed_node_indices.len() {
                for nz in topo.fixed_incidence.col_ptrs[fixed] as usize
                    ..topo.fixed_incidence.col_ptrs[fixed + 1] as usize
                {
                    let edge = topo.fixed_incidence.row_indices[nz] as usize;
                    triplets.push((
                        (row_offset + fixed) as u32,
                        edge as u32,
                        topo.fixed_incidence.values[nz] * directions[[edge, d]],
                    ));
                }
            }
            row_offset += topo.fixed_node_indices.len();
        }

        let a = SparseColMatOwned::from_triplets(n_eq, topo.num_edges, &triplets)
            .map_err(TheseusError::Shape)?;
        let mut p = vec![0.0; n_eq];
        for d in 0..3 {
            for i in 0..n_free {
                p[d * n_free + i] = problem.free_node_loads[[i, d]];
            }
        }
        Ok(Self {
            a,
            p,
            lengths,
            n_eq,
            n_edges: topo.num_edges,
            free_positions: target_free_xyz.clone(),
            n_free,
        })
    }
}

/// Production and referee null-space implementations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NullspaceMethod {
    Projector,
    Svd,
    /// Reserved comparison mode.  It is deliberately not used to publish rank.
    SparseQr,
}

/// Options for null-space analysis.
#[derive(Debug, Clone)]
pub struct NullspaceOptions {
    pub method: NullspaceMethod,
    pub max_modes: usize,
    pub include_rigid_bodies: bool,
    pub tolerance: f64,
    pub lsqr_max_iterations: usize,
}

impl Default for NullspaceOptions {
    fn default() -> Self {
        Self {
            method: NullspaceMethod::Projector,
            max_modes: 32,
            include_rigid_bodies: false,
            tolerance: DEFAULT_TOL,
            lsqr_max_iterations: 0,
        }
    }
}

/// Pellegrino--Calladine result.  Basis matrices store modes by columns.
#[derive(Debug, Clone)]
pub struct NullspaceReport {
    pub rank: usize,
    pub s: usize,
    pub m_raw: usize,
    pub m: usize,
    pub n_rigid: usize,
    pub particular_t: Array2<f64>,
    pub residual_r: Vec<f64>,
    pub residual_ratio: f64,
    pub self_stress: Array2<f64>,
    pub mechanisms: Array2<f64>,
    pub rigid_bodies: Array2<f64>,
}

/// Minimum-norm linear solver used by nonlinear length retraction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RowSpaceSolver {
    /// Augmented saddle LDL solve.
    Saddle,
    /// Direct zero-start LSQR action.
    Lsqr,
}

/// Controls Gauss--Newton retraction onto the per-member target lengths.
#[derive(Debug, Clone)]
pub struct LengthRetractionOptions {
    pub max_iterations: usize,
    pub tolerance: f64,
    pub lsqr_max_iterations: usize,
    pub solver: RowSpaceSolver,
}

impl Default for LengthRetractionOptions {
    fn default() -> Self {
        Self {
            max_iterations: 30,
            tolerance: 1e-10,
            lsqr_max_iterations: 0,
            // Retraction Jacobians are commonly wide, so their unregularized
            // saddle matrix is singular. Use the explicit minimum-norm path.
            solver: RowSpaceSolver::Lsqr,
        }
    }
}

/// Result of per-member length retraction.
#[derive(Debug, Clone)]
pub struct LengthRetractionResult {
    pub positions: Array2<f64>,
    pub iterations: usize,
    pub converged: bool,
    pub max_length_error: f64,
    pub residual_norm: f64,
}

/// Second-order classification of a mechanism eigenmode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MechanismClass {
    PrestressStable,
    FiniteCandidate,
    PrestressUnstable,
}

/// Geometric stiffness restricted to a supplied mechanism basis.
#[derive(Debug, Clone)]
pub struct PrestressClassification {
    /// `Phi^T K_g Phi`, before diagonalization.
    pub restricted_stiffness: Array2<f64>,
    /// Ascending eigenvalues of the restricted stiffness.
    pub eigenvalues: Vec<f64>,
    /// Eigenvectors by columns, in coordinates of the supplied basis.
    pub eigenvectors: Array2<f64>,
    pub classes: Vec<MechanismClass>,
}

/// Retract free-node positions onto each member's own target length.
///
/// The residual is `L^2 - L0^2`, whose Jacobian transpose is
/// `A(x) diag(2L)`. Every Gauss--Newton increment is the minimum-norm solution
/// and therefore lies in the Jacobian row space; no normal equations or Gram
/// matrix are formed.
pub fn retract_member_lengths(
    problem: &Problem,
    initial_free_xyz: &Array2<f64>,
    target_lengths: &[f64],
    options: &LengthRetractionOptions,
) -> Result<LengthRetractionResult, TheseusError> {
    let n_free = problem.topology.free_node_indices.len();
    if initial_free_xyz.dim() != (n_free, 3) {
        return Err(TheseusError::Shape(format!(
            "retraction XYZ is {:?}, expected ({n_free}, 3)",
            initial_free_xyz.dim()
        )));
    }
    if target_lengths.len() != problem.topology.num_edges {
        return Err(TheseusError::Shape(
            "retraction target length count does not match edge count".into(),
        ));
    }
    if target_lengths
        .iter()
        .any(|length| !length.is_finite() || *length <= ZERO_LENGTH_TOL)
    {
        return Err(TheseusError::Solver(
            "retraction target lengths must be finite and positive".into(),
        ));
    }

    let tolerance = options.tolerance.max(f64::EPSILON);
    let target_scale = target_lengths.iter().copied().fold(1.0_f64, f64::max);
    let mut positions = initial_free_xyz.clone();
    let mut iterations = 0;

    loop {
        let system = EquilibriumSystem::force(problem, &positions, false, false, false)?;
        let length_errors: Vec<f64> = system
            .lengths
            .iter()
            .zip(target_lengths)
            .map(|(length, target)| length - target)
            .collect();
        let residual: Vec<f64> = system
            .lengths
            .iter()
            .zip(target_lengths)
            .map(|(length, target)| length * length - target * target)
            .collect();
        let residual_norm = norm(&residual);
        let max_length_error = length_errors
            .iter()
            .map(|value| value.abs())
            .fold(0.0, f64::max);
        if max_length_error <= tolerance * target_scale {
            return Ok(LengthRetractionResult {
                positions,
                iterations,
                converged: true,
                max_length_error,
                residual_norm,
            });
        }
        if iterations >= options.max_iterations {
            return Ok(LengthRetractionResult {
                positions,
                iterations,
                converged: false,
                max_length_error,
                residual_norm,
            });
        }

        let mut jacobian_transpose = system.a.clone();
        for edge in 0..jacobian_transpose.ncols {
            let column_scale = 2.0 * system.lengths[edge];
            for nz in jacobian_transpose.col_ptrs[edge] as usize
                ..jacobian_transpose.col_ptrs[edge + 1] as usize
            {
                jacobian_transpose.values[nz] *= column_scale;
            }
        }
        let rhs: Vec<f64> = residual.iter().map(|value| -*value).collect();
        let correction = match options.solver {
            RowSpaceSolver::Saddle => solve_saddle_pseudoinverse(
                &jacobian_transpose.transpose(),
                &rhs,
                0.0,
                tolerance,
                options.lsqr_max_iterations,
            )?,
            RowSpaceSolver::Lsqr => apply_transpose_pseudoinverse(
                &jacobian_transpose,
                &rhs,
                tolerance,
                options.lsqr_max_iterations,
            )?,
        };
        let objective = 0.5 * residual_norm * residual_norm;
        let mut step = 1.0;
        let mut accepted = None;
        while step >= 1.0 / 4096.0 {
            let mut trial = positions.clone();
            for d in 0..3 {
                for i in 0..n_free {
                    trial[[i, d]] += step * correction[d * n_free + i];
                }
            }
            if let Ok(trial_system) = EquilibriumSystem::force(problem, &trial, false, false, false)
            {
                let trial_objective = 0.5
                    * trial_system
                        .lengths
                        .iter()
                        .zip(target_lengths)
                        .map(|(length, target)| (length * length - target * target).powi(2))
                        .sum::<f64>();
                if trial_objective < objective {
                    accepted = Some(trial);
                    break;
                }
            }
            step *= 0.5;
        }
        positions = accepted.ok_or_else(|| {
            TheseusError::Solver(
                "length retraction line search could not reduce the per-member residual".into(),
            )
        })?;
        iterations += 1;
    }
}

/// Restrict FDM geometric stiffness to a mechanism basis and classify it.
///
/// The sparse operator is applied edge-by-edge as
/// `C_n^T diag(t/L) C_n`; only the small restricted matrix is dense.
pub fn classify_prestress_stability(
    problem: &Problem,
    system: &EquilibriumSystem,
    prestress_t: &[f64],
    mechanisms: &Array2<f64>,
    tolerance: f64,
) -> Result<PrestressClassification, TheseusError> {
    let n_free = problem.topology.free_node_indices.len();
    if system.n_free != n_free || system.lengths.len() != problem.topology.num_edges {
        return Err(TheseusError::Shape(
            "prestress system does not match problem topology".into(),
        ));
    }
    if prestress_t.len() != problem.topology.num_edges {
        return Err(TheseusError::Shape(
            "prestress force count does not match edge count".into(),
        ));
    }
    if mechanisms.nrows() != 3 * n_free {
        return Err(TheseusError::Shape(format!(
            "mechanism basis has {} rows, expected {} free-coordinate rows",
            mechanisms.nrows(),
            3 * n_free
        )));
    }
    if prestress_t.iter().any(|value| !value.is_finite()) {
        return Err(TheseusError::Solver(
            "prestress forces must be finite".into(),
        ));
    }

    let mode_count = mechanisms.ncols();
    let mut restricted = Array2::<f64>::zeros((mode_count, mode_count));
    let mut node_to_free = vec![None; problem.topology.num_nodes];
    for (free_index, &node) in problem.topology.free_node_indices.iter().enumerate() {
        node_to_free[node] = Some(free_index);
    }
    let incidence = &problem.topology.incidence;
    for edge in 0..problem.topology.num_edges {
        let q = prestress_t[edge] / system.lengths[edge];
        let mut endpoints = Vec::with_capacity(2);
        for col in 0..incidence.ncols {
            for nz in incidence.col_ptrs[col] as usize..incidence.col_ptrs[col + 1] as usize {
                if incidence.row_indices[nz] as usize == edge {
                    endpoints.push((node_to_free[col], incidence.values[nz]));
                }
            }
        }
        for left in 0..mode_count {
            for right in left..mode_count {
                let mut value = 0.0;
                for d in 0..3 {
                    let left_difference: f64 = endpoints
                        .iter()
                        .filter_map(|(free, sign)| {
                            free.map(|i| sign * mechanisms[[d * n_free + i, left]])
                        })
                        .sum();
                    let right_difference: f64 = endpoints
                        .iter()
                        .filter_map(|(free, sign)| {
                            free.map(|i| sign * mechanisms[[d * n_free + i, right]])
                        })
                        .sum();
                    value += q * left_difference * right_difference;
                }
                restricted[[left, right]] += value;
                if left != right {
                    restricted[[right, left]] += value;
                }
            }
        }
    }

    let (eigenvalues, eigenvectors) = symmetric_eigen_jacobi(&restricted, tolerance)?;
    let scale = eigenvalues
        .iter()
        .map(|value| value.abs())
        .fold(1.0_f64, f64::max);
    let threshold = tolerance.max(f64::EPSILON) * scale;
    let classes = eigenvalues
        .iter()
        .map(|value| {
            if *value > threshold {
                MechanismClass::PrestressStable
            } else if *value < -threshold {
                MechanismClass::PrestressUnstable
            } else {
                MechanismClass::FiniteCandidate
            }
        })
        .collect();
    Ok(PrestressClassification {
        restricted_stiffness: restricted,
        eigenvalues,
        eigenvectors,
        classes,
    })
}

/// Apply the Moore--Penrose inverse with zero-start LSQR.
pub fn apply_pseudoinverse(
    a: &SparseColMatOwned,
    rhs: &[f64],
    tolerance: f64,
    max_iterations: usize,
) -> Result<Vec<f64>, TheseusError> {
    if rhs.len() != a.nrows {
        return Err(TheseusError::Shape(
            "pseudoinverse RHS length mismatch".into(),
        ));
    }
    Ok(lsqr(a, rhs, false, tolerance, max_iterations)?.solution)
}

/// Solve the augmented saddle system using LDL only.
///
/// At zero regularization a rank-deficient saddle is singular. Direct MP does
/// not silently run or fall back to LSQR: callers that need a minimum-norm
/// rank-deficient solution must explicitly select Iterative LSQR with λ = 0.
pub fn solve_saddle_pseudoinverse(
    a: &SparseColMatOwned,
    rhs: &[f64],
    regularization: f64,
    _tolerance: f64,
    _max_iterations: usize,
) -> Result<Vec<f64>, TheseusError> {
    if regularization < 0.0 {
        return Err(TheseusError::Solver(
            "regularization must be non-negative".into(),
        ));
    }
    if rhs.len() != a.nrows {
        return Err(TheseusError::Shape("saddle RHS length mismatch".into()));
    }

    let m = a.nrows;
    let n = a.ncols;
    let mut triplets = Vec::with_capacity(m + n + 2 * a.nnz());
    for i in 0..m {
        triplets.push((i as u32, i as u32, 1.0));
    }
    for col in 0..n {
        for nz in a.col_ptrs[col] as usize..a.col_ptrs[col + 1] as usize {
            let row = a.row_indices[nz] as usize;
            let value = a.values[nz];
            triplets.push((row as u32, (m + col) as u32, value));
            triplets.push(((m + col) as u32, row as u32, value));
        }
        triplets.push(((m + col) as u32, (m + col) as u32, -regularization));
    }
    let saddle =
        SparseColMatOwned::from_triplets(m + n, m + n, &triplets).map_err(TheseusError::Shape)?;
    let mut saddle_rhs = vec![0.0; m + n];
    saddle_rhs[..m].copy_from_slice(rhs);
    let solve_ldl = || {
        let mut factor_stack = GlobalPodBuffer::new(dyn_stack::StackReq::empty());
        let mut solve_stack = GlobalPodBuffer::new(dyn_stack::StackReq::empty());
        let factor = Factorization::new(&saddle, FactorizationStrategy::LDL, &mut factor_stack)?;
        let mut workspace = vec![0.0; (m + n).max(1)];
        factor.solve(&saddle_rhs, &mut workspace, &mut solve_stack)
    };

    solve_ldl()
        .and_then(|solution| {
            let candidate = solution[m..].to_vec();
            if candidate.iter().all(|value| value.is_finite()) {
                Ok(candidate)
            } else {
                Err(TheseusError::Solver(
                    "saddle LDL solve produced non-finite values".into(),
                ))
            }
        })
        .map_err(|error| {
            if regularization == 0.0 {
                TheseusError::Solver(format!(
                    "Direct MP requires a nonsingular zero-regularization saddle; \
                     switch to Iterative LSQR with lambda=0 for the minimum-norm solution \
                     ({error})"
                ))
            } else {
                error
            }
        })
}

/// Apply `(A^T)^+`.
fn apply_transpose_pseudoinverse(
    a: &SparseColMatOwned,
    rhs: &[f64],
    tolerance: f64,
    max_iterations: usize,
) -> Result<Vec<f64>, TheseusError> {
    if rhs.len() != a.ncols {
        return Err(TheseusError::Shape(
            "transpose pseudoinverse RHS length mismatch".into(),
        ));
    }
    Ok(lsqr(a, rhs, true, tolerance, max_iterations)?.solution)
}

/// Result of a zero-start LSQR solve.
#[derive(Debug, Clone)]
pub struct LsqrResult {
    pub solution: Vec<f64>,
    pub iterations: usize,
    pub converged: bool,
}

/// Solve `min ||Ax-b||² + λ||x||²` with recurrence-based LSQR stopping.
pub fn solve_lsqr(
    a: &SparseColMatOwned,
    rhs: &[f64],
    tolerance: f64,
    max_iterations: usize,
) -> Result<LsqrResult, TheseusError> {
    lsqr(a, rhs, false, tolerance, max_iterations)
}

fn lsqr(
    a: &SparseColMatOwned,
    rhs: &[f64],
    transposed: bool,
    tolerance: f64,
    max_iterations: usize,
) -> Result<LsqrResult, TheseusError> {
    let (rows, cols) = if transposed {
        (a.ncols, a.nrows)
    } else {
        (a.nrows, a.ncols)
    };
    let at = a.transpose();
    let apply = |x: &[f64]| {
        if transposed {
            at.matvec(x)
        } else {
            a.matvec(x)
        }
    };
    let apply_t = |x: &[f64]| {
        if transposed {
            a.matvec(x)
        } else {
            at.matvec(x)
        }
    };
    debug_assert_eq!(rhs.len(), rows);

    let rhs_norm = norm(rhs);
    if rhs_norm == 0.0 {
        return Ok(LsqrResult {
            solution: vec![0.0; cols],
            iterations: 0,
            converged: true,
        });
    }
    let mut u: Vec<f64> = rhs.iter().map(|v| v / rhs_norm).collect();
    let mut v = apply_t(&u);
    let mut alpha = norm(&v);
    if alpha == 0.0 {
        return Ok(LsqrResult {
            solution: vec![0.0; cols],
            iterations: 0,
            converged: true,
        });
    }
    scale(&mut v, 1.0 / alpha);
    let mut w = v.clone();
    let mut x = vec![0.0; cols];
    let mut phibar = rhs_norm;
    let mut rhobar = alpha;
    let operator_norm = norm(&a.values).max(f64::EPSILON);
    let iterations = if max_iterations == 0 {
        4 * (rows + cols).max(1)
    } else {
        max_iterations
    };

    let stopping_tolerance = tolerance.max(f64::EPSILON);
    let mut performed = 0;
    let mut converged = false;
    let mut anorm = alpha;
    for iteration in 0..iterations {
        let mut next_u = apply(&v);
        axpy(&mut next_u, -alpha, &u);
        let beta = norm(&next_u);
        if beta > 0.0 {
            scale(&mut next_u, 1.0 / beta);
        }
        u = next_u;

        let mut next_v = apply_t(&u);
        axpy(&mut next_v, -beta, &v);
        alpha = norm(&next_v);
        if alpha > 0.0 {
            scale(&mut next_v, 1.0 / alpha);
        }
        v = next_v;

        let rho = rhobar.hypot(beta);
        if rho == 0.0 {
            break;
        }
        let c = rhobar / rho;
        let s = beta / rho;
        let theta = s * alpha;
        rhobar = -c * alpha;
        let phi = c * phibar;
        phibar *= s;
        let tau = s * phi;
        axpy(&mut x, phi / rho, &w);
        for i in 0..cols {
            w[i] = v[i] - (theta / rho) * w[i];
        }
        performed = iteration + 1;
        // Standard LSQR recurrence estimates: ||r|| = |phibar| and
        // ||A^T r|| = alpha*|tau|. This preserves the previous primal and
        // stationarity tolerances without two extra sparse matvecs per step.
        anorm = anorm.hypot(beta).hypot(alpha);
        let residual_norm = phibar.abs();
        let normal_residual = alpha * tau.abs();
        if residual_norm <= stopping_tolerance * rhs_norm
            || normal_residual
                <= stopping_tolerance * anorm.max(operator_norm) * residual_norm.max(rhs_norm)
        {
            converged = true;
            break;
        }
    }
    if x.iter().any(|value| !value.is_finite()) {
        return Err(TheseusError::Solver(
            "LSQR pseudoinverse produced non-finite values".into(),
        ));
    }
    Ok(LsqrResult {
        solution: x,
        iterations: performed,
        converged,
    })
}

/// Analyze with the sparse Moore--Penrose projectors.
pub fn analyze_projector(
    system: &EquilibriumSystem,
    options: &NullspaceOptions,
) -> Result<NullspaceReport, TheseusError> {
    let tol = options.tolerance.max(f64::EPSILON);
    let max_iter = options.lsqr_max_iterations;
    let a = &system.a;

    // The projector traces give nullities without storing dense A or full bases.
    let mut right_trace = 0.0;
    for j in 0..a.ncols {
        let mut col = vec![0.0; a.ncols];
        col[j] = 1.0;
        let projected = project_right(a, &col, tol, max_iter)?;
        right_trace += projected[j];
    }
    let mut left_trace = 0.0;
    for i in 0..a.nrows {
        let mut col = vec![0.0; a.nrows];
        col[i] = 1.0;
        let projected = project_left(a, &col, tol, max_iter)?;
        left_trace += projected[i];
    }
    let s = rounded_nullity(right_trace, a.ncols, tol)?;
    let m_raw = rounded_nullity(left_trace, a.nrows, tol)?;
    let rank = a.ncols.saturating_sub(s);
    if a.nrows.saturating_sub(m_raw) != rank {
        return Err(TheseusError::Solver(format!(
            "projector Calladine/rank disagreement: right rank {rank}, left rank {}",
            a.nrows.saturating_sub(m_raw)
        )));
    }

    let right_panel = projected_panel(a.ncols, s, options.max_modes, |probe| {
        project_right(a, probe, tol, max_iter)
    })?;
    let raw_left_panel = projected_panel(a.nrows, m_raw, m_raw, |probe| {
        project_left(a, probe, tol, max_iter)
    })?;
    let rigid = rigid_basis(system, &raw_left_panel, tol);
    let n_rigid = rigid.ncols();
    let m = m_raw.saturating_sub(n_rigid);
    let mechanisms = if options.include_rigid_bodies {
        take_columns(&raw_left_panel, options.max_modes.min(m_raw))
    } else {
        remove_subspace(&raw_left_panel, &rigid, options.max_modes.min(m), tol)
    };
    let rigid_bodies = if options.include_rigid_bodies {
        rigid
    } else {
        Array2::zeros((a.nrows, 0))
    };

    build_report(
        system,
        rank,
        s,
        m_raw,
        m,
        n_rigid,
        right_panel,
        mechanisms,
        rigid_bodies,
        tol,
        max_iter,
    )
}

/// Dense `faer-svd` referee for paper baselines and tiny fixtures only.
pub fn analyze_dense_svd(
    system: &EquilibriumSystem,
    options: &NullspaceOptions,
) -> Result<NullspaceReport, TheseusError> {
    let (u, singular, v) = dense_svd(&system.a, true)?;
    let sigma_max = singular.first().copied().unwrap_or(0.0);
    let threshold = options
        .tolerance
        .max(f64::EPSILON * system.n_eq.max(system.n_edges) as f64)
        * sigma_max.max(1.0);
    let rank = singular.iter().filter(|&&sigma| sigma > threshold).count();
    let s = system.n_edges - rank;
    let m_raw = system.n_eq - rank;
    let right = columns_range(&v, rank, system.n_edges, options.max_modes);
    let raw_left = columns_range(&u, rank, system.n_eq, m_raw);
    let rigid = rigid_basis(system, &raw_left, threshold.max(DEFAULT_TOL));
    let n_rigid = rigid.ncols();
    let m = m_raw.saturating_sub(n_rigid);
    let mechanisms = if options.include_rigid_bodies {
        take_columns(&raw_left, options.max_modes.min(m_raw))
    } else {
        remove_subspace(
            &raw_left,
            &rigid,
            options.max_modes.min(m),
            threshold.max(DEFAULT_TOL),
        )
    };
    let rigid_bodies = if options.include_rigid_bodies {
        rigid
    } else {
        Array2::zeros((system.n_eq, 0))
    };
    build_report(
        system,
        rank,
        s,
        m_raw,
        m,
        n_rigid,
        right,
        mechanisms,
        rigid_bodies,
        options.tolerance,
        options.lsqr_max_iterations,
    )
}

/// Analyze a tall, numerically full-column-rank matrix with COLAMD sparse QR.
///
/// This comparison path deliberately does not infer rank from raw `R`
/// diagonals. It validates the numeric factorization by solving `A X = A`
/// and requiring `X ~= I`, then validates the load least-squares stationarity.
/// Rank-deficient or wide systems return a precise unsupported error.
pub fn analyze_sparse_qr(
    system: &EquilibriumSystem,
    options: &NullspaceOptions,
) -> Result<NullspaceReport, TheseusError> {
    let a = &system.a;
    if a.nrows < a.ncols {
        return Err(TheseusError::Solver(format!(
            "sparse QR comparison requires rows >= columns ({} < {})",
            a.nrows, a.ncols
        )));
    }
    let a_ref = a.as_faer_ref();
    let symbolic = factorize_symbolic_qr(a_ref.symbolic(), QrSymbolicParams::default())
        .map_err(|error| TheseusError::Linalg(format!("COLAMD sparse QR symbolic: {error:?}")))?;
    let mut indices = vec![0_u32; symbolic.len_indices()];
    let mut values = vec![0.0; symbolic.len_values()];
    let req = symbolic
        .factorize_numeric_qr_req::<f64>(Parallelism::Rayon(0))
        .map_err(|error| TheseusError::Linalg(format!("sparse QR workspace: {error:?}")))?;
    let mut factor_memory = GlobalPodBuffer::new(req);
    let qr = symbolic.factorize_numeric_qr(
        &mut indices,
        values.as_mut_slice(),
        a_ref,
        Parallelism::Rayon(0),
        PodStack::new(&mut factor_memory),
    );

    let solve = |right_hand_sides: &mut Mat<f64>| -> Result<(), TheseusError> {
        let req = symbolic
            .solve_in_place_req::<f64>(right_hand_sides.ncols(), Parallelism::Rayon(0))
            .map_err(|error| {
                TheseusError::Linalg(format!("sparse QR solve workspace: {error:?}"))
            })?;
        let mut solve_memory = GlobalPodBuffer::new(req);
        qr.solve_in_place_with_conj(
            Conj::No,
            right_hand_sides.as_mut(),
            Parallelism::Rayon(0),
            PodStack::new(&mut solve_memory),
        );
        Ok(())
    };

    // Numeric rank validation: for full column rank, QR must recover every
    // canonical coefficient vector from its exact image A e_j.
    let mut images = Mat::<f64>::zeros(a.nrows, a.ncols);
    for col in 0..a.ncols {
        for nz in a.col_ptrs[col] as usize..a.col_ptrs[col + 1] as usize {
            images.write(a.row_indices[nz] as usize, col, a.values[nz]);
        }
    }
    solve(&mut images)?;
    let mut identity_error_sq = 0.0;
    for col in 0..a.ncols {
        for row in 0..a.ncols {
            let expected = if row == col { 1.0 } else { 0.0 };
            identity_error_sq += (images.read(row, col) - expected).powi(2);
        }
    }
    let identity_error = identity_error_sq.sqrt();
    let tolerance = options.tolerance.max(DEFAULT_TOL);
    let rank_allowance = 1000.0 * tolerance * (a.ncols.max(1) as f64).sqrt();
    if !identity_error.is_finite() || identity_error > rank_allowance {
        return Err(TheseusError::Solver(format!(
            "sparse QR comparison cannot establish full column rank: numeric identity residual {identity_error:.3e} exceeds {rank_allowance:.3e}"
        )));
    }

    let mut load_rhs = Mat::<f64>::zeros(a.nrows, 1);
    for (row, &value) in system.p.iter().enumerate() {
        load_rhs.write(row, 0, value);
    }
    solve(&mut load_rhs)?;
    let qr_particular: Vec<f64> = (0..a.ncols).map(|row| load_rhs.read(row, 0)).collect();
    if qr_particular.iter().any(|value| !value.is_finite()) {
        return Err(TheseusError::Solver(
            "sparse QR comparison produced non-finite coefficients".into(),
        ));
    }
    let mut load_residual = a.matvec(&qr_particular);
    for (value, target) in load_residual.iter_mut().zip(&system.p) {
        *value -= target;
    }
    let normal_residual = norm(&a.transpose().matvec(&load_residual));
    let operator_norm = norm(&a.values).max(1.0);
    let stationarity_allowance =
        1000.0 * tolerance * operator_norm * norm(&load_residual).max(norm(&system.p)).max(1.0);
    if normal_residual > stationarity_allowance {
        return Err(TheseusError::Solver(format!(
            "sparse QR comparison failed least-squares residual validation: normal residual {normal_residual:.3e} exceeds {stationarity_allowance:.3e}"
        )));
    }

    let rank = a.ncols;
    let s = 0;
    let m_raw = a.nrows - rank;
    let raw_left = projected_panel(a.nrows, m_raw, m_raw, |probe| {
        project_left(a, probe, tolerance, options.lsqr_max_iterations)
    })?;
    let rigid = rigid_basis(system, &raw_left, tolerance);
    let n_rigid = rigid.ncols();
    let m = m_raw.saturating_sub(n_rigid);
    let mechanisms = if options.include_rigid_bodies {
        take_columns(&raw_left, options.max_modes.min(m_raw))
    } else {
        remove_subspace(&raw_left, &rigid, options.max_modes.min(m), tolerance)
    };
    let rigid_bodies = if options.include_rigid_bodies {
        rigid
    } else {
        Array2::zeros((a.nrows, 0))
    };
    build_report(
        system,
        rank,
        s,
        m_raw,
        m,
        n_rigid,
        Array2::zeros((a.ncols, 0)),
        mechanisms,
        rigid_bodies,
        tolerance,
        options.lsqr_max_iterations,
    )
}

/// Dispatch null-space analysis. Sparse QR remains comparison-only.
pub fn analyze(
    system: &EquilibriumSystem,
    options: &NullspaceOptions,
) -> Result<NullspaceReport, TheseusError> {
    match options.method {
        NullspaceMethod::Projector => analyze_projector(system, options),
        NullspaceMethod::Svd => analyze_dense_svd(system, options),
        NullspaceMethod::SparseQr => analyze_sparse_qr(system, options),
    }
}

fn build_report(
    system: &EquilibriumSystem,
    rank: usize,
    s: usize,
    m_raw: usize,
    m: usize,
    n_rigid: usize,
    self_stress: Array2<f64>,
    mechanisms: Array2<f64>,
    rigid_bodies: Array2<f64>,
    tolerance: f64,
    max_iterations: usize,
) -> Result<NullspaceReport, TheseusError> {
    verify_kernel(&system.a, &self_stress, false, tolerance)?;
    verify_kernel(&system.a, &mechanisms, true, tolerance)?;
    if s as isize - m_raw as isize != system.n_edges as isize - system.n_eq as isize {
        return Err(TheseusError::Solver("Calladine identity failed".into()));
    }
    let particular = apply_pseudoinverse(
        &system.a,
        &system.p,
        tolerance.max(DEFAULT_TOL),
        max_iterations,
    )?;
    let mut residual_r = system.a.matvec(&particular);
    for (value, load) in residual_r.iter_mut().zip(&system.p) {
        *value -= load;
    }
    let residual_ratio = norm(&residual_r) / norm(&system.p).max(f64::EPSILON);
    let particular_t = Array2::from_shape_vec((system.n_edges, 1), particular)
        .map_err(|error| TheseusError::Shape(format!("particular result shape error: {error}")))?;
    Ok(NullspaceReport {
        rank,
        s,
        m_raw,
        m,
        n_rigid,
        particular_t,
        residual_r,
        residual_ratio,
        self_stress,
        mechanisms,
        rigid_bodies,
    })
}

fn project_right(
    a: &SparseColMatOwned,
    vector: &[f64],
    tolerance: f64,
    max_iterations: usize,
) -> Result<Vec<f64>, TheseusError> {
    let av = a.matvec(vector);
    let row = apply_pseudoinverse(a, &av, tolerance, max_iterations)?;
    Ok(vector.iter().zip(row).map(|(&x, y)| x - y).collect())
}

fn project_left(
    a: &SparseColMatOwned,
    vector: &[f64],
    tolerance: f64,
    max_iterations: usize,
) -> Result<Vec<f64>, TheseusError> {
    let at = a.transpose();
    let atv = at.matvec(vector);
    let row = apply_transpose_pseudoinverse(a, &atv, tolerance, max_iterations)?;
    Ok(vector.iter().zip(row).map(|(&x, y)| x - y).collect())
}

fn projected_panel<F>(
    dimension: usize,
    nullity: usize,
    max_modes: usize,
    mut project: F,
) -> Result<Array2<f64>, TheseusError>
where
    F: FnMut(&[f64]) -> Result<Vec<f64>, TheseusError>,
{
    let wanted = nullity.min(max_modes);
    if wanted == 0 {
        return Ok(Array2::zeros((dimension, 0)));
    }
    let probes = dimension.min(wanted.saturating_add(8));
    let mut state = 0x9e37_79b9_7f4a_7c15_u64 ^ dimension as u64;
    let mut panel = Array2::<f64>::zeros((dimension, probes));
    for col in 0..probes {
        let mut probe = vec![0.0; dimension];
        for value in &mut probe {
            *value = gaussian(&mut state);
        }
        let projected = project(&probe)?;
        for row in 0..dimension {
            panel[[row, col]] = projected[row];
        }
    }
    let (_, singular, _) = svd_array(&panel, false)?;
    let sigma_max = singular.first().copied().unwrap_or(0.0);
    let numerical_rank = singular
        .iter()
        .filter(|&&sigma| sigma > DEFAULT_TOL * sigma_max.max(1.0))
        .count();
    if numerical_rank < wanted {
        return Err(TheseusError::Solver(format!(
            "projector probe panel found {numerical_rank} modes, expected at least {wanted}"
        )));
    }
    // faer-svd 0.17 can return inaccurate left vectors for an exactly
    // rank-deficient rectangular panel. The singular values still provide the
    // thin-panel rank check; orthonormalize the already projected columns to
    // preserve their verified kernel span.
    let basis = orthonormalize(&panel, DEFAULT_TOL * sigma_max.max(1.0));
    if basis.ncols() < wanted {
        return Err(TheseusError::Solver(format!(
            "projector orthonormalization retained {} modes, expected at least {wanted}",
            basis.ncols()
        )));
    }
    Ok(take_columns(&basis, wanted))
}

fn dense_svd(
    sparse: &SparseColMatOwned,
    full_vectors: bool,
) -> Result<(Array2<f64>, Vec<f64>, Array2<f64>), TheseusError> {
    let mut dense = Array2::<f64>::zeros((sparse.nrows, sparse.ncols));
    for col in 0..sparse.ncols {
        for nz in sparse.col_ptrs[col] as usize..sparse.col_ptrs[col + 1] as usize {
            dense[[sparse.row_indices[nz] as usize, col]] = sparse.values[nz];
        }
    }
    svd_array_full(&dense, full_vectors)
}

fn svd_array(
    dense: &Array2<f64>,
    full_vectors: bool,
) -> Result<(Array2<f64>, Vec<f64>, Array2<f64>), TheseusError> {
    svd_array_full(dense, full_vectors)
}

fn svd_array_full(
    dense: &Array2<f64>,
    full_vectors: bool,
) -> Result<(Array2<f64>, Vec<f64>, Array2<f64>), TheseusError> {
    let (m, n) = dense.dim();
    let k = m.min(n);
    let matrix = Mat::from_fn(m, n, |i, j| dense[[i, j]]);
    let mut singular = Mat::<f64>::zeros(k, 1);
    let u_cols = if full_vectors { m } else { k };
    let v_cols = if full_vectors { n } else { k };
    let mut u = Mat::<f64>::zeros(m, u_cols);
    let mut v = Mat::<f64>::zeros(n, v_cols);
    let vectors = if full_vectors {
        ComputeVectors::Full
    } else {
        ComputeVectors::Thin
    };
    let params = SvdParams::default();
    let req = compute_svd_req::<f64>(m, n, vectors, vectors, Parallelism::Rayon(0), params)
        .map_err(|error| TheseusError::Linalg(format!("SVD workspace: {error:?}")))?;
    let mut memory = GlobalPodBuffer::new(req);
    compute_svd(
        matrix.as_ref(),
        singular.as_mut(),
        Some(u.as_mut()),
        Some(v.as_mut()),
        Parallelism::Rayon(0),
        PodStack::new(&mut memory),
        params,
    );
    let singular_values = (0..k).map(|i| singular.read(i, 0)).collect();
    let u_array = Array2::from_shape_fn((m, u_cols), |(i, j)| u.read(i, j));
    let v_array = Array2::from_shape_fn((n, v_cols), |(i, j)| v.read(i, j));
    Ok((u_array, singular_values, v_array))
}

fn rounded_nullity(trace: f64, dimension: usize, tolerance: f64) -> Result<usize, TheseusError> {
    let rounded = trace.round().clamp(0.0, dimension as f64);
    let allowance = (50.0 * tolerance * dimension.max(1) as f64).max(1e-5);
    if (trace - rounded).abs() > allowance {
        return Err(TheseusError::Solver(format!(
            "projector trace {trace:.8} is not an integer within tolerance {allowance:.2e}"
        )));
    }
    Ok(rounded as usize)
}

fn rigid_basis(
    system: &EquilibriumSystem,
    _mechanism_basis: &Array2<f64>,
    tolerance: f64,
) -> Array2<f64> {
    if system.n_eq != 3 * system.n_free || system.n_free == 0 {
        return Array2::zeros((system.n_eq, 0));
    }
    let mut centroid = [0.0; 3];
    for i in 0..system.n_free {
        for d in 0..3 {
            centroid[d] += system.free_positions[[i, d]] / system.n_free as f64;
        }
    }
    let mut candidates = Array2::<f64>::zeros((system.n_eq, 6));
    for i in 0..system.n_free {
        for d in 0..3 {
            candidates[[d * system.n_free + i, d]] = 1.0;
        }
        let r = [
            system.free_positions[[i, 0]] - centroid[0],
            system.free_positions[[i, 1]] - centroid[1],
            system.free_positions[[i, 2]] - centroid[2],
        ];
        let rotations = [[0.0, -r[2], r[1]], [r[2], 0.0, -r[0]], [-r[1], r[0], 0.0]];
        for axis in 0..3 {
            for d in 0..3 {
                candidates[[d * system.n_free + i, 3 + axis]] = rotations[axis][d];
            }
        }
    }
    // A constrained free-node set does not admit global rigid motion. Keep a
    // candidate only when the candidate itself satisfies A^T d = 0.
    let at = system.a.transpose();
    let scale = system
        .a
        .values
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt()
        .max(1.0);
    let mut accepted = Vec::new();
    for j in 0..candidates.ncols() {
        let candidate: Vec<f64> = (0..candidates.nrows())
            .map(|i| candidates[[i, j]])
            .collect();
        let candidate_norm = norm(&candidate);
        if candidate_norm > tolerance
            && norm(&at.matvec(&candidate)) <= 100.0 * tolerance * scale * candidate_norm
        {
            accepted.push(candidate);
        }
    }
    let accepted = Array2::from_shape_fn((system.n_eq, accepted.len()), |(i, j)| accepted[j][i]);
    orthonormalize(&accepted, tolerance)
}

fn remove_subspace(
    basis: &Array2<f64>,
    removed: &Array2<f64>,
    max_modes: usize,
    tolerance: f64,
) -> Array2<f64> {
    let mut residual = basis.clone();
    for j in 0..residual.ncols() {
        for k in 0..removed.ncols() {
            let dot = (0..residual.nrows())
                .map(|i| removed[[i, k]] * residual[[i, j]])
                .sum::<f64>();
            for i in 0..residual.nrows() {
                residual[[i, j]] -= dot * removed[[i, k]];
            }
        }
    }
    take_columns(&orthonormalize(&residual, tolerance), max_modes)
}

fn orthonormalize(input: &Array2<f64>, tolerance: f64) -> Array2<f64> {
    let mut columns: Vec<Vec<f64>> = Vec::new();
    for j in 0..input.ncols() {
        let mut vector: Vec<f64> = (0..input.nrows()).map(|i| input[[i, j]]).collect();
        for _ in 0..2 {
            for column in &columns {
                let dot = vector.iter().zip(column).map(|(a, b)| a * b).sum::<f64>();
                axpy(&mut vector, -dot, column);
            }
        }
        let vector_norm = norm(&vector);
        if vector_norm > tolerance {
            scale(&mut vector, 1.0 / vector_norm);
            columns.push(vector);
        }
    }
    Array2::from_shape_fn((input.nrows(), columns.len()), |(i, j)| columns[j][i])
}

fn verify_kernel(
    a: &SparseColMatOwned,
    basis: &Array2<f64>,
    transpose: bool,
    tolerance: f64,
) -> Result<(), TheseusError> {
    let operator = if transpose { a.transpose() } else { a.clone() };
    let mut residual_sq = 0.0;
    for j in 0..basis.ncols() {
        let column: Vec<f64> = (0..basis.nrows()).map(|i| basis[[i, j]]).collect();
        residual_sq += operator
            .matvec(&column)
            .into_iter()
            .map(|value| value * value)
            .sum::<f64>();
    }
    let scale = a.values.iter().map(|v| v * v).sum::<f64>().sqrt().max(1.0);
    if residual_sq.sqrt() > 100.0 * tolerance.max(DEFAULT_TOL) * scale {
        return Err(TheseusError::Solver(format!(
            "null-space residual {:.3e} exceeds tolerance",
            residual_sq.sqrt()
        )));
    }
    Ok(())
}

fn columns_range(matrix: &Array2<f64>, start: usize, end: usize, cap: usize) -> Array2<f64> {
    let count = end.saturating_sub(start).min(cap);
    Array2::from_shape_fn((matrix.nrows(), count), |(i, j)| matrix[[i, start + j]])
}

fn take_columns(matrix: &Array2<f64>, count: usize) -> Array2<f64> {
    columns_range(matrix, 0, matrix.ncols(), count)
}

fn gaussian(state: &mut u64) -> f64 {
    let u1 = uniform(state).max(f64::MIN_POSITIVE);
    let u2 = uniform(state);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

fn uniform(state: &mut u64) -> f64 {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    let bits = state.wrapping_mul(0x2545_f491_4f6c_dd1d);
    (bits >> 11) as f64 * (1.0 / ((1_u64 << 53) as f64))
}

fn norm(values: &[f64]) -> f64 {
    values.iter().map(|value| value * value).sum::<f64>().sqrt()
}

fn scale(values: &mut [f64], factor: f64) {
    for value in values {
        *value *= factor;
    }
}

fn axpy(target: &mut [f64], alpha: f64, source: &[f64]) {
    for (target, source) in target.iter_mut().zip(source) {
        *target += alpha * source;
    }
}

fn symmetric_eigen_jacobi(
    matrix: &Array2<f64>,
    tolerance: f64,
) -> Result<(Vec<f64>, Array2<f64>), TheseusError> {
    let (rows, cols) = matrix.dim();
    if rows != cols {
        return Err(TheseusError::Shape(
            "symmetric eigenproblem must be square".into(),
        ));
    }
    if rows == 0 {
        return Ok((Vec::new(), Array2::zeros((0, 0))));
    }
    let mut a = matrix.clone();
    let mut vectors = Array2::from_shape_fn((rows, rows), |(i, j)| (i == j) as u8 as f64);
    let scale = a.iter().map(|value| value.abs()).fold(1.0_f64, f64::max);
    let threshold = tolerance.max(f64::EPSILON) * scale;
    let max_sweeps = 50 * rows * rows;
    for _ in 0..max_sweeps {
        let mut p = 0;
        let mut q = 0;
        let mut largest = 0.0;
        for i in 0..rows {
            for j in (i + 1)..rows {
                if a[[i, j]].abs() > largest {
                    largest = a[[i, j]].abs();
                    p = i;
                    q = j;
                }
            }
        }
        if largest <= threshold {
            let mut order: Vec<usize> = (0..rows).collect();
            order.sort_by(|&left, &right| {
                a[[left, left]]
                    .partial_cmp(&a[[right, right]])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let eigenvalues = order.iter().map(|&i| a[[i, i]]).collect();
            let eigenvectors = Array2::from_shape_fn((rows, rows), |(i, j)| vectors[[i, order[j]]]);
            return Ok((eigenvalues, eigenvectors));
        }

        let app = a[[p, p]];
        let aqq = a[[q, q]];
        let apq = a[[p, q]];
        let angle = 0.5 * (2.0 * apq).atan2(aqq - app);
        let c = angle.cos();
        let s = angle.sin();
        for k in 0..rows {
            if k != p && k != q {
                let akp = a[[k, p]];
                let akq = a[[k, q]];
                a[[k, p]] = c * akp - s * akq;
                a[[p, k]] = a[[k, p]];
                a[[k, q]] = s * akp + c * akq;
                a[[q, k]] = a[[k, q]];
            }
        }
        a[[p, p]] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
        a[[q, q]] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
        a[[p, q]] = 0.0;
        a[[q, p]] = 0.0;
        for k in 0..rows {
            let vkp = vectors[[k, p]];
            let vkq = vectors[[k, q]];
            vectors[[k, p]] = c * vkp - s * vkq;
            vectors[[k, q]] = s * vkp + c * vkq;
        }
    }
    Err(TheseusError::Solver(
        "restricted geometric-stiffness eigensolve did not converge".into(),
    ))
}
