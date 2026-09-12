use crate::sparse::SparseColMatOwned;
use ndarray::Array2;
use std::fmt;
use std::fmt::Debug;

// ─────────────────────────────────────────────────────────────
//  Error type
// ─────────────────────────────────────────────────────────────

/// Unified error type for all fallible operations in the crate.
///
/// Every function in the public Rust API returns `Result<T, TheseusError>`
/// instead of panicking.  The FFI layer translates these into integer
/// return codes + a thread-local error message.
#[derive(Debug)]
pub enum TheseusError {
    /// Linear algebra failure (singular / not-SPD matrix, etc.).
    Linalg(String),
    /// Sparsity pattern is inconsistent (should never happen after
    /// a correct `FdmCache::new`).
    SparsityMismatch { edge: usize, row: usize, col: usize },
    /// The factorization has not been computed yet.
    MissingFactorization,
    /// Argmin solver returned an error.
    Solver(String),
    /// Shape mismatch in input data.
    Shape(String),
    /// Optimization was cancelled by the caller via the progress callback.
    Cancelled,
}

impl fmt::Display for TheseusError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Linalg(e) => write!(f, "linear algebra error: {e}"),
            Self::SparsityMismatch { edge, row, col } => write!(
                f,
                "sparsity pattern mismatch: edge {edge}, ({row},{col}) not in A"
            ),
            Self::MissingFactorization => {
                write!(f, "factorization not computed (call solve_fdm first)")
            }
            Self::Solver(msg) => write!(f, "solver error: {msg}"),
            Self::Shape(msg) => write!(f, "shape error: {msg}"),
            Self::Cancelled => write!(f, "optimization cancelled by user"),
        }
    }
}

impl std::error::Error for TheseusError {}

impl From<faer_sparse::cholesky::CholeskyError> for TheseusError {
    fn from(e: faer_sparse::cholesky::CholeskyError) -> Self {
        Self::Linalg(format!("{:?}", e))
    }
}

impl From<faer_sparse::FaerError> for TheseusError {
    fn from(e: faer_sparse::FaerError) -> Self {
        Self::Linalg(e.to_string())
    }
}

impl From<argmin::core::Error> for TheseusError {
    fn from(e: argmin::core::Error) -> Self {
        let msg = e.to_string();
        if msg == Self::Cancelled.to_string() {
            Self::Cancelled
        } else {
            Self::Solver(msg)
        }
    }
}

/// Resize a faer workspace buffer to satisfy the required stack size.
fn ensure_pod_stack(stack: &mut dyn_stack::GlobalPodBuffer, req: dyn_stack::StackReq) {
    *stack = dyn_stack::GlobalPodBuffer::new(req);
}

// ─────────────────────────────────────────────────────────────
//  Constants
// ─────────────────────────────────────────────────────────────

pub const DEFAULT_BARRIER_SHARPNESS: f64 = 10.0;

// ─────────────────────────────────────────────────────────────
//  Objective trait  (extensible — implement for custom objectives)
// ─────────────────────────────────────────────────────────────

/// Trait for form-finding objectives.
///
/// Implement `loss` and `accumulate_gradient` to add custom objectives.
/// The gradient method must accumulate into `cache.grad_x` (for implicit
/// adjoint contributions) and/or `cache.grad_q` (for explicit q gradients).
pub trait ObjectiveTrait: Debug + Send + Sync {
    /// Scalar loss contribution from this objective.
    fn loss(&self, snap: &GeometrySnapshot) -> f64;

    /// Accumulate dJ/dx̂ into `cache.grad_x` and explicit dJ/dq into
    /// `cache.grad_q`.  Called before the adjoint solve.
    fn accumulate_gradient(&self, cache: &mut FdmCache, problem: &Problem);

    /// Weight of this objective (used for display/debugging).
    fn weight(&self) -> f64;

    /// Validate static configuration (e.g. degenerate constraint geometry).
    fn validate(&self) -> Result<(), TheseusError> {
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────
//  Built-in objective structs
// ─────────────────────────────────────────────────────────────

/// How per-node target-geometry residuals are reduced to a scalar.
///
/// Residuals are per-node squared Euclidean (or in-plane) distances; `n` is
/// the number of target nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TargetGeometryReduction {
    /// Σ_i ‖e_i‖². Historical default.
    #[default]
    Sse = 0,
    /// (1/n) Σ_i ‖e_i‖²
    Mse = 1,
    /// sqrt((1/n) Σ_i ‖e_i‖²)
    Rmse = 2,
}

impl TryFrom<i32> for TargetGeometryReduction {
    type Error = TheseusError;

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Sse),
            1 => Ok(Self::Mse),
            2 => Ok(Self::Rmse),
            _ => Err(TheseusError::Shape(format!(
                "invalid target geometry reduction: {value}"
            ))),
        }
    }
}

#[derive(Debug, Clone)]
pub struct TargetXYZ {
    pub weight: f64,
    pub node_indices: Vec<usize>,
    pub target: Array2<f64>, // n × 3
    pub reduction: TargetGeometryReduction,
}

#[derive(Debug, Clone)]
pub struct TargetXY {
    pub weight: f64,
    pub node_indices: Vec<usize>,
    pub target: Array2<f64>,
    pub reduction: TargetGeometryReduction,
}

/// Target positions on an arbitrary plane. Origin and axes are in world coordinates;
/// axes should be unit and orthogonal (e.g. Rhino plane Origin, XAxis, YAxis).
#[derive(Debug, Clone)]
pub struct TargetPlane {
    pub weight: f64,
    pub node_indices: Vec<usize>,
    pub target: Array2<f64>, // n × 3 world positions
    pub origin: [f64; 3],
    pub x_axis: [f64; 3],
    pub y_axis: [f64; 3],
    pub reduction: TargetGeometryReduction,
}

/// Planar constraint: pull nodes onto a plane along a given direction. No target positions —
/// loss is Σ t² where t = n·(O−P)/(n·d) (signed distance along d to the plane).
#[derive(Debug, Clone)]
pub struct PlanarConstraintAlongDirection {
    pub weight: f64,
    pub node_indices: Vec<usize>,
    pub origin: [f64; 3],
    pub x_axis: [f64; 3],
    pub y_axis: [f64; 3],
    pub direction: [f64; 3],
}

#[derive(Debug, Clone)]
pub struct TargetLength {
    pub weight: f64,
    pub edge_indices: Vec<usize>,
    pub target: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct TargetForce {
    pub weight: f64,
    pub edge_indices: Vec<usize>,
    pub target: Vec<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LengthVarianceNormalizationStrategy {
    /// Divides by mean(length)^2 + eps. Existing normalized variance behavior.
    SquaredMean,
    /// Uses raw variance to show scale sensitivity.
    None,
}

impl TryFrom<i32> for LengthVarianceNormalizationStrategy {
    type Error = TheseusError;

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::SquaredMean),
            1 => Ok(Self::None),
            _ => Err(TheseusError::Shape(format!(
                "invalid length variance normalization strategy: {value}"
            ))),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ForceVarianceNormalizationStrategy {
    /// Divides by mean(force)^2 + eps. Existing normalized variance behavior.
    SquaredMean,
    /// Divides by mean(abs(force))^2 + eps. Better for mixed-sign force magnitudes.
    AbsoluteSquaredMean,
    /// Uses raw variance without scale normalization.
    None,
}

impl TryFrom<i32> for ForceVarianceNormalizationStrategy {
    type Error = TheseusError;

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::SquaredMean),
            1 => Ok(Self::AbsoluteSquaredMean),
            2 => Ok(Self::None),
            _ => Err(TheseusError::Shape(format!(
                "invalid force variance normalization strategy: {value}"
            ))),
        }
    }
}

#[derive(Debug, Clone)]
pub struct LengthVariation {
    pub weight: f64,
    pub edge_indices: Vec<usize>,
    pub sharpness: f64,
    pub use_normalized_variance: bool,
    pub normalization_strategy: LengthVarianceNormalizationStrategy,
}

#[derive(Debug, Clone)]
pub struct ForceVariation {
    pub weight: f64,
    pub edge_indices: Vec<usize>,
    pub sharpness: f64,
    pub use_normalized_variance: bool,
    pub normalization_strategy: ForceVarianceNormalizationStrategy,
}

#[derive(Debug, Clone)]
pub struct SumForceLength {
    pub weight: f64,
    pub edge_indices: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct MinLength {
    pub weight: f64,
    pub edge_indices: Vec<usize>,
    pub threshold: Vec<f64>,
    pub sharpness: f64,
}

#[derive(Debug, Clone)]
pub struct MaxLength {
    pub weight: f64,
    pub edge_indices: Vec<usize>,
    pub threshold: Vec<f64>,
    pub sharpness: f64,
}

#[derive(Debug, Clone)]
pub struct MinForce {
    pub weight: f64,
    pub edge_indices: Vec<usize>,
    pub threshold: Vec<f64>,
    pub sharpness: f64,
}

#[derive(Debug, Clone)]
pub struct MaxForce {
    pub weight: f64,
    pub edge_indices: Vec<usize>,
    pub threshold: Vec<f64>,
    pub sharpness: f64,
}

#[derive(Debug, Clone)]
pub struct RigidSetCompare {
    pub weight: f64,
    pub node_indices: Vec<usize>,
    pub target: Array2<f64>,
}

#[derive(Debug, Clone)]
pub struct ReactionDirection {
    pub weight: f64,
    pub anchor_indices: Vec<usize>,
    pub target_directions: Array2<f64>, // n × 3, unit rows
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReactionMagnitudeBehavior {
    Target,
    Max,
    Min,
}

impl TryFrom<i32> for ReactionMagnitudeBehavior {
    type Error = TheseusError;

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Target),
            1 => Ok(Self::Max),
            2 => Ok(Self::Min),
            _ => Err(TheseusError::Shape(format!(
                "invalid reaction magnitude behavior: {value}"
            ))),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReactionMagnitudeSign {
    Unsigned,
    SignedProjected,
}

impl TryFrom<i32> for ReactionMagnitudeSign {
    type Error = TheseusError;

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Unsigned),
            1 => Ok(Self::SignedProjected),
            _ => Err(TheseusError::Shape(format!(
                "invalid reaction magnitude sign semantics: {value}"
            ))),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QParameterizationMode {
    /// Optimize unconstrained direct q values with soft bound penalties.
    DirectSoftBounds,
    /// Optimize physical q values within strict finite, two-sided edge bounds.
    ///
    /// This mode intentionally treats finite increasing q intervals as a
    /// model contract. Fixed or one-sided/unbounded q values should use
    /// [`QParameterizationMode::DirectSoftBounds`].
    DirectBoxBounds,
}

impl TryFrom<i32> for QParameterizationMode {
    type Error = TheseusError;

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::DirectSoftBounds),
            2 => Ok(Self::DirectBoxBounds),
            _ => Err(TheseusError::Shape(format!(
                "invalid q parameterization mode: {value}"
            ))),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ReactionMagnitude {
    pub weight: f64,
    pub anchor_indices: Vec<usize>,
    pub target_directions: Array2<f64>,
    pub target_magnitudes: Vec<f64>,
    pub behavior: ReactionMagnitudeBehavior,
    pub sign: ReactionMagnitudeSign,
}

#[derive(Debug, Clone)]
pub struct ReactionDirectionMagnitude {
    pub weight: f64,
    pub anchor_indices: Vec<usize>,
    pub target_directions: Array2<f64>,
    pub target_magnitudes: Vec<f64>,
    pub behavior: ReactionMagnitudeBehavior,
    pub sign: ReactionMagnitudeSign,
}

// ─────────────────────────────────────────────────────────────
//  Bounds
// ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Bounds {
    pub lower: Vec<f64>,
    pub upper: Vec<f64>,
}

impl Bounds {
    pub fn default_for(num_edges: usize) -> Self {
        Self {
            lower: vec![1e-8; num_edges],
            upper: vec![f64::INFINITY; num_edges],
        }
    }
}

// ─────────────────────────────────────────────────────────────
//  Solver / Tracing options
// ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct SolverOptions {
    pub absolute_tolerance: f64,
    pub relative_tolerance: f64,
    pub max_iterations: usize,
    pub report_frequency: usize,
    pub barrier_weight: f64,
    pub barrier_sharpness: f64,
    pub q_parameterization_mode: QParameterizationMode,
    pub anchor_saturation_lambda: f64,
}

impl Default for SolverOptions {
    fn default() -> Self {
        Self {
            absolute_tolerance: 1e-6,
            relative_tolerance: 1e-6,
            max_iterations: 500,
            report_frequency: 1,
            barrier_weight: 10.0,
            barrier_sharpness: DEFAULT_BARRIER_SHARPNESS,
            q_parameterization_mode: QParameterizationMode::DirectSoftBounds,
            anchor_saturation_lambda: 1.0,
        }
    }
}

// ─────────────────────────────────────────────────────────────
//  Network topology
// ─────────────────────────────────────────────────────────────

/// Compressed connectivity information built once from the incidence matrix.
#[derive(Debug, Clone)]
pub struct NetworkTopology {
    /// Full incidence matrix  (ne × nn)  with ±1 entries.
    pub incidence: SparseColMatOwned,
    /// Free-node incidence    (ne × nn_free)
    pub free_incidence: SparseColMatOwned,
    /// Fixed-node incidence   (ne × nn_fixed)
    pub fixed_incidence: SparseColMatOwned,
    pub num_edges: usize,
    pub num_nodes: usize,
    pub free_node_indices: Vec<usize>,
    pub fixed_node_indices: Vec<usize>,
}

// ─────────────────────────────────────────────────────────────
//  Anchor info  (variable / fixed supports)
// ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct AnchorInfo {
    pub variable_indices: Vec<usize>,
    pub fixed_indices: Vec<usize>,
    pub reference_positions: Array2<f64>,        // n_fixed × 3
    pub initial_variable_positions: Array2<f64>, // n_var × 3
    pub variable_supports: Vec<VariableSupport>,
}

#[derive(Debug, Clone)]
pub struct VariableSupport {
    /// Global node index in the full graph.
    pub node_index: usize,
    /// Anchor reference position (world XYZ) used for relative constraints.
    pub reference_position: [f64; 3],
    /// Positive dimensionless scale for this support's optimizer coordinate map.
    pub saturation_lambda: f64,
    pub kind: VariableSupportKind,
}

#[derive(Debug, Clone)]
pub enum VariableSupportKind {
    /// Hard bounded sphere around `reference_position`.
    Sphere { radius: f64 },
    /// Per-axis relative offset box with optional axis locking.
    Roller {
        enabled: [bool; 3],
        lower: [f64; 3],
        upper: [f64; 3],
    },
    /// Hard line-segment support: x = start + t(end-start), t in (0,1).
    Rail { start: [f64; 3], end: [f64; 3] },
    /// NURBS curve support: x = C(t), t in domain.
    NurbsCurve {
        curve: nurbsbook::NurbsCurve,
        domain: [f64; 2],
        initial_t: f64,
    },
    /// NURBS surface support: x = S(u, v), u/v in domains.
    NurbsSurface {
        surface: nurbsbook::NurbsSurface,
        domain_u: [f64; 2],
        domain_v: [f64; 2],
        initial_uv: [f64; 2],
    },
}

impl VariableSupportKind {
    pub fn latent_dim(&self) -> usize {
        match self {
            Self::Sphere { .. } => 3,
            Self::Roller { enabled, .. } => enabled.iter().filter(|&&b| b).count(),
            Self::Rail { .. } => 1,
            Self::NurbsCurve { .. } => 1,
            Self::NurbsSurface { .. } => 2,
        }
    }
}

impl AnchorInfo {
    /// All anchors fixed – no movable supports.
    pub fn all_fixed(reference_positions: Array2<f64>) -> Self {
        let n = reference_positions.nrows();
        Self {
            variable_indices: Vec::new(),
            fixed_indices: (0..n).collect(),
            reference_positions,
            initial_variable_positions: Array2::zeros((0, 3)),
            variable_supports: Vec::new(),
        }
    }
}

// ─────────────────────────────────────────────────────────────
//  Self-weight parameters
// ─────────────────────────────────────────────────────────────

/// Self-weight configuration.  When present on a [`Problem`], the forward
/// solve iterates until loads and geometry are mutually consistent.
#[derive(Debug, Clone)]
pub enum SelfWeightParams {
    /// User-prescribed linear density (mass/length) per edge.
    /// `pn_sw` depends on node positions `x` only.
    Prescribed {
        linear_densities: Vec<f64>,
        gravity: [f64; 3],
        max_iters: usize,
        tolerance: f64,
        relaxation: f64,
    },
    /// Cross-section derived from forces: `A_k = |F_k| / sigma`.
    /// Linear density becomes `mu_k = rho * |q_k| * L_k / sigma`,
    /// so `pn_sw` depends on both `x` and `q`.
    Sizing {
        rho: f64,
        sigma: f64,
        gravity: [f64; 3],
        max_iters: usize,
        tolerance: f64,
        relaxation: f64,
    },
}

impl SelfWeightParams {
    pub fn gravity(&self) -> &[f64; 3] {
        match self {
            Self::Prescribed { gravity, .. } | Self::Sizing { gravity, .. } => gravity,
        }
    }
    pub fn max_iters(&self) -> usize {
        match self {
            Self::Prescribed { max_iters, .. } | Self::Sizing { max_iters, .. } => *max_iters,
        }
    }
    pub fn tolerance(&self) -> f64 {
        match self {
            Self::Prescribed { tolerance, .. } | Self::Sizing { tolerance, .. } => *tolerance,
        }
    }
    pub fn relaxation(&self) -> f64 {
        match self {
            Self::Prescribed { relaxation, .. } | Self::Sizing { relaxation, .. } => *relaxation,
        }
    }
}

// ─────────────────────────────────────────────────────────────
//  Pressure load parameters
// ─────────────────────────────────────────────────────────────

/// Face topology for pressure loads: each face is an ordered list of
/// global vertex indices forming a closed polygon.
#[derive(Debug, Clone)]
pub struct FaceTopology {
    /// `faces[f]` = ordered vertex indices for face `f`.
    pub faces: Vec<Vec<usize>>,
}

/// Pressure load configuration.  When present on a [`Problem`], the forward
/// solve iterates until pressure loads and geometry converge.
#[derive(Debug, Clone)]
pub enum PressureParams {
    /// Constant pressure along each face's outward normal.
    /// `F_f = p_f * n_f` (Newell area-weighted normal).
    Normal {
        face_topology: FaceTopology,
        pressures: Vec<f64>,
        max_iters: usize,
        tolerance: f64,
        relaxation: f64,
    },
    /// Hydrostatic pressure varying linearly with depth below a datum.
    /// `p_f = rho_fluid * g_magnitude * max(0, z_datum - z_centroid)`,
    /// applied along each face's outward normal.
    Hydrostatic {
        face_topology: FaceTopology,
        rho_fluid: f64,
        g_magnitude: f64,
        z_datum: f64,
        /// Unit "up" direction (default `[0,0,1]`).  Depth is measured
        /// opposite to this direction from the datum.
        up_direction: [f64; 3],
        max_iters: usize,
        tolerance: f64,
        relaxation: f64,
    },
    /// Uniform directional pressure proportional to projected face area.
    /// `F_f = p * max(0, n_f · d_hat) * d_hat` per vertex.
    Directional {
        face_topology: FaceTopology,
        pressures: Vec<f64>,
        /// Unit load direction (e.g. `[0,0,-1]` for gravity dead load).
        direction: [f64; 3],
        max_iters: usize,
        tolerance: f64,
        relaxation: f64,
    },
}

impl PressureParams {
    pub fn face_topology(&self) -> &FaceTopology {
        match self {
            Self::Normal { face_topology, .. }
            | Self::Hydrostatic { face_topology, .. }
            | Self::Directional { face_topology, .. } => face_topology,
        }
    }
    pub fn max_iters(&self) -> usize {
        match self {
            Self::Normal { max_iters, .. }
            | Self::Hydrostatic { max_iters, .. }
            | Self::Directional { max_iters, .. } => *max_iters,
        }
    }
    pub fn tolerance(&self) -> f64 {
        match self {
            Self::Normal { tolerance, .. }
            | Self::Hydrostatic { tolerance, .. }
            | Self::Directional { tolerance, .. } => *tolerance,
        }
    }
    pub fn relaxation(&self) -> f64 {
        match self {
            Self::Normal { relaxation, .. }
            | Self::Hydrostatic { relaxation, .. }
            | Self::Directional { relaxation, .. } => *relaxation,
        }
    }
}

// ─────────────────────────────────────────────────────────────
//  Problem definition  (immutable after construction)
// ─────────────────────────────────────────────────────────────

#[derive(Debug)]
pub struct Problem {
    pub topology: NetworkTopology,
    pub free_node_loads: Array2<f64>,      // nn_free × 3
    pub fixed_node_positions: Array2<f64>, // n_fixed × 3  (reference)
    pub anchors: AnchorInfo,
    pub objectives: Vec<Box<dyn ObjectiveTrait>>,
    pub bounds: Bounds,
    pub solver: SolverOptions,
    pub self_weight: Option<SelfWeightParams>,
    pub pressure: Option<PressureParams>,
}

// ─────────────────────────────────────────────────────────────
//  Sparsity mapping  q_k  →  A.data[] indices
// ─────────────────────────────────────────────────────────────

/// Pre-computed contribution of edge `k` to the CSC `nzval` array of A.
#[derive(Debug, Clone)]
pub struct QToNz {
    /// For each edge k: list of (nz_index_in_A_data, coefficient)
    pub entries: Vec<Vec<(usize, f64)>>,
}

// ─────────────────────────────────────────────────────────────
//  Factorization strategy
// ─────────────────────────────────────────────────────────────

/// Adaptive factorization for A(q) = Cn^T diag(q) Cn.
/// Cholesky only when bounds guarantee strictly positive q; LDL otherwise.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FactorizationStrategy {
    /// All q_k > 0: A is SPD → Cholesky.
    /// Uses AMD fill-in reduction for better sparsity in L.
    Cholesky,
    /// Non-positive or mixed-sign q: symmetric diagonal LDL without numeric
    /// pivoting. Callers must validate solve accuracy near singular pivots.
    LDL,
}

impl FactorizationStrategy {
    /// Choose strategy from the bounds on q.
    pub fn from_bounds(bounds: &Bounds) -> Self {
        let all_positive = bounds.lower.iter().all(|&lb| lb > 0.0);
        if all_positive {
            Self::Cholesky
        } else {
            Self::LDL
        }
    }
}

/// Holds a numeric LDL^T (or Cholesky) factorization.
///
/// Both variants use faer-sparse internally.
/// The Cholesky path uses AMD fill-in reduction and validates D > 0.
/// The LDL path allows indefinite D.
pub enum Factorization {
    /// SPD path: AMD-ordered, D > 0 validated
    Cholesky {
        symbolic: faer_sparse::cholesky::SymbolicCholesky<u32>,
        l_values: Vec<f64>,
    },
    /// Indefinite path: no sign constraint on D
    Ldl {
        symbolic: faer_sparse::cholesky::SymbolicCholesky<u32>,
        l_values: Vec<f64>,
    },
}

impl std::fmt::Debug for Factorization {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cholesky { .. } => write!(f, "Factorization::Cholesky(...)"),
            Self::Ldl { .. } => write!(f, "Factorization::Ldl(...)"),
        }
    }
}

impl Factorization {
    fn symbolic(&self) -> &faer_sparse::cholesky::SymbolicCholesky<u32> {
        match self {
            Self::Cholesky { symbolic, .. } | Self::Ldl { symbolic, .. } => symbolic,
        }
    }

    /// Create an initial factorization from A and the chosen strategy.
    pub fn new(
        a: &SparseColMatOwned,
        strategy: FactorizationStrategy,
        stack: &mut dyn_stack::GlobalPodBuffer,
    ) -> Result<Self, TheseusError> {
        use dyn_stack::PodStack;
        use faer_core::{Parallelism, Side};
        use faer_sparse::cholesky::{
            factorize_symbolic_cholesky, CholeskySymbolicParams, LdltRegularization,
            LltRegularization,
        };

        let a_ref = a.as_faer_ref();
        let symbolic = factorize_symbolic_cholesky(
            a_ref.symbolic(),
            Side::Upper,
            CholeskySymbolicParams::default(),
        )
        .map_err(TheseusError::from)?;

        let len_values = symbolic.len_values();
        let mut l_values = vec![0.0; len_values];

        match strategy {
            FactorizationStrategy::Cholesky => {
                let req = symbolic
                    .factorize_numeric_llt_req::<f64>(Parallelism::Rayon(0))
                    .unwrap();
                ensure_pod_stack(stack, req);
                symbolic.factorize_numeric_llt(
                    l_values.as_mut_slice(),
                    a_ref,
                    Side::Upper,
                    LltRegularization::default(),
                    Parallelism::Rayon(0),
                    PodStack::new(stack),
                )?;
                Ok(Self::Cholesky { symbolic, l_values })
            }
            FactorizationStrategy::LDL => {
                let req = symbolic
                    .factorize_numeric_ldlt_req::<f64>(false, Parallelism::Rayon(0))
                    .unwrap();
                ensure_pod_stack(stack, req);
                symbolic.factorize_numeric_ldlt(
                    l_values.as_mut_slice(),
                    a_ref,
                    Side::Upper,
                    LdltRegularization::default(),
                    Parallelism::Rayon(0),
                    PodStack::new(stack),
                );
                if l_values.iter().any(|v| !v.is_finite()) {
                    return Err(TheseusError::Linalg(
                        "LDL factorization produced non-finite values (singular or ill-conditioned matrix)"
                            .into(),
                    ));
                }
                Ok(Self::Ldl { symbolic, l_values })
            }
        }
    }

    /// Re-factor with updated numeric values (same sparsity pattern).
    pub fn update(
        &mut self,
        a: &SparseColMatOwned,
        stack: &mut dyn_stack::GlobalPodBuffer,
    ) -> Result<(), TheseusError> {
        use dyn_stack::PodStack;
        use faer_core::{Parallelism, Side};
        use faer_sparse::cholesky::{LdltRegularization, LltRegularization};

        let a_ref = a.as_faer_ref();

        match self {
            Self::Cholesky { symbolic, l_values } => {
                let req = symbolic
                    .factorize_numeric_llt_req::<f64>(Parallelism::Rayon(0))
                    .unwrap();
                ensure_pod_stack(stack, req);
                symbolic.factorize_numeric_llt(
                    l_values.as_mut_slice(),
                    a_ref,
                    Side::Upper,
                    LltRegularization::default(),
                    Parallelism::Rayon(0),
                    PodStack::new(stack),
                )?;
                Ok(())
            }
            Self::Ldl { symbolic, l_values } => {
                let req = symbolic
                    .factorize_numeric_ldlt_req::<f64>(false, Parallelism::Rayon(0))
                    .unwrap();
                ensure_pod_stack(stack, req);
                symbolic.factorize_numeric_ldlt(
                    l_values.as_mut_slice(),
                    a_ref,
                    Side::Upper,
                    LdltRegularization::default(),
                    Parallelism::Rayon(0),
                    PodStack::new(stack),
                );
                if l_values.iter().any(|v| !v.is_finite()) {
                    return Err(TheseusError::Linalg(
                        "LDL refactorization produced non-finite values (singular or ill-conditioned matrix)"
                            .into(),
                    ));
                }
                Ok(())
            }
        }
    }

    /// Solve A X = B in place. `rhs` and `x` are (n × ncols) row-major ndarray arrays.
    pub fn solve_into(
        &self,
        rhs: &Array2<f64>,
        x: &mut Array2<f64>,
        workspace: &mut [f64],
        stack: &mut dyn_stack::GlobalPodBuffer,
    ) -> Result<(), TheseusError> {
        use dyn_stack::PodStack;
        use faer_core::{Conj, Mat, Parallelism};
        use faer_sparse::cholesky::{LdltRef, LltRef};

        let n = rhs.nrows();
        let ncols = rhs.ncols();
        assert_eq!(x.nrows(), n);
        assert_eq!(x.ncols(), ncols);
        assert!(workspace.len() >= n * ncols);

        // Pack row-major ndarray RHS into column-major faer layout.
        for j in 0..ncols {
            for i in 0..n {
                workspace[i + j * n] = rhs[[i, j]];
            }
        }

        let mut mat = Mat::from_fn(n, ncols, |i, j| workspace[i + j * n]);
        let req = self.symbolic().solve_in_place_req::<f64>(ncols).unwrap();
        ensure_pod_stack(stack, req);

        match self {
            Self::Cholesky { symbolic, l_values } => {
                let llt = LltRef::new(symbolic, l_values.as_slice());
                llt.solve_in_place_with_conj(
                    Conj::No,
                    mat.as_mut(),
                    Parallelism::Rayon(0),
                    PodStack::new(stack),
                );
            }
            Self::Ldl { symbolic, l_values } => {
                let ldlt = LdltRef::new(symbolic, l_values.as_slice());
                ldlt.solve_in_place_with_conj(
                    Conj::No,
                    mat.as_mut(),
                    Parallelism::Rayon(0),
                    PodStack::new(stack),
                );
            }
        }

        for j in 0..ncols {
            for i in 0..n {
                x[[i, j]] = mat.read(i, j);
            }
        }

        Ok(())
    }

    /// Solve A x = rhs for a single RHS column.
    pub fn solve(
        &self,
        rhs: &[f64],
        workspace: &mut [f64],
        stack: &mut dyn_stack::GlobalPodBuffer,
    ) -> Result<Vec<f64>, TheseusError> {
        let n = rhs.len();
        let rhs_arr = Array2::from_shape_fn((n, 1), |(i, _)| rhs[i]);
        let mut x = Array2::zeros((n, 1));
        self.solve_into(&rhs_arr, &mut x, workspace, stack)?;
        Ok((0..n).map(|i| x[[i, 0]]).collect())
    }

    /// The strategy this factorization was built with.
    pub fn strategy(&self) -> FactorizationStrategy {
        match self {
            Self::Cholesky { .. } => FactorizationStrategy::Cholesky,
            Self::Ldl { .. } => FactorizationStrategy::LDL,
        }
    }
}

// ─────────────────────────────────────────────────────────────
//  Pre-allocated solver cache
// ─────────────────────────────────────────────────────────────

/// All mutable workspace for the forward solve, adjoint, and gradient
/// accumulation.  Built once from a [`Problem`], reused across iterations.
pub struct FdmCache {
    // ── Sparse system ──────────────────────────────────────
    /// System matrix A = Cn^T diag(q) Cn  (CSC, nn_free × nn_free).
    /// Sparsity pattern is fixed; values are updated in-place each iteration.
    pub a_matrix: SparseColMatOwned,

    /// Numeric factorization — Cholesky (SPD) or LDL (indefinite).
    /// Created on first factor, reused via `.update()` thereafter.
    pub factorization: Option<Factorization>,

    pub q_to_nz: QToNz,

    /// Start / end node of each edge (global node indices, 0-based)
    pub edge_starts: Vec<usize>,
    pub edge_ends: Vec<usize>,
    /// Global-node → free-index mapping  (`None` if fixed)
    pub node_to_free_idx: Vec<Option<usize>>,
    /// Per-node lists of incident edge indices (for reaction gradients).
    pub node_incident_edges: Vec<Vec<usize>>,

    /// Cn  (ne × nn_free)  and  Cf  (ne × nn_fixed)  stored as CSC
    pub cn: SparseColMatOwned,
    /// Precomputed Cn^T  (nn_free × ne) — topology is fixed.
    pub cn_t: SparseColMatOwned,
    pub cf: SparseColMatOwned,

    // ── Primal buffers ─────────────────────────────────────
    /// Free-node positions         (nn_free × 3, column-major)
    pub x: Array2<f64>,
    /// Adjoint variables           (nn_free × 3)
    pub lambda: Array2<f64>,
    /// dJ / d(free-node positions) (nn_free × 3)
    pub grad_x: Array2<f64>,
    /// Force densities
    pub q: Vec<f64>,
    /// dJ / dq
    pub grad_q: Vec<f64>,
    /// dJ / dNf  (all nodes × 3, only fixed rows used)
    pub grad_nf: Array2<f64>,

    // ── Derived geometry ───────────────────────────────────
    pub member_lengths: Vec<f64>,
    pub member_forces: Vec<f64>,
    pub reactions: Array2<f64>, // nn × 3

    // ── Intermediate RHS buffers ───────────────────────────
    pub cf_nf: Array2<f64>,    // ne × 3
    pub q_cf_nf: Array2<f64>,  // ne × 3
    pub pn: Array2<f64>,       // nn_free × 3  (copy of free-node loads)
    pub nf: Array2<f64>,       // nn × 3       (full node positions)
    pub nf_fixed: Array2<f64>, // nn_fixed × 3

    // ── RHS buffer (reusable for linear solve input) ───────
    pub rhs: Array2<f64>, // nn_free × 3

    // ── Factorization ──────────────────────────────────────
    pub strategy: FactorizationStrategy,

    // ── Self-weight / pressure iteration buffers ──────────
    /// Copy of the original (user-specified) free-node loads, used as the
    /// base when self-weight or pressure loads are added iteratively.
    pub pn_base: Array2<f64>, // nn_free × 3
    /// Per-edge linear density (mass/length).  In prescribed mode this is
    /// constant; in sizing mode it is updated each self-weight iteration.
    pub sw_mu: Vec<f64>,
    /// Per-edge cross-section area (populated in sizing mode, zero otherwise).
    pub cross_section_areas: Vec<f64>,
    /// Previous load vector for self-weight/pressure convergence checks.
    pub pn_prev: Array2<f64>,
    /// Scratch for softmax weight computation in variation objectives.
    pub softmax_scratch: Vec<f64>,
    pub softmax_scratch_b: Vec<f64>,
    /// Column-major buffer for faer triangular solves (nn_free × 3).
    pub solve_workspace: Vec<f64>,
    /// Reused faer stack for factorization numeric updates.
    pub factor_stack: dyn_stack::GlobalPodBuffer,
    /// Reused faer stack for triangular solves.
    pub solve_stack: dyn_stack::GlobalPodBuffer,
}

impl FdmCache {
    /// Build a fully pre-allocated cache from a [`Problem`].
    ///
    /// Returns `Err` if the incidence sparsity pattern is inconsistent.
    pub fn new(problem: &Problem) -> Result<Self, TheseusError> {
        let topo = &problem.topology;
        let ne = topo.num_edges;
        let nn = topo.num_nodes;
        let nn_free = topo.free_node_indices.len();
        let nn_fixed = topo.fixed_node_indices.len();

        // ── 1. Build A's sparsity pattern from Cn^T * Cn ──
        let cn = &topo.free_incidence; // ne × nn_free
        let cn_t = cn.transpose();
        let a_matrix = SparseColMatOwned::sparse_times_sparse(&cn_t, cn)
            .map_err(|e| TheseusError::Solver(e))?;

        // ── 2. Build q_to_nz mapping ──────────────────────
        // For each edge k, find which free nodes it touches in Cn,
        // then map those (n1, n2) pairs to indices in a_matrix.values.
        let mut edge_to_free_nodes: Vec<Vec<(usize, f64)>> = vec![Vec::new(); ne];
        for col in 0..nn_free {
            let start = cn.col_ptrs[col] as usize;
            let end_ = cn.col_ptrs[col + 1] as usize;
            for idx in start..end_ {
                let row = cn.row_indices[idx] as usize;
                let val = cn.values[idx];
                edge_to_free_nodes[row].push((col, val));
            }
        }

        let mut q_to_nz_entries: Vec<Vec<(usize, f64)>> = vec![Vec::new(); ne];
        for k in 0..ne {
            let nodes = &edge_to_free_nodes[k];
            for &(n1, v1) in nodes {
                for &(n2, v2) in nodes {
                    let nz_idx = find_nz_index(&a_matrix.col_ptrs, &a_matrix.row_indices, n1, n2)
                        .ok_or(TheseusError::SparsityMismatch {
                        edge: k,
                        row: n1,
                        col: n2,
                    })?;
                    q_to_nz_entries[k].push((nz_idx, v1 * v2));
                }
            }
        }

        // ── 3. Edge start / end from incidence ────────────
        let mut edge_starts = vec![0usize; ne];
        let mut edge_ends = vec![0usize; ne];
        let inc = &topo.incidence;
        for col in 0..nn {
            let start = inc.col_ptrs[col] as usize;
            let end_ = inc.col_ptrs[col + 1] as usize;
            for idx in start..end_ {
                let row = inc.row_indices[idx] as usize;
                let val = inc.values[idx];
                if val == -1.0 {
                    edge_starts[row] = col;
                } else if val == 1.0 {
                    edge_ends[row] = col;
                }
            }
        }

        // ── 4. node_to_free_idx ───────────────────────────
        let mut node_to_free_idx = vec![None; nn];
        for (i, &node) in topo.free_node_indices.iter().enumerate() {
            node_to_free_idx[node] = Some(i);
        }

        // ── 4b. node → incident edges (for reaction gradients) ──
        let mut node_incident_edges = vec![Vec::new(); nn];
        for k in 0..ne {
            node_incident_edges[edge_starts[k]].push(k);
            node_incident_edges[edge_ends[k]].push(k);
        }

        // ── 5. Factorization strategy ─────────────────────
        let strategy = FactorizationStrategy::from_bounds(&problem.bounds);

        // ── 6. Pre-allocate all buffers ───────────────────
        let cf = topo.fixed_incidence.clone();
        let cn_owned = topo.free_incidence.clone();
        let cn_t = cn_owned.transpose();

        let sw_mu = match &problem.self_weight {
            Some(SelfWeightParams::Prescribed {
                linear_densities, ..
            }) => linear_densities.clone(),
            _ => vec![0.0; ne],
        };

        Ok(FdmCache {
            a_matrix,
            factorization: None,
            q_to_nz: QToNz {
                entries: q_to_nz_entries,
            },
            edge_starts,
            edge_ends,
            node_to_free_idx,
            node_incident_edges,
            cn: cn_owned,
            cn_t,
            cf,
            x: Array2::zeros((nn_free, 3)),
            lambda: Array2::zeros((nn_free, 3)),
            grad_x: Array2::zeros((nn_free, 3)),
            q: vec![0.0; ne],
            grad_q: vec![0.0; ne],
            grad_nf: Array2::zeros((nn, 3)),
            member_lengths: vec![0.0; ne],
            member_forces: vec![0.0; ne],
            reactions: Array2::zeros((nn, 3)),
            cf_nf: Array2::zeros((ne, 3)),
            q_cf_nf: Array2::zeros((ne, 3)),
            pn: problem.free_node_loads.clone(),
            nf: Array2::zeros((nn, 3)),
            nf_fixed: Array2::zeros((nn_fixed, 3)),
            rhs: Array2::zeros((nn_free, 3)),
            strategy,
            pn_base: problem.free_node_loads.clone(),
            sw_mu,
            cross_section_areas: vec![0.0; ne],
            pn_prev: Array2::zeros((nn_free, 3)),
            softmax_scratch: Vec::new(),
            softmax_scratch_b: Vec::new(),
            solve_workspace: vec![0.0; nn_free * 3],
            factor_stack: dyn_stack::GlobalPodBuffer::new(dyn_stack::StackReq::empty()),
            solve_stack: dyn_stack::GlobalPodBuffer::new(dyn_stack::StackReq::empty()),
        })
    }
}

// ─────────────────────────────────────────────────────────────
//  Geometry snapshot  (read-only view after forward solve)
// ─────────────────────────────────────────────────────────────

/// Immutable snapshot of the geometry after a forward FDM solve.
/// Views borrow from `FdmCache` buffers.
pub struct GeometrySnapshot<'a> {
    pub xyz_full: &'a Array2<f64>, // nn × 3
    pub member_lengths: &'a [f64],
    pub member_forces: &'a [f64],
    pub reactions: &'a Array2<f64>, // nn × 3
}

// ─────────────────────────────────────────────────────────────
//  Optimisation state  (mutable across iterations)
// ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct OptimizationState {
    pub force_densities: Vec<f64>,
    pub variable_anchor_positions: Array2<f64>, // n_var × 3
    pub variable_anchor_latents: Vec<f64>,
    pub loss_trace: Vec<f64>,
    pub iterations: usize,
}

impl OptimizationState {
    pub fn new(q: Vec<f64>, anchors: Array2<f64>) -> Self {
        let mut latents = Vec::with_capacity(anchors.nrows() * 3);
        for i in 0..anchors.nrows() {
            latents.push(anchors[[i, 0]]);
            latents.push(anchors[[i, 1]]);
            latents.push(anchors[[i, 2]]);
        }
        Self {
            force_densities: q,
            variable_anchor_positions: anchors,
            variable_anchor_latents: latents,
            loss_trace: Vec::new(),
            iterations: 0,
        }
    }
}

// ─────────────────────────────────────────────────────────────
//  Solver result  (returned from optimize)
// ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct SolverResult {
    pub q: Vec<f64>,
    pub anchor_positions: Array2<f64>,
    pub xyz: Array2<f64>, // nn × 3
    pub member_lengths: Vec<f64>,
    pub member_forces: Vec<f64>,
    pub reactions: Array2<f64>, // nn × 3
    pub loss_trace: Vec<f64>,
    pub iterations: usize,
    pub converged: bool,
    pub termination_reason: String,
    /// Per-edge cross-section areas (populated in self-weight sizing mode).
    pub cross_section_areas: Vec<f64>,
}

// ─────────────────────────────────────────────────────────────
//  Helper: find nz index in CSC
// ─────────────────────────────────────────────────────────────

/// Given CSC col_ptrs and row_indices arrays, find the position of element (row, col)
/// in the data array.  Returns `None` if the entry is not in the sparsity pattern.
pub fn find_nz_index(
    col_ptrs: &[u32],
    row_indices: &[u32],
    row: usize,
    col: usize,
) -> Option<usize> {
    let start = col_ptrs[col] as usize;
    let end_ = col_ptrs[col + 1] as usize;
    for nz in start..end_ {
        if row_indices[nz] as usize == row {
            return Some(nz);
        }
    }
    None
}
