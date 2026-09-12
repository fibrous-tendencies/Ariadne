//! Finite-difference gradient tests for both Cholesky and LDL Factorization
//! paths.
//!
//! Tests build a 7-node arch network with **two fixed anchors** (nodes 0, 6)
//! and 5 free interior nodes (1–5), connected by 8 edges (chain + cross-
//! bracing).  `value_and_gradient` is evaluated at a point and every
//! component of the analytic gradient is compared against a central-
//! difference estimate:
//!
//!     dJ/dθ_i  ≈  [ J(θ + h eᵢ) − J(θ − h eᵢ) ] / 2h
//!
//! We verify both:
//!   (a) FactorizationStrategy::Cholesky  (all q bounds > 0)
//!   (b) FactorizationStrategy::LDL       (mixed sign q bounds)
//!
//! Multiple objective types are exercised per test for coverage.

use ndarray::Array2;
use theseus::sparse::SparseColMatOwned;
use theseus::types::*;

// ─────────────────────────────────────────────────────────────
//  Helpers: build a small test network
// ─────────────────────────────────────────────────────────────

/// Build the signed incidence matrix C (ne × nn) from edge list.
///
/// Convention:  C[e, start] = −1,  C[e, end] = +1.
fn build_incidence(edges: &[(usize, usize)], num_nodes: usize) -> SparseColMatOwned {
    let ne = edges.len();
    let mut rows = Vec::with_capacity(ne * 2);
    let mut cols = Vec::with_capacity(ne * 2);
    let mut vals = Vec::with_capacity(ne * 2);
    for (e, &(s, t)) in edges.iter().enumerate() {
        rows.push(e);
        cols.push(s);
        vals.push(-1.0);
        rows.push(e);
        cols.push(t);
        vals.push(1.0);
    }
    SparseColMatOwned::from_coo(ne, num_nodes, &rows, &cols, &vals).unwrap()
}

/// Extract columns of a CSC matrix by index (duplicated from ffi for tests).
fn extract_columns(mat: &SparseColMatOwned, cols: &[usize]) -> SparseColMatOwned {
    mat.extract_columns(cols)
}

/// A 7-node arch with two pinned supports.
///
/// Topology (side elevation, x increases right, z is up):
///
///              (3)              ← crown
///             / | \
///          (2)  |  (4)
///         / |   |   | \
///      (1)  |   |   |  (5)
///      /    |   |   |    \
///   (0)─────┴───┴───┴─────(6)   ← ground
///    ^                      ^
///  fixed                  fixed
///
/// 8 edges (chain + cross-bracing for non-trivial sparsity):
///   0→1, 1→2, 2→3, 3→4, 4→5, 5→6   (chain along the arch)
///   1→5, 2→4                          (cross-braces)
///
/// Nodes 0, 6 are fixed anchors at ground level.
/// Nodes 1–5 are free interior nodes.
fn make_arch_problem(bounds: Bounds, objectives: Vec<Box<dyn ObjectiveTrait>>) -> Problem {
    let num_nodes = 7;
    let num_edges = 8;

    let edges = vec![
        (0, 1), // 0  left footing → first free
        (1, 2), // 1
        (2, 3), // 2  → crown
        (3, 4), // 3  crown →
        (4, 5), // 4
        (5, 6), // 5  last free → right footing
        (1, 5), // 6  cross-brace lower
        (2, 4), // 7  cross-brace upper
    ];

    let free_idx: Vec<usize> = vec![1, 2, 3, 4, 5];
    let fixed_idx: Vec<usize> = vec![0, 6];

    let incidence = build_incidence(&edges, num_nodes);
    let free_inc = extract_columns(&incidence, &free_idx);
    let fixed_inc = extract_columns(&incidence, &fixed_idx);

    let topology = NetworkTopology {
        incidence,
        free_incidence: free_inc,
        fixed_incidence: fixed_inc,
        num_edges,
        num_nodes,
        free_node_indices: free_idx,
        fixed_node_indices: fixed_idx,
    };

    // Loads: gravity-like in −z for each free node
    let nn_free = 5;
    let free_node_loads = Array2::from_shape_vec(
        (nn_free, 3),
        vec![
            0.0, 0.0, -1.0, // node 1
            0.0, 0.0, -1.0, // node 2
            0.0, 0.0, -2.0, // node 3  (crown carries more load)
            0.0, 0.0, -1.0, // node 4
            0.0, 0.0, -1.0, // node 5
        ],
    )
    .unwrap();

    // Fixed-node positions:
    //   node 0  at (0, 0, 0)    left support
    //   node 6  at (6, 0, 0)    right support
    let fixed_node_positions =
        Array2::from_shape_vec((2, 3), vec![0.0, 0.0, 0.0, 6.0, 0.0, 0.0]).unwrap();

    let anchors = AnchorInfo::all_fixed(fixed_node_positions.clone());

    Problem {
        topology,
        free_node_loads,
        fixed_node_positions,
        anchors,
        objectives,
        bounds,
        solver: SolverOptions::default(),
        self_weight: None,
        pressure: None,
    }
}

fn make_variable_rail_arch_problem(objectives: Vec<Box<dyn ObjectiveTrait>>) -> Problem {
    let mut problem = make_arch_problem(
        Bounds {
            lower: vec![0.1; 8],
            upper: vec![100.0; 8],
        },
        objectives,
    );
    let reference_positions = problem.anchors.reference_positions.clone();
    let initial_variable_positions = Array2::from_shape_vec((1, 3), vec![6.0, 0.0, 0.0]).unwrap();
    problem.anchors = AnchorInfo {
        variable_indices: vec![6],
        fixed_indices: vec![0, 6],
        reference_positions,
        initial_variable_positions,
        variable_supports: vec![VariableSupport {
            node_index: 6,
            reference_position: [6.0, 0.0, 0.0],
            saturation_lambda: 1.0,
            kind: VariableSupportKind::Rail {
                start: [5.0, -1.0, 0.0],
                end: [7.0, 1.0, 0.0],
            },
        }],
    };
    problem
}

fn make_variable_nurbs_curve_arch_problem(objectives: Vec<Box<dyn ObjectiveTrait>>) -> Problem {
    let mut problem = make_arch_problem(
        Bounds {
            lower: vec![0.1; 8],
            upper: vec![100.0; 8],
        },
        objectives,
    );
    let reference_positions = problem.anchors.reference_positions.clone();
    let initial_variable_positions = Array2::from_shape_vec((1, 3), vec![6.0, 0.0, 0.0]).unwrap();
    let curve = nurbsbook::NurbsCurve::from_cartesian(
        2,
        vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        vec![[5.0, -1.0, 0.0], [6.0, 0.0, 0.0], [7.0, 1.0, 0.0]],
        vec![1.0; 3],
    )
    .unwrap();
    problem.anchors = AnchorInfo {
        variable_indices: vec![6],
        fixed_indices: vec![0, 6],
        reference_positions,
        initial_variable_positions,
        variable_supports: vec![VariableSupport {
            node_index: 6,
            reference_position: [6.0, 0.0, 0.0],
            saturation_lambda: 1.0,
            kind: VariableSupportKind::NurbsCurve {
                curve,
                domain: [0.0, 1.0],
                initial_t: 0.5,
            },
        }],
    };
    problem
}

fn make_variable_nurbs_surface_arch_problem(objectives: Vec<Box<dyn ObjectiveTrait>>) -> Problem {
    let mut problem = make_arch_problem(
        Bounds {
            lower: vec![0.1; 8],
            upper: vec![100.0; 8],
        },
        objectives,
    );
    let reference_positions = problem.anchors.reference_positions.clone();
    let initial_variable_positions = Array2::from_shape_vec((1, 3), vec![6.0, 0.0, 0.0]).unwrap();
    let surface = nurbsbook::NurbsSurface::from_cartesian(
        1,
        1,
        vec![0.0, 0.0, 1.0, 1.0],
        vec![0.0, 0.0, 1.0, 1.0],
        2,
        2,
        vec![
            [5.0, -1.0, 0.0],
            [5.0, 1.0, 0.0],
            [7.0, -1.0, 0.0],
            [7.0, 1.0, 0.0],
        ],
        vec![1.0; 4],
    )
    .unwrap();
    problem.anchors = AnchorInfo {
        variable_indices: vec![6],
        fixed_indices: vec![0, 6],
        reference_positions,
        initial_variable_positions,
        variable_supports: vec![VariableSupport {
            node_index: 6,
            reference_position: [6.0, 0.0, 0.0],
            saturation_lambda: 1.0,
            kind: VariableSupportKind::NurbsSurface {
                surface,
                domain_u: [0.0, 1.0],
                domain_v: [0.0, 1.0],
                initial_uv: [0.5, 0.5],
            },
        }],
    };
    problem
}

// ─────────────────────────────────────────────────────────────
//  Core FD test driver
// ─────────────────────────────────────────────────────────────

/// Evaluate loss-only at θ (without gradient — fresh cache each call so the
/// Factorization is clean).
fn eval_loss(
    problem: &Problem,
    theta: &[f64],
    lb: &[f64],
    ub: &[f64],
    lb_idx: &[usize],
    ub_idx: &[usize],
) -> f64 {
    let mut cache = FdmCache::new(problem).unwrap();
    let mut grad = vec![0.0; theta.len()];
    theseus::gradients::value_and_gradient(
        &mut cache, problem, theta, &mut grad, lb, ub, lb_idx, ub_idx,
    )
    .unwrap()
}

/// Central-difference gradient test.
///
/// Returns (max_abs_error, max_rel_error) over all components.
fn fd_gradient_check(problem: &Problem, theta: &[f64], h: f64, tol_abs: f64, tol_rel: f64) {
    let ne = problem.topology.num_edges;
    let n = theta.len();

    // Bound arrays (for barrier — use wide bounds so barrier is negligible)
    let lb: Vec<f64> = problem
        .bounds
        .lower
        .iter()
        .chain(std::iter::repeat(&f64::NEG_INFINITY).take(n - ne))
        .take(n)
        .copied()
        .collect();
    let ub: Vec<f64> = problem
        .bounds
        .upper
        .iter()
        .chain(std::iter::repeat(&f64::INFINITY).take(n - ne))
        .take(n)
        .copied()
        .collect();

    // For bounds penalty we need to pass the relevant indices
    let lb_idx: Vec<usize> = (0..ne).filter(|&i| lb[i].is_finite()).collect();
    let ub_idx: Vec<usize> = (0..ne).filter(|&i| ub[i].is_finite()).collect();

    // Analytic gradient
    let mut cache = FdmCache::new(problem).unwrap();
    let mut grad_analytic = vec![0.0; n];
    let _loss = theseus::gradients::value_and_gradient(
        &mut cache,
        problem,
        theta,
        &mut grad_analytic,
        &lb,
        &ub,
        &lb_idx,
        &ub_idx,
    )
    .unwrap();

    // FD gradient
    let mut grad_fd = vec![0.0; n];
    let mut theta_plus = theta.to_vec();
    let mut theta_minus = theta.to_vec();

    for i in 0..n {
        theta_plus[i] = theta[i] + h;
        theta_minus[i] = theta[i] - h;

        let f_plus = eval_loss(problem, &theta_plus, &lb, &ub, &lb_idx, &ub_idx);
        let f_minus = eval_loss(problem, &theta_minus, &lb, &ub, &lb_idx, &ub_idx);

        grad_fd[i] = (f_plus - f_minus) / (2.0 * h);

        // Restore
        theta_plus[i] = theta[i];
        theta_minus[i] = theta[i];
    }

    // Compare
    let mut max_abs = 0.0_f64;
    let mut max_rel = 0.0_f64;
    let mut worst_i = 0;
    for i in 0..n {
        let abs_err = (grad_analytic[i] - grad_fd[i]).abs();
        let denom = grad_fd[i].abs().max(grad_analytic[i].abs()).max(1e-14);
        let rel_err = abs_err / denom;

        if abs_err > max_abs {
            max_abs = abs_err;
            worst_i = i;
        }
        max_rel = max_rel.max(rel_err);
    }

    // Print diagnostics before asserting
    eprintln!("──────────────────────────────────────────────");
    eprintln!("FD gradient check  (h = {h:.1e})");
    eprintln!("  max |g_a - g_fd|  = {max_abs:.3e}  at component {worst_i}");
    eprintln!("  max relative err  = {max_rel:.3e}");
    for i in 0..n {
        let abs_err = (grad_analytic[i] - grad_fd[i]).abs();
        let denom = grad_fd[i].abs().max(grad_analytic[i].abs()).max(1e-14);
        let rel_err = abs_err / denom;
        let tag = if i < ne {
            format!("q[{i}]")
        } else {
            format!("a[{}]", i - ne)
        };
        let flag = if abs_err > tol_abs && rel_err > tol_rel {
            " <<<"
        } else {
            ""
        };
        eprintln!(
            "  {tag:>6}  analytic={:+12.6e}  fd={:+12.6e}  abs={:.2e}  rel={:.2e}{flag}",
            grad_analytic[i], grad_fd[i], abs_err, rel_err,
        );
    }
    eprintln!("──────────────────────────────────────────────");

    // Assert
    for i in 0..n {
        let abs_err = (grad_analytic[i] - grad_fd[i]).abs();
        let denom = grad_fd[i].abs().max(grad_analytic[i].abs()).max(1e-14);
        let rel_err = abs_err / denom;
        assert!(
            abs_err < tol_abs || rel_err < tol_rel,
            "Component {i}: analytic={:.8e}, fd={:.8e}, abs_err={:.3e}, rel_err={:.3e}",
            grad_analytic[i],
            grad_fd[i],
            abs_err,
            rel_err,
        );
    }
}

// ─────────────────────────────────────────────────────────────
//  Tests:  Cholesky path  (all q > 0, SPD)
// ─────────────────────────────────────────────────────────────

/// TargetXYZ objective — Cholesky path.
#[test]
fn fd_cholesky_target_xyz() {
    let ne = 8;
    let bounds = Bounds {
        lower: vec![0.1; ne], // all positive → Cholesky
        upper: vec![100.0; ne],
    };

    // Target: free nodes at arch-like positions
    let target = Array2::from_shape_vec(
        (5, 3),
        vec![
            1.0, 0.0, 1.0, // node 1
            2.0, 0.0, 2.0, // node 2
            3.0, 0.0, 2.5, // node 3  (crown)
            4.0, 0.0, 2.0, // node 4
            5.0, 0.0, 1.0, // node 5
        ],
    )
    .unwrap();

    let objectives: Vec<Box<dyn ObjectiveTrait>> = vec![Box::new(TargetXYZ {
        weight: 1.0,
        node_indices: vec![1, 2, 3, 4, 5],
        target,
        reduction: TargetGeometryReduction::Sse,
    })];

    let problem = make_arch_problem(bounds, objectives);
    assert_eq!(
        FactorizationStrategy::from_bounds(&problem.bounds),
        FactorizationStrategy::Cholesky,
    );

    let theta: Vec<f64> = vec![2.0, 3.0, 1.5, 2.5, 1.0, 3.5, 2.0, 1.8];

    fd_gradient_check(&problem, &theta, 1e-6, 1e-4, 1e-3);
}

/// TargetLength objective — Cholesky path.
#[test]
fn fd_cholesky_target_length() {
    let ne = 8;
    let bounds = Bounds {
        lower: vec![0.5; ne],
        upper: vec![50.0; ne],
    };

    let objectives: Vec<Box<dyn ObjectiveTrait>> = vec![Box::new(TargetLength {
        weight: 1.0,
        edge_indices: (0..ne).collect(),
        target: vec![1.0; ne],
    })];

    let problem = make_arch_problem(bounds, objectives);
    assert_eq!(
        FactorizationStrategy::from_bounds(&problem.bounds),
        FactorizationStrategy::Cholesky,
    );

    let theta: Vec<f64> = vec![1.0, 2.0, 1.5, 2.5, 3.0, 1.2, 2.0, 1.8];

    fd_gradient_check(&problem, &theta, 1e-6, 1e-4, 1e-3);
}

/// TargetForce objective — Cholesky path.
#[test]
fn fd_cholesky_target_force() {
    let ne = 8;
    let bounds = Bounds {
        lower: vec![0.5; ne],
        upper: vec![50.0; ne],
    };

    let objectives: Vec<Box<dyn ObjectiveTrait>> = vec![Box::new(TargetForce {
        weight: 1.0,
        edge_indices: (0..ne).collect(),
        target: vec![1.0, -0.5, 2.0, 1.5, -1.0, 2.5, 0.75, -0.25],
    })];

    let problem = make_arch_problem(bounds, objectives);
    assert_eq!(
        FactorizationStrategy::from_bounds(&problem.bounds),
        FactorizationStrategy::Cholesky,
    );

    let theta: Vec<f64> = vec![1.0, 2.0, 1.5, 2.5, 3.0, 1.2, 2.0, 1.8];

    fd_gradient_check(&problem, &theta, 1e-6, 1e-4, 1e-3);
}

/// SumForceLength objective — Cholesky path.
#[test]
fn fd_cholesky_sum_force_length() {
    let ne = 8;
    let bounds = Bounds {
        lower: vec![0.1; ne],
        upper: vec![f64::INFINITY; ne],
    };

    let objectives: Vec<Box<dyn ObjectiveTrait>> = vec![Box::new(SumForceLength {
        weight: 1.0,
        edge_indices: (0..ne).collect(),
    })];

    let problem = make_arch_problem(bounds, objectives);
    assert_eq!(
        FactorizationStrategy::from_bounds(&problem.bounds),
        FactorizationStrategy::Cholesky,
    );

    let theta: Vec<f64> = vec![2.0, 1.5, 3.0, 2.5, 1.0, 4.0, 2.0, 1.5];

    fd_gradient_check(&problem, &theta, 1e-6, 1e-4, 1e-3);
}

/// Combined objectives — Cholesky path.
#[test]
fn fd_cholesky_combined() {
    let ne = 8;
    let bounds = Bounds {
        lower: vec![0.1; ne],
        upper: vec![20.0; ne],
    };

    let target_xyz = Array2::from_shape_vec(
        (5, 3),
        vec![
            1.0, 0.0, 0.8, 2.0, 0.0, 1.5, 3.0, 0.0, 2.0, 4.0, 0.0, 1.5, 5.0, 0.0, 0.8,
        ],
    )
    .unwrap();

    let objectives: Vec<Box<dyn ObjectiveTrait>> = vec![
        Box::new(TargetXYZ {
            weight: 1.0,
            node_indices: vec![1, 2, 3, 4, 5],
            target: target_xyz,
            reduction: TargetGeometryReduction::Sse,
        }),
        Box::new(TargetLength {
            weight: 0.5,
            edge_indices: vec![0, 1, 2, 3, 4, 5],
            target: vec![1.5; 6],
        }),
        Box::new(SumForceLength {
            weight: 0.1,
            edge_indices: (0..ne).collect(),
        }),
    ];

    let problem = make_arch_problem(bounds, objectives);
    assert_eq!(
        FactorizationStrategy::from_bounds(&problem.bounds),
        FactorizationStrategy::Cholesky,
    );

    let theta: Vec<f64> = vec![1.5, 2.0, 2.5, 3.0, 1.0, 1.5, 2.2, 1.8];

    fd_gradient_check(&problem, &theta, 1e-6, 1e-4, 1e-3);
}

// ─────────────────────────────────────────────────────────────
//  Tests:  LDL path  (mixed-sign q bounds)
// ─────────────────────────────────────────────────────────────

/// TargetXYZ objective — LDL path (mixed sign bounds).
#[test]
fn fd_ldl_target_xyz() {
    let ne = 8;
    let bounds = Bounds {
        lower: vec![-5.0, 0.1, -5.0, 0.1, -5.0, 0.1, -5.0, 0.1],
        upper: vec![5.0; ne],
    };

    let target = Array2::from_shape_vec(
        (5, 3),
        vec![
            1.0, 0.0, 1.0, 2.0, 0.0, 2.0, 3.0, 0.0, 2.5, 4.0, 0.0, 2.0, 5.0, 0.0, 1.0,
        ],
    )
    .unwrap();

    let objectives: Vec<Box<dyn ObjectiveTrait>> = vec![Box::new(TargetXYZ {
        weight: 1.0,
        node_indices: vec![1, 2, 3, 4, 5],
        target,
        reduction: TargetGeometryReduction::Sse,
    })];

    let problem = make_arch_problem(bounds, objectives);
    assert_eq!(
        FactorizationStrategy::from_bounds(&problem.bounds),
        FactorizationStrategy::LDL,
    );

    // Positive q values (A is SPD at this point even under LDL strategy)
    let theta: Vec<f64> = vec![2.0, 3.0, 1.5, 2.5, 1.0, 3.5, 2.0, 1.8];

    fd_gradient_check(&problem, &theta, 1e-6, 1e-4, 1e-3);
}

/// TargetLength — LDL path.
#[test]
fn fd_ldl_target_length() {
    let ne = 8;
    let bounds = Bounds {
        lower: vec![-10.0; ne],
        upper: vec![10.0; ne],
    };

    let objectives: Vec<Box<dyn ObjectiveTrait>> = vec![Box::new(TargetLength {
        weight: 1.0,
        edge_indices: (0..ne).collect(),
        target: vec![1.0; ne],
    })];

    let problem = make_arch_problem(bounds, objectives);
    assert_eq!(
        FactorizationStrategy::from_bounds(&problem.bounds),
        FactorizationStrategy::LDL,
    );

    let theta: Vec<f64> = vec![1.0, 2.0, 1.5, 2.5, 3.0, 1.2, 2.0, 1.8];

    fd_gradient_check(&problem, &theta, 1e-6, 1e-4, 1e-3);
}

/// TargetForce — LDL path.
#[test]
fn fd_ldl_target_force() {
    let ne = 8;
    let bounds = Bounds {
        lower: vec![-10.0; ne],
        upper: vec![10.0; ne],
    };

    let objectives: Vec<Box<dyn ObjectiveTrait>> = vec![Box::new(TargetForce {
        weight: 1.0,
        edge_indices: (0..ne).collect(),
        target: vec![1.0, -0.5, 2.0, 1.5, -1.0, 2.5, 0.75, -0.25],
    })];

    let problem = make_arch_problem(bounds, objectives);
    assert_eq!(
        FactorizationStrategy::from_bounds(&problem.bounds),
        FactorizationStrategy::LDL,
    );

    let theta: Vec<f64> = vec![1.0, 2.0, 1.5, 2.5, 3.0, 1.2, 2.0, 1.8];

    fd_gradient_check(&problem, &theta, 1e-6, 1e-4, 1e-3);
}

/// SumForceLength — LDL path.
#[test]
fn fd_ldl_sum_force_length() {
    let ne = 8;
    let bounds = Bounds {
        lower: vec![-10.0; ne],
        upper: vec![10.0; ne],
    };

    let objectives: Vec<Box<dyn ObjectiveTrait>> = vec![Box::new(SumForceLength {
        weight: 1.0,
        edge_indices: (0..ne).collect(),
    })];

    let problem = make_arch_problem(bounds, objectives);
    assert_eq!(
        FactorizationStrategy::from_bounds(&problem.bounds),
        FactorizationStrategy::LDL,
    );

    let theta: Vec<f64> = vec![2.0, 1.5, 3.0, 2.5, 1.0, 4.0, 2.0, 1.5];

    fd_gradient_check(&problem, &theta, 1e-6, 1e-4, 1e-3);
}

/// Combined objectives — LDL path.
#[test]
fn fd_ldl_combined() {
    let ne = 8;
    let bounds = Bounds {
        lower: vec![-5.0, 0.1, -5.0, 0.1, -5.0, 0.1, -5.0, 0.1],
        upper: vec![10.0; ne],
    };

    let target_xyz = Array2::from_shape_vec(
        (5, 3),
        vec![
            1.0, 0.0, 0.8, 2.0, 0.0, 1.5, 3.0, 0.0, 2.0, 4.0, 0.0, 1.5, 5.0, 0.0, 0.8,
        ],
    )
    .unwrap();

    let objectives: Vec<Box<dyn ObjectiveTrait>> = vec![
        Box::new(TargetXYZ {
            weight: 1.0,
            node_indices: vec![1, 2, 3, 4, 5],
            target: target_xyz,
            reduction: TargetGeometryReduction::Sse,
        }),
        Box::new(TargetLength {
            weight: 0.5,
            edge_indices: vec![0, 1, 2, 3, 4, 5],
            target: vec![1.5; 6],
        }),
        Box::new(SumForceLength {
            weight: 0.1,
            edge_indices: (0..ne).collect(),
        }),
    ];

    let problem = make_arch_problem(bounds, objectives);
    assert_eq!(
        FactorizationStrategy::from_bounds(&problem.bounds),
        FactorizationStrategy::LDL,
    );

    let theta: Vec<f64> = vec![1.5, 2.0, 2.5, 3.0, 1.0, 1.5, 2.2, 1.8];

    fd_gradient_check(&problem, &theta, 1e-6, 1e-4, 1e-3);
}

// ─────────────────────────────────────────────────────────────
//  Tests:  Variation objectives  (smooth log-sum-exp)
// ─────────────────────────────────────────────────────────────

/// LengthVariation — Cholesky path.
#[test]
fn fd_cholesky_length_variation() {
    let ne = 8;
    let bounds = Bounds {
        lower: vec![0.1; ne],
        upper: vec![50.0; ne],
    };

    let objectives: Vec<Box<dyn ObjectiveTrait>> = vec![Box::new(LengthVariation {
        weight: 1.0,
        edge_indices: (0..ne).collect(),
        sharpness: 20.0,
        use_normalized_variance: false,
        normalization_strategy: LengthVarianceNormalizationStrategy::SquaredMean,
    })];

    let problem = make_arch_problem(bounds, objectives);
    assert_eq!(
        FactorizationStrategy::from_bounds(&problem.bounds),
        FactorizationStrategy::Cholesky,
    );

    let theta: Vec<f64> = vec![1.0, 2.0, 1.5, 2.5, 3.0, 1.2, 2.0, 1.8];

    fd_gradient_check(&problem, &theta, 1e-6, 1e-4, 1e-3);
}

/// ForceVariation — Cholesky path.
#[test]
fn fd_cholesky_force_variation() {
    let ne = 8;
    let bounds = Bounds {
        lower: vec![0.1; ne],
        upper: vec![50.0; ne],
    };

    let objectives: Vec<Box<dyn ObjectiveTrait>> = vec![Box::new(ForceVariation {
        weight: 1.0,
        edge_indices: (0..ne).collect(),
        sharpness: 20.0,
        use_normalized_variance: false,
        normalization_strategy: ForceVarianceNormalizationStrategy::SquaredMean,
    })];

    let problem = make_arch_problem(bounds, objectives);
    assert_eq!(
        FactorizationStrategy::from_bounds(&problem.bounds),
        FactorizationStrategy::Cholesky,
    );

    let theta: Vec<f64> = vec![2.0, 3.0, 1.5, 2.5, 1.0, 3.5, 2.0, 1.8];

    fd_gradient_check(&problem, &theta, 1e-6, 1e-4, 1e-3);
}

/// LengthVariation normalized variance — Cholesky path.
#[test]
fn fd_cholesky_length_variation_normalized_variance() {
    let ne = 8;
    let bounds = Bounds {
        lower: vec![0.1; ne],
        upper: vec![50.0; ne],
    };

    let objectives: Vec<Box<dyn ObjectiveTrait>> = vec![Box::new(LengthVariation {
        weight: 1.0,
        edge_indices: (0..ne).collect(),
        sharpness: 20.0,
        use_normalized_variance: true,
        normalization_strategy: LengthVarianceNormalizationStrategy::SquaredMean,
    })];

    let problem = make_arch_problem(bounds, objectives);
    let theta: Vec<f64> = vec![1.0, 2.0, 1.5, 2.5, 3.0, 1.2, 2.0, 1.8];

    fd_gradient_check(&problem, &theta, 1e-6, 1e-4, 1e-3);
}

/// ForceVariation normalized variance — Cholesky path.
#[test]
fn fd_cholesky_force_variation_normalized_variance() {
    let ne = 8;
    let bounds = Bounds {
        lower: vec![0.1; ne],
        upper: vec![50.0; ne],
    };

    let objectives: Vec<Box<dyn ObjectiveTrait>> = vec![Box::new(ForceVariation {
        weight: 1.0,
        edge_indices: (0..ne).collect(),
        sharpness: 20.0,
        use_normalized_variance: true,
        normalization_strategy: ForceVarianceNormalizationStrategy::SquaredMean,
    })];

    let problem = make_arch_problem(bounds, objectives);
    let theta: Vec<f64> = vec![2.0, 3.0, 1.5, 2.5, 1.0, 3.5, 2.0, 1.8];

    fd_gradient_check(&problem, &theta, 1e-6, 1e-4, 1e-3);
}

/// LengthVariation raw variance — Cholesky path.
#[test]
fn fd_cholesky_length_variation_raw_variance() {
    let ne = 8;
    let bounds = Bounds {
        lower: vec![0.1; ne],
        upper: vec![50.0; ne],
    };

    let objectives: Vec<Box<dyn ObjectiveTrait>> = vec![Box::new(LengthVariation {
        weight: 1.0,
        edge_indices: (0..ne).collect(),
        sharpness: 20.0,
        use_normalized_variance: true,
        normalization_strategy: LengthVarianceNormalizationStrategy::None,
    })];

    let problem = make_arch_problem(bounds, objectives);
    let theta: Vec<f64> = vec![1.0, 2.0, 1.5, 2.5, 3.0, 1.2, 2.0, 1.8];

    fd_gradient_check(&problem, &theta, 1e-6, 1e-4, 1e-3);
}

/// ForceVariation absolute normalized variance — Cholesky path.
#[test]
fn fd_cholesky_force_variation_absolute_normalized_variance() {
    let ne = 8;
    let bounds = Bounds {
        lower: vec![0.1; ne],
        upper: vec![50.0; ne],
    };

    let objectives: Vec<Box<dyn ObjectiveTrait>> = vec![Box::new(ForceVariation {
        weight: 1.0,
        edge_indices: (0..ne).collect(),
        sharpness: 20.0,
        use_normalized_variance: true,
        normalization_strategy: ForceVarianceNormalizationStrategy::AbsoluteSquaredMean,
    })];

    let problem = make_arch_problem(bounds, objectives);
    let theta: Vec<f64> = vec![2.0, 3.0, 1.5, 2.5, 1.0, 3.5, 2.0, 1.8];

    fd_gradient_check(&problem, &theta, 1e-6, 1e-4, 1e-3);
}

#[test]
fn length_raw_variance_demonstrates_scale_sensitivity() {
    let xyz = Array2::zeros((0, 3));
    let reactions = Array2::zeros((0, 3));
    let unit_lengths = vec![1.0, 2.0, 3.0];
    let scaled_lengths = vec![10.0, 20.0, 30.0];
    let forces = Vec::new();
    let edge_indices = vec![0, 1, 2];

    let normalized = LengthVariation {
        weight: 1.0,
        edge_indices: edge_indices.clone(),
        sharpness: 20.0,
        use_normalized_variance: true,
        normalization_strategy: LengthVarianceNormalizationStrategy::SquaredMean,
    };
    let raw = LengthVariation {
        weight: 1.0,
        edge_indices,
        sharpness: 20.0,
        use_normalized_variance: true,
        normalization_strategy: LengthVarianceNormalizationStrategy::None,
    };

    let unit_snap = GeometrySnapshot {
        xyz_full: &xyz,
        member_lengths: &unit_lengths,
        member_forces: &forces,
        reactions: &reactions,
    };
    let scaled_snap = GeometrySnapshot {
        xyz_full: &xyz,
        member_lengths: &scaled_lengths,
        member_forces: &forces,
        reactions: &reactions,
    };

    let normalized_unit = normalized.loss(&unit_snap);
    let normalized_scaled = normalized.loss(&scaled_snap);
    assert!((normalized_unit - normalized_scaled).abs() < 1e-12);

    let raw_unit = raw.loss(&unit_snap);
    let raw_scaled = raw.loss(&scaled_snap);
    assert!((raw_scaled / raw_unit - 100.0).abs() < 1e-12);
}

#[test]
fn force_absolute_normalization_handles_zero_mean_mixed_sign_forces() {
    let xyz = Array2::zeros((0, 3));
    let reactions = Array2::zeros((0, 3));
    let lengths = Vec::new();
    let forces = vec![-2.0, -1.0, 1.0, 2.0];
    let edge_indices = vec![0, 1, 2, 3];

    let squared_mean = ForceVariation {
        weight: 1.0,
        edge_indices: edge_indices.clone(),
        sharpness: 20.0,
        use_normalized_variance: true,
        normalization_strategy: ForceVarianceNormalizationStrategy::SquaredMean,
    };
    let absolute_mean = ForceVariation {
        weight: 1.0,
        edge_indices: edge_indices.clone(),
        sharpness: 20.0,
        use_normalized_variance: true,
        normalization_strategy: ForceVarianceNormalizationStrategy::AbsoluteSquaredMean,
    };
    let raw = ForceVariation {
        weight: 1.0,
        edge_indices,
        sharpness: 20.0,
        use_normalized_variance: true,
        normalization_strategy: ForceVarianceNormalizationStrategy::None,
    };

    let snap = GeometrySnapshot {
        xyz_full: &xyz,
        member_lengths: &lengths,
        member_forces: &forces,
        reactions: &reactions,
    };

    let squared_mean_loss = squared_mean.loss(&snap);
    let absolute_mean_loss = absolute_mean.loss(&snap);
    let raw_loss = raw.loss(&snap);

    assert!(squared_mean_loss.is_finite());
    assert!((absolute_mean_loss - (2.5 / 2.25)).abs() < 1e-12);
    assert!((raw_loss - 2.5).abs() < 1e-12);
    assert!(absolute_mean_loss < squared_mean_loss * 1e-6);
}

/// LengthVariation — LDL path (mixed bounds).
#[test]
fn fd_ldl_length_variation() {
    let ne = 8;
    let bounds = Bounds {
        lower: vec![-10.0; ne],
        upper: vec![10.0; ne],
    };

    let objectives: Vec<Box<dyn ObjectiveTrait>> = vec![Box::new(LengthVariation {
        weight: 1.0,
        edge_indices: (0..ne).collect(),
        sharpness: 20.0,
        use_normalized_variance: false,
        normalization_strategy: LengthVarianceNormalizationStrategy::SquaredMean,
    })];

    let problem = make_arch_problem(bounds, objectives);
    assert_eq!(
        FactorizationStrategy::from_bounds(&problem.bounds),
        FactorizationStrategy::LDL,
    );

    let theta: Vec<f64> = vec![1.0, 2.0, 1.5, 2.5, 3.0, 1.2, 2.0, 1.8];

    fd_gradient_check(&problem, &theta, 1e-6, 1e-4, 1e-3);
}

/// ForceVariation — LDL path (mixed bounds).
#[test]
fn fd_ldl_force_variation() {
    let ne = 8;
    let bounds = Bounds {
        lower: vec![-10.0; ne],
        upper: vec![10.0; ne],
    };

    let objectives: Vec<Box<dyn ObjectiveTrait>> = vec![Box::new(ForceVariation {
        weight: 1.0,
        edge_indices: (0..ne).collect(),
        sharpness: 20.0,
        use_normalized_variance: false,
        normalization_strategy: ForceVarianceNormalizationStrategy::SquaredMean,
    })];

    let problem = make_arch_problem(bounds, objectives);
    assert_eq!(
        FactorizationStrategy::from_bounds(&problem.bounds),
        FactorizationStrategy::LDL,
    );

    let theta: Vec<f64> = vec![2.0, 1.5, 3.0, 2.5, 1.0, 4.0, 2.0, 1.5];

    fd_gradient_check(&problem, &theta, 1e-6, 1e-4, 1e-3);
}

/// Combined with variation objectives — Cholesky path.
#[test]
fn fd_cholesky_combined_with_variation() {
    let ne = 8;
    let bounds = Bounds {
        lower: vec![0.1; ne],
        upper: vec![20.0; ne],
    };

    let target_xyz = Array2::from_shape_vec(
        (5, 3),
        vec![
            1.0, 0.0, 0.8, 2.0, 0.0, 1.5, 3.0, 0.0, 2.0, 4.0, 0.0, 1.5, 5.0, 0.0, 0.8,
        ],
    )
    .unwrap();

    let objectives: Vec<Box<dyn ObjectiveTrait>> = vec![
        Box::new(TargetXYZ {
            weight: 1.0,
            node_indices: vec![1, 2, 3, 4, 5],
            target: target_xyz,
            reduction: TargetGeometryReduction::Sse,
        }),
        Box::new(LengthVariation {
            weight: 0.5,
            edge_indices: (0..ne).collect(),
            sharpness: 20.0,
            use_normalized_variance: false,
            normalization_strategy: LengthVarianceNormalizationStrategy::SquaredMean,
        }),
        Box::new(ForceVariation {
            weight: 0.3,
            edge_indices: vec![0, 1, 2, 3, 4, 5],
            sharpness: 15.0,
            use_normalized_variance: false,
            normalization_strategy: ForceVarianceNormalizationStrategy::SquaredMean,
        }),
    ];

    let problem = make_arch_problem(bounds, objectives);
    assert_eq!(
        FactorizationStrategy::from_bounds(&problem.bounds),
        FactorizationStrategy::Cholesky,
    );

    let theta: Vec<f64> = vec![1.5, 2.0, 2.5, 3.0, 1.0, 1.5, 2.2, 1.8];

    fd_gradient_check(&problem, &theta, 1e-6, 1e-4, 1e-3);
}

// ─────────────────────────────────────────────────────────────
//  Cross-validation: both strategies give same value at same θ
// ─────────────────────────────────────────────────────────────

/// Verify that Cholesky and LDL produce identical loss and gradient
/// when given the same positive q values on the arch network.
#[test]
fn cholesky_ldl_consistency() {
    let ne = 8;

    let target = Array2::from_shape_vec(
        (5, 3),
        vec![
            1.0, 0.0, 1.0, 2.0, 0.0, 2.0, 3.0, 0.0, 2.5, 4.0, 0.0, 2.0, 5.0, 0.0, 1.0,
        ],
    )
    .unwrap();

    let make_objectives = || -> Vec<Box<dyn ObjectiveTrait>> {
        vec![
            Box::new(TargetXYZ {
                weight: 1.0,
                node_indices: vec![1, 2, 3, 4, 5],
                target: target.clone(),
                reduction: TargetGeometryReduction::Sse,
            }),
            Box::new(TargetLength {
                weight: 0.5,
                edge_indices: (0..ne).collect(),
                target: vec![1.0; ne],
            }),
        ]
    };

    let theta: Vec<f64> = vec![2.0, 3.0, 1.5, 2.5, 1.0, 3.5, 2.0, 1.8];

    // Cholesky problem
    let bounds_chol = Bounds {
        lower: vec![0.1; ne],
        upper: vec![100.0; ne],
    };
    let problem_chol = make_arch_problem(bounds_chol, make_objectives());
    assert_eq!(
        FactorizationStrategy::from_bounds(&problem_chol.bounds),
        FactorizationStrategy::Cholesky,
    );

    // LDL problem
    let bounds_ldl = Bounds {
        lower: vec![-10.0; ne],
        upper: vec![100.0; ne],
    };
    let problem_ldl = make_arch_problem(bounds_ldl, make_objectives());
    assert_eq!(
        FactorizationStrategy::from_bounds(&problem_ldl.bounds),
        FactorizationStrategy::LDL,
    );

    // Evaluate both
    let lb_chol = problem_chol.bounds.lower.clone();
    let ub_chol = problem_chol.bounds.upper.clone();
    let lb_idx_chol: Vec<usize> = (0..ne).filter(|&i| lb_chol[i].is_finite()).collect();
    let ub_idx_chol: Vec<usize> = (0..ne).filter(|&i| ub_chol[i].is_finite()).collect();

    let lb_ldl = problem_ldl.bounds.lower.clone();
    let ub_ldl = problem_ldl.bounds.upper.clone();
    let lb_idx_ldl: Vec<usize> = (0..ne).filter(|&i| lb_ldl[i].is_finite()).collect();
    let ub_idx_ldl: Vec<usize> = (0..ne).filter(|&i| ub_ldl[i].is_finite()).collect();

    let mut cache_chol = FdmCache::new(&problem_chol).unwrap();
    let mut grad_chol = vec![0.0; ne];
    let _loss_chol = theseus::gradients::value_and_gradient(
        &mut cache_chol,
        &problem_chol,
        &theta,
        &mut grad_chol,
        &lb_chol,
        &ub_chol,
        &lb_idx_chol,
        &ub_idx_chol,
    )
    .unwrap();

    let mut cache_ldl = FdmCache::new(&problem_ldl).unwrap();
    let mut grad_ldl = vec![0.0; ne];
    let _loss_ldl = theseus::gradients::value_and_gradient(
        &mut cache_ldl,
        &problem_ldl,
        &theta,
        &mut grad_ldl,
        &lb_ldl,
        &ub_ldl,
        &lb_idx_ldl,
        &ub_idx_ldl,
    )
    .unwrap();

    // Positions should match (same q, same network)
    let nn = problem_chol.topology.num_nodes;
    for i in 0..nn {
        for d in 0..3 {
            let diff = (cache_chol.nf[[i, d]] - cache_ldl.nf[[i, d]]).abs();
            assert!(
                diff < 1e-12,
                "Position mismatch at node {i} dim {d}: chol={:.8e} ldl={:.8e}",
                cache_chol.nf[[i, d]],
                cache_ldl.nf[[i, d]],
            );
        }
    }

    // Geometric loss should match (barrier differs, that's expected)
    let snap_chol = GeometrySnapshot {
        xyz_full: &cache_chol.nf,
        member_lengths: &cache_chol.member_lengths,
        member_forces: &cache_chol.member_forces,
        reactions: &cache_chol.reactions,
    };
    let snap_ldl = GeometrySnapshot {
        xyz_full: &cache_ldl.nf,
        member_lengths: &cache_ldl.member_lengths,
        member_forces: &cache_ldl.member_forces,
        reactions: &cache_ldl.reactions,
    };
    let geo_chol = theseus::objectives::total_loss(&problem_chol.objectives, &snap_chol);
    let geo_ldl = theseus::objectives::total_loss(&problem_ldl.objectives, &snap_ldl);
    assert!(
        (geo_chol - geo_ldl).abs() < 1e-12,
        "Geometric loss mismatch: chol={geo_chol:.8e} ldl={geo_ldl:.8e}",
    );

    eprintln!("Cholesky/LDL consistency on arch: positions match within 1e-12, geometric loss match within 1e-12");
}

#[test]
fn fd_variable_rail_joint_q_anchor_combined_objectives() {
    let ne = 8;
    let target_xyz = Array2::from_shape_vec(
        (5, 3),
        vec![
            1.0, 0.0, 0.8, 2.0, 0.0, 1.5, 3.0, 0.0, 2.0, 4.0, 0.0, 1.5, 5.0, 0.0, 0.8,
        ],
    )
    .unwrap();
    let rigid_target =
        Array2::from_shape_vec((3, 3), vec![1.0, 0.0, 0.8, 3.0, 0.0, 2.0, 6.2, 0.2, 0.0]).unwrap();
    let objectives: Vec<Box<dyn ObjectiveTrait>> = vec![
        Box::new(TargetXYZ {
            weight: 1.0,
            node_indices: vec![1, 2, 3, 4, 5],
            target: target_xyz,
            reduction: TargetGeometryReduction::Sse,
        }),
        Box::new(TargetLength {
            weight: 0.25,
            edge_indices: (0..ne).collect(),
            target: vec![1.25; ne],
        }),
        Box::new(SumForceLength {
            weight: 0.05,
            edge_indices: (0..ne).collect(),
        }),
        Box::new(RigidSetCompare {
            weight: 0.2,
            node_indices: vec![1, 3, 6],
            target: rigid_target,
        }),
    ];
    let problem = make_variable_rail_arch_problem(objectives);
    let theta: Vec<f64> = vec![1.5, 2.0, 2.5, 3.0, 1.4, 1.7, 2.2, 1.8, 0.2];

    fd_gradient_check(&problem, &theta, 1e-6, 5e-4, 5e-3);
}

#[test]
fn fd_variable_nurbs_curve_anchor_target_objective() {
    let target = Array2::from_shape_vec((1, 3), vec![6.25, 0.15, 0.0]).unwrap();
    let objectives: Vec<Box<dyn ObjectiveTrait>> = vec![Box::new(TargetXYZ {
        weight: 1.0,
        node_indices: vec![6],
        target,
        reduction: TargetGeometryReduction::Sse,
    })];
    let problem = make_variable_nurbs_curve_arch_problem(objectives);
    let theta: Vec<f64> = vec![1.5, 2.0, 2.5, 3.0, 1.4, 1.7, 2.2, 1.8, 0.2];

    fd_gradient_check(&problem, &theta, 1e-6, 5e-4, 5e-3);
}

#[test]
fn fd_variable_nurbs_surface_anchor_target_objective() {
    let target = Array2::from_shape_vec((1, 3), vec![6.25, 0.15, 0.0]).unwrap();
    let objectives: Vec<Box<dyn ObjectiveTrait>> = vec![Box::new(TargetXYZ {
        weight: 1.0,
        node_indices: vec![6],
        target,
        reduction: TargetGeometryReduction::Sse,
    })];
    let problem = make_variable_nurbs_surface_arch_problem(objectives);
    let theta: Vec<f64> = vec![1.5, 2.0, 2.5, 3.0, 1.4, 1.7, 2.2, 1.8, 0.2, -0.1];

    fd_gradient_check(&problem, &theta, 1e-6, 5e-4, 5e-3);
}

// ─────────────────────────────────────────────────────────────
//  Factorization strategy dispatch tests
// ─────────────────────────────────────────────────────────────

#[test]
fn strategy_all_positive() {
    let b = Bounds {
        lower: vec![0.1, 0.5],
        upper: vec![10.0, 20.0],
    };
    assert_eq!(
        FactorizationStrategy::from_bounds(&b),
        FactorizationStrategy::Cholesky
    );
}

#[test]
fn strategy_all_negative() {
    let b = Bounds {
        lower: vec![-10.0, -5.0],
        upper: vec![-0.1, -0.5],
    };
    assert_eq!(
        FactorizationStrategy::from_bounds(&b),
        FactorizationStrategy::LDL
    );
}

#[test]
fn strategy_mixed() {
    let b = Bounds {
        lower: vec![-1.0, 0.1],
        upper: vec![1.0, 5.0],
    };
    assert_eq!(
        FactorizationStrategy::from_bounds(&b),
        FactorizationStrategy::LDL
    );
}

#[test]
fn strategy_zero_lower() {
    let b = Bounds {
        lower: vec![0.0, 0.0],
        upper: vec![10.0, 10.0],
    };
    assert_eq!(
        FactorizationStrategy::from_bounds(&b),
        FactorizationStrategy::LDL
    );
}
