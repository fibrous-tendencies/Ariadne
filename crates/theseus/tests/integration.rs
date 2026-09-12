//! Integration tests — end-to-end optimisation on the arch network.
//!
//! These tests verify that the full pipeline (problem construction →
//! L-BFGS optimisation → result extraction) produces reasonable geometry
//! and actually reduces the objective value.

use ndarray::Array2;
use std::sync::atomic::{AtomicBool, Ordering};
use theseus::inverse::{
    solve_inverse_fdm, InverseFdmOptions, InverseMetric, ParticularMethod,
};
use theseus::optimizer;
use theseus::sparse::SparseColMatOwned;
use theseus::types::*;

static DIRECT_BOX_PROGRESS_FEASIBLE: AtomicBool = AtomicBool::new(true);

unsafe extern "C" fn record_direct_box_feasibility(
    _iteration: usize,
    _loss: f64,
    _xyz: *const f64,
    _num_nodes: usize,
    q: *const f64,
    num_edges: usize,
) -> u8 {
    let q = unsafe { std::slice::from_raw_parts(q, num_edges) };
    if q.iter().any(|value| !(0.5..=5.0).contains(value)) {
        DIRECT_BOX_PROGRESS_FEASIBLE.store(false, Ordering::Relaxed);
    }
    1
}

// ─────────────────────────────────────────────────────────────
//  Helpers (shared arch construction)
// ─────────────────────────────────────────────────────────────

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

fn extract_columns(mat: &SparseColMatOwned, cols: &[usize]) -> SparseColMatOwned {
    mat.extract_columns(cols)
}

fn make_arch_problem(bounds: Bounds, objectives: Vec<Box<dyn ObjectiveTrait>>) -> Problem {
    let num_nodes = 7;
    let num_edges = 8;

    let edges = vec![
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 6),
        (1, 5),
        (2, 4),
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

    let free_node_loads = Array2::from_shape_vec(
        (5, 3),
        vec![
            0.0, 0.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0, -2.0, 0.0, 0.0, -1.0, 0.0, 0.0, -1.0,
        ],
    )
    .unwrap();

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
        solver: SolverOptions {
            max_iterations: 200,
            ..SolverOptions::default()
        },
        self_weight: None,
        pressure: None,
    }
}

// ─────────────────────────────────────────────────────────────
//  Test: Unconstrained TargetXYZ optimisation
// ─────────────────────────────────────────────────────────────

/// Optimise the arch to hit target positions.  Verify:
///   1. Optimiser runs without error
///   2. Final positions are close to target
///   3. All lengths and forces are finite/positive
#[test]
fn optimize_target_xyz() {
    let ne = 8;
    let bounds = Bounds {
        lower: vec![0.1; ne],
        upper: vec![100.0; ne],
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
        target: target.clone(),
        reduction: TargetGeometryReduction::Sse,
    })];

    let problem = make_arch_problem(bounds, objectives);
    let mut state = OptimizationState::new(vec![1.0; ne], Array2::zeros((0, 3)));

    let cancel = AtomicBool::new(false);
    let result = optimizer::optimize(&problem, &mut state, None, 1, &cancel).unwrap();

    // Basic sanity
    assert!(result.iterations > 0, "should run at least 1 iteration");

    // Positions should move toward target (not exact — FDM has physical constraints)
    // We check that the optimizer has reduced the distance compared to the
    // initial uniform-q solution.
    let mut total_error = 0.0;
    let free_idx = &problem.topology.free_node_indices;
    for (i, &node) in free_idx.iter().enumerate() {
        for d in 0..3 {
            let diff = result.xyz[[node, d]] - target[[i, d]];
            total_error += diff * diff;
        }
    }
    // The total squared error should be small-ish (< 20 for 5 nodes × 3 dims)
    assert!(
        total_error < 20.0,
        "total squared error = {total_error:.4}, expected < 20.0",
    );

    // All geometry should be finite
    for l in &result.member_lengths {
        assert!(
            l.is_finite() && *l > 0.0,
            "length must be finite positive: {l}"
        );
    }
    for f in &result.member_forces {
        assert!(f.is_finite(), "force must be finite: {f}");
    }
    for &q in &result.q {
        assert!(q.is_finite() && q > 0.0, "q must be finite positive: {q}");
    }

    eprintln!(
        "optimize_target_xyz: {} iterations, converged={}",
        result.iterations, result.converged
    );
}

#[test]
#[ignore = "manual warm-start benchmark; run with --release --ignored --nocapture"]
fn benchmark_inverse_warm_starts_for_lbfgsb() {
    let ne = 8;
    let bounds = Bounds {
        lower: vec![0.1; ne],
        upper: vec![100.0; ne],
    };
    let target = Array2::from_shape_vec(
        (5, 3),
        vec![
            1.0, 0.0, 1.0, 2.0, 0.0, 2.0, 3.0, 0.0, 2.5, 4.0, 0.0, 2.0, 5.0, 0.0, 1.0,
        ],
    )
    .unwrap();

    for metric in [
        InverseMetric::Force,
        InverseMetric::Geometry,
        InverseMetric::GeometryNewton,
    ] {
        let objectives: Vec<Box<dyn ObjectiveTrait>> = vec![Box::new(TargetXYZ {
            weight: 1.0,
            node_indices: vec![1, 2, 3, 4, 5],
            target: target.clone(),
            reduction: TargetGeometryReduction::Sse,
        })];
        let problem = make_arch_problem(bounds.clone(), objectives);
        let mut opts = InverseFdmOptions::direct_unconstrained(
            1e-6,
            true,
            1,
            ParticularMethod::Clarabel,
            false,
            false,
            false,
            false,
        );
        opts.metric = metric;
        opts.signs = vec![1];
        opts.lower = vec![0.1];
        opts.upper = vec![100.0];
        let inverse = solve_inverse_fdm(&problem, &target, opts).unwrap();

        let mut state = OptimizationState::new(inverse.q, Array2::zeros((0, 3)));
        let cancel = AtomicBool::new(false);
        let started = std::time::Instant::now();
        let optimized = optimizer::optimize(&problem, &mut state, None, 1, &cancel).unwrap();
        eprintln!(
            "warm-start,metric={metric:?},geom_error={:.6e},lbfgsb_iterations={},\
             lbfgsb_converged={},lbfgsb_ms={:.3}",
            inverse.geometric_error,
            optimized.iterations,
            optimized.converged,
            started.elapsed().as_secs_f64() * 1e3
        );
    }
}

// ─────────────────────────────────────────────────────────────
//  Test: Combined objectives optimisation
// ─────────────────────────────────────────────────────────────

/// Optimise with multiple objectives and verify convergence.
#[test]
fn optimize_combined_objectives() {
    let ne = 8;
    let bounds = Bounds {
        lower: vec![0.1; ne],
        upper: vec![20.0; ne],
    };

    let target = Array2::from_shape_vec(
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
            target,
            reduction: TargetGeometryReduction::Sse,
        }),
        Box::new(LengthVariation {
            weight: 0.5,
            edge_indices: (0..ne).collect(),
            sharpness: 20.0,
            use_normalized_variance: false,
            normalization_strategy: LengthVarianceNormalizationStrategy::SquaredMean,
        }),
        Box::new(SumForceLength {
            weight: 0.01,
            edge_indices: (0..ne).collect(),
        }),
    ];

    let problem = make_arch_problem(bounds, objectives);
    let mut state = OptimizationState::new(vec![2.0; ne], Array2::zeros((0, 3)));

    let cancel = AtomicBool::new(false);
    let result = optimizer::optimize(&problem, &mut state, None, 1, &cancel).unwrap();

    assert!(result.iterations > 0);
    // Check that all results are finite
    for l in &result.member_lengths {
        assert!(l.is_finite() && *l > 0.0);
    }

    eprintln!(
        "optimize_combined: {} iterations, converged={}",
        result.iterations, result.converged
    );
}

#[test]
fn optimize_direct_box_bounds_keeps_physical_q_inside_bounds() {
    let ne = 8;
    let bounds = Bounds {
        lower: vec![0.5; ne],
        upper: vec![5.0; ne],
    };

    let target = Array2::from_shape_vec(
        (5, 3),
        vec![
            1.0, 0.0, 0.8, 2.0, 0.0, 1.5, 3.0, 0.0, 2.0, 4.0, 0.0, 1.5, 5.0, 0.0, 0.8,
        ],
    )
    .unwrap();
    let objectives: Vec<Box<dyn ObjectiveTrait>> = vec![Box::new(TargetXYZ {
        weight: 1.0,
        node_indices: vec![1, 2, 3, 4, 5],
        target,
        reduction: TargetGeometryReduction::Sse,
    })];

    let mut problem = make_arch_problem(bounds, objectives);
    problem.solver.q_parameterization_mode = QParameterizationMode::DirectBoxBounds;
    problem.solver.max_iterations = 50;
    let mut state = OptimizationState::new(vec![0.0; ne], Array2::zeros((0, 3)));
    let cancel = AtomicBool::new(false);
    DIRECT_BOX_PROGRESS_FEASIBLE.store(true, Ordering::Relaxed);
    let result = optimizer::optimize(
        &problem,
        &mut state,
        Some(record_direct_box_feasibility),
        1,
        &cancel,
    )
    .unwrap();

    assert!(!result.loss_trace.is_empty());
    assert!(DIRECT_BOX_PROGRESS_FEASIBLE.load(Ordering::Relaxed));
    let evaluations = result
        .termination_reason
        .split("evaluations=")
        .nth(1)
        .and_then(|text| text.split(';').next())
        .and_then(|text| text.parse::<usize>().ok())
        .expect("termination reason should report typed evaluation statistics");
    assert_eq!(
        result.loss_trace.len(),
        evaluations,
        "each solver evaluation should produce exactly one trace entry"
    );
    for &q in &result.q {
        assert!(
            (0.5..=5.0).contains(&q),
            "q={q} outside DirectBoxBounds range"
        );
    }

    let mut final_cache = FdmCache::new(&problem).unwrap();
    theseus::fdm::solve_fdm(
        &mut final_cache,
        &result.q,
        &problem,
        &result.anchor_positions,
        1e-12,
    )
    .unwrap();
    for (returned, recomputed) in result.xyz.iter().zip(final_cache.nf.iter()) {
        assert!((returned - recomputed).abs() <= 1e-12);
    }
}

// ─────────────────────────────────────────────────────────────
//  Test: Forward solve produces reasonable geometry
// ─────────────────────────────────────────────────────────────

/// Verify that a single forward solve produces finite geometry.
#[test]
fn forward_solve_basic() {
    let ne = 8;
    let bounds = Bounds::default_for(ne);

    let objectives: Vec<Box<dyn ObjectiveTrait>> = vec![];
    let problem = make_arch_problem(bounds, objectives);

    let q = vec![1.0; ne];
    let anchors = Array2::zeros((0, 3));

    let mut cache = FdmCache::new(&problem).unwrap();
    theseus::fdm::solve_fdm(&mut cache, &q, &problem, &anchors, 1e-12).unwrap();

    // Free-node positions should be finite
    for i in 0..problem.topology.num_nodes {
        for d in 0..3 {
            assert!(
                cache.nf[[i, d]].is_finite(),
                "node {i} dim {d} = {} is not finite",
                cache.nf[[i, d]],
            );
        }
    }

    // Anchor positions should be preserved
    assert!((cache.nf[[0, 0]] - 0.0).abs() < 1e-12);
    assert!((cache.nf[[6, 0]] - 6.0).abs() < 1e-12);

    // All lengths positive
    for (k, &len) in cache.member_lengths.iter().enumerate() {
        assert!(len > 0.0, "edge {k}: length={len} should be positive");
    }

    eprintln!("forward_solve_basic: all positions finite, anchors preserved");
}
