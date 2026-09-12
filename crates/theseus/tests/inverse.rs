//! Integration tests for inverse FDM particulars (Direct, Iterative, box, IRLS).

use ndarray::Array2;
use std::time::Instant;
use theseus::fdm;
use theseus::inverse::{
    compose_box, geometric_error_vector, solve_inverse_fdm, solve_pseudoinverse_dispatch,
    solve_spg_box, InverseFdmOptions, InverseMetric, LinearAlgebra, ParticularMethod,
    DEFAULT_MAX_OUTER,
};
use theseus::sparse::SparseColMatOwned;
use theseus::types::{AnchorInfo, Bounds, FdmCache, NetworkTopology, Problem, SolverOptions};

fn build_incidence(edges: &[(usize, usize)], num_nodes: usize) -> SparseColMatOwned {
    let mut rows = Vec::with_capacity(edges.len() * 2);
    let mut cols = Vec::with_capacity(edges.len() * 2);
    let mut values = Vec::with_capacity(edges.len() * 2);
    for (edge, &(start, end)) in edges.iter().enumerate() {
        rows.extend([edge, edge]);
        cols.extend([start, end]);
        values.extend([-1.0, 1.0]);
    }
    SparseColMatOwned::from_coo(edges.len(), num_nodes, &rows, &cols, &values).unwrap()
}

fn make_problem(
    edges: &[(usize, usize)],
    num_nodes: usize,
    free_indices: Vec<usize>,
    fixed_indices: Vec<usize>,
    loads: Array2<f64>,
    fixed_positions: Array2<f64>,
) -> Problem {
    let incidence = build_incidence(edges, num_nodes);
    let free_incidence = incidence.extract_columns(&free_indices);
    let fixed_incidence = incidence.extract_columns(&fixed_indices);
    let anchors = AnchorInfo::all_fixed(fixed_positions.clone());

    Problem {
        topology: NetworkTopology {
            incidence,
            free_incidence,
            fixed_incidence,
            num_edges: edges.len(),
            num_nodes,
            free_node_indices: free_indices,
            fixed_node_indices: fixed_indices,
        },
        free_node_loads: loads,
        fixed_node_positions: fixed_positions,
        anchors,
        objectives: Vec::new(),
        bounds: Bounds::default_for(edges.len()),
        solver: SolverOptions::default(),
        self_weight: None,
        pressure: None,
    }
}

fn triangle_problem() -> (Problem, Vec<(usize, usize)>) {
    let edges = vec![(0, 1), (1, 2)];
    let problem = make_problem(
        &edges,
        3,
        vec![1],
        vec![0, 2],
        Array2::from_shape_vec((1, 3), vec![0.0, 0.0, -1.0]).unwrap(),
        Array2::from_shape_vec((2, 3), vec![0.0, 0.0, 0.0, 2.0, 0.0, 0.0]).unwrap(),
    );
    (problem, edges)
}

fn arch_problem(selective_load: bool) -> (Problem, Vec<(usize, usize)>) {
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
    let mut load_values = vec![0.0; 15];
    if selective_load {
        load_values[2 * 3 + 2] = -2.0;
    } else {
        for i in 0..5 {
            load_values[i * 3 + 2] = -1.0;
        }
    }
    let problem = make_problem(
        &edges,
        7,
        vec![1, 2, 3, 4, 5],
        vec![0, 6],
        Array2::from_shape_vec((5, 3), load_values).unwrap(),
        Array2::from_shape_vec((2, 3), vec![0.0, 0.0, 0.0, 6.0, 0.0, 0.0]).unwrap(),
    );
    (problem, edges)
}

fn duplicate_edge_problem() -> (Problem, Array2<f64>) {
    let edges = vec![(0, 1), (0, 1)];
    let problem = make_problem(
        &edges,
        2,
        vec![0],
        vec![1],
        Array2::from_shape_vec((1, 3), vec![0.0, 0.0, -1.0]).unwrap(),
        Array2::from_shape_vec((1, 3), vec![0.0, 0.0, 0.0]).unwrap(),
    );
    let (target, _) = forward_target(&problem, &[1.0, 1.0]);
    (problem, target)
}

fn forward_target(problem: &Problem, q: &[f64]) -> (Array2<f64>, Vec<f64>) {
    let mut cache = FdmCache::new(problem).unwrap();
    fdm::solve_fdm(&mut cache, q, problem, &Array2::zeros((0, 3)), 1e-12).unwrap();
    let target = Array2::from_shape_fn((problem.topology.free_node_indices.len(), 3), |(i, d)| {
        cache.nf[[problem.topology.free_node_indices[i], d]]
    });
    (target, cache.member_lengths)
}

fn equilibrium_residual(
    problem: &Problem,
    edges: &[(usize, usize)],
    target: &Array2<f64>,
    q: &[f64],
) -> f64 {
    let mut positions = Array2::<f64>::zeros((problem.topology.num_nodes, 3));
    for (i, &node) in problem.topology.free_node_indices.iter().enumerate() {
        for d in 0..3 {
            positions[[node, d]] = target[[i, d]];
        }
    }
    for (i, &node) in problem.topology.fixed_node_indices.iter().enumerate() {
        for d in 0..3 {
            positions[[node, d]] = problem.fixed_node_positions[[i, d]];
        }
    }

    let mut residual_sq = 0.0;
    for (free_row, &node) in problem.topology.free_node_indices.iter().enumerate() {
        for d in 0..3 {
            let mut equilibrium = -problem.free_node_loads[[free_row, d]];
            for (edge, &(start, end)) in edges.iter().enumerate() {
                if node == start {
                    equilibrium -= (positions[[end, d]] - positions[[start, d]]) * q[edge];
                } else if node == end {
                    equilibrium += (positions[[end, d]] - positions[[start, d]]) * q[edge];
                }
            }
            residual_sq += equilibrium * equilibrium;
        }
    }
    residual_sq.sqrt()
}

fn fixed_reaction_norm(
    problem: &Problem,
    edges: &[(usize, usize)],
    target: &Array2<f64>,
    q: &[f64],
) -> f64 {
    let mut positions = Array2::<f64>::zeros((problem.topology.num_nodes, 3));
    for (i, &node) in problem.topology.free_node_indices.iter().enumerate() {
        for d in 0..3 {
            positions[[node, d]] = target[[i, d]];
        }
    }
    for (i, &node) in problem.topology.fixed_node_indices.iter().enumerate() {
        for d in 0..3 {
            positions[[node, d]] = problem.fixed_node_positions[[i, d]];
        }
    }

    let mut reaction_sq = 0.0;
    for &node in &problem.topology.fixed_node_indices {
        for d in 0..3 {
            let mut reaction = 0.0;
            for (edge, &(start, end)) in edges.iter().enumerate() {
                if node == start {
                    reaction -= (positions[[end, d]] - positions[[start, d]]) * q[edge];
                } else if node == end {
                    reaction += (positions[[end, d]] - positions[[start, d]]) * q[edge];
                }
            }
            reaction_sq += reaction * reaction;
        }
    }
    reaction_sq.sqrt()
}

fn tension_spg(
    problem: &Problem,
    target: &Array2<f64>,
    max_iter: usize,
    tol: f64,
) -> theseus::inverse::SpgBoxResult {
    solve_spg_box(
        problem,
        target,
        max_iter,
        tol,
        &[1],
        &[],
        &[],
        0.0,
        false,
        false,
        false,
        true,
    )
    .unwrap()
}

fn inverse_opts(
    regularization: f64,
    use_l2: bool,
    particular: ParticularMethod,
    algebra: LinearAlgebra,
    solve_for_q: bool,
) -> InverseFdmOptions {
    InverseFdmOptions {
        regularization,
        use_l2,
        max_l1_iter: 20,
        particular_method: particular,
        linear_algebra: algebra,
        enforce_zero_rx: false,
        enforce_zero_ry: false,
        enforce_zero_rz: false,
        solve_for_q,
        signs: Vec::new(),
        lower: Vec::new(),
        upper: Vec::new(),
        max_iter: 4_000,
        tol: 1e-8,
        metric: InverseMetric::Force,
        q_ref: Vec::new(),
        max_frozen_outer: 0,
        max_outer: DEFAULT_MAX_OUTER,
        cwls_damping: 1e-6,
    }
}

fn pinv(problem: &Problem, target: &Array2<f64>, use_l2: bool, augmented: bool) -> Vec<f64> {
    let method = if augmented {
        ParticularMethod::Augmented
    } else {
        ParticularMethod::Gram
    };
    solve_pseudoinverse_dispatch(
        problem, target, 1e-10, use_l2, 30, method, false, false, false, true,
    )
    .unwrap()
}

#[test]
fn round_trip_recovers_positive_force_densities_with_both_solvers() {
    let (problem, edges) = triangle_problem();
    let expected = vec![2.0, 3.0];
    let (target, _) = forward_target(&problem, &expected);

    let pinv_q = pinv(&problem, &target, true, false);
    let nnls = tension_spg(&problem, &target, 2_000, 1e-10);

    for (actual, expected) in pinv_q.iter().zip(&expected) {
        assert!((actual - expected).abs() < 1e-7, "pinv q={pinv_q:?}");
    }
    for (actual, expected) in nnls.q.iter().zip(&expected) {
        assert!((actual - expected).abs() < 1e-7, "SPG q={:?}", nnls.q);
    }
    assert!(nnls.converged);
    assert!((1..=2_000).contains(&nnls.iterations));
    assert!(equilibrium_residual(&problem, &edges, &target, &nnls.q) < 1e-8);
}

#[test]
fn selective_node_loads_are_supported_by_both_solvers() {
    let (problem, _) = arch_problem(true);
    let (target, _) = forward_target(&problem, &[2.0; 8]);

    let pinv_q = pinv(&problem, &target, true, false);
    let nnls = tension_spg(&problem, &target, 4_000, 1e-8);

    assert_eq!(pinv_q.len(), 8);
    assert!(pinv_q.iter().all(|q| q.is_finite()));
    assert_eq!(nnls.q.len(), 8);
    assert!(nnls.q.iter().all(|q| q.is_finite() && *q >= 0.0));
}

#[test]
fn pseudoinverse_normal_and_augmented_l2_agree() {
    let (problem, _) = arch_problem(false);
    let (target, _) = forward_target(&problem, &[1.5; 8]);

    let normal = pinv(&problem, &target, true, false);
    let augmented = pinv(&problem, &target, true, true);

    for (left, right) in normal.iter().zip(&augmented) {
        assert!((left - right).abs() < 1e-5, "{left} != {right}");
    }
}

#[test]
fn pseudoinverse_l1_and_solve_for_force_return_finite_values() {
    let (problem, _) = arch_problem(false);
    let (target, lengths) = forward_target(&problem, &[1.0; 8]);

    let l1 = pinv(&problem, &target, false, false);
    let solve_for_force = solve_pseudoinverse_dispatch(
        &problem,
        &target,
        1e-10,
        true,
        20,
        ParticularMethod::Gram,
        false,
        false,
        false,
        false,
    )
    .unwrap();

    assert!(l1.iter().all(|q| q.is_finite()));
    assert!(solve_for_force.iter().all(|q| q.is_finite()));
    for (q, length) in solve_for_force.iter().zip(lengths) {
        assert!((q * length).is_finite());
    }
}

#[test]
fn pseudoinverse_solve_for_force_rejects_degenerate_target_edges() {
    let (problem, _) = triangle_problem();
    let target = Array2::from_shape_vec((1, 3), vec![0.0, 0.0, 0.0]).unwrap();

    let result = solve_pseudoinverse_dispatch(
        &problem,
        &target,
        1e-6,
        true,
        20,
        ParticularMethod::Gram,
        false,
        false,
        false,
        false,
    );

    assert!(result.is_err());
}

#[test]
fn pseudoinverse_reaction_constraints_reduce_reaction_norm() {
    let (problem, edges) = arch_problem(false);
    let (target, _) = forward_target(&problem, &[1.0; 8]);

    let unconstrained = pinv(&problem, &target, true, false);
    let constrained = solve_pseudoinverse_dispatch(
        &problem,
        &target,
        1e-8,
        true,
        20,
        ParticularMethod::Gram,
        true,
        true,
        true,
        true,
    )
    .unwrap();

    assert_eq!(constrained.len(), 8);
    assert!(constrained.iter().all(|q| q.is_finite()));
    let unconstrained_reaction = fixed_reaction_norm(&problem, &edges, &target, &unconstrained);
    let constrained_reaction = fixed_reaction_norm(&problem, &edges, &target, &constrained);
    assert!(
        constrained_reaction < unconstrained_reaction,
        "reaction rows should reduce the target-geometry reaction norm: \
         constrained={constrained_reaction}, unconstrained={unconstrained_reaction}"
    );
}

#[test]
fn sparse_qr_particular_agrees_with_gram_on_tall_full_rank_fixture() {
    let (problem, _) = arch_problem(false);
    let (target, _) = forward_target(&problem, &[1.25; 8]);

    let gram = pinv(&problem, &target, true, false);
    let qr = solve_pseudoinverse_dispatch(
        &problem,
        &target,
        0.0,
        true,
        20,
        ParticularMethod::SparseQr,
        false,
        false,
        false,
        true,
    )
    .unwrap();

    assert_eq!(qr.len(), gram.len());
    for (left, right) in gram.iter().zip(&qr) {
        assert!(
            (left - right).abs() < 1e-5,
            "sparse QR particular {right} disagreed with Gram {left}"
        );
    }
}

#[test]
fn sparse_qr_particular_agrees_with_unregularized_gram_on_tall_full_rank_fixture() {
    let (problem, _) = arch_problem(false);
    let (target, _) = forward_target(&problem, &[1.25; 8]);

    let gram = solve_pseudoinverse_dispatch(
        &problem,
        &target,
        0.0,
        true,
        20,
        ParticularMethod::Gram,
        false,
        false,
        false,
        true,
    )
    .unwrap();
    let qr = solve_pseudoinverse_dispatch(
        &problem,
        &target,
        0.0,
        true,
        20,
        ParticularMethod::SparseQr,
        false,
        false,
        false,
        true,
    )
    .unwrap();

    assert_eq!(qr.len(), gram.len());
    for (left, right) in gram.iter().zip(&qr) {
        assert!(
            (left - right).abs() < 1e-5,
            "sparse QR particular {right} disagreed with unregularized Gram {left}"
        );
    }
}

#[test]
fn spg_tension_is_nonnegative_and_beats_clipped_pseudoinverse_residual() {
    let (problem, edges) = arch_problem(false);
    let target = Array2::from_shape_vec(
        (5, 3),
        vec![
            1.0, 0.0, 1.0, 2.0, 0.0, 2.2, 3.0, 0.0, 1.4, 4.0, 0.0, 2.0, 5.0, 0.0, 0.8,
        ],
    )
    .unwrap();

    let unconstrained = pinv(&problem, &target, true, false);
    let clipped: Vec<f64> = unconstrained.iter().map(|q| q.max(0.0)).collect();
    let nnls = tension_spg(&problem, &target, 10_000, 1e-8);

    assert!(nnls.q.iter().all(|q| q.is_finite() && *q >= 0.0));
    let nnls_residual = equilibrium_residual(&problem, &edges, &target, &nnls.q);
    let clipped_residual = equilibrium_residual(&problem, &edges, &target, &clipped);
    assert!(
        nnls_residual <= clipped_residual + 1e-6,
        "SPG residual {nnls_residual} exceeded clipped Pinv residual {clipped_residual}"
    );
}

fn max_abs(values: &[f64]) -> f64 {
    values.iter().map(|v| v.abs()).fold(0.0_f64, f64::max)
}

fn near_flat_problem() -> (Problem, Array2<f64>) {
    let edges = vec![(0, 1), (1, 2)];
    let problem = make_problem(
        &edges,
        3,
        vec![1],
        vec![0, 2],
        Array2::from_shape_vec((1, 3), vec![0.0, 0.0, -1.0]).unwrap(),
        Array2::from_shape_vec((2, 3), vec![0.0, 0.0, 0.0, 2.0, 0.0, 0.0]).unwrap(),
    );
    let target = Array2::from_shape_vec((1, 3), vec![1.0, 0.0, -1e-4]).unwrap();
    (problem, target)
}

#[test]
fn compose_box_intersects_signs_and_errors_on_empty_interval() {
    let box_plus = compose_box(2, &[1], &[], &[]).unwrap();
    assert!(box_plus.has_finite());
    assert_eq!(box_plus.lower, vec![0.0, 0.0]);
    assert!(box_plus.upper.iter().all(|u| *u == f64::INFINITY));

    let box_minus = compose_box(1, &[-1], &[-2.0], &[0.5]).unwrap();
    assert_eq!(box_minus.lower[0], -2.0);
    assert_eq!(box_minus.upper[0], 0.0);

    let mixed = compose_box(2, &[1, -1], &[0.1, -5.0], &[4.0, 0.0]).unwrap();
    assert_eq!(mixed.lower, vec![0.1, -5.0]);
    assert_eq!(mixed.upper, vec![4.0, 0.0]);

    assert!(compose_box(1, &[1], &[1.0], &[-0.5]).is_err());
    assert!(!compose_box(2, &[], &[], &[]).unwrap().has_finite());
}

#[test]
fn compression_and_mixed_signs_on_spg_and_clarabel() {
    let (problem, _) = arch_problem(false);
    let (target, _) = forward_target(&problem, &[1.5; 8]);

    for algebra in [LinearAlgebra::Direct, LinearAlgebra::Iterative] {
        let mut compression = inverse_opts(0.0, true, ParticularMethod::Augmented, algebra, true);
        compression.signs = vec![-1];
        compression.max_iter = 8_000;
        compression.tol = 1e-7;
        let result = solve_inverse_fdm(&problem, &target, compression).unwrap();
        assert!(
            result.q.iter().all(|q| *q <= 1e-8),
            "compression {:?} {:?}",
            algebra,
            result.q
        );

        let mut mixed = inverse_opts(0.0, true, ParticularMethod::Augmented, algebra, true);
        mixed.signs = vec![1, 1, -1, -1, 1, 1, 0, 0];
        mixed.max_iter = 8_000;
        mixed.tol = 1e-7;
        let mixed_result = solve_inverse_fdm(&problem, &target, mixed).unwrap();
        assert_eq!(mixed_result.q.len(), 8);
        assert!(mixed_result.q[0] >= -1e-8);
        assert!(mixed_result.q[1] >= -1e-8);
        assert!(mixed_result.q[2] <= 1e-8);
        assert!(mixed_result.q[3] <= 1e-8);
        assert!(mixed_result.q.iter().all(|q| q.is_finite()));
    }
}

#[test]
fn lower_and_upper_bounds_are_respected_by_both_engines() {
    let (problem, _) = triangle_problem();
    let (target, _) = forward_target(&problem, &[2.0, 3.0]);
    for algebra in [LinearAlgebra::Direct, LinearAlgebra::Iterative] {
        let mut opts = inverse_opts(0.0, true, ParticularMethod::Augmented, algebra, true);
        opts.lower = vec![0.5];
        opts.upper = vec![4.0];
        opts.max_iter = 4_000;
        let result = solve_inverse_fdm(&problem, &target, opts).unwrap();
        assert!(result
            .q
            .iter()
            .all(|q| (0.5 - 1e-6..=4.0 + 1e-6).contains(q)));
    }
}

#[test]
fn near_flat_box_caps_unconstrained_particular() {
    let (problem, target) = near_flat_problem();
    let unconstrained = solve_inverse_fdm(
        &problem,
        &target,
        inverse_opts(
            0.0,
            true,
            ParticularMethod::Augmented,
            LinearAlgebra::Direct,
            false,
        ),
    )
    .unwrap();
    let unconstrained_max = max_abs(&unconstrained.q);
    assert!(
        unconstrained_max > 50.0,
        "expected a large unconstrained particular, got {unconstrained_max}"
    );

    for algebra in [LinearAlgebra::Direct, LinearAlgebra::Iterative] {
        let mut opts = inverse_opts(0.0, true, ParticularMethod::Augmented, algebra, false);
        opts.upper = vec![10.0];
        opts.max_iter = 8_000;
        opts.tol = 1e-7;
        let boxed = solve_inverse_fdm(&problem, &target, opts).unwrap();
        // Public bounds are on q even though Stage 1 solves member force.
        let boxed_max = max_abs(&boxed.q);
        assert!(
            boxed_max < unconstrained_max * 0.5,
            "{algebra:?} box did not cap |q|: boxed={boxed_max} unconstrained={unconstrained_max} q={:?}",
            boxed.q
        );
        assert!(
            boxed.q.iter().all(|q| q.abs() <= 10.0 + 1e-4),
            "{algebra:?} q={:?}",
            boxed.q
        );
    }
}

#[test]
fn clarabel_and_spg_agree_on_a_simple_box() {
    let (problem, _) = triangle_problem();
    let (target, _) = forward_target(&problem, &[2.0, 3.0]);
    let mut clarabel = inverse_opts(
        0.0,
        true,
        ParticularMethod::Augmented,
        LinearAlgebra::Direct,
        true,
    );
    clarabel.signs = vec![1];
    let mut spg = inverse_opts(
        0.0,
        true,
        ParticularMethod::Augmented,
        LinearAlgebra::Iterative,
        true,
    );
    spg.signs = vec![1];
    spg.max_iter = 8_000;
    spg.tol = 1e-8;
    let left = solve_inverse_fdm(&problem, &target, clarabel).unwrap();
    let right = solve_inverse_fdm(&problem, &target, spg).unwrap();
    assert_eq!(left.q.len(), right.q.len());
    for (a, b) in left.q.iter().zip(&right.q) {
        assert!((a - b).abs() < 1e-3, "Clarabel {a} vs SPG {b}");
    }
}

#[test]
fn unconstrained_lsqr_agrees_with_qr_on_tall_full_rank_arch() {
    let (problem, _) = arch_problem(false);
    let (target, _) = forward_target(&problem, &[1.25; 8]);
    let qr = solve_inverse_fdm(
        &problem,
        &target,
        inverse_opts(
            0.0,
            true,
            ParticularMethod::SparseQr,
            LinearAlgebra::Direct,
            true,
        ),
    )
    .unwrap();
    let lsqr = solve_inverse_fdm(
        &problem,
        &target,
        inverse_opts(
            0.0,
            true,
            ParticularMethod::Augmented,
            LinearAlgebra::Iterative,
            true,
        ),
    )
    .unwrap();
    assert_eq!(qr.q.len(), lsqr.q.len());
    for (a, b) in qr.q.iter().zip(&lsqr.q) {
        assert!((a - b).abs() < 1e-5, "QR {a} vs LSQR {b}");
    }
}

#[test]
fn irls_runs_on_every_inner_and_stays_finite() {
    let (problem, _) = arch_problem(false);
    let (target, _) = forward_target(&problem, &[1.0; 8]);
    let cases = [
        (
            ParticularMethod::Augmented,
            LinearAlgebra::Direct,
            Vec::new(),
            Vec::new(),
        ),
        (
            ParticularMethod::Gram,
            LinearAlgebra::Direct,
            Vec::new(),
            Vec::new(),
        ),
        (
            ParticularMethod::SparseQr,
            LinearAlgebra::Direct,
            Vec::new(),
            Vec::new(),
        ),
        (
            ParticularMethod::Augmented,
            LinearAlgebra::Iterative,
            Vec::new(),
            Vec::new(),
        ),
        (
            ParticularMethod::Augmented,
            LinearAlgebra::Direct,
            vec![1],
            Vec::new(),
        ),
        (
            ParticularMethod::Augmented,
            LinearAlgebra::Iterative,
            vec![1],
            Vec::new(),
        ),
    ];
    for (particular, algebra, signs, lower) in cases {
        // Direct augmented IRLS uses Tikhonov here because strict Direct MP
        // intentionally rejects a singular zero-regularization weighted saddle.
        let lambda =
            if particular == ParticularMethod::Augmented && algebra == LinearAlgebra::Direct {
                1.0
            } else {
                0.0
            };
        let mut opts = inverse_opts(lambda, false, particular, algebra, true);
        opts.signs = signs;
        opts.lower = lower;
        opts.max_l1_iter = 8;
        opts.max_iter = 2_000;
        let result = solve_inverse_fdm(&problem, &target, opts).unwrap();
        assert!(
            result.q.iter().all(|q| q.is_finite()),
            "IRLS {particular:?} {algebra:?} produced non-finite q {:?}",
            result.q
        );
    }
}

#[test]
fn reaction_rows_reduce_reaction_norm_on_clarabel_and_spg() {
    let (problem, edges) = arch_problem(false);
    let (target, _) = forward_target(&problem, &[1.0; 8]);
    for algebra in [LinearAlgebra::Direct, LinearAlgebra::Iterative] {
        let mut free = inverse_opts(1e-8, true, ParticularMethod::Augmented, algebra, true);
        free.signs = vec![1];
        free.max_iter = 8_000;
        let mut pinned = free.clone();
        pinned.enforce_zero_rx = true;
        pinned.enforce_zero_ry = true;
        pinned.enforce_zero_rz = true;
        let unconstrained = solve_inverse_fdm(&problem, &target, free).unwrap();
        let constrained = solve_inverse_fdm(&problem, &target, pinned).unwrap();
        let unconstrained_reaction =
            fixed_reaction_norm(&problem, &edges, &target, &unconstrained.q);
        let constrained_reaction = fixed_reaction_norm(&problem, &edges, &target, &constrained.q);
        assert!(
            constrained_reaction < unconstrained_reaction,
            "{algebra:?} reaction rows should reduce the target-geometry reaction norm: \
             constrained={constrained_reaction}, unconstrained={unconstrained_reaction}"
        );
    }
}

#[test]
fn particular_method_ffi_mapping_includes_clarabel() {
    assert_eq!(
        ParticularMethod::try_from(0).unwrap(),
        ParticularMethod::Gram
    );
    assert_eq!(
        ParticularMethod::try_from(1).unwrap(),
        ParticularMethod::Augmented
    );
    assert_eq!(
        ParticularMethod::try_from(2).unwrap(),
        ParticularMethod::SparseQr
    );
    assert_eq!(
        ParticularMethod::try_from(3).unwrap(),
        ParticularMethod::Clarabel
    );
    assert!(ParticularMethod::try_from(4).is_err());
}

#[test]
fn unconstrained_clarabel_matches_tikhonov_and_solves_lambda_zero() {
    let (problem, edges) = arch_problem(false);
    let (target, _) = forward_target(&problem, &[1.25; 8]);
    let lambda = 1e-5;
    let clarabel = solve_inverse_fdm(
        &problem,
        &target,
        inverse_opts(
            lambda,
            true,
            ParticularMethod::Clarabel,
            LinearAlgebra::Direct,
            true,
        ),
    )
    .unwrap();
    let tikhonov = solve_inverse_fdm(
        &problem,
        &target,
        inverse_opts(
            lambda,
            true,
            ParticularMethod::Augmented,
            LinearAlgebra::Direct,
            true,
        ),
    )
    .unwrap();
    for (left, right) in clarabel.q.iter().zip(&tikhonov.q) {
        assert!(
            (left - right).abs() < 2e-4,
            "Clarabel {left} vs Tikhonov {right}"
        );
    }

    let zero = solve_inverse_fdm(
        &problem,
        &target,
        inverse_opts(
            0.0,
            true,
            ParticularMethod::Clarabel,
            LinearAlgebra::Direct,
            true,
        ),
    )
    .unwrap();
    assert!(equilibrium_residual(&problem, &edges, &target, &zero.q) < 1e-6);
}

#[test]
fn direct_mp_errors_on_singular_saddle_but_zero_lambda_lsqr_is_minimum_norm() {
    let (problem, target) = duplicate_edge_problem();
    let direct = solve_inverse_fdm(
        &problem,
        &target,
        inverse_opts(
            0.0,
            true,
            ParticularMethod::Augmented,
            LinearAlgebra::Direct,
            true,
        ),
    )
    .unwrap_err();
    let message = direct.to_string();
    assert!(message.contains("Iterative LSQR"), "{message}");
    assert!(message.contains("lambda=0"), "{message}");

    let iterative = solve_inverse_fdm(
        &problem,
        &target,
        inverse_opts(
            0.0,
            true,
            ParticularMethod::Gram,
            LinearAlgebra::Iterative,
            true,
        ),
    )
    .unwrap();
    assert!(iterative.converged);
    assert!(iterative.iterations > 0);
    assert!((iterative.q[0] - iterative.q[1]).abs() < 1e-9);
    assert!((iterative.q[0] - 1.0).abs() < 1e-7, "{:?}", iterative.q);
}

#[test]
fn direct_mp_matches_zero_lambda_lsqr_when_saddle_is_nonsingular() {
    let (problem, target) = near_flat_problem();
    let direct = solve_inverse_fdm(
        &problem,
        &target,
        inverse_opts(
            0.0,
            true,
            ParticularMethod::Augmented,
            LinearAlgebra::Direct,
            false,
        ),
    )
    .unwrap();
    let iterative = solve_inverse_fdm(
        &problem,
        &target,
        inverse_opts(
            0.0,
            true,
            ParticularMethod::Clarabel,
            LinearAlgebra::Iterative,
            false,
        ),
    )
    .unwrap();
    for (mp, lsqr) in direct.q.iter().zip(&iterative.q) {
        assert!(
            (mp - lsqr).abs() < 1e-8 * lsqr.abs().max(1.0),
            "Direct MP {mp} vs LSQR {lsqr}"
        );
    }
}

#[test]
fn regularized_lsqr_matches_direct_solvers_and_reports_iterations() {
    let (problem, _) = arch_problem(false);
    let (target, _) = forward_target(&problem, &[1.25; 8]);
    let lambda = 1e-4;
    let iterative = solve_inverse_fdm(
        &problem,
        &target,
        inverse_opts(
            lambda,
            true,
            ParticularMethod::SparseQr,
            LinearAlgebra::Iterative,
            true,
        ),
    )
    .unwrap();
    let tikhonov = solve_inverse_fdm(
        &problem,
        &target,
        inverse_opts(
            lambda,
            true,
            ParticularMethod::Augmented,
            LinearAlgebra::Direct,
            true,
        ),
    )
    .unwrap();
    let gram = solve_inverse_fdm(
        &problem,
        &target,
        inverse_opts(
            lambda,
            true,
            ParticularMethod::Gram,
            LinearAlgebra::Direct,
            true,
        ),
    )
    .unwrap();
    assert!(iterative.converged);
    assert!(iterative.iterations > 1);
    for ((lsqr, saddle), normal) in iterative.q.iter().zip(&tikhonov.q).zip(&gram.q) {
        assert!(
            (lsqr - saddle).abs() < 2e-5,
            "LSQR {lsqr} vs saddle {saddle}"
        );
        assert!((lsqr - normal).abs() < 2e-5, "LSQR {lsqr} vs Gram {normal}");
    }
}

#[test]
fn iterative_ignores_every_direct_particular_selection() {
    let (problem, _) = arch_problem(false);
    let (target, _) = forward_target(&problem, &[1.25; 8]);
    let methods = [
        ParticularMethod::Gram,
        ParticularMethod::Augmented,
        ParticularMethod::SparseQr,
        ParticularMethod::Clarabel,
    ];
    let baseline = solve_inverse_fdm(
        &problem,
        &target,
        inverse_opts(1e-5, true, methods[0], LinearAlgebra::Iterative, true),
    )
    .unwrap();
    for method in methods.into_iter().skip(1) {
        let result = solve_inverse_fdm(
            &problem,
            &target,
            inverse_opts(1e-5, true, method, LinearAlgebra::Iterative, true),
        )
        .unwrap();
        assert_eq!(result.iterations, baseline.iterations);
        for (left, right) in result.q.iter().zip(&baseline.q) {
            assert!(
                (left - right).abs() < 1e-12,
                "{method:?}: {left} vs {right}"
            );
        }
    }
}

#[test]
fn regularization_materially_changes_scaled_lsqr_fixture() {
    let (problem, target) = near_flat_problem();
    let zero = solve_inverse_fdm(
        &problem,
        &target,
        inverse_opts(
            0.0,
            true,
            ParticularMethod::Gram,
            LinearAlgebra::Iterative,
            false,
        ),
    )
    .unwrap();
    let regularized = solve_inverse_fdm(
        &problem,
        &target,
        inverse_opts(
            1e-2,
            true,
            ParticularMethod::Gram,
            LinearAlgebra::Iterative,
            false,
        ),
    )
    .unwrap();
    assert!(
        max_abs(&regularized.q) < 0.5 * max_abs(&zero.q),
        "lambda did not materially change LSQR: zero={:?}, regularized={:?}",
        zero.q,
        regularized.q
    );
}

#[test]
fn sparse_qr_rejects_wide_and_rank_deficient_systems() {
    let (rank_deficient, target) = duplicate_edge_problem();
    let error = solve_inverse_fdm(
        &rank_deficient,
        &target,
        inverse_opts(
            0.0,
            true,
            ParticularMethod::SparseQr,
            LinearAlgebra::Direct,
            true,
        ),
    )
    .unwrap_err()
    .to_string();
    assert!(
        error.contains("full column rank") || error.contains("rows >= columns"),
        "{error}"
    );
}

fn inverse_benchmark_fixture(edge_count: usize) -> (Problem, Array2<f64>) {
    // Independent struts give all methods the same full-column-rank sparse
    // equilibrium problem, avoiding method-specific "unsupported" timings.
    let nodes = 2 * edge_count;
    let edges: Vec<(usize, usize)> = (0..edge_count).map(|i| (i, edge_count + i)).collect();
    let free: Vec<usize> = (0..edge_count).collect();
    let fixed: Vec<usize> = (edge_count..nodes).collect();
    let target = Array2::from_shape_fn((edge_count, 3), |(i, d)| match d {
        0 => 0.01 * i as f64 + 1.0,
        1 => (0.13 * i as f64).sin(),
        _ => (0.17 * i as f64).cos(),
    });
    let fixed_positions = Array2::from_shape_fn((edge_count, 3), |(i, d)| match d {
        0 => 0.01 * i as f64,
        1 => (0.13 * i as f64).sin() - 0.2,
        _ => (0.17 * i as f64).cos() + 0.1,
    });
    let mut problem = make_problem(
        &edges,
        nodes,
        free,
        fixed,
        Array2::zeros((target.nrows(), 3)),
        fixed_positions,
    );
    let system = theseus::nullspace::EquilibriumSystem::assemble(
        &problem,
        &target,
        theseus::nullspace::EquilibriumUnknown::ForceDensity,
        false,
        false,
        false,
    )
    .unwrap();
    let load = system.a.matvec(&vec![1.0; edges.len()]);
    for d in 0..3 {
        for i in 0..target.nrows() {
            problem.free_node_loads[[i, d]] = load[d * target.nrows() + i];
        }
    }
    (problem, target)
}

#[test]
#[ignore = "manual release benchmark; run with --release --ignored --nocapture"]
fn benchmark_inverse_approximately_800_edges() {
    let (problem, target) = inverse_benchmark_fixture(800);
    let edge_count = problem.topology.num_edges;
    assert_eq!(edge_count, 800);
    let assembly_start = Instant::now();
    let _ = theseus::nullspace::EquilibriumSystem::assemble(
        &problem,
        &target,
        theseus::nullspace::EquilibriumUnknown::ForceDensity,
        false,
        false,
        false,
    )
    .unwrap();
    let assembly = assembly_start.elapsed();
    eprintln!(
        "inverse,assembly,edges={edge_count},ms={:.3}",
        assembly.as_secs_f64() * 1e3
    );

    let cases = [
        (
            "mp",
            0.0,
            ParticularMethod::Augmented,
            LinearAlgebra::Direct,
        ),
        (
            "tikhonov",
            1e-6,
            ParticularMethod::Augmented,
            LinearAlgebra::Direct,
        ),
        ("qr", 0.0, ParticularMethod::SparseQr, LinearAlgebra::Direct),
        ("gram", 1e-6, ParticularMethod::Gram, LinearAlgebra::Direct),
        (
            "lsqr-zero",
            0.0,
            ParticularMethod::Clarabel,
            LinearAlgebra::Iterative,
        ),
        (
            "lsqr-regularized",
            1e-6,
            ParticularMethod::Gram,
            LinearAlgebra::Iterative,
        ),
    ];
    for (name, lambda, method, algebra) in cases {
        let started = Instant::now();
        let result = solve_inverse_fdm(
            &problem,
            &target,
            inverse_opts(lambda, true, method, algebra, true),
        );
        let total = started.elapsed();
        match result {
            Ok(result) => eprintln!(
                "inverse,{name},edges={edge_count},total_ms={:.3},solve_estimate_ms={:.3},iterations={},converged={}",
                total.as_secs_f64() * 1e3,
                total.saturating_sub(assembly).as_secs_f64() * 1e3,
                result.iterations,
                result.converged
            ),
            Err(error) => eprintln!(
                "inverse,{name},edges={edge_count},total_ms={:.3},status=error,error={error}",
                total.as_secs_f64() * 1e3
            ),
        }
    }
}

// ─────────────────────────────────────────────────────────────
//  Geometric metric
// ─────────────────────────────────────────────────────────────

/// Nudge a target off the equilibrium manifold so it is near- but not exactly
/// funicular.
fn perturb(target: &Array2<f64>, amount: f64) -> Array2<f64> {
    Array2::from_shape_fn(target.dim(), |(i, d)| {
        let wobble = ((i * 3 + d) as f64 * 1.7).sin();
        target[[i, d]] + amount * wobble
    })
}

fn column_norm(v: &Array2<f64>) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

#[test]
fn geometric_error_identity_matches_an_independent_forward_solve() {
    // The whole geometric metric rests on x(q) - x* = -D(q)^-1 (E(x*)q - p).
    // Check it against a real forward solve rather than against itself.
    let (problem, _) = arch_problem(false);
    let q = vec![1.3, 0.7, 2.1, 0.9, 1.6, 1.1, 0.5, 2.4];
    let (funicular, _) = forward_target(&problem, &q);

    for amount in [0.0, 1e-3, 0.05, 0.4] {
        let target = perturb(&funicular, amount);
        let predicted = geometric_error_vector(&problem, &target, &q).unwrap();
        let actual =
            Array2::from_shape_fn(funicular.dim(), |(i, d)| funicular[[i, d]] - target[[i, d]]);

        for i in 0..actual.nrows() {
            for d in 0..3 {
                assert!(
                    (predicted[[i, d]] - actual[[i, d]]).abs() < 1e-9,
                    "identity failed at node {i} axis {d} for perturbation {amount}: \
                     predicted {} vs forward-solved {}",
                    predicted[[i, d]],
                    actual[[i, d]]
                );
            }
        }
    }
}

#[test]
fn geometric_error_is_zero_when_the_target_is_already_funicular() {
    let (problem, _) = arch_problem(false);
    let q = vec![1.0, 1.4, 0.8, 1.2, 0.6, 1.9, 1.1, 0.7];
    let (funicular, _) = forward_target(&problem, &q);

    let error = geometric_error_vector(&problem, &funicular, &q).unwrap();
    assert!(
        column_norm(&error) < 1e-10,
        "expected a funicular target to have zero geometric error, got {}",
        column_norm(&error)
    );
}

fn geometric_opts(
    metric: InverseMetric,
    particular: ParticularMethod,
    algebra: LinearAlgebra,
    lambda: f64,
) -> InverseFdmOptions {
    InverseFdmOptions {
        metric,
        // A positive box keeps the Laplacian SPD, which the compliance needs.
        signs: vec![1],
        lower: vec![1e-6],
        max_outer: 12,
        tol: 1e-10,
        ..inverse_opts(lambda, true, particular, algebra, true)
    }
}

/// Distance from the forward solve to the target for a recovered q.
fn achieved_error(problem: &Problem, target: &Array2<f64>, q: &[f64]) -> f64 {
    let (reached, _) = forward_target(problem, q);
    let mut sum = 0.0;
    for i in 0..target.nrows() {
        for d in 0..3 {
            let diff = reached[[i, d]] - target[[i, d]];
            sum += diff * diff;
        }
    }
    sum.sqrt()
}

#[test]
fn geometric_metric_lands_closer_to_a_near_funicular_target_than_the_force_metric() {
    let (problem, _) = arch_problem(false);
    let q = vec![1.2, 0.9, 1.5, 1.1, 0.8, 1.3, 0.6, 1.7];
    let (funicular, _) = forward_target(&problem, &q);
    let target = perturb(&funicular, 0.02);

    let force = solve_inverse_fdm(
        &problem,
        &target,
        geometric_opts(
            InverseMetric::Force,
            ParticularMethod::Clarabel,
            LinearAlgebra::Direct,
            1e-8,
        ),
    )
    .unwrap();
    let geometry = solve_inverse_fdm(
        &problem,
        &target,
        geometric_opts(
            InverseMetric::Geometry,
            ParticularMethod::Clarabel,
            LinearAlgebra::Direct,
            1e-8,
        ),
    )
    .unwrap();
    let newton = solve_inverse_fdm(
        &problem,
        &target,
        geometric_opts(
            InverseMetric::GeometryNewton,
            ParticularMethod::Clarabel,
            LinearAlgebra::Direct,
            1e-8,
        ),
    )
    .unwrap();

    let force_error = achieved_error(&problem, &target, &force.q);
    let geometry_error = achieved_error(&problem, &target, &geometry.q);
    let newton_error = achieved_error(&problem, &target, &newton.q);

    eprintln!("force={force_error:.6e} geometry={geometry_error:.6e} newton={newton_error:.6e}");
    assert!(
        geometry_error < force_error,
        "geometric metric ({geometry_error:.6e}) should beat the force metric \
         ({force_error:.6e})"
    );
    assert!(
        newton_error <= geometry_error * 1.05,
        "Gauss-Newton ({newton_error:.6e}) should not be worse than the frozen-target \
         metric ({geometry_error:.6e})"
    );
}

#[test]
fn reported_geometric_error_matches_the_forward_solve() {
    let (problem, _) = arch_problem(false);
    let q = vec![1.1, 1.4, 0.9, 1.2, 1.0, 1.5, 0.7, 1.3];
    let (funicular, _) = forward_target(&problem, &q);
    let target = perturb(&funicular, 0.03);

    let result = solve_inverse_fdm(
        &problem,
        &target,
        geometric_opts(
            InverseMetric::GeometryNewton,
            ParticularMethod::Clarabel,
            LinearAlgebra::Direct,
            1e-8,
        ),
    )
    .unwrap();

    let measured = achieved_error(&problem, &target, &result.q);
    assert!(
        (result.geometric_error - measured).abs() < 1e-7,
        "reported geometric error {} disagrees with the forward solve {measured}",
        result.geometric_error
    );
}

#[test]
fn geometric_backends_agree_on_the_same_weighted_problem() {
    let (problem, _) = arch_problem(false);
    let q = vec![1.0, 1.2, 0.9, 1.1, 1.3, 0.8, 1.4, 1.0];
    let (funicular, _) = forward_target(&problem, &q);
    let target = perturb(&funicular, 0.02);

    let cases = [
        (
            "clarabel",
            ParticularMethod::Clarabel,
            LinearAlgebra::Direct,
        ),
        ("saddle", ParticularMethod::Augmented, LinearAlgebra::Direct),
        ("spg", ParticularMethod::Clarabel, LinearAlgebra::Iterative),
    ];

    let mut errors = Vec::new();
    for (name, particular, algebra) in cases {
        let result = solve_inverse_fdm(
            &problem,
            &target,
            geometric_opts(InverseMetric::Geometry, particular, algebra, 1e-8),
        )
        .unwrap_or_else(|e| panic!("{name} failed: {e}"));
        let error = achieved_error(&problem, &target, &result.q);
        eprintln!("{name}: geometric error {error:.6e}");
        errors.push((name, error));
    }

    let best = errors.iter().map(|(_, e)| *e).fold(f64::MAX, f64::min);
    for (name, error) in &errors {
        assert!(
            *error <= best * 10.0 + 1e-9,
            "{name} reached {error:.6e}, far off the best backend {best:.6e}"
        );
    }
}

#[test]
fn gram_and_sparse_qr_can_initialize_geometric_stage_two() {
    let (problem, _) = arch_problem(false);
    let q = vec![1.0; 8];
    let (funicular, _) = forward_target(&problem, &q);

    for particular in [ParticularMethod::Gram, ParticularMethod::SparseQr] {
        let result = solve_inverse_fdm(
            &problem,
            &funicular,
            InverseFdmOptions {
                metric: InverseMetric::Geometry,
                max_outer: 4,
                ..inverse_opts(1e-8, true, particular, LinearAlgebra::Direct, true)
            },
        )
        .unwrap_or_else(|error| panic!("{particular:?} initializer failed: {error}"));
        assert!(result.q.iter().all(|q| q.is_finite()));
        assert!(result.geometric_error < 1e-6);
    }
}

#[test]
fn geometric_metric_rejects_l1() {
    let (problem, _) = arch_problem(false);
    let q = vec![1.0; 8];
    let (funicular, _) = forward_target(&problem, &q);

    let error = solve_inverse_fdm(
        &problem,
        &funicular,
        InverseFdmOptions {
            metric: InverseMetric::Geometry,
            max_outer: 4,
            ..inverse_opts(
                1e-8,
                false,
                ParticularMethod::Clarabel,
                LinearAlgebra::Direct,
                true,
            )
        },
    )
    .expect_err("expected the geometric metric to reject IRLS");
    assert!(
        error.to_string().contains("L1/IRLS"),
        "unexpected rejection message: {error}"
    );
}

#[test]
fn force_metric_is_unchanged_by_the_metric_plumbing() {
    // Same fixture and options as the historical path; only the new default
    // fields are present. The recovered q must still round-trip exactly.
    let (problem, _) = triangle_problem();
    let expected = vec![2.0, 3.0];
    let (target, _) = forward_target(&problem, &expected);

    let result = solve_inverse_fdm(
        &problem,
        &target,
        inverse_opts(
            0.0,
            true,
            ParticularMethod::Augmented,
            LinearAlgebra::Direct,
            true,
        ),
    )
    .unwrap();

    for (got, want) in result.q.iter().zip(&expected) {
        assert!(
            (got - want).abs() < 1e-8,
            "force metric changed: got {got}, expected {want}"
        );
    }
}

fn square_grid_problem(side: usize) -> (Problem, Vec<f64>) {
    let node = |row: usize, col: usize| row * side + col;
    let mut edges = Vec::new();
    for row in 0..side {
        for col in 0..side {
            if col + 1 < side {
                edges.push((node(row, col), node(row, col + 1)));
            }
            if row + 1 < side {
                edges.push((node(row, col), node(row + 1, col)));
            }
        }
    }
    let fixed = vec![
        node(0, 0),
        node(0, side - 1),
        node(side - 1, 0),
        node(side - 1, side - 1),
    ];
    let free: Vec<usize> = (0..side * side)
        .filter(|index| !fixed.contains(index))
        .collect();
    let fixed_positions = Array2::from_shape_fn((4, 3), |(i, axis)| {
        let locations = [
            [0.0, 0.0, 0.0],
            [(side - 1) as f64, 0.0, 0.0],
            [0.0, (side - 1) as f64, 0.0],
            [(side - 1) as f64, (side - 1) as f64, 0.0],
        ];
        locations[i][axis]
    });
    let mut loads = Array2::zeros((free.len(), 3));
    for i in 0..free.len() {
        loads[[i, 2]] = -1.0;
    }
    let problem = make_problem(&edges, side * side, free, fixed, loads, fixed_positions);

    let mut state = 0x5eed_u64;
    let q = (0..edges.len())
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let unit = ((state >> 32) as u32) as f64 / u32::MAX as f64;
            10.0_f64.powf(-3.0 + 5.0 * unit)
        })
        .collect();
    (problem, q)
}

fn jitter_z(target: &Array2<f64>) -> Array2<f64> {
    let mut state = 0xc0ffee_u64;
    let mut result = target.clone();
    for i in 0..result.nrows() {
        state = state
            .wrapping_mul(2862933555777941757)
            .wrapping_add(3037000493);
        let unit = ((state >> 32) as u32) as f64 / u32::MAX as f64;
        result[[i, 2]] += 0.2 * unit - 0.1;
    }
    result
}

#[test]
fn staged_metrics_run_on_deterministic_corner_anchored_grid() {
    let (problem, known_q) = square_grid_problem(4);
    let (funicular, _) = forward_target(&problem, &known_q);
    let target = jitter_z(&funicular);
    let mut results = Vec::new();
    for metric in [
        InverseMetric::Force,
        InverseMetric::Geometry,
        InverseMetric::GeometryNewton,
    ] {
        let mut opts = inverse_opts(
            1e-8,
            true,
            ParticularMethod::Clarabel,
            LinearAlgebra::Direct,
            false,
        );
        opts.metric = metric;
        opts.lower = vec![1e-3];
        opts.upper = vec![100.0];
        let result = solve_inverse_fdm(&problem, &target, opts).unwrap();
        assert!(result
            .q
            .iter()
            .all(|q| q.is_finite() && (1e-3 - 1e-8..=100.0 + 1e-8).contains(q)));
        assert!(result.geometric_error.is_finite());
        results.push(result.geometric_error);
    }
    assert!(
        results[1] <= results[0] * 1.05 && results[2] <= results[0] * 1.05,
        "staged metrics should preserve or improve geometry: {results:?}"
    );
}

#[test]
fn stage_one_initializes_geometry_and_lower_bound_is_sensitive() {
    let (problem, known_q) = square_grid_problem(4);
    let (funicular, _) = forward_target(&problem, &known_q);

    let mut force_opts = inverse_opts(
        1e-8,
        true,
        ParticularMethod::Gram,
        LinearAlgebra::Direct,
        true,
    );
    force_opts.lower = vec![1e-3];
    let stage_one = solve_inverse_fdm(&problem, &funicular, force_opts.clone()).unwrap();
    force_opts.metric = InverseMetric::Geometry;
    let staged = solve_inverse_fdm(&problem, &funicular, force_opts).unwrap();
    assert!(staged.geometric_error <= stage_one.geometric_error + 1e-8);

    let target = jitter_z(&funicular);
    let mut low = inverse_opts(
        0.0,
        true,
        ParticularMethod::Clarabel,
        LinearAlgebra::Direct,
        false,
    );
    low.lower = vec![1e-3];
    low.upper = vec![100.0];
    let mut high = low.clone();
    high.lower = vec![10.0];
    let low_result = solve_inverse_fdm(&problem, &target, low).unwrap();
    let high_result = solve_inverse_fdm(&problem, &target, high).unwrap();
    assert!(high_result.q.iter().all(|q| *q >= 10.0 - 1e-7));
    assert!(low_result
        .q
        .iter()
        .zip(&high_result.q)
        .any(|(left, right)| (left - right).abs() > 1e-3));
}

#[test]
fn mixed_sign_and_compression_geometric_identity_match_forward_solve() {
    let (problem, _) = arch_problem(false);
    for q in [
        vec![1.3, -0.7, 2.1, -0.9, 1.6, -1.1, 0.5, 2.4],
        vec![-1.3, -0.7, -2.1, -0.9, -1.6, -1.1, -0.5, -2.4],
    ] {
        let (funicular, _) = forward_target(&problem, &q);
        let target = perturb(&funicular, 0.02);
        let predicted = geometric_error_vector(&problem, &target, &q).unwrap();
        let actual =
            Array2::from_shape_fn(funicular.dim(), |(i, d)| funicular[[i, d]] - target[[i, d]]);
        for (left, right) in predicted.iter().zip(actual.iter()) {
            assert!((left - right).abs() < 1e-8, "{left} != {right}");
        }
    }
}

#[test]
fn mixed_sign_geometric_direct_and_iterative_backends_agree() {
    let (problem, _) = arch_problem(false);
    let q = vec![1.3, -0.7, 2.1, -0.9, 1.6, -1.1, 0.5, 2.4];
    let (funicular, _) = forward_target(&problem, &q);
    let target = perturb(&funicular, 0.01);
    let mut errors = Vec::new();
    for algebra in [LinearAlgebra::Direct, LinearAlgebra::Iterative] {
        let mut opts = inverse_opts(1e-8, true, ParticularMethod::Gram, algebra, true);
        opts.metric = InverseMetric::Geometry;
        opts.q_ref = q.clone();
        opts.max_iter = 8_000;
        opts.tol = 1e-8;
        let result = solve_inverse_fdm(&problem, &target, opts).unwrap();
        errors.push(result.geometric_error);
    }
    let scale = errors[0].max(errors[1]).max(1e-12);
    assert!((errors[0] - errors[1]).abs() <= 0.1 * scale, "{errors:?}");
}

#[test]
fn mixed_sign_particular_initializes_cwls_in_q_and_force_coordinates() {
    let (problem, _) = arch_problem(false);
    let known_q = vec![1.3, -0.7, 2.1, -0.9, 1.6, -1.1, 0.5, 2.4];
    let signs: Vec<i32> = known_q
        .iter()
        .map(|value| if *value > 0.0 { 1 } else { -1 })
        .collect();
    let lower: Vec<f64> = signs
        .iter()
        .map(|sign| if *sign > 0 { 0.1 } else { -3.0 })
        .collect();
    let upper: Vec<f64> = signs
        .iter()
        .map(|sign| if *sign > 0 { 3.0 } else { -0.1 })
        .collect();
    let (funicular, _) = forward_target(&problem, &known_q);
    let target = perturb(&funicular, 0.005);

    for solve_for_q in [true, false] {
        let mut opts = inverse_opts(
            1e-6,
            true,
            ParticularMethod::Clarabel,
            LinearAlgebra::Direct,
            solve_for_q,
        );
        opts.metric = InverseMetric::GeometryNewton;
        opts.signs = signs.clone();
        opts.lower = lower.clone();
        opts.upper = upper.clone();
        opts.q_ref.clear();
        let result = solve_inverse_fdm(&problem, &target, opts).unwrap();
        assert!(result.geometric_error.is_finite());
        for (edge, value) in result.q.iter().enumerate() {
            assert!(
                (lower[edge] - 1e-7..=upper[edge] + 1e-7).contains(value),
                "solve_for_q={solve_for_q}, edge={edge}, q={value}"
            );
        }
    }
}

#[test]
fn singular_mixed_sign_cancellation_is_rejected() {
    let (problem, target) = duplicate_edge_problem();
    let error = geometric_error_vector(&problem, &target, &[1.0, -1.0])
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("nonsingular") || error.contains("factorization"),
        "{error}"
    );
}

#[test]
fn geometric_exact_stage_one_is_converged_without_a_step() {
    let (problem, _) = arch_problem(false);
    let q = vec![1.0; 8];
    let (target, _) = forward_target(&problem, &q);
    let mut opts = geometric_opts(
        InverseMetric::Geometry,
        ParticularMethod::Clarabel,
        LinearAlgebra::Direct,
        0.0,
    );
    opts.q_ref = q;
    opts.max_outer = 1;
    let result = solve_inverse_fdm(&problem, &target, opts).unwrap();
    assert!(result.converged);
    assert_eq!(result.iterations, 0);
    assert!(result.geometric_error < 1e-10);
}

#[test]
fn geometric_blocked_step_is_not_reported_converged() {
    let (problem, _) = arch_problem(false);
    let q = vec![1.0; 8];
    let (funicular, _) = forward_target(&problem, &q);
    let target = perturb(&funicular, 0.02);
    let mut opts = geometric_opts(
        InverseMetric::Geometry,
        ParticularMethod::Clarabel,
        LinearAlgebra::Direct,
        0.0,
    );
    opts.q_ref = q.clone();
    opts.lower = q.clone();
    opts.upper = q;
    opts.max_outer = 1;
    let result = solve_inverse_fdm(&problem, &target, opts).unwrap();
    assert!(!result.converged);
    assert!(result.geometric_error > 1e-8);
}

#[test]
fn frozen_cwls_honors_a_single_update_budget() {
    let (problem, known_q) = square_grid_problem(4);
    let (funicular, _) = forward_target(&problem, &known_q);
    let target = jitter_z(&funicular);
    let mut opts = geometric_opts(
        InverseMetric::Geometry,
        ParticularMethod::Clarabel,
        LinearAlgebra::Direct,
        1e-8,
    );
    opts.max_outer = 1;
    opts.tol = 1e-12;
    let result = solve_inverse_fdm(&problem, &target, opts).unwrap();
    assert_eq!(result.iterations, 1);
}

#[test]
fn phased_frozen_only_matches_legacy_frozen_metric() {
    let (problem, known_q) = square_grid_problem(4);
    let (funicular, _) = forward_target(&problem, &known_q);
    let target = jitter_z(&funicular);

    let mut legacy = geometric_opts(
        InverseMetric::Geometry,
        ParticularMethod::Clarabel,
        LinearAlgebra::Direct,
        1e-8,
    );
    legacy.max_outer = 3;
    legacy.tol = 1e-12;
    let legacy_result = solve_inverse_fdm(&problem, &target, legacy).unwrap();

    let mut phased = geometric_opts(
        InverseMetric::GeometryNewton,
        ParticularMethod::Clarabel,
        LinearAlgebra::Direct,
        1e-8,
    );
    phased.max_frozen_outer = 3;
    phased.max_outer = 0;
    phased.tol = 1e-12;
    let phased_result = solve_inverse_fdm(&problem, &target, phased).unwrap();

    assert_eq!(phased_result.iterations, legacy_result.iterations);
    assert!((phased_result.geometric_error - legacy_result.geometric_error).abs() < 1e-10);
    for (actual, expected) in phased_result.q.iter().zip(&legacy_result.q) {
        assert!((actual - expected).abs() < 1e-10);
    }
}

#[test]
fn zero_geometric_phase_budgets_return_stage_one() {
    let (problem, known_q) = square_grid_problem(4);
    let (funicular, _) = forward_target(&problem, &known_q);
    let target = jitter_z(&funicular);

    let force = solve_inverse_fdm(
        &problem,
        &target,
        geometric_opts(
            InverseMetric::Force,
            ParticularMethod::Clarabel,
            LinearAlgebra::Direct,
            1e-8,
        ),
    )
    .unwrap();
    let mut stage_one_only = geometric_opts(
        InverseMetric::GeometryNewton,
        ParticularMethod::Clarabel,
        LinearAlgebra::Direct,
        1e-8,
    );
    stage_one_only.max_frozen_outer = 0;
    stage_one_only.max_outer = 0;
    let result = solve_inverse_fdm(&problem, &target, stage_one_only).unwrap();

    assert_eq!(result.iterations, 0);
    for (actual, expected) in result.q.iter().zip(&force.q) {
        assert!((actual - expected).abs() < 1e-10);
    }
}

#[test]
fn sequential_phases_preserve_the_best_frozen_result() {
    let (problem, known_q) = square_grid_problem(4);
    let (funicular, _) = forward_target(&problem, &known_q);
    let target = jitter_z(&funicular);

    let mut frozen = geometric_opts(
        InverseMetric::Geometry,
        ParticularMethod::Clarabel,
        LinearAlgebra::Direct,
        1e-8,
    );
    frozen.max_outer = 3;
    frozen.tol = 1e-12;
    let frozen_result = solve_inverse_fdm(&problem, &target, frozen).unwrap();

    let mut sequential = geometric_opts(
        InverseMetric::GeometryNewton,
        ParticularMethod::Clarabel,
        LinearAlgebra::Direct,
        1e-8,
    );
    sequential.max_frozen_outer = 3;
    sequential.max_outer = 3;
    sequential.tol = 1e-12;
    let sequential_result = solve_inverse_fdm(&problem, &target, sequential).unwrap();

    assert!(
        sequential_result.geometric_error <= frozen_result.geometric_error + 1e-10,
        "sequential error {} exceeded frozen error {}",
        sequential_result.geometric_error,
        frozen_result.geometric_error
    );
}

#[test]
fn geometry_newton_honors_a_budget_above_the_default() {
    let (problem, known_q) = square_grid_problem(4);
    let (funicular, _) = forward_target(&problem, &known_q);
    let target = jitter_z(&funicular);
    let mut opts = geometric_opts(
        InverseMetric::GeometryNewton,
        ParticularMethod::Clarabel,
        LinearAlgebra::Direct,
        1e-8,
    );
    opts.max_outer = DEFAULT_MAX_OUTER + 2;
    opts.tol = 1e-12;
    let requested = opts.max_outer;
    let result = solve_inverse_fdm(&problem, &target, opts).unwrap();
    assert!(
        result.iterations > DEFAULT_MAX_OUTER,
        "requested {} updates, but only {} ran",
        requested,
        result.iterations
    );
    assert!(result.iterations <= requested);
}

#[test]
fn public_q_bounds_hold_for_q_and_member_force_stage_one() {
    let (problem, _) = triangle_problem();
    let (target, _) = forward_target(&problem, &[2.0, 3.0]);
    for solve_for_q in [true, false] {
        let mut opts = inverse_opts(
            0.0,
            true,
            ParticularMethod::Clarabel,
            LinearAlgebra::Direct,
            solve_for_q,
        );
        opts.lower = vec![2.5];
        opts.upper = vec![2.75];
        let result = solve_inverse_fdm(&problem, &target, opts).unwrap();
        assert!(
            result
                .q
                .iter()
                .all(|q| (2.5 - 1e-7..=2.75 + 1e-7).contains(q)),
            "solve_for_q={solve_for_q}: {:?}",
            result.q
        );
    }
}

#[test]
#[ignore = "manual release benchmark; run with --release --ignored --nocapture"]
fn benchmark_staged_inverse_21_by_21_nodes() {
    let (problem, q) = square_grid_problem(21);
    let (funicular, _) = forward_target(&problem, &q);
    let target = jitter_z(&funicular);
    for (metric, updates) in std::iter::once((InverseMetric::Force, 0)).chain(
        [InverseMetric::Geometry, InverseMetric::GeometryNewton]
            .into_iter()
            .flat_map(|metric| (1..=3).map(move |updates| (metric, updates))),
    ) {
        let mut opts = inverse_opts(
            1e-8,
            true,
            ParticularMethod::Clarabel,
            LinearAlgebra::Direct,
            false,
        );
        opts.metric = metric;
        opts.max_outer = updates.max(1);
        opts.lower = vec![1e-3];
        opts.upper = vec![100.0];
        let started = Instant::now();
        let result = solve_inverse_fdm(&problem, &target, opts).unwrap();
        eprintln!(
            "staged-grid,metric={metric:?},updates={updates},nodes=441,edges={},total_ms={:.3},iterations={},error={:.6e}",
            problem.topology.num_edges,
            started.elapsed().as_secs_f64() * 1e3,
            result.iterations,
            result.geometric_error
        );
    }
}
