//! Integration tests for inverse FDM particulars (Direct, Iterative, box, IRLS).

use ndarray::Array2;
use std::time::Instant;
use theseus::fdm;
use theseus::inverse::{
    compose_box, solve_inverse_fdm, solve_pseudoinverse_dispatch, solve_spg_box, InverseFdmOptions,
    LinearAlgebra, ParticularMethod,
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
        problem, target, max_iter, tol, &[1], &[], &[], 0.0, false, false, false, true,
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
        &problem, &target, 1e-10, true, 20, ParticularMethod::Gram, false, false, false, false,
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
        &problem, &target, 1e-6, true, 20, ParticularMethod::Gram, false, false, false, false,
    );

    assert!(result.is_err());
}

#[test]
fn pseudoinverse_reaction_constraints_reduce_reaction_norm() {
    let (problem, edges) = arch_problem(false);
    let (target, _) = forward_target(&problem, &[1.0; 8]);

    let unconstrained = pinv(&problem, &target, true, false);
    let constrained = solve_pseudoinverse_dispatch(
        &problem, &target, 1e-8, true, 20, ParticularMethod::Gram, true, true, true, true,
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
        assert!(result.q.iter().all(|q| (0.5 - 1e-6..=4.0 + 1e-6).contains(q)));
    }
}

#[test]
fn near_flat_box_caps_unconstrained_particular() {
    let (problem, target) = near_flat_problem();
    let unconstrained = solve_inverse_fdm(
        &problem,
        &target,
        inverse_opts(0.0, true, ParticularMethod::Augmented, LinearAlgebra::Direct, false),
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
        // Bound is on member force t; recovered q = t / L so |q| is also capped near 10/L ≈ 10.
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
    let mut clarabel = inverse_opts(0.0, true, ParticularMethod::Augmented, LinearAlgebra::Direct, true);
    clarabel.signs = vec![1];
    let mut spg = inverse_opts(0.0, true, ParticularMethod::Augmented, LinearAlgebra::Iterative, true);
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
        inverse_opts(0.0, true, ParticularMethod::SparseQr, LinearAlgebra::Direct, true),
    )
    .unwrap();
    let lsqr = solve_inverse_fdm(
        &problem,
        &target,
        inverse_opts(0.0, true, ParticularMethod::Augmented, LinearAlgebra::Iterative, true),
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
        (ParticularMethod::Augmented, LinearAlgebra::Direct, Vec::new(), Vec::new()),
        (ParticularMethod::Gram, LinearAlgebra::Direct, Vec::new(), Vec::new()),
        (ParticularMethod::SparseQr, LinearAlgebra::Direct, Vec::new(), Vec::new()),
        (ParticularMethod::Augmented, LinearAlgebra::Iterative, Vec::new(), Vec::new()),
        (ParticularMethod::Augmented, LinearAlgebra::Direct, vec![1], Vec::new()),
        (ParticularMethod::Augmented, LinearAlgebra::Iterative, vec![1], Vec::new()),
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
    assert_eq!(ParticularMethod::try_from(0).unwrap(), ParticularMethod::Gram);
    assert_eq!(ParticularMethod::try_from(1).unwrap(), ParticularMethod::Augmented);
    assert_eq!(ParticularMethod::try_from(2).unwrap(), ParticularMethod::SparseQr);
    assert_eq!(ParticularMethod::try_from(3).unwrap(), ParticularMethod::Clarabel);
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
        inverse_opts(lambda, true, ParticularMethod::Clarabel, LinearAlgebra::Direct, true),
    )
    .unwrap();
    let tikhonov = solve_inverse_fdm(
        &problem,
        &target,
        inverse_opts(lambda, true, ParticularMethod::Augmented, LinearAlgebra::Direct, true),
    )
    .unwrap();
    for (left, right) in clarabel.q.iter().zip(&tikhonov.q) {
        assert!((left - right).abs() < 2e-4, "Clarabel {left} vs Tikhonov {right}");
    }

    let zero = solve_inverse_fdm(
        &problem,
        &target,
        inverse_opts(0.0, true, ParticularMethod::Clarabel, LinearAlgebra::Direct, true),
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
        inverse_opts(0.0, true, ParticularMethod::Augmented, LinearAlgebra::Direct, true),
    )
    .unwrap_err();
    let message = direct.to_string();
    assert!(message.contains("Iterative LSQR"), "{message}");
    assert!(message.contains("lambda=0"), "{message}");

    let iterative = solve_inverse_fdm(
        &problem,
        &target,
        inverse_opts(0.0, true, ParticularMethod::Gram, LinearAlgebra::Iterative, true),
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
        inverse_opts(0.0, true, ParticularMethod::Augmented, LinearAlgebra::Direct, false),
    )
    .unwrap();
    let iterative = solve_inverse_fdm(
        &problem,
        &target,
        inverse_opts(0.0, true, ParticularMethod::Clarabel, LinearAlgebra::Iterative, false),
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
        inverse_opts(lambda, true, ParticularMethod::SparseQr, LinearAlgebra::Iterative, true),
    )
    .unwrap();
    let tikhonov = solve_inverse_fdm(
        &problem,
        &target,
        inverse_opts(lambda, true, ParticularMethod::Augmented, LinearAlgebra::Direct, true),
    )
    .unwrap();
    let gram = solve_inverse_fdm(
        &problem,
        &target,
        inverse_opts(lambda, true, ParticularMethod::Gram, LinearAlgebra::Direct, true),
    )
    .unwrap();
    assert!(iterative.converged);
    assert!(iterative.iterations > 1);
    for ((lsqr, saddle), normal) in iterative.q.iter().zip(&tikhonov.q).zip(&gram.q) {
        assert!((lsqr - saddle).abs() < 2e-5, "LSQR {lsqr} vs saddle {saddle}");
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
            assert!((left - right).abs() < 1e-12, "{method:?}: {left} vs {right}");
        }
    }
}

#[test]
fn regularization_materially_changes_scaled_lsqr_fixture() {
    let (problem, target) = near_flat_problem();
    let zero = solve_inverse_fdm(
        &problem,
        &target,
        inverse_opts(0.0, true, ParticularMethod::Gram, LinearAlgebra::Iterative, false),
    )
    .unwrap();
    let regularized = solve_inverse_fdm(
        &problem,
        &target,
        inverse_opts(1e-2, true, ParticularMethod::Gram, LinearAlgebra::Iterative, false),
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
        inverse_opts(0.0, true, ParticularMethod::SparseQr, LinearAlgebra::Direct, true),
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
        ("mp", 0.0, ParticularMethod::Augmented, LinearAlgebra::Direct),
        ("tikhonov", 1e-6, ParticularMethod::Augmented, LinearAlgebra::Direct),
        ("qr", 0.0, ParticularMethod::SparseQr, LinearAlgebra::Direct),
        ("gram", 1e-6, ParticularMethod::Gram, LinearAlgebra::Direct),
        ("lsqr-zero", 0.0, ParticularMethod::Clarabel, LinearAlgebra::Iterative),
        ("lsqr-regularized", 1e-6, ParticularMethod::Gram, LinearAlgebra::Iterative),
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
