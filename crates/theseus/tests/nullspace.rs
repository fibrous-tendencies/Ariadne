//! Focused correctness tests for Pellegrino--Calladine analysis.

use ndarray::Array2;
use theseus::inverse::solve_inverse_fdm_default;
use theseus::nullspace::{
    analyze, analyze_dense_svd, analyze_projector, apply_pseudoinverse,
    classify_prestress_stability, retract_member_lengths, EquilibriumSystem, EquilibriumUnknown,
    LengthRetractionOptions, MechanismClass, NullspaceMethod, NullspaceOptions, RowSpaceSolver,
};
use theseus::sparse::SparseColMatOwned;
use theseus::types::{AnchorInfo, Bounds, NetworkTopology, Problem, SolverOptions};

fn incidence(edges: &[(usize, usize)], nodes: usize) -> SparseColMatOwned {
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut values = Vec::new();
    for (edge, &(start, end)) in edges.iter().enumerate() {
        rows.extend([edge, edge]);
        cols.extend([start, end]);
        values.extend([-1.0, 1.0]);
    }
    SparseColMatOwned::from_coo(edges.len(), nodes, &rows, &cols, &values).unwrap()
}

fn problem(
    edges: &[(usize, usize)],
    positions: &[[f64; 3]],
    free: Vec<usize>,
    fixed: Vec<usize>,
    loads: Vec<f64>,
) -> (Problem, Array2<f64>) {
    let c = incidence(edges, positions.len());
    let free_positions = Array2::from_shape_fn((free.len(), 3), |(i, d)| positions[free[i]][d]);
    let fixed_positions = Array2::from_shape_fn((fixed.len(), 3), |(i, d)| positions[fixed[i]][d]);
    let free_incidence = c.extract_columns(&free);
    let fixed_incidence = c.extract_columns(&fixed);
    let anchors = AnchorInfo::all_fixed(fixed_positions.clone());
    (
        Problem {
            topology: NetworkTopology {
                incidence: c,
                free_incidence,
                fixed_incidence,
                num_edges: edges.len(),
                num_nodes: positions.len(),
                free_node_indices: free,
                fixed_node_indices: fixed,
            },
            free_node_loads: Array2::from_shape_vec((free_positions.nrows(), 3), loads).unwrap(),
            fixed_node_positions: fixed_positions,
            anchors,
            objectives: Vec::new(),
            bounds: Bounds::default_for(edges.len()),
            solver: SolverOptions::default(),
            self_weight: None,
            pressure: None,
        },
        free_positions,
    )
}

fn options(max_modes: usize) -> NullspaceOptions {
    NullspaceOptions {
        max_modes,
        tolerance: 1e-9,
        ..NullspaceOptions::default()
    }
}

fn subspace_projection_error(reference: &Array2<f64>, candidate: &Array2<f64>) -> f64 {
    let mut worst = 0.0_f64;
    for j in 0..reference.ncols() {
        let mut captured = 0.0;
        for k in 0..candidate.ncols() {
            let dot = (0..reference.nrows())
                .map(|i| reference[[i, j]] * candidate[[i, k]])
                .sum::<f64>();
            captured += dot * dot;
        }
        worst = worst.max((1.0 - captured).abs().sqrt());
    }
    worst
}

#[test]
fn reduced_four_bar_with_fixed_side_has_one_mechanism() {
    // A planar four-bar after eliminating the fixed side's coordinates. The
    // fifth equilibrium row is dependent, so rank(A)=4, s=0, and m=1.
    let a = SparseColMatOwned::from_coo(
        5,
        4,
        &[0, 1, 2, 3, 4, 4],
        &[0, 1, 2, 3, 0, 1],
        &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    )
    .unwrap();
    let system = EquilibriumSystem {
        a,
        p: vec![0.0; 5],
        lengths: vec![1.0; 4],
        n_eq: 5,
        n_edges: 4,
        free_positions: Array2::zeros((0, 3)),
        n_free: 0,
    };
    let report = analyze_projector(&system, &options(4)).unwrap();

    assert_eq!(
        (report.rank, report.s, report.m_raw, report.m),
        (4, 0, 1, 1)
    );
}

#[test]
fn sparse_qr_comparison_reports_valid_full_column_rank_fixture() {
    let a = SparseColMatOwned::from_coo(
        5,
        4,
        &[0, 1, 2, 3, 4, 4],
        &[0, 1, 2, 3, 0, 1],
        &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    )
    .unwrap();
    let system = EquilibriumSystem {
        a,
        p: vec![1.0, -0.5, 0.25, 0.75, 0.5],
        lengths: vec![1.0; 4],
        n_eq: 5,
        n_edges: 4,
        free_positions: Array2::zeros((0, 3)),
        n_free: 0,
    };
    let report = analyze(
        &system,
        &NullspaceOptions {
            method: NullspaceMethod::SparseQr,
            ..options(4)
        },
    )
    .unwrap();

    assert_eq!(
        (report.rank, report.s, report.m_raw, report.m),
        (4, 0, 1, 1)
    );
    assert!(report.self_stress.is_empty());
    assert!(report.residual_ratio.is_finite());
}

#[test]
fn sparse_qr_comparison_rejects_unvalidated_rank_deficiency() {
    let a = SparseColMatOwned::from_coo(
        3,
        3,
        &[0, 1, 0, 1, 0, 1],
        &[0, 0, 1, 1, 2, 2],
        &[1.0, 2.0, 2.0, 4.0, -1.0, -2.0],
    )
    .unwrap();
    let system = EquilibriumSystem {
        a,
        p: vec![1.0, 2.0, 0.0],
        lengths: vec![1.0; 3],
        n_eq: 3,
        n_edges: 3,
        free_positions: Array2::zeros((0, 3)),
        n_free: 0,
    };
    let error = analyze(
        &system,
        &NullspaceOptions {
            method: NullspaceMethod::SparseQr,
            ..options(3)
        },
    )
    .unwrap_err();

    assert!(error
        .to_string()
        .contains("cannot establish full column rank"));
}

#[test]
fn prestressed_three_spoke_triangle_has_one_self_stress() {
    let positions = [
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.6, 0.7, 0.0],
    ];
    let edges = [(3, 0), (3, 1), (3, 2)];
    let (problem, target) = problem(&edges, &positions, vec![3], vec![0, 1, 2], vec![0.0; 3]);
    let system = EquilibriumSystem::force(&problem, &target, false, false, false).unwrap();
    let report = analyze_projector(&system, &options(8)).unwrap();

    assert_eq!(report.rank, 2);
    assert_eq!(report.s, 1);
    assert_eq!(report.m_raw, 1);
    assert_eq!(
        report.s as isize - report.m_raw as isize,
        system.n_edges as isize - system.n_eq as isize
    );
}

#[test]
fn projector_and_dense_svd_agree_on_counts_and_subspaces() {
    let positions = [
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.6, 0.7, 0.0],
    ];
    let edges = [(3, 0), (3, 1), (3, 2)];
    let (problem, target) = problem(
        &edges,
        &positions,
        vec![3],
        vec![0, 1, 2],
        vec![0.3, -0.2, 0.0],
    );
    let system = EquilibriumSystem::force(&problem, &target, false, false, false).unwrap();
    let projector = analyze_projector(&system, &options(8)).unwrap();
    let svd = analyze_dense_svd(&system, &options(8)).unwrap();

    assert_eq!(
        (projector.rank, projector.s, projector.m_raw),
        (svd.rank, svd.s, svd.m_raw)
    );
    assert!(subspace_projection_error(&svd.self_stress, &projector.self_stress) < 1e-6);
    assert!(subspace_projection_error(&svd.mechanisms, &projector.mechanisms) < 1e-6);
}

#[test]
fn planar_normal_load_stays_bounded_and_in_the_residual() {
    let positions = [
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.6, 0.7, 0.0],
    ];
    let edges = [(3, 0), (3, 1), (3, 2)];
    let (problem, target) = problem(
        &edges,
        &positions,
        vec![3],
        vec![0, 1, 2],
        vec![0.0, 0.0, -2.0],
    );
    let system = EquilibriumSystem::force(&problem, &target, false, false, false).unwrap();
    let report = analyze_projector(&system, &options(2)).unwrap();

    assert!(report
        .particular_t
        .iter()
        .all(|value| value.is_finite() && value.abs() < 1e3));
    assert!((report.residual_r[2] - 2.0).abs() < 1e-8);
    assert!((report.residual_ratio - 1.0).abs() < 1e-8);
}

#[test]
fn planar_grid_projector_accepts_inconsistent_normal_load() {
    let n = 4;
    let positions: Vec<[f64; 3]> = (0..n)
        .flat_map(|row| (0..n).map(move |col| [col as f64, row as f64, 0.0]))
        .collect();
    let mut edges = Vec::new();
    for row in 0..n {
        for col in 0..n - 1 {
            edges.push((row * n + col, row * n + col + 1));
        }
    }
    for row in 0..n - 1 {
        for col in 0..n {
            edges.push((row * n + col, (row + 1) * n + col));
        }
    }
    let fixed = vec![0, n - 1, n * (n - 1), n * n - 1];
    let free: Vec<usize> = (0..n * n).filter(|node| !fixed.contains(node)).collect();
    let mut loads = vec![0.0; 3 * free.len()];
    for load in loads.iter_mut().skip(2).step_by(3) {
        *load = -1.0;
    }
    let (problem, target) = problem(&edges, &positions, free, fixed, loads);
    let system = EquilibriumSystem::force(&problem, &target, false, false, false).unwrap();
    let at = system.a.transpose();
    let mut worst_left_projector_residual = 0.0_f64;
    for row in 0..system.n_eq {
        let mut probe = vec![0.0; system.n_eq];
        probe[row] = 1.0;
        let at_probe = at.matvec(&probe);
        let range_component = apply_pseudoinverse(&at, &at_probe, 1e-9, 0).unwrap();
        let projected: Vec<f64> = probe
            .iter()
            .zip(range_component)
            .map(|(probe, range)| probe - range)
            .collect();
        worst_left_projector_residual = worst_left_projector_residual.max(
            at.matvec(&projected)
                .iter()
                .map(|v| v * v)
                .sum::<f64>()
                .sqrt(),
        );
    }
    assert!(worst_left_projector_residual < 1e-7);

    let report = analyze_projector(&system, &options(32)).unwrap();

    assert_eq!((report.rank, report.s, report.m_raw), (20, 4, 16));
    assert!((report.residual_ratio - 1.0).abs() < 1e-8);
    assert!(report.particular_t.iter().all(|value| value.is_finite()));
    assert!(
        report
            .particular_t
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt()
            < 1e-8
    );
}

#[test]
fn unconstrained_tetrahedron_strips_six_rigid_body_modes() {
    let positions = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.2, 1.1, 0.0],
        [0.3, 0.4, 1.2],
    ];
    let edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
    let (problem, target) = problem(&edges, &positions, vec![0, 1, 2, 3], vec![], vec![0.0; 12]);
    let system = EquilibriumSystem::force(&problem, &target, false, false, false).unwrap();
    let report = analyze_projector(&system, &options(12)).unwrap();

    assert_eq!(report.rank, 6);
    assert_eq!(report.s, 0);
    assert_eq!(report.m_raw, 6);
    assert_eq!(report.n_rigid, 6);
    assert_eq!(report.m, 0);
    assert_eq!(report.mechanisms.ncols(), 0);
}

#[test]
fn max_modes_caps_returned_bases_without_changing_counts() {
    let positions = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
    ];
    let edges = [(0, 1), (2, 3)];
    let (problem, target) = problem(&edges, &positions, vec![0, 1, 2, 3], vec![], vec![0.0; 12]);
    let system = EquilibriumSystem::force(&problem, &target, false, false, false).unwrap();
    let report = analyze_projector(&system, &options(1)).unwrap();

    assert!(report.m > 1);
    assert!(report.self_stress.ncols() <= 1);
    assert_eq!(report.mechanisms.ncols(), 1);
    assert_eq!(report.m_raw, system.n_eq - report.rank);
}

#[test]
fn force_and_density_moore_penrose_particulars_differ_with_lengths() {
    let positions = [
        [0.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.7, 0.4, 0.0],
    ];
    let edges = [(3, 0), (3, 1), (3, 2)];
    let (problem, target) = problem(
        &edges,
        &positions,
        vec![3],
        vec![0, 1, 2],
        vec![1.0, -0.4, 0.0],
    );
    let density_system = EquilibriumSystem::assemble(
        &problem,
        &target,
        EquilibriumUnknown::ForceDensity,
        false,
        false,
        false,
    )
    .unwrap();
    let q_mp = apply_pseudoinverse(
        &density_system.a,
        &density_system.p,
        options(32).tolerance,
        0,
    )
    .unwrap();
    let q_from_t = solve_inverse_fdm_default(&problem, &target, false, false, false).unwrap();
    let difference = q_mp
        .iter()
        .zip(&q_from_t)
        .map(|(left, right)| (left - right).powi(2))
        .sum::<f64>()
        .sqrt();

    assert!(difference > 1e-4, "q-MP and mapped t-MP unexpectedly agree");
}

#[test]
fn length_retraction_restores_distinct_member_lengths() {
    let positions = [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.8, 1.1, 0.0]];
    let edges = [(2, 0), (2, 1)];
    let (problem, target) = problem(&edges, &positions, vec![2], vec![0, 1], vec![0.0; 3]);
    let reference = EquilibriumSystem::force(&problem, &target, false, false, false).unwrap();
    assert!((reference.lengths[0] - reference.lengths[1]).abs() > 0.5);

    let displaced = Array2::from_shape_vec((1, 3), vec![1.15, 0.72, 0.35]).unwrap();
    let result = retract_member_lengths(
        &problem,
        &displaced,
        &reference.lengths,
        &LengthRetractionOptions {
            tolerance: 1e-11,
            ..LengthRetractionOptions::default()
        },
    )
    .unwrap();
    let restored =
        EquilibriumSystem::force(&problem, &result.positions, false, false, false).unwrap();

    assert!(result.converged);
    assert!(result.iterations > 0);
    for (actual, expected) in restored.lengths.iter().zip(&reference.lengths) {
        assert!((actual - expected).abs() < 1e-9);
    }
}

#[test]
fn lsqr_retraction_uses_only_radial_row_space_correction() {
    let positions = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
    let edges = [(1, 0)];
    let (problem, _target) = problem(&edges, &positions, vec![1], vec![0], vec![0.0; 3]);
    let displaced = Array2::from_shape_vec((1, 3), vec![1.0, 0.2, 0.0]).unwrap();
    let result = retract_member_lengths(
        &problem,
        &displaced,
        &[1.0],
        &LengthRetractionOptions {
            solver: RowSpaceSolver::Lsqr,
            tolerance: 1e-12,
            ..LengthRetractionOptions::default()
        },
    )
    .unwrap();

    assert!(result.converged);
    let radius = (result.positions[[0, 0]].powi(2) + result.positions[[0, 1]].powi(2)).sqrt();
    assert!((radius - 1.0).abs() < 1e-10);
    assert!(result.positions[[0, 1]] > 0.15);
}

#[test]
fn geometric_stiffness_classifies_stable_finite_and_unstable_modes() {
    let positions = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
    let edges = [(1, 0)];
    let (problem, target) = problem(&edges, &positions, vec![1], vec![0], vec![0.0; 3]);
    let system = EquilibriumSystem::force(&problem, &target, false, false, false).unwrap();
    // Transverse free-node displacement is an infinitesimal mechanism.
    let mechanism = Array2::from_shape_vec((3, 1), vec![0.0, 1.0, 0.0]).unwrap();

    for (force, expected) in [
        (4.0, MechanismClass::PrestressStable),
        (0.0, MechanismClass::FiniteCandidate),
        (-4.0, MechanismClass::PrestressUnstable),
    ] {
        let result =
            classify_prestress_stability(&problem, &system, &[force], &mechanism, 1e-10).unwrap();
        assert_eq!(result.classes, vec![expected]);
        assert!((result.restricted_stiffness[[0, 0]] - force / 2.0).abs() < 1e-12);
    }
}

#[test]
fn geometric_stiffness_diagonalizes_coupled_mechanism_basis() {
    let positions = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
    let edges = [(1, 0)];
    let (problem, target) = problem(&edges, &positions, vec![1], vec![0], vec![0.0; 3]);
    let system = EquilibriumSystem::force(&problem, &target, false, false, false).unwrap();
    let mechanisms = Array2::from_shape_vec((3, 2), vec![0.0, 0.0, 1.0, 1.0, 0.0, 1.0]).unwrap();
    let result =
        classify_prestress_stability(&problem, &system, &[2.0], &mechanisms, 1e-12).unwrap();

    assert_eq!(result.classes.len(), 2);
    assert!(result.eigenvalues[0] >= -1e-12);
    let root_five = 5.0_f64.sqrt();
    assert!((result.eigenvalues[0] - (3.0 - root_five) / 2.0).abs() < 1e-10);
    assert!((result.eigenvalues[1] - (3.0 + root_five) / 2.0).abs() < 1e-10);
}
