//! Fresh-process paper benchmarks for the null-space explorer.
//!
//! Each benchmark cell is run by a new invocation of this test executable so
//! peak working set and out-of-memory failures belong to one method/fixture.
//! See `crates/theseus/BENCHMARKS.md` for commands and metric definitions.

use dyn_stack::{GlobalPodBuffer, PodStack};
use faer_core::{Conj, Mat, Parallelism};
use faer_sparse::qr::{factorize_symbolic_qr, QrSymbolicParams};
use faer_svd::{compute_svd, compute_svd_req, ComputeVectors, SvdParams};
use ndarray::Array2;
use std::env;
use std::process::Command;
use std::time::Instant;
use theseus::nullspace::{
    analyze_projector, solve_saddle_pseudoinverse, EquilibriumSystem, NullspaceOptions,
    NullspaceReport,
};
use theseus::sparse::SparseColMatOwned;
use theseus::types::{
    AnchorInfo, Bounds, Factorization, FactorizationStrategy, NetworkTopology, Problem,
    SolverOptions,
};

const CHILD_ENV: &str = "THESEUS_NULLSPACE_BENCH_CHILD";
const FIXTURE_ENV: &str = "THESEUS_NULLSPACE_FIXTURES";
const GRID_ENV: &str = "THESEUS_NULLSPACE_GRIDS";
const METHOD_ENV: &str = "THESEUS_NULLSPACE_METHODS";
const LAMBDA_ENV: &str = "THESEUS_NULLSPACE_LAMBDAS";

#[derive(Clone)]
struct Fixture {
    name: String,
    system: EquilibriumSystem,
}

#[derive(Default)]
struct Metrics {
    status: String,
    rank: Option<usize>,
    s: Option<usize>,
    m_raw: Option<usize>,
    calladine: Option<isize>,
    an: Option<f64>,
    at_phi: Option<f64>,
    load_residual: Option<f64>,
    self_angle_deg: Option<f64>,
    mechanism_angle_deg: Option<f64>,
    note: String,
}

struct DenseBench {
    rank: usize,
    s: usize,
    m_raw: usize,
    self_stress: Array2<f64>,
    mechanisms: Array2<f64>,
    load_residual: f64,
}

#[test]
#[ignore = "manual paper benchmark; fresh process per method and fixture"]
fn bench_nullspace_fresh_processes() {
    if let Ok(spec) = env::var(CHILD_ENV) {
        run_child(&spec);
        return;
    }

    let fixtures = fixture_names();
    let methods = list_env(METHOD_ENV, "dense,projector,qr,angles,saddle,gram");
    let lambdas = list_env(LAMBDA_ENV, "0,1e-10,1e-6");
    println!(
        "RESULT\tfixture\tmethod\tlambda\trows\tcols\tnnz\twall_ms\tpeak_mib\tstatus\trank\ts\tm_raw\tcalladine\tAN_F\tATPhi_F\tload_residual\tself_angle_deg\tmechanism_angle_deg\tnote"
    );

    for fixture in fixtures {
        for method in &methods {
            let method_lambdas: Vec<&str> = if method == "saddle" || method == "gram" {
                lambdas.iter().map(String::as_str).collect()
            } else {
                vec!["-"]
            };
            for lambda in method_lambdas {
                let spec = format!("{fixture}|{method}|{lambda}");
                let output = Command::new(env::current_exe().expect("current test executable"))
                    .arg("--exact")
                    .arg("bench_nullspace_fresh_processes")
                    .arg("--ignored")
                    .arg("--nocapture")
                    .env(CHILD_ENV, &spec)
                    .output()
                    .expect("launch benchmark child");
                let stdout = String::from_utf8_lossy(&output.stdout);
                for line in stdout.lines().filter(|line| line.starts_with("RESULT\t")) {
                    println!("{line}");
                }
                if !output.status.success() {
                    println!(
                        "RESULT\t{fixture}\t{method}\t{lambda}\t-\t-\t-\t-\t-\tprocess-failed\t-\t-\t-\t-\t-\t-\t-\t-\t-\texit={}",
                        output
                            .status
                            .code()
                            .map_or_else(|| "terminated".into(), |code| code.to_string())
                    );
                }
            }
        }
    }
}

fn run_child(spec: &str) {
    let mut fields = spec.split('|');
    let fixture_name = fields.next().expect("fixture");
    let method = fields.next().expect("method");
    let lambda_text = fields.next().unwrap_or("-");
    let fixture = make_fixture(fixture_name);
    let start = Instant::now();
    let metrics = match method {
        "dense" => dense_metrics(&fixture.system),
        "projector" => measured_report_metrics(
            &fixture.system,
            analyze_projector(&fixture.system, &options()),
        ),
        "qr" => qr_metrics(&fixture.system),
        "angles" => angle_metrics(&fixture.system),
        "saddle" => particular_metrics(
            &fixture.system,
            lambda_text.parse().expect("numeric saddle lambda"),
            false,
        ),
        "gram" => particular_metrics(
            &fixture.system,
            lambda_text.parse().expect("numeric Gram lambda"),
            true,
        ),
        other => panic!("unknown benchmark method {other}"),
    };
    let elapsed_ms = start.elapsed().as_secs_f64() * 1e3;
    let a = &fixture.system.a;
    println!(
        "RESULT\t{}\t{}\t{}\t{}\t{}\t{}\t{:.6}\t{:.3}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
        fixture.name,
        method,
        lambda_text,
        a.nrows,
        a.ncols,
        a.nnz(),
        elapsed_ms,
        peak_working_set_bytes() as f64 / (1024.0 * 1024.0),
        text(&metrics.status),
        opt_usize(metrics.rank),
        opt_usize(metrics.s),
        opt_usize(metrics.m_raw),
        opt_isize(metrics.calladine),
        opt_float(metrics.an),
        opt_float(metrics.at_phi),
        opt_float(metrics.load_residual),
        opt_float(metrics.self_angle_deg),
        opt_float(metrics.mechanism_angle_deg),
        text(&metrics.note),
    );
}

fn options() -> NullspaceOptions {
    NullspaceOptions {
        max_modes: env::var("THESEUS_NULLSPACE_MAX_MODES")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(32),
        tolerance: 1e-9,
        ..NullspaceOptions::default()
    }
}

fn dense_metrics(system: &EquilibriumSystem) -> Metrics {
    match dense_bench(system) {
        Ok(report) => Metrics {
            status: "ok".into(),
            rank: Some(report.rank),
            s: Some(report.s),
            m_raw: Some(report.m_raw),
            calladine: Some(
                report.s as isize
                    - report.m_raw as isize
                    - (system.n_edges as isize - system.n_eq as isize),
            ),
            an: Some(basis_residual(&system.a, &report.self_stress, false)),
            at_phi: Some(basis_residual(&system.a, &report.mechanisms, true)),
            load_residual: Some(report.load_residual),
            note: format!(
                "raw dense bases; returned_s={},returned_m_raw={}",
                report.self_stress.ncols(),
                report.mechanisms.ncols()
            ),
            ..Metrics::default()
        },
        Err(error) => Metrics {
            status: "error".into(),
            note: error,
            ..Metrics::default()
        },
    }
}

fn dense_bench(system: &EquilibriumSystem) -> Result<DenseBench, String> {
    let a = &system.a;
    let mut matrix = Mat::<f64>::zeros(a.nrows, a.ncols);
    for col in 0..a.ncols {
        for nz in a.col_ptrs[col] as usize..a.col_ptrs[col + 1] as usize {
            matrix.write(a.row_indices[nz] as usize, col, a.values[nz]);
        }
    }
    let k = a.nrows.min(a.ncols);
    let mut singular = Mat::<f64>::zeros(k, 1);
    let mut u = Mat::<f64>::zeros(a.nrows, k);
    let mut v = Mat::<f64>::zeros(a.ncols, k);
    let params = SvdParams::default();
    let req = compute_svd_req::<f64>(
        a.nrows,
        a.ncols,
        ComputeVectors::Thin,
        ComputeVectors::Thin,
        Parallelism::Rayon(0),
        params,
    )
    .map_err(|error| format!("dense SVD workspace: {error:?}"))?;
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
    let sigma_max = (0..k).map(|i| singular.read(i, 0)).fold(0.0_f64, f64::max);
    let threshold = options()
        .tolerance
        .max(f64::EPSILON * a.nrows.max(a.ncols) as f64)
        * sigma_max.max(1.0);
    let nonzero: Vec<usize> = (0..k)
        .filter(|&i| singular.read(i, 0) > threshold)
        .collect();
    let rank = nonzero.len();
    let cap = options().max_modes;
    // faer-svd 0.17's thin left vectors are not reliable for the exact-zero
    // complement of these highly rank-deficient rectangular fixtures. Build
    // the same rank-dimensional range directly from dense A columns, while
    // retaining SVD for rank and the right singular vectors.
    let left_range = column_range(a, rank, threshold);
    let right_range = Array2::from_shape_fn((a.ncols, rank), |(i, j)| v.read(i, nonzero[j]));
    let self_stress = orthogonal_complement(&right_range, (a.ncols - rank).min(cap), threshold);
    let mechanisms = orthogonal_complement(&left_range, (a.nrows - rank).min(cap), threshold);
    let particular = solve_saddle_pseudoinverse(a, &system.p, 0.0, options().tolerance, 0)
        .map_err(|error| error.to_string())?;
    Ok(DenseBench {
        rank,
        s: a.ncols - rank,
        m_raw: a.nrows - rank,
        self_stress,
        mechanisms,
        load_residual: relative_residual(a, &particular, &system.p),
    })
}

fn orthogonal_complement(range: &Array2<f64>, wanted: usize, tolerance: f64) -> Array2<f64> {
    let dimension = range.nrows();
    let mut accepted: Vec<Vec<f64>> = Vec::with_capacity(wanted);
    for pivot in 0..dimension {
        if accepted.len() == wanted {
            break;
        }
        let mut vector = vec![0.0; dimension];
        vector[pivot] = 1.0;
        for _ in 0..2 {
            for col in 0..range.ncols() {
                let dot: f64 = (0..dimension).map(|i| range[[i, col]] * vector[i]).sum();
                for i in 0..dimension {
                    vector[i] -= dot * range[[i, col]];
                }
            }
            for basis in &accepted {
                let dot: f64 = vector.iter().zip(basis).map(|(a, b)| a * b).sum();
                for i in 0..dimension {
                    vector[i] -= dot * basis[i];
                }
            }
        }
        let length = norm(&vector);
        if length > tolerance.max(1e-12) {
            for value in &mut vector {
                *value /= length;
            }
            accepted.push(vector);
        }
    }
    Array2::from_shape_fn((dimension, accepted.len()), |(i, j)| accepted[j][i])
}

fn column_range(a: &SparseColMatOwned, rank: usize, tolerance: f64) -> Array2<f64> {
    let mut accepted: Vec<Vec<f64>> = Vec::with_capacity(rank);
    for col in 0..a.ncols {
        if accepted.len() == rank {
            break;
        }
        let mut vector = vec![0.0; a.nrows];
        for nz in a.col_ptrs[col] as usize..a.col_ptrs[col + 1] as usize {
            vector[a.row_indices[nz] as usize] = a.values[nz];
        }
        for _ in 0..2 {
            for basis in &accepted {
                let dot: f64 = vector.iter().zip(basis).map(|(x, y)| x * y).sum();
                for i in 0..vector.len() {
                    vector[i] -= dot * basis[i];
                }
            }
        }
        let length = norm(&vector);
        if length > tolerance.max(1e-12) {
            for value in &mut vector {
                *value /= length;
            }
            accepted.push(vector);
        }
    }
    Array2::from_shape_fn((a.nrows, accepted.len()), |(i, j)| accepted[j][i])
}

fn measured_report_metrics(
    system: &EquilibriumSystem,
    result: Result<NullspaceReport, theseus::TheseusError>,
) -> Metrics {
    match result {
        Ok(report) => Metrics {
            status: "ok".into(),
            rank: Some(report.rank),
            s: Some(report.s),
            m_raw: Some(report.m_raw),
            calladine: Some(
                report.s as isize
                    - report.m_raw as isize
                    - (system.n_edges as isize - system.n_eq as isize),
            ),
            an: Some(basis_residual(&system.a, &report.self_stress, false)),
            at_phi: Some(basis_residual(&system.a, &report.mechanisms, true)),
            load_residual: Some(report.residual_ratio),
            note: format!(
                "m={},rigid={},returned_s={},returned_m={}",
                report.m,
                report.n_rigid,
                report.self_stress.ncols(),
                report.mechanisms.ncols()
            ),
            ..Metrics::default()
        },
        Err(error) => Metrics {
            status: "error".into(),
            note: error.to_string(),
            ..Metrics::default()
        },
    }
}

fn qr_metrics(system: &EquilibriumSystem) -> Metrics {
    let a = &system.a;
    if a.nrows < a.ncols {
        return Metrics {
            status: "unsupported".into(),
            note: "faer sparse QR requires rows>=columns".into(),
            ..Metrics::default()
        };
    }
    let a_ref = a.as_faer_ref();
    let symbolic = match factorize_symbolic_qr(a_ref.symbolic(), QrSymbolicParams::default()) {
        Ok(value) => value,
        Err(error) => {
            return Metrics {
                status: "error".into(),
                note: format!("COLAMD symbolic QR: {error:?}"),
                ..Metrics::default()
            }
        }
    };
    let mut indices = vec![0_u32; symbolic.len_indices()];
    let mut values = vec![0.0; symbolic.len_values()];
    let req = symbolic
        .factorize_numeric_qr_req::<f64>(Parallelism::Rayon(0))
        .expect("QR numeric workspace");
    let mut factor_mem = GlobalPodBuffer::new(req);
    let qr = symbolic.factorize_numeric_qr(
        &mut indices,
        values.as_mut_slice(),
        a_ref,
        Parallelism::Rayon(0),
        PodStack::new(&mut factor_mem),
    );
    let mut rhs = Mat::<f64>::zeros(a.nrows, 1);
    for (i, &value) in system.p.iter().enumerate() {
        rhs.write(i, 0, value);
    }
    let req = symbolic
        .solve_in_place_req::<f64>(1, Parallelism::Rayon(0))
        .expect("QR solve workspace");
    let mut solve_mem = GlobalPodBuffer::new(req);
    qr.solve_in_place_with_conj(
        Conj::No,
        rhs.as_mut(),
        Parallelism::Rayon(0),
        PodStack::new(&mut solve_mem),
    );
    let x: Vec<f64> = (0..a.ncols).map(|i| rhs.read(i, 0)).collect();
    if x.iter().any(|value| !value.is_finite()) {
        return Metrics {
            status: "unsupported-rank-deficient".into(),
            note: "QR basic solve produced non-finite values; no rank published".into(),
            ..Metrics::default()
        };
    }
    let ratio = relative_residual(a, &x, &system.p);
    Metrics {
        status: if ratio.is_finite() {
            "ok-residual-checked".into()
        } else {
            "error".into()
        },
        load_residual: Some(ratio),
        note: "COLAMD ordering; comparison solve only; no rank published".into(),
        ..Metrics::default()
    }
}

fn particular_metrics(system: &EquilibriumSystem, lambda: f64, gram: bool) -> Metrics {
    let solve = if gram {
        solve_gram(system, lambda)
    } else {
        solve_saddle_pseudoinverse(&system.a, &system.p, lambda, 1e-9, 0)
            .map_err(|error| error.to_string())
    };
    match solve {
        Ok(x) if x.iter().all(|value| value.is_finite()) => Metrics {
            status: "ok".into(),
            load_residual: Some(relative_residual(&system.a, &x, &system.p)),
            note: if gram {
                "normal equations (condition squared)".into()
            } else if lambda == 0.0 {
                "Moore-Penrose saddle with LSQR fallback".into()
            } else {
                "Tikhonov saddle".into()
            },
            ..Metrics::default()
        },
        Ok(_) => Metrics {
            status: if gram && lambda == 0.0 {
                "unsupported-singular".into()
            } else {
                "error".into()
            },
            note: "non-finite particular".into(),
            ..Metrics::default()
        },
        Err(error) => Metrics {
            status: if gram && lambda == 0.0 {
                "unsupported-singular".into()
            } else {
                "error".into()
            },
            note: error,
            ..Metrics::default()
        },
    }
}

fn solve_gram(system: &EquilibriumSystem, lambda: f64) -> Result<Vec<f64>, String> {
    let at = system.a.transpose();
    let mut gram = SparseColMatOwned::sparse_times_sparse(&at, &system.a)
        .map_err(|error| error.to_string())?;
    if lambda > 0.0 {
        gram.add_diagonal(lambda);
    }
    let rhs = at.matvec(&system.p);
    let mut factor_mem = GlobalPodBuffer::new(dyn_stack::StackReq::empty());
    let factor = Factorization::new(&gram, FactorizationStrategy::LDL, &mut factor_mem)
        .map_err(|error| error.to_string())?;
    let mut solve_mem = GlobalPodBuffer::new(dyn_stack::StackReq::empty());
    let mut workspace = vec![0.0; rhs.len().max(1)];
    factor
        .solve(&rhs, &mut workspace, &mut solve_mem)
        .map_err(|error| error.to_string())
}

fn angle_metrics(system: &EquilibriumSystem) -> Metrics {
    let dense = match dense_bench(system) {
        Ok(report) => report,
        Err(error) => {
            return Metrics {
                status: "reference-error".into(),
                note: error,
                ..Metrics::default()
            }
        }
    };
    let projector = match analyze_projector(system, &options()) {
        Ok(report) => report,
        Err(error) => {
            return Metrics {
                status: "projector-error".into(),
                note: error.to_string(),
                ..Metrics::default()
            }
        }
    };
    let counts_match =
        (dense.rank, dense.s, dense.m_raw) == (projector.rank, projector.s, projector.m_raw);
    Metrics {
        status: if counts_match {
            "ok".into()
        } else {
            "count-mismatch".into()
        },
        rank: Some(projector.rank),
        s: Some(projector.s),
        m_raw: Some(projector.m_raw),
        calladine: Some(
            projector.s as isize
                - projector.m_raw as isize
                - (system.n_edges as isize - system.n_eq as isize),
        ),
        self_angle_deg: if dense.s <= options().max_modes {
            principal_angle_bound(&dense.self_stress, &projector.self_stress)
        } else {
            None
        },
        mechanism_angle_deg: if projector.n_rigid == 0 && dense.m_raw <= options().max_modes {
            principal_angle_bound(&dense.mechanisms, &projector.mechanisms)
        } else {
            None
        },
        note: "separate comparison process; excluded from method memory figures".into(),
        ..Metrics::default()
    }
}

fn principal_angle_bound(reference: &Array2<f64>, candidate: &Array2<f64>) -> Option<f64> {
    if reference.ncols() == 0 && candidate.ncols() == 0 {
        return Some(0.0);
    }
    if reference.ncols() != candidate.ncols() || reference.nrows() != candidate.nrows() {
        return None;
    }
    let k = reference.ncols();
    let cross = Mat::from_fn(k, k, |i, j| {
        (0..reference.nrows())
            .map(|row| reference[[row, i]] * candidate[[row, j]])
            .sum::<f64>()
    });
    let mut singular = Mat::<f64>::zeros(k, 1);
    let params = SvdParams::default();
    let req = compute_svd_req::<f64>(
        k,
        k,
        ComputeVectors::No,
        ComputeVectors::No,
        Parallelism::None,
        params,
    )
    .expect("principal-angle SVD workspace");
    let mut memory = GlobalPodBuffer::new(req);
    compute_svd(
        cross.as_ref(),
        singular.as_mut(),
        None,
        None,
        Parallelism::None,
        PodStack::new(&mut memory),
        params,
    );
    let sigma_min = (0..k)
        .map(|i| singular.read(i, 0))
        .fold(1.0_f64, f64::min)
        .clamp(0.0, 1.0);
    Some(sigma_min.acos().to_degrees())
}

fn basis_residual(a: &SparseColMatOwned, basis: &Array2<f64>, transpose: bool) -> f64 {
    let operator = if transpose { a.transpose() } else { a.clone() };
    (0..basis.ncols())
        .flat_map(|j| {
            let column: Vec<f64> = (0..basis.nrows()).map(|i| basis[[i, j]]).collect();
            operator.matvec(&column)
        })
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt()
}

fn relative_residual(a: &SparseColMatOwned, x: &[f64], rhs: &[f64]) -> f64 {
    let mut residual = a.matvec(x);
    for (value, target) in residual.iter_mut().zip(rhs) {
        *value -= target;
    }
    norm(&residual) / norm(rhs).max(f64::EPSILON)
}

fn fixture_names() -> Vec<String> {
    let mut names = list_env(FIXTURE_ENV, "four-bar,prestressed-triangle,planar-z-grid");
    for size in list_env(GRID_ENV, "4,8,12") {
        let n: usize = size
            .parse()
            .unwrap_or_else(|_| panic!("invalid grid size {size}"));
        if n < 3 {
            panic!("grid size must be at least 3");
        }
        names.push(format!("grid-{n}"));
    }
    names
}

fn make_fixture(name: &str) -> Fixture {
    match name {
        "four-bar" => {
            let a = SparseColMatOwned::from_coo(
                5,
                4,
                &[0, 1, 2, 3, 4, 4],
                &[0, 1, 2, 3, 0, 1],
                &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            )
            .unwrap();
            Fixture {
                name: name.into(),
                system: EquilibriumSystem {
                    a,
                    p: vec![1.0, -0.5, 0.25, 0.75, 0.5],
                    lengths: vec![1.0; 4],
                    n_eq: 5,
                    n_edges: 4,
                    free_positions: Array2::zeros((0, 3)),
                    n_free: 0,
                },
            }
        }
        "prestressed-triangle" => {
            let positions = [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.6, 0.7, 0.0],
            ];
            assembled_fixture(
                name,
                &[(3, 0), (3, 1), (3, 2)],
                &positions,
                vec![3],
                vec![0, 1, 2],
                vec![0.3, -0.2, 0.0],
            )
        }
        "planar-z-grid" => make_grid_fixture(name, 4, true),
        _ if name.starts_with("grid-") => {
            let n = name[5..].parse().expect("grid-N fixture");
            make_grid_fixture(name, n, false)
        }
        _ => panic!("unknown fixture {name}"),
    }
}

fn make_grid_fixture(name: &str, n: usize, planar: bool) -> Fixture {
    let mut positions = Vec::with_capacity(n * n);
    for row in 0..n {
        for col in 0..n {
            let z = if planar {
                0.0
            } else {
                -0.08 * (row as f64 * 0.7).sin() * (col as f64 * 0.9).sin()
            };
            positions.push([col as f64, row as f64, z]);
        }
    }
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
    for i in 0..free.len() {
        loads[3 * i + 2] = -1.0;
    }
    assembled_fixture(name, &edges, &positions, free, fixed, loads)
}

fn assembled_fixture(
    name: &str,
    edges: &[(usize, usize)],
    positions: &[[f64; 3]],
    free: Vec<usize>,
    fixed: Vec<usize>,
    loads: Vec<f64>,
) -> Fixture {
    let incidence = incidence(edges, positions.len());
    let target = Array2::from_shape_fn((free.len(), 3), |(i, d)| positions[free[i]][d]);
    let fixed_positions = Array2::from_shape_fn((fixed.len(), 3), |(i, d)| positions[fixed[i]][d]);
    let problem = Problem {
        topology: NetworkTopology {
            free_incidence: incidence.extract_columns(&free),
            fixed_incidence: incidence.extract_columns(&fixed),
            incidence,
            num_edges: edges.len(),
            num_nodes: positions.len(),
            free_node_indices: free,
            fixed_node_indices: fixed,
        },
        free_node_loads: Array2::from_shape_vec((target.nrows(), 3), loads).unwrap(),
        fixed_node_positions: fixed_positions.clone(),
        anchors: AnchorInfo::all_fixed(fixed_positions),
        objectives: Vec::new(),
        bounds: Bounds::default_for(edges.len()),
        solver: SolverOptions::default(),
        self_weight: None,
        pressure: None,
    };
    Fixture {
        name: name.into(),
        system: EquilibriumSystem::force(&problem, &target, false, false, false).unwrap(),
    }
}

fn incidence(edges: &[(usize, usize)], nodes: usize) -> SparseColMatOwned {
    let mut rows = Vec::with_capacity(2 * edges.len());
    let mut cols = Vec::with_capacity(2 * edges.len());
    let mut values = Vec::with_capacity(2 * edges.len());
    for (edge, &(start, end)) in edges.iter().enumerate() {
        rows.extend([edge, edge]);
        cols.extend([start, end]);
        values.extend([-1.0, 1.0]);
    }
    SparseColMatOwned::from_coo(edges.len(), nodes, &rows, &cols, &values).unwrap()
}

fn list_env(name: &str, default: &str) -> Vec<String> {
    env::var(name)
        .unwrap_or_else(|_| default.into())
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
        .collect()
}

fn norm(values: &[f64]) -> f64 {
    values.iter().map(|value| value * value).sum::<f64>().sqrt()
}

fn opt_float(value: Option<f64>) -> String {
    value.map_or_else(|| "-".into(), |value| format!("{value:.6e}"))
}

fn opt_usize(value: Option<usize>) -> String {
    value.map_or_else(|| "-".into(), |value| value.to_string())
}

fn opt_isize(value: Option<isize>) -> String {
    value.map_or_else(|| "-".into(), |value| value.to_string())
}

fn text(value: &str) -> String {
    value.replace(['\t', '\r', '\n'], " ")
}

#[cfg(windows)]
fn peak_working_set_bytes() -> u64 {
    #[repr(C)]
    struct ProcessMemoryCounters {
        cb: u32,
        page_fault_count: u32,
        peak_working_set_size: usize,
        working_set_size: usize,
        quota_peak_paged_pool_usage: usize,
        quota_paged_pool_usage: usize,
        quota_peak_non_paged_pool_usage: usize,
        quota_non_paged_pool_usage: usize,
        pagefile_usage: usize,
        peak_pagefile_usage: usize,
    }
    #[link(name = "kernel32")]
    extern "system" {
        fn GetCurrentProcess() -> isize;
    }
    #[link(name = "psapi")]
    extern "system" {
        fn GetProcessMemoryInfo(
            process: isize,
            counters: *mut ProcessMemoryCounters,
            size: u32,
        ) -> i32;
    }
    let mut counters = ProcessMemoryCounters {
        cb: std::mem::size_of::<ProcessMemoryCounters>() as u32,
        page_fault_count: 0,
        peak_working_set_size: 0,
        working_set_size: 0,
        quota_peak_paged_pool_usage: 0,
        quota_paged_pool_usage: 0,
        quota_peak_non_paged_pool_usage: 0,
        quota_non_paged_pool_usage: 0,
        pagefile_usage: 0,
        peak_pagefile_usage: 0,
    };
    let ok = unsafe {
        GetProcessMemoryInfo(
            GetCurrentProcess(),
            &mut counters,
            std::mem::size_of::<ProcessMemoryCounters>() as u32,
        )
    };
    if ok == 0 {
        0
    } else {
        counters.peak_working_set_size as u64
    }
}

#[cfg(target_os = "linux")]
fn peak_working_set_bytes() -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|status| {
            status.lines().find_map(|line| {
                line.strip_prefix("VmHWM:")
                    .and_then(|value| value.split_whitespace().next())
                    .and_then(|value| value.parse::<u64>().ok())
                    .map(|kib| kib * 1024)
            })
        })
        .unwrap_or(0)
}

#[cfg(not(any(windows, target_os = "linux")))]
fn peak_working_set_bytes() -> u64 {
    0
}
