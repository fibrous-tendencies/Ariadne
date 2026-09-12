/// Default inverse-FDM particular: Clarabel in member-force coordinates.
pub fn solve_inverse_fdm_default(
    problem: &Problem,
    target_free_xyz: &Array2<f64>,
    enforce_zero_rx: bool,
    enforce_zero_ry: bool,
    enforce_zero_rz: bool,
) -> Result<Vec<f64>, TheseusError> {
    solve_inverse_fdm(
        problem,
        target_free_xyz,
        InverseFdmOptions::direct_unconstrained(
            0.0,
            true,
            20,
            ParticularMethod::Clarabel,
            enforce_zero_rx,
            enforce_zero_ry,
            enforce_zero_rz,
            false,
        ),
    )
    .map(|result| result.q)
}

#[derive(Clone, Copy)]
enum InnerKind {
    Gram,
    Saddle,
    Qr,
    Lsqr,
    Clarabel,
    Spg,
}

fn pick_inner(opts: &InverseFdmOptions, bounds: &BoxBounds) -> InnerKind {
    if bounds.has_finite() {
        return match opts.linear_algebra {
            LinearAlgebra::Direct => InnerKind::Clarabel,
            LinearAlgebra::Iterative => InnerKind::Spg,
        };
    }
    match opts.linear_algebra {
        LinearAlgebra::Iterative => InnerKind::Lsqr,
        LinearAlgebra::Direct => match opts.particular_method {
            ParticularMethod::Gram => InnerKind::Gram,
            ParticularMethod::Augmented => InnerKind::Saddle,
            ParticularMethod::SparseQr => InnerKind::Qr,
            ParticularMethod::Clarabel => InnerKind::Clarabel,
        },
    }
}

fn pick_stage2_inner(opts: &InverseFdmOptions, bounds: &BoxBounds) -> InnerKind {
    match (opts.linear_algebra, bounds.has_finite()) {
        (LinearAlgebra::Direct, true) => InnerKind::Clarabel,
        (LinearAlgebra::Direct, false) => InnerKind::Saddle,
        (LinearAlgebra::Iterative, true) => InnerKind::Spg,
        (LinearAlgebra::Iterative, false) => InnerKind::Lsqr,
    }
}

fn broadcast_f64(values: &[f64], n: usize, fill: f64) -> Result<Vec<f64>, TheseusError> {
    if values.is_empty() {
        Ok(vec![fill; n])
    } else if values.len() == 1 {
        Ok(vec![values[0]; n])
    } else if values.len() == n {
        Ok(values.to_vec())
    } else {
        Err(TheseusError::Shape(format!(
            "bound list length {} must be 0, 1, or {n}",
            values.len()
        )))
    }
}

fn broadcast_i32(values: &[i32], n: usize) -> Result<Vec<i32>, TheseusError> {
    if values.is_empty() {
        Ok(vec![0; n])
    } else if values.len() == 1 {
        Ok(vec![values[0]; n])
    } else if values.len() == n {
        Ok(values.to_vec())
    } else {
        Err(TheseusError::Shape(format!(
            "sign list length {} must be 0, 1, or {n}",
            values.len()
        )))
    }
}

/// Compose Signs × Lower × Upper into a per-edge box.
pub fn compose_box(
    n: usize,
    signs: &[i32],
    lower: &[f64],
    upper: &[f64],
) -> Result<BoxBounds, TheseusError> {
    let signs = broadcast_i32(signs, n)?;
    let mut lo = broadcast_f64(lower, n, f64::NEG_INFINITY)?;
    let mut hi = broadcast_f64(upper, n, f64::INFINITY)?;
    for i in 0..n {
        if signs[i] > 0 {
            lo[i] = lo[i].max(0.0);
        } else if signs[i] < 0 {
            hi[i] = hi[i].min(0.0);
        }
        if lo[i] > hi[i] {
            return Err(TheseusError::Solver(format!(
                "empty bound interval at edge {i}: [{}, {}]",
                lo[i], hi[i]
            )));
        }
    }
    Ok(BoxBounds {
        lower: lo,
        upper: hi,
    })
}

/// Convert public q bounds to member-force bounds `t = L*q` for Stage 1.
fn q_box_to_force_box(q_bounds: &BoxBounds, lengths: &[f64]) -> Result<BoxBounds, TheseusError> {
    let mut lower = Vec::with_capacity(lengths.len());
    let mut upper = Vec::with_capacity(lengths.len());
    for (edge, ((&lo, &hi), &length)) in q_bounds
        .lower
        .iter()
        .zip(&q_bounds.upper)
        .zip(lengths)
        .enumerate()
    {
        if !length.is_finite() || length <= 0.0 {
            return Err(TheseusError::Solver(format!(
                "cannot transform q bounds to member-force bounds at edge {edge}: \
                 target length is {length}"
            )));
        }
        lower.push(lo * length);
        upper.push(hi * length);
    }
    Ok(BoxBounds { lower, upper })
}

fn feasible_start(bounds: &BoxBounds) -> Vec<f64> {
    bounds
        .lower
        .iter()
        .zip(&bounds.upper)
        .map(|(&lo, &hi)| {
            let guess: f64 = if hi <= 0.0 {
                -1.0
            } else if lo >= 0.0 {
                1.0
            } else {
                0.0
            };
            guess.clamp(lo, hi)
        })
        .collect()
}

fn clip_to_box(x: &mut [f64], bounds: &BoxBounds) {
    for (i, value) in x.iter_mut().enumerate() {
        *value = value.clamp(bounds.lower[i], bounds.upper[i]);
    }
}

fn validate_unknown(x: &[f64], n: usize, label: &str) -> Result<(), TheseusError> {
    if x.len() != n {
        return Err(TheseusError::Shape(format!(
            "{label} returned {} values, expected {n}",
            x.len()
        )));
    }
    for (i, &v) in x.iter().enumerate() {
        if !v.is_finite() {
            return Err(TheseusError::Solver(format!(
                "{label} produced non-finite value at edge {i}"
            )));
        }
    }
    Ok(())
}

fn stack_tikhonov(m_mat: &SparseColMatOwned, lambda: f64) -> SparseColMatOwned {
    if lambda <= 0.0 {
        return m_mat.clone();
    }
    let scale = lambda.sqrt();
    let m = m_mat.nrows;
    let n = m_mat.ncols;
    let mut triplets = Vec::with_capacity(m_mat.nnz() + n);
    for col in 0..n {
        for nz in m_mat.col_ptrs[col] as usize..m_mat.col_ptrs[col + 1] as usize {
            triplets.push((m_mat.row_indices[nz], col as u32, m_mat.values[nz]));
        }
        triplets.push(((m + col) as u32, col as u32, scale));
    }
    SparseColMatOwned::from_triplets(m + n, n, &triplets).expect("stack_tikhonov")
}

fn solve_lsqr_on(
    m_mat: &SparseColMatOwned,
    p: &[f64],
    lambda: f64,
    tol: f64,
    max_iter: usize,
    weight: Option<&MetricWeight>,
) -> Result<crate::nullspace::LsqrResult, TheseusError> {
    let Some(w) = weight else {
        if lambda > 0.0 {
            let stacked = stack_tikhonov(m_mat, lambda);
            let mut rhs = vec![0.0; stacked.nrows];
            rhs[..p.len()].copy_from_slice(p);
            return solve_lsqr(&stacked, &rhs, tol, max_iter);
        }
        return solve_lsqr(m_mat, p, tol, max_iter);
    };

    // Weighted: minimise ‖S⁻¹(Mx − p)‖² + λ‖x‖² over the operator S⁻¹M.
    // The Tikhonov rows are appended inside the closures rather than
    // materialised, since S⁻¹M has no sparse representation.
    let m_rows = m_mat.nrows;
    let n = m_mat.ncols;
    let m_t = m_mat.transpose();
    let damping = if lambda > 0.0 { lambda.sqrt() } else { 0.0 };
    let rows = if damping > 0.0 { m_rows + n } else { m_rows };

    let apply = |x: &[f64]| -> Result<Vec<f64>, TheseusError> {
        let mut out = vec![0.0; rows];
        let weighted = w.apply_inverse(&m_mat.matvec(x))?;
        out[..m_rows].copy_from_slice(&weighted);
        if damping > 0.0 {
            for (j, &xj) in x.iter().enumerate() {
                out[m_rows + j] = damping * xj;
            }
        }
        Ok(out)
    };
    let apply_t = |y: &[f64]| -> Result<Vec<f64>, TheseusError> {
        let weighted = w.apply_inverse(&y[..m_rows])?;
        let mut out = m_t.matvec(&weighted);
        if damping > 0.0 {
            for (j, value) in out.iter_mut().enumerate() {
                *value += damping * y[m_rows + j];
            }
        }
        Ok(out)
    };

    let mut rhs = vec![0.0; rows];
    rhs[..m_rows].copy_from_slice(&w.apply_inverse(p)?);
    crate::nullspace::lsqr_operator(
        apply,
        apply_t,
        rows,
        n,
        &rhs,
        l2_norm_prefix(&m_mat.values, m_mat.values.len()),
        tol,
        max_iter,
    )
}

fn ldl_solve_cached(
    g: &SparseColMatOwned,
    rhs: &[f64],
    lambda: f64,
    cache: &mut Option<Factorization>,
    factor_stack: &mut GlobalPodBuffer,
    solve_stack: &mut GlobalPodBuffer,
) -> Result<Vec<f64>, TheseusError> {
    let factor_result = if cache.is_none() {
        Factorization::new(g, FactorizationStrategy::LDL, factor_stack).map(|fac| {
            *cache = Some(fac);
        })
    } else {
        cache.as_mut().expect("LDL cache").update(g, factor_stack)
    };
    if let Err(error) = factor_result {
        *cache = None;
        match Factorization::new(g, FactorizationStrategy::LDL, factor_stack) {
            Ok(fac) => *cache = Some(fac),
            Err(_) if lambda == 0.0 => {
                return Err(TheseusError::Solver("Gram at λ=0: MᵀM is singular".into()));
            }
            Err(_) => return Err(error),
        }
    }
    let mut workspace = vec![0.0; rhs.len().max(1)];
    match cache
        .as_ref()
        .expect("LDL cache")
        .solve(rhs, &mut workspace, solve_stack)
    {
        Ok(sol) => Ok(sol),
        Err(_) if lambda == 0.0 => {
            *cache = None;
            Err(TheseusError::Solver("Gram at λ=0: MᵀM is singular".into()))
        }
        Err(error) => {
            *cache = None;
            Err(error)
        }
    }
}

fn solve_gram_on(
    m_mat: &SparseColMatOwned,
    p: &[f64],
    lambda: f64,
    cache: &mut Option<Factorization>,
    factor_stack: &mut GlobalPodBuffer,
    solve_stack: &mut GlobalPodBuffer,
) -> Result<Vec<f64>, TheseusError> {
    let m_t = m_mat.transpose();
    let mut g =
        SparseColMatOwned::sparse_times_sparse(&m_t, m_mat).map_err(TheseusError::Solver)?;
    if lambda > 0.0 {
        g.add_diagonal(lambda);
    }
    let h = m_t.matvec(p);
    ldl_solve_cached(&g, &h, lambda, cache, factor_stack, solve_stack)
}

/// Build the weighted 3-block KKT system for `min ½‖e‖² + ½λ‖q‖²`
/// subject to `S e − M q = −p`.
///
/// Stationarity of the Lagrangian in `(e, q, y)` gives the symmetric
/// indefinite system
///
/// ```text
/// [ I    0    Sᵀ ] [e]   [ 0 ]
/// [ 0   λI   −Mᵀ ] [q] = [ 0 ]
/// [ S   −M    0  ] [y]   [−p ]
/// ```
///
/// Every block is sparse, so this is the weighted analogue of
/// [`build_augmented_system`] without ever forming `S⁻¹M`.
fn build_weighted_saddle(
    m_mat: &SparseColMatOwned,
    s_mat: &SparseColMatOwned,
    p: &[f64],
    lambda: f64,
) -> (SparseColMatOwned, Vec<f64>) {
    let m = m_mat.nrows;
    let n = m_mat.ncols;
    let total = 2 * m + n;
    let y0 = m + n;

    let mut triplets: Vec<(u32, u32, f64)> = Vec::with_capacity(m + n + 4 * m_mat.nnz());

    // (1,1) = I
    for i in 0..m {
        triplets.push((i as u32, i as u32, 1.0));
    }
    // (2,2) = λI
    for j in 0..n {
        triplets.push(((m + j) as u32, (m + j) as u32, lambda));
    }
    // (1,3) = Sᵀ and (3,1) = S
    for col in 0..s_mat.ncols {
        for nz in s_mat.col_ptrs[col] as usize..s_mat.col_ptrs[col + 1] as usize {
            let row = s_mat.row_indices[nz] as usize;
            let value = s_mat.values[nz];
            triplets.push((col as u32, (y0 + row) as u32, value));
            triplets.push(((y0 + row) as u32, col as u32, value));
        }
    }
    // (2,3) = −Mᵀ and (3,2) = −M
    for col in 0..n {
        for nz in m_mat.col_ptrs[col] as usize..m_mat.col_ptrs[col + 1] as usize {
            let row = m_mat.row_indices[nz] as usize;
            let value = -m_mat.values[nz];
            triplets.push(((m + col) as u32, (y0 + row) as u32, value));
            triplets.push(((y0 + row) as u32, (m + col) as u32, value));
        }
    }

    let k_mat =
        SparseColMatOwned::from_triplets(total, total, &triplets).expect("build_weighted_saddle");
    let mut rhs = vec![0.0; total];
    for (i, &pi) in p.iter().enumerate() {
        rhs[y0 + i] = -pi;
    }
    (k_mat, rhs)
}

fn solve_saddle_on(
    m_mat: &SparseColMatOwned,
    p: &[f64],
    lambda: f64,
    cache: &mut Option<Factorization>,
    factor_stack: &mut GlobalPodBuffer,
    solve_stack: &mut GlobalPodBuffer,
    weight: Option<&MetricWeight>,
) -> Result<Vec<f64>, TheseusError> {
    let m = p.len();
    let n = m_mat.ncols;
    if let Some(w) = weight {
        let (k_mat, rhs) = build_weighted_saddle(m_mat, &w.s, p, lambda);
        let sol = ldl_solve_cached(&k_mat, &rhs, lambda, cache, factor_stack, solve_stack)?;
        return Ok(sol[m..m + n].to_vec());
    }
    if lambda == 0.0 {
        return solve_saddle_pseudoinverse(m_mat, p, lambda, 1e-11, 0);
    }
    let (k_mat, rhs) = build_augmented_system(m_mat, p, lambda);
    let sol = ldl_solve_cached(&k_mat, &rhs, lambda, cache, factor_stack, solve_stack)?;
    Ok(sol[m..m + n].to_vec())
}

fn solve_qr_on(
    a: &SparseColMatOwned,
    p: &[f64],
    check_rank: bool,
    symbolic: &mut Option<SymbolicQr<u32>>,
) -> Result<Vec<f64>, TheseusError> {
    if a.nrows < a.ncols {
        return Err(TheseusError::Solver(format!(
            "sparse QR particular requires rows >= columns ({} < {})",
            a.nrows, a.ncols
        )));
    }
    let a_ref = a.as_faer_ref();
    if symbolic.is_none() {
        // Force simplicial storage so the sparse R column pointers, row
        // indices, and values have a stable public buffer layout. This lets us
        // inspect pivots without materializing A or A*I as a dense matrix.
        let params = QrSymbolicParams {
            supernodal_flop_ratio_threshold: faer_sparse::SupernodalThreshold::FORCE_SIMPLICIAL,
            ..QrSymbolicParams::default()
        };
        *symbolic = Some(
            factorize_symbolic_qr(a_ref.symbolic(), params).map_err(|error| {
                TheseusError::Linalg(format!("COLAMD sparse QR symbolic: {error:?}"))
            })?,
        );
    }
    let symbolic = symbolic.as_ref().expect("QR symbolic");
    let mut indices = vec![0_u32; symbolic.len_indices()];
    let mut values = vec![0.0; symbolic.len_values()];
    let req = symbolic
        .factorize_numeric_qr_req::<f64>(Parallelism::Rayon(0))
        .map_err(|error| TheseusError::Linalg(format!("sparse QR workspace: {error:?}")))?;
    let mut factor_memory = GlobalPodBuffer::new(req);
    {
        let _factor = symbolic.factorize_numeric_qr(
            &mut indices,
            values.as_mut_slice(),
            a_ref,
            Parallelism::Rayon(0),
            PodStack::new(&mut factor_memory),
        );
    }
    if check_rank {
        // Forced-simplicial numeric storage starts with:
        // indices = [R col_ptrs (n+1), R row_indices, ...]
        // values  = [R values, ...].
        let n = a.ncols;
        let r_nnz = indices[n] as usize;
        let r_rows = &indices[n + 1..n + 1 + r_nnz];
        let r_values = &values[..r_nnz];
        let mut pivots = Vec::with_capacity(n);
        for col in 0..n {
            let start = indices[col] as usize;
            let end = indices[col + 1] as usize;
            let pivot = (start..end)
                .find(|&nz| r_rows[nz] as usize == col)
                .map(|nz| r_values[nz])
                .unwrap_or(0.0);
            pivots.push(pivot);
        }
        let pivot_scale = pivots.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        let pivot_tol = 100.0 * f64::EPSILON * pivot_scale.max(1.0) * a.nrows.max(n).max(1) as f64;
        let rank = pivots
            .iter()
            .filter(|pivot| pivot.is_finite() && pivot.abs() > pivot_tol)
            .count();
        if rank != n {
            return Err(TheseusError::Solver(format!(
                "sparse QR particular requires full column rank; sparse R has \
                 {rank} usable pivots for {n} columns (threshold {pivot_tol:.3e})"
            )));
        }
    }
    // Reborrow the already-computed sparse factors after pivot inspection.
    let qr =
        unsafe { faer_sparse::qr::QrRef::new_unchecked(symbolic, &indices, values.as_slice()) };
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
    let mut rhs = Mat::<f64>::zeros(a.nrows, 1);
    for (row, &value) in p.iter().enumerate() {
        rhs.write(row, 0, value);
    }
    solve(&mut rhs)?;
    Ok((0..a.ncols).map(|i| rhs.read(i, 0)).collect())
}

fn to_clarabel_csc(
    mat: &SparseColMatOwned,
) -> Result<clarabel::algebra::CscMatrix<f64>, TheseusError> {
    let csc = clarabel::algebra::CscMatrix::new(
        mat.nrows,
        mat.ncols,
        mat.col_ptrs.iter().map(|&v| v as usize).collect(),
        mat.row_indices.iter().map(|&v| v as usize).collect(),
        mat.values.clone(),
    );
    csc.check_format()
        .map_err(|error| TheseusError::Solver(format!("Clarabel CSC: {error}")))?;
    Ok(csc)
}

/// Split-variable QP: `min ½‖e‖² + ½λ‖q‖²` s.t. `S e − M q = −p`, `q` in box.
///
/// With no weight, `S = I` pins `e` to the plain force residual `Mq − p`.
/// Under the geometric metric `S` is the sparse Laplacian block, so `e` becomes
/// `S⁻¹(Mq − p)`, the geometric error, and the inverse is never formed.
fn solve_clarabel_once(
    m_mat: &SparseColMatOwned,
    p: &[f64],
    lambda: f64,
    bounds: &BoxBounds,
    weight: Option<&MetricWeight>,
    max_iter: usize,
    tol: f64,
    geometric_step: bool,
) -> Result<Vec<f64>, TheseusError> {
    use clarabel::algebra::CscMatrix;
    use clarabel::solver::{
        DefaultSettings, DefaultSolver, IPSolver, SolverStatus, SupportedConeT,
    };

    let total_started = std::time::Instant::now();
    let m = m_mat.nrows;
    let n = m_mat.ncols;
    let n_var = m + n;

    let mut p_colptr = vec![0usize; n_var + 1];
    let mut p_rowval = Vec::with_capacity(m + n);
    let mut p_nz = Vec::with_capacity(m + n);
    for j in 0..m {
        p_colptr[j] = p_rowval.len();
        p_rowval.push(j);
        p_nz.push(1.0);
    }
    for j in 0..n {
        p_colptr[m + j] = p_rowval.len();
        if lambda > 0.0 {
            p_rowval.push(m + j);
            p_nz.push(lambda);
        }
    }
    p_colptr[n_var] = p_rowval.len();
    let p_mat = CscMatrix::new(n_var, n_var, p_colptr, p_rowval, p_nz);
    p_mat
        .check_format()
        .map_err(|error| TheseusError::Solver(format!("Clarabel P: {error}")))?;
    let q_lin = vec![0.0; n_var];

    let mut a_triplets: Vec<(u32, u32, f64)> = Vec::new();
    match weight {
        None => {
            for i in 0..m {
                a_triplets.push((i as u32, i as u32, 1.0));
            }
        }
        Some(w) => {
            for col in 0..w.s.ncols {
                for nz in w.s.col_ptrs[col] as usize..w.s.col_ptrs[col + 1] as usize {
                    a_triplets.push((w.s.row_indices[nz], col as u32, w.s.values[nz]));
                }
            }
        }
    }
    for col in 0..n {
        for nz in m_mat.col_ptrs[col] as usize..m_mat.col_ptrs[col + 1] as usize {
            a_triplets.push((m_mat.row_indices[nz], (m + col) as u32, -m_mat.values[nz]));
        }
    }
    let mut b = vec![0.0; m];
    for i in 0..m {
        b[i] = -p[i];
    }
    let mut n_ineq = 0usize;
    for i in 0..n {
        if bounds.lower[i].is_finite() {
            a_triplets.push(((m + n_ineq) as u32, (m + i) as u32, -1.0));
            b.push(-bounds.lower[i]);
            n_ineq += 1;
        }
        if bounds.upper[i].is_finite() {
            a_triplets.push(((m + n_ineq) as u32, (m + i) as u32, 1.0));
            b.push(bounds.upper[i]);
            n_ineq += 1;
        }
    }
    let a_mat = SparseColMatOwned::from_triplets(m + n_ineq, n_var, &a_triplets)
        .map_err(TheseusError::Shape)?;
    let a_csc = to_clarabel_csc(&a_mat)?;

    let mut cones = vec![SupportedConeT::ZeroConeT(m)];
    if n_ineq > 0 {
        cones.push(SupportedConeT::NonnegativeConeT(n_ineq));
    }
    let requested_tol = tol.max(1e-10);
    let make_settings = |retry: bool| DefaultSettings::<f64> {
        verbose: false,
        max_iter: max_iter.max(1).min(u32::MAX as usize) as u32,
        tol_gap_abs: requested_tol,
        tol_gap_rel: requested_tol,
        tol_feas: requested_tol,
        presolve_enable: !retry,
        equilibrate_min_scaling: if retry { 1e-8 } else { 1e-4 },
        equilibrate_max_scaling: if retry { 1e8 } else { 1e4 },
        ..Default::default()
    };
    let solve_attempt = |retry: bool| {
        let mut solver =
            DefaultSolver::new(&p_mat, &q_lin, &a_csc, &b, &cones, make_settings(retry));
        solver.solve();
        (
            solver.solution.status,
            solver.solution.x.clone(),
            solver.solution.iterations,
            solver.solution.r_prim,
            solver.solution.r_dual,
            solver.solution.solve_time,
        )
    };

    let mut attempt = solve_attempt(false);
    let mut retried = false;
    if matches!(
        attempt.0,
        SolverStatus::PrimalInfeasible | SolverStatus::MaxIterations
    ) {
        retried = true;
        attempt = solve_attempt(true);
    }
    let (status, solution, iterations, r_prim, r_dual, solve_time) = attempt;
    let diagnostics = || {
        format!(
            "status={status:?}, iterations={iterations}, primal_residual={r_prim:.3e}, \
             dual_residual={r_dual:.3e}, solve_time={solve_time:.6}s, \
             P={}x{} nnz={}, A={}x{} nnz={}",
            n_var,
            n_var,
            p_mat.nzval.len(),
            a_mat.nrows,
            a_mat.ncols,
            a_mat.nnz()
        )
    };
    if !matches!(status, SolverStatus::Solved | SolverStatus::AlmostSolved) {
        let classification = if geometric_step && status == SolverStatus::PrimalInfeasible {
            "Clarabel numerical failure: geometric Δ=0 is feasible"
        } else {
            "Clarabel failed"
        };
        return Err(TheseusError::Solver(format!(
            "{classification} ({})",
            diagnostics()
        )));
    }

    let unknown = &solution[m..];
    let check_tol = 100.0 * requested_tol;
    let mut equality = match weight {
        Some(w) => w.s.matvec(&solution[..m]),
        None => solution[..m].to_vec(),
    };
    let mq = m_mat.matvec(unknown);
    for i in 0..m {
        equality[i] += p[i] - mq[i];
    }
    let equality_scale = 1.0 + l2_norm_prefix(p, p.len());
    let equality_error = l2_norm_prefix(&equality, equality.len()) / equality_scale;
    let mut box_error = 0.0_f64;
    for i in 0..n {
        box_error = box_error.max((bounds.lower[i] - unknown[i]).max(0.0));
        box_error = box_error.max((unknown[i] - bounds.upper[i]).max(0.0));
    }
    if !equality_error.is_finite()
        || !box_error.is_finite()
        || equality_error > check_tol
        || box_error > check_tol
    {
        return Err(TheseusError::Solver(format!(
            "Clarabel returned an inaccurate solution: equality_residual={equality_error:.3e}, \
             box_violation={box_error:.3e}, check_tolerance={check_tol:.3e} ({})",
            diagnostics()
        )));
    }
    if std::env::var_os("THESEUS_CLARABEL_DIAGNOSTICS").is_some() {
        eprintln!(
            "clarabel,geometric_step={geometric_step},retried={retried},\
             iterations={iterations},solver_ms={:.3},total_ms={:.3},\
             primal_residual={r_prim:.3e},dual_residual={r_dual:.3e},\
             rows={},cols={},nnz={}",
            solve_time * 1e3,
            total_started.elapsed().as_secs_f64() * 1e3,
            a_mat.nrows,
            a_mat.ncols,
            a_mat.nnz()
        );
    }
    Ok(unknown.to_vec())
}

fn solve_clarabel_box(
    m_mat: &SparseColMatOwned,
    p: &[f64],
    lambda: f64,
    bounds: &BoxBounds,
    weight: Option<&MetricWeight>,
    max_iter: usize,
    tol: f64,
    geometric_step: bool,
) -> Result<Vec<f64>, TheseusError> {
    solve_clarabel_once(
        m_mat,
        p,
        lambda,
        bounds,
        weight,
        max_iter,
        tol,
        geometric_step,
    )
}

fn solve_spg_on(
    m_mat: &SparseColMatOwned,
    p: &[f64],
    lambda: f64,
    bounds: &BoxBounds,
    max_iter: usize,
    tol: f64,
    x0: &[f64],
    weight: Option<&MetricWeight>,
) -> Result<SpgBoxResult, TheseusError> {
    let n = m_mat.ncols;
    let m_t = m_mat.transpose();
    let mut x = x0.to_vec();
    clip_to_box(&mut x, bounds);
    let mut prev_g = vec![0.0; n];
    let mut prev_x = vec![0.0; n];
    let mut alpha = 1.0;
    let mut iterations = 0;
    let mut converged = false;
    let max_iter = max_iter.max(1);

    for iter in 0..max_iter {
        iterations = iter + 1;
        let mut r = m_mat.matvec(&x);
        for (ri, &pi) in r.iter_mut().zip(p.iter()) {
            *ri -= pi;
        }
        // Objective ½‖S⁻¹(Mx − p)‖² has gradient Mᵀ S⁻ᵀ S⁻¹ (Mx − p); S is
        // symmetric, so S⁻¹ is applied twice.
        if let Some(w) = weight {
            r = w.apply_inverse(&r)?;
            r = w.apply_inverse(&r)?;
        }
        let mut g = m_t.matvec(&r);
        if lambda > 0.0 {
            for (gk, &xk) in g.iter_mut().zip(x.iter()) {
                *gk += lambda * xk;
            }
        }
        if iter > 0 {
            let mut dx_dot_dg = 0.0;
            let mut dg_dot_dg = 0.0;
            for k in 0..n {
                let dx = x[k] - prev_x[k];
                let dg = g[k] - prev_g[k];
                dx_dot_dg += dx * dg;
                dg_dot_dg += dg * dg;
            }
            if dg_dot_dg > 0.0 && dx_dot_dg > 0.0 {
                alpha = dx_dot_dg / dg_dot_dg;
            }
        }
        prev_x.copy_from_slice(&x);
        prev_g.copy_from_slice(&g);
        let mut proj_grad_norm_sq = 0.0;
        for k in 0..n {
            let trial = (x[k] - alpha * g[k]).clamp(bounds.lower[k], bounds.upper[k]);
            let pg = x[k] - trial;
            proj_grad_norm_sq += pg * pg;
            x[k] = trial;
        }
        if proj_grad_norm_sq.sqrt() < tol {
            converged = true;
            break;
        }
    }
    validate_unknown(&x, n, "SPG")?;
    Ok(SpgBoxResult {
        q: x,
        iterations,
        converged,
    })
}

fn irls_weights(residual: &[f64]) -> (Vec<f64>, f64, f64) {
    const ABS_EPS: f64 = 1e-12;
    let max_abs_r = residual.iter().map(|ri| ri.abs()).fold(0.0_f64, f64::max);
    let eps_iter = (1e-4 * max_abs_r).max(ABS_EPS);
    let mut l1_obj = 0.0;
    let mut sqrt_w = vec![0.0; residual.len()];
    for i in 0..residual.len() {
        let abs_r = residual[i].abs();
        l1_obj += abs_r;
        sqrt_w[i] = (1.0 / abs_r.max(eps_iter)).sqrt();
    }
    let w_max = sqrt_w.iter().map(|s| s * s).fold(0.0_f64, f64::max);
    if w_max > 0.0 {
        let inv = 1.0 / w_max.sqrt();
        for sw in sqrt_w.iter_mut() {
            *sw *= inv;
        }
    }
    (sqrt_w, l1_obj, w_max)
}

// ─────────────────────────────────────────────────────────────
//  Geometric metric  (Laplacian compliance weighting)
// ─────────────────────────────────────────────────────────────

/// Euclidean norm of the first `len` entries.
fn l2_norm_prefix(v: &[f64], len: usize) -> f64 {
    v[..len.min(v.len())]
        .iter()
        .map(|x| x * x)
        .sum::<f64>()
        .sqrt()
}

/// Left weight `S` for the geometric metric.
///
/// `S = blkdiag(D, D, D, I)` over the axis-major equilibrium rows, where
/// `D = Cnᵀ diag(q) Cn` is the FDM Laplacian. The trailing identity covers the
/// optional zero-reaction rows, which are support constraints rather than
/// free-node equilibrium and carry no compliance.
///
/// `S` is kept assembled for the backends that embed it (Clarabel, saddle);
/// `D` is kept factored for the backends that apply `S⁻¹` per matvec (LSQR,
/// SPG) and for evaluating the exact geometric error. `D` is never inverted.
struct MetricWeight {
    s: SparseColMatOwned,
    d: SparseColMatOwned,
    d_factor: Factorization,
    n_free: usize,
    n_eq: usize,
    workspace: std::cell::RefCell<Vec<f64>>,
    stack: std::cell::RefCell<GlobalPodBuffer>,
}

impl MetricWeight {
    /// Assemble and factor the weight at force densities `q`.
    fn build(
        problem: &Problem,
        q: &[f64],
        n_eq: usize,
        n_free: usize,
    ) -> Result<Self, TheseusError> {
        let cn = &problem.topology.free_incidence;
        let cn_t = cn.transpose();
        let scaled = row_scaled_copy(cn, q);
        let d =
            SparseColMatOwned::sparse_times_sparse(&cn_t, &scaled).map_err(TheseusError::Solver)?;

        // S = blkdiag(D, D, D, I) in axis-major row order.
        let mut triplets = Vec::with_capacity(3 * d.nnz() + (n_eq - 3 * n_free));
        for col in 0..d.ncols {
            for nz in d.col_ptrs[col] as usize..d.col_ptrs[col + 1] as usize {
                let row = d.row_indices[nz] as usize;
                let value = d.values[nz];
                for axis in 0..3 {
                    let offset = axis * n_free;
                    triplets.push(((offset + row) as u32, (offset + col) as u32, value));
                }
            }
        }
        for row in (3 * n_free)..n_eq {
            triplets.push((row as u32, row as u32, 1.0));
        }
        let s =
            SparseColMatOwned::from_triplets(n_eq, n_eq, &triplets).map_err(TheseusError::Shape)?;

        // Cholesky is valid only when every current q is strictly positive.
        // Compression-only and mixed-sign systems are intentionally handled
        // by LDL; indefiniteness is valid as long as D is nonsingular.
        let strategy = if q.iter().all(|&v| v > 0.0) {
            FactorizationStrategy::Cholesky
        } else {
            FactorizationStrategy::LDL
        };
        let mut stack = GlobalPodBuffer::new(dyn_stack::StackReq::empty());
        let d_factor = Factorization::new(&d, strategy, &mut stack).map_err(|_| {
            TheseusError::Solver(format!(
                "geometric metric: {} factorization of D=Cnᵀ diag(q) Cn failed; \
                 mixed signs and indefiniteness are permitted, but D must be nonsingular \
                 and numerically factorizable by the current sparse LDL ordering",
                if strategy == FactorizationStrategy::Cholesky {
                    "Cholesky"
                } else {
                    "LDL"
                }
            ))
        })?;

        Ok(Self {
            s,
            d,
            d_factor,
            n_free,
            n_eq,
            workspace: std::cell::RefCell::new(vec![0.0; (n_free * 3).max(1)]),
            stack: std::cell::RefCell::new(stack),
        })
    }

    /// Apply `S⁻¹` to an equilibrium-row vector.
    ///
    /// Solves the three axis blocks in one triangular solve and passes the
    /// reaction rows through unchanged.
    fn apply_inverse(&self, r: &[f64]) -> Result<Vec<f64>, TheseusError> {
        if r.len() != self.n_eq {
            return Err(TheseusError::Shape(format!(
                "geometric weight expected {} rows, got {}",
                self.n_eq,
                r.len()
            )));
        }
        let n_free = self.n_free;
        let mut out = vec![0.0; self.n_eq];
        if n_free > 0 {
            let rhs = Array2::from_shape_fn((n_free, 3), |(i, axis)| r[axis * n_free + i]);
            let mut x = Array2::zeros((n_free, 3));
            let mut workspace = self.workspace.borrow_mut();
            let mut stack = self.stack.borrow_mut();
            if workspace.len() < n_free * 3 {
                workspace.resize(n_free * 3, 0.0);
            }
            self.d_factor
                .solve_into(&rhs, &mut x, &mut workspace, &mut stack)?;
            let d_norm = self
                .d
                .values
                .iter()
                .map(|value| value * value)
                .sum::<f64>()
                .sqrt();
            for axis in 0..3 {
                let b: Vec<f64> = (0..n_free).map(|i| rhs[[i, axis]]).collect();
                let y: Vec<f64> = (0..n_free).map(|i| x[[i, axis]]).collect();
                let dy = self.d.matvec(&y);
                let residual = dy
                    .iter()
                    .zip(&b)
                    .map(|(&left, &right)| {
                        let difference = left - right;
                        difference * difference
                    })
                    .sum::<f64>()
                    .sqrt();
                let y_norm = l2_norm_prefix(&y, y.len());
                let b_norm = l2_norm_prefix(&b, b.len());
                let denominator = d_norm * y_norm + b_norm;
                let backward_error = if denominator > 0.0 {
                    residual / denominator
                } else {
                    residual
                };
                if !backward_error.is_finite() || backward_error > 1e-8 {
                    return Err(TheseusError::Solver(format!(
                        "geometric metric: inaccurate D inverse solve on axis {axis}; \
                         scale-aware backward error {backward_error:.3e} exceeds 1.000e-8. \
                         D must be nonsingular and numerically solvable"
                    )));
                }
                for i in 0..n_free {
                    let value = x[[i, axis]];
                    if !value.is_finite() {
                        return Err(TheseusError::Solver(
                            "geometric metric: applying D⁻¹ produced a non-finite value; \
                             the Laplacian is effectively singular at the current q"
                                .into(),
                        ));
                    }
                    out[axis * n_free + i] = value;
                }
            }
        }
        out[3 * n_free..].copy_from_slice(&r[3 * n_free..]);
        Ok(out)
    }
}

/// Current geometry and error for an unknown vector under the geometric metric.
struct GeometryProbe {
    weight: MetricWeight,
    /// `S⁻¹ r = x* − x(q)`, the negated geometric error.
    neg_error: Vec<f64>,
    /// `‖x(q) − x*‖` over the free-node rows only.
    error: f64,
}

/// Evaluate the exact geometric error at an unknown vector.
///
/// Uses the identity `x(q) − x* = −D(q)⁻¹(E(x*)q − p)`, so no forward solve is
/// needed. Exact for geometry-independent loads.
fn probe_geometry(
    problem: &Problem,
    system: &EquilibriumSystem,
    unknown: &[f64],
    solve_for_q: bool,
) -> Result<GeometryProbe, TheseusError> {
    let q = if solve_for_q {
        unknown.to_vec()
    } else {
        forces_to_q(unknown, &system.lengths)
    };
    let weight = MetricWeight::build(problem, &q, system.n_eq, system.n_free)?;
    let mut r = system.a.matvec(unknown);
    for (ri, &pi) in r.iter_mut().zip(system.p.iter()) {
        *ri -= pi;
    }
    let neg_error = weight.apply_inverse(&r)?;
    let error = l2_norm_prefix(&neg_error, 3 * system.n_free);
    Ok(GeometryProbe {
        weight,
        neg_error,
        error,
    })
}

/// Exact offset from the forward-solved geometry to the target, `x(q) − x*`,
/// as an `n_free × 3` array.
///
/// Evaluated through the identity `x(q) − x* = −D(q)⁻¹(E(x*)q − p)` rather than
/// by running a forward solve, so it costs one Laplacian factorisation. Exact
/// whenever the loads do not move with the geometry; with self-weight or
/// pressure active the forward solve is the authority.
pub fn geometric_error_vector(
    problem: &Problem,
    target_free_xyz: &Array2<f64>,
    q: &[f64],
) -> Result<Array2<f64>, TheseusError> {
    let system = EquilibriumSystem::assemble(
        problem,
        target_free_xyz,
        EquilibriumUnknown::ForceDensity,
        false,
        false,
        false,
    )?;
    let probe = probe_geometry(problem, &system, q, true)?;
    let n_free = system.n_free;
    // probe.neg_error is x* − x(q).
    Ok(Array2::from_shape_fn((n_free, 3), |(i, d)| {
        -probe.neg_error[d * n_free + i]
    }))
}

/// Reject metric/backend combinations that cannot be left-weighted sparsely.
fn validate_geometric_options(
    problem: &Problem,
    opts: &InverseFdmOptions,
) -> Result<(), TheseusError> {
    if !opts.use_l2 {
        return Err(TheseusError::Solver(
            "geometric metric does not support L1/IRLS. The IRLS reweighting is a \
             diagonal row scaling of the force residual, which has no agreed meaning \
             once the rows are already weighted by the Laplacian compliance. Set L2 = true."
                .into(),
        ));
    }
    if problem.self_weight.is_some() || problem.pressure.is_some() {
        return Err(TheseusError::Solver(
            "geometric metric requires geometry-independent loads, but self-weight or \
             pressure is active. The identity x(q) − x* = −D(q)⁻¹r(q) assumes the load \
             vector does not move with the geometry."
                .into(),
        ));
    }
    Ok(())
}

/// Shift a box from the unknown `x` to the step `Δ`: `lo − x ≤ Δ ≤ hi − x`.
fn shift_box(bounds: &BoxBounds, x: &[f64]) -> BoxBounds {
    BoxBounds {
        lower: bounds
            .lower
            .iter()
            .zip(x)
            .map(|(&lo, &xi)| lo - xi)
            .collect(),
        upper: bounds
            .upper
            .iter()
            .zip(x)
            .map(|(&hi, &xi)| hi - xi)
            .collect(),
    }
}

/// Solve the inverse-FDM particular with Direct/Iterative linear algebra,
/// optional box bounds, and optional IRLS.
pub fn solve_inverse_fdm(
    problem: &Problem,
    target_free_xyz: &Array2<f64>,
    opts: InverseFdmOptions,
) -> Result<InverseFdmResult, TheseusError> {
    if opts.regularization < 0.0 {
        return Err(TheseusError::Solver(
            "regularization must be non-negative".into(),
        ));
    }
    if !opts.cwls_damping.is_finite() || opts.cwls_damping < 0.0 {
        return Err(TheseusError::Solver(
            "cwls_damping must be finite and non-negative".into(),
        ));
    }
    let ne = problem.topology.num_edges;
    let stage1_system = EquilibriumSystem::assemble(
        problem,
        target_free_xyz,
        if opts.solve_for_q {
            EquilibriumUnknown::ForceDensity
        } else {
            EquilibriumUnknown::Force
        },
        opts.enforce_zero_rx,
        opts.enforce_zero_ry,
        opts.enforce_zero_rz,
    )?;
    let q_bounds = compose_box(ne, &opts.signs, &opts.lower, &opts.upper)?;
    let stage1_bounds = if opts.solve_for_q {
        q_bounds.clone()
    } else {
        q_box_to_force_box(&q_bounds, &stage1_system.lengths)?
    };
    let stage1_kind = pick_inner(&opts, &stage1_bounds);

    let mut ldl_cache = None;
    let mut stage2_ldl_cache = None;
    let mut qr_symbolic: Option<SymbolicQr<u32>> = None;
    let mut factor_stack = GlobalPodBuffer::new(dyn_stack::StackReq::empty());
    let mut solve_stack = GlobalPodBuffer::new(dyn_stack::StackReq::empty());
    let mut start = feasible_start(&stage1_bounds);

    let mut solve_inner = |kind: InnerKind,
                           m_mat: &SparseColMatOwned,
                           p: &[f64],
                           lambda: f64,
                           x0: &[f64],
                           box_bounds: &BoxBounds,
                           weight: Option<&MetricWeight>,
                           geometric_step: bool|
     -> Result<(Vec<f64>, usize, bool), TheseusError> {
        match kind {
            InnerKind::Gram => Ok((
                solve_gram_on(
                    m_mat,
                    p,
                    lambda,
                    &mut ldl_cache,
                    &mut factor_stack,
                    &mut solve_stack,
                )?,
                1,
                true,
            )),
            InnerKind::Saddle => Ok((
                solve_saddle_on(
                    m_mat,
                    p,
                    lambda,
                    if geometric_step {
                        &mut stage2_ldl_cache
                    } else {
                        &mut ldl_cache
                    },
                    &mut factor_stack,
                    &mut solve_stack,
                    weight,
                )?,
                1,
                true,
            )),
            InnerKind::Qr => Ok((solve_qr_on(m_mat, p, true, &mut qr_symbolic)?, 1, true)),
            InnerKind::Lsqr => {
                let result =
                    solve_lsqr_on(m_mat, p, lambda, opts.tol.max(1e-11), opts.max_iter, weight)?;
                Ok((result.solution, result.iterations, result.converged))
            }
            InnerKind::Clarabel => Ok((
                solve_clarabel_box(
                    m_mat,
                    p,
                    lambda,
                    box_bounds,
                    weight,
                    opts.max_iter,
                    opts.tol,
                    geometric_step,
                )?,
                1,
                true,
            )),
            InnerKind::Spg => {
                let result = solve_spg_on(
                    m_mat,
                    p,
                    lambda,
                    box_bounds,
                    opts.max_iter,
                    opts.tol,
                    x0,
                    weight,
                )?;
                Ok((result.q, result.iterations, result.converged))
            }
        }
    };

    let (mut x, mut iterations, mut converged) = solve_inner(
        stage1_kind,
        &stage1_system.a,
        &stage1_system.p,
        opts.regularization,
        &start,
        &stage1_bounds,
        None,
        false,
    )?;
    clip_to_box(&mut x, &stage1_bounds);

    if !opts.use_l2 {
        let max_l1 = opts.max_l1_iter.max(1);
        let mut prev_l1 = f64::MAX;
        const ABS_EPS: f64 = 1e-12;
        for outer in 0..max_l1 {
            let mut r = stage1_system.a.matvec(&x);
            for (ri, &pi) in r.iter_mut().zip(stage1_system.p.iter()) {
                *ri -= pi;
            }
            let (sqrt_w, l1_obj, w_max) = irls_weights(&r);
            if prev_l1 < f64::MAX {
                let rel_change = (prev_l1 - l1_obj).abs() / (prev_l1 + ABS_EPS);
                if rel_change < 1e-8 {
                    converged = true;
                    iterations = outer + 1;
                    break;
                }
            }
            prev_l1 = l1_obj;
            let m_w = row_scaled_copy(&stage1_system.a, &sqrt_w);
            let p_w: Vec<f64> = sqrt_w
                .iter()
                .zip(&stage1_system.p)
                .map(|(w, p)| w * p)
                .collect();
            let effective_reg = if w_max > 0.0 {
                opts.regularization / w_max
            } else {
                opts.regularization
            };
            start.copy_from_slice(&x);
            let (next, _inner_iters, inner_ok) = solve_inner(
                stage1_kind,
                &m_w,
                &p_w,
                effective_reg,
                &start,
                &stage1_bounds,
                None,
                false,
            )?;
            x = next;
            clip_to_box(&mut x, &stage1_bounds);
            iterations = outer + 1;
            converged = inner_ok;
        }
    }

    validate_unknown(&x, ne, "inverse FDM")?;
    let stage1_q = if opts.solve_for_q {
        x
    } else {
        forces_to_q(&x, &stage1_system.lengths)
    };

    if opts.metric.is_geometric() {
        validate_geometric_options(problem, &opts)?;
        let stage2_system = EquilibriumSystem::assemble(
            problem,
            target_free_xyz,
            EquilibriumUnknown::ForceDensity,
            opts.enforce_zero_rx,
            opts.enforce_zero_ry,
            opts.enforce_zero_rz,
        )?;
        let stage2_kind = pick_stage2_inner(&opts, &q_bounds);
        let (frozen_budget, newton_budget) = match opts.metric {
            InverseMetric::Force => unreachable!("force metric is not geometric"),
            // Legacy native Geometry callers use max_outer as their frozen budget.
            InverseMetric::Geometry => (opts.max_outer, 0),
            InverseMetric::GeometryNewton => (opts.max_frozen_outer, opts.max_outer),
        };

        let mut total_iterations = 0usize;
        let mut phase_seed = if opts.q_ref.is_empty() {
            stage1_q.clone()
        } else {
            opts.q_ref.clone()
        };
        let mut phase_result = None;

        if frozen_budget > 0 {
            let mut frozen_opts = opts.clone();
            frozen_opts.metric = InverseMetric::Geometry;
            frozen_opts.max_frozen_outer = 0;
            frozen_opts.max_outer = frozen_budget;
            let result = solve_geometric_outer(
                problem,
                &stage2_system,
                &frozen_opts,
                &q_bounds,
                &phase_seed,
                stage2_kind,
                &mut solve_inner,
            )?;
            total_iterations += result.iterations;
            phase_seed = result.q.clone();
            phase_result = Some(result);
        }

        if newton_budget > 0 {
            let mut newton_opts = opts.clone();
            newton_opts.metric = InverseMetric::GeometryNewton;
            newton_opts.q_ref = phase_seed.clone();
            newton_opts.max_frozen_outer = 0;
            newton_opts.max_outer = newton_budget;
            let mut result = solve_geometric_outer(
                problem,
                &stage2_system,
                &newton_opts,
                &q_bounds,
                &phase_seed,
                stage2_kind,
                &mut solve_inner,
            )?;
            result.iterations += total_iterations;
            return Ok(result);
        }

        if let Some(result) = phase_result {
            return Ok(result);
        }

        // Both phase budgets are zero: report the Stage-1 seed without a
        // compliance-weighted update.
        let mut zero_opts = opts.clone();
        zero_opts.metric = InverseMetric::Geometry;
        zero_opts.max_frozen_outer = 0;
        zero_opts.max_outer = 0;
        return solve_geometric_outer(
            problem,
            &stage2_system,
            &zero_opts,
            &q_bounds,
            &phase_seed,
            stage2_kind,
            &mut solve_inner,
        );
    }

    // Report the geometric error even for a force-metric solve so the two are
    // comparable on the same scale. A singular Laplacian just means "unknown".
    let geometric_error = EquilibriumSystem::assemble(
        problem,
        target_free_xyz,
        EquilibriumUnknown::ForceDensity,
        opts.enforce_zero_rx,
        opts.enforce_zero_ry,
        opts.enforce_zero_rz,
    )
    .and_then(|q_system| probe_geometry(problem, &q_system, &stage1_q, true))
    .map(|probe| probe.error)
    .unwrap_or(f64::NAN);
    Ok(InverseFdmResult {
        q: stage1_q,
        iterations,
        converged,
        geometric_error,
    })
}

/// Stage-2 outer loop for the geometric metrics, always in q coordinates.
///
/// Each iteration rebuilds the compliance at the current q, measures the exact
/// geometric error, solves one weighted least-squares step, and backtracks on
/// the measured error. The step form makes the existing λ act as
/// Levenberg--Marquardt damping.
fn solve_geometric_outer<F>(
    problem: &Problem,
    system: &EquilibriumSystem,
    opts: &InverseFdmOptions,
    bounds: &BoxBounds,
    stage1_q: &[f64],
    kind: InnerKind,
    solve_inner: &mut F,
) -> Result<InverseFdmResult, TheseusError>
where
    F: FnMut(
        InnerKind,
        &SparseColMatOwned,
        &[f64],
        f64,
        &[f64],
        &BoxBounds,
        Option<&MetricWeight>,
        bool,
    ) -> Result<(Vec<f64>, usize, bool), TheseusError>,
{
    const MAX_BACKTRACK: usize = 12;
    let ne = problem.topology.num_edges;

    // Stage 1 is the normal initializer. q_ref remains only as an explicit
    // internal test override.
    let q_seed: Vec<f64> = if opts.q_ref.is_empty() {
        stage1_q.to_vec()
    } else {
        if opts.q_ref.len() != ne {
            return Err(TheseusError::Shape(format!(
                "q_ref has {} entries, expected {ne}",
                opts.q_ref.len()
            )));
        }
        opts.q_ref.clone()
    };
    let mut x = q_seed;
    clip_to_box(&mut x, bounds);

    let mut probe = probe_geometry(problem, system, &x, true)?;
    let mut best_x = x.clone();
    let mut best_error = probe.error;
    let mut iterations = 0;
    let mut converged = false;
    let max_outer = opts.max_outer;
    if best_error <= opts.tol.max(1e-12) {
        return Ok(InverseFdmResult {
            q: best_x,
            iterations,
            converged: true,
            geometric_error: best_error,
        });
    }

    for outer in 0..max_outer {
        iterations = outer + 1;

        // r_k is always measured against the target; only the Jacobian moves.
        let mut r = system.a.matvec(&x);
        for (ri, &pi) in r.iter_mut().zip(system.p.iter()) {
            *ri -= pi;
        }

        let jacobian = match opts.metric {
            InverseMetric::Force => unreachable!("force metric does not reach the outer loop"),
            InverseMetric::Geometry => None,
            InverseMetric::GeometryNewton => {
                // x(q_k) = x* + e_k = x* − S⁻¹r_k, no forward solve needed.
                let n_free = system.n_free;
                let current = Array2::from_shape_fn((n_free, 3), |(i, d)| {
                    system.free_positions[[i, d]] - probe.neg_error[d * n_free + i]
                });
                let at_current = EquilibriumSystem::assemble(
                    problem,
                    &current,
                    EquilibriumUnknown::ForceDensity,
                    opts.enforce_zero_rx,
                    opts.enforce_zero_ry,
                    opts.enforce_zero_rz,
                )?;
                Some(at_current.a)
            }
        };
        let jacobian = jacobian.as_ref().unwrap_or(&system.a);

        let neg_r: Vec<f64> = r.iter().map(|v| -v).collect();
        let step_bounds = shift_box(bounds, &x);
        let zero_start = vec![0.0; ne];
        // The inner convergence flag is advisory only: acceptance is decided by
        // the measured geometric error below.
        let (step, _inner_iters, _inner_ok) = solve_inner(
            kind,
            jacobian,
            &neg_r,
            opts.cwls_damping,
            &zero_start,
            &step_bounds,
            Some(&probe.weight),
            true,
        )?;
        validate_unknown(&step, ne, "geometric step")?;

        // Backtrack on the measured error; the linear model can overshoot when
        // the target is far from funicular.
        let mut accepted = false;
        let mut failed_probes = 0usize;
        let mut last_probe_error = None;
        let mut scale = 1.0;
        for _ in 0..MAX_BACKTRACK {
            let mut candidate: Vec<f64> = x
                .iter()
                .zip(&step)
                .map(|(&xi, &di)| xi + scale * di)
                .collect();
            clip_to_box(&mut candidate, bounds);
            match probe_geometry(problem, system, &candidate, true) {
                Ok(next) => {
                    if next.error < probe.error {
                        let improvement =
                            (probe.error - next.error) / probe.error.max(f64::MIN_POSITIVE);
                        let step_norm = candidate
                            .iter()
                            .zip(&x)
                            .map(|(&next_q, &q)| (next_q - q).powi(2))
                            .sum::<f64>()
                            .sqrt();
                        let q_norm = l2_norm_prefix(&x, x.len()).max(1.0);
                        let relative_step = step_norm / q_norm;
                        x = candidate;
                        probe = next;
                        accepted = true;
                        if probe.error < best_error {
                            best_error = probe.error;
                            best_x = x.clone();
                        }
                        let tolerance = opts.tol.max(1e-12);
                        converged = probe.error <= tolerance
                            || (improvement <= tolerance && relative_step <= tolerance);
                        break;
                    }
                }
                Err(error) => {
                    failed_probes += 1;
                    last_probe_error = Some(error);
                }
            }
            scale *= 0.5;
        }

        if !accepted {
            if failed_probes == MAX_BACKTRACK {
                return Err(TheseusError::Solver(format!(
                    "CWLS backtracking could not find a nonsingular trial Laplacian after \
                     {MAX_BACKTRACK} step reductions; last probe: {}",
                    last_probe_error.expect("a failed probe records its error")
                )));
            }
            // No downhill step along this direction; the linear model has
            // nothing left to offer and the best-so-far point stands.
            converged = false;
            break;
        }
        if converged {
            break;
        }
    }

    validate_unknown(&best_x, ne, "inverse FDM (geometric)")?;
    Ok(InverseFdmResult {
        q: best_x,
        iterations,
        converged,
        geometric_error: best_error,
    })
}

/// Box-constrained SPG particular.
pub fn solve_spg_box(
    problem: &Problem,
    target_free_xyz: &Array2<f64>,
    max_iter: usize,
    tol: f64,
    signs: &[i32],
    lower: &[f64],
    upper: &[f64],
    regularization: f64,
    enforce_zero_rx: bool,
    enforce_zero_ry: bool,
    enforce_zero_rz: bool,
    solve_for_q: bool,
) -> Result<SpgBoxResult, TheseusError> {
    let result = solve_inverse_fdm(
        problem,
        target_free_xyz,
        InverseFdmOptions {
            regularization,
            use_l2: true,
            max_l1_iter: 1,
            particular_method: ParticularMethod::Augmented,
            linear_algebra: LinearAlgebra::Iterative,
            enforce_zero_rx,
            enforce_zero_ry,
            enforce_zero_rz,
            solve_for_q,
            signs: signs.to_vec(),
            lower: lower.to_vec(),
            upper: upper.to_vec(),
            max_iter,
            tol,
            metric: InverseMetric::Force,
            q_ref: Vec::new(),
            max_frozen_outer: 0,
            max_outer: DEFAULT_MAX_OUTER,
            cwls_damping: 1e-6,
        },
    )?;
    Ok(SpgBoxResult {
        q: result.q,
        iterations: result.iterations,
        converged: result.converged,
    })
}

// (geometric helpers live above `solve_inverse_fdm`)

/// Create a copy of a CSC matrix with each row `i` scaled by `row_scales[i]`.
fn row_scaled_copy(mat: &SparseColMatOwned, row_scales: &[f64]) -> SparseColMatOwned {
    let mut scaled = mat.clone();
    for col in 0..scaled.ncols {
        let start = scaled.col_ptrs[col] as usize;
        let end_ = scaled.col_ptrs[col + 1] as usize;
        for nz in start..end_ {
            let row = scaled.row_indices[nz] as usize;
            scaled.values[nz] *= row_scales[row];
        }
    }
    scaled
}
