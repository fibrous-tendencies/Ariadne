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
) -> Result<crate::nullspace::LsqrResult, TheseusError> {
    if lambda > 0.0 {
        let stacked = stack_tikhonov(m_mat, lambda);
        let mut rhs = vec![0.0; stacked.nrows];
        rhs[..p.len()].copy_from_slice(p);
        solve_lsqr(&stacked, &rhs, tol, max_iter)
    } else {
        solve_lsqr(m_mat, p, tol, max_iter)
    }
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
        cache
            .as_mut()
            .expect("LDL cache")
            .update(g, factor_stack)
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
    let mut g = SparseColMatOwned::sparse_times_sparse(&m_t, m_mat).map_err(TheseusError::Solver)?;
    if lambda > 0.0 {
        g.add_diagonal(lambda);
    }
    let h = m_t.matvec(p);
    ldl_solve_cached(&g, &h, lambda, cache, factor_stack, solve_stack)
}

fn solve_saddle_on(
    m_mat: &SparseColMatOwned,
    p: &[f64],
    lambda: f64,
    cache: &mut Option<Factorization>,
    factor_stack: &mut GlobalPodBuffer,
    solve_stack: &mut GlobalPodBuffer,
) -> Result<Vec<f64>, TheseusError> {
    if lambda == 0.0 {
        return solve_saddle_pseudoinverse(m_mat, p, lambda, 1e-11, 0);
    }
    let m = p.len();
    let n = m_mat.ncols;
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
            supernodal_flop_ratio_threshold:
                faer_sparse::SupernodalThreshold::FORCE_SIMPLICIAL,
            ..QrSymbolicParams::default()
        };
        *symbolic = Some(
            factorize_symbolic_qr(a_ref.symbolic(), params).map_err(
                |error| TheseusError::Linalg(format!("COLAMD sparse QR symbolic: {error:?}")),
            )?,
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
        let pivot_tol =
            100.0 * f64::EPSILON * pivot_scale.max(1.0) * a.nrows.max(n).max(1) as f64;
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
    let qr = unsafe {
        faer_sparse::qr::QrRef::new_unchecked(symbolic, &indices, values.as_slice())
    };
    let solve = |right_hand_sides: &mut Mat<f64>| -> Result<(), TheseusError> {
        let req = symbolic
            .solve_in_place_req::<f64>(right_hand_sides.ncols(), Parallelism::Rayon(0))
            .map_err(|error| TheseusError::Linalg(format!("sparse QR solve workspace: {error:?}")))?;
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

fn to_clarabel_csc(mat: &SparseColMatOwned) -> Result<clarabel::algebra::CscMatrix<f64>, TheseusError> {
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

fn solve_clarabel_once(
    m_mat: &SparseColMatOwned,
    p: &[f64],
    lambda: f64,
    bounds: &BoxBounds,
) -> Result<Vec<f64>, TheseusError> {
    use clarabel::algebra::CscMatrix;
    use clarabel::solver::{DefaultSettings, DefaultSolver, IPSolver, SolverStatus, SupportedConeT};

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
    for i in 0..m {
        a_triplets.push((i as u32, i as u32, 1.0));
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
    let settings = DefaultSettings::<f64> {
        verbose: false,
        ..Default::default()
    };
    let mut solver = DefaultSolver::new(&p_mat, &q_lin, &a_csc, &b, &cones, settings);
    solver.solve();
    let status = solver.solution.status;
    if !matches!(status, SolverStatus::Solved | SolverStatus::AlmostSolved) {
        return Err(TheseusError::Solver(format!(
            "Clarabel did not solve the bound-constrained particular ({status:?})"
        )));
    }
    Ok(solver.solution.x[m..].to_vec())
}

fn solve_clarabel_box(
    m_mat: &SparseColMatOwned,
    p: &[f64],
    lambda: f64,
    bounds: &BoxBounds,
) -> Result<Vec<f64>, TheseusError> {
    match solve_clarabel_once(m_mat, p, lambda, bounds) {
        Ok(x) => Ok(x),
        Err(_) if lambda == 0.0 => solve_clarabel_once(m_mat, p, 1e-12, bounds),
        Err(error) => Err(error),
    }
}

fn solve_spg_on(
    m_mat: &SparseColMatOwned,
    p: &[f64],
    lambda: f64,
    bounds: &BoxBounds,
    max_iter: usize,
    tol: f64,
    x0: &[f64],
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
    let ne = problem.topology.num_edges;
    let system = EquilibriumSystem::assemble(
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
    let bounds = compose_box(ne, &opts.signs, &opts.lower, &opts.upper)?;
    let kind = pick_inner(&opts, &bounds);

    let mut ldl_cache = None;
    let mut qr_symbolic: Option<SymbolicQr<u32>> = None;
    let mut factor_stack = GlobalPodBuffer::new(dyn_stack::StackReq::empty());
    let mut solve_stack = GlobalPodBuffer::new(dyn_stack::StackReq::empty());
    let mut start = feasible_start(&bounds);

    let mut solve_inner = |m_mat: &SparseColMatOwned,
                           p: &[f64],
                           lambda: f64,
                           x0: &[f64]|
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
                    &mut ldl_cache,
                    &mut factor_stack,
                    &mut solve_stack,
                )?,
                1,
                true,
            )),
            InnerKind::Qr => Ok((solve_qr_on(m_mat, p, true, &mut qr_symbolic)?, 1, true)),
            InnerKind::Lsqr => {
                let result =
                    solve_lsqr_on(m_mat, p, lambda, opts.tol.max(1e-11), opts.max_iter)?;
                Ok((result.solution, result.iterations, result.converged))
            }
            InnerKind::Clarabel => Ok((solve_clarabel_box(m_mat, p, lambda, &bounds)?, 1, true)),
            InnerKind::Spg => {
                let result = solve_spg_on(m_mat, p, lambda, &bounds, opts.max_iter, opts.tol, x0)?;
                Ok((result.q, result.iterations, result.converged))
            }
        }
    };

    let (mut x, mut iterations, mut converged) =
        solve_inner(&system.a, &system.p, opts.regularization, &start)?;
    clip_to_box(&mut x, &bounds);

    if !opts.use_l2 {
        let max_l1 = opts.max_l1_iter.max(1);
        let mut prev_l1 = f64::MAX;
        const ABS_EPS: f64 = 1e-12;
        for outer in 0..max_l1 {
            let mut r = system.a.matvec(&x);
            for (ri, &pi) in r.iter_mut().zip(system.p.iter()) {
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
            let m_w = row_scaled_copy(&system.a, &sqrt_w);
            let p_w: Vec<f64> = sqrt_w.iter().zip(&system.p).map(|(w, p)| w * p).collect();
            let effective_reg = if w_max > 0.0 {
                opts.regularization / w_max
            } else {
                opts.regularization
            };
            start.copy_from_slice(&x);
            let (next, _inner_iters, inner_ok) = solve_inner(&m_w, &p_w, effective_reg, &start)?;
            x = next;
            clip_to_box(&mut x, &bounds);
            iterations = outer + 1;
            converged = inner_ok;
        }
    }

    validate_unknown(&x, ne, "inverse FDM")?;
    let q = if opts.solve_for_q {
        x
    } else {
        forces_to_q(&x, &system.lengths)
    };
    Ok(InverseFdmResult {
        q,
        iterations,
        converged,
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
        },
    )?;
    Ok(SpgBoxResult {
        q: result.q,
        iterations: result.iterations,
        converged: result.converged,
    })
}

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
