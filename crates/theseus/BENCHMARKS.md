# Null-space paper benchmarks

`tests/bench_nullspace.rs` is an ignored, dependency-free harness for the
Pellegrino--Calladine implementation. It compares the dense referee with the
sparse production path and keeps every method/fixture cell in a fresh process.
Consequently, an out-of-memory dense SVD is recorded as `process-failed` and
does not invalidate later projector cells.

Run the default release sweep from the repository root:

```powershell
cargo test --release -p theseus --test bench_nullspace `
  bench_nullspace_fresh_processes -- --ignored --nocapture
```

The default fixtures are the reduced 4-bar, a prestressed three-spoke triangle,
a planar grid under normal loads, and corner-anchored warped grids of sizes 4,
8, and 12. Override the matrix without changing the harness:

```powershell
$env:THESEUS_NULLSPACE_FIXTURES = 'four-bar,prestressed-triangle,planar-z-grid'
$env:THESEUS_NULLSPACE_GRIDS = '8,16,24,32,48,64,96,128'
$env:THESEUS_NULLSPACE_METHODS = 'dense,projector,qr,angles,saddle,gram'
$env:THESEUS_NULLSPACE_LAMBDAS = '0,1e-12,1e-10,1e-8,1e-6'
$env:THESEUS_NULLSPACE_MAX_MODES = '32'
cargo test --release -p theseus --test bench_nullspace `
  bench_nullspace_fresh_processes -- --ignored --nocapture
```

Increase the grid list until the dense child is killed or cannot allocate.
Do not run an intentionally memory-exhausting sweep alongside Grasshopper or
other unsaved work. The projector currently obtains exact nullities from
projector traces, requiring one pseudoinverse action per row and column; this
is a correctness-first implementation, so its runtime can become the limiting
factor before its sparse memory scaling does.

## Methods and validity

- `dense` densifies the force-form equilibrium matrix \(A\) and computes a
  full `faer-svd`. This is the naive paper baseline and small-fixture referee.
- `projector` applies \(I-A^+A\) and \(I-AA^+\) using the production LSQR
  Moore--Penrose action, followed by SVD only on small probe panels. It never
  densifies \(A\).
- `qr` uses faer sparse QR's built-in COLAMD ordering. It is available only
  when rows are at least columns. The basic least-squares solution is always
  residual-checked; rank and Calladine counts are deliberately never
  published from its diagonal. Rank-deficient non-finite solves are reported
  as `unsupported-rank-deficient`.
- `angles` runs dense SVD and projector together in a separate comparison
  process. It reports the largest principal angle from the singular values of
  \(N_\mathrm{svd}^T N_\mathrm{projector}\), and likewise for mechanisms.
  Its time and memory are not attributed to either primary method.
- `saddle` measures the augmented particular solve. At \(\lambda=0\), this is
  the Moore--Penrose particular with the LSQR fallback; positive values are
  Tikhonov solves.
- `gram` explicitly forms \(A^TA+\lambda I\), demonstrating normal-equation
  fill and condition squaring. A singular zero-\(\lambda\) cell is expected to
  report unsupported. The lambda sweep applies only to these particular
  solves, never to kernel rank.

All methods use the same assembled force-form \(A\) within a fixture.

## Output

Rows beginning with `RESULT` are tab-separated for direct import. The columns
are:

- `wall_ms`: complete child method time after fixture assembly;
- `peak_mib`: process peak working set (`GetProcessMemoryInfo` on Windows,
  `VmHWM` on Linux);
- `rank`, `s`, `m_raw`: SVD/projector counts only;
- `calladine`: integer residual
  \(s-m_\mathrm{raw}-(n_e-n_\mathrm{eq})\), which must be zero;
- `AN_F`, `ATPhi_F`: measured Frobenius kernel residuals
  \(\lVert AN\rVert_F\) and \(\lVert A^T\Phi\rVert_F\);
- `load_residual`: \(\lVert At^+-p\rVert_2/\max(\lVert p\rVert_2,\epsilon)\);
- `self_angle_deg`, `mechanism_angle_deg`: largest principal angles, present
  only in `angles` cells where both complete returned subspaces fit the mode
  cap;
- `status` and `note`: unsupported shapes/rank, allocation/process failures,
  rigid-body stripping, and returned-mode truncation.

Peak working set includes the Rust test process baseline and loaded libraries.
Compare cells from the same build and machine; do not subtract two peaks or
interpret small differences as allocator-exact live memory.

## Publication protocol

Use a release build, close unrelated high-memory applications, record OS, CPU,
RAM, Rust/Cargo versions, commit, thread settings, grid list, tolerance, mode
cap, and the complete `RESULT` stream. Repeat successful cells in independent
sweeps. Treat a dense process failure as a memory-wall observation only after
confirming the preceding grid succeeds; record the failing grid and available
physical memory. Never infer QR rank from this harness.

This document intentionally contains no paper baseline yet. A focused smoke
run validates compilation and process isolation, but it is not a controlled
end-to-end measurement suitable for publication.
