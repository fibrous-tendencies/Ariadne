using System;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using Ariadne.Solver;

namespace Theseus.Interop;

/// <summary>
/// Result of an optimisation or forward solve.
/// All arrays use row-major layout: xyz[node * 3 + dim].
/// </summary>
public sealed class SolverResult
{
    public double[] Xyz { get; init; } = [];
    public double[] MemberLengths { get; init; } = [];
    public double[] MemberForces { get; init; } = [];
    public double[] ForceDensities { get; init; } = [];
    public double[] Reactions { get; init; } = [];
    public double[] LossTrace { get; init; } = [];
    public int Iterations { get; init; }
    public bool Converged { get; init; }
    public string TerminationReason { get; init; } = "";

    /// <summary>
    /// ‖x(q) − x*‖ for inverse solves: how far the forward solve lands from the
    /// target. NaN when not applicable or when the Laplacian was singular.
    /// </summary>
    public double GeometricError { get; init; } = double.NaN;
}

public enum RigidityMethod
{
    Projector = 0,
    DenseSvd = 1,
    SparseQr = 2,
}

/// <summary>Managed Pellegrino–Calladine report; basis matrices are row-major.</summary>
public sealed class RigidityReport
{
    public int Rank { get; init; }
    public int SelfStressCount { get; init; }
    public int RawMechanismCount { get; init; }
    public int MechanismCount { get; init; }
    public int RigidBodyCount { get; init; }
    public double[] ParticularForces { get; init; } = [];
    public double[] Residual { get; init; } = [];
    public double ResidualRatio { get; init; }
    public double[] SelfStressBasis { get; init; } = [];
    public double[] MechanismBasis { get; init; } = [];
    public double[] RigidBodyBasis { get; init; } = [];
    public int SelfStressModeCount =>
        SelfStressCount == 0 ? 0 : SelfStressBasis.Length / _numEdges;
    public int MechanismModeCount =>
        _equilibriumRows == 0 ? 0 : MechanismBasis.Length / _equilibriumRows;

    internal int _numEdges;
    internal int _equilibriumRows;
}

public enum MechanismClass
{
    PrestressUnstable = -1,
    FiniteCandidate = 0,
    PrestressStable = 1,
}

public sealed class LengthRetractionResult
{
    public double[] FreeXyz { get; init; } = [];
    public int Iterations { get; init; }
    public bool Converged { get; init; }
    public double MaxLengthError { get; init; }
    public double ResidualNorm { get; init; }
}

public sealed class PrestressClassification
{
    public double[] Eigenvalues { get; init; } = [];
    public MechanismClass[] Classes { get; init; } = [];
    public double[] RotatedMechanisms { get; init; } = [];
}

/// <summary>
/// Managed wrapper around the native Theseus solver (theseus.dll).
///
/// Implements <see cref="IDisposable"/> to ensure the native handle is freed.
/// A destructor is provided as a safety net for cases where Dispose is not called.
/// </summary>
public sealed class TheseusSolver : IDisposable
{
    private IntPtr _handle;
    private readonly int _numNodes;
    private readonly int _numEdges;
    private readonly int _numFree;
    private bool _disposed;
    private TheseusInterop.NativeProgressCallback? _pinnedCallback;

    private TheseusSolver(IntPtr handle, int numNodes, int numEdges, int numFree)
    {
        _handle = handle;
        _numNodes = numNodes;
        _numEdges = numEdges;
        _numFree = numFree;
    }

    ~TheseusSolver()
    {
        Dispose();
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(TheseusSolver));
    }

    public static string GetLastError()
    {
        var buf = new byte[2048];
        int n = TheseusInterop.theseus_last_error(buf, (nuint)buf.Length);
        if (n <= 0) return string.Empty;
        return Encoding.UTF8.GetString(buf, 0, n);
    }

    private static void Check(int rc)
    {
        if (rc != 0)
            throw new TheseusException(GetLastError(), rc);
    }

    // ── Construction ─────────────────────────────────────────

    public static TheseusSolver Create(
        int numEdges, int numNodes, int numFree,
        int[] cooRows, int[] cooCols, double[] cooVals,
        int[] freeNodeIndices, int[] fixedNodeIndices,
        double[] loads, double[] fixedPositions,
        double[] qInit, double[] lowerBounds, double[] upperBounds,
        int[]? variableNodeIndices = null,
        int[]? variableSupportKinds = null,
        double[]? variableSupportLambdas = null,
        double[]? sphereRadii = null,
        byte[]? rollerEnabled = null,
        double[]? rollerLower = null,
        double[]? rollerUpper = null,
        double[]? railStart = null,
        double[]? railEnd = null,
        int[]? nurbsOffsets = null,
        int[]? nurbsLengths = null,
        double[]? nurbsData = null)
    {
        int numFixed = fixedNodeIndices.Length;
        IntPtr handle;
        if (variableNodeIndices is { Length: > 0 })
        {
            int n = variableNodeIndices.Length;
            variableSupportKinds ??= new int[n];
            variableSupportLambdas ??= Enumerable.Repeat(1.0, n).ToArray();
            sphereRadii ??= new double[n];
            rollerEnabled ??= new byte[n * 3];
            rollerLower ??= new double[n * 3];
            rollerUpper ??= new double[n * 3];
            railStart ??= new double[n * 3];
            railEnd ??= new double[n * 3];
            nurbsOffsets ??= new int[n];
            nurbsLengths ??= new int[n];
            nurbsData ??= [];
            handle = TheseusInterop.theseus_create_with_variable_supports(
                (nuint)numEdges, (nuint)numNodes, (nuint)numFree,
                ToNuint(cooRows), ToNuint(cooCols), cooVals, (nuint)cooRows.Length,
                ToNuint(freeNodeIndices), ToNuint(fixedNodeIndices), (nuint)numFixed,
                loads, fixedPositions,
                qInit, lowerBounds, upperBounds,
                (nuint)n,
                ToNuint(variableNodeIndices),
                variableSupportKinds,
                variableSupportLambdas,
                sphereRadii,
                rollerEnabled,
                rollerLower,
                rollerUpper,
                railStart,
                railEnd,
                ToNuint(nurbsOffsets),
                ToNuint(nurbsLengths),
                nurbsData,
                (nuint)nurbsData.Length);
        }
        else
        {
            handle = TheseusInterop.theseus_create(
                (nuint)numEdges, (nuint)numNodes, (nuint)numFree,
                ToNuint(cooRows), ToNuint(cooCols), cooVals, (nuint)cooRows.Length,
                ToNuint(freeNodeIndices), ToNuint(fixedNodeIndices), (nuint)numFixed,
                loads, fixedPositions,
                qInit, lowerBounds, upperBounds);
        }

        if (handle == IntPtr.Zero)
            throw new TheseusException(GetLastError(), -1);

        return new TheseusSolver(handle, numNodes, numEdges, numFree);
    }

    // ── Objectives ───────────────────────────────────────────

    public void AddTargetXyz(double weight, int[] nodeIndices, double[] targetXyz, int reduction = 0)
    {
        ThrowIfDisposed();
        Check(TheseusInterop.theseus_add_target_xyz(
            _handle, weight, ToNuint(nodeIndices), (nuint)nodeIndices.Length, targetXyz, reduction));
    }

    public void AddTargetXy(double weight, int[] nodeIndices, double[] targetXy, int reduction = 0)
    {
        ThrowIfDisposed();
        Check(TheseusInterop.theseus_add_target_xy(
            _handle, weight, ToNuint(nodeIndices), (nuint)nodeIndices.Length, targetXy, reduction));
    }

    public void AddTargetPlane(double weight, int[] nodeIndices, double[] targetXyz, double[] origin, double[] xAxis, double[] yAxis, int reduction = 0)
    {
        ThrowIfDisposed();
        if (targetXyz.Length != nodeIndices.Length * 3)
            throw new ArgumentException("targetXyz length must be nodeIndices.Length * 3.", nameof(targetXyz));
        if (origin == null || origin.Length != 3)
            throw new ArgumentException("origin must have length 3.", nameof(origin));
        if (xAxis == null || xAxis.Length != 3)
            throw new ArgumentException("xAxis must have length 3.", nameof(xAxis));
        if (yAxis == null || yAxis.Length != 3)
            throw new ArgumentException("yAxis must have length 3.", nameof(yAxis));
        Check(TheseusInterop.theseus_add_target_plane(
            _handle, weight, ToNuint(nodeIndices), (nuint)nodeIndices.Length,
            targetXyz, origin, xAxis, yAxis, reduction));
    }

    public void AddPlanarConstraintAlongDirection(double weight, int[] nodeIndices, double[] origin, double[] xAxis, double[] yAxis, double[] direction)
    {
        ThrowIfDisposed();
        if (origin == null || origin.Length != 3)
            throw new ArgumentException("origin must have length 3.", nameof(origin));
        if (xAxis == null || xAxis.Length != 3)
            throw new ArgumentException("xAxis must have length 3.", nameof(xAxis));
        if (yAxis == null || yAxis.Length != 3)
            throw new ArgumentException("yAxis must have length 3.", nameof(yAxis));
        if (direction == null || direction.Length != 3)
            throw new ArgumentException("direction must have length 3.", nameof(direction));
        Check(TheseusInterop.theseus_add_planar_constraint_along_direction(
            _handle, weight, ToNuint(nodeIndices), (nuint)nodeIndices.Length,
            origin, xAxis, yAxis, direction));
    }

    public void AddTargetLength(double weight, int[] edgeIndices, double[] targets)
    {
        ThrowIfDisposed();
        Check(TheseusInterop.theseus_add_target_length(
            _handle, weight, ToNuint(edgeIndices), (nuint)edgeIndices.Length, targets));
    }

    public void AddTargetForce(double weight, int[] edgeIndices, double[] targets)
    {
        ThrowIfDisposed();
        Check(TheseusInterop.theseus_add_target_force(
            _handle, weight, ToNuint(edgeIndices), (nuint)edgeIndices.Length, targets));
    }

    public void AddLengthVariation(
        double weight,
        int[] edgeIndices,
        double sharpness,
        bool useNormalizedVariance,
        int normalizationStrategy)
    {
        ThrowIfDisposed();
        Check(TheseusInterop.theseus_add_length_variation(
            _handle,
            weight,
            ToNuint(edgeIndices),
            (nuint)edgeIndices.Length,
            sharpness,
            useNormalizedVariance ? (byte)1 : (byte)0,
            normalizationStrategy));
    }

    public void AddForceVariation(
        double weight,
        int[] edgeIndices,
        double sharpness,
        bool useNormalizedVariance,
        int normalizationStrategy)
    {
        ThrowIfDisposed();
        Check(TheseusInterop.theseus_add_force_variation(
            _handle,
            weight,
            ToNuint(edgeIndices),
            (nuint)edgeIndices.Length,
            sharpness,
            useNormalizedVariance ? (byte)1 : (byte)0,
            normalizationStrategy));
    }

    public void AddSumForceLength(double weight, int[] edgeIndices)
    {
        ThrowIfDisposed();
        Check(TheseusInterop.theseus_add_sum_force_length(
            _handle, weight, ToNuint(edgeIndices), (nuint)edgeIndices.Length));
    }

    public void AddMinLength(double weight, int[] edgeIndices, double[] thresholds, double sharpness)
    {
        ThrowIfDisposed();
        Check(TheseusInterop.theseus_add_min_length(
            _handle, weight, ToNuint(edgeIndices), (nuint)edgeIndices.Length, thresholds, sharpness));
    }

    public void AddMaxLength(double weight, int[] edgeIndices, double[] thresholds, double sharpness)
    {
        ThrowIfDisposed();
        Check(TheseusInterop.theseus_add_max_length(
            _handle, weight, ToNuint(edgeIndices), (nuint)edgeIndices.Length, thresholds, sharpness));
    }

    public void AddMinForce(double weight, int[] edgeIndices, double[] thresholds, double sharpness)
    {
        ThrowIfDisposed();
        Check(TheseusInterop.theseus_add_min_force(
            _handle, weight, ToNuint(edgeIndices), (nuint)edgeIndices.Length, thresholds, sharpness));
    }

    public void AddMaxForce(double weight, int[] edgeIndices, double[] thresholds, double sharpness)
    {
        ThrowIfDisposed();
        Check(TheseusInterop.theseus_add_max_force(
            _handle, weight, ToNuint(edgeIndices), (nuint)edgeIndices.Length, thresholds, sharpness));
    }

    public void AddRigidSetCompare(double weight, int[] nodeIndices, double[] targetXyz)
    {
        ThrowIfDisposed();
        Check(TheseusInterop.theseus_add_rigid_set_compare(
            _handle, weight, ToNuint(nodeIndices), (nuint)nodeIndices.Length, targetXyz));
    }

    public void AddReactionDirection(double weight, int[] anchorIndices, double[] targetDirs)
    {
        ThrowIfDisposed();
        Check(TheseusInterop.theseus_add_reaction_direction(
            _handle, weight, ToNuint(anchorIndices), (nuint)anchorIndices.Length, targetDirs));
    }

    public void AddReactionDirectionMagnitude(double weight, int[] anchorIndices, double[] targetDirs, double[] targetMags)
    {
        ThrowIfDisposed();
        Check(TheseusInterop.theseus_add_reaction_direction_magnitude(
            _handle, weight, ToNuint(anchorIndices), (nuint)anchorIndices.Length, targetDirs, targetMags));
    }

    public void AddReactionMagnitude(
        double weight,
        int[] anchorIndices,
        double[] targetDirs,
        double[] targetMags,
        int behavior,
        int signSemantics)
    {
        ThrowIfDisposed();
        Check(TheseusInterop.theseus_add_reaction_magnitude(
            _handle,
            weight,
            ToNuint(anchorIndices),
            (nuint)anchorIndices.Length,
            targetDirs,
            targetMags,
            behavior,
            signSemantics));
    }

    public void AddReactionDirectionMagnitude(
        double weight,
        int[] anchorIndices,
        double[] targetDirs,
        double[] targetMags,
        int behavior,
        int signSemantics)
    {
        ThrowIfDisposed();
        Check(TheseusInterop.theseus_add_reaction_direction_magnitude_with_options(
            _handle,
            weight,
            ToNuint(anchorIndices),
            (nuint)anchorIndices.Length,
            targetDirs,
            targetMags,
            behavior,
            signSemantics));
    }

    // ── Self-weight ───────────────────────────────────────────

    public void SetSelfWeightPrescribed(
        double[] linearDensities, double[] gravity,
        int maxIters = 50, double tolerance = 1e-6, double relaxation = 1.0)
    {
        ThrowIfDisposed();
        Check(TheseusInterop.theseus_set_self_weight_prescribed(
            _handle, linearDensities, gravity,
            (nuint)maxIters, tolerance, relaxation));
    }

    public void SetSelfWeightSizing(
        double rho, double sigma, double[] gravity,
        int maxIters = 50, double tolerance = 1e-6, double relaxation = 1.0)
    {
        ThrowIfDisposed();
        Check(TheseusInterop.theseus_set_self_weight_sizing(
            _handle, rho, sigma, gravity,
            (nuint)maxIters, tolerance, relaxation));
    }

    public void ClearSelfWeight()
    {
        ThrowIfDisposed();
        Check(TheseusInterop.theseus_clear_self_weight(_handle));
    }

    // ── Pressure loads ──────────────────────────────────────

    public void SetPressure(
        int[][] faces, double[] pressures,
        int maxIters = 50, double tolerance = 1e-6, double relaxation = 1.0)
    {
        ThrowIfDisposed();
        int numFaces = faces.Length;
        var offsets = new nuint[numFaces + 1];
        int totalVerts = 0;
        for (int f = 0; f < numFaces; f++)
        {
            offsets[f] = (nuint)totalVerts;
            totalVerts += faces[f].Length;
        }
        offsets[numFaces] = (nuint)totalVerts;

        var verts = new nuint[totalVerts];
        int idx = 0;
        for (int f = 0; f < numFaces; f++)
            foreach (int v in faces[f])
                verts[idx++] = (nuint)v;

        Check(TheseusInterop.theseus_set_pressure(
            _handle, (nuint)numFaces, offsets, verts, pressures,
            (nuint)maxIters, tolerance, relaxation));
    }

    public void SetPressureHydrostatic(
        int[][] faces, double rhoFluid, double gMagnitude, double zDatum,
        double[] upDirection,
        int maxIters = 50, double tolerance = 1e-6, double relaxation = 1.0)
    {
        ThrowIfDisposed();
        int numFaces = faces.Length;
        var offsets = new nuint[numFaces + 1];
        int totalVerts = 0;
        for (int f = 0; f < numFaces; f++)
        {
            offsets[f] = (nuint)totalVerts;
            totalVerts += faces[f].Length;
        }
        offsets[numFaces] = (nuint)totalVerts;

        var verts = new nuint[totalVerts];
        int idx = 0;
        for (int f = 0; f < numFaces; f++)
            foreach (int v in faces[f])
                verts[idx++] = (nuint)v;

        Check(TheseusInterop.theseus_set_pressure_hydrostatic(
            _handle, (nuint)numFaces, offsets, verts,
            rhoFluid, gMagnitude, zDatum, upDirection,
            (nuint)maxIters, tolerance, relaxation));
    }

    public void SetPressureDirectional(
        int[][] faces, double[] pressures, double[] direction,
        int maxIters = 50, double tolerance = 1e-6, double relaxation = 1.0)
    {
        ThrowIfDisposed();
        int numFaces = faces.Length;
        var offsets = new nuint[numFaces + 1];
        int totalVerts = 0;
        for (int f = 0; f < numFaces; f++)
        {
            offsets[f] = (nuint)totalVerts;
            totalVerts += faces[f].Length;
        }
        offsets[numFaces] = (nuint)totalVerts;

        var verts = new nuint[totalVerts];
        int idx = 0;
        for (int f = 0; f < numFaces; f++)
            foreach (int v in faces[f])
                verts[idx++] = (nuint)v;

        Check(TheseusInterop.theseus_set_pressure_directional(
            _handle, (nuint)numFaces, offsets, verts,
            pressures, direction,
            (nuint)maxIters, tolerance, relaxation));
    }

    public void ClearPressure()
    {
        ThrowIfDisposed();
        Check(TheseusInterop.theseus_clear_pressure(_handle));
    }

    // ── Solver options ───────────────────────────────────────

    public void SetSolverOptions(
        int maxIterations = 500,
        double absTol = 1e-6,
        double relTol = 1e-6,
        double barrierWeight = 10.0,
        double barrierSharpness = 10.0,
        double anchorSaturationLambda = 1.0)
    {
        ThrowIfDisposed();
        Check(TheseusInterop.theseus_set_solver_options(
            _handle, (nuint)maxIterations, absTol, relTol, barrierWeight, barrierSharpness,
            anchorSaturationLambda));
    }

    public void SetQParameterizationMode(int mode)
    {
        ThrowIfDisposed();
        Check(TheseusInterop.theseus_set_q_parameterization_mode(_handle, mode));
    }

    // ── Progress callback ─────────────────────────────────────

    /// <summary>
    /// Register a managed callback invoked every <paramref name="frequency"/>
    /// accepted L-BFGS iterations with (majorIteration, loss, xyz[numNodes*3], q[numEdges]).
    /// Return <c>true</c> to continue, <c>false</c> to cancel.
    /// Pass null to clear.  The delegate is pinned for the lifetime of this solver.
    /// When <paramref name="copySolverState"/> is false, xyz/q arrays are empty and
    /// no native-to-managed copy is performed (cancellation-only callbacks).
    /// </summary>
    public void SetProgressCallback(
        Func<int, double, double[], double[], bool>? callback,
        int frequency,
        bool copySolverState = true)
    {
        ThrowIfDisposed();
        if (callback == null)
        {
            _pinnedCallback = null;
            Check(TheseusInterop.theseus_set_progress_callback(_handle, null, (nuint)1));
            return;
        }

        int nn = _numNodes;
        int ne = _numEdges;
        if (copySolverState)
        {
            _pinnedCallback = (nuint majorIteration, double loss, IntPtr xyzPtr, nuint numNodes, IntPtr qPtr, nuint numEdges) =>
            {
                var xyz = new double[nn * 3];
                Marshal.Copy(xyzPtr, xyz, 0, nn * 3);
                var q = new double[ne];
                Marshal.Copy(qPtr, q, 0, ne);
                bool shouldContinue = callback((int)majorIteration, loss, xyz, q);
                return shouldContinue ? (byte)1 : (byte)0;
            };
        }
        else
        {
            _pinnedCallback = (nuint majorIteration, double loss, IntPtr _, nuint __, IntPtr ___, nuint ____) =>
            {
                bool shouldContinue = callback((int)majorIteration, loss, [], []);
                return shouldContinue ? (byte)1 : (byte)0;
            };
        }

        Check(TheseusInterop.theseus_set_progress_callback(
            _handle, _pinnedCallback, (nuint)Math.Max(1, frequency)));
    }

    /// <summary>
    /// Requests cancellation of an in-flight optimization when the native library
    /// exports <c>theseus_cancel</c>. Safe to call if cancellation is unsupported.
    /// </summary>
    public void RequestCancel()
    {
        ThrowIfDisposed();
        TheseusInterop.TryCancel(_handle);
    }

    private ulong BeginCancelScope()
    {
        ThrowIfDisposed();
        ulong scopeId = 0;
        Check(TheseusInterop.theseus_begin_cancel_scope(_handle, ref scopeId));
        return scopeId;
    }

    private void RequestScopedCancel(ulong scopeId)
    {
        if (!_disposed && _handle != IntPtr.Zero)
            TheseusInterop.theseus_cancel_scope(_handle, scopeId);
    }

    private void CompleteCancelScope(ulong scopeId)
    {
        if (!_disposed && _handle != IntPtr.Zero)
            TheseusInterop.theseus_complete_cancel_scope(_handle, scopeId);
    }

    // ── Solve ────────────────────────────────────────────────

    public SolverResult Optimize()
    {
        return Optimize(CancellationToken.None);
    }

    /// <summary>
    /// Runs optimization with a cancellation scope reserved before the native
    /// run token is published.
    /// </summary>
    public SolverResult Optimize(CancellationToken cancellationToken)
    {
        return OptimizeCore(cancellationToken, null);
    }

    internal SolverResult OptimizeWithScopeReadyHook(
        CancellationToken cancellationToken,
        Action scopeReady)
    {
        return OptimizeCore(cancellationToken, scopeReady);
    }

    private SolverResult OptimizeCore(
        CancellationToken cancellationToken,
        Action? scopeReady)
    {
        ThrowIfDisposed();
        ulong scopeId = BeginCancelScope();
        using var cancellationScope = PerSolveCancellation.Begin(
            cancellationToken,
            () => RequestScopedCancel(scopeId),
            () => CompleteCancelScope(scopeId));
        cancellationToken.ThrowIfCancellationRequested();
        scopeReady?.Invoke();
        var xyz = new double[_numNodes * 3];
        var lengths = new double[_numEdges];
        var forces = new double[_numEdges];
        var q = new double[_numEdges];
        var reactions = new double[_numNodes * 3];
        nuint iterations = 0;
        byte converged = 0;

        try
        {
            Check(TheseusInterop.theseus_optimize_scoped(
                _handle, scopeId, xyz, lengths, forces, q, reactions,
                ref iterations, ref converged));
        }
        catch (TheseusException) when (cancellationToken.IsCancellationRequested)
        {
            throw new OperationCanceledException(cancellationToken);
        }
        finally
        {
            cancellationScope.Complete();
        }
        cancellationToken.ThrowIfCancellationRequested();

        return new SolverResult
        {
            Xyz = xyz,
            MemberLengths = lengths,
            MemberForces = forces,
            ForceDensities = q,
            Reactions = reactions,
            LossTrace = GetLossTrace(),
            Iterations = (int)iterations,
            Converged = converged != 0,
            TerminationReason = GetTerminationReason(),
        };
    }

    private double[] GetLossTrace()
    {
        nuint len = TheseusInterop.theseus_get_loss_trace_len(_handle);
        if (len == 0)
            return [];

        var trace = new double[(int)len];
        nuint copied = TheseusInterop.theseus_get_loss_trace(_handle, trace, len);
        if (copied == len)
            return trace;

        Array.Resize(ref trace, (int)copied);
        return trace;
    }

    private string GetTerminationReason()
    {
        var buf = new byte[2048];
        int n = TheseusInterop.theseus_get_termination_reason(_handle, buf, (nuint)buf.Length);
        return n > 0 ? Encoding.UTF8.GetString(buf, 0, n) : string.Empty;
    }

    public SolverResult SolveForward()
    {
        return SolveForward(CancellationToken.None);
    }

    /// <summary>
    /// Runs a forward solve with cancellation bound to this solver handle.
    /// Native cancellation is cooperative between nonlinear/GMRES iterations
    /// and after each sparse factorization/solve.
    /// </summary>
    public SolverResult SolveForward(CancellationToken cancellationToken)
    {
        return SolveForwardCore(cancellationToken, null);
    }

    internal SolverResult SolveForwardWithScopeReadyHook(
        CancellationToken cancellationToken,
        Action scopeReady)
    {
        return SolveForwardCore(cancellationToken, scopeReady);
    }

    private SolverResult SolveForwardCore(
        CancellationToken cancellationToken,
        Action? scopeReady)
    {
        ThrowIfDisposed();
        ulong scopeId = BeginCancelScope();
        using var cancellationScope = PerSolveCancellation.Begin(
            cancellationToken,
            () => RequestScopedCancel(scopeId),
            () => CompleteCancelScope(scopeId));
        cancellationToken.ThrowIfCancellationRequested();
        scopeReady?.Invoke();
        var xyz = new double[_numNodes * 3];
        var lengths = new double[_numEdges];
        var forces = new double[_numEdges];
        var q = new double[_numEdges];
        var reactions = new double[_numNodes * 3];

        try
        {
            Check(TheseusInterop.theseus_solve_forward_scoped(
                _handle, scopeId, xyz, lengths, forces, q, reactions));
        }
        catch (TheseusException) when (cancellationToken.IsCancellationRequested)
        {
            throw new OperationCanceledException(cancellationToken);
        }
        finally
        {
            cancellationScope.Complete();
        }
        cancellationToken.ThrowIfCancellationRequested();

        return new SolverResult
        {
            Xyz = xyz,
            MemberLengths = lengths,
            MemberForces = forces,
            ForceDensities = q,
            Reactions = reactions,
            Iterations = 1,
            Converged = true,
        };
    }

    // ── Inverse solvers (experimental) ──────────────────────

    /// particularMethod: 0 = Gram, 1 = Augmented, 2 = Sparse QR, 3 = Clarabel
    /// (Direct unconstrained only; constrained Direct always uses Clarabel).
    /// linearAlgebra: 0 = Direct, 1 = Iterative.
    /// metric: 0 = Force (min ‖Mx − p‖), 1 = Geometry, 2 = GeometryNewton.
    /// solveForQ selects only the Stage-1 particular coordinate. Geometric
    /// Stage 2 always refines q using the Laplacian compliance. Gram and sparse
    /// QR can initialize Stage 2 but L1 remains unsupported there.
    public SolverResult SolveInverseFdm(
        double[] targetFreeXyz, double regularization,
        bool useL2 = true, int maxL1Iter = 20, int particularMethod = 3,
        int linearAlgebra = 0,
        bool enforceZeroRx = false, bool enforceZeroRy = false,
        bool enforceZeroRz = false, bool solveForQ = true,
        int[]? signs = null, double[]? lower = null, double[]? upper = null,
        int maxIter = 500, double tol = 1e-6,
        int metric = 0, double[]? qRef = null, int maxOuter = 0,
        double cwlsDamping = 1e-6)
    {
        ThrowIfDisposed();
        var q = new double[_numEdges];
        var xyz = new double[_numNodes * 3];
        var lengths = new double[_numEdges];
        var forces = new double[_numEdges];
        var reactions = new double[_numNodes * 3];
        nuint iterations = 0;
        byte converged = 0;
        double geometricError = double.NaN;
        int[] signsArr = signs ?? [];
        double[] lowerArr = lower ?? [];
        double[] upperArr = upper ?? [];

        Check(TheseusInterop.theseus_solve_inverse_fdm_metric_cwls(
            _handle, targetFreeXyz, regularization, cwlsDamping,
            useL2 ? 1 : 0, (nuint)maxL1Iter, particularMethod, linearAlgebra,
            enforceZeroRx ? 1 : 0, enforceZeroRy ? 1 : 0,
            enforceZeroRz ? 1 : 0, solveForQ ? 1 : 0,
            signsArr, (nuint)signsArr.Length,
            lowerArr, (nuint)lowerArr.Length,
            upperArr, (nuint)upperArr.Length,
            (nuint)maxIter, tol,
            metric, qRef, (nuint)(qRef?.Length ?? 0), (nuint)maxOuter,
            q, xyz, lengths, forces, reactions,
            ref iterations, ref converged, ref geometricError));

        return new SolverResult
        {
            Xyz = xyz,
            MemberLengths = lengths,
            MemberForces = forces,
            ForceDensities = q,
            Reactions = reactions,
            Iterations = (int)iterations,
            Converged = converged != 0,
            GeometricError = geometricError,
        };
    }

    public RigidityReport AnalyzeRigidity(
        double[] targetFreeXyz,
        RigidityMethod method = RigidityMethod.Projector,
        bool includeRigidBodies = false,
        int maxModes = 32)
    {
        ThrowIfDisposed();
        if (maxModes < 0)
            throw new ArgumentOutOfRangeException(nameof(maxModes));

        nuint rank = 0, selfStressCount = 0, rawMechanismCount = 0;
        nuint mechanismCount = 0, rigidCount = 0, particularLen = 0;
        nuint residualLen = 0, selfStressLen = 0, mechanismLen = 0, rigidLen = 0;
        Check(TheseusInterop.theseus_rigidity_report_sizes(
            _handle, targetFreeXyz, (int)method, includeRigidBodies ? 1 : 0, (nuint)maxModes,
            ref rank, ref selfStressCount, ref rawMechanismCount, ref mechanismCount,
            ref rigidCount, ref particularLen, ref residualLen, ref selfStressLen,
            ref mechanismLen, ref rigidLen));

        var particular = new double[(int)particularLen];
        var residual = new double[(int)residualLen];
        var selfStress = new double[(int)selfStressLen];
        var mechanisms = new double[(int)mechanismLen];
        var rigidBodies = new double[(int)rigidLen];
        double residualRatio = 0.0;
        Check(TheseusInterop.theseus_rigidity_report_fill(
            _handle, particular, particularLen, residual, residualLen, selfStress, selfStressLen,
            mechanisms, mechanismLen, rigidBodies, rigidLen, ref residualRatio));

        int equilibriumRows = (int)residualLen;
        return new RigidityReport
        {
            Rank = (int)rank,
            SelfStressCount = (int)selfStressCount,
            RawMechanismCount = (int)rawMechanismCount,
            MechanismCount = (int)mechanismCount,
            RigidBodyCount = (int)rigidCount,
            ParticularForces = particular,
            Residual = residual,
            ResidualRatio = residualRatio,
            SelfStressBasis = selfStress,
            MechanismBasis = mechanisms,
            RigidBodyBasis = rigidBodies,
            _numEdges = _numEdges,
            _equilibriumRows = equilibriumRows,
        };
    }

    public LengthRetractionResult RetractMemberLengths(
        double[] initialFreeXyz,
        double[] targetLengths,
        int maxIterations = 30,
        double tolerance = 1e-10)
    {
        ThrowIfDisposed();
        if (initialFreeXyz.Length != _numFree * 3)
            throw new ArgumentException("Coordinates must contain XYZ for every free node.", nameof(initialFreeXyz));
        if (targetLengths.Length != _numEdges)
            throw new ArgumentException("Target length count must match edge count.", nameof(targetLengths));
        var output = new double[initialFreeXyz.Length];
        nuint iterations = 0;
        byte converged = 0;
        double maxLengthError = 0.0, residualNorm = 0.0;
        Check(TheseusInterop.theseus_retract_member_lengths(
            _handle, initialFreeXyz, targetLengths, (nuint)Math.Max(0, maxIterations),
            tolerance, output, ref iterations, ref converged, ref maxLengthError,
            ref residualNorm));
        return new LengthRetractionResult
        {
            FreeXyz = output,
            Iterations = (int)iterations,
            Converged = converged != 0,
            MaxLengthError = maxLengthError,
            ResidualNorm = residualNorm,
        };
    }

    public PrestressClassification ClassifyPrestress(
        double[] targetFreeXyz,
        double[] prestressForces,
        double[] mechanisms,
        int mechanismCount,
        double tolerance = 1e-10)
    {
        ThrowIfDisposed();
        if (prestressForces.Length != _numEdges)
            throw new ArgumentException("Prestress force count must match edge count.", nameof(prestressForces));
        if (mechanismCount < 0 || (mechanismCount == 0
            ? mechanisms.Length != 0
            : mechanisms.Length % mechanismCount != 0))
            throw new ArgumentException("Mechanism basis shape is inconsistent.", nameof(mechanisms));
        var eigenvalues = new double[mechanismCount];
        var classes = new int[mechanismCount];
        var rotated = new double[mechanisms.Length];
        Check(TheseusInterop.theseus_classify_prestress(
            _handle, targetFreeXyz, prestressForces, mechanisms,
            (nuint)mechanismCount, tolerance, eigenvalues, classes, rotated));
        return new PrestressClassification
        {
            Eigenvalues = eigenvalues,
            Classes = Array.ConvertAll(classes, value => (MechanismClass)value),
            RotatedMechanisms = rotated,
        };
    }

    // ── IDisposable ──────────────────────────────────────────

    public void Dispose()
    {
        if (!_disposed && _handle != IntPtr.Zero)
        {
            TheseusInterop.theseus_set_progress_callback(_handle, null, (nuint)1);
            TheseusInterop.theseus_free(_handle);
            _handle = IntPtr.Zero;
            _pinnedCallback = null;
            _disposed = true;
        }
        GC.SuppressFinalize(this);
    }

    // ── Helpers ──────────────────────────────────────────────

    private static nuint[] ToNuint(int[] arr)
    {
        var result = new nuint[arr.Length];
        for (int i = 0; i < arr.Length; i++)
            result[i] = (nuint)arr[i];
        return result;
    }
}

/// <summary>
/// Exception thrown when the native Theseus library returns an error.
/// </summary>
public class TheseusException : Exception
{
    public int NativeCode { get; }

    public TheseusException(string message, int nativeCode)
        : base(message)
    {
        NativeCode = nativeCode;
    }
}
