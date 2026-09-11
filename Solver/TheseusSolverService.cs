namespace Ariadne.Solver;

using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using Ariadne.FDM;
using Ariadne.Graphs;
using Ariadne.Utilities;
using Rhino.Geometry;
using Theseus.Interop;

/// <summary>
/// Encapsulates all Theseus solver operations. Provides a clean interface
/// for solving FDM networks without Grasshopper dependencies.
/// </summary>
public static class TheseusSolverService
{
    /// <summary>
    /// Solve an FDM network with optimization.
    /// </summary>
    /// <param name="network">The FDM network to solve.</param>
    /// <param name="inputs">Solver inputs including loads, bounds, and objectives.</param>
    /// <param name="options">Optional solver options (iterations, tolerances, etc.).</param>
    /// <param name="progressCallback">
    /// Optional callback invoked every ReportFrequency accepted L-BFGS iterations with
    /// (majorIteration, loss, xyz[numNodes*3], q[numEdges]).
    /// Return <c>true</c> to continue, <c>false</c> to cancel.
    /// </param>
    public static SolveResult Solve(
        FDM_Network network,
        SolverInputs inputs,
        SolverOptions? options = null,
        Func<int, double, double[], double[], bool>? progressCallback = null)
    {
        return Solve(network, inputs, options, progressCallback, CancellationToken.None);
    }

    /// <summary>
    /// Solve an FDM network with optimization and per-solve cancellation.
    /// </summary>
    public static SolveResult Solve(
        FDM_Network network,
        SolverInputs inputs,
        SolverOptions? options,
        Func<int, double, double[], double[], bool>? progressCallback,
        CancellationToken cancellationToken)
    {
        ValidateCommon(network, inputs);
        ValidateOptimizationBounds(inputs);
        options ??= new SolverOptions();

        var context = BuildContext(network);
        var data = BuildSolverData(network, inputs, context);

        using var solver = TheseusSolver.Create(
            data.NumEdges, data.NumNodes, data.NumFree,
            data.CooRows, data.CooCols, data.CooVals,
            data.FreeIndices, data.FixedIndices,
            data.Loads, data.FixedPositions,
            data.QInit, data.LowerBounds, data.UpperBounds,
            data.VariableNodeIndices, data.VariableSupportKinds, data.VariableSupportLambdas, data.SphereRadii,
            data.RollerEnabled, data.RollerLower, data.RollerUpper, data.RailStart, data.RailEnd,
            data.NurbsOffsets, data.NurbsLengths, data.NurbsData);

        cancellationToken.ThrowIfCancellationRequested();

        foreach (var objective in inputs.Objectives)
        {
            if (!objective.IsValid)
                throw new ArgumentException(
                    $"Objective {objective.GetType().Name} is invalid and cannot be used for optimization.",
                    nameof(inputs));
            objective.ApplyTo(solver, context);
        }

        ApplyLoadConfig(solver, inputs, context);

        solver.SetSolverOptions(
            maxIterations: options.MaxIterations,
            absTol: options.AbsTol,
            relTol: options.RelTol,
            barrierWeight: options.BarrierWeight,
            barrierSharpness: options.BarrierSharpness,
            anchorSaturationLambda: 1.0);
        solver.SetQParameterizationMode((int)ResolveQParameterizationMode(
            inputs.QParameterizationMode, data.LowerBounds, data.UpperBounds));

        if (cancellationToken.CanBeCanceled)
        {
            var callback = progressCallback;
            progressCallback = (iteration, loss, xyz, q) =>
                !cancellationToken.IsCancellationRequested &&
                (callback?.Invoke(iteration, loss, xyz, q) ?? true);
        }

        if (progressCallback != null)
            solver.SetProgressCallback(progressCallback, options.ReportFrequency, options.CopyProgressState);

        cancellationToken.ThrowIfCancellationRequested();
        SolverResult result;
        try
        {
            result = solver.Optimize(cancellationToken);
        }
        catch (TheseusException) when (cancellationToken.IsCancellationRequested)
        {
            throw new OperationCanceledException(cancellationToken);
        }
        cancellationToken.ThrowIfCancellationRequested();

        return BuildResult(network, result, context);
    }

    /// <summary>
    /// Solve an FDM network without optimization (forward solve only).
    /// Supplies unconstrained bounds so the Rust solver selects LDL
    /// factorization, which handles mixed-sign q values correctly.
    /// </summary>
    public static SolveResult SolveForward(FDM_Network network, SolverInputs inputs)
    {
        return SolveForward(network, inputs, CancellationToken.None);
    }

    /// <summary>
    /// Solve an FDM network without optimization with per-solve cancellation.
    /// Cancellation is cooperative at native forward-solve boundaries.
    /// </summary>
    public static SolveResult SolveForward(
        FDM_Network network,
        SolverInputs inputs,
        CancellationToken cancellationToken)
    {
        ValidateCommon(network, inputs);
        cancellationToken.ThrowIfCancellationRequested();

        var context = BuildContext(network);
        var data = BuildSolverData(network, inputs, context);

        using var solver = TheseusSolver.Create(
            data.NumEdges, data.NumNodes, data.NumFree,
            data.CooRows, data.CooCols, data.CooVals,
            data.FreeIndices, data.FixedIndices,
            data.Loads, data.FixedPositions,
            data.QInit, data.LowerBounds, data.UpperBounds,
            data.VariableNodeIndices, data.VariableSupportKinds, data.VariableSupportLambdas, data.SphereRadii,
            data.RollerEnabled, data.RollerLower, data.RollerUpper, data.RailStart, data.RailEnd,
            data.NurbsOffsets, data.NurbsLengths, data.NurbsData);
        solver.SetQParameterizationMode((int)ResolveQParameterizationMode(
            inputs.QParameterizationMode, data.LowerBounds, data.UpperBounds));

        ApplyLoadConfig(solver, inputs, context);

        cancellationToken.ThrowIfCancellationRequested();
        var result = solver.SolveForward(cancellationToken);

        return BuildResult(network, result, context);
    }

    /// Solve inverse FDM at a target geometry, then forward-solve with the recovered q.
    /// particularMethod: 0 = Gram, 1 = Augmented, 2 = Sparse QR, 3 = Clarabel
    /// (Direct unconstrained only; constrained Direct always uses Clarabel).
    /// linearAlgebra: 0 = Direct, 1 = Iterative.
    public static SolveResult SolveInverseFdm(
        FDM_Network network,
        SolverInputs inputs,
        double[] targetFreeXyz,
        double regularization = 1e-6,
        bool useL2 = true,
        int maxL1Iter = 20,
        int particularMethod = 3,
        int linearAlgebra = 0,
        bool enforceZeroRx = false,
        bool enforceZeroRy = false,
        bool enforceZeroRz = false,
        bool solveForQ = true,
        int[]? signs = null,
        double[]? lower = null,
        double[]? upper = null,
        int maxIter = 500,
        double tol = 1e-6)
    {
        ValidateCommon(network, inputs);
        var context = BuildContext(network);
        var data = BuildSolverData(network, inputs, context);

        using var solver = TheseusSolver.Create(
            data.NumEdges, data.NumNodes, data.NumFree,
            data.CooRows, data.CooCols, data.CooVals,
            data.FreeIndices, data.FixedIndices,
            data.Loads, data.FixedPositions,
            data.QInit, data.LowerBounds, data.UpperBounds,
            data.VariableNodeIndices, data.VariableSupportKinds, data.VariableSupportLambdas, data.SphereRadii,
            data.RollerEnabled, data.RollerLower, data.RollerUpper, data.RailStart, data.RailEnd,
            data.NurbsOffsets, data.NurbsLengths, data.NurbsData);
        solver.SetQParameterizationMode((int)ResolveQParameterizationMode(
            inputs.QParameterizationMode, data.LowerBounds, data.UpperBounds));

        ApplyLoadConfig(solver, inputs, context);

        var result = solver.SolveInverseFdm(targetFreeXyz, regularization, useL2, maxL1Iter, particularMethod,
            linearAlgebra, enforceZeroRx, enforceZeroRy, enforceZeroRz, solveForQ,
            signs, lower, upper, maxIter, tol);
        return BuildResult(network, result, context);
    }

    /// <summary>Analyze force and displacement null spaces at a target geometry.</summary>
    public static RigidityReport AnalyzeRigidity(
        FDM_Network network,
        SolverInputs inputs,
        double[] targetFreeXyz,
        RigidityMethod method = RigidityMethod.Projector,
        bool includeRigidBodies = false,
        int maxModes = 32)
    {
        ValidateCommon(network, inputs);
        if (targetFreeXyz.Length != network.FreeNodes.Count * 3)
            throw new ArgumentException(
                "Target coordinates must contain XYZ for every free node.",
                nameof(targetFreeXyz));

        var context = BuildContext(network);
        var data = BuildSolverData(network, inputs, context);
        using var solver = TheseusSolver.Create(
            data.NumEdges, data.NumNodes, data.NumFree,
            data.CooRows, data.CooCols, data.CooVals,
            data.FreeIndices, data.FixedIndices,
            data.Loads, data.FixedPositions,
            data.QInit, data.LowerBounds, data.UpperBounds);
        return solver.AnalyzeRigidity(targetFreeXyz, method, includeRigidBodies, maxModes);
    }

    /// <summary>Retract free-node coordinates onto per-member target lengths.</summary>
    public static LengthRetractionResult RetractMemberLengths(
        FDM_Network network,
        SolverInputs inputs,
        double[] initialFreeXyz,
        double[] targetLengths,
        int maxIterations = 30,
        double tolerance = 1e-10)
    {
        ValidateCommon(network, inputs);
        if (initialFreeXyz.Length != network.FreeNodes.Count * 3)
            throw new ArgumentException(
                "Coordinates must contain XYZ for every free node.",
                nameof(initialFreeXyz));
        if (targetLengths.Length != network.Graph.Ne)
            throw new ArgumentException(
                "Target length count must match edge count.",
                nameof(targetLengths));

        var context = BuildContext(network);
        var data = BuildSolverData(network, inputs, context);
        using var solver = TheseusSolver.Create(
            data.NumEdges, data.NumNodes, data.NumFree,
            data.CooRows, data.CooCols, data.CooVals,
            data.FreeIndices, data.FixedIndices,
            data.Loads, data.FixedPositions,
            data.QInit, data.LowerBounds, data.UpperBounds);
        return solver.RetractMemberLengths(
            initialFreeXyz, targetLengths, maxIterations, tolerance);
    }

    /// <summary>Classify and rotate mechanism modes using prestress stiffness.</summary>
    public static PrestressClassification ClassifyPrestress(
        FDM_Network network,
        SolverInputs inputs,
        double[] targetFreeXyz,
        double[] prestressForces,
        double[] mechanisms,
        int mechanismCount,
        double tolerance = 1e-10)
    {
        ValidateCommon(network, inputs);
        var context = BuildContext(network);
        var data = BuildSolverData(network, inputs, context);
        using var solver = TheseusSolver.Create(
            data.NumEdges, data.NumNodes, data.NumFree,
            data.CooRows, data.CooCols, data.CooVals,
            data.FreeIndices, data.FixedIndices,
            data.Loads, data.FixedPositions,
            data.QInit, data.LowerBounds, data.UpperBounds);
        return solver.ClassifyPrestress(
            targetFreeXyz, prestressForces, mechanisms, mechanismCount, tolerance);
    }

    /// <summary>
    /// Resolves a list of load-node positions to their indices in the network's
    /// free-node list (0-based). Matches by proximity using the network's ATol.
    /// </summary>
    /// <returns>List of free-node indices corresponding to each input point.</returns>
    /// <exception cref="ArgumentException">Thrown when a point doesn't match any free node.</exception>
    public static List<int> ResolveLoadNodeIndices(FDM_Network network, IReadOnlyList<Point3d> loadNodePoints)
    {
        var indices = new List<int>(loadNodePoints.Count);
        double tol = network.ATol;

        for (int p = 0; p < loadNodePoints.Count; p++)
        {
            var pt = loadNodePoints[p];
            int bestIdx = -1;
            double bestDist = double.MaxValue;

            for (int i = 0; i < network.Free.Count; i++)
            {
                double dist = pt.DistanceTo(network.Free[i].Value);
                if (dist < bestDist)
                {
                    bestDist = dist;
                    bestIdx = i;
                }
            }

            if (bestIdx < 0 || bestDist > tol)
                throw new ArgumentException(
                    $"Load node at ({pt.X:G4}, {pt.Y:G4}, {pt.Z:G4}) does not match any free node within tolerance {tol}.");

            indices.Add(bestIdx);
        }

        return indices;
    }

    internal static Vector3d[] PackFreeNodeLoads(
        int numFree,
        IReadOnlyList<Vector3d> loads,
        IReadOnlyList<int>? loadNodeIndices)
    {
        if (loads.Count == 0)
            throw new ArgumentException("Loads list cannot be empty.", nameof(loads));

        var packed = new Vector3d[numFree];
        if (loadNodeIndices is { Count: > 0 } targetIndices)
        {
            if (loads.Count != 1 && loads.Count != targetIndices.Count)
                throw new ArgumentException(
                    $"When load nodes are specified ({targetIndices.Count}), " +
                    $"provide either 1 load or exactly {targetIndices.Count} loads (got {loads.Count}).");

            for (int i = 0; i < targetIndices.Count; i++)
            {
                int freeIdx = targetIndices[i];
                if ((uint)freeIdx >= (uint)numFree)
                    throw new ArgumentOutOfRangeException(
                        nameof(loadNodeIndices),
                        $"Load node index {freeIdx} is outside the free-node range [0, {numFree}).");
                packed[freeIdx] = loads.Count == 1 ? loads[0] : loads[i];
            }
        }
        else
        {
            for (int i = 0; i < numFree; i++)
                packed[i] = i < loads.Count ? loads[i] : loads[^1];
        }

        return packed;
    }

    #region Validation

    private static void ValidateCommon(FDM_Network network, SolverInputs inputs)
    {
        if (network == null)
            throw new ArgumentNullException(nameof(network));
        if (!network.Valid)
            throw new ArgumentException("Network is not valid. Check anchor definitions.", nameof(network));
        if (inputs.Loads == null || inputs.Loads.Count == 0)
            throw new ArgumentException("Loads list cannot be empty.", nameof(inputs));
        if (inputs.QInit == null || inputs.QInit.Count == 0)
            throw new ArgumentException("Initial force densities cannot be empty.", nameof(inputs));
    }

    private static void ValidateOptimizationBounds(SolverInputs inputs)
    {
        if (inputs.LowerBounds == null || inputs.LowerBounds.Count == 0)
            throw new ArgumentException("Lower bounds cannot be empty for optimization.", nameof(inputs));
        if (inputs.UpperBounds == null || inputs.UpperBounds.Count == 0)
            throw new ArgumentException("Upper bounds cannot be empty for optimization.", nameof(inputs));
    }

    internal static QParameterizationMode ResolveQParameterizationMode(
        QParameterizationMode mode,
        IReadOnlyList<double> lowerBounds,
        IReadOnlyList<double> upperBounds)
    {
        const int legacyImplicitBounded = 1;
        if ((int)mode != legacyImplicitBounded)
            return mode;

        return HasFiniteTwoSidedBounds(lowerBounds, upperBounds)
            ? QParameterizationMode.DirectBoxBounds
            : QParameterizationMode.DirectSoftBounds;
    }

    private static bool HasFiniteTwoSidedBounds(
        IReadOnlyList<double> lowerBounds,
        IReadOnlyList<double> upperBounds)
    {
        int n = Math.Min(lowerBounds.Count, upperBounds.Count);
        if (n == 0)
            return false;

        for (int i = 0; i < n; i++)
        {
            if (!double.IsFinite(lowerBounds[i])
                || !double.IsFinite(upperBounds[i])
                || upperBounds[i] <= lowerBounds[i])
                return false;
        }
        return true;
    }

    #endregion

    #region Private Methods

    private static SolverContext BuildContext(FDM_Network network)
    {
        var nodeMap = new Dictionary<Node, int>(network.Graph.Nn);
        for (int i = 0; i < network.Graph.Nn; i++)
            nodeMap[network.Graph.Nodes[i]] = i;

        var edgeMap = new Dictionary<Edge, int>(network.Graph.Ne);
        for (int i = 0; i < network.Graph.Ne; i++)
            edgeMap[network.Graph.Edges[i]] = i;

        return new SolverContext
        {
            Network = network,
            NodeIndexMap = nodeMap,
            EdgeIndexMap = edgeMap
        };
    }

    private static SolverData BuildSolverData(FDM_Network network, SolverInputs inputs, SolverContext context)
    {
        var graph = network.Graph;
        int numEdges = graph.Ne;
        int numNodes = graph.Nn;
        int numFree = network.FreeNodes.Count;
        int numFixed = network.FixedNodes.Count;

        var cooRows = new int[numEdges * 2];
        var cooCols = new int[numEdges * 2];
        var cooVals = new double[numEdges * 2];

        for (int e = 0; e < numEdges; e++)
        {
            var edge = graph.Edges[e];
            int startIdx = context.NodeIndexMap[edge.Start];
            int endIdx = context.NodeIndexMap[edge.End];

            cooRows[e * 2] = e;
            cooCols[e * 2] = startIdx;
            cooVals[e * 2] = -1.0;
            cooRows[e * 2 + 1] = e;
            cooCols[e * 2 + 1] = endIdx;
            cooVals[e * 2 + 1] = 1.0;
        }

        var packedLoads = PackFreeNodeLoads(numFree, inputs.Loads, inputs.LoadNodeIndices);
        double[] loads = new double[numFree * 3];
        for (int i = 0; i < numFree; i++)
        {
            loads[i * 3 + 0] = packedLoads[i].X;
            loads[i * 3 + 1] = packedLoads[i].Y;
            loads[i * 3 + 2] = packedLoads[i].Z;
        }

        double[] fixedPos = new double[numFixed * 3];
        for (int i = 0; i < numFixed; i++)
        {
            var pos = network.Fixed[i].Value;
            fixedPos[i * 3 + 0] = pos.X;
            fixedPos[i * 3 + 1] = pos.Y;
            fixedPos[i * 3 + 2] = pos.Z;
        }

        double[] lowerBounds = inputs.LowerBounds != null
            ? Expand(inputs.LowerBounds, numEdges)
            : ExpandConstant(double.NegativeInfinity, numEdges);

        double[] upperBounds = inputs.UpperBounds != null
            ? Expand(inputs.UpperBounds, numEdges)
            : ExpandConstant(double.PositiveInfinity, numEdges);

        var supportData = BuildVariableSupportData(inputs.VariableSupports, network, context);

        return new SolverData(
            numEdges, numNodes, numFree,
            cooRows, cooCols, cooVals,
            [.. network.FreeNodes], [.. network.FixedNodes],
            loads, fixedPos,
            Expand(inputs.QInit, numEdges),
            lowerBounds,
            upperBounds,
            supportData.VariableNodeIndices,
            supportData.VariableSupportKinds,
            supportData.VariableSupportLambdas,
            supportData.SphereRadii,
            supportData.RollerEnabled,
            supportData.RollerLower,
            supportData.RollerUpper,
            supportData.RailStart,
            supportData.RailEnd,
            supportData.NurbsOffsets,
            supportData.NurbsLengths,
            supportData.NurbsData);
    }

    private readonly record struct VariableSupportData(
        int[] VariableNodeIndices,
        int[] VariableSupportKinds,
        double[] VariableSupportLambdas,
        double[] SphereRadii,
        byte[] RollerEnabled,
        double[] RollerLower,
        double[] RollerUpper,
        double[] RailStart,
        double[] RailEnd,
        int[] NurbsOffsets,
        int[] NurbsLengths,
        double[] NurbsData);

    private static VariableSupportData BuildVariableSupportData(
        List<VariableSupportConfig>? supports,
        FDM_Network network,
        SolverContext context)
    {
        if (supports == null || supports.Count == 0)
        {
            return new VariableSupportData(
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                []);
        }

        var nodeIndices = new List<int>();
        var kinds = new List<int>();
        var lambdas = new List<double>();
        var sphereRadii = new List<double>();
        var rollerEnabled = new List<byte>();
        var rollerLower = new List<double>();
        var rollerUpper = new List<double>();
        var railStart = new List<double>();
        var railEnd = new List<double>();
        var nurbsOffsets = new List<int>();
        var nurbsLengths = new List<int>();
        var nurbsData = new List<double>();

        var seen = new HashSet<int>();
        var fixedNodeSet = new HashSet<int>(network.FixedNodes);
        foreach (var support in supports)
        {
            if (support.Nodes == null || support.Nodes.Count == 0)
                continue;

            foreach (var node in support.Nodes)
            {
                if (!context.NodeIndexMap.TryGetValue(node, out int nodeIdx))
                    throw new ArgumentException("Variable support node is not part of the current network.");
                if (!fixedNodeSet.Contains(nodeIdx))
                    throw new ArgumentException("Variable supports can only target fixed nodes.");
                if (!seen.Add(nodeIdx))
                    throw new ArgumentException($"Node index {nodeIdx} has multiple variable support definitions.");
                if (!double.IsFinite(support.SaturationLambda) || support.SaturationLambda <= 0.0)
                    throw new ArgumentException("Variable support saturation lambda must be positive and finite.");

                nodeIndices.Add(nodeIdx);
                lambdas.Add(support.SaturationLambda);

                switch (support)
                {
                    case SphereVariableSupport sphere:
                        if (!double.IsFinite(sphere.Radius) || sphere.Radius <= 0.0)
                            throw new ArgumentException("Sphere variable support radius must be positive and finite.");
                        kinds.Add(0);
                        sphereRadii.Add(sphere.Radius);
                        rollerEnabled.AddRange([0, 0, 0]);
                        rollerLower.AddRange([0.0, 0.0, 0.0]);
                        rollerUpper.AddRange([0.0, 0.0, 0.0]);
                        railStart.AddRange([0.0, 0.0, 0.0]);
                        railEnd.AddRange([0.0, 0.0, 0.0]);
                        AddEmptyNurbs(nurbsOffsets, nurbsLengths);
                        break;

                    case RollerVariableSupport roller:
                        if (!(roller.FreeX || roller.FreeY || roller.FreeZ))
                            throw new ArgumentException("Roller variable support must free at least one axis.");
                        if ((roller.FreeX && !roller.DomainX.IsIncreasing) ||
                            (roller.FreeY && !roller.DomainY.IsIncreasing) ||
                            (roller.FreeZ && !roller.DomainZ.IsIncreasing))
                            throw new ArgumentException("Roller enabled-axis domains must be increasing.");

                        kinds.Add(1);
                        sphereRadii.Add(0.0);
                        rollerEnabled.Add((byte)(roller.FreeX ? 1 : 0));
                        rollerEnabled.Add((byte)(roller.FreeY ? 1 : 0));
                        rollerEnabled.Add((byte)(roller.FreeZ ? 1 : 0));
                        rollerLower.Add(roller.DomainX.T0);
                        rollerLower.Add(roller.DomainY.T0);
                        rollerLower.Add(roller.DomainZ.T0);
                        rollerUpper.Add(roller.DomainX.T1);
                        rollerUpper.Add(roller.DomainY.T1);
                        rollerUpper.Add(roller.DomainZ.T1);
                        railStart.AddRange([0.0, 0.0, 0.0]);
                        railEnd.AddRange([0.0, 0.0, 0.0]);
                        AddEmptyNurbs(nurbsOffsets, nurbsLengths);
                        break;

                    case RailVariableSupport rail:
                        if (!rail.Rail.IsValid || rail.Rail.Length <= 1e-12)
                            throw new ArgumentException("Rail variable support requires a non-degenerate line segment.");
                        kinds.Add(2);
                        sphereRadii.Add(0.0);
                        rollerEnabled.AddRange([0, 0, 0]);
                        rollerLower.AddRange([0.0, 0.0, 0.0]);
                        rollerUpper.AddRange([0.0, 0.0, 0.0]);
                        railStart.Add(rail.Rail.FromX);
                        railStart.Add(rail.Rail.FromY);
                        railStart.Add(rail.Rail.FromZ);
                        railEnd.Add(rail.Rail.ToX);
                        railEnd.Add(rail.Rail.ToY);
                        railEnd.Add(rail.Rail.ToZ);
                        AddEmptyNurbs(nurbsOffsets, nurbsLengths);
                        break;

                    case CurveVariableSupport curveSupport:
                    {
                        var curveData = NurbsSerialization.FromCurve(curveSupport.Curve);
                        var domain = curveSupport.Domain ?? new Interval(curveData.Domain[0], curveData.Domain[1]);
                        if (!domain.IsIncreasing)
                            throw new ArgumentException("NURBS curve support domain must be increasing.");
                        if (!curveSupport.Curve.ClosestPoint(node.Value, out double t))
                            throw new ArgumentException("Could not find closest curve parameter for variable support node.");

                        kinds.Add(3);
                        sphereRadii.Add(0.0);
                        rollerEnabled.AddRange([0, 0, 0]);
                        rollerLower.AddRange([0.0, 0.0, 0.0]);
                        rollerUpper.AddRange([0.0, 0.0, 0.0]);
                        railStart.AddRange([0.0, 0.0, 0.0]);
                        railEnd.AddRange([0.0, 0.0, 0.0]);
                        AddNurbsCurveData(nurbsOffsets, nurbsLengths, nurbsData, curveData, domain, t);
                        break;
                    }

                    case SurfaceVariableSupport surfaceSupport:
                    {
                        var surfaceData = NurbsSerialization.FromSurface(surfaceSupport.Surface);
                        var domainU = surfaceSupport.DomainU ?? new Interval(surfaceData.DomainU[0], surfaceData.DomainU[1]);
                        var domainV = surfaceSupport.DomainV ?? new Interval(surfaceData.DomainV[0], surfaceData.DomainV[1]);
                        if (!domainU.IsIncreasing || !domainV.IsIncreasing)
                            throw new ArgumentException("NURBS surface support domains must be increasing.");
                        if (!surfaceSupport.Surface.ClosestPoint(node.Value, out double u, out double v))
                            throw new ArgumentException("Could not find closest surface parameters for variable support node.");

                        kinds.Add(4);
                        sphereRadii.Add(0.0);
                        rollerEnabled.AddRange([0, 0, 0]);
                        rollerLower.AddRange([0.0, 0.0, 0.0]);
                        rollerUpper.AddRange([0.0, 0.0, 0.0]);
                        railStart.AddRange([0.0, 0.0, 0.0]);
                        railEnd.AddRange([0.0, 0.0, 0.0]);
                        AddNurbsSurfaceData(nurbsOffsets, nurbsLengths, nurbsData, surfaceData, domainU, domainV, u, v);
                        break;
                    }

                    default:
                        throw new ArgumentException($"Unsupported variable support type {support.GetType().Name}.");
                }
            }
        }

        return new VariableSupportData(
            [.. nodeIndices],
            [.. kinds],
            [.. lambdas],
            [.. sphereRadii],
            [.. rollerEnabled],
            [.. rollerLower],
            [.. rollerUpper],
            [.. railStart],
            [.. railEnd],
            [.. nurbsOffsets],
            [.. nurbsLengths],
            [.. nurbsData]);
    }

    private static void AddEmptyNurbs(List<int> offsets, List<int> lengths)
    {
        offsets.Add(0);
        lengths.Add(0);
    }

    private static void AddNurbsCurveData(
        List<int> offsets,
        List<int> lengths,
        List<double> data,
        NurbsSerialization.CurveData curve,
        Interval domain,
        double initialT)
    {
        int offset = data.Count;
        offsets.Add(offset);
        data.Add(curve.Degree);
        data.Add(curve.Knots.Length);
        data.Add(curve.ControlPoints.Length / 4);
        data.Add(domain.T0);
        data.Add(domain.T1);
        data.Add(Math.Clamp(initialT, domain.T0, domain.T1));
        data.AddRange(curve.Knots);
        data.AddRange(curve.ControlPoints);
        lengths.Add(data.Count - offset);
    }

    private static void AddNurbsSurfaceData(
        List<int> offsets,
        List<int> lengths,
        List<double> data,
        NurbsSerialization.SurfaceData surface,
        Interval domainU,
        Interval domainV,
        double initialU,
        double initialV)
    {
        int offset = data.Count;
        offsets.Add(offset);
        data.Add(surface.DegreeU);
        data.Add(surface.DegreeV);
        data.Add(surface.CountU);
        data.Add(surface.CountV);
        data.Add(surface.KnotsU.Length);
        data.Add(surface.KnotsV.Length);
        data.Add(domainU.T0);
        data.Add(domainU.T1);
        data.Add(domainV.T0);
        data.Add(domainV.T1);
        data.Add(Math.Clamp(initialU, domainU.T0, domainU.T1));
        data.Add(Math.Clamp(initialV, domainV.T0, domainV.T1));
        data.AddRange(surface.KnotsU);
        data.AddRange(surface.KnotsV);
        data.AddRange(surface.ControlPoints);
        lengths.Add(data.Count - offset);
    }

    private static void ApplyLoadConfig(TheseusSolver solver, SolverInputs inputs, SolverContext context)
    {
        if (inputs.SelfWeight is SelfWeightConfig.Prescribed prescribed)
        {
            int numEdges = context.EdgeIndexMap.Count;
            var mu = new double[numEdges];
            for (int i = 0; i < numEdges; i++)
                mu[i] = i < prescribed.LinearDensities.Count
                    ? prescribed.LinearDensities[i]
                    : prescribed.LinearDensities[^1];

            var g = new[] { prescribed.Gravity.X, prescribed.Gravity.Y, prescribed.Gravity.Z };
            solver.SetSelfWeightPrescribed(mu, g,
                prescribed.MaxIters, prescribed.Tolerance, prescribed.Relaxation);
        }
        else if (inputs.SelfWeight is SelfWeightConfig.Sizing sizing)
        {
            var g = new[] { sizing.Gravity.X, sizing.Gravity.Y, sizing.Gravity.Z };
            solver.SetSelfWeightSizing(sizing.Rho, sizing.Sigma, g,
                sizing.MaxIters, sizing.Tolerance, sizing.Relaxation);
        }

        if (inputs.Pressure is { } pressure)
        {
            int numFaces = pressure.Faces.Count;
            var faces = BuildFaceArray(pressure.Faces, context);

            switch (pressure)
            {
                case PressureConfig.Normal normal:
                {
                    var press = ExpandPerFace(normal.Pressures, numFaces);
                    solver.SetPressure(faces, press,
                        normal.MaxIters, normal.Tolerance, normal.Relaxation);
                    break;
                }
                case PressureConfig.Hydrostatic hydro:
                {
                    var up = new[] { hydro.UpDirection.X, hydro.UpDirection.Y, hydro.UpDirection.Z };
                    solver.SetPressureHydrostatic(faces,
                        hydro.RhoFluid, hydro.GMagnitude, hydro.ZDatum, up,
                        hydro.MaxIters, hydro.Tolerance, hydro.Relaxation);
                    break;
                }
                case PressureConfig.Directional dir:
                {
                    var press = ExpandPerFace(dir.Pressures, numFaces);
                    var d = new[] { dir.Direction.X, dir.Direction.Y, dir.Direction.Z };
                    solver.SetPressureDirectional(faces, press, d,
                        dir.MaxIters, dir.Tolerance, dir.Relaxation);
                    break;
                }
            }
        }
    }

    private static int[][] BuildFaceArray(List<List<int>> faces, SolverContext context)
    {
        int numFaces = faces.Count;
        int numNodes = context.Network.Graph.Nn;
        var result = new int[numFaces][];
        for (int f = 0; f < numFaces; f++)
        {
            var faceVerts = faces[f];
            var error = FaceGeometry.ValidateFaceIndices(faceVerts, f, numNodes);
            if (error != null)
                throw new ArgumentException(error);

            result[f] = new int[faceVerts.Count];
            for (int v = 0; v < faceVerts.Count; v++)
            {
                var node = context.Network.Graph.Nodes[faceVerts[v]];
                result[f][v] = context.NodeIndexMap[node];
            }
        }
        return result;
    }

    private static double[] ExpandPerFace(List<double> values, int numFaces)
    {
        var result = new double[numFaces];
        for (int f = 0; f < numFaces; f++)
            result[f] = f < values.Count ? values[f] : values[^1];
        return result;
    }

    private static SolveResult BuildResult(FDM_Network network, SolverResult result, SolverContext context)
    {
        var solvedNetwork = new FDM_Network(network);
        var solvedContext = BuildContext(solvedNetwork);
        ApplySolverResult(solvedNetwork, result, solvedContext);

        return new SolveResult
        {
            Network = solvedNetwork,
            ForceDensities = result.ForceDensities,
            MemberForces = result.MemberForces,
            MemberLengths = result.MemberLengths,
            Reactions = result.Reactions,
            LossTrace = result.LossTrace,
            Iterations = result.Iterations,
            Converged = result.Converged,
            TerminationReason = result.TerminationReason
        };
    }

    /// <summary>
    /// Updates an existing network in-place with solver geometry and force densities.
    /// Avoids allocating a new graph when topology is unchanged.
    /// </summary>
    internal static void ApplySolverResult(FDM_Network network, SolverResult result, SolverContext context)
    {
        var graph = network.Graph;
        int numNodes = graph.Nn;
        int numEdges = graph.Ne;

        for (int i = 0; i < numNodes; i++)
        {
            graph.Nodes[i].Value = new Point3d(
                result.Xyz[i * 3 + 0],
                result.Xyz[i * 3 + 1],
                result.Xyz[i * 3 + 2]);
        }

        for (int i = 0; i < numEdges; i++)
        {
            int startIdx = context.NodeIndexMap[graph.Edges[i].Start];
            int endIdx = context.NodeIndexMap[graph.Edges[i].End];
            graph.Edges[i].Q = result.ForceDensities[i];
            graph.Edges[i].Value = new LineCurve(
                graph.Nodes[startIdx].Value,
                graph.Nodes[endIdx].Value);
        }

        network.Free = network.FreeNodes.ConvertAll(i => graph.Nodes[i]);
        network.Fixed = network.FixedNodes.ConvertAll(i => graph.Nodes[i]);
        network.Anchors = network.FixedNodes.ConvertAll(i => graph.Nodes[i].Value);
        network.Valid = true;
    }

    private static FDM_Network BuildSolvedNetwork(FDM_Network oldNetwork, SolverResult result, SolverContext context)
    {
        var solvedNetwork = new FDM_Network(oldNetwork);
        ApplySolverResult(solvedNetwork, result, BuildContext(solvedNetwork));
        return solvedNetwork;
    }

    private static double[] Expand(IReadOnlyList<double> values, int length)
    {
        if (values.Count == 0)
            throw new ArgumentException("Values list cannot be empty", nameof(values));

        var result = new double[length];
        for (int i = 0; i < length; i++)
            result[i] = i < values.Count ? values[i] : values[^1];
        return result;
    }

    private static double[] ExpandConstant(double value, int length)
    {
        var result = new double[length];
        Array.Fill(result, value);
        return result;
    }

    #endregion
}
