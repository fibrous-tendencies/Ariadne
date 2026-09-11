using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Windows.Forms;
using Ariadne.FDM;
using GH_IO.Serialization;
using Grasshopper.Kernel;
using Grasshopper.Kernel.Data;
using Grasshopper.Kernel.Types;
using Rhino.Geometry;
using Theseus.Interop;

namespace Ariadne.Solver.Components.Experimental;

/// <summary>Finite mechanism paths with per-step projector recomputation.</summary>
public sealed class LengthPreservingTransformComponent : GH_Component
{
    private const string RestoreKey = "NetXformRestoreLengths";
    private const string ConeKey = "NetXformSignCone";
    private const int MaxModes = 32;
    private bool _restoreLengths = true;
    private bool _signCone = true;

    public LengthPreservingTransformComponent()
        : base("Length-Preserving Transform", "NetXform",
            "Trace finite, per-member length-preserving mechanism paths.",
            "Ariadne", "Experimental")
    {
        UpdateMessage();
    }

    protected override void RegisterInputParams(GH_InputParamManager pManager)
    {
        pManager.AddGenericParameter("Network", "Network", "FDM network topology and supports", GH_ParamAccess.item);
        pManager.AddPointParameter("Target Points", "Target", "Initial position of every free node", GH_ParamAccess.list);
        pManager.AddVectorParameter("Loads", "Loads", "Loads on free nodes", GH_ParamAccess.list, new Vector3d(0, 0, -1));
        pManager.AddNumberParameter("Coupled Mode Coefficients", "Beta", "Simultaneous finite-mode direction coefficients", GH_ParamAccess.list, 0.0);
        pManager.AddIntegerParameter("Steps", "Steps", "Number of finite continuation steps", GH_ParamAccess.item, 10);
        pManager.AddNumberParameter("Step Size", "Step", "Tangent step size before retraction", GH_ParamAccess.item, 0.05);
        pManager.AddPointParameter("Anchor Handles", "Handles", "Optional proposed positions, in fixed-node order", GH_ParamAccess.list);
        pManager[6].Optional = true;
    }

    protected override void RegisterOutputParams(GH_OutputParamManager pManager)
    {
        pManager.AddGenericParameter("Network", "Network", "Final transformed network", GH_ParamAccess.item);
        pManager.AddPointParameter("Path", "Path", "Free-node positions, one branch per continuation step", GH_ParamAccess.tree);
        pManager.AddPointParameter("Projected Handles", "Handles", "Nearest feasible handle configuration", GH_ParamAccess.list);
        pManager.AddNumberParameter("Member Lengths", "Lengths", "Final per-member lengths", GH_ParamAccess.list);
        pManager.AddNumberParameter("Member Forces", "t", "Final sign-feasible equilibrium forces", GH_ParamAccess.list);
        pManager.AddNumberParameter("Force Densities", "q", "Final force densities", GH_ParamAccess.list);
        pManager.AddNumberParameter("Maximum Length Error", "Lerr", "Maximum error from each member's own reference length", GH_ParamAccess.item);
        pManager.AddNumberParameter("Projection Distance", "Dist", "Weighted distance from requested to feasible configuration", GH_ParamAccess.item);
        pManager.AddNumberParameter("Projection Feasibility", "Feas", "Maximum length or sign-cone violation", GH_ParamAccess.item);
        pManager.AddIntegerParameter("Projection Iterations", "Iter", "Gauss-Newton active-set iterations", GH_ParamAccess.item);
        pManager.AddBooleanParameter("Projection Converged", "OK", "True when feasibility and nearest-point stationarity converged", GH_ParamAccess.item);
    }

    protected override void SolveInstance(IGH_DataAccess DA)
    {
        FDM_Network? network = null;
        List<Point3d> target = [];
        List<Vector3d> loads = [];
        List<double> beta = [];
        List<Point3d> handles = [];
        int steps = 10;
        double stepSize = 0.05;
        if (!DA.GetData(0, ref network) || network == null) return;
        if (!DA.GetDataList(1, target) || !DA.GetDataList(2, loads)) return;
        DA.GetDataList(3, beta);
        DA.GetData(4, ref steps);
        DA.GetData(5, ref stepSize);
        DA.GetDataList(6, handles);
        if (!network.Valid || target.Count != network.FreeNodes.Count)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Error,
                "Network must be valid and Target must contain every free node.");
            return;
        }
        if (handles.Count != 0 && handles.Count != network.FixedNodes.Count)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Error,
                "Handles must be empty or contain one point for every fixed node.");
            return;
        }
        steps = Math.Clamp(steps, 0, 1000);

        try
        {
            double[] referenceLengths = NullspaceComponentUtilities.MemberLengths(network, target);
            int[] signs = network.Graph.Edges.Select(edge =>
                edge.Q > 0.0 ? 1 : edge.Q < 0.0 ? -1 : 0).ToArray();
            var path = new GH_Structure<GH_Point>();
            FDM_Network current;
            double[] currentXyz;
            double[] currentForces;
            ConstrainedProjectionResult handleProjection;
            (current, currentXyz, currentForces, handleProjection) = ProjectHandles(
                network, target, loads, handles, referenceLengths, signs);

            AppendPath(path, 0, currentXyz);
            for (int step = 0; step < steps; step++)
            {
                List<Point3d> currentPoints = [.. NullspaceComponentUtilities.UnpackPoints(currentXyz)];
                int key = NullspaceComponentUtilities.AnalysisKey(current, currentPoints, loads, MaxModes);
                ManagedNullspaceAnalysis analysis = NullspaceComponentUtilities.Analyze(
                    current, currentPoints, loads, MaxModes, key);
                int allModes = analysis.Mechanisms.Length == 0
                    ? 0 : analysis.Mechanisms.Length / currentXyz.Length;
                int[] finite = Enumerable.Range(0, allModes)
                    .Where(index => index >= analysis.Classes.Length
                        || analysis.Classes[index] == MechanismClass.FiniteCandidate)
                    .ToArray();
                if (finite.Length == 0)
                {
                    AddRuntimeMessage(GH_RuntimeMessageLevel.Warning,
                        $"Continuation stopped at step {step}: no finite-classified mechanism.");
                    break;
                }

                double[] finiteBasis = SelectColumns(
                    analysis.Mechanisms, currentXyz.Length, allModes, finite);
                double[] trial = NullspaceComponentUtilities.Displace(
                    currentXyz, finiteBasis, finite.Length, beta, stepSize);
                double[] nextXyz = trial;
                if (_restoreLengths)
                {
                    LengthRetractionResult retraction = TheseusSolverService.RetractMemberLengths(
                        current, NullspaceComponentUtilities.Inputs(current, loads),
                        trial, referenceLengths);
                    if (!retraction.Converged)
                    {
                        AddRuntimeMessage(GH_RuntimeMessageLevel.Warning,
                            $"Continuation stopped at step {step}: length retraction error {retraction.MaxLengthError:G3}.");
                        break;
                    }
                    nextXyz = retraction.FreeXyz;
                }

                List<Point3d> nextPoints = [.. NullspaceComponentUtilities.UnpackPoints(nextXyz)];
                ManagedNullspaceAnalysis nextAnalysis = NullspaceComponentUtilities.Analyze(
                    current, nextPoints, loads, MaxModes,
                    NullspaceComponentUtilities.AnalysisKey(current, nextPoints, loads, MaxModes));
                currentForces = nextAnalysis.Report.ParticularForces;
                if (_signCone && !NullspaceComponentUtilities.ProjectForcesToSignCone(
                    currentForces, nextAnalysis.Report.SelfStressBasis,
                    nextAnalysis.Report.SelfStressModeCount, signs, out currentForces))
                {
                    AddRuntimeMessage(GH_RuntimeMessageLevel.Warning,
                        $"Continuation stopped at step {step}: the sign cone is infeasible.");
                    break;
                }
                current = NullspaceComponentUtilities.BuildNetwork(current, nextXyz, currentForces);
                currentXyz = nextXyz;
                AppendPath(path, step + 1, currentXyz);
            }

            Point3d[] projectedHandles = current.FixedNodes
                .ConvertAll(index => current.Graph.Nodes[index].Value).ToArray();
            double[] lengths = current.Graph.Edges
                .Select(edge => edge.Start.Value.DistanceTo(edge.End.Value)).ToArray();
            double maxError = lengths.Select((length, index) =>
                Math.Abs(length - referenceLengths[index])).DefaultIfEmpty(0.0).Max();
            DA.SetData(0, current);
            DA.SetDataTree(1, path);
            DA.SetDataList(2, projectedHandles);
            DA.SetDataList(3, lengths);
            DA.SetDataList(4, currentForces);
            DA.SetDataList(5, current.Graph.Edges.Select(edge => edge.Q));
            DA.SetData(6, maxError);
            DA.SetData(7, handleProjection.ObjectiveDistance);
            DA.SetData(8, handleProjection.Feasibility);
            DA.SetData(9, handleProjection.Iterations);
            DA.SetData(10, handleProjection.Converged);
        }
        catch (Exception error)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Error, error.Message);
        }
    }

    private (FDM_Network Network, double[] Xyz, double[] Forces,
        ConstrainedProjectionResult Projection) ProjectHandles(
        FDM_Network network,
        List<Point3d> target,
        List<Vector3d> loads,
        List<Point3d> handles,
        double[] referenceLengths,
        int[] signs)
    {
        double[] targetXyz = NullspaceComponentUtilities.PackPoints(target);
        if (handles.Count == 0)
        {
            ManagedNullspaceAnalysis analysis = NullspaceComponentUtilities.Analyze(
                network, target, loads, MaxModes,
                NullspaceComponentUtilities.AnalysisKey(network, target, loads, MaxModes));
            double[] forces = analysis.Report.ParticularForces;
            if (_signCone && !NullspaceComponentUtilities.ProjectForcesToSignCone(
                    forces, analysis.Report.SelfStressBasis,
                    analysis.Report.SelfStressModeCount, signs, out forces))
                throw new InvalidOperationException(
                    "The initial geometry has no equilibrium force in the requested sign cone.");
            var emptyProjection = new ConstrainedProjectionResult
            {
                Values = [],
                ObjectiveDistance = 0.0,
                MaxEqualityResidual = 0.0,
                MaxInequalityViolation = 0.0,
                Iterations = 0,
                Converged = true,
            };
            return (NullspaceComponentUtilities.BuildNetwork(network, targetXyz, forces),
                targetXyz, forces, emptyProjection);
        }

        // Freeze a self-stress basis once at the seed geometry. The inner
        // projection must not re-run full kernel analysis for every finite-
        // difference sample — that previously froze Rhino.
        double[] selfStress = [];
        int selfStressModes = 0;
        if (_signCone)
        {
            ManagedNullspaceAnalysis seed = NullspaceComponentUtilities.Analyze(
                network, target, loads, MaxModes,
                NullspaceComponentUtilities.AnalysisKey(network, target, loads, MaxModes));
            selfStress = seed.Report.SelfStressBasis;
            selfStressModes = seed.Report.SelfStressModeCount;
        }

        var initial = new double[handles.Count * 3];
        var desired = new double[initial.Length];
        var weights = new double[initial.Length];
        for (int i = 0; i < handles.Count; i++)
        {
            Point3d original = network.Graph.Nodes[network.FixedNodes[i]].Value;
            initial[3 * i] = original.X;
            initial[3 * i + 1] = original.Y;
            initial[3 * i + 2] = original.Z;
            desired[3 * i] = handles[i].X;
            desired[3 * i + 1] = handles[i].Y;
            desired[3 * i + 2] = handles[i].Z;
            weights[3 * i] = weights[3 * i + 1] = weights[3 * i + 2] = 1.0;
        }

        // Prefer the requested handles when already feasible.
        var requested = EvaluateHandleConfiguration(
            network, targetXyz, loads, referenceLengths, signs,
            desired, selfStress, selfStressModes);
        ConstrainedProjectionResult projection;
        HandleEvaluation final;
        if (requested.Feasible)
        {
            final = requested;
            projection = new ConstrainedProjectionResult
            {
                Values = desired,
                ObjectiveDistance = Math.Sqrt(initial
                    .Select((value, index) => Math.Pow(value - desired[index], 2)).Sum()),
                MaxEqualityResidual = MaxAbs(requested.LengthErrors),
                MaxInequalityViolation = requested.SignMargins
                    .Select(value => Math.Max(0.0, -value)).DefaultIfEmpty(0.0).Max(),
                Iterations = 0,
                Converged = true,
            };
        }
        else
        {
            projection = NullspaceComponentUtilities.ProjectNearestFeasible(
                initial, desired, weights, values =>
                {
                    HandleEvaluation evaluation = EvaluateHandleConfiguration(
                        network, targetXyz, loads, referenceLengths, signs,
                        values, selfStress, selfStressModes);
                    return (evaluation.LengthErrors, evaluation.SignMargins);
                },
                maxIterations: 25,
                tolerance: 1e-7);
            final = EvaluateHandleConfiguration(
                network, targetXyz, loads, referenceLengths, signs,
                projection.Values, selfStress, selfStressModes);
        }

        if (projection.Feasibility > 1e-5)
            throw new InvalidOperationException(
                $"Nearest-handle projection did not reach feasibility: {projection.Feasibility:G3}.");
        if (!projection.Converged)
            AddRuntimeMessage(GH_RuntimeMessageLevel.Warning,
                $"Handle projection is feasible but stationarity stopped after {projection.Iterations} iterations.");
        else if (projection.ObjectiveDistance > 1e-8)
            AddRuntimeMessage(GH_RuntimeMessageLevel.Remark,
                $"Handles projected to the nearest feasible configuration (distance {projection.ObjectiveDistance:G3}).");
        return (NullspaceComponentUtilities.BuildNetwork(
                final.Network, final.FreeXyz, final.Forces),
            final.FreeXyz, final.Forces, projection);
    }

    private sealed class HandleEvaluation
    {
        public required FDM_Network Network { get; init; }
        public required double[] FreeXyz { get; init; }
        public required double[] Forces { get; init; }
        public required double[] LengthErrors { get; init; }
        public required double[] SignMargins { get; init; }
        public bool Feasible =>
            MaxAbs(LengthErrors) <= 1e-5
            && SignMargins.All(value => value >= -1e-5);
    }

    /// <summary>
    /// Cheap handle evaluation: move anchors, retract free nodes onto the
    /// reference lengths, then recover a particular force (no kernel SVD).
    /// </summary>
    private HandleEvaluation EvaluateHandleConfiguration(
        FDM_Network source,
        double[] seedFreeXyz,
        List<Vector3d> loads,
        IReadOnlyList<double> referenceLengths,
        IReadOnlyList<int> signs,
        IReadOnlyList<double> handleValues,
        IReadOnlyList<double> selfStress,
        int selfStressModes)
    {
        Point3d[] anchors = Enumerable.Range(0, source.FixedNodes.Count)
            .Select(i => new Point3d(
                handleValues[3 * i],
                handleValues[3 * i + 1],
                handleValues[3 * i + 2]))
            .ToArray();
        FDM_Network candidate = NullspaceComponentUtilities.WithAnchors(source, anchors);
        SolverInputs inputs = NullspaceComponentUtilities.Inputs(candidate, loads);

        double[] freeXyz = seedFreeXyz;
        if (_restoreLengths)
        {
            LengthRetractionResult retraction = TheseusSolverService.RetractMemberLengths(
                candidate, inputs, seedFreeXyz, [.. referenceLengths],
                maxIterations: 20, tolerance: 1e-8);
            freeXyz = retraction.FreeXyz;
        }

        List<Point3d> points = [.. NullspaceComponentUtilities.UnpackPoints(freeXyz)];
        double[] actualLengths = NullspaceComponentUtilities.MemberLengths(candidate, points);
        double[] lengthErrors = _restoreLengths
            ? actualLengths.Select((length, index) => length - referenceLengths[index]).ToArray()
            : [];

        // Moore–Penrose particular in force coordinates — saddle only.
        SolveResult particular = TheseusSolverService.SolveInverseFdm(
            candidate, inputs, freeXyz,
            regularization: 0.0, useL2: true, maxL1Iter: 1, particularMethod: 1,
            solveForQ: false);
        double[] forces = new double[actualLengths.Length];
        for (int edge = 0; edge < forces.Length; edge++)
            forces[edge] = particular.ForceDensities[edge] * actualLengths[edge];

        if (_signCone)
        {
            NullspaceComponentUtilities.ProjectForcesToSignCone(
                forces, selfStress, selfStressModes, signs, out forces);
        }

        double[] signMargins = _signCone
            ? forces.Select((force, index) =>
                index < signs.Count && signs[index] != 0 ? signs[index] * force : 1.0).ToArray()
            : [];

        return new HandleEvaluation
        {
            Network = candidate,
            FreeXyz = freeXyz,
            Forces = forces,
            LengthErrors = lengthErrors,
            SignMargins = signMargins,
        };
    }

    private static double MaxAbs(IReadOnlyList<double> values) =>
        values.Select(Math.Abs).DefaultIfEmpty(0.0).Max();

    private static double[] SelectColumns(
        IReadOnlyList<double> basis, int rows, int sourceColumns, IReadOnlyList<int> selected)
    {
        var result = new double[rows * selected.Count];
        for (int row = 0; row < rows; row++)
            for (int column = 0; column < selected.Count; column++)
                result[row * selected.Count + column] =
                    basis[row * sourceColumns + selected[column]];
        return result;
    }

    private static void AppendPath(GH_Structure<GH_Point> tree, int step, IReadOnlyList<double> xyz)
    {
        var path = new GH_Path(step);
        foreach (Point3d point in NullspaceComponentUtilities.UnpackPoints(xyz))
            tree.Append(new GH_Point(point), path);
    }

    protected override void AppendAdditionalComponentMenuItems(ToolStripDropDown menu)
    {
        base.AppendAdditionalComponentMenuItems(menu);
        Menu_AppendSeparator(menu);
        Menu_AppendItem(menu, "Restore member lengths", (_, _) => ToggleRestore(), true, _restoreLengths);
        Menu_AppendItem(menu, "Preserve force sign cone", (_, _) => ToggleCone(), true, _signCone);
    }

    private void ToggleRestore()
    {
        RecordUndoEvent("Restore Member Lengths");
        _restoreLengths = !_restoreLengths;
        UpdateMessage();
        ExpireSolution(true);
    }

    private void ToggleCone()
    {
        RecordUndoEvent("Preserve Sign Cone");
        _signCone = !_signCone;
        UpdateMessage();
        ExpireSolution(true);
    }

    private void UpdateMessage() =>
        Message = $"{(_restoreLengths ? "Lengths" : "Tangent")} | {(_signCone ? "Cone" : "Any sign")}";

    public override bool Write(GH_IWriter writer)
    {
        writer.SetBoolean(RestoreKey, _restoreLengths);
        writer.SetBoolean(ConeKey, _signCone);
        return base.Write(writer);
    }

    public override bool Read(GH_IReader reader)
    {
        if (reader.ItemExists(RestoreKey)) _restoreLengths = reader.GetBoolean(RestoreKey);
        if (reader.ItemExists(ConeKey)) _signCone = reader.GetBoolean(ConeKey);
        UpdateMessage();
        return base.Read(reader);
    }

    protected override string HtmlHelp_Source() =>
"""
<html><body>
<h1>Length-Preserving Transform</h1>
<p>
Traces a finite path by stepping along coupled finite-classified modes,
retracting onto every member's own reference length, and recomputing the
sparse projector at each new geometry.
</p>
<br/>
<h2>Anchor handles</h2>
<p>
Handles are optional proposed fixed-node positions (one per support, same order).
Projection optimizes <b>handle coordinates only</b>: free nodes are restored by
length retraction, and member forces come from a Moore–Penrose particular.
A self-stress basis is computed once at the seed geometry for sign-cone mixing;
the inner loop does <b>not</b> re-run full null-space analysis (that previously froze Rhino).
</p>
</body></html>
""";

    protected override Bitmap Icon => Properties.Resources.parameters;
    public override Guid ComponentGuid => new("F0D73E8B-9A43-4D40-A9A3-45B77781B2E7");
    public override GH_Exposure Exposure => GH_Exposure.hidden;
}
