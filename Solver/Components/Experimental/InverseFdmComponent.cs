using System;
using System.Collections.Generic;
using System.Drawing;
using System.Globalization;
using System.Linq;
using System.Windows.Forms;
using GH_IO.Serialization;
using Grasshopper.Kernel;
using Grasshopper.Kernel.Data;
using Grasshopper.Kernel.Types;
using Rhino.Geometry;
using Ariadne.FDM;
using Ariadne.Solver;

namespace Ariadne.Solver.Components.Experimental;

/// <summary>
/// Experimental inverse force-density solve at a target geometry.
/// </summary>
public class InverseFdmComponent : GH_Component
{
    private const string ParticularKey = "InverseFdmParticular";
    private const string LinearAlgebraKey = "InverseFdmLinearAlgebra";
    private const string MetricKey = "InverseFdmMetric";
    private ParticularMode _particular = InverseFdmUiState.DefaultParticular;
    private LinearAlgebraMode _linearAlgebra = LinearAlgebraMode.Direct;
    private MetricMode _metric = InverseFdmUiState.DefaultMetric;
    private double _lambda = 1e-6;
    private double _cwlsDamping = 1e-6;
    private int _gnIterations = InverseFdmUiState.DefaultGnIterations;
    private bool _useL2 = true;
    private bool _solveForQ = InverseFdmUiState.DefaultSolveForQ;
    private bool _hasBox;

    public InverseFdmComponent()
        : base("Inverse FDM", "InvFDM",
            "Find one equilibrium particular at a target geometry, then forward-solve.",
            "Ariadne", "Experimental")
    {
        UpdateMessage();
    }

    protected override void RegisterInputParams(GH_InputParamManager pManager)
    {
        pManager.AddGenericParameter("Network", "Network", "FDM Network (topology + anchors)", GH_ParamAccess.item);
        pManager.AddPointParameter("Target Points", "Target", "Desired free-node positions (one per free node, matching order)", GH_ParamAccess.list);
        pManager.AddVectorParameter("Loads", "Loads", "Loads on free nodes", GH_ParamAccess.list, new Vector3d(0, 0, -1));
        pManager.AddPointParameter("Load Nodes", "LN", "Nodes to apply loads to (optional; if empty, loads apply to all free nodes)", GH_ParamAccess.list);
        pManager.AddNumberParameter("Regularization", "λ", "Stage-1 particular regularization used by Tikhonov, Gram, LSQR, Clarabel, and SPG. Ignored for Moore–Penrose and QR.", GH_ParamAccess.item, 1e-6);
        pManager.AddBooleanParameter("L2", "L2", "True = L2 least-squares residual, False = L1 absolute residual (IRLS around the inner solver)", GH_ParamAccess.item, true);
        pManager.AddIntegerParameter("Gauss–Newton Iterations", "GNiter", "Geometric metric only: 0 runs one frozen-target CWLS update; a positive value is the maximum number of CWLS-GN updates. Stops early at Tol.", GH_ParamAccess.item, InverseFdmUiState.DefaultGnIterations);
        pManager.AddBooleanParameter("Enforce Rx=0", "Rx0", "Strictly enforce zero X-reaction at supports", GH_ParamAccess.item, false);
        pManager.AddBooleanParameter("Enforce Ry=0", "Ry0", "Strictly enforce zero Y-reaction at supports", GH_ParamAccess.item, false);
        pManager.AddBooleanParameter("Enforce Rz=0", "Rz0", "Strictly enforce zero Z-reaction at supports", GH_ParamAccess.item, false);
        pManager.AddBooleanParameter("Solve Q", "SolveQ", "Stage 1 only: True solves the initial particular in force densities q. False (default) solves member forces t, then recovers q = t / target length. CWLS and L-BFGS-B operate in q.", GH_ParamAccess.item, InverseFdmUiState.DefaultSolveForQ);
        pManager.AddIntegerParameter("Signs", "Signs",
            "+1 tension (q ≥ 0), -1 compression (q ≤ 0), 0 free. Bounds always apply to q, even when Stage 1 solves member forces.",
            GH_ParamAccess.tree);
        pManager.AddNumberParameter("Lower", "Lower",
            "Lower bound on q. For a force-space Stage 1 this is internally multiplied by target edge length.",
            GH_ParamAccess.tree);
        pManager.AddNumberParameter("Upper", "Upper",
            "Upper bound on q. For a force-space Stage 1 this is internally multiplied by target edge length.",
            GH_ParamAccess.tree);
        pManager.AddIntegerParameter("Max Iterations", "MaxIter", "Iteration budget per inner solve for Clarabel, SPG, and LSQR", GH_ParamAccess.item, 500);
        pManager.AddNumberParameter("Tolerance", "Tol", "Convergence tolerance for Clarabel, SPG, and LSQR", GH_ParamAccess.item, 1e-6);
        pManager.AddNumberParameter("CWLS Damping", "λcwls", "Stage-2 Levenberg–Marquardt damping λcwls‖Δq‖². Separate from Stage-1 particular regularization.", GH_ParamAccess.item, 1e-6);
        pManager[3].Optional = true;
        pManager[11].Optional = true;
        pManager[12].Optional = true;
        pManager[13].Optional = true;
    }

    protected override void RegisterOutputParams(GH_OutputParamManager pManager)
    {
        pManager.AddGenericParameter("Network", "Network", "Solved network with updated geometry", GH_ParamAccess.item);
        pManager.AddPointParameter("Nodes", "Nodes", "Solved node positions", GH_ParamAccess.list);
        pManager.AddCurveParameter("Edges", "Edges", "Solved edge curves", GH_ParamAccess.list);
        pManager.AddNumberParameter("Force Densities", "Q", "Computed force densities", GH_ParamAccess.list);
        pManager.AddNumberParameter("Member Forces", "Forces", "Target-geometry member forces", GH_ParamAccess.list);
        pManager.AddVectorParameter("Residual", "Residual", "Free-node equilibrium residual at the target", GH_ParamAccess.list);
        pManager.AddNumberParameter("Residual Ratio", "RelRes", "Residual norm divided by load norm", GH_ParamAccess.item);
        pManager.AddNumberParameter("Geometric Error", "GeomErr",
            "‖x(q) − x*‖: distance from the forward-solved geometry to the target. Unlike RelRes this is a length, not a force ratio.",
            GH_ParamAccess.item);
    }

    protected override void SolveInstance(IGH_DataAccess DA)
    {
        DA.DisableGapLogic();

        FDM_Network? network = null;
        List<Point3d> targetPoints = [];
        List<Vector3d> loads = [];
        List<Point3d> loadNodes = [];
        double regularization = 1e-6;
        bool useL2 = true;
        const int maxL1Iter = 20;
        int gnIterations = InverseFdmUiState.DefaultGnIterations;
        bool enforceZeroRx = false;
        bool enforceZeroRy = false;
        bool enforceZeroRz = false;
        bool solveForQ = false;
        var signTree = new GH_Structure<GH_Integer>();
        var lowerTree = new GH_Structure<GH_Number>();
        var upperTree = new GH_Structure<GH_Number>();
        int maxIter = 500;
        double tol = 1e-6;
        double cwlsDamping = 1e-6;

        if (!DA.GetData(0, ref network)) return;
        if (!DA.GetDataList(1, targetPoints)) return;
        DA.GetDataList(2, loads);
        DA.GetDataList(3, loadNodes);
        DA.GetData(4, ref regularization);
        DA.GetData(5, ref useL2);
        DA.GetData(6, ref gnIterations);
        DA.GetData(7, ref enforceZeroRx);
        DA.GetData(8, ref enforceZeroRy);
        DA.GetData(9, ref enforceZeroRz);
        DA.GetData(10, ref solveForQ);
        DA.GetDataTree(11, out signTree);
        DA.GetDataTree(12, out lowerTree);
        DA.GetDataTree(13, out upperTree);
        DA.GetData(14, ref maxIter);
        DA.GetData(15, ref tol);
        DA.GetData(16, ref cwlsDamping);

        _lambda = regularization;
        _cwlsDamping = cwlsDamping;
        _gnIterations = Math.Max(0, gnIterations);
        _useL2 = useL2;
        _solveForQ = solveForQ;

        if (network == null || !network.Valid)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Error, "Invalid or null network.");
            return;
        }

        if (!TryMapOptionalTree(signTree, network.Graph.EdgeInputPaths, "Signs",
                out int[] signs, out string? signError, out string? signWarning))
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Error, signError!);
            return;
        }
        if (!TryMapOptionalTree(lowerTree, network.Graph.EdgeInputPaths, "Lower",
                out double[] lower, out string? lowerError, out string? lowerWarning))
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Error, lowerError!);
            return;
        }
        if (!TryMapOptionalTree(upperTree, network.Graph.EdgeInputPaths, "Upper",
                out double[] upper, out string? upperError, out string? upperWarning))
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Error, upperError!);
            return;
        }

        if (signWarning is not null)
            AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, signWarning);
        if (lowerWarning is not null)
            AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, lowerWarning);
        if (upperWarning is not null)
            AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, upperWarning);

        _hasBox = InverseFdmUiState.HasEffectiveBounds(signs, lower, upper);
        _particular = InverseFdmUiState.UpdateParticular(_linearAlgebra, _particular, _hasBox);
        UpdateMessage();

        int numFree = network.Free.Count;
        if (targetPoints.Count != numFree)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Error,
                $"Target points count ({targetPoints.Count}) must match free node count ({numFree}).");
            return;
        }

        bool unconstrainedDirect = _linearAlgebra == LinearAlgebraMode.Direct && !_hasBox;
        if (unconstrainedDirect && _particular == ParticularMode.Tikhonov && regularization <= 0.0)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Error,
                "Tikhonov mode requires λ > 0. Switch Particular to Moore–Penrose for λ = 0.");
            return;
        }

        if (_metric == MetricMode.Geometric)
        {
            if (!useL2)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Error,
                    "The geometric metric requires L2 = true. IRLS reweights the force residual, which "
                    + "has no agreed meaning once the rows carry the Laplacian compliance.");
                return;
            }
            if (!InverseFdmUiState.HasStrictSignDefiniteBounds(lower, upper))
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Warning,
                    "CWLS requires a numerically invertible FDM Laplacian. Mixed-sign and "
                    + "all-compression q are supported through sparse LDLᵀ, but cancellation, zero "
                    + "densities, or an unstable unpivoted factorization can block a trial.");
            }
        }

        double[] targetFreeXyz = new double[numFree * 3];
        for (int i = 0; i < numFree; i++)
        {
            targetFreeXyz[i * 3 + 0] = targetPoints[i].X;
            targetFreeXyz[i * 3 + 1] = targetPoints[i].Y;
            targetFreeXyz[i * 3 + 2] = targetPoints[i].Z;
        }

        var q = new List<double>(network.Graph.Ne);
        foreach (var edge in network.Graph.Edges)
            q.Add(double.IsFinite(edge.Q) ? edge.Q : 1.0);

        try
        {
            var loadNodeIndices = loadNodes.Count > 0
                ? TheseusSolverService.ResolveLoadNodeIndices(network, loadNodes)
                : null;
            var inputs = new SolverInputs
            {
                QInit = q,
                Loads = loads,
                LoadNodeIndices = loadNodeIndices,
            };

            double effectiveRegularization = unconstrainedDirect
                ? _particular switch
                {
                    ParticularMode.MoorePenrose => 0.0,
                    ParticularMode.Tikhonov => regularization,
                    ParticularMode.QrLeastSquares => 0.0,
                    _ => regularization,
                }
                : regularization;
            int particularMethod = InverseFdmUiState.NativeParticularMethod(_particular);

            int nativeMetric = InverseFdmUiState.NativeMetric(_metric, _gnIterations);
            int maxOuter = InverseFdmUiState.GeometricIterationBudget(_metric, _gnIterations);
            var result = TheseusSolverService.SolveInverseFdm(
                network, inputs, targetFreeXyz, effectiveRegularization,
                useL2, maxL1Iter, particularMethod, (int)_linearAlgebra,
                enforceZeroRx, enforceZeroRy, enforceZeroRz, solveForQ,
                [.. signs], [.. lower], [.. upper], maxIter, tol,
                nativeMetric, maxOuter, cwlsDamping);

            if (_metric == MetricMode.Force
                && _hasBox
                && _linearAlgebra == LinearAlgebraMode.Iterative
                && !result.Converged)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Warning,
                    $"SPG did not converge within MaxIter={maxIter}.");
            }

            var packedLoads = TheseusSolverService.PackFreeNodeLoads(
                network.FreeNodes.Count, loads, loadNodeIndices);
            var (forces, residuals, ratio) = TargetResidual(
                network, targetPoints, packedLoads, result.ForceDensities);

            DA.SetData(0, result.Network);
            DA.SetDataList(1, result.NodePositions);
            DA.SetDataList(2, result.EdgeCurves);
            DA.SetDataList(3, result.ForceDensities);
            DA.SetDataList(4, forces);
            DA.SetDataList(5, residuals);
            DA.SetData(6, ratio);
            DA.SetData(7, result.GeometricError);

            if (ratio > 0.25 && _metric == MetricMode.Force)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Warning,
                    $"Large residual ratio ({ratio:0.###}). The target/load combination may be inconsistent with equilibrium; Forces are from the particular at the target, Network is the forward solve.");
            }
        }
        catch (Exception ex)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Error, ex.Message);
        }
    }

    private static bool TryMapOptionalTree(
        GH_Structure<GH_Number> tree,
        IReadOnlyList<GH_Path> edgePaths,
        string label,
        out double[] values,
        out string? error,
        out string? warning)
    {
        values = [];
        error = null;
        warning = null;
        if (tree.DataCount == 0)
            return true;
        var mapping = QTreeMapper.Map(tree, edgePaths, label);
        if (!mapping.Success)
        {
            error = mapping.Error;
            return false;
        }
        values = mapping.Values!.ToArray();
        warning = mapping.Warning;
        return true;
    }

    private static bool TryMapOptionalTree(
        GH_Structure<GH_Integer> tree,
        IReadOnlyList<GH_Path> edgePaths,
        string label,
        out int[] values,
        out string? error,
        out string? warning)
    {
        values = [];
        error = null;
        warning = null;
        if (tree.DataCount == 0)
            return true;
        var mapping = QTreeMapper.Map(tree, edgePaths, label);
        if (!mapping.Success)
        {
            error = mapping.Error;
            return false;
        }
        values = mapping.Values!.Select(value => (int)Math.Round(value)).ToArray();
        warning = mapping.Warning;
        return true;
    }

    private static (double[] Forces, Vector3d[] Residuals, double Ratio) TargetResidual(
        FDM_Network network, IReadOnlyList<Point3d> target, IReadOnlyList<Vector3d> packedLoads,
        IReadOnlyList<double> q)
    {
        var positions = new Point3d[network.Graph.Nn];
        for (int i = 0; i < network.FreeNodes.Count; i++)
            positions[network.FreeNodes[i]] = target[i];
        for (int i = 0; i < network.FixedNodes.Count; i++)
            positions[network.FixedNodes[i]] = network.Fixed[i].Value;
        var residuals = new Vector3d[network.FreeNodes.Count];
        for (int i = 0; i < residuals.Length; i++)
            residuals[i] = -packedLoads[i];
        var freeLookup = new Dictionary<int, int>();
        for (int i = 0; i < network.FreeNodes.Count; i++) freeLookup[network.FreeNodes[i]] = i;
        var forces = new double[network.Graph.Ne];
        for (int e = 0; e < network.Graph.Ne; e++)
        {
            var edge = network.Graph.Edges[e];
            Vector3d delta = positions[edge.End.Index] - positions[edge.Start.Index];
            forces[e] = q[e] * delta.Length;
            if (freeLookup.TryGetValue(edge.Start.Index, out int start))
                residuals[start] -= q[e] * delta;
            if (freeLookup.TryGetValue(edge.End.Index, out int end))
                residuals[end] += q[e] * delta;
        }
        double residualNorm = 0.0, loadNorm = 0.0;
        for (int i = 0; i < residuals.Length; i++)
        {
            residualNorm += residuals[i].SquareLength;
            loadNorm += packedLoads[i].SquareLength;
        }
        return (forces, residuals, Math.Sqrt(residualNorm) / Math.Max(Math.Sqrt(loadNorm), double.Epsilon));
    }

    protected override void AppendAdditionalComponentMenuItems(ToolStripDropDown menu)
    {
        base.AppendAdditionalComponentMenuItems(menu);
        Menu_AppendSeparator(menu);
        var metricMenu = new ToolStripMenuItem("Metric");
        AppendMetricItem(metricMenu, "Force residual", MetricMode.Force);
        AppendMetricItem(metricMenu, "Geometric residual", MetricMode.Geometric);
        menu.Items.Add(metricMenu);
        Menu_AppendSeparator(menu);
        Menu_AppendItem(menu, "Linear algebra: Direct", (_, _) => SetLinearAlgebra(LinearAlgebraMode.Direct), true, _linearAlgebra == LinearAlgebraMode.Direct);
        Menu_AppendItem(menu, "Linear algebra: Iterative", (_, _) => SetLinearAlgebra(LinearAlgebraMode.Iterative), true, _linearAlgebra == LinearAlgebraMode.Iterative);
        Menu_AppendSeparator(menu);
        var directSolverMenu = new ToolStripMenuItem("Direct solver")
        {
            Enabled = _linearAlgebra == LinearAlgebraMode.Direct,
        };
        AppendDirectSolverItem(directSolverMenu, "Clarabel", ParticularMode.Clarabel);
        AppendDirectSolverItem(directSolverMenu, "Moore–Penrose", ParticularMode.MoorePenrose);
        AppendDirectSolverItem(directSolverMenu, "Tikhonov", ParticularMode.Tikhonov);
        AppendDirectSolverItem(directSolverMenu, "QR least squares", ParticularMode.QrLeastSquares);
        AppendDirectSolverItem(directSolverMenu, "Gram (normal equations)", ParticularMode.Gram);
        menu.Items.Add(directSolverMenu);
    }

    private void AppendDirectSolverItem(
        ToolStripMenuItem parent,
        string label,
        ParticularMode mode)
    {
        var item = new ToolStripMenuItem(label)
        {
            Checked = _particular == mode,
        };
        item.Click += (_, _) => SetParticular(mode);
        parent.DropDownItems.Add(item);
    }

    private void AppendMetricItem(ToolStripMenuItem parent, string label, MetricMode mode)
    {
        var item = new ToolStripMenuItem(label)
        {
            Checked = _metric == mode,
        };
        item.Click += (_, _) => SetMetric(mode);
        parent.DropDownItems.Add(item);
    }

    private void SetMetric(MetricMode mode)
    {
        if (_metric == mode) return;
        RecordUndoEvent("Set Inverse FDM Metric");
        _metric = mode;
        UpdateMessage();
        ExpireSolution(true);
    }

    private void SetParticular(ParticularMode mode)
    {
        if (_particular == mode) return;
        RecordUndoEvent("Set Inverse FDM Particular");
        _particular = mode;
        UpdateMessage();
        ExpireSolution(true);
    }

    private void SetLinearAlgebra(LinearAlgebraMode mode)
    {
        if (_linearAlgebra == mode) return;
        RecordUndoEvent("Set Inverse FDM Linear Algebra");
        _linearAlgebra = mode;
        _particular = InverseFdmUiState.UpdateParticular(
            _linearAlgebra, _particular, _hasBox);
        UpdateMessage();
        ExpireSolution(true);
    }

    private void UpdateMessage()
    {
        string residual = _useL2 ? "L2" : "L1";
        string unknown = _solveForQ ? "q-init" : "t-init";
        ActiveInverseEngine engine = InverseFdmUiState.ResolveEngine(
            _linearAlgebra, _particular, _hasBox);
        string engineLabel = engine switch
        {
            ActiveInverseEngine.Clarabel => $"Clarabel λ={FormatLambda(_lambda)}",
            ActiveInverseEngine.MoorePenrose => "MP",
            ActiveInverseEngine.Tikhonov => $"Tikh λ={FormatLambda(_lambda)}",
            ActiveInverseEngine.QrLeastSquares => "QR",
            ActiveInverseEngine.Gram => $"Gram λ={FormatLambda(_lambda)}",
            ActiveInverseEngine.Lsqr when _lambda == 0.0 => "LSQR min-norm",
            ActiveInverseEngine.Lsqr => $"LSQR λ={FormatLambda(_lambda)}",
            _ => $"SPG λ={FormatLambda(_lambda)}",
        };
        string metricLabel = _metric switch
        {
            MetricMode.Geometric when _gnIterations == 0 =>
                $" · CWLS λ={FormatLambda(_cwlsDamping)}",
            MetricMode.Geometric =>
                $" · CWLS-GN×{_gnIterations} λ={FormatLambda(_cwlsDamping)}",
            _ => "",
        };
        Message = $"{engineLabel} · {residual} · {unknown}{metricLabel}";
    }

    private static string FormatLambda(double lambda)
    {
        if (lambda == 0.0) return "0";
        return lambda.ToString("0.###e0", CultureInfo.InvariantCulture);
    }

    public override bool Write(GH_IWriter writer)
    {
        writer.SetInt32(ParticularKey, (int)_particular);
        writer.SetInt32(LinearAlgebraKey, (int)_linearAlgebra);
        writer.SetInt32(MetricKey, (int)_metric);
        return base.Write(writer);
    }

    public override bool Read(GH_IReader reader)
    {
        if (reader.ItemExists(ParticularKey) && Enum.IsDefined(typeof(ParticularMode), reader.GetInt32(ParticularKey)))
            _particular = (ParticularMode)reader.GetInt32(ParticularKey);
        if (reader.ItemExists(LinearAlgebraKey) && Enum.IsDefined(typeof(LinearAlgebraMode), reader.GetInt32(LinearAlgebraKey)))
            _linearAlgebra = (LinearAlgebraMode)reader.GetInt32(LinearAlgebraKey);
        if (reader.ItemExists(MetricKey))
            _metric = reader.GetInt32(MetricKey) == 0 ? MetricMode.Force : MetricMode.Geometric;
        UpdateMessage();
        return base.Read(reader);
    }

    protected override string HtmlHelp_Source() =>
"""
<html>
<body>
<h1>Inverse FDM</h1>
<p>
This component solves a <b>rectangular</b> equilibrium system at a prescribed
target geometry, then runs a forward FDM solve with the recovered force densities.
It is <b>not</b> the inverse of the square form-finding matrix <code>A(q)</code>.
(The older name was Pinv.)
</p>

<br/>

<h2>When to bound the particular</h2>
<p>
A flat or nearly-flat plate cannot carry out-of-plane load with in-plane members.
Unconstrained least squares can then produce huge <code>q</code> or <code>t</code>.
<b>Signs</b>, <b>Lower</b>, and <b>Upper</b> always constrain force density q.
When Stage 1 solves member force, the component transforms the q box with
<code>t = L* q</code>. RelRes will not go to 0 when the target is inconsistent
with the box.
</p>
<p>
These bounds clip the least-squares particular. They are <b>not</b> the
length-preserving transform's sign cone, which stays in equilibrium by adding
self-stress.
</p>

<br/>

<h2>Metric (right-click)</h2>
<p>
Choosing what to minimise matters more than choosing a solver. In FDM the force
residual in the free-node rows at a frozen target is the geometric error
pre-conditioned by the Laplacian:
</p>
<p>
<code>r(q) = E(x*)q − p = D(q)(x* − x(q))</code>, so
<code>x(q) − x* = −D(q)⁻¹ r(q)</code>.
</p>
<ul>
<li><b>Force residual</b> — returns the selected Stage-1 particular without a
compliance-weighted update. It minimises <code>‖r‖</code>. Every nodal
force error counts the same regardless of how flexible that node is, so a small
RelRes can still forward-solve far from the target. This is the historical
behaviour and the right choice for a rigidity/particular question.</li>
<li><b>Geometric residual (default)</b> — applies compliance-weighted
least-squares (CWLS) warm-start updates in q. <b>GNiter = 0</b> performs one
frozen-target update with the Jacobian held at <code>x*</code>. A positive
<b>GNiter</b> performs up to that many Gauss–Newton updates, rebuilding the
Jacobian at <code>x(q_k)</code> and stopping early at Tol. These are true
Gauss–Newton steps for <code>‖x(q) − x*‖</code> and need no forward solve because
<code>x(q_k) = x* − D⁻¹r_k</code>.</li>
</ul>
<p>
Use a geometric metric when you want a <b>warm start</b>: a q whose form-found
shape lands near the target. Use Force when you want the particular that best
closes equilibrium at the frozen shape.
</p>
<p>
CWLS requires <b>L2 = true</b>, geometry-independent loads, and nonsingular
<code>D(q)</code>. Positive q guarantees a positive-definite Laplacian on a
properly anchored connected net. Mixed-sign and all-compression q are supported
with sparse LDLᵀ when D is nonsingular and numerically factorizable under the
current ordering; indefiniteness itself is not an error.
Gram and QR may generate Stage 1, while Stage 2 independently selects a
left-weightable backend.
</p>
<p>
<b>Regularization λ</b> applies only to the Stage-1 particular.
<b>CWLS Damping λcwls</b> applies Levenberg–Marquardt damping
<code>λcwls‖Δq‖²</code> to Stage 2. The geometric objective is nonconvex even
though each inner CWLS problem is convex. <b>GeomErr</b> reports
<code>‖x(q) − x*‖</code> for every metric.
</p>

<br/>

<h2>Linear algebra (right-click)</h2>
<p>
The particular and linear-algebra menus select Stage 1. Stage 2 always works
in q and dispatches independently.
</p>
<ul>
<li><b>Direct</b> — uses the selected Direct solver when unconstrained.
A nonzero Sign or finite q bound selects Clarabel. Bounded CWLS also uses
Clarabel; unbounded CWLS uses the weighted saddle system.</li>
<li><b>Iterative</b> — uses LSQR when unconstrained and SPG with a nonzero Sign
or finite q bound, in both stages. The Direct solver menu is inactive.</li>
</ul>
<p>
Connected ±∞ bounds do not constrain the solve.
</p>

<br/>

<h2>Direct solvers</h2>
<ul>
<li><b>Clarabel (default)</b> — QP least squares, with or without bounds. Uses λ.</li>
<li><b>Moore–Penrose</b> — augmented saddle at λ = 0. Min-norm particular
<code>x = M⁺ p</code>. The λ wire is unused.</li>
<li><b>Tikhonov</b> — damped saddle using the λ wire (λ must be &gt; 0).
<code>min ‖Mx − p‖² + λ‖x‖²</code>.</li>
<li><b>QR least squares</b> — COLAMD sparse QR for tall, full-column-rank systems.
Rank-deficient or wide nets error instead of guessing a particular. λ is unused.
L2 = false wraps QR in IRLS.</li>
<li><b>Gram (normal equations)</b> — <code>(MᵀM + λI)x = Mᵀp</code>. λ is used as-is,
including 0 (pure <code>MᵀM</code>, which fails if singular).</li>
</ul>

<p>
Iterative λ = 0 uses LSQR for a minimum-norm solution; λ &gt; 0 uses regularized LSQR.
Bounded Iterative uses SPG and λ.
</p>

<br/>

<h2>L1 / IRLS</h2>
<p>
For the Force metric, <code>L2 = false</code> wraps the Stage-1 inner solver in
iteratively reweighted least squares with an internal iteration budget. That is
still weighted L2, not a linear program. Bound + IRLS is not exact ℓ₁ with
bounds. The Geometric metric requires <code>L2 = true</code>.
</p>

<br/>

<h2>Input wires</h2>
<ul>
<li><b>Loads / Load Nodes</b> — with no Load Nodes, loads apply to all free nodes in order
(the last load repeats if needed). With Load Nodes, one load broadcasts to every listed node,
or provide one load per listed node; unlisted free nodes receive zero load.</li>
<li><b>Rx0 / Ry0 / Rz0</b> — independently force zero support reaction in X, Y, and/or Z.</li>
<li><b>GNiter</b> — Geometric metric only. 0 runs one frozen-target CWLS update.
A positive integer runs at most that many CWLS-GN updates; Tol may stop them
earlier. The default is 3. Force ignores this input.</li>
<li><b>SolveQ</b> — controls Stage 1 only. False (default) solves member forces
<code>t</code> using unit directions and converts <code>q=t/L*</code>; True
solves the initial particular directly in q. CWLS and downstream L-BFGS-B
always operate in q.</li>
<li><b>Signs / Lower / Upper</b> — always constrain q. A force-space Stage 1
multiplies the bounds by positive target edge lengths. Match or graft to the
edge tree: one value broadcasts globally or within a branch, or provide one
value per edge. Empty channels are unconstrained. Older experimental files that
treated these values as force bounds must divide them by target lengths.</li>
<li><b>MaxIter</b> — inner iteration budget for Clarabel, SPG, and LSQR.
It is independent of GNiter.</li>
<li><b>Tol</b> — inner-solver tolerance and early stopping tolerance for CWLS-GN.</li>
</ul>

<br/>

<h2>Outputs</h2>
<p>
<b>Network / Nodes / Edges</b> come from the forward solve. <b>Forces / Residual / RelRes</b>
are evaluated at the target geometry for the recovered particular.
</p>
<p>
<b>GeomErr</b> is <code>‖x(q) − x*‖</code>, the distance from the forward-solved
geometry to the target. It is a length, not a ratio, and it is the quantity a
warm start should be judged on. RelRes near zero does <i>not</i> imply GeomErr
near zero when the target is not funicular.
</p>
</body>
</html>
""";

    protected override Bitmap Icon => Properties.Resources.parameters;

    public override Guid ComponentGuid => new("E1F2A3B4-C5D6-7890-E1F2-A3B4C5D60001");
}

internal enum ParticularMode
{
    MoorePenrose = 0,
    Tikhonov = 1,
    QrLeastSquares = 2,
    Gram = 3,
    Clarabel = 4,
}

internal enum LinearAlgebraMode { Direct = 0, Iterative = 1 }

/// <summary>Residual family exposed by the component.</summary>
internal enum MetricMode
{
    /// <summary>Minimize the force residual ‖Mx − p‖.</summary>
    Force = 0,
    /// <summary>Minimize compliance-weighted geometric error.</summary>
    Geometric = 1,
}

internal enum ActiveInverseEngine
{
    Clarabel,
    MoorePenrose,
    Tikhonov,
    QrLeastSquares,
    Gram,
    Lsqr,
    Spg,
}

internal static class InverseFdmUiState
{
    internal const ParticularMode DefaultParticular = ParticularMode.Clarabel;
    internal const MetricMode DefaultMetric = MetricMode.Geometric;
    internal const int DefaultGnIterations = 3;
    internal const bool DefaultSolveForQ = false;

    internal static int NativeMetric(MetricMode metric, int gnIterations) =>
        metric switch
        {
            MetricMode.Force => 0,
            _ when gnIterations <= 0 => 1,
            _ => 2,
        };

    internal static int GeometricIterationBudget(MetricMode metric, int gnIterations) =>
        metric == MetricMode.Force ? 0 : Math.Max(1, gnIterations);

    internal static bool HasEffectiveBounds(
        IReadOnlyList<int> signs,
        IReadOnlyList<double> lower,
        IReadOnlyList<double> upper) =>
        signs.Any(sign => sign != 0)
        || lower.Any(double.IsFinite)
        || upper.Any(double.IsFinite);

    internal static bool HasStrictSignDefiniteBounds(
        IReadOnlyList<double> lower,
        IReadOnlyList<double> upper)
    {
        bool allPositive = lower.Count > 0 && lower.All(value => value > 0.0);
        bool allNegative = upper.Count > 0 && upper.All(value => value < 0.0);
        return allPositive || allNegative;
    }

    internal static ParticularMode UpdateParticular(
        LinearAlgebraMode linearAlgebra,
        ParticularMode particular,
        bool hasEffectiveBounds) =>
        linearAlgebra == LinearAlgebraMode.Direct && hasEffectiveBounds
            ? ParticularMode.Clarabel
            : particular;

    /// <summary>
    /// Every Stage-1 backend can initialize CWLS because Stage 2 dispatches
    /// independently to a left-weightable backend.
    /// </summary>
    internal static bool SupportsGeometricMetric(
        LinearAlgebraMode linearAlgebra,
        ParticularMode particular,
        bool hasEffectiveBounds)
    {
        _ = linearAlgebra;
        _ = particular;
        _ = hasEffectiveBounds;
        return true;
    }

    internal static int NativeParticularMethod(ParticularMode particular) =>
        particular switch
        {
            ParticularMode.Gram => 0,
            ParticularMode.QrLeastSquares => 2,
            ParticularMode.Clarabel => 3,
            _ => 1,
        };

    internal static ActiveInverseEngine ResolveEngine(
        LinearAlgebraMode linearAlgebra,
        ParticularMode particular,
        bool hasEffectiveBounds)
    {
        if (hasEffectiveBounds)
        {
            return linearAlgebra == LinearAlgebraMode.Direct
                ? ActiveInverseEngine.Clarabel
                : ActiveInverseEngine.Spg;
        }

        if (linearAlgebra == LinearAlgebraMode.Iterative)
            return ActiveInverseEngine.Lsqr;

        return particular switch
        {
            ParticularMode.Clarabel => ActiveInverseEngine.Clarabel,
            ParticularMode.MoorePenrose => ActiveInverseEngine.MoorePenrose,
            ParticularMode.Tikhonov => ActiveInverseEngine.Tikhonov,
            ParticularMode.QrLeastSquares => ActiveInverseEngine.QrLeastSquares,
            _ => ActiveInverseEngine.Gram,
        };
    }
}
