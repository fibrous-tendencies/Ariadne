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
    private int _frozenIterations = InverseFdmUiState.DefaultFrozenIterations;
    private int _gnIterations = InverseFdmUiState.DefaultGnIterations;
    private bool _solveForQ = InverseFdmUiState.DefaultSolveForQ;
    private bool _hasBox;

    public InverseFdmComponent()
        : base("Inverse FDM", "InvFDM",
            "Build a q warm start from a target particular, optional frozen CWLS, and optional CWLS-GN, then forward-solve.",
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
        pManager.AddIntegerParameter("Frozen CWLS Iterations", "FrozenIter", "Geometric metric only: maximum frozen-target CWLS updates before Gauss–Newton. 0 skips this phase.", GH_ParamAccess.item, InverseFdmUiState.DefaultFrozenIterations);
        pManager.AddIntegerParameter("Gauss–Newton Iterations", "GNiter", "Geometric metric only: maximum CWLS-GN updates after the frozen phase. 0 skips this phase. Stops early at Tol.", GH_ParamAccess.item, InverseFdmUiState.DefaultGnIterations);
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
        const int maxL1Iter = 20;
        int frozenIterations = InverseFdmUiState.DefaultFrozenIterations;
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
        DA.GetData(5, ref frozenIterations);
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
        _frozenIterations = Math.Max(0, frozenIterations);
        _gnIterations = Math.Max(0, gnIterations);
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

            int nativeMetric = InverseFdmUiState.NativeMetric(_metric);
            int frozenBudget = InverseFdmUiState.FrozenIterationBudget(_metric, _frozenIterations);
            int gnBudget = InverseFdmUiState.GaussNewtonIterationBudget(_metric, _gnIterations);
            var result = TheseusSolverService.SolveInverseFdm(
                network, inputs, targetFreeXyz, effectiveRegularization,
                true, maxL1Iter, particularMethod, (int)_linearAlgebra,
                enforceZeroRx, enforceZeroRy, enforceZeroRz, solveForQ,
                [.. signs], [.. lower], [.. upper], maxIter, tol,
                nativeMetric, gnBudget, cwlsDamping, frozenBudget);

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
        string metricLabel = _metric == MetricMode.Geometric
            ? $" · {InverseFdmUiState.PhaseLabel(_frozenIterations, _gnIterations)} λ={FormatLambda(_cwlsDamping)}"
            : "";
        Message = $"{engineLabel} · L2 · {unknown}{metricLabel}";
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
This component constructs a force-density warm start for a prescribed target
geometry <code>x*</code>, then forward-solves the network with the recovered
<code>q</code>. The pipeline is:
</p>
<p>
<b>Stage-1 particular → Frozen CWLS → CWLS-GN → forward FDM.</b>
</p>
<p>
The two CWLS phases are optional and independently budgeted. This is an inverse
equilibrium solve at a frozen geometry, not an algebraic inverse of the square
forward matrix <code>D(q)</code>.
</p>

<h2>Stage 1: equilibrium particular</h2>
<p>
At <code>x*</code>, Stage 1 solves a rectangular linear least-squares problem
for one equilibrium particular. <b>SolveQ = false</b> (default) solves member
forces <code>t</code> using target unit directions, then converts
<code>q = t/L*</code>. <b>SolveQ = true</b> solves force densities directly.
The choice affects only this initializer; all CWLS updates use q.
</p>
<ul>
<li><b>Clarabel (default)</b> — convex quadratic least squares with optional
q bounds and reaction equalities.</li>
<li><b>Moore–Penrose</b> — augmented saddle solve at λ = 0 for a minimum-norm
unconstrained particular.</li>
<li><b>Tikhonov</b> — solves
<code>min ½‖Mz−p‖² + ½λ‖z‖²</code>; λ must be positive.</li>
<li><b>QR least squares</b> — sparse QR for tall, full-column-rank systems;
rank-deficient or wide systems report an error.</li>
<li><b>Gram</b> — solves
<code>(MᵀM + λI)z = Mᵀp</code>; it may be less well-conditioned than QR or
the augmented formulations.</li>
</ul>
<p>
With <b>Iterative</b> linear algebra, unconstrained problems use LSQR and
bounded problems use SPG. <b>MaxIter</b> is the per-inner-solve iteration
budget for Clarabel, LSQR, and SPG; it is not a CWLS phase budget.
</p>

<h2>Metric and compliance weighting</h2>
<p>
For target equilibrium matrix <code>E(x*)</code>, target residual
<code>r(q) = E(x*)q − p</code>, and free-node FDM Laplacian
<code>D(q) = C_fᵀ diag(q) C_f</code> (applied to each coordinate), the exact
constant-load identity is
</p>
<p>
<code>r(q) = D(q)(x* − x(q))</code>, hence
<code>x(q) − x* = −D(q)⁻¹r(q)</code>.
</p>
<ul>
<li><b>Force residual</b> — returns Stage 1 directly and minimises the Euclidean
equilibrium residual. CWLS budgets are ignored.</li>
<li><b>Geometric residual (default)</b> — runs the requested compliance-weighted
phases in q. Set both phase budgets to zero to inspect Stage 1 alone.</li>
</ul>
<p>
All residual objectives in this component are L2. The former L2/L1 toggle and
its IRLS approximation were removed because IRLS was not an exact bounded L1
solve and has no consistent role in compliance-weighted CWLS.
</p>

<h2>Frozen CWLS phase</h2>
<p>
At iteration <code>q_k</code>, Frozen CWLS rebuilds the compliance
<code>D(q_k)⁻¹</code> but keeps the target Jacobian <code>E(x*)</code>. Its
step is the bounded convex least-squares model
</p>
<p>
<code>min_Δq ½‖D(q_k)⁻¹(r(q_k)+E(x*)Δq)‖²
+ ½λcwls‖Δq‖²</code>.
</p>
<p>
This is useful as compliance reweighting of the Stage-1 particular, but it is
not the exact Jacobian of the nonlinear landing map away from the target.
<b>FrozenIter</b> sets its maximum accepted-step attempts; 0 skips it.
</p>

<h2>Gauss–Newton CWLS phase</h2>
<p>
After Frozen CWLS, Gauss–Newton rebuilds both <code>D(q_k)</code> and the
Jacobian <code>E(x(q_k))</code>. For geometry-independent loads,
<code>−D(q_k)⁻¹E(x(q_k))</code> is the derivative of the forward coordinates
with respect to q, so the CWLS-GN step is a true Gauss–Newton model of
<code>½‖x(q)−x*‖²</code>. <b>GNiter</b> sets its maximum; 0 skips it.
</p>
<p>
Both phases backtrack against the exact <b>GeomErr</b>, accept only improving
trials, stop early at Tol, and retain the best point across the whole sequence.
Thus a GN phase cannot replace a better frozen result. λcwls is q-space
Levenberg–Marquardt damping; it does not regularize or shift <code>D(q)</code>.
</p>

<h2>Phase controls</h2>
<ul>
<li><b>FrozenIter = 0, GNiter = 3</b> — default, Gauss–Newton only.</li>
<li><b>FrozenIter = 3, GNiter = 0</b> — frozen-target CWLS only.</li>
<li><b>FrozenIter = 1, GNiter = 0</b> — one compliance reweight.</li>
<li><b>FrozenIter = 3, GNiter = 3</b> — frozen warm-up followed by GN.</li>
<li><b>FrozenIter = 0, GNiter = 0</b> — Stage 1 only.</li>
</ul>
<p>
Budgets are nonnegative and have no hard upper cap. Tol stops a phase when
GeomErr is small, or when both relative improvement and relative q-step are
small. Frozen-phase stagnation does not prevent the requested GN phase from
trying its different Jacobian.
</p>
<p>
Migration note: the former experimental <b>L2</b> input slot is now
<b>FrozenIter</b>. Remove any old Boolean wire and supply a nonnegative integer.
</p>

<h2>Bounds, signs, and invertibility</h2>
<p>
<b>Signs</b>, <b>Lower</b>, and <b>Upper</b> always constrain q in every stage.
When Stage 1 solves t, the component maps the q box through
<code>t = L* q</code> using positive target lengths. One value broadcasts;
otherwise data must match the edge tree. Empty channels are unconstrained.
</p>
<p>
Positive q gives a positive-definite D for a connected, properly anchored net.
Mixed-sign and all-compression systems use sparse LDLᵀ and are valid only when
D is nonsingular and numerically factorizable. Near-zero q and sign cancellation
can create mechanisms or failed trial factors. The solver deliberately uses the
exact compliance: no shifted inverse or pseudoinverse is substituted.
</p>

<h2>Other inputs</h2>
<ul>
<li><b>Loads / Load Nodes</b> — without Load Nodes, loads apply to free nodes
in order and the final load repeats. With Load Nodes, one load broadcasts or
one load per listed node is required; unlisted free nodes receive zero.</li>
<li><b>Rx0 / Ry0 / Rz0</b> — add exact linear zero-reaction constraints to the
particular and CWLS subproblems.</li>
<li><b>Regularization λ</b> — Stage 1 only.</li>
<li><b>CWLS Damping λcwls</b> — both geometric phases only.</li>
<li><b>MaxIter</b> — each inner Clarabel, SPG, or LSQR solve. Independent of
FrozenIter and GNiter.</li>
<li><b>Tol</b> — inner-solver tolerance and geometric phase stopping tolerance.</li>
</ul>

<h2>Outputs</h2>
<p>
<b>Network / Nodes / Edges</b> are the final forward solve.
<b>Forces / Residual / RelRes</b> are evaluated at x* using the returned q.
<b>GeomErr = ‖x(q)−x*‖</b> is a length and directly measures warm-start landing
error. A small force residual ratio does not imply a small GeomErr when the
target is not funicular.
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
    internal const int DefaultFrozenIterations = 0;
    internal const int DefaultGnIterations = 3;
    internal const bool DefaultSolveForQ = false;

    internal static int NativeMetric(MetricMode metric) =>
        metric == MetricMode.Force ? 0 : 2;

    internal static int FrozenIterationBudget(MetricMode metric, int frozenIterations) =>
        metric == MetricMode.Force ? 0 : Math.Max(0, frozenIterations);

    internal static int GaussNewtonIterationBudget(MetricMode metric, int gnIterations) =>
        metric == MetricMode.Force ? 0 : Math.Max(0, gnIterations);

    internal static string PhaseLabel(int frozenIterations, int gnIterations)
    {
        int frozen = Math.Max(0, frozenIterations);
        int gn = Math.Max(0, gnIterations);
        if (frozen == 0 && gn == 0)
            return "Stage 1 only";
        if (frozen == 0)
            return $"GN×{gn}";
        if (gn == 0)
            return $"Frozen×{frozen}";
        return $"Frozen×{frozen} → GN×{gn}";
    }

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
