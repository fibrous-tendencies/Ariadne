using System;
using System.Collections.Generic;
using System.Drawing;
using System.Windows.Forms;
using Ariadne.FDM;
using GH_IO.Serialization;
using Grasshopper.Kernel;
using Grasshopper.Kernel.Data;
using Grasshopper.Kernel.Types;
using Rhino.Geometry;
using Theseus.Interop;

namespace Ariadne.Solver.Components.Experimental;

/// <summary>Experimental Pellegrino–Calladine analysis of a target network.</summary>
public sealed class RigidityReportComponent : GH_Component
{
    private const string MethodKey = "RigidityMethod";
    private const string IncludeRigidKey = "IncludeRigidBodies";
    private const int MaxModes = 32;

    private RigidityMethod _method = RigidityMethod.Projector;
    private bool _includeRigidBodies;

    public RigidityReportComponent()
        : base("Rigidity Report", "Rigidity",
            "Report equilibrium rank, self-stress, and infinitesimal mechanisms.",
            "Ariadne", "Experimental")
    {
        UpdateMessage();
    }

    protected override void RegisterInputParams(GH_InputParamManager pManager)
    {
        pManager.AddGenericParameter("Network", "Network", "FDM network topology and supports", GH_ParamAccess.item);
        pManager.AddPointParameter("Target Points", "Target", "Target position of every free node", GH_ParamAccess.list);
        pManager.AddVectorParameter("Loads", "Loads", "Loads on free nodes", GH_ParamAccess.list, new Vector3d(0, 0, -1));
    }

    protected override void RegisterOutputParams(GH_OutputParamManager pManager)
    {
        pManager.AddIntegerParameter("Rank", "Rank", "Numerical rank of the force equilibrium matrix", GH_ParamAccess.item);
        pManager.AddIntegerParameter("Self Stress Count", "s", "Self-stress nullity", GH_ParamAccess.item);
        pManager.AddIntegerParameter("Raw Mechanism Count", "mRaw", "Mechanisms before rigid-body removal", GH_ParamAccess.item);
        pManager.AddIntegerParameter("Mechanism Count", "m", "Mechanisms after rigid-body removal", GH_ParamAccess.item);
        pManager.AddIntegerParameter("Rigid Body Count", "Rigid", "Detected rigid-body modes", GH_ParamAccess.item);
        pManager.AddNumberParameter("Particular Forces", "t+", "Minimum-norm particular member forces", GH_ParamAccess.list);
        pManager.AddNumberParameter("Self Stress Basis", "N", "Self-stress basis; one tree branch per mode", GH_ParamAccess.tree);
        pManager.AddVectorParameter("Mechanism Basis", "Phi", "Free-node displacement vectors; one tree branch per mode", GH_ParamAccess.tree);
        pManager.AddVectorParameter("Rigid Body Basis", "RB", "Rigid-body vectors when included", GH_ParamAccess.tree);
        pManager.AddNumberParameter("Residual Ratio", "RelRes", "Relative equilibrium residual", GH_ParamAccess.item);
    }

    protected override void SolveInstance(IGH_DataAccess DA)
    {
        FDM_Network? network = null;
        List<Point3d> target = [];
        List<Vector3d> loads = [];
        if (!DA.GetData(0, ref network) || network == null) return;
        if (!DA.GetDataList(1, target)) return;
        if (!DA.GetDataList(2, loads)) return;

        if (!network.Valid)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Error, "Network is invalid.");
            return;
        }
        if (target.Count != network.FreeNodes.Count)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Error,
                $"Target count {target.Count} must match free-node count {network.FreeNodes.Count}.");
            return;
        }

        var targetXyz = new double[target.Count * 3];
        for (int i = 0; i < target.Count; i++)
        {
            targetXyz[i * 3] = target[i].X;
            targetXyz[i * 3 + 1] = target[i].Y;
            targetXyz[i * 3 + 2] = target[i].Z;
        }
        var q = new List<double>(network.Graph.Ne);
        foreach (var edge in network.Graph.Edges)
            q.Add(double.IsFinite(edge.Q) ? edge.Q : 1.0);

        try
        {
            var report = TheseusSolverService.AnalyzeRigidity(
                network,
                new SolverInputs { QInit = q, Loads = loads },
                targetXyz,
                _method,
                _includeRigidBodies,
                MaxModes);

            DA.SetData(0, report.Rank);
            DA.SetData(1, report.SelfStressCount);
            DA.SetData(2, report.RawMechanismCount);
            DA.SetData(3, report.MechanismCount);
            DA.SetData(4, report.RigidBodyCount);
            DA.SetDataList(5, report.ParticularForces);
            DA.SetDataTree(6, NumberModeTree(report.SelfStressBasis, network.Graph.Ne));
            DA.SetDataTree(7, VectorModeTree(report.MechanismBasis, network.FreeNodes.Count));
            DA.SetDataTree(8, VectorModeTree(report.RigidBodyBasis, network.FreeNodes.Count));
            DA.SetData(9, report.ResidualRatio);
            if (report.ResidualRatio > 1e-4)
                AddRuntimeMessage(GH_RuntimeMessageLevel.Warning,
                    $"Equilibrium residual is {report.ResidualRatio:G3}.");
        }
        catch (Exception ex)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Error, ex.Message);
        }
    }

    private static GH_Structure<GH_Number> NumberModeTree(double[] values, int rows)
    {
        var tree = new GH_Structure<GH_Number>();
        if (rows == 0) return tree;
        int modes = values.Length / rows;
        for (int mode = 0; mode < modes; mode++)
        {
            var path = new GH_Path(mode);
            for (int row = 0; row < rows; row++)
                tree.Append(new GH_Number(values[row * modes + mode]), path);
        }
        return tree;
    }

    private static GH_Structure<GH_Vector> VectorModeTree(double[] values, int nodes)
    {
        var tree = new GH_Structure<GH_Vector>();
        int rows = nodes * 3;
        if (rows == 0) return tree;
        int modes = values.Length / rows;
        for (int mode = 0; mode < modes; mode++)
        {
            var path = new GH_Path(mode);
            for (int node = 0; node < nodes; node++)
            {
                tree.Append(new GH_Vector(new Vector3d(
                    values[node * modes + mode],
                    values[(nodes + node) * modes + mode],
                    values[(2 * nodes + node) * modes + mode])), path);
            }
        }
        return tree;
    }

    protected override void AppendAdditionalComponentMenuItems(ToolStripDropDown menu)
    {
        base.AppendAdditionalComponentMenuItems(menu);
        Menu_AppendSeparator(menu);
        AppendMethod(menu, "Projector", RigidityMethod.Projector);
        AppendMethod(menu, "Dense SVD", RigidityMethod.DenseSvd);
        AppendMethod(menu, "Sparse QR (comparison)", RigidityMethod.SparseQr);
        Menu_AppendSeparator(menu);
        Menu_AppendItem(menu, "Include rigid bodies", (_, _) => SetIncludeRigidBodies(),
            true, _includeRigidBodies);
    }

    private void AppendMethod(ToolStripDropDown menu, string label, RigidityMethod method) =>
        Menu_AppendItem(menu, label, (_, _) => SetMethod(method), true, _method == method);

    private void SetMethod(RigidityMethod method)
    {
        if (_method == method) return;
        RecordUndoEvent("Set Rigidity Method");
        _method = method;
        UpdateMessage();
        ExpireSolution(true);
    }

    private void SetIncludeRigidBodies()
    {
        RecordUndoEvent("Include Rigid Bodies");
        _includeRigidBodies = !_includeRigidBodies;
        UpdateMessage();
        ExpireSolution(true);
    }

    private void UpdateMessage() =>
        Message = $"{(_method switch { RigidityMethod.Projector => "Projector", RigidityMethod.DenseSvd => "Dense SVD", _ => "Sparse QR" })} | {(_includeRigidBodies ? "Rigid on" : "Rigid off")}";

    public override bool Write(GH_IWriter writer)
    {
        writer.SetInt32(MethodKey, (int)_method);
        writer.SetBoolean(IncludeRigidKey, _includeRigidBodies);
        return base.Write(writer);
    }

    public override bool Read(GH_IReader reader)
    {
        if (reader.ItemExists(MethodKey))
        {
            int value = reader.GetInt32(MethodKey);
            if (Enum.IsDefined(typeof(RigidityMethod), value))
                _method = (RigidityMethod)value;
        }
        if (reader.ItemExists(IncludeRigidKey))
            _includeRigidBodies = reader.GetBoolean(IncludeRigidKey);
        UpdateMessage();
        return base.Read(reader);
    }

    protected override string HtmlHelp_Source() =>
        "<h1>Rigidity Report</h1><p>Computes the Pellegrino–Calladine null spaces of the force equilibrium matrix A. " +
        "Projector is the sparse large-network method. Dense SVD densifies A and is intended for small fixtures. " +
        "Sparse QR is an explicit comparison mode and may reject rank-deficient systems. This is rigidity analysis, not form-finding.</p>";

    protected override Bitmap Icon => Properties.Resources.parameters;
    public override Guid ComponentGuid => new("D36DBFF8-6266-45A2-A5AF-017C60943191");
    public override GH_Exposure Exposure => GH_Exposure.hidden;
}
