using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Windows.Forms;
using Ariadne.FDM;
using GH_IO.Serialization;
using Grasshopper.Kernel;
using Rhino.Geometry;
using Theseus.Interop;

namespace Ariadne.Solver.Components.Experimental;

/// <summary>Managed self-stress and first-order mechanism mixer.</summary>
public sealed class InfinitesimalMixComponent : GH_Component
{
    private const string RestoreKey = "FlexRestoreLengths";
    private const int MaxModes = 32;
    private ManagedNullspaceAnalysis? _analysis;
    private bool _restoreLengths;
    private Line[] _arrows = [];

    public InfinitesimalMixComponent()
        : base("Infinitesimal Mix", "Flex",
            "Mix cached self-stress and first-order mechanism modes.",
            "Ariadne", "Experimental")
    {
        UpdateMessage();
    }

    protected override void RegisterInputParams(GH_InputParamManager pManager)
    {
        pManager.AddGenericParameter("Network", "Network", "FDM network topology and supports", GH_ParamAccess.item);
        pManager.AddPointParameter("Target Points", "Target", "Target position of every free node", GH_ParamAccess.list);
        pManager.AddVectorParameter("Loads", "Loads", "Loads on free nodes", GH_ParamAccess.list, new Vector3d(0, 0, -1));
        pManager.AddNumberParameter("Self-Stress Coefficients", "Alpha", "Coefficients alpha in t=t+ + N alpha", GH_ParamAccess.list, 0.0);
        pManager.AddNumberParameter("Mechanism Coefficients", "Beta", "Coefficients beta in x=x0 + Phi beta", GH_ParamAccess.list, 0.0);
        pManager.AddNumberParameter("Mechanism Scale", "k", "Scale applied to the first-order displacement and arrows", GH_ParamAccess.item, 1.0);
    }

    protected override void RegisterOutputParams(GH_OutputParamManager pManager)
    {
        pManager.AddGenericParameter("Network", "Network", "Mixed network", GH_ParamAccess.item);
        pManager.AddNumberParameter("Member Forces", "t", "Mixed member forces", GH_ParamAccess.list);
        pManager.AddNumberParameter("Force Densities", "q", "Mixed forces divided by current member lengths", GH_ParamAccess.list);
        pManager.AddPointParameter("Points", "X", "Mixed free-node positions", GH_ParamAccess.list);
        pManager.AddVectorParameter("Mechanism Arrows", "Arrows", "First-order displacement arrows", GH_ParamAccess.list);
        pManager.AddTextParameter("Mechanism Classes", "Class", "Prestress classification of rotated mechanism modes", GH_ParamAccess.list);
        pManager.AddNumberParameter("Stiffness Eigenvalues", "Eig", "Restricted geometric-stiffness eigenvalues", GH_ParamAccess.list);
        pManager.AddNumberParameter("Maximum Length Error", "Lerr", "Maximum per-member length error after optional retraction", GH_ParamAccess.item);
    }

    protected override void SolveInstance(IGH_DataAccess DA)
    {
        _arrows = [];
        FDM_Network? network = null;
        List<Point3d> target = [];
        List<Vector3d> loads = [];
        List<double> alpha = [];
        List<double> beta = [];
        double scale = 1.0;
        if (!DA.GetData(0, ref network) || network == null) return;
        if (!DA.GetDataList(1, target) || !DA.GetDataList(2, loads)) return;
        DA.GetDataList(3, alpha);
        DA.GetDataList(4, beta);
        DA.GetData(5, ref scale);
        if (!network.Valid || target.Count != network.FreeNodes.Count)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Error,
                "Network must be valid and Target must contain every free node.");
            return;
        }

        try
        {
            int key = NullspaceComponentUtilities.AnalysisKey(network, target, loads, MaxModes);
            if (_analysis == null || _analysis.Key != key)
                _analysis = NullspaceComponentUtilities.Analyze(network, target, loads, MaxModes, key);

            RigidityReport report = _analysis.Report;
            int stressModes = report.SelfStressModeCount;
            int mechanismModes = _analysis.Mechanisms.Length == 0
                ? 0 : _analysis.Mechanisms.Length / (target.Count * 3);
            double[] forces = NullspaceComponentUtilities.MixColumns(
                report.ParticularForces, report.SelfStressBasis, stressModes, alpha);
            double[] targetXyz = NullspaceComponentUtilities.PackPoints(target);
            double[] linearXyz = NullspaceComponentUtilities.Displace(
                targetXyz, _analysis.Mechanisms, mechanismModes, beta, scale);
            double[] finalXyz = linearXyz;
            if (_restoreLengths)
            {
                LengthRetractionResult retraction = TheseusSolverService.RetractMemberLengths(
                    network, NullspaceComponentUtilities.Inputs(network, loads),
                    linearXyz, _analysis.ReferenceLengths);
                finalXyz = retraction.FreeXyz;
                if (!retraction.Converged)
                    AddRuntimeMessage(GH_RuntimeMessageLevel.Warning,
                        $"Length retraction stopped at error {retraction.MaxLengthError:G3}.");
            }

            FDM_Network output = NullspaceComponentUtilities.BuildNetwork(network, finalXyz, forces);
            Point3d[] points = NullspaceComponentUtilities.UnpackPoints(finalXyz);
            var arrows = new Vector3d[target.Count];
            _arrows = new Line[target.Count];
            for (int i = 0; i < target.Count; i++)
            {
                Point3d linear = new(linearXyz[i * 3], linearXyz[i * 3 + 1], linearXyz[i * 3 + 2]);
                arrows[i] = linear - target[i];
                _arrows[i] = new Line(target[i], linear);
            }
            double maxLengthError = output.Graph.Edges
                .Select((edge, index) => Math.Abs(edge.Start.Value.DistanceTo(edge.End.Value)
                    - _analysis.ReferenceLengths[index]))
                .DefaultIfEmpty(0.0).Max();

            DA.SetData(0, output);
            DA.SetDataList(1, forces);
            DA.SetDataList(2, output.Graph.Edges.Select(edge => edge.Q));
            DA.SetDataList(3, points);
            DA.SetDataList(4, arrows);
            DA.SetDataList(5, _analysis.Classes.Select(value => value switch
            {
                MechanismClass.PrestressStable => "Prestress stable",
                MechanismClass.PrestressUnstable => "Prestress unstable",
                _ => "Finite candidate",
            }));
            DA.SetDataList(6, _analysis.Eigenvalues);
            DA.SetData(7, maxLengthError);
        }
        catch (Exception error)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Error, error.Message);
        }
    }

    public override void DrawViewportWires(IGH_PreviewArgs args)
    {
        base.DrawViewportWires(args);
        if (_arrows.Length > 0)
            args.Display.DrawArrows(_arrows, Color.FromArgb(230, 244, 129, 32));
    }

    protected override void AppendAdditionalComponentMenuItems(ToolStripDropDown menu)
    {
        base.AppendAdditionalComponentMenuItems(menu);
        Menu_AppendSeparator(menu);
        Menu_AppendItem(menu, "Restore member lengths", (_, _) =>
        {
            RecordUndoEvent("Restore Member Lengths");
            _restoreLengths = !_restoreLengths;
            UpdateMessage();
            ExpireSolution(true);
        }, true, _restoreLengths);
    }

    private void UpdateMessage() => Message = _restoreLengths ? "Managed mix | Retract" : "Managed mix | First order";

    public override bool Write(GH_IWriter writer)
    {
        writer.SetBoolean(RestoreKey, _restoreLengths);
        return base.Write(writer);
    }

    public override bool Read(GH_IReader reader)
    {
        if (reader.ItemExists(RestoreKey)) _restoreLengths = reader.GetBoolean(RestoreKey);
        UpdateMessage();
        return base.Read(reader);
    }

    protected override string HtmlHelp_Source() =>
        "<h1>Infinitesimal Mix</h1><p>Analyzes once, caches managed bases, and evaluates t=t+ + N alpha and x=x0 + Phi beta in C#. " +
        "Coefficient slider changes therefore do not acquire the Theseus analysis lock. Phi beta is first-order: a mode classified as prestress-stable can look mobile without defining a finite path. " +
        "Optional retraction restores every member's own reference length; use Length-Preserving Transform for finite motion.</p>";

    protected override Bitmap Icon => Properties.Resources.parameters;
    public override Guid ComponentGuid => new("5BE7A99B-37A0-4ECB-ACFD-791D64BE6430");
    public override GH_Exposure Exposure => GH_Exposure.hidden;
}
