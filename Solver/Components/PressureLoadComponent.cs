using System;
using System.Collections.Generic;
using System.Drawing;
using Ariadne.FDM;
using Ariadne.Graphs;
using Grasshopper.Kernel;
using Grasshopper.Kernel.Data;
using Grasshopper.Kernel.Types;

namespace Ariadne.Solver.Components;

/// <summary>
/// Creates a pressure load configuration for the Theseus solver.
/// Pressure acts normal to each face with magnitude proportional to area.
/// </summary>
public class PressureLoadComponent : GH_Component
{
    public PressureLoadComponent()
        : base("Pressure Load", "Press",
            "Configure pressure loads on faces for form-finding.",
            "Ariadne", "Design")
    { }

    protected override void RegisterInputParams(GH_InputParamManager pManager)
    {
        pManager.AddGenericParameter("Network", "Network",
            "FDM Network for vertex index validation (optional)", GH_ParamAccess.item);
        pManager.AddIntegerParameter("Face Vertices", "FV",
            "Ordered vertex indices per face (tree: one branch per face)", GH_ParamAccess.tree);
        pManager.AddNumberParameter("Pressure", "p",
            "Pressure magnitude per face (positive = along outward normal). One value = uniform.", GH_ParamAccess.list);
        pManager.AddIntegerParameter("Max Iterations", "MaxIter", "Maximum pressure iterations", GH_ParamAccess.item, 50);
        pManager.AddNumberParameter("Tolerance", "Tol", "Convergence tolerance", GH_ParamAccess.item, 1e-6);
        pManager.AddNumberParameter("Relaxation", "α", "Relaxation factor", GH_ParamAccess.item, 1.0);
        pManager[0].Optional = true;
    }

    protected override void RegisterOutputParams(GH_OutputParamManager pManager)
    {
        pManager.AddGenericParameter("Pressure", "Press", "Pressure load configuration", GH_ParamAccess.item);
    }

    protected override void SolveInstance(IGH_DataAccess DA)
    {
        object? networkObj = null;
        DA.GetData(0, ref networkObj);
        Graph? graph = ResolveGraph(networkObj);

        GH_Structure<GH_Integer> faceTree = new();
        if (!DA.GetDataTree(1, out faceTree)) return;

        List<double> pressures = [];
        int maxIter = 50;
        double tol = 1e-6;
        double relax = 1.0;

        DA.GetDataList(2, pressures);
        DA.GetData(3, ref maxIter);
        DA.GetData(4, ref tol);
        DA.GetData(5, ref relax);

        if (faceTree.PathCount == 0)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Error, "At least one face required.");
            return;
        }
        if (pressures.Count == 0)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Error, "At least one pressure value required.");
            return;
        }

        var faces = new List<List<int>>(faceTree.PathCount);
        for (int b = 0; b < faceTree.PathCount; b++)
        {
            var branch = faceTree.get_Branch(b);
            var face = new List<int>(branch.Count);
            foreach (var item in branch)
            {
                if (item is GH_Integer ghInt)
                    face.Add(ghInt.Value);
            }

            if (graph != null)
            {
                var error = FaceGeometry.ValidateFaceIndices(face, b, graph.Nn);
                if (error != null)
                {
                    AddRuntimeMessage(GH_RuntimeMessageLevel.Error, error);
                    return;
                }
            }

            if (face.Count < 3)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Warning,
                    $"Face {b}: fewer than 3 vertices, skipped.");
                continue;
            }
            faces.Add(face);
        }

        // Expand pressure list to match face count
        var expandedPressures = new List<double>(faces.Count);
        for (int f = 0; f < faces.Count; f++)
            expandedPressures.Add(f < pressures.Count ? pressures[f] : pressures[^1]);

        var config = new PressureConfig.Normal
        {
            Faces = faces,
            Pressures = expandedPressures,
            MaxIters = maxIter,
            Tolerance = tol,
            Relaxation = relax,
        };

        DA.SetData(0, config);
    }

    private static Graph? ResolveGraph(object? graphObj)
    {
        if (graphObj is Graph g)
            return g;
        if (graphObj is FDM_Network net)
            return net.Graph;
        if (graphObj is GH_ObjectWrapper wrapper)
        {
            if (wrapper.Value is Graph wg)
                return wg;
            if (wrapper.Value is FDM_Network wn)
                return wn.Graph;
        }
        return null;
    }

    protected override Bitmap Icon => null!;
    public override Guid ComponentGuid => new("E1F7A6B5-DB82-4EC3-F0A9-6B5C4D7E8F90");
    public override GH_Exposure Exposure => GH_Exposure.hidden;
}
