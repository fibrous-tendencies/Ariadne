using System;
using System.Collections.Generic;
using System.Drawing;
using Grasshopper.Kernel;
using Grasshopper.Kernel.Data;
using Grasshopper.Kernel.Types;
using Rhino.Geometry;

namespace Ariadne.Solver.Components;

/// <summary>
/// Creates a directional pressure load configuration.
/// Load is proportional to each face's projected area in a fixed direction,
/// suitable for uniform dead loads (snow, soil pressure).
/// </summary>
public class DirectionalPressureComponent : GH_Component
{
    public DirectionalPressureComponent()
        : base("Pressure Load (Directional)", "Press-D",
            "Directional pressure proportional to projected face area (e.g. dead load, soil pressure).",
            "Ariadne", "Design")
    { }

    protected override void RegisterInputParams(GH_InputParamManager pManager)
    {
        pManager.AddIntegerParameter("Face Vertices", "FV",
            "Ordered vertex indices per face (tree: one branch per face)", GH_ParamAccess.tree);
        pManager.AddNumberParameter("Pressure", "p",
            "Pressure magnitude per face. One value = uniform.", GH_ParamAccess.list);
        pManager.AddVectorParameter("Direction", "d",
            "Unit load direction (e.g. {0,0,-1} for gravity dead load)", GH_ParamAccess.item);
        pManager.AddIntegerParameter("Max Iterations", "MaxIter",
            "Maximum pressure iterations", GH_ParamAccess.item, 50);
        pManager.AddNumberParameter("Tolerance", "Tol",
            "Convergence tolerance", GH_ParamAccess.item, 1e-6);
        pManager.AddNumberParameter("Relaxation", "α",
            "Relaxation factor", GH_ParamAccess.item, 1.0);
    }

    protected override void RegisterOutputParams(GH_OutputParamManager pManager)
    {
        pManager.AddGenericParameter("Pressure", "Press", "Directional pressure configuration", GH_ParamAccess.item);
    }

    protected override void SolveInstance(IGH_DataAccess DA)
    {
        GH_Structure<GH_Integer> faceTree = new();
        if (!DA.GetDataTree(0, out faceTree)) return;

        List<double> pressures = [];
        Vector3d direction = Vector3d.Unset;
        int maxIter = 50;
        double tol = 1e-6;
        double relax = 1.0;

        DA.GetDataList(1, pressures);
        if (!DA.GetData(2, ref direction)) return;
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

        direction.Unitize();

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
            if (face.Count < 3)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Warning,
                    $"Face {b}: fewer than 3 vertices, skipped.");
                continue;
            }
            faces.Add(face);
        }

        var expandedPressures = new List<double>(faces.Count);
        for (int f = 0; f < faces.Count; f++)
            expandedPressures.Add(f < pressures.Count ? pressures[f] : pressures[^1]);

        var config = new PressureConfig.Directional
        {
            Faces = faces,
            Pressures = expandedPressures,
            Direction = direction,
            MaxIters = maxIter,
            Tolerance = tol,
            Relaxation = relax,
        };

        DA.SetData(0, config);
    }

    protected override Bitmap Icon => null!;
    public override Guid ComponentGuid => new("A3B9C8D7-FE04-4AE5-B2C1-8D7E6F9A0B12");
    public override GH_Exposure Exposure => GH_Exposure.hidden;
}
