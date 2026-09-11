using System;
using System.Collections.Generic;
using System.Drawing;
using Grasshopper.Kernel;
using Grasshopper.Kernel.Data;
using Grasshopper.Kernel.Types;
using Rhino.Geometry;

namespace Ariadne.Solver.Components;

/// <summary>
/// Creates a hydrostatic pressure load configuration.
/// Pressure varies linearly with depth below a reference datum,
/// suitable for fabric formwork with wet concrete.
/// </summary>
public class HydrostaticPressureComponent : GH_Component
{
    public HydrostaticPressureComponent()
        : base("Pressure Load (Hydrostatic)", "Press-H",
            "Hydrostatic pressure varying with depth below a datum (e.g. fabric formwork).",
            "Ariadne", "Design")
    { }

    protected override void RegisterInputParams(GH_InputParamManager pManager)
    {
        pManager.AddIntegerParameter("Face Vertices", "FV",
            "Ordered vertex indices per face (tree: one branch per face)", GH_ParamAccess.tree);
        pManager.AddNumberParameter("Fluid Density", "ρ",
            "Fluid density in kg/m³ (e.g. 2400 for wet concrete)", GH_ParamAccess.item);
        pManager.AddNumberParameter("g Magnitude", "g",
            "Gravitational acceleration magnitude", GH_ParamAccess.item, 9.81);
        pManager.AddNumberParameter("Datum", "z₀",
            "Free surface elevation (top of pour) along the up direction", GH_ParamAccess.item);
        pManager.AddVectorParameter("Up Direction", "Up",
            "Unit 'up' direction (depth measured opposite)", GH_ParamAccess.item, new Vector3d(0, 0, 1));
        pManager.AddIntegerParameter("Max Iterations", "MaxIter",
            "Maximum pressure iterations", GH_ParamAccess.item, 50);
        pManager.AddNumberParameter("Tolerance", "Tol",
            "Convergence tolerance", GH_ParamAccess.item, 1e-6);
        pManager.AddNumberParameter("Relaxation", "α",
            "Relaxation factor", GH_ParamAccess.item, 1.0);
    }

    protected override void RegisterOutputParams(GH_OutputParamManager pManager)
    {
        pManager.AddGenericParameter("Pressure", "Press", "Hydrostatic pressure configuration", GH_ParamAccess.item);
    }

    protected override void SolveInstance(IGH_DataAccess DA)
    {
        GH_Structure<GH_Integer> faceTree = new();
        if (!DA.GetDataTree(0, out faceTree)) return;

        double rhoFluid = 0;
        double gMag = 9.81;
        double zDatum = 0;
        Vector3d upDir = new(0, 0, 1);
        int maxIter = 50;
        double tol = 1e-6;
        double relax = 1.0;

        if (!DA.GetData(1, ref rhoFluid)) return;
        DA.GetData(2, ref gMag);
        if (!DA.GetData(3, ref zDatum)) return;
        DA.GetData(4, ref upDir);
        DA.GetData(5, ref maxIter);
        DA.GetData(6, ref tol);
        DA.GetData(7, ref relax);

        if (faceTree.PathCount == 0)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Error, "At least one face required.");
            return;
        }
        if (rhoFluid <= 0)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Error, "Fluid density must be positive.");
            return;
        }

        upDir.Unitize();

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

        var config = new PressureConfig.Hydrostatic
        {
            Faces = faces,
            RhoFluid = rhoFluid,
            GMagnitude = gMag,
            ZDatum = zDatum,
            UpDirection = upDir,
            MaxIters = maxIter,
            Tolerance = tol,
            Relaxation = relax,
        };

        DA.SetData(0, config);
    }

    protected override Bitmap Icon => null!;
    public override Guid ComponentGuid => new("F2A8B7C6-ED93-4FD4-A1B0-7C6D5E8F9A01");
    public override GH_Exposure Exposure => GH_Exposure.hidden;
}
