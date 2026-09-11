using System;
using System.Collections.Generic;
using System.Drawing;
using Grasshopper.Kernel;
using Rhino.Geometry;

namespace Ariadne.Solver.Components;

/// <summary>
/// Creates a self-weight configuration for the Theseus solver.
/// Two modes: prescribed linear density per edge, or force-based sizing.
/// </summary>
public class SelfWeightPrescribedComponent : GH_Component
{
    public SelfWeightPrescribedComponent()
        : base("Self-Weight (Prescribed)", "SW-Presc",
            "Configure prescribed linear density self-weight for form-finding.",
            "Ariadne", "Design")
    { }

    protected override void RegisterInputParams(GH_InputParamManager pManager)
    {
        pManager.AddNumberParameter("Linear Density", "μ",
            "Linear density per edge (kg/m). One value = uniform for all edges.", GH_ParamAccess.list);
        pManager.AddVectorParameter("Gravity", "g", "Gravity vector", GH_ParamAccess.item, new Vector3d(0, 0, -9.81));
        pManager.AddIntegerParameter("Max Iterations", "MaxIter", "Maximum self-weight iterations", GH_ParamAccess.item, 50);
        pManager.AddNumberParameter("Tolerance", "Tol", "Convergence tolerance", GH_ParamAccess.item, 1e-6);
        pManager.AddNumberParameter("Relaxation", "α", "Relaxation factor (1.0 = no relaxation)", GH_ParamAccess.item, 1.0);
    }

    protected override void RegisterOutputParams(GH_OutputParamManager pManager)
    {
        pManager.AddGenericParameter("Self-Weight", "SW", "Self-weight configuration", GH_ParamAccess.item);
    }

    protected override void SolveInstance(IGH_DataAccess DA)
    {
        List<double> mu = [];
        Vector3d gravity = new(0, 0, -9.81);
        int maxIter = 50;
        double tol = 1e-6;
        double relax = 1.0;

        DA.GetDataList(0, mu);
        DA.GetData(1, ref gravity);
        DA.GetData(2, ref maxIter);
        DA.GetData(3, ref tol);
        DA.GetData(4, ref relax);

        if (mu.Count == 0)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Error, "At least one linear density value required.");
            return;
        }

        var config = new SelfWeightConfig.Prescribed
        {
            LinearDensities = mu,
            Gravity = gravity,
            MaxIters = maxIter,
            Tolerance = tol,
            Relaxation = relax,
        };

        DA.SetData(0, config);
    }

    protected override Bitmap Icon => null!;
    public override Guid ComponentGuid => new("C9D5E4F3-BF60-4CA1-D8E7-4A3F2B5C6D7E");
    public override GH_Exposure Exposure => GH_Exposure.hidden;
}

/// <summary>
/// Creates a force-based sizing self-weight configuration.
/// Cross-section area is derived as A_k = |F_k| / sigma.
/// </summary>
public class SelfWeightSizingComponent : GH_Component
{
    public SelfWeightSizingComponent()
        : base("Self-Weight (Sizing)", "SW-Size",
            "Configure force-based sizing self-weight. A_k = |F_k| / σ.",
            "Ariadne", "Design")
    { }

    protected override void RegisterInputParams(GH_InputParamManager pManager)
    {
        pManager.AddNumberParameter("Density", "ρ", "Material density (kg/m³)", GH_ParamAccess.item);
        pManager.AddNumberParameter("Allowable Stress", "σ", "Allowable stress (Pa or consistent units)", GH_ParamAccess.item);
        pManager.AddVectorParameter("Gravity", "g", "Gravity vector", GH_ParamAccess.item, new Vector3d(0, 0, -9.81));
        pManager.AddIntegerParameter("Max Iterations", "MaxIter", "Maximum self-weight iterations", GH_ParamAccess.item, 50);
        pManager.AddNumberParameter("Tolerance", "Tol", "Convergence tolerance", GH_ParamAccess.item, 1e-6);
        pManager.AddNumberParameter("Relaxation", "α", "Relaxation factor (1.0 = no relaxation)", GH_ParamAccess.item, 1.0);
    }

    protected override void RegisterOutputParams(GH_OutputParamManager pManager)
    {
        pManager.AddGenericParameter("Self-Weight", "SW", "Self-weight configuration", GH_ParamAccess.item);
    }

    protected override void SolveInstance(IGH_DataAccess DA)
    {
        double rho = 0;
        double sigma = 0;
        Vector3d gravity = new(0, 0, -9.81);
        int maxIter = 50;
        double tol = 1e-6;
        double relax = 1.0;

        if (!DA.GetData(0, ref rho)) return;
        if (!DA.GetData(1, ref sigma)) return;
        DA.GetData(2, ref gravity);
        DA.GetData(3, ref maxIter);
        DA.GetData(4, ref tol);
        DA.GetData(5, ref relax);

        if (rho <= 0)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Error, "Density must be positive.");
            return;
        }
        if (sigma <= 0)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Error, "Allowable stress must be positive.");
            return;
        }

        var config = new SelfWeightConfig.Sizing
        {
            Rho = rho,
            Sigma = sigma,
            Gravity = gravity,
            MaxIters = maxIter,
            Tolerance = tol,
            Relaxation = relax,
        };

        DA.SetData(0, config);
    }

    protected override Bitmap Icon => null!;
    public override Guid ComponentGuid => new("D0E6F5A4-CA71-4DB2-E9F8-5A4B3C6D7E8F");
    public override GH_Exposure Exposure => GH_Exposure.hidden;
}
