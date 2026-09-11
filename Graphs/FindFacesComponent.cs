using Grasshopper.Kernel;
using Grasshopper.Kernel.Data;
using Grasshopper.Kernel.Types;
using Rhino.Geometry;
using System;
using System.Collections.Generic;
using System.Drawing;

namespace Ariadne.Graphs;

/// <summary>
/// Discovers all bounded face cycles in a topologically planar graph embedded in 3D.
/// Uses local tangent frames at each vertex for angular sorting, then walks
/// half-edges to extract faces. Validates with Euler's formula and detects non-planarity.
/// </summary>
public class FindFacesComponent : GH_Component
{
    public FindFacesComponent()
        : base("Find Faces", "FindFaces",
            "Find bounded face cycles in a graph/network. Works directly on 3D geometry without projection.",
            "Ariadne", "Graphs")
    { }

    protected override void RegisterInputParams(GH_InputParamManager pManager)
    {
        pManager.AddGenericParameter("Graph", "Graph", "Graph or FDM Network to find faces in", GH_ParamAccess.item);
    }

    protected override void RegisterOutputParams(GH_OutputParamManager pManager)
    {
        pManager.AddIntegerParameter("Faces", "F", "Face vertex indices (tree: one branch per face)", GH_ParamAccess.tree);
        pManager.AddCurveParameter("Face Polylines", "P", "Face boundary polylines for visualization", GH_ParamAccess.list);
        pManager.AddVectorParameter("Face Normals", "N", "Face normals (Newell method)", GH_ParamAccess.list);
        pManager.AddIntegerParameter("Euler Characteristic", "χ", "Euler characteristic V-E+F", GH_ParamAccess.item);
        pManager.AddBooleanParameter("Is Manifold", "M", "True if graph is a valid 2-manifold", GH_ParamAccess.item);
    }

    protected override void SolveInstance(IGH_DataAccess DA)
    {
        object graphObj = null!;
        if (!DA.GetData(0, ref graphObj)) return;

        Graph? graph = null;
        if (graphObj is Graph g)
            graph = g;
        else if (graphObj is FDM.FDM_Network net)
            graph = net.Graph;
        else if (graphObj is GH_ObjectWrapper wrapper)
        {
            if (wrapper.Value is Graph wg)
                graph = wg;
            else if (wrapper.Value is FDM.FDM_Network wn)
                graph = wn.Graph;
        }

        if (graph == null)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Error, "Input must be a Graph or FDM Network.");
            return;
        }

        if (graph.Nn < 3 || graph.Ne < 3)
        {
            AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "Graph too small for face-finding (need ≥3 nodes and edges).");
            return;
        }

        graph.UpdateNodeIndices();
        var result = Graph.FindFaces(graph);

        foreach (var warning in result.Warnings)
            AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, warning);

        // Build face index tree
        var faceTree = new GH_Structure<GH_Integer>();
        for (int f = 0; f < result.Faces.Count; f++)
        {
            var path = new GH_Path(f);
            foreach (int idx in result.Faces[f])
                faceTree.Append(new GH_Integer(idx), path);
        }

        // Build polylines and normals
        var polylines = new List<Polyline>(result.Faces.Count);
        var normals = new List<Vector3d>(result.Faces.Count);
        foreach (var face in result.Faces)
        {
            var pts = new List<Point3d>(face.Count + 1);
            foreach (int idx in face)
                pts.Add(graph.Nodes[idx].Value);
            pts.Add(pts[0]); // close
            polylines.Add(new Polyline(pts));
            normals.Add(FaceGeometry.ComputeNewellNormal(pts));
        }

        DA.SetDataTree(0, faceTree);
        DA.SetDataList(1, polylines.ConvertAll(p => new PolylineCurve(p)));
        DA.SetDataList(2, normals);
        DA.SetData(3, result.EulerCharacteristic);
        DA.SetData(4, result.IsManifold);
    }

    protected override Bitmap Icon => null!;
    public override Guid ComponentGuid => new Guid("A7F3B2C1-9D4E-4A8F-B6C5-2E1D0F3A4B5C");
    public override GH_Exposure Exposure => GH_Exposure.hidden;
}
