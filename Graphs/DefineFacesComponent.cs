using Grasshopper.Kernel;
using Grasshopper.Kernel.Data;
using Grasshopper.Kernel.Types;
using Rhino.Geometry;
using System;
using System.Collections.Generic;
using System.Drawing;

namespace Ariadne.Graphs;

/// <summary>
/// Accepts a DataTree of integers (one branch per face, each branch = ordered vertex indices)
/// and a Graph/Network, validates indices, and outputs face data suitable for pressure loads.
/// </summary>
public class DefineFacesComponent : GH_Component
{
    public DefineFacesComponent()
        : base("Define Faces", "DefFaces",
            "Define face topology from vertex index lists. Each branch is one face.",
            "Ariadne", "Graphs")
    { }

    protected override void RegisterInputParams(GH_InputParamManager pManager)
    {
        pManager.AddGenericParameter("Graph", "Graph", "Graph or FDM Network", GH_ParamAccess.item);
        pManager.AddIntegerParameter("Face Vertices", "FV",
            "Ordered vertex indices per face (tree: one branch per face)", GH_ParamAccess.tree);
    }

    protected override void RegisterOutputParams(GH_OutputParamManager pManager)
    {
        pManager.AddIntegerParameter("Faces", "F", "Validated face vertex indices (tree)", GH_ParamAccess.tree);
        pManager.AddCurveParameter("Face Polylines", "P", "Face boundary polylines", GH_ParamAccess.list);
        pManager.AddVectorParameter("Face Normals", "N", "Face normals (Newell method)", GH_ParamAccess.list);
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

        GH_Structure<GH_Integer> faceTree = new();
        if (!DA.GetDataTree(1, out faceTree)) return;

        var faces = new List<List<int>>();
        var polylines = new List<PolylineCurve>();
        var normals = new List<Vector3d>();

        for (int b = 0; b < faceTree.PathCount; b++)
        {
            var branch = faceTree.get_Branch(b);
            var face = new List<int>(branch.Count);
            bool valid = true;

            foreach (var item in branch)
            {
                if (item is GH_Integer ghInt)
                {
                    int idx = ghInt.Value;
                    if (idx < 0 || idx >= graph.Nn)
                    {
                        AddRuntimeMessage(GH_RuntimeMessageLevel.Error,
                            $"Face {b}: vertex index {idx} out of range [0, {graph.Nn - 1}].");
                        valid = false;
                        break;
                    }
                    face.Add(idx);
                }
            }

            if (!valid || face.Count < 3)
            {
                if (valid)
                    AddRuntimeMessage(GH_RuntimeMessageLevel.Warning,
                        $"Face {b}: fewer than 3 vertices, skipped.");
                continue;
            }

            faces.Add(face);
            var pts = new List<Point3d>(face.Count);
            foreach (int idx in face)
                pts.Add(graph.Nodes[idx].Value);
            polylines.Add(FaceGeometry.BuildFacePolyline(pts));
            normals.Add(FaceGeometry.ComputeNewellNormal(pts));
        }

        // Rebuild output tree
        var outTree = new GH_Structure<GH_Integer>();
        for (int f = 0; f < faces.Count; f++)
        {
            var path = new GH_Path(f);
            foreach (int idx in faces[f])
                outTree.Append(new GH_Integer(idx), path);
        }

        DA.SetDataTree(0, outTree);
        DA.SetDataList(1, polylines);
        DA.SetDataList(2, normals);
    }

    protected override Bitmap Icon => null!;
    public override Guid ComponentGuid => new Guid("B8E4C3D2-AE5F-4B90-C7D6-3F2E1A4B5C6D");
    public override GH_Exposure Exposure => GH_Exposure.hidden;
}
