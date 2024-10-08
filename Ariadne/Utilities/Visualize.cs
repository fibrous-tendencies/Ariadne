﻿using System;
using System.Collections.Generic;
using System.Linq;
using Grasshopper.Kernel;
using Rhino.Geometry;
using Grasshopper.Kernel.Parameters;
using Grasshopper.GUI.Gradient;
using System.Runtime.InteropServices;
using Grasshopper.Kernel.Types.Transforms;
using System.Data.SqlClient;
using Ariadne.FDM;
using Rhino.Collections;
using Ariadne.Graphs;
using System.Drawing;

namespace Ariadne.Utilities
{
    internal class Visualize : GH_Component
    {
        //persistent data
        Line[] edges;
        List<double> property;
        Line[] externalforces;
        Line[] reactionforces;
        System.Drawing.Color c0;
        System.Drawing.Color cmed;
        System.Drawing.Color c1;
        GH_Gradient grad;
        int thickness;
        System.Drawing.Color cload;
        System.Drawing.Color creact;
        bool load;
        bool react;
        int prop;
        bool show;
        FDM_Problem network;

        //default colours
        readonly System.Drawing.Color lightgray = System.Drawing.Color.FromArgb(230, 231, 232);
        readonly System.Drawing.Color blue = System.Drawing.Color.FromArgb(62, 168, 222);
        readonly System.Drawing.Color pink = System.Drawing.Color.FromArgb(255, 123, 172);
        readonly System.Drawing.Color green = System.Drawing.Color.FromArgb(71, 181, 116);
        readonly System.Drawing.Color red = System.Drawing.Color.FromArgb(150, 235, 52, 73);

        //double minprop;
        //double maxprop;


        /// <summary>
        /// Initializes a new instance of the Visualize class.
        /// </summary>
        public Visualize()
          : base("VisualizeNetwork", "Visualize",
              "Visualize a FDM network",
              "Ariadne", "Utilities")
        {
        }

        /// <summary>
        /// Registers all the input parameters for this component.
        /// </summary>
        protected override void RegisterInputParams(GH_InputParamManager pManager)
        {
            pManager.AddBooleanParameter("Show", "Show", "Active status of component", GH_ParamAccess.item, true);
            pManager.AddGenericParameter("Network", "Network", "Network to visualize", GH_ParamAccess.item);
            pManager.AddVectorParameter("Loads", "P", "Applied loads", GH_ParamAccess.list, new Vector3d(0, 0, 0));
            pManager.AddNumberParameter("Load Scale", "Pscale", "Scale factor for length of arrows", GH_ParamAccess.item, 100);
            pManager.AddColourParameter("ColourMin", "Cmin", "Colour for minimum value", GH_ParamAccess.item, pink);
            pManager.AddColourParameter("ColourMed", "Cmed", "Colour for neutral value", GH_ParamAccess.item, lightgray);
            pManager.AddColourParameter("ColourMax", "Cmax", "Colour for maximum value",
                GH_ParamAccess.item, blue);
            pManager.AddIntegerParameter("Color Property", "Property", "Property displayed by colour gradient", GH_ParamAccess.item, 0);
            pManager.AddIntegerParameter("Line Thickness", "Thickness", "Thickness of preview lines", GH_ParamAccess.item, 8);
            pManager.AddColourParameter("Load Colour", "Cload", "Colour for applied loads", GH_ParamAccess.item, red);
            pManager.AddBooleanParameter("Show Loads", "Load", "Show external loads in preview", GH_ParamAccess.item, true);
            pManager.AddColourParameter("Reaction Colour", "Creaction", "Colour for support reactions", GH_ParamAccess.item, green);
            pManager.AddBooleanParameter("Show Reactions", "Reaction", "Show anchor reactions in preview", GH_ParamAccess.item, true);

            Param_Integer param = pManager[7] as Param_Integer;
            param.AddNamedValue("None", -1);
            param.AddNamedValue("Force", 0);
            param.AddNamedValue("Q", 1);

        }

        /// <summary>
        /// Registers all the output parameters for this component.
        /// </summary>
        protected override void RegisterOutputParams(GH_OutputParamManager pManager)
        {
        }

        /// <summary>
        /// This is the method that actually does the work.
        /// </summary>
        /// <param name="DA">The DA object is used to retrieve from inputs and store in outputs.</param>
        protected override void SolveInstance(IGH_DataAccess DA)
        {
            ClearData();
            //Initialize
            FDM_Network network = new();
            List<Vector3d> loads = new List<Vector3d>();
            double scale = 1.0;
            c0 = pink; // min colour (light gray)
            cmed = lightgray;
            c1 = blue; // max colour (blue)
            cload = pink; // load colour (pink)
            load = true;
            creact = green; // reaction colour green)
            react = false;

            //Assign
            DA.GetData(0, ref show);
            if (!DA.GetData(1, ref network)) return;
            DA.GetDataList(2, loads);
            DA.GetData(3, ref scale);
            DA.GetData(4, ref c0);
            DA.GetData(5, ref cmed);
            DA.GetData(6, ref c1);
            DA.GetData(7, ref prop);
            DA.GetData(8, ref thickness);
            DA.GetData(9, ref cload);
            DA.GetData(10, ref load);
            DA.GetData(11, ref creact);
            DA.GetData(12, ref react);

            List<int> freeIndices = network.FreeNodes;
            List<int> fixedIndices = network.FixedNodes;

            //Lines and forces
            externalforces = LoadMaker(network.Graph.Nodes, freeIndices , loads, scale);
            //edges = network.Network.Graph.Edges.Select(edge => edge.Curve);
            //reactionforces = ReactionMaker(Solver.Reactions(network), network.Points, network.Network.F, scale);

            //element-wise values
            if (prop == 0)
            {
                List<double> length = UtilityFunctions.GetLengths(network.Graph.Edges);
                property = UtilityFunctions.GetForces(length, network.Graph.Edges);
                GradientMaker(property);
                //var propabs = property.Select(x => Math.Abs(x)).ToList();

                //SetGradient(propabs.Max());
            }
            else if (prop == 1)
            {
                property = network.Graph.Edges.Select(x => x.Q).ToList();
                GradientMaker(property);

                //var propabs = property.Select(x => Math.Abs(x)).ToList();
                //SetGradient(propabs.Max());
            }

        }

        public override BoundingBox ClippingBox
        {
            get
            {
                BoundingBox bb = new BoundingBox(network.Network.Graph.Nodes.Select(node => node.Value));
                for (int i = 0; i < externalforces.Length; i++) bb.Union(externalforces[i].BoundingBox);
                for (int i = 0; i < reactionforces.Length; i++) bb.Union(reactionforces[i].BoundingBox);

                return bb;
            }
        }

        public override void DrawViewportWires(IGH_PreviewArgs args)
        {
            base.DrawViewportWires(args);

            if (show)
            {
                if (load) args.Display.DrawArrows(externalforces, cload);

                if (react) args.Display.DrawArrows(reactionforces, creact);

                if (prop == -1) args.Display.DrawLines(edges, c1, thickness);
                else
                {
                    for (int i = 0; i < edges.Length; i++)
                    {
                        args.Display.DrawLine(edges[i], grad.ColourAt(property[i]), thickness);
                    }
                }
            }
        }

        public Line[] ToLines(CurveList curves)
        {
            Line[] lines = new Line[curves.Count];
            for (int i = 0; i < curves.Count; i++)
            {
                Curve curve = curves[i];
                Line line = new Line(curve.PointAtStart, curve.PointAtEnd);
                lines[i] = line;
            }

            return lines;
        }

        public void GradientMaker(List<double> property)
        {
            double minprop = property.Min();
            double maxprop = property.Max();

            int signmin = Math.Sign(minprop);
            int signmax = Math.Sign(maxprop);

            //all data is negative
            if (signmin <= 0 && signmax <= 0)
            {
                grad = new GH_Gradient();
                grad.AddGrip(minprop, c0);
                grad.AddGrip(0, cmed);
            }
            //negative and positive values
            else if (signmin <= 0 && signmax >= 0)
            {
                grad = new GH_Gradient();
                grad.AddGrip(minprop, c0);
                grad.AddGrip(0, cmed);
                grad.AddGrip(maxprop, c1);
            }
            //all positive
            else
            {
                grad = new GH_Gradient();
                grad.AddGrip(0, cmed);
                grad.AddGrip(maxprop, c1);
            }
        }

        public void SetGradient(double max)
        {
            grad = new GH_Gradient();
            grad.AddGrip(-max, c0);
            grad.AddGrip(0, cmed);
            grad.AddGrip(max, c1);
        }

        public Line[] ReactionMaker(List<Vector3d> anchorforces, List<Point3d> points, List<int> F, double scale)
        {
            var mags = anchorforces.Select(p => p.Length).ToList();
            var normalizer = mags.Max();

            List<Line> reactions = new List<Line>();

            for (int i = 0; i < F.Count; i++)
            {
                var index = F[i];
                reactions.Add(new Line(points[index], anchorforces[i] * 3 * scale / normalizer));
            }

            return reactions.ToArray();
        }
        public Line[] LoadMaker(List<Node> nodes, List<int> N, List<Vector3d> loads, double scale)
        {
            List<Line> loadvectors = new List<Line>();

            if (N.Count != loads.Count && loads.Count != 1) throw new ArgumentException("Length of force vectors must be 1 or match length of free nodes.");

            if (loads.Count == 1)
            {
                for (int i = 0; i < N.Count; i++)
                {
                    int index = N[i];

                    Point3d p = nodes[index].Value;

                    loadvectors.Add(new Line(p, loads[0] / loads[0].Length * scale));
                }
            }
            else
            {
                //extract magnitudes
                var lns = loads.Select(p => p.Length).ToList();
                var normalizer = lns.Max();

                for (int i = 0; i < N.Count; i++)
                {
                    int index = N[i];
                    Point3d p = nodes[index].Value;
                    Vector3d l = loads[i];

                    if (l.Length < 0.1)
                    {
                        continue;
                    }
                    else
                    {
                        loadvectors.Add(new Line(p, l * scale / normalizer));
                    }

                }
            }

            return loadvectors.ToArray();
        }

        /// <summary>
        /// Provides an Icon for the component.
        /// </summary>
        protected override Bitmap Icon => Properties.Resources.visualize;

        /// <summary>
        /// Gets the unique ID for this component. Do not change this ID after release.
        /// </summary>
        public override Guid ComponentGuid
        {
            get { return new Guid("5B9176B3-B940-4C2C-AFFE-BF4532FB2111"); }
        }
    }
}