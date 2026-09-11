using System;
using System.Collections.Generic;

using Grasshopper.Kernel;
using Rhino.Collections;
using Rhino.Geometry;
using Rhino.Geometry.Intersect;
using Ariadne.Utilities;
using Ariadne.Properties;
using Rhino.Display;
using Ariadne.FDM;
using System.Drawing;
using Ariadne.Graphs;

namespace Ariadne.GH_Design.Experimental
{
    /// <summary>
    /// Experimental: constrains or visualizes network points relative to a control surface (e.g. NURBS).
    /// </summary>
    public class CtrlSurfPoint : GH_Component
    {
        private FDM_Network network = null!;
        private List<Point3d> inputPoints = null!;
        private NurbsSurface surf = null!;
        private NurbsSurface offsetSurf = null!;
        private List<Point3d> points = null!;
        private List<Point3d> offsetPoints = null!;
        private List<string> names = null!;
        private List<double> zvalues = null!;
        private List<double> values = null!;
        private BoundingBox bb;
        private double height;
        private double baseline;
        private GH_Document ghd = null!;
        private int ctrlidx = 7;
        private int u;
        private int v;
        private double vmax;
        private double vmin;

        private bool show;
        private int scale;
        private Point3d pbl;
        private Point3d pbr;
        private Point3d ptl;

        /// <summary>
        /// Initializes a new instance of the CtrlSurf class.
        /// </summary>
        public CtrlSurfPoint()
          : base("ControlSurfaceP", "CtrlSurfP",
              "Provides reduced-dimension values for free nodes based on NURBs control surface",
              "Ariadne", "Experimental")
        {
        }

        /// <summary>
        /// Registers all the input parameters for this component.
        /// </summary>
        protected override void RegisterInputParams(GH_Component.GH_InputParamManager pManager)
        {
            //pManager.AddGeometryParameter("Geometry", "Geo", "Geometry to reference", GH_ParamAccess.list);
            pManager.AddBooleanParameter("Generate", "Generate", "Generate the control surface", GH_ParamAccess.item, false);
            pManager.AddGenericParameter("Network", "Network", "Network to analyze", GH_ParamAccess.item);
            pManager.AddIntegerParameter("Ucount", "nU", "Number of points in u direction", GH_ParamAccess.item, 3);
            pManager.AddIntegerParameter("Vcount", "nV", "Number of points in v direction", GH_ParamAccess.item, 3);
            pManager.AddVectorParameter("SurfaceOffset", "Offset", "Offset of displayed control surface (independent of actual value calculation)", GH_ParamAccess.item, new Vector3d(0, 0, -150));
            pManager.AddNumberParameter("MaximumValue", "Max", "Maximum value represented by surface", GH_ParamAccess.item, 100);
            pManager.AddNumberParameter("MinimumValue", "Min", "Minimum value represented by surface",
                GH_ParamAccess.item, -100);
            pManager.AddNumberParameter("CtrlValue", "Value", "Surface control point values", GH_ParamAccess.list, 0);
            pManager.AddBooleanParameter("ShowSurface", "Show", "Show the control surface", GH_ParamAccess.item, true);
            pManager.AddIntegerParameter("TextScale", "TextScale", "Scale of text tags", GH_ParamAccess.item, 20);
        }

        /// <summary>
        /// Registers all the output parameters for this component.
        /// </summary>
        protected override void RegisterOutputParams(GH_Component.GH_OutputParamManager pManager)
        {
            pManager.AddSurfaceParameter("Surface", "Control Surface", "Output control surface", GH_ParamAccess.item);
            pManager.AddNumberParameter("Values", "Vals", "Output values", GH_ParamAccess.list);

            pManager.HideParameter(0);
        }

        /// <summary>
        /// This is the method that actually does the work.
        /// </summary>
        /// <param name="DA">The DA object is used to retrieve from inputs and store in outputs.</param>
        protected override void SolveInstance(IGH_DataAccess DA)
        {
            //Initialize
            inputPoints = new List<Point3d>();
            network = new FDM_Network();
            u = 3;
            v = 3;
            Vector3d offset = new Vector3d(0, 0, -100);
            vmax = 1e3;
            vmin = -1e3;
            bool reset = false;
            show = true;
            scale = 20;

            //assign
            DA.GetData(0, ref reset);
            if (!DA.GetData(1, ref network)) return;
            DA.GetData(2, ref u);
            DA.GetData(3, ref v);
            DA.GetData(8, ref show);
            DA.GetData(9, ref scale);

            //upper limit for density of control points
            if (u > 5 && v > 5)
            {
                AddRuntimeMessage(GH_RuntimeMessageLevel.Warning, "Number of control points is exceeding human-usable limits");
            }

            DA.GetData(4, ref offset);
            DA.GetData(5, ref vmax);
            DA.GetData(6, ref vmin);

            //active doc
            ghd = this.OnPingDocument();

            //get points
            GetFreePoints();

            //get global bounding box
            GetBB();

            //create sliders
            if (reset)
            {
                ghd.ScheduleSolution(5, SolutionCallback);
            }

            zvalues = new List<double>();
            DA.GetDataList(7, zvalues);

            //create control points
            GetPoints();

            //generate surface
            surf = NurbsSurface.CreateFromPoints(points, u, v, 3, 3);

            //interpolate values
            GetValues();

            //visualized nurbs surface
            GetOffsets(offset);

            DA.SetData(0, offsetSurf);
            DA.SetDataList(1, values);
        }

        public void GetFreePoints()
        {
            inputPoints = new List<Point3d>();

            foreach (Node node in network.Free)
            {
                inputPoints.Add(new Point3d(node.Value));
            }
        }

        //public override void DrawViewportMeshes(IGH_PreviewArgs args)
        //{
        //    base.DrawViewportMeshes(args);

        //}


        private void GetOffsets(Vector3d offset)
        {
            offsetSurf = (NurbsSurface)surf.Duplicate();
            offsetSurf.Translate(offset);

            offsetPoints = new List<Point3d>();

            foreach (Point3d point in points)
            {
                var offsetpoint = new Point3d(point);

                offsetPoints.Add(offsetpoint + offset);
            }
        }


        private void SolutionCallback(GH_Document gdoc)
        {
            MakeSliders();
        }

        /// <summary>
        /// create sliders
        /// </summary>
        private void MakeSliders()
        {
            if (this.Params.Input[ctrlidx].SourceCount > 0)
            {
                List<IGH_Param> sources = new List<IGH_Param>(this.Params.Input[ctrlidx].Sources);
                ghd.RemoveObjects(sources, false);
            }

            int nsliders = u * v;

            GetNames();

            //instantiate new slider
            for (int i = 0; i < nsliders; i++)
            {
                Grasshopper.Kernel.Special.GH_NumberSlider slider = new Grasshopper.Kernel.Special.GH_NumberSlider();

                slider.CreateAttributes();

                int inputcount = this.Params.Input[ctrlidx].SourceCount;
                var xpos = (float)this.Attributes.DocObject.Attributes.Bounds.Left - slider.Attributes.Bounds.Width - 30;
                var ypos = (float)this.Params.Input[ctrlidx].Attributes.Bounds.Y + inputcount * 30;

                slider.Attributes.Pivot = new System.Drawing.PointF(xpos, ypos);
                slider.Slider.Maximum = 1;
                slider.Slider.Minimum = 0;
                slider.Slider.DecimalPlaces = 2;
                slider.Slider.Value = (decimal)0.5;
                slider.NickName = names[i];

                ghd.AddObject(slider, false);

                this.Params.Input[ctrlidx].AddSource(slider);

            }
        }

        /// <summary>
        /// get bounding box
        /// </summary>
        private void GetBB()
        {
            var pointlist = new Point3dList(inputPoints);
            bb = pointlist.BoundingBox;

            

            //get total height of geometry

            pbl = bb.Corner(true, true, true);
            pbr = bb.Corner(false, true, true);
            ptl = bb.Corner(true, false, true);
            var pzl = bb.Corner(true, true, false);

            height = 1.25 * pbl.DistanceTo(pzl);
            baseline = pbl.Z - height;
        }

        /// <summary>
        /// Extract nicknames for sliders
        /// </summary>
        private void GetNames()
        {
            names = new List<string>();
            for (int i = 0; i < u; i++)
            {
                for (int j = 0; j < v; j++)
                {
                    string name = "[" + i.ToString() + ", " + j.ToString() + "]";
                    names.Add(name);
                }
            }
        }

        /// <summary>
        /// Generate point grid
        /// </summary>
        private void GetPoints()
        {
            points = new List<Point3d>();
            //extract spans
            double x = pbl.DistanceTo(pbr);
            double y = pbl.DistanceTo(ptl);

            //spacings
            double xspacing = x / (u - 1);
            double yspacing = y / (v - 1);

            int k = 0;

            for (int i = 0; i < u; i++)
            {
                for (int j = 0; j < v; j++)
                {
                    double px = pbl.X + xspacing * i;
                    double py = pbl.Y + yspacing * j;
                    double pz;

                    if (zvalues.Count == 1)
                    {
                        pz = height * zvalues[0] + baseline;
                    }
                    else
                    {
                        pz = height * zvalues[k] + baseline;
                    }


                    points.Add(new Point3d(px, py, pz));
                    k++;
                }
            }

        }

        private void GetValues()
        {
            values = new List<double>();
            double range = vmax - vmin;
            foreach (Point3d point in inputPoints)
            {
                Vector3d ray = -2 * height * Vector3d.ZAxis;
                Line line = new Line(point, ray);

                var inter = Intersection.CurveSurface(line.ToNurbsCurve(), surf, 1e-2, 1e-2)[0];

                Point3d interpoint = inter.PointB;

                double val = vmin + (interpoint.Z - baseline) / height * range;

                values.Add(val);
            }
        }

        /// <summary>
        /// Provides an Icon for the component.
        /// </summary>
        protected override Bitmap Icon => Resources.SurfP;

        /// <summary>
        /// Gets the unique ID for this component. Do not change this ID after release.
        /// </summary>
        public override Guid ComponentGuid
        {
            get { return new Guid("5C6A252F-A943-46A1-B609-7E80B9DDAD73"); }
        }

        public override GH_Exposure Exposure => GH_Exposure.hidden;
    }
}