﻿using System;
using System.Collections.Generic;

using Grasshopper.Kernel;
using Rhino.Geometry;
using Ariadne.Objectives;
using System.Drawing;

namespace Ariadne.Objectives
{
    public class ObjectivePerformance : GH_Component
    {
        /// <summary>
        /// Initializes a new instance of the ObjectivePerformance class.
        /// </summary>
        public ObjectivePerformance()
          : base("Structural Performance", "FL",
              "Minimize sum(Force x Length)",
              "Ariadne", "ObjectiveFunctions")
        {
        }

        /// <summary>
        /// Registers all the input parameters for this component.
        /// </summary>
        protected override void RegisterInputParams(GH_InputParamManager pManager)
        {
            pManager.AddNumberParameter("Weight", "W", "Weight of objective", GH_ParamAccess.item, 1.0);
        }

        /// <summary>
        /// Registers all the output parameters for this component.
        /// </summary>
        protected override void RegisterOutputParams(GH_OutputParamManager pManager)
        {
            pManager.AddGenericParameter("Performance Objective Fuction", "OBJ", "Performance Objective Function", GH_ParamAccess.item);
        }

        /// <summary>
        /// This is the method that actually does the work.
        /// </summary>
        /// <param name="DA">The DA object is used to retrieve from inputs and store in outputs.</param>
        protected override void SolveInstance(IGH_DataAccess DA)
        {
            double weight = 1.0;

            if (!DA.GetData(0, ref weight)) return;

            OBJPerformance obj = new OBJPerformance(weight);

            DA.SetData(0, obj);
        }

        /// <summary>
        /// Provides an Icon for the component.
        /// </summary>
        protected override Bitmap Icon
        {
            get
            {
                return Properties.Resources.performance;
            }
        }

        /// <summary>
        /// Gets the unique ID for this component. Do not change this ID after release.
        /// </summary>
        public override Guid ComponentGuid
        {
            get { return new Guid("D11889BC-9F3F-4CD7-9767-2C771B1E9DF9"); }
        }
    }
}