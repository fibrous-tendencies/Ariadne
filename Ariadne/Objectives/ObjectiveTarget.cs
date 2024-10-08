﻿using System;
using System.Collections.Generic;

using Grasshopper.Kernel;
using Rhino.Geometry;
using Ariadne.Objectives;
using Ariadne.FDM;
using Rhino.DocObjects;
using System.Linq;
using Ariadne.Graphs;
using System.Drawing;

namespace Ariadne.Objectives
{
    public class ObjectiveTarget : GH_Component
    {
        /// <summary>
        /// Initializes a new instance of the ObjectiveTarget class.
        /// </summary>
        public ObjectiveTarget()
          : base("Target Geometry", "TargetGeo",
              "Minimize the deviation from starting nodal Values",
              "Ariadne", "ObjectiveFunctions")
        {
        }

        /// <summary>
        /// Registers all the input parameters for this component.
        /// </summary>
        protected override void RegisterInputParams(GH_InputParamManager pManager)
        {
            pManager.AddGenericParameter("Nodes", "Nodes", "Nodes to target", GH_ParamAccess.list);
            pManager.AddNumberParameter("Weight", "Weight", "Weight of this objective function", GH_ParamAccess.item, 1.0);

            pManager[0].Optional = true;
        }

        /// <summary>
        /// Registers all the output parameters for this component.
        /// </summary>
        protected override void RegisterOutputParams(GH_OutputParamManager pManager)
        {
            pManager.AddGenericParameter("Target Geometry Objective Function", "OBJ", "Target Geometry Objective Function", GH_ParamAccess.item);
        }

        /// <summary>
        /// This is the method that actually does the work.
        /// </summary>
        /// <param name="DA">The DA object is used to retrieve from inputs and store in outputs.</param>
        protected override void SolveInstance(IGH_DataAccess DA)
        {
            List<Node> nodes = new();
            double weight = 1.0;

                
            DA.GetDataList(0, nodes);
            if (!DA.GetData(1, ref weight)) return;


            if (nodes.Count > 0)
            {                    
                OBJTarget obj = new OBJTarget(weight, nodes);
                DA.SetData(0, obj);
            }
            else 
            {
                OBJTarget obj = new OBJTarget(weight);
                DA.SetData(0, obj);
            }
        }

        /// <summary>
        /// Provides an Icon for the component.
        /// </summary>
        protected override Bitmap Icon => Properties.Resources.Target;

        /// <summary>
        /// Gets the unique ID for this component. Do not change this ID after release.
        /// </summary>
        public override Guid ComponentGuid
        {
            get { return new Guid("6AA55256-288F-49A6-9648-5314691BB659"); }
        }
    }
}