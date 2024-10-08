﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Rhino.Geometry;
using Rhino.Collections;
using Grasshopper.Kernel.Data;
using Grasshopper.Kernel.Types;
using System.Text.Json;
using System.Text.Json.Serialization;
using static Ariadne.Utilities.UtilityFunctions;
using Grasshopper.Kernel;
using Eto.Forms;
using GH_IO.Types;

namespace Ariadne.Graphs
{
    internal class Graph
    {
        /// <summary>
        /// Tolerance for merging curves into a graph
        /// </summary>
        [JsonIgnore]
        public double Tolerance { get; set;}

        /// <summary>
        /// All nodes in the graph represented as 3d points.
        /// </summary>
        [JsonIgnore]
        public List<Node> Nodes { get; set; }

        /// <summary>
        /// All edges of the graph represented as curve geometry.
        /// </summary>
        [JsonIgnore]
        public List<Edge> Edges { get; set; }

        /// <summary>
        /// Grasshopper tree representation of the indices list.
        /// </summary>
        [JsonIgnore]
        public GH_Structure<GH_Number> IndicesTree { get; set; }

        /// <summary>
        /// Grasshopper tree representation of the adjacency list.
        /// </summary>
        [JsonIgnore]
        public GH_Structure<GH_Number> AdjacencyTree { get; set; }

        /// <summary>
        /// Empty constructor
        /// </summary>
        public Graph(){ }

        /// <summary>
        /// Geometric constructor
        /// </summary>
        /// <param name="_InputCurves"></param>
        /// <param name="_Tolerance"></param>
        public Graph(List<GH_Curve> _InputCurves, double _Tolerance)
        {
            ConstructGraph(_InputCurves, _Tolerance);
        }

        /// <summary>
        /// Copy of an existing network
        /// </summary>
        /// <param name="other"></param>
        public Graph(Graph other)
        {
            Tolerance = other.Tolerance;
            Nodes = other.Nodes;
            Edges = other.Edges;
            IndicesTree = other.IndicesTree;
            AdjacencyTree = other.AdjacencyTree;
        }

        /// <summary>
        /// Construct a graph representation of curve network
        /// </summary>
        /// <param name="edges"></param>
        /// 
        private void ConstructGraph(List<GH_Curve> curves, double tol)
        {
            Nodes = new List<Node>();
            Edges = new List<Edge>();

            foreach (GH_Curve curve in curves)
            {
                Point3d start = curve.Value.PointAtStart;
                Point3d end = curve.Value.PointAtEnd;

                Edge edge = new() {
                    Value = curve.Value};

                if (curve.IsReferencedGeometry)
                {
                    edge.ReferenceID = curve.ReferenceID;
                }
                
                // check if start and end points are within tolerance of existing nodes
                (bool startFound, int istart) = WithinTolerance(Nodes, start, tol);                

                // analyze starting point
                if (!startFound)
                {
                    Node nodeStart = new() { Value = start };
                    Nodes.Add(nodeStart);
                    edge.Start = nodeStart;
                }
                else
                {
                    if (istart != -1)
                    {
                        edge.Start = Nodes[istart];
                    }                    
                }

                (bool endFound, int iend) = WithinTolerance(Nodes, end, tol);

                // analyze end point
                if (!endFound)
                {
                    Node nodeEnd = new() { Value = end };
                    Nodes.Add(nodeEnd);
                    edge.End = nodeEnd;
                }
                else
                {
                    if (iend != -1)
                    {
                        edge.End = Nodes[iend];
                    }
                }

                edge.Start.Neighbors.Add(edge.End);
                edge.End.Neighbors.Add(edge.Start);

                Edges.Add(edge);
                
            }
        }

        //create tree structure representation of edge indices array
        public void EdgeIndicesToTree()
        {
            int counter = 0;
            IndicesTree = new GH_Structure<GH_Number>();

            foreach (Edge edge in Edges)
            {
                //Add start and end indices to a tree structure for output
                GH_Number ghStart = new(Nodes.IndexOf(edge.Start));
                GH_Number ghEnd = new(Nodes.IndexOf(edge.End));
                IndicesTree.Append(ghStart, new GH_Path(counter));
                IndicesTree.Append(ghEnd, new GH_Path(counter));
                counter++;
            }
        }

        public void AdjacencyListToTree()
        {
            int counter = 0;
            AdjacencyTree = new GH_Structure<GH_Number>();

            foreach (Node node in Nodes)
            {
                foreach (Node neighbor in node.Neighbors)
                {
                    GH_Number ID = new(Nodes.IndexOf(neighbor));
                    AdjacencyTree.Append(ID, new GH_Path(counter));
                }
                counter++;
            }

        }

        public Graph Duplicate()
        {
            return new Graph(this);
        }

        //public functions to get number of edges and number of nodes
        public int Ne
        {
            get
            {
                return Edges.Count;
            }
        }
        public int Nn
        {
            get
            {
                return Nodes.Count;
            }
        }
    }

    internal class Node : GH_Point
    {
        [JsonIgnore]
        public bool Anchor { get; set; }

        [JsonIgnore]
        public List<Node> Neighbors { get; set; }

        /// <summary>
        /// Default constructor
        /// </summary>
        public Node()
        {
            Anchor = false;
            Neighbors = new List<Node>();
            Value = new Point3d();
        }

        public Node(Node other)
        {
            Anchor = other.Anchor;
            Neighbors = other.Neighbors;
        }

        public Node(bool anchor, List<Node> neighbors)
        {
            Anchor = anchor;
            Neighbors = neighbors;
        }
    }



    internal class Edge : GH_Curve
    {
        [JsonIgnore]
        public Node Start { get; set; }

        [JsonIgnore]
        public Node End { get; set; }

        [JsonIgnore]
        public double Q { get; set; }

        public Edge()
        {
            Start = new();
            End = new();
            Q = 0;
        }

        public Edge(Node start, Node end, double q, GH_Curve curve)
        {
            Start = start;
            End = end;
            Q = q;
            Value = curve.Value;
        }

        public Edge(Edge other)
        {
            Start = other.Start;
            End = other.End;
            Q = other.Q;
            Value = other.Value;
        }
        
    }
}
