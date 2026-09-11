namespace Ariadne.Tests;

using System;
using System.Collections.Generic;
using Ariadne.FDM;
using Ariadne.Graphs;
using Ariadne.Solver;
using Rhino.Geometry;
using Xunit;

public sealed class LoadNodeResolutionTests
{
    [Fact]
    public void ResolvesPointsWithinNetworkTolerance()
    {
        var network = Network(
            tolerance: 0.01,
            new Point3d(0, 0, 0),
            new Point3d(1, 0, 0));

        var indices = TheseusSolverService.ResolveLoadNodeIndices(
            network,
            [new Point3d(1.005, 0, 0)]);

        Assert.Equal([1], indices);
    }

    [Fact]
    public void ResolvesEachPointToNearestFreeNode()
    {
        var network = Network(
            tolerance: 0.25,
            new Point3d(0, 0, 0),
            new Point3d(1, 0, 0),
            new Point3d(2, 0, 0));

        var indices = TheseusSolverService.ResolveLoadNodeIndices(
            network,
            [new Point3d(1.1, 0, 0), new Point3d(0.05, 0, 0)]);

        Assert.Equal([1, 0], indices);
    }

    [Fact]
    public void RejectsPointOutsideNetworkTolerance()
    {
        var network = Network(
            tolerance: 0.01,
            new Point3d(0, 0, 0),
            new Point3d(1, 0, 0));

        var exception = Assert.Throws<ArgumentException>(
            () => TheseusSolverService.ResolveLoadNodeIndices(
                network,
                [new Point3d(1.02, 0, 0)]));

        Assert.Contains("does not match any free node", exception.Message);
        Assert.Contains("0.01", exception.Message);
    }

    [Fact]
    public void BroadcastsSingleLoadToSpecifiedNodesAndZerosOthers()
    {
        var load = new Vector3d(1, 2, 3);

        var packed = TheseusSolverService.PackFreeNodeLoads(4, [load], [1, 3]);

        Assert.Equal(
            [Vector3d.Zero, load, Vector3d.Zero, load],
            packed);
    }

    [Fact]
    public void AppliesMatchingLoadsToSpecifiedNodesInOrder()
    {
        var first = new Vector3d(1, 0, 0);
        var second = new Vector3d(0, 2, 0);

        var packed = TheseusSolverService.PackFreeNodeLoads(
            3, [first, second], [2, 0]);

        Assert.Equal(
            [second, Vector3d.Zero, first],
            packed);
    }

    [Fact]
    public void RepeatsLastLoadAcrossAllNodesWhenNoLoadNodesAreSpecified()
    {
        var first = new Vector3d(1, 0, 0);
        var second = new Vector3d(0, 2, 0);

        var packed = TheseusSolverService.PackFreeNodeLoads(
            4, [first, second], null);

        Assert.Equal(
            [first, second, second, second],
            packed);
    }

    [Fact]
    public void RejectsMismatchedLoadAndLoadNodeCounts()
    {
        var exception = Assert.Throws<ArgumentException>(
            () => TheseusSolverService.PackFreeNodeLoads(
                4,
                [new Vector3d(1, 0, 0), new Vector3d(0, 1, 0)],
                [0, 1, 2]));

        Assert.Contains("provide either 1 load or exactly 3 loads", exception.Message);
    }

    private static FDM_Network Network(double tolerance, params Point3d[] points)
    {
        var nodes = new List<Node>(points.Length);
        for (int i = 0; i < points.Length; i++)
            nodes.Add(new Node { Index = i, Value = points[i] });

        return new FDM_Network
        {
            Graph = new Graph
            {
                Nodes = nodes,
                Edges = [],
                EdgeInputMap = [],
                Tolerance = tolerance,
            },
            Anchors = [],
            ATol = tolerance,
            ETol = tolerance,
            Free = nodes,
            Fixed = [],
            FreeNodes = [.. nodes.ConvertAll(node => node.Index)],
            FixedNodes = [],
            Valid = true,
        };
    }
}
