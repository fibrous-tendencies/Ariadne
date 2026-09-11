namespace Ariadne.Tests;

using System.Collections.Generic;
using Ariadne.Solver;
using Grasshopper.Kernel.Data;
using Grasshopper.Kernel.Types;
using Xunit;

public sealed class QTreeMapperTests
{
    [Fact]
    public void OneValueBroadcastsAcrossAllEdgeBranches()
    {
        var result = QTreeMapper.Map(
            Tree((Path(7), new[] { 2.5 })),
            [Path(0), Path(0), Path(1)]);

        Assert.True(result.Success);
        Assert.Equal([2.5, 2.5, 2.5], result.Values);
        Assert.Null(result.Warning);
    }

    [Fact]
    public void MatchingFlatValuesMapPerEdgeWithoutWarningForSingleEdgeBranch()
    {
        var result = QTreeMapper.Map(
            Tree((Path(0), new[] { 1.0, 2.0, 3.0 })),
            [Path(0), Path(0), Path(0)]);

        Assert.True(result.Success);
        Assert.Equal([1.0, 2.0, 3.0], result.Values);
        Assert.Null(result.Warning);
    }

    [Fact]
    public void MatchingFlatValuesMapInEdgeOrderWithWarningForMultipleEdgeBranches()
    {
        var result = QTreeMapper.Map(
            Tree((Path(0), new[] { 1.0, 2.0, 3.0 })),
            [Path(2), Path(2), Path(4)]);

        Assert.True(result.Success);
        Assert.Equal([1.0, 2.0, 3.0], result.Values);
        Assert.Contains("flattened edge order", result.Warning);
    }

    [Fact]
    public void PathMatchingSupportsMixedBroadcastAndPerEdgeValues()
    {
        var result = QTreeMapper.Map(
            Tree(
                (Path(0, 1), new[] { 4.0 }),
                (Path(2, 3), new[] { 5.0, 6.0 })),
            [Path(0, 1), Path(0, 1), Path(2, 3), Path(2, 3)]);

        Assert.True(result.Success);
        Assert.Equal([4.0, 4.0, 5.0, 6.0], result.Values);
        Assert.Null(result.Warning);
    }

    [Fact]
    public void MissingQPathReturnsUsefulError()
    {
        var result = QTreeMapper.Map(
            Tree((Path(0), new[] { 1.0, 2.0 })),
            [Path(0), Path(0), Path(1)]);

        Assert.False(result.Success);
        Assert.Contains("{1}", result.Error);
        Assert.Contains("No q values were supplied", result.Error);
    }

    [Fact]
    public void ExtraQPathReturnsUsefulError()
    {
        var result = QTreeMapper.Map(
            Tree(
                (Path(0), new[] { 1.0, 2.0 }),
                (Path(1), new[] { 3.0 }),
                (Path(9), new[] { 4.0 })),
            [Path(0), Path(0), Path(1)]);

        Assert.False(result.Success);
        Assert.Contains("{9}", result.Error);
        Assert.Contains("does not match any edge branch", result.Error);
    }

    [Fact]
    public void BranchCountMismatchReturnsExpectedCounts()
    {
        var result = QTreeMapper.Map(
            Tree((Path(0, 1), new[] { 1.0, 2.0 })),
            [Path(0, 1), Path(0, 1), Path(0, 1)]);

        Assert.False(result.Success);
        Assert.Contains("{0;1}", result.Error);
        Assert.Contains("has 2 value(s)", result.Error);
        Assert.Contains("has 3 edge(s)", result.Error);
    }

    [Fact]
    public void EmptyQTreeReturnsError()
    {
        var result = QTreeMapper.Map(
            new GH_Structure<GH_Number>(),
            [Path(0)]);

        Assert.False(result.Success);
        Assert.Contains("cannot be empty", result.Error);
    }

    [Fact]
    public void BoundsUseSameMixedBroadcastAndPerEdgeDispatch()
    {
        var bounds = new EdgeValueTree(
        [
            new EdgeValueBranch("{0;1}", [4.0]),
            new EdgeValueBranch("{2;3}", [5.0, 6.0]),
        ]);

        var result = QTreeMapper.Map(
            bounds,
            [Path(0, 1), Path(0, 1), Path(2, 3), Path(2, 3)],
            "qMin");

        Assert.True(result.Success);
        Assert.Equal([4.0, 4.0, 5.0, 6.0], result.Values);
    }

    [Fact]
    public void LowerAndUpperBoundsCanUseIndependentTrees()
    {
        var lower = QTreeMapper.Map(
            EdgeValueTree.Broadcast(0.1),
            [Path(0), Path(0), Path(1)],
            "qMin");
        var upper = QTreeMapper.Map(
            new EdgeValueTree(
            [
                new EdgeValueBranch("{0}", [10.0]),
                new EdgeValueBranch("{1}", [20.0]),
            ]),
            [Path(0), Path(0), Path(1)],
            "qMax");

        Assert.Equal([0.1, 0.1, 0.1], lower.Values);
        Assert.Equal([10.0, 10.0, 20.0], upper.Values);
    }

    [Fact]
    public void BoundsErrorsUseTheirInputLabel()
    {
        var result = QTreeMapper.Map(
            new EdgeValueTree([new EdgeValueBranch("{0}", [1.0, 2.0])]),
            [Path(0), Path(0), Path(0)],
            "qMax");

        Assert.False(result.Success);
        Assert.Contains("qMax at {0}", result.Error);
        Assert.Contains("has 3 edge(s)", result.Error);
    }

    [Fact]
    public void IntegerTreesBroadcastAndMatchLikeNumberBounds()
    {
        var tree = new GH_Structure<GH_Integer>();
        tree.Append(new GH_Integer(1), Path(0, 1));
        tree.Append(new GH_Integer(-1), Path(2, 3));
        tree.Append(new GH_Integer(-1), Path(2, 3));

        var result = QTreeMapper.Map(
            tree,
            [Path(0, 1), Path(0, 1), Path(2, 3), Path(2, 3)],
            "Signs");

        Assert.True(result.Success);
        Assert.Equal([1.0, 1.0, -1.0, -1.0], result.Values);
    }

    private static GH_Path Path(params int[] indices) => new(indices);

    private static GH_Structure<GH_Number> Tree(
        params (GH_Path Path, IReadOnlyList<double> Values)[] branches)
    {
        var tree = new GH_Structure<GH_Number>();
        foreach (var (path, values) in branches)
        {
            foreach (double value in values)
                tree.Append(new GH_Number(value), path);
        }
        return tree;
    }
}
