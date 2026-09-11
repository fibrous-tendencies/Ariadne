namespace Ariadne.Solver;

using System;
using System.Collections.Generic;
using System.Linq;
using Grasshopper.Kernel.Data;
using Grasshopper.Kernel.Types;

internal sealed record QTreeMappingResult(
    IReadOnlyList<double>? Values,
    string? Error,
    string? Warning)
{
    public bool Success => Error is null && Values is not null;
}

public sealed record EdgeValueBranch(string Path, IReadOnlyList<double> Values);

public sealed record EdgeValueTree(IReadOnlyList<EdgeValueBranch> Branches)
{
    public static EdgeValueTree Broadcast(double value) =>
        new([new EdgeValueBranch("{0}", [value])]);

    public IEnumerable<double> FlattenedValues =>
        Branches.SelectMany(branch => branch.Values);

    internal static EdgeValueTree FromGh(GH_Structure<GH_Number> tree)
    {
        ArgumentNullException.ThrowIfNull(tree);
        var branches = new List<EdgeValueBranch>(tree.PathCount);
        for (int branchIndex = 0; branchIndex < tree.PathCount; branchIndex++)
        {
            GH_Path path = tree.Paths[branchIndex];
            var values = NumericValues(tree.get_Branch(path));
            branches.Add(new EdgeValueBranch(path.ToString(), values));
        }
        return new EdgeValueTree(branches);
    }

    internal static EdgeValueTree FromGh(GH_Structure<GH_Integer> tree)
    {
        ArgumentNullException.ThrowIfNull(tree);
        var branches = new List<EdgeValueBranch>(tree.PathCount);
        for (int branchIndex = 0; branchIndex < tree.PathCount; branchIndex++)
        {
            GH_Path path = tree.Paths[branchIndex];
            var values = NumericValues(tree.get_Branch(path));
            branches.Add(new EdgeValueBranch(path.ToString(), values));
        }
        return new EdgeValueTree(branches);
    }

    private static double[] NumericValues(System.Collections.IList branch)
    {
        var values = new List<double>(branch.Count);
        foreach (var item in branch)
        {
            switch (item)
            {
                case GH_Number number:
                    values.Add(number.Value);
                    break;
                case GH_Integer integer:
                    values.Add(integer.Value);
                    break;
            }
        }
        return values.ToArray();
    }
}

internal static class QTreeMapper
{
    public static QTreeMappingResult Map(
        GH_Structure<GH_Number> qTree,
        IReadOnlyList<GH_Path> edgePaths)
    {
        ArgumentNullException.ThrowIfNull(qTree);
        return Map(EdgeValueTree.FromGh(qTree), edgePaths, "q");
    }

    public static QTreeMappingResult Map(
        GH_Structure<GH_Number> tree,
        IReadOnlyList<GH_Path> edgePaths,
        string label)
    {
        ArgumentNullException.ThrowIfNull(tree);
        return Map(EdgeValueTree.FromGh(tree), edgePaths, label);
    }

    public static QTreeMappingResult Map(
        GH_Structure<GH_Integer> tree,
        IReadOnlyList<GH_Path> edgePaths,
        string label)
    {
        ArgumentNullException.ThrowIfNull(tree);
        return Map(EdgeValueTree.FromGh(tree), edgePaths, label);
    }

    public static QTreeMappingResult Map(
        EdgeValueTree valueTree,
        IReadOnlyList<GH_Path> edgePaths,
        string label)
    {
        ArgumentNullException.ThrowIfNull(valueTree);
        ArgumentNullException.ThrowIfNull(edgePaths);
        ArgumentException.ThrowIfNullOrWhiteSpace(label);

        int edgeCount = edgePaths.Count;
        if (edgeCount == 0)
            return Failure($"The network has no edges, so {label} cannot be assigned.");

        int totalValueCount = valueTree.Branches.Sum(branch => branch.Values.Count);
        if (totalValueCount == 0)
            return Failure($"{label} cannot be empty.");

        if (totalValueCount == 1)
        {
            double value = valueTree.FlattenedValues.Single();
            return Success(Enumerable.Repeat(value, edgeCount).ToArray());
        }

        var edgeIndicesByPath = new Dictionary<string, List<int>>(StringComparer.Ordinal);
        for (int edgeIndex = 0; edgeIndex < edgePaths.Count; edgeIndex++)
        {
            string path = edgePaths[edgeIndex].ToString();
            if (!edgeIndicesByPath.TryGetValue(path, out var indices))
            {
                indices = [];
                edgeIndicesByPath.Add(path, indices);
            }
            indices.Add(edgeIndex);
        }

        if (valueTree.Branches.Count == 1 && totalValueCount == edgeCount)
        {
            string? warning = edgeIndicesByPath.Count > 1
                ? $"{label} has {edgeCount} values matching the total edge count; values were applied 1:1 " +
                  $"in flattened edge order, not per edge-tree branch. Graft/match {label} to the edge " +
                  "tree to assign per branch."
                : null;
            return Success(valueTree.Branches[0].Values.ToArray(), warning);
        }

        var valuesByPath = new Dictionary<string, IReadOnlyList<double>>(StringComparer.Ordinal);
        foreach (var branch in valueTree.Branches)
        {
            if (!valuesByPath.TryAdd(branch.Path, branch.Values))
                return Failure($"The {label} tree contains duplicate path {branch.Path}.");
        }

        foreach (var (path, edgeIndices) in edgeIndicesByPath)
        {
            if (!valuesByPath.ContainsKey(path))
            {
                return Failure(
                    $"No {label} values were supplied for edge branch {path}, which contains " +
                    $"{edgeIndices.Count} edge(s). Supply {label} values for every edge branch.");
            }
        }

        foreach (var (path, values) in valuesByPath)
        {
            if (values.Count > 0 && !edgeIndicesByPath.ContainsKey(path))
                return Failure($"{label} branch {path} does not match any edge branch.");
        }

        var result = new double[edgeCount];
        foreach (var (path, edgeIndices) in edgeIndicesByPath)
        {
            IReadOnlyList<double> values = valuesByPath[path];
            if (values.Count == 1)
            {
                foreach (int edgeIndex in edgeIndices)
                    result[edgeIndex] = values[0];
                continue;
            }

            if (values.Count != edgeIndices.Count)
            {
                return Failure(
                    $"{label} at {path} has {values.Count} value(s), but that branch has " +
                    $"{edgeIndices.Count} edge(s). Supply 1 value to broadcast or " +
                    $"{edgeIndices.Count} values to assign per edge.");
            }

            for (int itemIndex = 0; itemIndex < edgeIndices.Count; itemIndex++)
                result[edgeIndices[itemIndex]] = values[itemIndex];
        }

        return Success(result);
    }

    private static QTreeMappingResult Success(
        IReadOnlyList<double> values,
        string? warning = null) =>
        new(values, null, warning);

    private static QTreeMappingResult Failure(string error) =>
        new(null, error, null);
}
