using System;
using System.Collections.Generic;
using System.Linq;
using Ariadne.FDM;
using Rhino.Geometry;
using Theseus.Interop;

namespace Ariadne.Solver.Components.Experimental;

internal sealed class ManagedNullspaceAnalysis
{
    public required int Key { get; init; }
    public required RigidityReport Report { get; init; }
    public required double[] Mechanisms { get; init; }
    public required double[] Eigenvalues { get; init; }
    public required MechanismClass[] Classes { get; init; }
    public required double[] ReferenceLengths { get; init; }
}

internal sealed class ConstrainedProjectionResult
{
    public required double[] Values { get; init; }
    public required double ObjectiveDistance { get; init; }
    public required double MaxEqualityResidual { get; init; }
    public required double MaxInequalityViolation { get; init; }
    public required int Iterations { get; init; }
    public required bool Converged { get; init; }
    public double Feasibility => Math.Max(MaxEqualityResidual, MaxInequalityViolation);
}

internal static class NullspaceComponentUtilities
{
    public static double[] PackPoints(IReadOnlyList<Point3d> points)
    {
        var values = new double[points.Count * 3];
        for (int i = 0; i < points.Count; i++)
        {
            values[i * 3] = points[i].X;
            values[i * 3 + 1] = points[i].Y;
            values[i * 3 + 2] = points[i].Z;
        }
        return values;
    }

    public static Point3d[] UnpackPoints(IReadOnlyList<double> values)
    {
        var points = new Point3d[values.Count / 3];
        for (int i = 0; i < points.Length; i++)
            points[i] = new Point3d(values[i * 3], values[i * 3 + 1], values[i * 3 + 2]);
        return points;
    }

    public static List<double> InitialQ(FDM_Network network)
    {
        var q = new List<double>(network.Graph.Ne);
        foreach (var edge in network.Graph.Edges)
            q.Add(double.IsFinite(edge.Q) ? edge.Q : 1.0);
        return q;
    }

    public static SolverInputs Inputs(FDM_Network network, List<Vector3d> loads) =>
        new() { QInit = InitialQ(network), Loads = loads };

    public static int AnalysisKey(
        FDM_Network network,
        IReadOnlyList<Point3d> target,
        IReadOnlyList<Vector3d> loads,
        int maxModes)
    {
        var hash = new HashCode();
        hash.Add(network.GetTopologyHashCode());
        hash.Add(maxModes);
        foreach (var point in target)
        {
            hash.Add(point.X); hash.Add(point.Y); hash.Add(point.Z);
        }
        foreach (var load in loads)
        {
            hash.Add(load.X); hash.Add(load.Y); hash.Add(load.Z);
        }
        return hash.ToHashCode();
    }

    public static ManagedNullspaceAnalysis Analyze(
        FDM_Network network,
        List<Point3d> target,
        List<Vector3d> loads,
        int maxModes,
        int key)
    {
        double[] xyz = PackPoints(target);
        SolverInputs inputs = Inputs(network, loads);
        RigidityReport report = TheseusSolverService.AnalyzeRigidity(
            network, inputs, xyz, RigidityMethod.Projector, false, maxModes);
        double[] referenceLengths = MemberLengths(network, target);
        double[] mechanisms = report.MechanismBasis;
        double[] eigenvalues = [];
        MechanismClass[] classes = [];
        int modeCount = report.MechanismModeCount;
        if (modeCount > 0)
        {
            var prestressForces = new double[network.Graph.Ne];
            for (int edge = 0; edge < prestressForces.Length; edge++)
                prestressForces[edge] = network.Graph.Edges[edge].Q * referenceLengths[edge];
            PrestressClassification classification = TheseusSolverService.ClassifyPrestress(
                network, inputs, xyz, prestressForces,
                report.MechanismBasis, modeCount);
            mechanisms = classification.RotatedMechanisms;
            eigenvalues = classification.Eigenvalues;
            classes = classification.Classes;
        }
        return new ManagedNullspaceAnalysis
        {
            Key = key,
            Report = report,
            Mechanisms = mechanisms,
            Eigenvalues = eigenvalues,
            Classes = classes,
            ReferenceLengths = referenceLengths,
        };
    }

    public static double[] MixColumns(
        IReadOnlyList<double> origin,
        IReadOnlyList<double> basis,
        int columns,
        IReadOnlyList<double> coefficients,
        double scale = 1.0)
    {
        var result = new double[origin.Count];
        for (int row = 0; row < result.Length; row++)
        {
            double value = origin[row];
            for (int mode = 0; mode < columns; mode++)
            {
                double coefficient = mode < coefficients.Count ? coefficients[mode] : 0.0;
                value += scale * basis[row * columns + mode] * coefficient;
            }
            result[row] = value;
        }
        return result;
    }

    public static double[] Displace(
        IReadOnlyList<double> xyz,
        IReadOnlyList<double> basis,
        int columns,
        IReadOnlyList<double> coefficients,
        double scale)
    {
        int nodes = xyz.Count / 3;
        var result = new double[xyz.Count];
        for (int node = 0; node < nodes; node++)
        {
            for (int dimension = 0; dimension < 3; dimension++)
            {
                int xyzIndex = node * 3 + dimension;
                int basisRow = dimension * nodes + node;
                double value = xyz[xyzIndex];
                for (int mode = 0; mode < columns; mode++)
                {
                    double coefficient = mode < coefficients.Count ? coefficients[mode] : 0.0;
                    value += scale * basis[basisRow * columns + mode] * coefficient;
                }
                result[xyzIndex] = value;
            }
        }
        return result;
    }

    public static FDM_Network BuildNetwork(
        FDM_Network source,
        IReadOnlyList<double> freeXyz,
        IReadOnlyList<double> forces)
    {
        var result = new FDM_Network(source);
        for (int i = 0; i < result.FreeNodes.Count; i++)
            result.Graph.Nodes[result.FreeNodes[i]].Value =
                new Point3d(freeXyz[i * 3], freeXyz[i * 3 + 1], freeXyz[i * 3 + 2]);
        for (int edgeIndex = 0; edgeIndex < result.Graph.Ne; edgeIndex++)
        {
            var edge = result.Graph.Edges[edgeIndex];
            double length = edge.Start.Value.DistanceTo(edge.End.Value);
            edge.Q = length > 1e-14 && edgeIndex < forces.Count
                ? forces[edgeIndex] / length
                : 0.0;
            edge.Value = new LineCurve(edge.Start.Value, edge.End.Value);
        }
        result.Free = result.FreeNodes.ConvertAll(index => result.Graph.Nodes[index]);
        result.Fixed = result.FixedNodes.ConvertAll(index => result.Graph.Nodes[index]);
        result.Anchors = result.FixedNodes.ConvertAll(index => result.Graph.Nodes[index].Value);
        return result;
    }

    public static FDM_Network WithAnchors(
        FDM_Network source,
        IReadOnlyList<Point3d> anchors,
        double fraction)
    {
        Point3d[] interpolated = source.FixedNodes
            .Select((node, index) =>
            {
                Point3d start = source.Graph.Nodes[node].Value;
                return start + fraction * (anchors[index] - start);
            })
            .ToArray();
        return WithAnchors(source, interpolated);
    }

    public static FDM_Network WithAnchors(
        FDM_Network source,
        IReadOnlyList<Point3d> anchors)
    {
        var result = new FDM_Network(source);
        for (int i = 0; i < result.FixedNodes.Count && i < anchors.Count; i++)
        {
            int node = result.FixedNodes[i];
            result.Graph.Nodes[node].Value = anchors[i];
        }
        result.Fixed = result.FixedNodes.ConvertAll(index => result.Graph.Nodes[index]);
        result.Anchors = result.FixedNodes.ConvertAll(index => result.Graph.Nodes[index].Value);
        foreach (var edge in result.Graph.Edges)
            edge.Value = new LineCurve(edge.Start.Value, edge.End.Value);
        return result;
    }

    public static double[] MemberLengths(FDM_Network network, IReadOnlyList<Point3d> freePoints)
    {
        var positions = new Point3d[network.Graph.Nn];
        for (int i = 0; i < network.FreeNodes.Count; i++)
            positions[network.FreeNodes[i]] = freePoints[i];
        for (int i = 0; i < network.FixedNodes.Count; i++)
            positions[network.FixedNodes[i]] = network.Graph.Nodes[network.FixedNodes[i]].Value;
        var lengths = new double[network.Graph.Ne];
        for (int i = 0; i < lengths.Length; i++)
        {
            var edge = network.Graph.Edges[i];
            lengths[i] = positions[edge.Start.Index].DistanceTo(positions[edge.End.Index]);
        }
        return lengths;
    }

    public static bool ProjectForcesToSignCone(
        IReadOnlyList<double> particular,
        IReadOnlyList<double> basis,
        int columns,
        IReadOnlyList<int> signs,
        out double[] forces)
    {
        var alpha = new double[columns];
        double normSq = 1e-12;
        for (int i = 0; i < basis.Count; i++) normSq += basis[i] * basis[i];
        double step = 0.8 / normSq;
        forces = [.. particular];
        for (int iteration = 0; iteration < 500; iteration++)
        {
            forces = MixColumns(particular, basis, columns, alpha);
            var gradient = new double[columns];
            double worst = 0.0;
            for (int row = 0; row < forces.Length; row++)
            {
                int sign = row < signs.Count ? signs[row] : 0;
                double violation = sign == 0 ? 0.0 : Math.Min(0.0, sign * forces[row]);
                worst = Math.Max(worst, -violation);
                if (violation == 0.0) continue;
                for (int mode = 0; mode < columns; mode++)
                    gradient[mode] += sign * basis[row * columns + mode] * violation;
            }
            if (worst <= 1e-9) return true;
            for (int mode = 0; mode < columns; mode++)
                alpha[mode] -= step * gradient[mode];
        }
        return false;
    }

    /// <summary>
    /// Sequential Gauss-Newton/active-set projection onto nonlinear equality
    /// constraints and inequalities represented as g(x) >= 0.
    /// </summary>
    public static ConstrainedProjectionResult ProjectNearestFeasible(
        IReadOnlyList<double> initial,
        IReadOnlyList<double> desired,
        IReadOnlyList<double> weights,
        Func<double[], (double[] Equalities, double[] Inequalities)> constraints,
        int maxIterations = 40,
        double tolerance = 1e-8)
    {
        if (initial.Count != desired.Count || initial.Count != weights.Count)
            throw new ArgumentException("Projection vectors and weights must have equal lengths.");
        if (weights.Any(weight => !double.IsFinite(weight) || weight <= 0.0))
            throw new ArgumentException("Projection weights must be finite and positive.");

        double[] x = [.. initial];
        int iterations = 0;
        for (; iterations < maxIterations; iterations++)
        {
            (double[] equalities, double[] inequalities) = constraints(x);
            ValidateConstraints(equalities, inequalities);
            double maxEquality = MaxAbs(equalities);
            double maxViolation = inequalities
                .Select(value => Math.Max(0.0, -value))
                .DefaultIfEmpty(0.0)
                .Max();
            List<int> active = inequalities
                .Select((value, index) => (value, index))
                .Where(item => item.value <= Math.Max(10.0 * tolerance, 1e-7))
                .Select(item => item.index)
                .ToList();
            double[] constraintValues = [.. equalities, .. active.Select(index => inequalities[index])];
            double[,] jacobian = NumericalConstraintJacobian(
                x, equalities.Length, active, constraints);
            double[] towardDesired = desired.Select((value, index) => value - x[index]).ToArray();
            double[] multipliers = SolveConstraintMultipliers(
                jacobian, constraintValues, towardDesired, weights);
            var step = new double[x.Length];
            for (int variable = 0; variable < x.Length; variable++)
            {
                double correction = 0.0;
                for (int row = 0; row < constraintValues.Length; row++)
                    correction += jacobian[row, variable] * multipliers[row];
                step[variable] = towardDesired[variable] - correction / weights[variable];
            }

            double stepNorm = Math.Sqrt(step.Sum(value => value * value));
            if (maxEquality <= tolerance && maxViolation <= tolerance
                && stepNorm <= tolerance * (1.0 + Math.Sqrt(x.Sum(value => value * value))))
                return ProjectionResult(x, desired, weights, equalities, inequalities, iterations, true);

            double currentMerit = ProjectionMerit(
                x, desired, weights, equalities, inequalities);
            bool accepted = false;
            double scale = 1.0;
            while (scale >= 1.0 / 4096.0)
            {
                double[] trial = x.Select((value, index) => value + scale * step[index]).ToArray();
                (double[] trialEqualities, double[] trialInequalities) = constraints(trial);
                ValidateConstraints(trialEqualities, trialInequalities);
                double trialMerit = ProjectionMerit(
                    trial, desired, weights, trialEqualities, trialInequalities);
                if (trialMerit < currentMerit - 1e-12 || trialMerit <= tolerance * tolerance)
                {
                    x = trial;
                    accepted = true;
                    break;
                }
                scale *= 0.5;
            }
            if (!accepted)
            {
                bool feasible = maxEquality <= tolerance && maxViolation <= tolerance;
                return ProjectionResult(
                    x, desired, weights, equalities, inequalities, iterations, feasible);
            }
        }

        (double[] finalEqualities, double[] finalInequalities) = constraints(x);
        bool converged = MaxAbs(finalEqualities) <= tolerance
            && finalInequalities.All(value => value >= -tolerance);
        return ProjectionResult(
            x, desired, weights, finalEqualities, finalInequalities, iterations, converged);
    }

    private static double[,] NumericalConstraintJacobian(
        double[] x,
        int equalityCount,
        IReadOnlyList<int> activeInequalities,
        Func<double[], (double[] Equalities, double[] Inequalities)> constraints)
    {
        int rows = equalityCount + activeInequalities.Count;
        var jacobian = new double[rows, x.Length];
        for (int variable = 0; variable < x.Length; variable++)
        {
            double delta = 1e-6 * Math.Max(1.0, Math.Abs(x[variable]));
            double[] plus = [.. x];
            double[] minus = [.. x];
            plus[variable] += delta;
            minus[variable] -= delta;
            var plusValues = constraints(plus);
            var minusValues = constraints(minus);
            for (int row = 0; row < equalityCount; row++)
                jacobian[row, variable] =
                    (plusValues.Equalities[row] - minusValues.Equalities[row]) / (2.0 * delta);
            for (int active = 0; active < activeInequalities.Count; active++)
            {
                int index = activeInequalities[active];
                jacobian[equalityCount + active, variable] =
                    (plusValues.Inequalities[index] - minusValues.Inequalities[index])
                    / (2.0 * delta);
            }
        }
        return jacobian;
    }

    private static double[] SolveConstraintMultipliers(
        double[,] jacobian,
        IReadOnlyList<double> constraintValues,
        IReadOnlyList<double> towardDesired,
        IReadOnlyList<double> weights)
    {
        int rows = constraintValues.Count;
        if (rows == 0) return [];
        var matrix = new double[rows, rows];
        var rhs = new double[rows];
        for (int left = 0; left < rows; left++)
        {
            rhs[left] = constraintValues[left];
            for (int variable = 0; variable < towardDesired.Count; variable++)
                rhs[left] += jacobian[left, variable] * towardDesired[variable];
            for (int right = 0; right < rows; right++)
            {
                double value = 0.0;
                for (int variable = 0; variable < towardDesired.Count; variable++)
                    value += jacobian[left, variable] * jacobian[right, variable]
                        / weights[variable];
                matrix[left, right] = value;
            }
            matrix[left, left] += 1e-12;
        }
        return SolveDense(matrix, rhs);
    }

    private static double[] SolveDense(double[,] matrix, double[] rhs)
    {
        int n = rhs.Length;
        var augmented = new double[n, n + 1];
        for (int row = 0; row < n; row++)
        {
            for (int column = 0; column < n; column++)
                augmented[row, column] = matrix[row, column];
            augmented[row, n] = rhs[row];
        }
        for (int pivot = 0; pivot < n; pivot++)
        {
            int best = pivot;
            for (int row = pivot + 1; row < n; row++)
                if (Math.Abs(augmented[row, pivot]) > Math.Abs(augmented[best, pivot]))
                    best = row;
            if (Math.Abs(augmented[best, pivot]) < 1e-14) continue;
            if (best != pivot)
                for (int column = pivot; column <= n; column++)
                    (augmented[pivot, column], augmented[best, column]) =
                        (augmented[best, column], augmented[pivot, column]);
            double diagonal = augmented[pivot, pivot];
            for (int column = pivot; column <= n; column++)
                augmented[pivot, column] /= diagonal;
            for (int row = 0; row < n; row++)
            {
                if (row == pivot) continue;
                double factor = augmented[row, pivot];
                for (int column = pivot; column <= n; column++)
                    augmented[row, column] -= factor * augmented[pivot, column];
            }
        }
        return Enumerable.Range(0, n).Select(row => augmented[row, n]).ToArray();
    }

    private static double ProjectionMerit(
        IReadOnlyList<double> values,
        IReadOnlyList<double> desired,
        IReadOnlyList<double> weights,
        IReadOnlyList<double> equalities,
        IReadOnlyList<double> inequalities)
    {
        double objective = values
            .Select((value, index) => 0.5 * weights[index] * Math.Pow(value - desired[index], 2))
            .Sum();
        double violation = equalities.Sum(value => Math.Abs(value))
            + inequalities.Sum(value => Math.Max(0.0, -value));
        return objective + 1e4 * violation;
    }

    private static ConstrainedProjectionResult ProjectionResult(
        double[] values,
        IReadOnlyList<double> desired,
        IReadOnlyList<double> weights,
        IReadOnlyList<double> equalities,
        IReadOnlyList<double> inequalities,
        int iterations,
        bool converged) =>
        new()
        {
            Values = [.. values],
            ObjectiveDistance = Math.Sqrt(values
                .Select((value, index) => weights[index] * Math.Pow(value - desired[index], 2))
                .Sum()),
            MaxEqualityResidual = MaxAbs(equalities),
            MaxInequalityViolation = inequalities
                .Select(value => Math.Max(0.0, -value))
                .DefaultIfEmpty(0.0)
                .Max(),
            Iterations = iterations,
            Converged = converged,
        };

    private static double MaxAbs(IReadOnlyList<double> values) =>
        values.Select(Math.Abs).DefaultIfEmpty(0.0).Max();

    private static void ValidateConstraints(
        IReadOnlyList<double> equalities,
        IReadOnlyList<double> inequalities)
    {
        if (equalities.Any(value => !double.IsFinite(value))
            || inequalities.Any(value => !double.IsFinite(value)))
            throw new InvalidOperationException("Projection constraints produced non-finite values.");
    }
}
