using System;
using Ariadne.Solver.Components.Experimental;
using Xunit;

namespace Ariadne.Tests;

public class NullspaceComponentUtilitiesTests
{
    [Fact]
    public void MixColumnsUsesRowMajorManagedBasis()
    {
        double[] result = NullspaceComponentUtilities.MixColumns(
            [10.0, 20.0],
            [
                1.0, 2.0,
                3.0, 4.0,
            ],
            2,
            [2.0, -1.0]);

        Assert.Equal([10.0, 22.0], result);
    }

    [Fact]
    public void DisplaceMapsDimensionMajorMechanismsToNodeMajorXyz()
    {
        double[] result = NullspaceComponentUtilities.Displace(
            [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            [
                1.0, 2.0,
                3.0, 4.0,
                5.0, 6.0,
                7.0, 8.0,
                9.0, 10.0,
                11.0, 12.0,
            ],
            2,
            [2.0, -1.0],
            0.5);

        Assert.Equal([10.0, 22.0, 34.0, 41.0, 53.0, 65.0], result);
    }

    [Fact]
    public void SignConeProjectionUsesSelfStressCoordinates()
    {
        bool feasible = NullspaceComponentUtilities.ProjectForcesToSignCone(
            [-1.0, 2.0],
            [1.0, 0.0],
            1,
            [1, 1],
            out double[] forces);

        Assert.True(feasible);
        Assert.True(forces[0] >= -1e-8);
        Assert.True(forces[1] >= 0.0);
    }

    [Fact]
    public void NearestFeasibleProjectionCanLeaveRequestedHandleSegment()
    {
        double[] initial = [1.0, 0.0];
        double[] requested = [2.0, 1.0];

        ConstrainedProjectionResult result =
            NullspaceComponentUtilities.ProjectNearestFeasible(
                initial,
                requested,
                [1.0, 1.0],
                values => ([values[0] * values[0] + values[1] * values[1] - 1.0], []),
                tolerance: 1e-10);

        Assert.True(result.Converged);
        Assert.True(result.Feasibility < 1e-9);
        Assert.True(result.ObjectiveDistance < Math.Sqrt(2.0));
        double segmentCross = (result.Values[0] - initial[0])
            * (requested[1] - initial[1])
            - (result.Values[1] - initial[1])
            * (requested[0] - initial[0]);
        Assert.True(Math.Abs(segmentCross) > 1e-3);
        Assert.InRange(result.Values[0], 0.893, 0.895);
        Assert.InRange(result.Values[1], 0.446, 0.448);
    }

    [Fact]
    public void ActiveSetProjectionEnforcesInequality()
    {
        ConstrainedProjectionResult result =
            NullspaceComponentUtilities.ProjectNearestFeasible(
                [0.0, 1.0],
                [-2.0, 0.2],
                [1.0, 1.0],
                values => (
                    [values[0] * values[0] + values[1] * values[1] - 1.0],
                    [values[0]]),
                tolerance: 1e-9);

        Assert.True(result.Converged);
        Assert.True(result.Feasibility < 1e-8);
        Assert.True(result.Values[0] >= -1e-8);
        Assert.True(result.Values[1] > 0.99);
    }
}
