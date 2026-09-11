using Ariadne.Solver.Components.Experimental;
using Xunit;

namespace Ariadne.Tests;

public sealed class InverseFdmUiStateTests
{
    [Fact]
    public void ClarabelIsTheDefaultDirectSelection()
    {
        Assert.Equal(ParticularMode.Clarabel, InverseFdmUiState.DefaultParticular);
        Assert.Equal(3, InverseFdmUiState.NativeParticularMethod(ParticularMode.Clarabel));
        Assert.Equal(
            ActiveInverseEngine.Clarabel,
            InverseFdmUiState.ResolveEngine(
                LinearAlgebraMode.Direct,
                InverseFdmUiState.DefaultParticular,
                hasEffectiveBounds: false));
    }

    [Fact]
    public void EffectiveBoundsMatchNativeFiniteBoxRules()
    {
        Assert.False(InverseFdmUiState.HasEffectiveBounds(
            [0, 0],
            [double.NegativeInfinity, double.NegativeInfinity],
            [double.PositiveInfinity, double.PositiveInfinity]));

        Assert.True(InverseFdmUiState.HasEffectiveBounds([0, 1], [], []));
        Assert.True(InverseFdmUiState.HasEffectiveBounds([], [-2.0], []));
        Assert.True(InverseFdmUiState.HasEffectiveBounds([], [], [3.0]));
    }

    [Theory]
    [InlineData(4, 0)]
    [InlineData(0, 1)]
    [InlineData(1, 2)]
    [InlineData(2, 3)]
    [InlineData(3, 4)]
    public void UnconstrainedDirectUsesSelectedEngine(int particularValue, int expectedValue)
    {
        var particular = (ParticularMode)particularValue;
        var expected = (ActiveInverseEngine)expectedValue;
        Assert.Equal(
            expected,
            InverseFdmUiState.ResolveEngine(
                LinearAlgebraMode.Direct,
                particular,
                hasEffectiveBounds: false));
    }

    [Fact]
    public void DirectConstraintsSelectClarabelAndRemovalRetainsIt()
    {
        ParticularMode selected = InverseFdmUiState.UpdateParticular(
            LinearAlgebraMode.Direct,
            ParticularMode.QrLeastSquares,
            hasEffectiveBounds: true);

        Assert.Equal(ParticularMode.Clarabel, selected);
        Assert.Equal(
            ActiveInverseEngine.Clarabel,
            InverseFdmUiState.ResolveEngine(
                LinearAlgebraMode.Direct,
                selected,
                hasEffectiveBounds: true));

        selected = InverseFdmUiState.UpdateParticular(
            LinearAlgebraMode.Direct,
            selected,
            hasEffectiveBounds: false);

        Assert.Equal(ParticularMode.Clarabel, selected);
        Assert.Equal(
            ActiveInverseEngine.Clarabel,
            InverseFdmUiState.ResolveEngine(
                LinearAlgebraMode.Direct,
                selected,
                hasEffectiveBounds: false));
    }

    [Fact]
    public void IterativeEngineIgnoresDirectSelection()
    {
        Assert.Equal(
            ActiveInverseEngine.Lsqr,
            InverseFdmUiState.ResolveEngine(
                LinearAlgebraMode.Iterative,
                ParticularMode.MoorePenrose,
                hasEffectiveBounds: false));
        Assert.Equal(
            ActiveInverseEngine.Spg,
            InverseFdmUiState.ResolveEngine(
                LinearAlgebraMode.Iterative,
                ParticularMode.MoorePenrose,
                hasEffectiveBounds: true));
        Assert.Equal(
            ParticularMode.MoorePenrose,
            InverseFdmUiState.UpdateParticular(
                LinearAlgebraMode.Iterative,
                ParticularMode.MoorePenrose,
                hasEffectiveBounds: true));
    }
}
