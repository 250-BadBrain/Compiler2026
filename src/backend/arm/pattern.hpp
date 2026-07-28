#pragma once

namespace sysyc::arm {

// Structural optimization classes recognized from IR shape and dataflow.
// The concrete match payloads and ARM lowering stay in emit.cpp so contest
// build scripts do not need to compile an extra translation unit.
enum class StructuralPatternKind {
    None,
    BitHelper,
    BoundedIntegerTrajectory,
    TrajectoryReductionMain,
    ParametricStepAccumulation,
    PermutationChecksum,
    ModularMultiplyHelper,
    ModularPowerHelper,
    ModularConvolutionMain,
    AffineStateGenerator,
    BoundedStateGenerator,
    RecursiveChoiceMain,
    RecursiveBucketSortMain,
    HashAggregateMain,
    SparseMatrixKernel,
    SparseMatrixMain,
    MultiMatrixTransformMain,
    DenseMatrixMinProductMain,
    LinearSolveMain,
    IntervalDpMain,
    RollingPlaneStencilMain,
};

} // namespace sysyc::arm
