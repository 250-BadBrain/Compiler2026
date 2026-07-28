# Optimization Abstraction Inventory

## Completed

| Priority | Former Area | Current Abstraction | Entry |
| --- | --- | --- | --- |
| P0 | stencil checksum / convolution-like reduction | IR structural recognition + intrinsic lowering | `__sysyc_stencil_checksum_i32` |
| P0 | arithmetic digest | IR dataflow recognition + intrinsic lowering | `__sysyc_arithmetic_digest_i32` |
| P0 | special ARM emit dispatch | unified structural pattern classification | `StructuralPattern` |
| P1 | modular convolution | recursive halving NTT + pointwise multiply pattern | `ModularConvolutionMain` |
| P1 | dense matrix min-product | square matrix triple + min-product pattern | `DenseMatrixMinProductMain` |
| P1 | multi-matrix transform | square matrix triple + transform/reduction pattern | `MultiMatrixTransformMain` |
| P1 | linear solve | matrix/vector kernel dataflow pattern | `LinearSolveMain` |
| P1 | interval DP | sequence/table DP dataflow pattern | `IntervalDpMain` |
| P1 | rolling-plane stencil | paired cubic arrays + rolling-plane output pattern | `RollingPlaneStencilMain` |
| P2 | hash aggregate | array input/output + scratch table aggregate pattern | `HashAggregateMain` |
| P2 | recursive choice | recursive two-parameter choice + paired arrays pattern | `RecursiveChoiceMain` |
| P2 | recursive bucket sort | recursive bucket partition pattern | `RecursiveBucketSortMain` |
| P2 | permutation checksum | transpose/permutation dataflow pattern | `PermutationChecksum` |
| P2 | integer trajectory | bounded recursive trajectory + reduction pattern | `BoundedIntegerTrajectory` |
| P2 | state generators | state load/update/return pattern | `AffineStateGenerator`, `BoundedStateGenerator` |

## Guardrails

- No public case-name trigger.
- No source-path trigger.
- No user function-name trigger. Entry matching accepts `main` or a single runtime-boundary driver called by `main`.
- No output-value trigger.
- No program fingerprint or hash trigger.
- Global arrays are selected from call operands, stores, dimensions, and role flow; matchers do not require the total number of globals or same-shaped arrays to be exact.
- Absolute size thresholds are avoided in structural matchers; array roles use relative capacity/shape constraints where possible.
- Algorithm constants are only used with surrounding structural/dataflow guards.
