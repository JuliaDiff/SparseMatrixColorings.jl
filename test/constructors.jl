using SparseMatrixColorings
using Test

@test ColoringProblem{:nonsymmetric,:column}() == ColoringProblem()
@test ColoringProblem{:symmetric,:column}() ==
    ColoringProblem(; structure=:symmetric, partition=:column)

# the two-parameter form still works and defaults to the full matrix
@test ColoringProblem{:nonsymmetric,:column,:F}() ==
    ColoringProblem{:nonsymmetric,:column}()
@test ColoringProblem{:symmetric,:column,:U}() ==
    ColoringProblem(; structure=:symmetric, partition=:column, uplo=:U)
@test ColoringProblem{:symmetric,:column,:L}() ==
    ColoringProblem(; structure=:symmetric, partition=:column, uplo=:L)

@test_throws ArgumentError ColoringProblem(; structure=:weird, partition=:column)
@test_throws ArgumentError ColoringProblem(; structure=:symmetric, partition=:row)

# uplo must be one of :F, :L, :U and only makes sense for symmetric problems
@test_throws ArgumentError ColoringProblem(;
    structure=:symmetric, partition=:column, uplo=:weird
)
@test_throws ArgumentError ColoringProblem(;
    structure=:nonsymmetric, partition=:column, uplo=:L
)
@test_throws ArgumentError ColoringProblem(;
    structure=:nonsymmetric, partition=:bidirectional, uplo=:L
)

@test GreedyColoringAlgorithm{:direct}() == GreedyColoringAlgorithm()
@test GreedyColoringAlgorithm{:substitution}() ==
    GreedyColoringAlgorithm(; decompression=:substitution)

@test_throws ArgumentError GreedyColoringAlgorithm(decompression=:weird)
