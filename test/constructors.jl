using SparseMatrixColorings
using Test

@test ColoringProblem{:nonsymmetric,:column}() == ColoringProblem()
@test ColoringProblem{:symmetric,:column}() ==
    ColoringProblem(; structure=:symmetric, partition=:column)

@test_throws ArgumentError ColoringProblem(; structure=:weird, partition=:column)
@test_throws ArgumentError ColoringProblem(; structure=:symmetric, partition=:row)

let A = [1 0; 0 1], algo = GreedyColoringAlgorithm()
    @test_throws ArgumentError coloring(
        A,
        ColoringProblem(; structure=:symmetric, partition=:column),
        algo;
        decompression_uplo=:weird,
    )
    @test_throws ArgumentError coloring(
        A,
        ColoringProblem(; structure=:nonsymmetric, partition=:column),
        algo;
        decompression_uplo=:L,
    )
    @test_throws ArgumentError coloring(
        A,
        ColoringProblem(; structure=:nonsymmetric, partition=:bidirectional),
        algo;
        decompression_uplo=:L,
    )
end

@test GreedyColoringAlgorithm{:direct}() == GreedyColoringAlgorithm()
@test GreedyColoringAlgorithm{:substitution}() ==
    GreedyColoringAlgorithm(; decompression=:substitution)

@test_throws ArgumentError GreedyColoringAlgorithm(decompression=:weird)
