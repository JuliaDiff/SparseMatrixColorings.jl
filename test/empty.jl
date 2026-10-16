using LinearAlgebra
using SparseArrays
using SparseMatrixColorings
using Test

@testset "Empty symmetric coloring" begin
    problem = ColoringProblem(; structure=:symmetric, partition=:column)
    orders = (NaturalOrder(), LargestFirst(), (NaturalOrder(), LargestFirst()))
    matrices = (
        zeros(0, 0),
        spzeros(0, 0),
        Symmetric(spzeros(0, 0)),
        SparseMatrixColorings.SparsityPatternCSC(spzeros(0, 0)),
    )
    @testset "$decompression, $postprocessing, $order, $(typeof(A))" for decompression in (
            :direct, :substitution
        ),
        postprocessing in (false, true), order in orders,
        A in matrices

        algo = GreedyColoringAlgorithm(order; decompression, postprocessing)
        result = coloring(A, problem, algo)
        @test ncolors(result) == 0
        @test isempty(column_colors(result))
        @test isempty(column_groups(result))
        @test fast_coloring(A, problem, algo) == Int[]
        B = compress(A, result)
        @test size(B) == (0, 0)
        @test decompress(B, result) == A
        target = spzeros(0, 0)
        @test decompress!(target, B, result) === target
        @test size(target) == (0, 0)
    end
end
