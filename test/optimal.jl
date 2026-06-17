using SparseArrays
using SparseMatrixColorings
using StableRNGs
using Test
using JuMP
using MiniZinc
using HiGHS

rng = StableRNG(0)

asymmetric_params = vcat(
    [(10, 20, p) for p in (0.0:0.1:0.5)], [(20, 10, p) for p in (0.0:0.1:0.5)]
)

algo = GreedyColoringAlgorithm()
optalgo = OptimalColoringAlgorithm(() -> MiniZinc.Optimizer{Float64}("highs"); silent=false)

# TODO: reactivate tests once https://github.com/jump-dev/MiniZinc.jl/issues/103 is fixed

@testset "Column coloring" begin
    problem = ColoringProblem(; structure=:nonsymmetric, partition=:column)
    for (m, n, p) in asymmetric_params
        A = sprand(rng, m, n, p)
        result = coloring(A, problem, algo)
        @test_skip ncolors(result) >= ncolors(coloring(A, problem, optalgo))
    end
end

@testset "Row coloring" begin
    problem = ColoringProblem(; structure=:nonsymmetric, partition=:row)
    for (m, n, p) in asymmetric_params
        A = sprand(rng, m, n, p)
        result = coloring(A, problem, algo)
        @test_skip ncolors(result) >= ncolors(coloring(A, problem, optalgo))
    end
end

@testset "Too big" begin
    A = sprand(rng, Bool, 100, 100, 0.1)
    optalgo_timelimit = OptimalColoringAlgorithm(
        optimizer_with_attributes(HiGHS.Optimizer, "time_limit" => 10.0); # 1 second
        silent=false,
        assert_solved=false,
    )
    @test_throws AssertionError coloring(A, ColoringProblem(), optalgo_timelimit)
end
