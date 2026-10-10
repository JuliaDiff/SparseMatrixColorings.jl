using ADTypes: column_coloring, row_coloring
using LinearAlgebra
using SparseArrays
using SparseMatrixColorings
using SparseMatrixColorings: quotient_pattern, structurally_orthogonal_columns
using Test

# 5-point Laplacian on an m × m grid
function laplacian(m)
    T = spdiagm(-1 => ones(m - 1), 0 => ones(m), 1 => ones(m - 1))
    return kron(sparse(I, m, m), T) + kron(T, sparse(I, m, m))
end

# position of the grid points (or elements) modulo an a × a cell
cell(m, a) = [1 + mod(i, a) + a * mod(j, a) for j in 0:(m - 1) for i in 0:(m - 1)]

# column `j` in block `e` at local index `ℓ` gets the label of `(ℓ, block_labels[e])`
function block_classes(block_size, block_labels)
    return [ℓ + block_size * (c - 1) for c in block_labels for ℓ in 1:block_size]
end

@testset "Quotient pattern" begin
    A = sparse([
        1 1 0 0
        0 1 1 0
        0 0 1 1
    ])
    Q, class_index = quotient_pattern(A, [7, 3, 7, 3])
    @test class_index == [1, 2, 1, 2]
    @test Q == sparse(Bool[1 1])  # the three rows have the same set of classes
    @test_throws ArgumentError quotient_pattern(A, [1, 1, 2, 2])
    @test_throws DimensionMismatch quotient_pattern(A, [1, 2, 3])
end

@testset "Laplacian" begin
    m = 20
    A = laplacian(m)
    greedy = ncolors(coloring(A, ColoringProblem(), GreedyColoringAlgorithm()))
    @testset "$partition" for partition in (:column, :row)
        problem = ColoringProblem(; structure=:nonsymmetric, partition)
        result = coloring(A, problem, QuotientColoringAlgorithm(cell(m, 5)))
        @test ncolors(result) == 5  # optimal
        @test ncolors(result) < greedy
        @test decompress(compress(A, result), result) == A
    end
    colors = column_coloring(A, QuotientColoringAlgorithm(cell(m, 5)))
    @test structurally_orthogonal_columns(A, colors)
    @test all(
        colors[j] == colors[k] for
        j in axes(A, 2), k in axes(A, 2) if cell(m, 5)[j] == cell(m, 5)[k]
    )
    row_colors = row_coloring(A, QuotientColoringAlgorithm(cell(m, 5)))
    @test structurally_orthogonal_columns(sparse(transpose(A)), row_colors)
    # without recoloring of the classes
    algo = QuotientColoringAlgorithm(cell(m, 5), GreedyColoringAlgorithm())
    @test structurally_orthogonal_columns(A, column_coloring(A, algo))
    # trivial classes: one class per column
    algo = QuotientColoringAlgorithm(1:(m ^ 2), GreedyColoringAlgorithm())
    @test ncolors(coloring(A, ColoringProblem(), algo)) == greedy
    # a 2 × 2 cell is too small: neighbors share a class
    @test_throws ArgumentError coloring(
        A, ColoringProblem(), QuotientColoringAlgorithm(cell(m, 2))
    )
end

@testset "Block pattern" begin
    # 12 × 12 elements with 4 dofs each, dense coupling to face neighbors
    ne, b = 12, 4
    A = kron(laplacian(ne), ones(b, b))
    lower_bound = 5b
    classes = block_classes(b, cell(ne, 5))
    result = coloring(A, ColoringProblem(), QuotientColoringAlgorithm(classes))
    @test ncolors(result) == lower_bound
    @test ncolors(coloring(A, ColoringProblem(), GreedyColoringAlgorithm())) > lower_bound
    @test structurally_orthogonal_columns(A, column_colors(result))
end
