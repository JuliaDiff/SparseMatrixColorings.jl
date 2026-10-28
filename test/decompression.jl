using LinearAlgebra
using SparseArrays
using SparseMatrixColorings
using SparseMatrixColorings: SparsityPatternCSC, decompress_csc!
using Test

function test_star_csc_decompression(T, Ti, order, postprocessing)
    problem = ColoringProblem(; structure=:symmetric, partition=:column)
    algorithm = GreedyColoringAlgorithm(order; decompression=:direct, postprocessing)
    @testset "edges = $edges" for edges in (
        [(2, 4), (4, 7), (7, 9)],
        [(2, 4), (2, 7), (2, 9), (7, 7)],
        [(2, 4), (4, 7), (7, 9), (2, 9), (4, 4), (9, 9)],
        [(2, 2), (9, 9)],
        Tuple{Int,Int}[],
    )
        A = spzeros(T, Ti, 9, 9)
        for (i, j) in edges
            A[i, j] = A[j, i] = T <: Complex ? T(10i + j, i - j) : T(10i + j)
        end
        # Keep a structural zero as well as empty rows and columns.
        if !isempty(edges)
            A[5, 5] = one(T)
            A[5, 5] = zero(T)
        end
        result = coloring(A, problem, algorithm)
        B = compress(A, result)
        @testset "$uplo" for uplo in (:F, :L, :U)
            expected = if uplo == :F
                A
            elseif uplo == :L
                tril(A)
            else
                triu(A)
            end
            V = fill(T(NaN), nnz(expected))
            @test decompress_csc!(V, expected.colptr, B, result, uplo) === nothing
            @test V == nonzeros(expected)
            @test (@allocated decompress_csc!(V, expected.colptr, B, result, uplo)) == 0

            dest = copy(expected)
            fill!(nonzeros(dest), T(NaN))
            @test decompress!(dest, B, result, uplo) === dest
            @test dest == expected
            @test (@allocated decompress!(dest, B, result, uplo)) == 0

            # Abstract buffers and a different destination eltype remain supported.
            storage = fill(ComplexF64(NaN), 2nnz(expected))
            values = @view storage[1:2:end]
            compressed = @view B[:, :]
            @test decompress_csc!(values, expected.colptr, compressed, result, uplo) ===
                nothing
            @test values == nonzeros(expected)
            @test (@allocated decompress_csc!(
                values, expected.colptr, compressed, result, uplo
            )) == 0
            mixed = SparseMatrixCSC{ComplexF64,Ti}(expected)
            fill!(nonzeros(mixed), ComplexF64(NaN))
            @test decompress!(mixed, compressed, result, uplo) === mixed
            @test mixed == expected
        end
        full_values = similar(nonzeros(A))
        @test decompress_csc!(full_values, A.colptr, B, result) === nothing
        @test full_values == nonzeros(A)
    end
    return nothing
end

@testset "Star CSC decompression" begin
    @testset "$T, $Ti, $order, postprocessing=$postprocessing" for T in (
            Float32, Float64, ComplexF64
        ),
        Ti in (Int32, Int64), order in (NaturalOrder(), LargestFirst()),
        postprocessing in (false, true)

        test_star_csc_decompression(T, Ti, order, postprocessing)
    end
end

function test_tree_csc_decompression(T, Ti, order, postprocessing)
    problem = ColoringProblem(; structure=:symmetric)
    algorithm = GreedyColoringAlgorithm(order; decompression=:substitution, postprocessing)
    @testset "$pattern" for pattern in (
        :blocks, :star_low, :star_middle, :star_high, :two_edges, :mixed, :diagonal, :empty
    )
        A = spzeros(T, Ti, 10, 10)
        if pattern in (:blocks, :mixed)
            clique_size = pattern == :blocks ? 4 : 3
            for j in 1:clique_size, i in 1:(j - 1)
                A[i, j] = A[j, i] = T(10i + j)
            end
            edges = if pattern == :blocks
                [(6, 7), (6, 8), (7, 8)]
            else
                [(4, 5), (5, 6), (6, 7), (8, 9)]
            end
            for (i, j) in edges
                A[i, j] = A[j, i] = T(10i + j)
            end
        end
        if pattern in (:star_low, :star_middle, :star_high, :two_edges)
            hub = if pattern == :star_low
                1
            elseif pattern == :star_high
                9
            else
                5
            end
            leaves = pattern == :two_edges ? (2, 7) : (2, 4, 7)
            for leaf in leaves
                A[leaf, hub] = A[hub, leaf] = T(10leaf + hub)
            end
        end
        if pattern != :empty
            # Missing diagonals, a structural zero, and an empty last column.
            A[2, 2] = T(2)
            A[8, 8] = T(8)
            A[3, 3] = one(T)
            A[3, 3] = zero(T)
        end
        result = coloring(SparsityPatternCSC(A), problem, algorithm; decompression_eltype=T)
        sizes = diff(result.tree_edge_indices)
        if pattern == :blocks
            @test !isempty(sizes)
            @test all(==(1), sizes)
            @test all(result.is_star)
        elseif pattern in (:star_low, :star_middle, :star_high, :two_edges)
            @test sizes == [pattern == :two_edges ? 2 : 3]
            @test all(result.is_star)
        elseif pattern == :mixed
            @test any(==(1), sizes)
            @test any(>(1), sizes)
            @test any(!, result.is_star)
        end
        B = compress(A, result)
        @testset "$uplo" for uplo in (:F, :L, :U)
            expected = if uplo == :F
                A
            elseif uplo == :L
                tril(A)
            else
                triu(A)
            end
            values = fill(T(NaN), nnz(expected))
            # Reuse the result with changed compressed values and stale scratch data.
            for scale in (one(T), T(-2))
                compressed = scale .* B
                fill!(result.buffer, T(NaN))
                @test decompress_csc!(values, expected.colptr, compressed, result, uplo) ===
                    nothing
                @test values == scale .* nonzeros(expected)
                if all(result.is_star)
                    @test all(isnan, result.buffer)
                end
                @test (@allocated decompress_csc!(
                    values, expected.colptr, compressed, result, uplo
                )) == 0

                dest = copy(expected)
                fill!(nonzeros(dest), T(NaN))
                @test decompress!(dest, compressed, result, uplo) === dest
                @test dest == scale .* expected
                @test (@allocated decompress!(dest, compressed, result, uplo)) == 0
            end

            # The CSC entry point also accepts views of its input/output buffers.
            storage = fill(T(NaN), 2nnz(expected))
            values_view = @view storage[1:2:end]
            compressed_view = @view B[:, :]
            @test decompress_csc!(
                values_view, expected.colptr, compressed_view, result, uplo
            ) === nothing
            @test values_view == nonzeros(expected)
            @test (@allocated decompress_csc!(
                values_view, expected.colptr, compressed_view, result, uplo
            )) == 0
        end
    end
    return nothing
end

@testset "Tree CSC decompression" begin
    @testset "$T, $Ti, $order, postprocessing=$postprocessing" for T in (Float32, Float64),
        Ti in (Int32, Int64), order in (NaturalOrder(), LargestFirst()),
        postprocessing in (false, true)

        test_tree_csc_decompression(T, Ti, order, postprocessing)
    end
end

struct ZeroBasedDecompressionArray{T,N,A<:AbstractArray{T,N}} <: AbstractArray{T,N}
    parent::A
end
Base.size(A::ZeroBasedDecompressionArray) = size(A.parent)
Base.axes(A::ZeroBasedDecompressionArray) = map(n -> 0:(n - 1), size(A))

@testset "Tree CSC buffer validation" begin
    A = sparse([1, 2, 2, 3, 3, 4, 4], [2, 1, 3, 2, 4, 3, 4], ones(7), 6, 6)
    result = coloring(
        A,
        ColoringProblem(; structure=:symmetric),
        GreedyColoringAlgorithm(NaturalOrder(); decompression=:substitution),
    )
    B = compress(A, result)
    for uplo in (:F, :L, :U)
        expected = if uplo == :F
            A
        elseif uplo == :L
            tril(A)
        else
            triu(A)
        end
        values = fill(NaN, nnz(expected))
        pointers = expected.colptr
        fill!(result.buffer, NaN)
        for bad_B in (B[1:(end - 1), :], B[:, 1:(end - 1)], hcat(B, B))
            @test_throws DimensionMismatch decompress_csc!(
                values, pointers, bad_B, result, uplo
            )
        end
        for bad_values in (values[1:(end - 1)], vcat(values, NaN))
            @test_throws DimensionMismatch decompress_csc!(
                bad_values, pointers, B, result, uplo
            )
        end
        @test_throws DimensionMismatch decompress_csc!(
            values, pointers[1:(end - 1)], B, result, uplo
        )
        @test_throws ArgumentError decompress_csc!(values, pointers, B, result, :invalid)
        @test_throws ArgumentError decompress_csc!(
            ZeroBasedDecompressionArray(values), pointers, B, result, uplo
        )
        @test_throws ArgumentError decompress_csc!(
            values, ZeroBasedDecompressionArray(pointers), B, result, uplo
        )
        @test_throws ArgumentError decompress_csc!(
            values, pointers, ZeroBasedDecompressionArray(B), result, uplo
        )
        # Every invalid call must fail before writing output or scratch data.
        @test all(isnan, values)
        @test all(isnan, result.buffer)
        @test decompress_csc!(values, pointers, B, result, uplo) === nothing
        @test values == nonzeros(expected)

        # Destination writes retain their ordinary bounds checks. A bad
        # pointer can fail after earlier coefficients have already been written.
        for bad_pointer in (0, 2length(values) + 2)
            bad_pointers = fill(bad_pointer, length(pointers))
            @test_throws BoundsError decompress_csc!(values, bad_pointers, B, result, uplo)
            storage = zeros(2length(values))
            values_view = @view storage[1:2:end]
            @test_throws BoundsError decompress_csc!(
                values_view, bad_pointers, B, result, uplo
            )
        end
    end
end

function test_tree_bicoloring_decompression(T, Ti, order, target)
    problem = ColoringProblem(; structure=:nonsymmetric, partition=:bidirectional)
    algorithm = GreedyColoringAlgorithm(
        order;
        decompression=:substitution,
        postprocessing=target != :none,
        postprocessing_minimizes=target == :none ? :all_colors : target,
    )
    @testset "$pattern" for pattern in (
        :single_edge, :row_star, :column_star, :two_edges, :mixed, :empty
    )
        A = spzeros(T, Ti, 8, 9)
        edges = if pattern == :single_edge
            [(2, 3)]
        elseif pattern == :row_star
            [(2, 1), (2, 3), (2, 5)]
        elseif pattern == :column_star
            [(1, 2), (3, 2), (5, 2)]
        elseif pattern == :two_edges
            [(2, 1), (2, 3)]
        elseif pattern == :mixed
            [(1, 1), (1, 2), (2, 2), (4, 4), (5, 5), (5, 6), (5, 7)]
        else
            Tuple{Int,Int}[]
        end
        for (i, j) in edges
            A[i, j] = T <: Complex ? T(10i + j, i - j) : T(10i + j)
        end
        # Keep a structural zero, empty rows, and empty columns.
        if !isempty(edges)
            nonzeros(A)[end] = zero(T)
        end
        result = coloring(A, problem, algorithm; decompression_eltype=T)
        sizes = diff(result.tree_edge_indices)
        if pattern == :mixed
            @test any(!, result.is_star)
            @test any(result.is_star .& (sizes .> 1))
            @test any(==(1), sizes)
        else
            @test all(result.is_star)
            if pattern in (:row_star, :column_star)
                @test sizes == [3]
            elseif pattern == :two_edges
                @test sizes == [2]
            elseif pattern == :single_edge
                @test sizes == [1]
            else
                @test isempty(sizes)
            end
        end
        Br, Bc = compress(A, result)
        @testset "$(typeof(dest))" for dest in (copy(A), Matrix(A))
            for scale in (one(T), T(-2))
                rows, columns = scale .* Br, scale .* Bc
                rows_view, columns_view = @view(rows[:, :]), @view(columns[:, :])
                fill!(dest isa SparseMatrixCSC ? nonzeros(dest) : dest, T(NaN))
                fill!(result.buffer, T(NaN))
                @test decompress!(dest, rows_view, columns_view, result) === dest
                @test dest == scale .* A
                if all(result.is_star)
                    @test all(isnan, result.buffer)
                end
                @test (@allocated decompress!(dest, rows_view, columns_view, result)) == 0
            end
        end
    end
    return nothing
end

@testset "Tree bicoloring decompression" begin
    @testset "$T, $Ti, $order, target=$target" for T in (Float32, Float64, ComplexF64),
        Ti in (Int32, Int64), order in (NaturalOrder(), LargestFirst()),
        target in (:none, :all_colors, :row_colors, :column_colors)

        test_tree_bicoloring_decompression(T, Ti, order, target)
    end
end
