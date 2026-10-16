using LinearAlgebra
using SparseArrays
using SparseMatrixColorings
using SparseMatrixColorings: decompress_csc!
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
