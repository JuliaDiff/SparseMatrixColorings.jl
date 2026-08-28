module SparseMatrixColoringsCUDAExt
using LinearAlgebra
import SparseMatrixColorings as SMC
using SparseArrays: SparseMatrixCSC, rowvals, nnz, nzrange
using CUDA: CuArray, CuVector, CuMatrix
using cuSPARSE: AbstractCuSparseMatrix, CuSparseMatrixCSC, CuSparseMatrixCSR

## Basic support for GPU sparsity pattern stuff

function SMC.SparsityPatternCSC(A::CuSparseMatrixCSC)
    SMC.SparsityPatternCSC(first(A.dims), last(A.dims), A.colPtr, A.rowVal)
end

for R in (:Diagonal, :Bidiagonal, :Tridiagonal)
    @eval function SMC.BipartiteGraph(
        A::$R{T,<:CuArray}; symmetric_pattern::Bool=false
    ) where {T}
        return SMC.BipartiteGraph(CuSparseMatrixCSC(A); symmetric_pattern)
    end
end

function SMC.BipartiteGraph(A::CuSparseMatrixCSC; symmetric_pattern::Bool=false)
    S2 = SMC.SparsityPatternCSC(A)
    if symmetric_pattern
        checksquare(A)  # proxy for checking full symmetry
        S1 = S2
    else
        S1 = transpose(S2)  # rows to columns
    end
    return SMC.BipartiteGraph(S1, S2)
end

## CSC Result

function SMC.ColumnColoringResult(
    A::CuSparseMatrixCSC, bg::SMC.BipartiteGraph{T}, color::Vector{<:Integer}
) where {T<:Integer}
    group = SMC.group_by_color(T, color)
    compressed_indices = SMC.column_csc_indices(bg, color)
    additional_info = (; compressed_indices_gpu_csc=CuVector(compressed_indices))
    return SMC.ColumnColoringResult(
        A, bg, color, group, compressed_indices, additional_info
    )
end

function SMC.RowColoringResult(
    A::CuSparseMatrixCSC, bg::SMC.BipartiteGraph{T}, color::Vector{<:Integer}
) where {T<:Integer}
    group = SMC.group_by_color(T, color)
    compressed_indices = SMC.row_csc_indices(bg, color)
    additional_info = (; compressed_indices_gpu_csc=CuVector(compressed_indices))
    return SMC.RowColoringResult(A, bg, color, group, compressed_indices, additional_info)
end

function SMC.StarSetColoringResult(
    A::CuSparseMatrixCSC,
    ag::SMC.AdjacencyGraph{T},
    color::Vector{<:Integer},
    star_set::SMC.StarSet{<:Integer},
) where {T<:Integer}
    group = SMC.group_by_color(T, color)
    compressed_indices = SMC.star_csc_indices(ag, color, star_set)
    additional_info = (; compressed_indices_gpu_csc=CuVector(compressed_indices))
    return SMC.StarSetColoringResult(
        A, ag, color, group, compressed_indices, additional_info
    )
end

## CSR Result

function SMC.ColumnColoringResult(
    A::CuSparseMatrixCSR, bg::SMC.BipartiteGraph{T}, color::Vector{<:Integer}
) where {T<:Integer}
    group = SMC.group_by_color(T, color)
    compressed_indices = SMC.column_csc_indices(bg, color)
    compressed_indices_csr = SMC.column_csr_indices(bg, color)
    additional_info = (; compressed_indices_gpu_csr=CuVector(compressed_indices_csr))
    return SMC.ColumnColoringResult(
        A, bg, color, group, compressed_indices, additional_info
    )
end

function SMC.RowColoringResult(
    A::CuSparseMatrixCSR, bg::SMC.BipartiteGraph{T}, color::Vector{<:Integer}
) where {T<:Integer}
    group = SMC.group_by_color(T, color)
    compressed_indices = SMC.row_csc_indices(bg, color)
    compressed_indices_csr = SMC.row_csr_indices(bg, color)
    additional_info = (; compressed_indices_gpu_csr=CuVector(compressed_indices_csr))
    return SMC.RowColoringResult(A, bg, color, group, compressed_indices, additional_info)
end

function SMC.StarSetColoringResult(
    A::CuSparseMatrixCSR,
    ag::SMC.AdjacencyGraph{T},
    color::Vector{<:Integer},
    star_set::SMC.StarSet{<:Integer},
) where {T<:Integer}
    group = SMC.group_by_color(T, color)
    compressed_indices = SMC.star_csc_indices(ag, color, star_set)
    additional_info = (; compressed_indices_gpu_csr=CuVector(compressed_indices))
    return SMC.StarSetColoringResult(
        A, ag, color, group, compressed_indices, additional_info
    )
end

## Decompression

for R in (:ColumnColoringResult, :RowColoringResult)
    # loop to avoid method ambiguity
    @eval function SMC.decompress!(
        A::CuSparseMatrixCSC, B::CuMatrix, result::SMC.$R{<:CuSparseMatrixCSC}
    )
        compressed_indices = result.additional_info.compressed_indices_gpu_csc
        copyto!(A.nzVal, view(B, compressed_indices))
        return A
    end

    @eval function SMC.decompress!(
        A::CuSparseMatrixCSR, B::CuMatrix, result::SMC.$R{<:CuSparseMatrixCSR}
    )
        compressed_indices = result.additional_info.compressed_indices_gpu_csr
        copyto!(A.nzVal, view(B, compressed_indices))
        return A
    end
end

function SMC.decompress!(
    A::CuSparseMatrixCSC,
    B::CuMatrix,
    result::SMC.StarSetColoringResult{<:CuSparseMatrixCSC},
    uplo::Symbol=:F,
)
    if uplo != :F
        throw(
            SMC.UnsupportedDecompressionError(
                "Single-triangle decompression is not supported on GPU matrices"
            ),
        )
    end
    compressed_indices = result.additional_info.compressed_indices_gpu_csc
    copyto!(A.nzVal, view(B, compressed_indices))
    return A
end

function SMC.decompress!(
    A::CuSparseMatrixCSR,
    B::CuMatrix,
    result::SMC.StarSetColoringResult{<:CuSparseMatrixCSR},
    uplo::Symbol=:F,
)
    if uplo != :F
        throw(
            SMC.UnsupportedDecompressionError(
                "Single-triangle decompression is not supported on GPU matrices"
            ),
        )
    end
    compressed_indices = result.additional_info.compressed_indices_gpu_csr
    copyto!(A.nzVal, view(B, compressed_indices))
    return A
end

end
