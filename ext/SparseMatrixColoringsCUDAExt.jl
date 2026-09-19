module SparseMatrixColoringsCUDAExt

import SparseMatrixColorings as SMC
using SparseArrays: SparseMatrixCSC, rowvals, nnz, nzrange
using CUDA: CuVector, CuMatrix
using cuSPARSE: AbstractCuSparseMatrix, CuSparseMatrixCSC, CuSparseMatrixCSR

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

function SMC.StarSetBicoloringResult(
    A::CuSparseMatrixCSC,
    S::SMC.SparsityPatternCSC{T},
    ag::SMC.AdjacencyGraph{T},
    symmetric_color::Vector{<:Integer},
    star_set::SMC.StarSet{<:Integer},
    row_color::Vector{T},
    column_color::Vector{T},
    symmetric_to_row::Vector{T},
    symmetric_to_column::Vector{T},
) where {T<:Integer}
    column_group = SMC.group_by_color(T, column_color)
    row_group = SMC.group_by_color(T, row_color)
    num_row_colors = length(row_group)
    A_indices_bc, compressed_indices_bc, A_indices_br, compressed_indices_br = SMC.star_bicoloring_csc_indices(
        S, symmetric_color, star_set, symmetric_to_row, symmetric_to_column, num_row_colors
    )
    additional_info = (;
        A_indices_gpu_bc_csc=CuVector(A_indices_bc),
        compressed_indices_gpu_bc_csc=CuVector(compressed_indices_bc),
        A_indices_gpu_br_csc=CuVector(A_indices_br),
        compressed_indices_gpu_br_csc=CuVector(compressed_indices_br),
    )
    return SMC.StarSetBicoloringResult(
        A,
        S,
        ag,
        symmetric_color,
        column_color,
        row_color,
        column_group,
        row_group,
        symmetric_to_column,
        symmetric_to_row,
        A_indices_bc,
        compressed_indices_bc,
        A_indices_br,
        compressed_indices_br,
        additional_info,
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

function SMC.StarSetBicoloringResult(
    A::CuSparseMatrixCSR,
    S::SMC.SparsityPatternCSC{T},
    ag::SMC.AdjacencyGraph{T},
    symmetric_color::Vector{<:Integer},
    star_set::SMC.StarSet{<:Integer},
    row_color::Vector{T},
    column_color::Vector{T},
    symmetric_to_row::Vector{T},
    symmetric_to_column::Vector{T},
) where {T<:Integer}
    column_group = SMC.group_by_color(T, column_color)
    row_group = SMC.group_by_color(T, row_color)
    num_row_colors = length(row_group)
    A_indices_bc, compressed_indices_bc, A_indices_br, compressed_indices_br = SMC.star_bicoloring_csr_indices(
        ag,
        S,
        symmetric_color,
        star_set,
        symmetric_to_row,
        symmetric_to_column,
        num_row_colors,
    )
    additional_info = (;
        A_indices_gpu_bc_csr=CuVector(A_indices_bc),
        compressed_indices_gpu_bc_csr=CuVector(compressed_indices_bc),
        A_indices_gpu_br_csr=CuVector(A_indices_br),
        compressed_indices_gpu_br_csr=CuVector(compressed_indices_br),
    )
    return SMC.StarSetBicoloringResult(
        A,
        S,
        ag,
        symmetric_color,
        column_color,
        row_color,
        column_group,
        row_group,
        symmetric_to_column,
        symmetric_to_row,
        A_indices_bc,
        compressed_indices_bc,
        A_indices_br,
        compressed_indices_br,
        additional_info,
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

function SMC.decompress!(
    A::CuSparseMatrixCSC,
    Br::CuMatrix,
    Bc::CuMatrix,
    result::SMC.StarSetBicoloringResult{<:CuSparseMatrixCSC},
)
    (;
        A_indices_gpu_bc_csc,
        compressed_indices_gpu_bc_csc,
        A_indices_gpu_br_csc,
        compressed_indices_gpu_br_csc,
    ) = result.additional_info
    nzA = A.nzVal
    view(nzA, A_indices_gpu_bc_csc) .= view(Bc, compressed_indices_gpu_bc_csc)
    view(nzA, A_indices_gpu_br_csc) .= view(Br, compressed_indices_gpu_br_csc)
    return A
end

function SMC.decompress!(
    A::CuSparseMatrixCSR,
    Br::CuMatrix,
    Bc::CuMatrix,
    result::SMC.StarSetBicoloringResult{<:CuSparseMatrixCSR},
)
    (;
        A_indices_gpu_bc_csr,
        compressed_indices_gpu_bc_csr,
        A_indices_gpu_br_csr,
        compressed_indices_gpu_br_csr,
    ) = result.additional_info
    nzA = A.nzVal
    view(nzA, A_indices_gpu_bc_csr) .= view(Bc, compressed_indices_gpu_bc_csr)
    view(nzA, A_indices_gpu_br_csr) .= view(Br, compressed_indices_gpu_br_csr)
    return A
end

end
