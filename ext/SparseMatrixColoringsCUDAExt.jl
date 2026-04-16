module SparseMatrixColoringsCUDAExt

import SparseMatrixColorings as SMC
using SparseArrays: SparseMatrixCSC, rowvals, nnz, nzrange
using CUDA: CuVector, CuMatrix
using cuSPARSE: AbstractCuSparseMatrix, CuSparseMatrixCSC, CuSparseMatrixCSR

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
