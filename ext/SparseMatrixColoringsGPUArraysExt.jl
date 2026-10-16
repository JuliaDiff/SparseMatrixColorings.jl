module SparseMatrixColoringsGPUArraysExt

using GPUArrays: AbstractGPUSparseMatrix, dense_array_type
using LinearAlgebra: tril, triu
using SparseArrays: SparseMatrixCSC
import SparseMatrixColorings as SMC

SMC.matrix_versions(A::AbstractGPUSparseMatrix) = (A,)

## Compression (slow, through CPU)

function SMC.compress(
    A::AbstractGPUSparseMatrix, result::SMC.AbstractColoringResult{structure,:column}
) where {structure}
    A_cpu = SparseMatrixCSC(A)
    B_cpu = SMC.compress(A_cpu, result)
    B = dense_array_type(A)(B_cpu)
    return B
end

function SMC.compress(
    A::AbstractGPUSparseMatrix, result::SMC.AbstractColoringResult{structure,:row}
) where {structure}
    A_cpu = SparseMatrixCSC(A)
    B_cpu = SMC.compress(A_cpu, result)
    B = dense_array_type(A)(B_cpu)
    return B
end

function SMC.compress(
    A::AbstractGPUSparseMatrix, result::SMC.AbstractColoringResult{structure,:bidirectional}
) where {structure}
    A_cpu = SparseMatrixCSC(A)
    Br_cpu, Bc_cpu = SMC.compress(A_cpu, result)
    M = dense_array_type(A)
    return M(Br_cpu), M(Bc_cpu)
end

## Decompression

function SMC.decompress(
    B::AbstractMatrix, result::SMC.StarSetColoringResult{<:AbstractGPUSparseMatrix}
)
    (; decompression_uplo) = result
    A = SMC.respectful_similar(result.A, eltype(B))
    if decompression_uplo != :F
        fill!(A.nzVal, one(eltype(B)))
        A = decompression_uplo == :L ? tril(A) : triu(A)
    end
    return SMC.decompress!(A, B, result)
end

function SMC.decompress!(
    A::AbstractGPUSparseMatrix, B::AbstractMatrix, result::SMC.TreeSetColoringResult
)
    return throw(
        SMC.UnsupportedDecompressionError(
            "Symmetric decompression by substitution is not supported on GPU matrices"
        ),
    )
end

function SMC.decompress!(
    A::AbstractGPUSparseMatrix,
    Br::AbstractMatrix,
    Bc::AbstractMatrix,
    result::SMC.TreeSetBicoloringResult,
)
    return throw(
        SMC.UnsupportedDecompressionError(
            "Bidirectional decompression by substitution is not supported on GPU matrices"
        ),
    )
end

end
