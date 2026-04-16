module SparseMatrixColoringsGPUArraysExt

using GPUArrays: AbstractGPUSparseMatrix, dense_array_type
using SparseArrays: SparseMatrixCSC
import SparseMatrixColorings as SMC

SMC.matrix_versions(A::AbstractGPUSparseMatrix) = (A,)

## Compression (slow, through CPU)

function SMC.compress(A::AbstractGPUSparseMatrix, result::SMC.AbstractColoringResult)
    A_cpu = SparseMatrixCSC(A)
    B_cpu = SMC.compress(A_cpu, result)
    B = dense_array_type(A)(B_cpu)
    return B
end

end
