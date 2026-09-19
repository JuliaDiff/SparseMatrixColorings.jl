for (structure, partition, decompression, uplo) in [
    (:nonsymmetric, :column, :direct, :F),
    (:nonsymmetric, :row, :direct, :F),
    (:symmetric, :column, :direct, :F),
    (:symmetric, :column, :direct, :L),
    (:symmetric, :column, :substitution, :F),
    (:symmetric, :column, :substitution, :L),
    (:nonsymmetric, :bidirectional, :direct, :F),
    (:nonsymmetric, :bidirectional, :substitution, :F),
]
    A = sparse(Bool[1 0; 0 1])
    problem = ColoringProblem(; structure, partition, uplo)
    algo = GreedyColoringAlgorithm(; decompression, postprocessing=true)
    result = coloring(A, problem, algo)
    if partition == :bidirectional
        Br, Bc = compress(A, result)
        decompress(Br, Bc, result)
    else
        B = compress(A, result)
        decompress(B, result)
    end
end
