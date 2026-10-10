"""
    QuotientColoringAlgorithm <: ADTypes.AbstractColoringAlgorithm

Coloring algorithm which gives the same color to all columns (or rows) with the same class label, and colors the classes with another algorithm.

Two classes conflict if any of their columns (rows) have a nonzero in a common row (column).
The conflict pattern of the classes is usually much smaller than the matrix, so the coloring of the classes can afford many passes of iterated greedy recoloring.
This exploits translation structure that a greedy coloring of the matrix does not see, e.g., for discretizations on structured meshes: label each column by its local index in the element (or stencil) and the position of the element modulo a small periodic cell of elements.
The cell must be compatible with a good coloring of the infinite periodic pattern (e.g., `5 × 5` cells for a 5-point stencil); otherwise, all classes may conflict.

It is passed as an argument to the main function [`coloring`](@ref), but will only work if the associated `problem` has a `:column` or `:row` partition and a `:nonsymmetric` structure.

# Constructor

    QuotientColoringAlgorithm(classes, algo=GreedyColoringAlgorithm(; recoloring_iterations=30))

- `classes::AbstractVector{<:Integer}`: a class label for each column (for a `:column` partition) or row (for a `:row` partition). Columns (rows) of the same class must be structurally orthogonal, otherwise an `ArgumentError` is thrown.
- `algo::ADTypes.AbstractColoringAlgorithm`: the algorithm for the column coloring of the conflict pattern of the classes.

# Example

```jldoctest
julia> using SparseMatrixColorings, SparseArrays, LinearAlgebra

julia> m = 10;  # 5-point Laplacian on an m × m grid

julia> T = spdiagm(-1 => ones(m - 1), 0 => ones(m), 1 => ones(m - 1));

julia> A = kron(sparse(I, m, m), T) + kron(T, sparse(I, m, m));

julia> classes = [1 + mod(i, 5) + 5 * mod(j, 5) for j in 0:(m - 1) for i in 0:(m - 1)];

julia> problem = ColoringProblem(; structure=:nonsymmetric, partition=:column);

julia> ncolors(coloring(A, problem, GreedyColoringAlgorithm()))
7

julia> ncolors(coloring(A, problem, QuotientColoringAlgorithm(classes)))
5
```

# ADTypes coloring interface

`QuotientColoringAlgorithm` is a subtype of [`ADTypes.AbstractColoringAlgorithm`](@extref ADTypes.AbstractColoringAlgorithm), which means the following methods are also applicable:

- [`ADTypes.column_coloring`](@extref ADTypes.column_coloring)
- [`ADTypes.row_coloring`](@extref ADTypes.row_coloring)
"""
struct QuotientColoringAlgorithm{
    V<:AbstractVector{<:Integer},A<:ADTypes.AbstractColoringAlgorithm
} <: ADTypes.AbstractColoringAlgorithm
    classes::V
    algo::A
end

function QuotientColoringAlgorithm(classes::AbstractVector{<:Integer})
    return QuotientColoringAlgorithm(
        classes, GreedyColoringAlgorithm(; recoloring_iterations=30)
    )
end

"""
    quotient_pattern(A::AbstractMatrix, classes::AbstractVector{<:Integer})

Return a tuple `(Q, class_index)`, where `class_index[j]` is the index of the class of column `j` of `A`, and the `Bool` matrix `Q` is the conflict pattern of the classes: its columns are the classes, and its rows are the distinct sets of classes with a nonzero in a common row of `A`.

Throws an `ArgumentError` if two columns of the same class have a nonzero in a common row.
"""
function quotient_pattern(A::AbstractMatrix, classes::AbstractVector{<:Integer})
    S = SparseMatrixCSC{Bool,Int}(sparse(A) .!= 0)
    dropzeros!(S)
    n = size(S, 2)
    if length(classes) != n
        throw(
            DimensionMismatch(
                "`QuotientColoringAlgorithm` expected $n classes but got $(length(classes))"
            ),
        )
    end
    labels = unique(classes)
    index = Dict(c => k for (k, c) in enumerate(labels))
    class_index = [index[c] for c in classes]
    # Qᵀ[k, i] != 0: row `i` of `A` has a nonzero in a column of class `k`
    C = sparse(class_index, 1:n, true, length(labels), n)
    Qᵀ = SparseMatrixCSC{Bool,Int}(C * sparse(transpose(S)) .!= 0)
    if nnz(Qᵀ) != nnz(S)
        throw(
            ArgumentError(
                "`QuotientColoringAlgorithm`: two columns of the same class have a nonzero in a common row",
            ),
        )
    end
    # rows of `A` with the same set of classes add no conflicts
    rv = rowvals(Qᵀ)
    distinct_rows = unique(i -> view(rv, nzrange(Qᵀ, i)), axes(Qᵀ, 2))
    Q = sparse(transpose(Qᵀ[:, distinct_rows]))
    return Q, class_index
end

function ADTypes.column_coloring(A::AbstractMatrix, algo::QuotientColoringAlgorithm)
    Q, class_index = quotient_pattern(A, algo.classes)
    class_color = ADTypes.column_coloring(Q, algo.algo)
    return class_color[class_index]
end

function ADTypes.row_coloring(A::AbstractMatrix, algo::QuotientColoringAlgorithm)
    return ADTypes.column_coloring(transpose(A), algo)
end
