using LinearAlgebra
using SparseMatrixColorings:
    structurally_orthogonal_columns,
    symmetrically_orthogonal_columns,
    structurally_biorthogonal,
    directly_recoverable_columns,
    substitutable_columns,
    substitutable_bidirectional,
    what_fig_41,
    efficient_fig_1
using Test

@testset "Structurally orthogonal columns" begin
    A = [
        1 0 0
        0 2 0
        0 3 4
    ]

    # success

    @test structurally_orthogonal_columns(A, [1, 2, 3])
    @test structurally_orthogonal_columns(A, [1, 2, 1])
    @test structurally_orthogonal_columns(A, [1, 1, 2])

    @test directly_recoverable_columns(A, [1, 2, 3])
    @test directly_recoverable_columns(A, [1, 2, 1])
    @test directly_recoverable_columns(A, [1, 1, 2])

    # failure

    @test !structurally_orthogonal_columns(A, [1, 2])
    log = (:warn, "2 colors provided for 3 columns.")
    @test_logs log structurally_orthogonal_columns(A, [1, 2]; verbose=true)

    @test !directly_recoverable_columns(A, [1, 2])
    log = (:warn, "2 colors provided for 3 columns.")
    @test_logs log !directly_recoverable_columns(A, [1, 2]; verbose=true)

    @test !structurally_orthogonal_columns(A, [1, 2, 2])
    log = (:warn, "In color 2, columns [2, 3] all have nonzeros in row 3.")
    @test_logs log structurally_orthogonal_columns(A, [1, 2, 2]; verbose=true)

    @test !directly_recoverable_columns(A, [1, 2, 2])
    log = (:warn, "Coefficients [3, 4] are not directly recoverable.")
end

@testset "Structurally orthogonal rows" begin
    A = [
        1 0 0
        0 2 0
        0 3 4
    ]

    # success

    @test structurally_orthogonal_columns(transpose(A), [1, 2, 3])
    @test structurally_orthogonal_columns(transpose(A), [1, 2, 1])
    @test structurally_orthogonal_columns(transpose(A), [1, 1, 2])

    @test directly_recoverable_columns(transpose(A), [1, 2, 3])
    @test directly_recoverable_columns(transpose(A), [1, 2, 1])
    @test directly_recoverable_columns(transpose(A), [1, 1, 2])

    # failure

    @test !structurally_orthogonal_columns(transpose(A), [1, 2, 2, 3])
    log = (:warn, "4 colors provided for 3 columns.")
    @test_logs log structurally_orthogonal_columns(transpose(A), [1, 2, 2, 3]; verbose=true)

    @test !directly_recoverable_columns(transpose(A), [1, 2, 2, 3])
    log = (:warn, "4 colors provided for 3 columns.")
    @test_logs log directly_recoverable_columns(transpose(A), [1, 2, 2, 3]; verbose=true)

    @test !structurally_orthogonal_columns(transpose(A), [1, 2, 2])
    log = (:warn, "In color 2, columns [2, 3] all have nonzeros in row 2.")
    @test_logs log !structurally_orthogonal_columns(transpose(A), [1, 2, 2]; verbose=true)

    @test !directly_recoverable_columns(transpose(A), [1, 2, 2])
    log = (:warn, "Coefficients [2, 3] are not directly recoverable.")
    @test_logs log directly_recoverable_columns(transpose(A), [1, 2, 2]; verbose=true)
end

@testset "Symmetrically orthogonal" begin
    A = what_fig_41().A
    @test issymmetric(A)

    # success

    @test symmetrically_orthogonal_columns(A, [1, 2, 1, 3, 1, 1])

    @test directly_recoverable_columns(A, [1, 2, 1, 3, 1, 1])

    # failure

    @test !symmetrically_orthogonal_columns(A, [1, 2, 1, 3, 1])
    @test_logs (:warn, "5 colors provided for 6 columns.") symmetrically_orthogonal_columns(
        A, [1, 2, 1, 3, 1]; verbose=true
    )

    @test !symmetrically_orthogonal_columns(A, [1, 3, 1, 3, 1, 1])
    @test_logs (
        :warn,
        """
For coefficient (i=2, j=3) with colors (ci=3, cj=1):
- In row color ci=3, rows [2, 4] all have nonzeros in column j=3.
- In column color cj=1, columns [1, 3, 5, 6] all have nonzeros in row i=2.
""",
    ) symmetrically_orthogonal_columns(A, [1, 3, 1, 3, 1, 1]; verbose=true)

    A = efficient_fig_1().A
    @test issymmetric(A)

    # success

    @test symmetrically_orthogonal_columns(A, [1, 2, 1, 3, 1, 4, 3, 5, 1, 2])

    # failure

    @test !symmetrically_orthogonal_columns(A, [1, 2, 1, 3, 1, 4, 3, 4, 1, 2])
    @test !symmetrically_orthogonal_columns(A, [1, 2, 1, 3, 1, 4, 2, 5, 1, 2])
    @test !symmetrically_orthogonal_columns(A, [1, 2, 1, 4, 1, 4, 3, 5, 1, 2])

    @test !directly_recoverable_columns(A, [1, 2, 1, 3, 1, 4, 3, 4, 1, 2])
    @test !directly_recoverable_columns(A, [1, 2, 1, 3, 1, 4, 2, 5, 1, 2])
    @test !directly_recoverable_columns(A, [1, 2, 1, 4, 1, 4, 3, 5, 1, 2])
end

@testset "Structurally biorthogonal" begin
    A = [
        1 5 7 9 11
        2 0 0 0 12
        3 0 0 0 13
        4 6 8 10 14
    ]

    # success

    @test structurally_biorthogonal(A, [1, 2, 2, 3], [1, 2, 2, 2, 3])

    # failure

    @test !structurally_biorthogonal(A, [1, 2, 2, 3], [1, 2, 2, 2])
    @test !structurally_biorthogonal(A, [1, 2, 2, 3, 4], [1, 2, 2, 2, 3])
    @test !structurally_biorthogonal(A, [1, 1, 1, 2], [1, 1, 1, 1, 2])

    @test_logs (:warn, "4 colors provided for 5 columns.") !structurally_biorthogonal(
        A, [1, 2, 2, 3], [1, 2, 2, 2]; verbose=true
    )
    @test_logs (:warn, "5 colors provided for 4 rows.") !structurally_biorthogonal(
        A, [1, 2, 2, 3, 4], [1, 2, 2, 2, 3]; verbose=true
    )
    @test_logs (
        :warn,
        """
For coefficient (i=1, j=1) with colors (ci=1, cj=1):
- In row color ci=1, rows [1, 2, 3] all have nonzeros in column j=1.
- In column color cj=1, columns [1, 2, 3, 4] all have nonzeros in row i=1.
""",
    ) !structurally_biorthogonal(A, [1, 1, 1, 2], [1, 1, 1, 1, 2]; verbose=true)

    @test_logs (
        :warn,
        """
For coefficient (i=1, j=2) with colors (ci=0, cj=2):
- Row color ci=0 is neutral.
- In column color cj=2, columns [2, 3, 4] all have nonzeros in row i=1.
""",
    ) structurally_biorthogonal(A, [0, 2, 2, 3], [1, 2, 2, 2, 3], verbose=true)

    @test_logs (
        :warn,
        """
For coefficient (i=2, j=1) with colors (ci=2, cj=0):
- In row color ci=2, rows [2, 3] all have nonzeros in column j=1.
- Column color cj=0 is neutral.
""",
    ) structurally_biorthogonal(A, [1, 2, 2, 3], [0, 2, 2, 2, 3], verbose=true)

    @test_logs (
        :warn,
        """
For coefficient (i=1, j=1) with colors (ci=0, cj=0):
- Row color ci=0 is neutral.
- Column color cj=0 is neutral.
""",
    ) structurally_biorthogonal(A, [0, 2, 2, 3], [0, 2, 2, 2, 3], verbose=true)
end

@testset "Substitutable columns" begin
    A1 = [
        1 1 1 1 1
        1 1 0 0 0
        1 0 1 0 0
        1 0 0 1 0
        1 0 0 0 1
    ]
    B1 = [
        1 6 7 8 9
        6 2 0 0 0
        7 0 3 0 0
        8 0 0 4 0
        9 0 0 0 5
    ]
    A2 = [
        1 1 0 0 0
        1 1 1 0 0
        0 1 1 1 0
        0 0 1 1 1
        0 0 0 1 1
    ]
    B2 = [
        5 1 0 0 0
        1 6 2 0 0
        0 2 7 3 0
        0 0 3 8 4
        0 0 0 4 9
    ]
    A3 = [
        0 1 1 1 1
        1 0 1 1 1
        1 1 0 1 1
        1 1 1 0 1
        1 1 1 1 0
    ]
    B3 = [
        0 1 2 3 4
        1 0 5 6 7
        2 5 0 8 9
        3 6 8 0 10
        4 7 9 10 0
    ]

    # success

    substitutable_columns(A1, B1, [1, 2, 2, 2, 2])
    substitutable_columns(A2, B2, [1, 2, 3, 1, 2])
    substitutable_columns(A3, B3, [1, 2, 3, 4, 0])

    # failure

    @test !substitutable_columns(A1, B1, [1, 1, 1, 1])
    log = (:warn, "4 colors provided for 5 columns.")
    @test_logs log substitutable_columns(A1, B1, [1, 1, 1, 1]; verbose=true)

    @test !substitutable_columns(A1, B1, [1, 1, 1, 1, 1])
    @test_logs (
        :warn,
        """
For coefficient (i=1, j=1) with colors (ci=1, cj=1):
- For the row 5 in row color ci=1, A[5, 1] is ordered after A[1, 1].
- For the column 5 in column color cj=1, A[1, 5] is ordered after A[1, 1].
""",
    ) substitutable_columns(A1, B1, [1, 1, 1, 1, 1]; verbose=true)

    @test !substitutable_columns(A2, B2, [1, 2, 0, 1, 2])
    @test_logs (
        :warn,
        """
For coefficient (i=3, j=3) with colors (ci=0, cj=0):
- Row color ci=0 is neutral.
- Column color cj=0 is neutral.
""",
    ) substitutable_columns(A2, B2, [1, 2, 0, 1, 2]; verbose=true)

    @test !substitutable_columns(A3, B3, [0, 1, 2, 3, 3])
    @test_logs (
        :warn,
        """
For coefficient (i=1, j=4) with colors (ci=0, cj=3):
- Row color ci=0 is neutral.
- For the column 5 in column color cj=3, A[1, 5] is ordered after A[1, 4].
""",
    ) substitutable_columns(A3, B3, [0, 1, 2, 3, 3]; verbose=true)

    @test !substitutable_columns(A3, B3, [1, 2, 3, 3, 0])
    @test_logs (
        :warn,
        """
For coefficient (i=3, j=5) with colors (ci=3, cj=0):
- For the row 4 in row color ci=3, A[4, 5] is ordered after A[3, 5].
- Column color cj=0 is neutral.
""",
    ) substitutable_columns(A3, B3, [1, 2, 3, 3, 0]; verbose=true)
end

@testset "Substitutable bidirectional" begin
    A = [
        1 0 0
        0 1 0
        0 0 1
    ]
    B = [
        1 0 0
        0 2 0
        0 0 3
    ]

    # success

    substitutable_bidirectional(A, B, [1, 0, 0], [0, 1, 1])

    # failure

    log = (:warn, "2 colors provided for 3 columns.")
    @test_logs log !substitutable_bidirectional(A, B, [1, 0, 0], [0, 1]; verbose=true)

    log = (:warn, "4 colors provided for 3 rows.")
    @test_logs log !substitutable_bidirectional(A, B, [1, 0, 0, 1], [0, 1, 1]; verbose=true)
end
