# SparseMatrixColorings.jl

[![Build Status](https://github.com/juliadiff/SparseMatrixColorings.jl/actions/workflows/Test.yml/badge.svg?branch=main)](https://github.com/juliadiff/SparseMatrixColorings.jl/actions/workflows/Test.yml?query=branch%3Amain)
[![GPU build status](https://badge.buildkite.com/7d8ed289d7bdb5a25ae48b2c778a202ce4990b7ee558cdfef8.svg?branch=main)](https://buildkite.com/julialang/sparsematrixcolorings-dot-jl)
[![Coverage](https://codecov.io/gh/juliadiff/SparseMatrixColorings.jl/branch/main/graph/badge.svg)](https://app.codecov.io/gh/juliadiff/SparseMatrixColorings.jl)

[![Stable Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://juliadiff.org/SparseMatrixColorings.jl/stable/)
[![Dev Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://juliadiff.org/SparseMatrixColorings.jl/dev/)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/JuliaDiff/BlueStyle)
[![arXiv](https://img.shields.io/badge/arXiv-2505.07308-b31b1b.svg)](https://arxiv.org/abs/2505.07308)
[![DOI](https://img.shields.io/badge/DOI-10.5281/zenodo.11314275-blue.svg)](https://zenodo.org/doi/10.5281/zenodo.11314275)
[![All Contributors](https://img.shields.io/github/all-contributors/JuliaDiff/SparseMatrixColorings.jl?color=ee8449)](#contributors)

Coloring algorithms for sparse Jacobian and Hessian matrices.

## Getting started

To install this package, run the following in a Julia Pkg REPL:

```julia
pkg> add SparseMatrixColorings
```

## Background

The algorithms implemented in this package are described in the following preprint:

- [_Revisiting Sparse Matrix Coloring and Bicoloring_](https://arxiv.org/abs/2505.07308), Montoison et al. (2025)

and inspired by previous works:

- [_What Color Is Your Jacobian? Graph Coloring for Computing Derivatives_](https://epubs.siam.org/doi/10.1137/S0036144504444711), Gebremedhin et al. (2005)
- [_New Acyclic and Star Coloring Algorithms with Application to Computing Hessians_](https://epubs.siam.org/doi/abs/10.1137/050639879), Gebremedhin et al. (2007)
- [_Efficient Computation of Sparse Hessians Using Coloring and Automatic Differentiation_](https://pubsonline.informs.org/doi/abs/10.1287/ijoc.1080.0286), Gebremedhin et al. (2009)
- [_ColPack: Software for graph coloring and related problems in scientific computing_](https://dl.acm.org/doi/10.1145/2513109.2513110), Gebremedhin et al. (2013)

Some parts of the articles (like definitions) are thus copied verbatim in the documentation.

## Alternatives

In Python:

- [pysparsematrixcolorings](https://github.com/gdalle/pysparsematrixcolorings): an experimental Python interface to the present package
- [asdex](https://github.com/adrhill/asdex): a Python-native sparse differentiation library, with coloring utilities

In Julia (unmaintained):

- [ColPack.jl](https://github.com/exanauts/ColPack.jl): a Julia interface to the C++ library [ColPack](https://github.com/CSCsw/ColPack)
- [SparseDiffTools.jl](https://github.com/JuliaDiff/SparseDiffTools.jl): contains older Julia implementations of some coloring algorithms

## Citing

Please cite this software using the provided `CITATION.cff` file or the `.bib` entry below:

```bibtex
@unpublished{montoison2025revisitingsparsematrixcoloring,
      title={Revisiting Sparse Matrix Coloring and Bicoloring}, 
      author={Alexis Montoison and Guillaume Dalle and Assefaw Gebremedhin},
      year={2025},
      eprint={2505.07308},
      archivePrefix={arXiv},
      primaryClass={math.NA},
      url={https://arxiv.org/abs/2505.07308}, 
}
```

The link <https://zenodo.org/doi/10.5281/zenodo.11314275> resolves to the latest version on Zenodo.

## Contributors

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/amontoison"><img src="https://avatars.githubusercontent.com/u/35051714?v=4?s=100" width="100px;" alt="Alexis Montoison"/><br /><sub><b>Alexis Montoison</b></sub></a><br /><a href="#ideas-amontoison" title="Ideas, Planning, & Feedback">🤔</a> <a href="#code-amontoison" title="Code">💻</a> <a href="#doc-amontoison" title="Documentation">📖</a> <a href="#maintenance-amontoison" title="Maintenance">🚧</a> <a href="#research-amontoison" title="Research">🔬</a> <a href="#review-amontoison" title="Reviewed Pull Requests">👀</a> <a href="#talk-amontoison" title="Talks">📢</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://gdalle.github.io/"><img src="https://avatars.githubusercontent.com/u/22795598?v=4?s=100" width="100px;" alt="Guillaume Dalle"/><br /><sub><b>Guillaume Dalle</b></sub></a><br /><a href="#ideas-gdalle" title="Ideas, Planning, & Feedback">🤔</a> <a href="#code-gdalle" title="Code">💻</a> <a href="#doc-gdalle" title="Documentation">📖</a> <a href="#maintenance-gdalle" title="Maintenance">🚧</a> <a href="#research-gdalle" title="Research">🔬</a> <a href="#review-gdalle" title="Reviewed Pull Requests">👀</a> <a href="#talk-gdalle" title="Talks">📢</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->
