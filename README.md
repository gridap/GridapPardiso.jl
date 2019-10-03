# GridapPardiso

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://gridap.github.io/GridapPardiso.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://gridap.github.io/GridapPardiso.jl/dev)
[![Build Status](https://travis-ci.com/gridap/GridapPardiso.jl.svg?branch=master)](https://travis-ci.com/gridap/GridapPardiso.jl)
[![Codecov](https://codecov.io/gh/gridap/GridapPardiso.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/gridap/GridapPardiso.jl)

[Gridap](https://github.com/gridap/Gridap.jl) (Grid-based approximation of partial differential equations in Julia) plugin to use the [Intel Pardiso MKL direct sparse solver](https://software.intel.com/en-us/mkl-developer-reference-fortran-intel-mkl-pardiso-parallel-direct-sparse-solver-interface).

## Usage

```julia
using Gridap
using GridapPardiso
A = sparse([1,2,3,4,5],[1,2,3,4,5],[1.0,2.0,3.0,4.0,5.0])
b = ones(A.n)
x = similar(b)
msglvl = 1
ps = PardisoSolver(GridapPardiso.MTYPE_REAL_NON_SYMMETRIC, new_iparm(A), msglvl)
ss = symbolic_setup(ps, A)
ns = numerical_setup(ss, A)
solve!(x, ns, b)
```

## Notes

Currently **GridapPardiso** only works with `SparseMatrixCSC` matrices. Any other `AbstractMatrix{Float64}` matrix is converted to `SparseMatrixCSC{Float64,Integer}`.
