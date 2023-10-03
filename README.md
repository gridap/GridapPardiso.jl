# GridapPardiso

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://gridap.github.io/GridapPardiso.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://gridap.github.io/GridapPardiso.jl/dev)
[![Build Status](https://github.com/gridap/GridapPardiso.jl/workflows/CI/badge.svg?branch=master)](https://github.com/gridap/GridapPardiso.jl/actions?query=workflow%3ACI) 
[![Codecov](https://codecov.io/gh/gridap/GridapPardiso.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/gridap/GridapPardiso.jl)

[Gridap](https://github.com/gridap/Gridap.jl) (Grid-based approximation of partial differential equations in Julia) plugin to use the [Intel Pardiso OneAPI MKL direct sparse solver](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html).

## Basic Usage

```julia
using Gridap
using GridapPardiso
A = sparse([1,2,3,4,5],[1,2,3,4,5],[1.0,2.0,3.0,4.0,5.0])
b = ones(A.n)
x = similar(b)
msglvl = 1
ps = PardisoSolver(mtype=GridapPardiso.MTYPE_REAL_NON_SYMMETRIC, msglvl=msglvl)
ss = symbolic_setup(ps, A)
ns = numerical_setup(ss, A)
solve!(x, ns, b)
```

## Usage in a Finite Element computation

```julia
using Gridap
using GridapPardiso
using SparseMatricesCSR

# Define the FE problem
# -Δu = x*y in (0,1)^3, u = 0 on the boundary.

model = CartesianDiscreteModel((0,1,0,1,0,1), (10,10,10))

order=1
reffe = ReferenceFE(lagrangian,Float64,order)
V = FESpace(model,
            reffe,
            conformity=:H1,
            dirichlet_tags="boundary")
U = TrialFESpace(V)

trian = get_triangulation(model)
dΩ    = Measure(trian,2)

a(u,v)=∫(∇(v)⋅∇(u))dΩ
f(x)=x[1]*x[2]
l(v)=∫(v*f)dΩ

assem = SparseMatrixAssembler(SparseMatrixCSR{1,Float64,Int},Vector{Float64},U,V)
op = AffineFEOperator(a,l,U,V,assem)

ls = PardisoSolver()
solver = LinearFESolver(ls)

uh = solve(solver,op)
```

## Installation

**GridPardiso** itself is installed when you add and use it into another project.

First, ensure that your system fulfills the requirements (see instructions below). Only after these steps, to include into your project from the Julia REPL, use the following commands:

```
pkg> add GridapPardiso
julia> using GridapPardiso
```

If, for any reason, you need to manually build the project (e.g., you added the project with the wrong environment resulting in a build that fails, you have fixed the environment and want to re-build the project), write down the following commands in Julia REPL:

```
pkg> add GridapPardiso
pkg> build GridPardiso
julia> using GridapPardiso
```

### Requirements

**GridapPardiso** requires the following software to be installed on your system:

1. Intel oneAPI MKL library. In particular, **GridapPardiso** relies on the 
   [Intel Pardiso oneAPI MKL direct sparse solver](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html).
2. GNU C compiler (`gcc`) + GNU `OpenMP` library (`libgomp`). 

In order to find 1., the build system of **GridapPardiso** relies on the `MKLROOT` environment variable. This variable must point to the MKL installation directory on your system. [Intel oneAPI MKL](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html) includes the `mklvars.sh` Unix shell script in order to set up appropriately this environment variable. Assuming that `/opt/intel/mkl/` is the Intel MKL installation directory on your system, you have to run this script using the following command (most preferably in a script that is executed automatically when a new shell is opened):

```
$ source /opt/intel/mkl/bin/mklvars.sh intel64
```

In order to find 2., there are two alternatives:

* The user may optionally set the `GRIDAP_PARDISO_LIBGOMP_DIR` environment variable. This variable must contain the absolute path to the folder in which the `libgomp` dynamic library file resides on your system.
* The build system tries to do its best to find `libgomp` on the system.

If `GRIDAP_PARDISO_LIBGOMP_DIR` is defined, then the build system follows the first alternative. If not, then it follows the second. Thus, the environment variable has precedence over the default behaviour of trying to find the library automatically.

In general, the user may let the build system to find `libgomp` in the first place. If the build system fails, or it finds an undesired version of `libgomp`, then the environment variable can be used as a fallback solution, e.g., for those systems with a non-standard installation of `libgomp`, and/or several simultaneous installations of `libgomp`. 

We note that, in Debian-based Linux OSs, the following commands can be installed in order to satisfy requirement 2. (typically executed as sudo):

```
$ apt-get update
$ apt-get install -y gcc libgomp1
```

In such systems, the build system is able to automatically find `libgomp`.

