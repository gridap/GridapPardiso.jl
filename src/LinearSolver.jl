"""
Wrapper of the MKL Pardiso solver available in julia
"""
struct PardisoSolver <: LinearSolver
  mtype  :: Int
  iparm  :: Vector{Int32} 
  msglvl :: Int
  pt     :: Vector{Int}
end

PardisoSolver() = PardisoSolver(11, new_iparm(), 0, new_pardiso_handle())
PardisoSolver(mtype) = PardisoSolver(mtype, new_iparm(), 0, new_pardiso_handle())
PardisoSolver(mtype, iparm) = PardisoSolver(mtype, iparm, 0, new_pardiso_handle())
PardisoSolver(mtype, iparm, msglvl) = PardisoSolver(mtype, iparm, msglvl, new_pardiso_handle())

mutable struct PardisoSymbolicSetup{T} <: SymbolicSetup 
    phase  :: Int 
    mat    :: AbstractMatrix{T}
    solver :: PardisoSolver
end

mutable struct PardisoNumericalSetup{T} <: NumericalSetup
    phase  :: Int 
    mat    :: AbstractMatrix{T}
    solver :: PardisoSolver
end

symbolic_setup(ps::PardisoSolver, mat::AbstractMatrix) = symbolic_setup(ps, sparse(mat))

function symbolic_setup(ps::PardisoSolver, mat::SparseMatrixCSC{T}) where {T}

    maxfct = 1
    mnum = 1
    nrhs = 1
    phase = 11 

    pss = PardisoSymbolicSetup(phase, mat, ps)

    pardisoinit!(ps.pt, ps.mtype, ps.iparm)

    err = pardiso!( pss.solver.pt,                # Handle to internal data structure. The entries must be set to zero prior to the first call to pardiso
                    maxfct,                       # Maximum number of factors with identical sparsity structure that must be kept in memory at the same time
                    mnum,                         # Actual matrix for the solution phase. The value must be: 1 <= mnum <= maxfct. 
                    pss.solver.mtype,             # Defines the matrix type, which influences the pivoting method
                    pss.phase,                    # Controls the execution of the solver (11 == Analysis)
                    mat.n,                        # Number of equations in the sparse linear systems of equations
                    mat.nzval,                    # Contains the non-zero elements of the coefficient matrix A corresponding to the indices in ja
                    Vector{Int32}(mat.colptr),    # Pointers to columns in CSR format
                    Vector{Int32}(mat.rowval),    # Column indices of the CSR sparse matrix
                    Vector{Int32}(),              # Permutation vector 
                    nrhs,                         # Number of right-hand sides that need to be solved for
                    pss.solver.iparm,             # This array is used to pass various parameters to Intel MKL PARDISO 
                    pss.solver.msglvl,            # Message level information
                    Vector{T}(),                  # Array, size (n, nrhs). On entry, contains the right-hand side vector/matrix
                    Vector{T}())                  # Array, size (n, nrhs). If iparm(6)=0 it contains solution vector/matrix X
    @assert err == 0
    return finalizer(symbolic_setup_finalize, pss)
end

function symbolic_setup_finalize(pss::PardisoSymbolicSetup{T}) where {T}

    maxfct = 1
    mnum = 1
    nrhs = 1
    pss.phase = -1

    err = pardiso!( pss.solver.pt,                # Handle to internal data structure. The entries must be set to zero prior to the first call to pardiso
                    maxfct,                       # Maximum number of factors with identical sparsity structure that must be kept in memory at the same time
                    mnum,                         # Actual matrix for the solution phase. The value must be: 1 <= mnum <= maxfct. 
                    pss.solver.mtype,             # Defines the matrix type, which influences the pivoting method
                    pss.phase,                    # Controls the execution of the solver (11 == Analysis)
                    pss.mat.n,                    # Number of equations in the sparse linear systems of equations
                    Vector{T}(),                  # Contains the non-zero elements of the coefficient matrix A corresponding to the indices in ja
                    Vector{Int32}(),              # Pointers to columns in CSR format
                    Vector{Int32}(),              # Column indices of the CSR sparse matrix
                    Vector{Int32}(),              # Permutation vector 
                    nrhs,                         # Number of right-hand sides that need to be solved for
                    pss.solver.iparm,             # This array is used to pass various parameters to Intel MKL PARDISO 
                    pss.solver.msglvl,            # Message level information
                    Vector{T}(),                  # Array, size (n, nrhs). On entry, contains the right-hand side vector/matrix
                    Vector{T}())                  # Array, size (n, nrhs). If iparm(6)=0 it contains solution vector/matrix X
    @assert err == 0
end

numerical_setup(pss::PardisoSymbolicSetup, mat::AbstractMatrix) = numerical_setup(pss, sparse(mat))

function numerical_setup(pss::PardisoSymbolicSetup, mat::SparseMatrixCSC{T}) where {T}

    maxfct = 1
    mnum = 1
    nrhs = 1
    phase = 22 

    pns = PardisoNumericalSetup(phase, mat, pss.solver)

    err = pardiso!( pns.solver.pt,                # Handle to internal data structure. The entries must be set to zero prior to the first call to pardiso
                    maxfct,                       # Maximum number of factors with identical sparsity structure that must be kept in memory at the same time
                    mnum,                         # Actual matrix for the solution phase. The value must be: 1 <= mnum <= maxfct. 
                    pns.solver.mtype,             # Defines the matrix type, which influences the pivoting method
                    phase,                        # Controls the execution of the solver (11 == Analysis)
                    mat.n,                        # Number of equations in the sparse linear systems of equations
                    Vector{T}(),                  # Contains the non-zero elements of the coefficient matrix A corresponding to the indices in ja
                    Vector{Int32}(mat.colptr),    # Pointers to columns in CSR format
                    Vector{Int32}(mat.rowval),    # Column indices of the CSR sparse matrix
                    Vector{Int32}(),              # Permutation vector 
                    nrhs,                         # Number of right-hand sides that need to be solved for
                    pns.solver.iparm,             # This array is used to pass various parameters to Intel MKL PARDISO 
                    pns.solver.msglvl,            # Message level information
                    Vector{T}(),                  # Array, size (n, nrhs). On entry, contains the right-hand side vector/matrix
                    Vector{T}())                  # Array, size (n, nrhs). If iparm(6)=0 it contains solution vector/matrix X
    return finalizer(numerical_setup_finalize, pns)
end

function numerical_setup_finalize(pns::PardisoNumericalSetup{T}) where {T}

    maxfct = 1
    mnum = 1
    nrhs = 1
    pns.phase = 0 

    err = pardiso!( pns.solver.pt,                # Handle to internal data structure. The entries must be set to zero prior to the first call to pardiso
                    maxfct,                       # Maximum number of factors with identical sparsity structure that must be kept in memory at the same time
                    mnum,                         # Actual matrix for the solution phase. The value must be: 1 <= mnum <= maxfct. 
                    pns.solver.mtype,             # Defines the matrix type, which influences the pivoting method
                    pns.phase,                    # Controls the execution of the solver (11 == Analysis)
                    pns.mat.n,                    # Number of equations in the sparse linear systems of equations
                    Vector{T}(),                  # Contains the non-zero elements of the coefficient matrix A corresponding to the indices in ja
                    Vector{Int32}(),              # Pointers to columns in CSR format
                    Vector{Int32}(),              # Column indices of the CSR sparse matrix
                    Vector{Int32}(),              # Permutation vector 
                    nrhs,                         # Number of right-hand sides that need to be solved for
                    pns.solver.iparm,             # This array is used to pass various parameters to Intel MKL PARDISO 
                    pns.solver.msglvl,            # Message level information
                    Vector{T}(),                  # Array, size (n, nrhs). On entry, contains the right-hand side vector/matrix
                    Vector{T}())                  # Array, size (n, nrhs). If iparm(6)=0 it contains solution vector/matrix X
    @assert err == 0
end

function solve!(x::AbstractVector{T}, ns::PardisoNumericalSetup{T}, b::AbstractVector{T}) where {T}

    maxfct = 1
    mnum = 1
    nrhs = 1
    phase = 33

    @assert ns.phase != 0

    err = pardiso!( ns.solver.pt,                 # Handle to internal data structure. The entries must be set to zero prior to the first call to pardiso
                    maxfct,                       # Maximum number of factors with identical sparsity structure that must be kept in memory at the same time
                    mnum,                         # Actual matrix for the solution phase. The value must be: 1 <= mnum <= maxfct. 
                    ns.solver.mtype,              # Defines the matrix type, which influences the pivoting method
                    phase,                        # Controls the execution of the solver (11 == Analysis)
                    ns.mat.n,                     # Number of equations in the sparse linear systems of equations
                    ns.mat.nzval,                 # Contains the non-zero elements of the coefficient matrix A corresponding to the indices in ja
                    Vector{Int32}(ns.mat.colptr), # Pointers to columns in CSR format
                    Vector{Int32}(ns.mat.rowval), # Column indices of the CSR sparse matrix
                    Vector{Int32}(),              # Permutation vector 
                    nrhs,                         # Number of right-hand sides that need to be solved for
                    ns.solver.iparm,              # This array is used to pass various parameters to Intel MKL PARDISO 
                    ns.solver.msglvl,             # Message level information
                    b,                            # Array, size (n, nrhs). On entry, contains the right-hand side vector/matrix
                    x)                            # Array, size (n, nrhs). If iparm(6)=0 it contains solution vector/matrix X
    @assert err == 0
end

