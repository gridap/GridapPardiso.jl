"""
Wrapper of the MKL Pardiso solver available in julia
"""
struct PardisoSolver <: LinearSolver
  mtype  :: Int
  iparm  :: Vector{Int32} 
  msglvl :: Int
  pt     :: Vector{Int}
end

PardisoSolver() = PardisoSolver(MatrixTypes["Real_NonSymmetric"], new_iparm(), 0, new_pardiso_handle())
PardisoSolver(mtype) = PardisoSolver(mtype, new_iparm(), 0, new_pardiso_handle())
PardisoSolver(mtype, iparm) = PardisoSolver(mtype, iparm, 0, new_pardiso_handle())
PardisoSolver(mtype, iparm, msglvl) = PardisoSolver(mtype, iparm, msglvl, new_pardiso_handle())

mutable struct PardisoSymbolicSetup{T,Ti} <: SymbolicSetup 
    phase  :: Int 
    mat    :: SparseMatrixCSC{T,Ti}
    solver :: PardisoSolver
end

mutable struct PardisoNumericalSetup{T,Ti} <: NumericalSetup
    phase  :: Int
    mat    :: SparseMatrixCSC{T,Ti}
    solver :: PardisoSolver
end

symbolic_setup(ps::PardisoSolver, mat::AbstractMatrix) = symbolic_setup(ps, sparse(mat))

function symbolic_setup(ps::PardisoSolver, mat::SparseMatrixCSC{T,Ti}) where {T<:Float64,Ti<:Int32}

    maxfct = 1
    mnum   = 1
    nrhs   = 1
    phase  = Phases["Analysis"]

    pss = PardisoSymbolicSetup(phase, mat, ps)

    pardisoinit!(ps.pt, ps.mtype, ps.iparm)

    err = pardiso!( pss.solver.pt,                # Handle to internal data structure. The entries must be set to zero prior to the first call to pardiso
                    maxfct,                       # Maximum number of factors with identical sparsity structure that must be kept in memory at the same time
                    mnum,                         # Actual matrix for the solution phase. The value must be: 1 <= mnum <= maxfct. 
                    pss.solver.mtype,             # Defines the matrix type, which influences the pivoting method
                    pss.phase,                    # Controls the execution of the solver (11 == Analysis)
                    mat.n,                        # Number of equations in the sparse linear systems of equations
                    mat.nzval,                    # Contains the non-zero elements of the coefficient matrix A corresponding to the indices in ja
                    mat.colptr,                   # Pointers to columns in CSR format
                    mat.rowval,                   # Column indices of the CSR sparse matrix
                    Vector{Ti}(),                 # Permutation vector 
                    nrhs,                         # Number of right-hand sides that need to be solved for
                    pss.solver.iparm,             # This array is used to pass various parameters to Intel MKL PARDISO 
                    pss.solver.msglvl,            # Message level information
                    Vector{T}(),                  # Array, size (n, nrhs). On entry, contains the right-hand side vector/matrix
                    Vector{T}())                  # Array, size (n, nrhs). If iparm(6)=0 it contains solution vector/matrix X
    pardiso_report_error(err)
    return finalizer(symbolic_setup_finalize, pss)
end

function symbolic_setup_finalize(pss::PardisoSymbolicSetup{T,Ti}) where {T,Ti<:Int32}

    maxfct    = 1
    mnum      = 1
    nrhs      = 1
    pss.phase = Phases["ReleaseAllInternalMemory"]

    err = pardiso!( pss.solver.pt,                # Handle to internal data structure. The entries must be set to zero prior to the first call to pardiso
                    maxfct,                       # Maximum number of factors with identical sparsity structure that must be kept in memory at the same time
                    mnum,                         # Actual matrix for the solution phase. The value must be: 1 <= mnum <= maxfct. 
                    pss.solver.mtype,             # Defines the matrix type, which influences the pivoting method
                    pss.phase,                    # Controls the execution of the solver (11 == Analysis)
                    pss.mat.n,                    # Number of equations in the sparse linear systems of equations
                    Vector{T}(),                  # Contains the non-zero elements of the coefficient matrix A corresponding to the indices in ja
                    Vector{Ti}(),                 # Pointers to columns in CSR format
                    Vector{Ti}(),                 # Column indices of the CSR sparse matrix
                    Vector{Ti}(),                 # Permutation vector 
                    nrhs,                         # Number of right-hand sides that need to be solved for
                    pss.solver.iparm,             # This array is used to pass various parameters to Intel MKL PARDISO 
                    pss.solver.msglvl,            # Message level information
                    Vector{T}(),                  # Array, size (n, nrhs). On entry, contains the right-hand side vector/matrix
                    Vector{T}())                  # Array, size (n, nrhs). If iparm(6)=0 it contains solution vector/matrix X
    pardiso_report_error(err)
end

function symbolic_setup(ps::PardisoSolver, mat::SparseMatrixCSC{T,Ti}) where {T<:Float64,Ti<:Int64}

    maxfct = 1
    mnum   = 1
    nrhs   = 1
    phase  = Phases["Analysis"]

    pss = PardisoSymbolicSetup(phase, mat, ps)

    pardisoinit!(ps.pt, ps.mtype, ps.iparm)

    err = pardiso_64!( pss.solver.pt,             # Handle to internal data structure. The entries must be set to zero prior to the first call to pardiso
                    maxfct,                       # Maximum number of factors with identical sparsity structure that must be kept in memory at the same time
                    mnum,                         # Actual matrix for the solution phase. The value must be: 1 <= mnum <= maxfct. 
                    pss.solver.mtype,             # Defines the matrix type, which influences the pivoting method
                    pss.phase,                    # Controls the execution of the solver (11 == Analysis)
                    mat.n,                        # Number of equations in the sparse linear systems of equations
                    mat.nzval,                    # Contains the non-zero elements of the coefficient matrix A corresponding to the indices in ja
                    mat.colptr,                   # Pointers to columns in CSR format
                    mat.rowval,                   # Column indices of the CSR sparse matrix
                    Vector{Ti}(),                 # Permutation vector 
                    nrhs,                         # Number of right-hand sides that need to be solved for
                    pss.solver.iparm,             # This array is used to pass various parameters to Intel MKL PARDISO 
                    pss.solver.msglvl,            # Message level information
                    Vector{T}(),                  # Array, size (n, nrhs). On entry, contains the right-hand side vector/matrix
                    Vector{T}())                  # Array, size (n, nrhs). If iparm(6)=0 it contains solution vector/matrix X
    pardiso_report_error(err)
    return finalizer(symbolic_setup_finalize, pss)
end

function symbolic_setup_finalize(pss::PardisoSymbolicSetup{T,Ti}) where {T,Ti<:Int64}

    maxfct    = 1
    mnum      = 1
    nrhs      = 1
    pss.phase = Phases["ReleaseAllInternalMemory"]

    err = pardiso_64!( pss.solver.pt,             # Handle to internal data structure. The entries must be set to zero prior to the first call to pardiso
                    maxfct,                       # Maximum number of factors with identical sparsity structure that must be kept in memory at the same time
                    mnum,                         # Actual matrix for the solution phase. The value must be: 1 <= mnum <= maxfct. 
                    pss.solver.mtype,             # Defines the matrix type, which influences the pivoting method
                    pss.phase,                    # Controls the execution of the solver (11 == Analysis)
                    pss.mat.n,                    # Number of equations in the sparse linear systems of equations
                    Vector{T}(),                  # Contains the non-zero elements of the coefficient matrix A corresponding to the indices in ja
                    Vector{Ti}(),                 # Pointers to columns in CSR format
                    Vector{Ti}(),                 # Column indices of the CSR sparse matrix
                    Vector{Ti}(),                 # Permutation vector 
                    nrhs,                         # Number of right-hand sides that need to be solved for
                    pss.solver.iparm,             # This array is used to pass various parameters to Intel MKL PARDISO 
                    pss.solver.msglvl,            # Message level information
                    Vector{T}(),                  # Array, size (n, nrhs). On entry, contains the right-hand side vector/matrix
                    Vector{T}())                  # Array, size (n, nrhs). If iparm(6)=0 it contains solution vector/matrix X
    pardiso_report_error(err)
end

numerical_setup(pss::PardisoSymbolicSetup, mat::AbstractMatrix) = numerical_setup(pss, sparse(mat))

function numerical_setup(pss::PardisoSymbolicSetup, mat::SparseMatrixCSC{T,Ti}) where {T<:Float64,Ti<:Int32}

    maxfct = 1
    mnum   = 1
    nrhs   = 1
    phase  = Phases["NumericalFactorization"]


    pns = PardisoNumericalSetup(phase, mat, pss.solver)

    err = pardiso!( pns.solver.pt,                # Handle to internal data structure. The entries must be set to zero prior to the first call to pardiso
                    maxfct,                       # Maximum number of factors with identical sparsity structure that must be kept in memory at the same time
                    mnum,                         # Actual matrix for the solution phase. The value must be: 1 <= mnum <= maxfct. 
                    pns.solver.mtype,             # Defines the matrix type, which influences the pivoting method
                    phase,                        # Controls the execution of the solver (11 == Analysis)
                    mat.n,                        # Number of equations in the sparse linear systems of equations
                    Vector{T}(),                  # Contains the non-zero elements of the coefficient matrix A corresponding to the indices in ja
                    mat.colptr,                   # Pointers to columns in CSR format
                    mat.rowval,                   # Column indices of the CSR sparse matrix
                    Vector{Ti}(),                 # Permutation vector 
                    nrhs,                         # Number of right-hand sides that need to be solved for
                    pns.solver.iparm,             # This array is used to pass various parameters to Intel MKL PARDISO 
                    pns.solver.msglvl,            # Message level information
                    Vector{T}(),                  # Array, size (n, nrhs). On entry, contains the right-hand side vector/matrix
                    Vector{T}())                  # Array, size (n, nrhs). If iparm(6)=0 it contains solution vector/matrix X
    pardiso_report_error(err)
    return finalizer(numerical_setup_finalize, pns)
end

function numerical_setup(pss::PardisoSymbolicSetup, mat::SparseMatrixCSC{T,Ti}) where {T<:Float64,Ti<:Int64}

    maxfct = 1
    mnum   = 1
    nrhs   = 1
    phase  = Phases["NumericalFactorization"]


    pns = PardisoNumericalSetup(phase, mat, pss.solver)

    err = pardiso_64!( pns.solver.pt,                # Handle to internal data structure. The entries must be set to zero prior to the first call to pardiso
                    maxfct,                       # Maximum number of factors with identical sparsity structure that must be kept in memory at the same time
                    mnum,                         # Actual matrix for the solution phase. The value must be: 1 <= mnum <= maxfct. 
                    pns.solver.mtype,             # Defines the matrix type, which influences the pivoting method
                    phase,                        # Controls the execution of the solver (11 == Analysis)
                    mat.n,                        # Number of equations in the sparse linear systems of equations
                    Vector{T}(),                  # Contains the non-zero elements of the coefficient matrix A corresponding to the indices in ja
                    mat.colptr,                   # Pointers to columns in CSR format
                    mat.rowval,                   # Column indices of the CSR sparse matrix
                    Vector{Ti}(),                 # Permutation vector 
                    nrhs,                         # Number of right-hand sides that need to be solved for
                    pns.solver.iparm,             # This array is used to pass various parameters to Intel MKL PARDISO 
                    pns.solver.msglvl,            # Message level information
                    Vector{T}(),                  # Array, size (n, nrhs). On entry, contains the right-hand side vector/matrix
                    Vector{T}())                  # Array, size (n, nrhs). If iparm(6)=0 it contains solution vector/matrix X
    pardiso_report_error(err)
    return finalizer(numerical_setup_finalize, pns)
end

function numerical_setup_finalize(pns::PardisoNumericalSetup{T,Ti}) where {T, Ti<:Int32}

    maxfct    = 1
    mnum      = 1
    nrhs      = 1
    pns.phase = Phases["ReleaseInternalMemory"]

    err = pardiso!( pns.solver.pt,                # Handle to internal data structure. The entries must be set to zero prior to the first call to pardiso
                    maxfct,                       # Maximum number of factors with identical sparsity structure that must be kept in memory at the same time
                    mnum,                         # Actual matrix for the solution phase. The value must be: 1 <= mnum <= maxfct. 
                    pns.solver.mtype,             # Defines the matrix type, which influences the pivoting method
                    pns.phase,                    # Controls the execution of the solver (11 == Analysis)
                    pns.mat.n,                    # Number of equations in the sparse linear systems of equations
                    Vector{T}(),                  # Contains the non-zero elements of the coefficient matrix A corresponding to the indices in ja
                    Vector{Ti}(),                 # Pointers to columns in CSR format
                    Vector{Ti}(),                 # Column indices of the CSR sparse matrix
                    Vector{Ti}(),                 # Permutation vector 
                    nrhs,                         # Number of right-hand sides that need to be solved for
                    pns.solver.iparm,             # This array is used to pass various parameters to Intel MKL PARDISO 
                    pns.solver.msglvl,            # Message level information
                    Vector{T}(),                  # Array, size (n, nrhs). On entry, contains the right-hand side vector/matrix
                    Vector{T}())                  # Array, size (n, nrhs). If iparm(6)=0 it contains solution vector/matrix X
    pardiso_report_error(err)
end

function numerical_setup_finalize(pns::PardisoNumericalSetup{T,Ti}) where {T, Ti<:Int64}

    maxfct    = 1
    mnum      = 1
    nrhs      = 1
    pns.phase = Phases["ReleaseInternalMemory"]

    err = pardiso_64!( pns.solver.pt,             # Handle to internal data structure. The entries must be set to zero prior to the first call to pardiso
                    maxfct,                       # Maximum number of factors with identical sparsity structure that must be kept in memory at the same time
                    mnum,                         # Actual matrix for the solution phase. The value must be: 1 <= mnum <= maxfct. 
                    pns.solver.mtype,             # Defines the matrix type, which influences the pivoting method
                    pns.phase,                    # Controls the execution of the solver (11 == Analysis)
                    pns.mat.n,                    # Number of equations in the sparse linear systems of equations
                    Vector{T}(),                  # Contains the non-zero elements of the coefficient matrix A corresponding to the indices in ja
                    Vector{Ti}(),                 # Pointers to columns in CSR format
                    Vector{Ti}(),                 # Column indices of the CSR sparse matrix
                    Vector{Ti}(),                 # Permutation vector 
                    nrhs,                         # Number of right-hand sides that need to be solved for
                    pns.solver.iparm,             # This array is used to pass various parameters to Intel MKL PARDISO 
                    pns.solver.msglvl,            # Message level information
                    Vector{T}(),                  # Array, size (n, nrhs). On entry, contains the right-hand side vector/matrix
                    Vector{T}())                  # Array, size (n, nrhs). If iparm(6)=0 it contains solution vector/matrix X
    pardiso_report_error(err)
end

function solve!(x::AbstractVector{T}, ns::PardisoNumericalSetup{T,Ti}, b::AbstractVector{T}) where {T<:Float64,Ti<:Int32}

    maxfct = 1
    mnum   = 1
    nrhs   = 1
    phase  = Phases["SolveIterativeRefinement"]


    err = pardiso!( ns.solver.pt,                 # Handle to internal data structure. The entries must be set to zero prior to the first call to pardiso
                    maxfct,                       # Maximum number of factors with identical sparsity structure that must be kept in memory at the same time
                    mnum,                         # Actual matrix for the solution phase. The value must be: 1 <= mnum <= maxfct. 
                    ns.solver.mtype,              # Defines the matrix type, which influences the pivoting method
                    phase,                        # Controls the execution of the solver (11 == Analysis)
                    ns.mat.n,                     # Number of equations in the sparse linear systems of equations
                    ns.mat.nzval,                 # Contains the non-zero elements of the coefficient matrix A corresponding to the indices in ja
                    ns.mat.colptr,                # Pointers to columns in CSR format
                    ns.mat.rowval,                # Column indices of the CSR sparse matrix
                    Vector{Ti}(),                 # Permutation vector 
                    nrhs,                         # Number of right-hand sides that need to be solved for
                    ns.solver.iparm,              # This array is used to pass various parameters to Intel MKL PARDISO 
                    ns.solver.msglvl,             # Message level information
                    b,                            # Array, size (n, nrhs). On entry, contains the right-hand side vector/matrix
                    x)                            # Array, size (n, nrhs). If iparm(6)=0 it contains solution vector/matrix X

    pardiso_report_error(err)
end

function solve!(x::AbstractVector{T}, ns::PardisoNumericalSetup{T,Ti}, b::AbstractVector{T}) where {T<:Float64,Ti<:Int64}

    maxfct = 1
    mnum   = 1
    nrhs   = 1
    phase  = Phases["SolveIterativeRefinement"]


    err = pardiso_64!( ns.solver.pt,                 # Handle to internal data structure. The entries must be set to zero prior to the first call to pardiso
                    maxfct,                       # Maximum number of factors with identical sparsity structure that must be kept in memory at the same time
                    mnum,                         # Actual matrix for the solution phase. The value must be: 1 <= mnum <= maxfct. 
                    ns.solver.mtype,              # Defines the matrix type, which influences the pivoting method
                    phase,                        # Controls the execution of the solver (11 == Analysis)
                    ns.mat.n,                     # Number of equations in the sparse linear systems of equations
                    ns.mat.nzval,                 # Contains the non-zero elements of the coefficient matrix A corresponding to the indices in ja
                    ns.mat.colptr,                # Pointers to columns in CSR format
                    ns.mat.rowval,                # Column indices of the CSR sparse matrix
                    Vector{Ti}(),                 # Permutation vector 
                    nrhs,                         # Number of right-hand sides that need to be solved for
                    ns.solver.iparm,              # This array is used to pass various parameters to Intel MKL PARDISO 
                    ns.solver.msglvl,             # Message level information
                    b,                            # Array, size (n, nrhs). On entry, contains the right-hand side vector/matrix
                    x)                            # Array, size (n, nrhs). If iparm(6)=0 it contains solution vector/matrix X

    pardiso_report_error(err)
end

