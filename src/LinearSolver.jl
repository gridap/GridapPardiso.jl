"""
Wrapper of the MKL Pardiso solver available in julia
"""

# Maximum number of factors with identical sparsity structure
# that must be kept in memory at the same time
const maxfct = 1
# Actual matrix for the solution phase. The value must be: 1 <= mnum <= maxfct.
const mnum   = 1
# Number of right-hand sides that need to be solved for
const nrhs   = 1

"""
    struct PardisoSolver{Ti} <: LinearSolver
Gridap LinearSolver implementation for Intel Pardiso MKL solver.
Official Intel Pardiso MKL documentation:
https://software.intel.com/en-us/mkl-developer-reference-fortran-intel-mkl-pardiso-parallel-direct-sparse-solver-interface
"""
struct PardisoSolver{Ti} <: LinearSolver
  mtype     :: Int
  iparm     :: Vector{Ti}
  msglvl    :: Int
  pt        :: Vector{Int}

"""
    function PardisoSolver(mtype::Int, iparm::Vector{Ti}, msglvl::Int, pt:: Vector{Int}) where {Ti<:Integer}
PardisoSolver inner constructor.
"""
  function PardisoSolver(
                mtype::Int,
                iparm::Vector{Ti},
                msglvl::Int, pt:: Vector{Int}) where {Ti<:Integer}

    @assert length(iparm) == length(pt) == 64
    @assert mtype in (MTYPE_REAL_STRUCTURALLY_SYMMETRIC,
                      MTYPE_REAL_SYMMETRIC_POSITIVE_DEFINITE,
                      MTYPE_REAL_SYMMETRIC_INDEFINITE,
                      MTYPE_REAL_NON_SYMMETRIC,
                      MTYPE_COMPLEX_STRUCTURALLY_SYMMETRIC,
                      MTYPE_COMPLEX_HERMITIAN_POSITIVE_DEFINITE,
                      MTYPE_COMPLEX_HERMITIAN_INDEFINITE,
                      MTYPE_COMPLEX_SYMMETRIC,
                      MTYPE_COMPLEX_NON_SYMMETRIC
                    )
    tmpiparm = Vector{Int32}(iparm)
    pardisoinit!(pt, mtype, tmpiparm)
    new{Ti}(Int(mtype), Vector{Ti}(tmpiparm), Int(msglvl), pt)
  end
end

"""
    mutable struct PardisoSymbolicSetup{T,Ti} <: SymbolicSetup
Gridap SymbolicSetup implementation for Intel Pardiso MKL solver.
"""
mutable struct PardisoSymbolicSetup{T,Ti} <: SymbolicSetup
    phase     :: Int
    mat       :: AbstractSparseMatrix{T,Ti}
    solver    :: PardisoSolver
end

"""
    mutable struct PardisoNumericalSetup{T,Ti} <: NumericalSetup
Gridap NumericalSetup implementation for Intel Pardiso MKL solver.
"""
mutable struct PardisoNumericalSetup{T,Ti} <: NumericalSetup
    phase     :: Int
    mat       :: AbstractSparseMatrix{T,Ti}
    solver    :: PardisoSolver
end

"""
    function PardisoSolver()
PardisoSolver constructor overloading with default values.
Returns a PardisoSolver.
"""
PardisoSolver() = PardisoSolver(MTYPE_REAL_NON_SYMMETRIC,
                                new_iparm(),
                                MSGLVL_QUIET,
                                new_pardiso_handle())

"""
    function PardisoSolver(mtype)
PardisoSolver constructor overloading with default values.
Returns a PardisoSolver given its matrix type.
"""
PardisoSolver(mtype::Int) =
    PardisoSolver(mtype, new_iparm(), MSGLVL_QUIET, new_pardiso_handle())

PardisoSolver(::Type{<:AbstractSparseMatrix{Tv,Ti}}) where {Tv,Ti}=
    PardisoSolver(GridapPardiso.MTYPE_REAL_NON_SYMMETRIC)

PardisoSolver(::Type{SymSparseMatrixCSR{Bi,Tv,Ti}}) where {Bi,Tv,Ti}=
    PardisoSolver(GridapPardiso.MTYPE_REAL_SYMMETRIC_INDEFINITE)

PardisoSolver(op::FEOperator) = PardisoSolver(get_matrix_type(op))

function get_matrix_type(op::FEOperator)
  @abstractmethod
end

function get_matrix_type(op::AffineFEOperator)
  typeof(get_matrix(op))
end

"""
    function PardisoSolver(mtype, iparm)
PardisoSolver constructor overloading with default values.
Returns a PardisoSolver given its matrix type and Pardiso parameters.
"""
PardisoSolver(mtype::Int, iparm) =
    PardisoSolver(mtype, iparm, MSGLVL_QUIET, new_pardiso_handle())

"""
    function PardisoSolver(mtype, iparm, msglvl)
PardisoSolver constructor overloading with default values.
Returns a PardisoSolver given its matrix type, Pardiso parameters and verbosity.
"""
PardisoSolver(mtype::Int, iparm, msglvl) =
    PardisoSolver(mtype, iparm, msglvl, new_pardiso_handle())


"""
    function build_PardisoSymbolicSetup(phase::Integer, mat::AbstractSparseMatrix{T,Ti}, solver::PardisoSolver) where {T,Ti}
PardisoSymbolicSetup constructor overloading with default values.
Returns a PardisoSymbolicSetup from a given AbstractSparseMatrix.
"""
function build_PardisoSymbolicSetup(phase::Integer,
                mat::AbstractSparseMatrix{T,Ti},
                solver::PardisoSolver) where {T,Ti}
    if !(solver.mtype in (MTYPE_REAL_STRUCTURALLY_SYMMETRIC,
                          MTYPE_REAL_NON_SYMMETRIC))
        @warn "Pardiso matrix type ($(solver.mtype)) does not match with $(typeof(mat))."
    end
    PardisoSymbolicSetup(phase,SparseMatrixCSC{T,Ti}(mat),solver)
end

"""
    function build_PardisoNumericalSetup(phase::Integer, mat::AbstractSparseMatrix{T,Ti}, solver::PardisoSolver) where {T,Ti}
PardisoNumericalSetup constructor overloading with default values.
Returns a PardisoNumericalSetup from a given AbstractSparseMatrix.
"""
function build_PardisoNumericalSetup(phase::Integer,
                mat::AbstractSparseMatrix{T,Ti},
                solver::PardisoSolver) where {T,Ti}
    if !(solver.mtype in (MTYPE_REAL_STRUCTURALLY_SYMMETRIC,
                          MTYPE_REAL_NON_SYMMETRIC))
        @warn "Pardiso matrix type ($(solver.mtype)) does not match with $(typeof(mat))."
    end
    PardisoNumericalSetup(phase,SparseMatrixCSC{T,Ti}(mat),solver)
end

"""
    function build_PardisoSymbolicSetup(phase::Integer, mat::SparseMatrixCSC{T,Ti}, solver::PardisoSolver) where {T,Ti}
PardisoSymbolicSetup constructor overloading.
Returns a PardisoSymbolicSetup from a given SparseMatrixCSC.
"""
function build_PardisoSymbolicSetup(phase::Integer,
                mat::SparseMatrixCSC{T,Ti},
                solver::PardisoSolver) where {T,Ti}
    if !(solver.mtype in (MTYPE_REAL_STRUCTURALLY_SYMMETRIC,
                          MTYPE_REAL_NON_SYMMETRIC))
        @warn "Pardiso matrix type ($(solver.mtype)) does not match with $(typeof(mat))."
    end
    PardisoSymbolicSetup(phase,mat,solver)
end

"""
    function build_PardisoNumericalSetup(phase::Integer, mat::SparseMatrixCSC{T,Ti}, solver::PardisoSolver) where {T,Ti}
PardisoNumericalSetup constructor overloading.
Returns a PardisoNumericalSetup from a given SparseMatrixCSC.
"""
function build_PardisoNumericalSetup(phase::Integer,
                mat::SparseMatrixCSC{T,Ti},
                solver::PardisoSolver) where {T,Ti}
    if !(solver.mtype in (MTYPE_REAL_STRUCTURALLY_SYMMETRIC,
                          MTYPE_REAL_NON_SYMMETRIC))
        @warn "Pardiso matrix type ($(solver.mtype)) does not match with $(typeof(mat))."
    end
    PardisoNumericalSetup(phase,mat,solver)
end

"""
    function build_PardisoSymbolicSetup(phase::Integer, mat::SparseMatrixCSR{T,Ti}, solver::PardisoSolver) where {T,Ti}
PardisoSymbolicSetup constructor overloading.
Returns a PardisoSymbolicSetup from a given SparseMatrixCSR.
"""
function build_PardisoSymbolicSetup(phase::Integer,
                mat::SparseMatrixCSR{Bi},
                solver::PardisoSolver) where {Bi}
    if !(solver.mtype in (MTYPE_REAL_STRUCTURALLY_SYMMETRIC,
                          MTYPE_REAL_NON_SYMMETRIC))
        @warn "Pardiso matrix type ($(solver.mtype)) does not match with $(typeof(mat))."
    end
    solver.iparm[IPARM_ONE_OR_ZERO_BASED_INDEXING] = Bi == 0 ? PARDISO_ZERO_BASED_INDEXING : PARDISO_ONE_BASED_INDEXING
    PardisoSymbolicSetup(phase,mat,solver)
end

"""
    function build_PardisoNumericalSetup(phase::Integer, mat::SparseMatrixCSR{T,Ti}, solver::PardisoSolver) where {T,Ti}
PardisoNumericalSetup constructor overloading.
Returns a PardisoNumericalSetup from a given SparseMatrixCSR.
"""
function build_PardisoNumericalSetup(phase::Integer,
                mat::SparseMatrixCSR{Bi},
                solver::PardisoSolver) where {Bi}
    if !(solver.mtype in (MTYPE_REAL_STRUCTURALLY_SYMMETRIC,
                          MTYPE_REAL_NON_SYMMETRIC))
        @warn "Pardiso matrix type ($(solver.mtype)) does not match with $(typeof(mat))."
    end
    solver.iparm[IPARM_ONE_OR_ZERO_BASED_INDEXING] = Bi == 0 ? PARDISO_ZERO_BASED_INDEXING : PARDISO_ONE_BASED_INDEXING
    PardisoNumericalSetup(phase,mat,solver)
end

"""
    function build_PardisoSymbolicSetup(phase::Integer, mat::SymSparseMatrixCSR{T,Ti}, solver::PardisoSolver) where {T,Ti}
PardisoSymbolicSetup constructor overloading.
Returns a PardisoSymbolicSetup from a given SymSparseMatrixCSR.
"""
function build_PardisoSymbolicSetup(phase::Integer,
                mat::SymSparseMatrixCSR{Bi},
                solver::PardisoSolver) where {Bi}
    if !(solver.mtype in (MTYPE_REAL_SYMMETRIC_POSITIVE_DEFINITE,
                          MTYPE_REAL_SYMMETRIC_INDEFINITE))
        @warn "Pardiso matrix type ($(solver.mtype)) does not match with $(typeof(mat))."
    end
    solver.iparm[IPARM_ONE_OR_ZERO_BASED_INDEXING] = Bi == 0 ? PARDISO_ZERO_BASED_INDEXING : PARDISO_ONE_BASED_INDEXING
    PardisoSymbolicSetup(phase,mat.uppertrian,solver)
end

"""
    function build_PardisoNumericalSetup(phase::Integer, mat::SymSparseMatrixCSR{T,Ti}, solver::PardisoSolver) where {T,Ti}
PardisoNumericalSetup constructor overloading.
Returns a PardisoNumericalSetup from a given SparseMatrixCSR.
"""
function build_PardisoNumericalSetup(phase::Integer,
                mat::SymSparseMatrixCSR{Bi},
                solver::PardisoSolver) where {Bi}
    if !(solver.mtype in (MTYPE_REAL_SYMMETRIC_POSITIVE_DEFINITE,
                          MTYPE_REAL_SYMMETRIC_INDEFINITE))
        @warn "Pardiso matrix type ($(solver.mtype)) does not match with $(typeof(mat))."
    end
    solver.iparm[IPARM_ONE_OR_ZERO_BASED_INDEXING] = Bi == 0 ? PARDISO_ZERO_BASED_INDEXING : PARDISO_ONE_BASED_INDEXING
    PardisoNumericalSetup(phase,mat.uppertrian,solver)
end

"""
    function symbolic_setup(ps::PardisoSolver{Ti}, mat::M) where {T<:Float64,Ti<:Int32,M<:AbstractSparseMatrix{T,Ti}}
Gridap symbolic_setup overload.
Use Intel Pardiso MKL to perform the analisys phase.
"""
function symbolic_setup(
        ps::PardisoSolver{Ti},
        mat::M) where {T<:Float64,Ti<:Int32,M<:AbstractSparseMatrix{T,Ti}}

    pss = build_PardisoSymbolicSetup(GridapPardiso.PHASE_ANALYSIS, mat, ps)
    m,n = size(pss.mat)

    err = pardiso!( pss.solver.pt,                # Handle to internal data structure. The entries must be set to zero prior to the first call to pardiso
                    maxfct,                       # Maximum number of factors with identical sparsity structure that must be kept in memory at the same time
                    mnum,                         # Actual matrix for the solution phase. The value must be: 1 <= mnum <= maxfct.
                    pss.solver.mtype,             # Defines the matrix type, which influences the pivoting method
                    pss.phase,                    # Controls the execution of the solver (11 == Analysis)
                    n,                            # Number of equations in the sparse linear systems of equations
                    nonzeros(pss.mat),            # Contains the non-zero elements of the coefficient matrix A corresponding to the indices in ja
                    getptr(pss.mat),              # Pointers to columns in CSR format
                    getindices(pss.mat),          # Column indices of the CSR sparse matrix
                    Vector{Ti}(),                 # Permutation vector
                    nrhs,                         # Number of right-hand sides that need to be solved for
                    pss.solver.iparm,             # This array is used to pass various parameters to Intel MKL PARDISO
                    pss.solver.msglvl,            # Message level information
                    Vector{T}(),                  # Array, size (n, nrhs). On entry, contains the right-hand side vector/matrix
                    Vector{T}())                  # Array, size (n, nrhs). If iparm(6)=0 it contains solution vector/matrix X
    pardiso_report_error(err)
    return finalizer(symbolic_setup_finalize, pss)
end

"""
    function symbolic_setup(ps::PardisoSolver{Ti}, mat::M) where {T<:Float64,Ti<:Int64,M<:AbstractSparseMatrix{T,Ti}}
Gridap symbolic_setup overload.
Use Intel Pardiso MKL to perform the analisys phase.
"""
function symbolic_setup(
        ps::PardisoSolver{Ti},
        mat::M) where {T<:Float64,Ti<:Int64,M<:AbstractSparseMatrix{T,Ti}}

    pss = build_PardisoSymbolicSetup(GridapPardiso.PHASE_ANALYSIS, mat, ps)
    m,n = size(pss.mat)

    err = pardiso_64!( pss.solver.pt,             # Handle to internal data structure. The entries must be set to zero prior to the first call to pardiso
                    maxfct,                       # Maximum number of factors with identical sparsity structure that must be kept in memory at the same time
                    mnum,                         # Actual matrix for the solution phase. The value must be: 1 <= mnum <= maxfct.
                    pss.solver.mtype,             # Defines the matrix type, which influences the pivoting method
                    pss.phase,                    # Controls the execution of the solver (11 == Analysis)
                    n,                            # Number of equations in the sparse linear systems of equations
                    nonzeros(pss.mat),            # Contains the non-zero elements of the coefficient matrix A corresponding to the indices in ja
                    getptr(pss.mat),              # Pointers to columns in CSR format
                    getindices(pss.mat),          # Column indices of the CSR sparse matrix
                    Vector{Ti}(),                 # Permutation vector
                    nrhs,                         # Number of right-hand sides that need to be solved for
                    pss.solver.iparm,             # This array is used to pass various parameters to Intel MKL PARDISO
                    pss.solver.msglvl,            # Message level information
                    Vector{T}(),                  # Array, size (n, nrhs). On entry, contains the right-hand side vector/matrix
                    Vector{T}())                  # Array, size (n, nrhs). If iparm(6)=0 it contains solution vector/matrix X
    pardiso_report_error(err)
    return finalizer(symbolic_setup_finalize, pss)
end

"""
    function symbolic_setup_finalize(pss::PardisoSymbolicSetup{T,Ti}) where {T,Ti<:Int32}
Finalization of `PardisoSymbolicSetup{T,Ti}` object.
Release internal Pardiso memory.
"""
function symbolic_setup_finalize(
        pss::PardisoSymbolicSetup{T,Ti}) where {T,Ti<:Int32}

    pss.phase = GridapPardiso.PHASE_RELEASE_INTERNAL_MEMORY
    m,n = size(pss.mat)

    err = pardiso!( pss.solver.pt,                # Handle to internal data structure. The entries must be set to zero prior to the first call to pardiso
                    maxfct,                       # Maximum number of factors with identical sparsity structure that must be kept in memory at the same time
                    mnum,                         # Actual matrix for the solution phase. The value must be: 1 <= mnum <= maxfct.
                    pss.solver.mtype,             # Defines the matrix type, which influences the pivoting method
                    pss.phase,                    # Controls the execution of the solver (11 == Analysis)
                    n,                            # Number of equations in the sparse linear systems of equations
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

"""
    function symbolic_setup_finalize(pss::PardisoSymbolicSetup{T,Ti}) where {T,Ti<:Int64}
Finalization of `PardisoSymbolicSetup{T,Ti}` object.
Release internal Pardiso memory.
"""
function symbolic_setup_finalize(
        pss::PardisoSymbolicSetup{T,Ti}) where {T,Ti<:Int64}

    pss.phase = GridapPardiso.PHASE_RELEASE_INTERNAL_MEMORY
    m,n = size(pss.mat)

    err = pardiso_64!( pss.solver.pt,             # Handle to internal data structure. The entries must be set to zero prior to the first call to pardiso
                    maxfct,                       # Maximum number of factors with identical sparsity structure that must be kept in memory at the same time
                    mnum,                         # Actual matrix for the solution phase. The value must be: 1 <= mnum <= maxfct.
                    pss.solver.mtype,             # Defines the matrix type, which influences the pivoting method
                    pss.phase,                    # Controls the execution of the solver (11 == Analysis)
                    n,                            # Number of equations in the sparse linear systems of equations
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

"""
    numerical_setup!(pns::PardisoNumericalSetup{T,Ti}, mat::M) where {T<:Float64,Ti<:Int32,M<:AbstractSparseMatrix{T,Ti}}
Gridap numerical_setup overload.
Use Intel Pardiso MKL to perform the numerical factorization phase.
"""
function numerical_setup!(
        pns::PardisoNumericalSetup{T,Ti},
        mat::M) where {T<:Float64,Ti<:Int32,M<:AbstractSparseMatrix{T,Ti}}

    m,n = size(pns.mat)

    err = pardiso!( pns.solver.pt,                # Handle to internal data structure. The entries must be set to zero prior to the first call to pardiso
                    maxfct,                       # Maximum number of factors with identical sparsity structure that must be kept in memory at the same time
                    mnum,                         # Actual matrix for the solution phase. The value must be: 1 <= mnum <= maxfct.
                    pns.solver.mtype,             # Defines the matrix type, which influences the pivoting method
                    pns.phase,                    # Controls the execution of the solver (11 == Analysis)
                    n,                            # Number of equations in the sparse linear systems of equations
                    nonzeros(pns.mat),            # Contains the non-zero elements of the coefficient matrix A corresponding to the indices in ja
                    getptr(pns.mat),              # Pointers to columns in CSR format
                    getindices(pns.mat),          # Column indices of the CSR sparse matrix
                    Vector{Ti}(),                 # Permutation vector
                    nrhs,                         # Number of right-hand sides that need to be solved for
                    pns.solver.iparm,             # This array is used to pass various parameters to Intel MKL PARDISO
                    pns.solver.msglvl,            # Message level information
                    Vector{T}(),                  # Array, size (n, nrhs). On entry, contains the right-hand side vector/matrix
                    Vector{T}())                  # Array, size (n, nrhs). If iparm(6)=0 it contains solution vector/matrix X
    pardiso_report_error(err)
    return finalizer(numerical_setup_finalize!, pns)
end

"""
    function numerical_setup!(pns::PardisoNumericalSetup{T,Ti}, mat::M) where {T<:Float64,Ti<:Int64,M<:AbstractSparseMatrix{T,Ti}}
Gridap numerical_setup overload.
Use Intel Pardiso MKL to perform the numerical factorization phase.
"""
function numerical_setup!(
        pns::PardisoNumericalSetup{T,Ti},
        mat::M) where {T<:Float64,Ti<:Int64,M<:AbstractSparseMatrix{T,Ti}}

    m,n = size(pns.mat)

    err = pardiso_64!( pns.solver.pt,             # Handle to internal data structure. The entries must be set to zero prior to the first call to pardiso
                    maxfct,                       # Maximum number of factors with identical sparsity structure that must be kept in memory at the same time
                    mnum,                         # Actual matrix for the solution phase. The value must be: 1 <= mnum <= maxfct.
                    pns.solver.mtype,             # Defines the matrix type, which influences the pivoting method
                    pns.phase,                    # Controls the execution of the solver (11 == Analysis)
                    n,                            # Number of equations in the sparse linear systems of equations
                    nonzeros(pns.mat),            # Contains the non-zero elements of the coefficient matrix A corresponding to the indices in ja
                    getptr(pns.mat),              # Pointers to columns in CSR format
                    getindices(pns.mat),          # Column indices of the CSR sparse matrix
                    Vector{Ti}(),                 # Permutation vector
                    nrhs,                         # Number of right-hand sides that need to be solved for
                    pns.solver.iparm,             # This array is used to pass various parameters to Intel MKL PARDISO
                    pns.solver.msglvl,            # Message level information
                    Vector{T}(),                  # Array, size (n, nrhs). On entry, contains the right-hand side vector/matrix
                    Vector{T}())                  # Array, size (n, nrhs). If iparm(6)=0 it contains solution vector/matrix X
    pardiso_report_error(err)
    return finalizer(numerical_setup_finalize!, pns)
end

"""
    function numerical_setup_finalize(pss::PardisoNumericalSetup{T,Ti}) where {T,Ti<:Int32}
Finalization of `PardisoNumericalSetup{T,Ti}` object.
Release internal Pardiso memory.
"""
function numerical_setup_finalize!(
        pns::PardisoNumericalSetup{T,Ti}) where {T, Ti<:Int32}

    pns.phase = GridapPardiso.PHASE_RELEASE_INTERNAL_MEMORY
    m,n = size(pns.mat)

    err = pardiso!( pns.solver.pt,                # Handle to internal data structure. The entries must be set to zero prior to the first call to pardiso
                    maxfct,                       # Maximum number of factors with identical sparsity structure that must be kept in memory at the same time
                    mnum,                         # Actual matrix for the solution phase. The value must be: 1 <= mnum <= maxfct.
                    pns.solver.mtype,             # Defines the matrix type, which influences the pivoting method
                    pns.phase,                    # Controls the execution of the solver (11 == Analysis)
                    n,                            # Number of equations in the sparse linear systems of equations
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

"""
    function numerical_setup_finalize(pss::PardisoNumericalSetup{T,Ti}) where {T,Ti<:Int64}
Finalization of `PardisoNumericalSetup{T,Ti}` object.
Release internal Pardiso memory.
"""
function numerical_setup_finalize!(
        pns::PardisoNumericalSetup{T,Ti}) where {T, Ti<:Int64}

    pns.phase = GridapPardiso.PHASE_RELEASE_INTERNAL_MEMORY
    m,n = size(pns.mat)

    err = pardiso_64!( pns.solver.pt,             # Handle to internal data structure. The entries must be set to zero prior to the first call to pardiso
                    maxfct,                       # Maximum number of factors with identical sparsity structure that must be kept in memory at the same time
                    mnum,                         # Actual matrix for the solution phase. The value must be: 1 <= mnum <= maxfct.
                    pns.solver.mtype,             # Defines the matrix type, which influences the pivoting method
                    pns.phase,                    # Controls the execution of the solver (11 == Analysis)
                    n,                            # Number of equations in the sparse linear systems of equations
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

"""
    function numerical_setup(pss::PardisoSymbolicSetup{T,Ti}, mat::M) where {T<:Float64,Ti<:Integer,M<:AbstractSparseMatrix{T,Ti}}
Gridap numerical_setup overload.
Create the PardisoSymbolicSetup object and use Intel Pardiso MKL to perform the numerical factorization phase.
"""
function numerical_setup(
        pss::PardisoSymbolicSetup{T,Ti},
        mat::M) where {T<:Float64,Ti<:Integer,M<:AbstractSparseMatrix{T,Ti}}
    pns = build_PardisoNumericalSetup(GridapPardiso.PHASE_NUMERICAL_FACTORIZATION, mat, pss.solver)
    return numerical_setup!(pns, mat)
end

"""
    function solve!(x::AbstractVector{T}, ns::PardisoNumericalSetup{T,Ti}, b::AbstractVector{T}) where {T<:Float64,Ti<:Int32}
Gridap solve! method overload.
Use Intel Pardiso MKL to perform the solve iterative refinement phase.
"""
function solve!(
        x::AbstractVector{T},
        ns::PardisoNumericalSetup{T,Ti},
        b::AbstractVector{T}) where {T<:Float64,Ti<:Int32}

    phase  = GridapPardiso.PHASE_SOLVE_ITERATIVE_REFINEMENT
    m,n = size(ns.mat)

    # Here we assume that the users defines iparm for CSR matrix
    iparmcopy = copy(ns.solver.iparm)
    if hascolmajororder(ns.mat)
        if ns.solver.iparm[IPARM_TRANSPOSED_OR_CONJUGATED_TRANSPOSED] == PARDISO_SOLVE_LINEAR_SYSTEM
            iparmcopy[IPARM_TRANSPOSED_OR_CONJUGATED_TRANSPOSED] = PARDISO_SOLVE_TRANSPOSED_SYSTEM
        elseif ns.solver.iparm[IPARM_TRANSPOSED_OR_CONJUGATED_TRANSPOSED] == PARDISO_SOLVE_TRANSPOSED_SYSTEM
            iparmcopy[IPARM_TRANSPOSED_OR_CONJUGATED_TRANSPOSED] = PARDISO_SOLVE_LINEAR_SYSTEM
        else
            error(string("GridapPardiso Error: iparm[",
                        IPARM_TRANSPOSED_OR_CONJUGATED_TRANSPOSED,"] = ",
                        ns.solver.iparm[IPARM_TRANSPOSED_OR_CONJUGATED_TRANSPOSED],
                        " not supported."))
        end
    end

    err = pardiso!( ns.solver.pt,                 # Handle to internal data structure. The entries must be set to zero prior to the first call to pardiso
                    maxfct,                       # Maximum number of factors with identical sparsity structure that must be kept in memory at the same time
                    mnum,                         # Actual matrix for the solution phase. The value must be: 1 <= mnum <= maxfct.
                    ns.solver.mtype,              # Defines the matrix type, which influences the pivoting method
                    phase,                        # Controls the execution of the solver (11 == Analysis)
                    n,                            # Number of equations in the sparse linear systems of equations
                    nonzeros(ns.mat),            # Contains the non-zero elements of the coefficient matrix A corresponding to the indices in ja
                    getptr(ns.mat),              # Pointers to columns in CSR format
                    getindices(ns.mat),          # Column indices of the CSR sparse matrix
                    Vector{Ti}(),                 # Permutation vector
                    nrhs,                         # Number of right-hand sides that need to be solved for
                    iparmcopy,                    # This array is used to pass various parameters to Intel MKL PARDISO
                    ns.solver.msglvl,             # Message level information
                    b,                            # Array, size (n, nrhs). On entry, contains the right-hand side vector/matrix
                    x)                            # Array, size (n, nrhs). If iparm(6)=0 it contains solution vector/matrix X

    pardiso_report_error(err)
end

"""
    function solve!(x::AbstractVector{T}, ns::PardisoNumericalSetup{T,Ti}, b::AbstractVector{T}) where {T<:Float64,Ti<:Int64}
Gridap solve! method overload.
Use Intel Pardiso MKL to perform the solve iterative refinement phase.
"""
function solve!(
        x::AbstractVector{T},
        ns::PardisoNumericalSetup{T,Ti},
        b::AbstractVector{T}) where {T<:Float64,Ti<:Int64}

    phase  = GridapPardiso.PHASE_SOLVE_ITERATIVE_REFINEMENT
    m,n = size(ns.mat)

    # Here we assume that the users defines iparm for CSR matrix
    iparmcopy = copy(ns.solver.iparm)
    if hascolmajororder(ns.mat)
        if ns.solver.iparm[IPARM_TRANSPOSED_OR_CONJUGATED_TRANSPOSED] == PARDISO_SOLVE_LINEAR_SYSTEM
            iparmcopy[IPARM_TRANSPOSED_OR_CONJUGATED_TRANSPOSED] = PARDISO_SOLVE_TRANSPOSED_SYSTEM
        elseif ns.solver.iparm[IPARM_TRANSPOSED_OR_CONJUGATED_TRANSPOSED] == PARDISO_SOLVE_TRANSPOSED_SYSTEM
            iparmcopy[IPARM_TRANSPOSED_OR_CONJUGATED_TRANSPOSED] = PARDISO_SOLVE_LINEAR_SYSTEM
        else
            error(string("GridapPardiso Error: iparm[",
                        IPARM_TRANSPOSED_OR_CONJUGATED_TRANSPOSED,"] = ",
                        ns.solver.iparm[IPARM_TRANSPOSED_OR_CONJUGATED_TRANSPOSED],
                        " not supported."))
        end
    end

    err = pardiso_64!( ns.solver.pt,              # Handle to internal data structure. The entries must be set to zero prior to the first call to pardiso
                    maxfct,                       # Maximum number of factors with identical sparsity structure that must be kept in memory at the same time
                    mnum,                         # Actual matrix for the solution phase. The value must be: 1 <= mnum <= maxfct.
                    ns.solver.mtype,              # Defines the matrix type, which influences the pivoting method
                    phase,                        # Controls the execution of the solver (11 == Analysis)
                    n,                            # Number of equations in the sparse linear systems of equations
                    nonzeros(ns.mat),             # Contains the non-zero elements of the coefficient matrix A corresponding to the indices in ja
                    getptr(ns.mat),               # Pointers to columns in CSR format
                    getindices(ns.mat),           # Column indices of the CSR sparse matrix
                    Vector{Ti}(),                 # Permutation vector
                    nrhs,                         # Number of right-hand sides that need to be solved for
                    iparmcopy,                    # This array is used to pass various parameters to Intel MKL PARDISO
                    ns.solver.msglvl,             # Message level information
                    b,                            # Array, size (n, nrhs). On entry, contains the right-hand side vector/matrix
                    x)                            # Array, size (n, nrhs). If iparm(6)=0 it contains solution vector/matrix X

    pardiso_report_error(err)
end
