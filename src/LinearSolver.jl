#
# Maximum number of factors with identical sparsity structure
# that must be kept in memory at the same time
const maxfct = 1
# Actual matrix for the solution phase. The value must be: 1 <= mnum <= maxfct.
const mnum   = 1
# Number of right-hand sides that need to be solved for
const nrhs   = 1

const MTYPE_UNKNOWN = 0

new_pardiso_handle() = zeros(Int, 64)
new_iparm() = zeros(Int, 64)
function new_iparm(mtype::Integer)
  pt = new_pardiso_handle()
  iparm = Vector{Int32}(new_iparm())
  pardisoinit!(pt,mtype,iparm)
  iparm
end

getptr(S::SparseMatrixCSC) = S.colptr
getptr(S::SparseMatrixCSR) = S.rowptr
getptr(S::SymSparseMatrixCSR) = getptr(S.uppertrian)

getindices(S::SymSparseMatrixCSR) = colvals(S)
getindices(S::SparseMatrixCSC) = rowvals(S)
getindices(S::SparseMatrixCSR) = colvals(S)

hascolmajororder(::Type{<:SymSparseMatrixCSR}) = false
hascolmajororder(a::SymSparseMatrixCSR) = hascolmajororder(SymSparseMatrixCSR)
hascolmajororder(::Type{<:SparseMatrixCSC}) = true
hascolmajororder(a::SparseMatrixCSC) = hascolmajororder(SparseMatrixCSC)
hascolmajororder(a::SparseMatrixCSR) = false

get_pardiso(::Type{<:Int32}) = pardiso!
get_pardiso(::Type{<:Int64}) = pardiso_64!

has_0_based_storage(mat) = false
has_0_based_storage(mat::SparseMatrixCSR{0}) = true
has_0_based_storage(mat::SymSparseMatrixCSR{0}) = true

function get_mtype(mtype,mat::AbstractSparseMatrix{T}) where T
  error("Unsupported eltype $(T)")
end

# For the moment we use the matrix type but we could use
# other properties of the matrix as well

function get_mtype(mtype,mat::AbstractSparseMatrix{Float64})
  mtype == MTYPE_UNKNOWN ? MTYPE_REAL_NON_SYMMETRIC : mtype
end

function get_mtype(mtype,mat::AbstractSparseMatrix{Complex{Float64}})
  mtype == MTYPE_UNKNOWN ? MTYPE_COMPLEX_NON_SYMMETRIC : mtype
end

function get_mtype(mtype,mat::SymSparseMatrixCSR{Bi,Float64} where Bi)
  mtype == MTYPE_UNKNOWN ? MTYPE_REAL_SYMMETRIC_INDEFINITE : mtype
end

function get_mtype(mtype,mat::SymSparseMatrixCSR{Bi,Complex{Float64}} where Bi)
  mtype == MTYPE_UNKNOWN ? MTYPE_COMPLEX_SYMMETRIC : mtype
end

"""
    struct PardisoSolver{Ti} <: LinearSolver
Gridap LinearSolver implementation for Intel Pardiso MKL solver.
Official Intel Pardiso MKL documentation:
https://software.intel.com/en-us/mkl-developer-reference-fortran-intel-mkl-pardiso-parallel-direct-sparse-solver-interface
"""
struct PardisoSolver <: LinearSolver
  mtype     :: Int
  iparm     :: Vector{Int32}
  msglvl    :: Int
"""
    function PardisoSolver(
      mtype::Integer,
      iparm::AbstractVector{<:Integer},
      msglvl::Integer)
PardisoSolver inner constructor.
"""
  function PardisoSolver(
                mtype::Integer,
                iparm::AbstractVector{<:Integer},
                msglvl::Integer)

    @assert length(iparm) == 64
    @assert mtype in (MTYPE_UNKNOWN,
                      MTYPE_REAL_STRUCTURALLY_SYMMETRIC,
                      MTYPE_REAL_SYMMETRIC_POSITIVE_DEFINITE,
                      MTYPE_REAL_SYMMETRIC_INDEFINITE,
                      MTYPE_REAL_NON_SYMMETRIC,
                      MTYPE_COMPLEX_STRUCTURALLY_SYMMETRIC,
                      MTYPE_COMPLEX_HERMITIAN_POSITIVE_DEFINITE,
                      MTYPE_COMPLEX_HERMITIAN_INDEFINITE,
                      MTYPE_COMPLEX_SYMMETRIC,
                      MTYPE_COMPLEX_NON_SYMMETRIC
                    )
    new(Int(mtype), Vector{Int32}(iparm), Int(msglvl))
  end
end

"""
    function PardisoSolver(;
      mtype=MTYPE_UNKNOWN,
      iparm=new_iparm(mtype),
      msglvl=MSGLVL_QUIET)

PardisoSolver outer constructor via optional key-word arguments.
"""
function PardisoSolver(;
      mtype=MTYPE_UNKNOWN,
      iparm=new_iparm(mtype),
      msglvl=MSGLVL_QUIET)

  PardisoSolver(mtype,iparm,msglvl)
end

# mutable needed for the finalizer
mutable struct PardisoSymbolicSetup{T,Ti,A<:AbstractSparseMatrix} <: SymbolicSetup
  mtype::Int
  iparm::Vector{Ti}
  msglvl::Int
  eltype::Type{T}
  pt::Vector{Int}
  mat::A # We need to take ownership
end

function symbolic_setup(ps::PardisoSolver,mat::AbstractSparseMatrix{T,Ti}) where {T,Ti}
  pt = new_pardiso_handle()
  mtype = get_mtype(ps.mtype,mat)
  #pardisoinit!(pt,mtype,ps.iparm) # Warning! This would overwrite iparm
  iparm = Vector{Ti}(copy(ps.iparm))
  indexing = has_0_based_storage(mat) ? PARDISO_ZERO_BASED_INDEXING : PARDISO_ONE_BASED_INDEXING
  iparm[IPARM_ONE_OR_ZERO_BASED_INDEXING] = indexing
  msglvl = ps.msglvl
  m,n = size(mat)
  phase = PHASE_ANALYSIS
  f! = get_pardiso(Ti)
  err = f!( pt,               # Handle to internal data structure. The entries must be set to zero prior to the first call to pardiso
            maxfct,           # Maximum number of factors with identical sparsity structure that must be kept in memory at the same time
            mnum,             # Actual matrix for the solution phase. The value must be: 1 <= mnum <= maxfct.
            mtype,            # Defines the matrix type, which influences the pivoting method
            phase,            # Controls the execution of the solver (11 == Analysis)
            n,                # Number of equations in the sparse linear systems of equations
            nonzeros(mat),    # Contains the non-zero elements of the coefficient matrix A corresponding to the indices in ja
            getptr(mat),      # Pointers to columns in CSR format
            getindices(mat),  # Column indices of the CSR sparse matrix
            Vector{Ti}(),     # Permutation vector
            nrhs,             # Number of right-hand sides that need to be solved for
            iparm,            # This array is used to pass various parameters to Intel MKL PARDISO
            msglvl,           # Message level information
            Vector{T}(),      # Array, size (n, nrhs). On entry, contains the right-hand side vector/matrix
            Vector{T}())      # Array, size (n, nrhs). If iparm(6)=0 it contains solution vector/matrix X

  pardiso_report_error(err)
  pss = PardisoSymbolicSetup(mtype,iparm,msglvl,T,pt,mat)
  return finalizer(pardiso_finalize, pss)
end

function pardiso_finalize(pss::PardisoSymbolicSetup{T,Ti}) where {T,Ti}
  mtype = pss.mtype
  pt = pss.pt
  iparm = pss.iparm
  msglvl = pss.msglvl
  mat = pss.mat
  m,n = size(mat)
  phase = PHASE_RELEASE_INTERNAL_MEMORY
  f! = get_pardiso(Ti)
  err = f!( pt,               # Handle to internal data structure. The entries must be set to zero prior to the first call to pardiso
            maxfct,           # Maximum number of factors with identical sparsity structure that must be kept in memory at the same time
            mnum,             # Actual matrix for the solution phase. The value must be: 1 <= mnum <= maxfct.
            mtype,            # Defines the matrix type, which influences the pivoting method
            phase,            # Controls the execution of the solver (11 == Analysis)
            n,                # Number of equations in the sparse linear systems of equations
            nonzeros(mat),    # Contains the non-zero elements of the coefficient matrix A corresponding to the indices in ja
            getptr(mat),      # Pointers to columns in CSR format
            getindices(mat),  # Column indices of the CSR sparse matrix
            Vector{Ti}(),     # Permutation vector
            nrhs,             # Number of right-hand sides that need to be solved for
            iparm,            # This array is used to pass various parameters to Intel MKL PARDISO
            msglvl,           # Message level information
            Vector{T}(),      # Array, size (n, nrhs). On entry, contains the right-hand side vector/matrix
            Vector{T}())      # Array, size (n, nrhs). If iparm(6)=0 it contains solution vector/matrix X

  pardiso_report_error(err)
end

struct PardisoNumericalSetup{T,Ti} <: NumericalSetup
  pss::PardisoSymbolicSetup{T,Ti} # We need to take ownership here
end

function numerical_setup(pss::PardisoSymbolicSetup{T,Ti},mat::AbstractSparseMatrix{T,Ti}) where {T,Ti}
  pns = PardisoNumericalSetup(pss)
  numerical_setup!(pns,mat)
  pns
end

function numerical_setup!(pns::PardisoNumericalSetup{T,Ti},mat::AbstractSparseMatrix{T,Ti}) where {T,Ti}
  mtype = pns.pss.mtype
  iparm = pns.pss.iparm
  msglvl = pns.pss.msglvl
  pt = pns.pss.pt
  m,n = size(mat)
  phase = PHASE_NUMERICAL_FACTORIZATION
  f! = get_pardiso(Ti)
  err = f!( pt,               # Handle to internal data structure. The entries must be set to zero prior to the first call to pardiso
            maxfct,           # Maximum number of factors with identical sparsity structure that must be kept in memory at the same time
            mnum,             # Actual matrix for the solution phase. The value must be: 1 <= mnum <= maxfct.
            mtype,            # Defines the matrix type, which influences the pivoting method
            phase,            # Controls the execution of the solver (11 == Analysis)
            n,                # Number of equations in the sparse linear systems of equations
            nonzeros(mat),    # Contains the non-zero elements of the coefficient matrix A corresponding to the indices in ja
            getptr(mat),      # Pointers to columns in CSR format
            getindices(mat),  # Column indices of the CSR sparse matrix
            Vector{Ti}(),     # Permutation vector
            nrhs,             # Number of right-hand sides that need to be solved for
            iparm,            # This array is used to pass various parameters to Intel MKL PARDISO
            msglvl,           # Message level information
            Vector{T}(),      # Array, size (n, nrhs). On entry, contains the right-hand side vector/matrix
            Vector{T}())      # Array, size (n, nrhs). If iparm(6)=0 it contains solution vector/matrix X
  pardiso_report_error(err)
end

function solve!(x::AbstractVector{T},pns::PardisoNumericalSetup{T,Ti},b::AbstractVector{T}) where {T,Ti}
  mtype = pns.pss.mtype
  iparm = pns.pss.iparm
  msglvl = pns.pss.msglvl
  pt = pns.pss.pt
  mat = pns.pss.mat
  n = length(x)
  @assert n == length(b)
  @assert n == size(mat,1)
  @assert n == size(mat,2)
  phase = PHASE_SOLVE_ITERATIVE_REFINEMENT
  set_iparm_transpose!(iparm,mat)
  f! = get_pardiso(Ti)
  err = f!( pt,               # Handle to internal data structure. The entries must be set to zero prior to the first call to pardiso
            maxfct,           # Maximum number of factors with identical sparsity structure that must be kept in memory at the same time
            mnum,             # Actual matrix for the solution phase. The value must be: 1 <= mnum <= maxfct.
            mtype,            # Defines the matrix type, which influences the pivoting method
            phase,            # Controls the execution of the solver (11 == Analysis)
            n,                # Number of equations in the sparse linear systems of equations
            nonzeros(mat),    # Contains the non-zero elements of the coefficient matrix A corresponding to the indices in ja
            getptr(mat),      # Pointers to columns in CSR format
            getindices(mat),  # Column indices of the CSR sparse matrix
            Vector{Ti}(),     # Permutation vector
            nrhs,             # Number of right-hand sides that need to be solved for
            iparm,            # This array is used to pass various parameters to Intel MKL PARDISO
            msglvl,           # Message level information
            b,                # Array, size (n, nrhs). On entry, contains the right-hand side vector/matrix
            x)                # Array, size (n, nrhs). If iparm(6)=0 it contains solution vector/matrix X
  reset_iparm_transpose!(iparm,mat)
  pardiso_report_error(err)
  x
end

function set_iparm_transpose!(iparm,mat)
  if hascolmajororder(mat)
      if iparm[IPARM_TRANSPOSED_OR_CONJUGATED_TRANSPOSED] == PARDISO_SOLVE_LINEAR_SYSTEM
          iparm[IPARM_TRANSPOSED_OR_CONJUGATED_TRANSPOSED] = PARDISO_SOLVE_TRANSPOSED_SYSTEM
      elseif iparm[IPARM_TRANSPOSED_OR_CONJUGATED_TRANSPOSED] == PARDISO_SOLVE_TRANSPOSED_SYSTEM
          iparm[IPARM_TRANSPOSED_OR_CONJUGATED_TRANSPOSED] = PARDISO_SOLVE_LINEAR_SYSTEM
      else
          error(string("GridapPardiso Error: iparm[",
                      IPARM_TRANSPOSED_OR_CONJUGATED_TRANSPOSED,"] = ",
                      iparm[IPARM_TRANSPOSED_OR_CONJUGATED_TRANSPOSED],
                      " not supported."))
      end
  end
end

function reset_iparm_transpose!(iparm,mat)
  set_iparm_transpose!(iparm,mat)
end

