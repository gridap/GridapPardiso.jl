
# https://software.intel.com/en-us/mkl-developer-reference-fortran-intel-mkl-pardiso-parameters-in-tabular-form

###############################################################
# MTYPE: Pardiso matrix type
# This scalar value defines the matrix type. PARDISO supports the following matrices
###############################################################
# Real matrices
const MTYPE_REAL_STRUCTURALLY_SYMMETRIC         = 1
const MTYPE_REAL_SYMMETRIC_POSITIVE_DEFINITE    = 2
const MTYPE_REAL_SYMMETRIC_INDEFINITE           = -2
const MTYPE_REAL_NON_SYMMETRIC                  = 11
# Complex matrices
const MTYPE_COMPLEX_STRUCTURALLY_SYMMETRIC      = 3
const MTYPE_COMPLEX_HERMITIAN_POSITIVE_DEFINITE = 4
const MTYPE_COMPLEX_HERMITIAN_INDEFINITE        = -4
const MTYPE_COMPLEX_SYMMETRIC                   = 6
const MTYPE_COMPLEX_NON_SYMMETRIC               = 13


###############################################################
# PHASE: Controls the execution of the solver
# 
# Usually it is a two- or three-digit integer. 
# The first digit indicates the starting phase of execution and the second digit indicates the ending phase. 
# Intel MKL PARDISO has the following phases of execution:
# 1. Phase 1: Fill-reduction analysis and symbolic factorization
# 2. Phase 2: Numerical factorization
# 3. Phase 3: Forward and Backward solve including iterative refinement.
#             This phase can be divided into two or three separate substitutions: forward, backward, and diagonal.
# 4. Termination and Memory Release Phase (PHASE ≤ 0)
###############################################################

const PHASE_ANALYSIS                                                    = 11
const PHASE_ANALYSIS_NUMERICAL_FACTORIZATION                            = 12
const PHASE_ANALYSIS_NUMERICAL_FACTORIZATION_SOLVE_ITERATIVE_REFINEMENT = 13
const PHASE_NUMERICAL_FACTORIZATION                                     = 22
const PHASE_SELECTED_INVERSION                                          = -22
const PHASE_NUMERICAL_FACTORIZATION_SOLVE_ITERATIVE_REFINEMENT          = 23
const PHASE_SOLVE_ITERATIVE_REFINEMENT                                  = 33
const PHASE_SOLVE_ITERATIVE_REFINEMENT_FORWARD_SUBSTITUTION             = 331
const PHASE_SOLVE_ITERATIVE_REFINEMENT_DIAGONAL_SUBSTITUTION            = 332
const PHASE_SOLVE_ITERATIVE_REFINEMENT_BACKWARD_SUBSTITUTION            = 333
const PHASE_RELEASE_INTERNAL_MEMORY                                     = 0 #@fverdugo cual queremos nosotros? release, o release all?
const PHASE_RELEASE_ALL_INTERNAL_MEMORY                                 = -1


"""
    pardiso_report_error(code::Int)
Report Pardiso error given its code.
"""
function pardiso_report_error(code::Int)

    if code < 0
        code == -1   && error(string("Pardiso Error (", code, "): ", "Input inconsistent."))
        code == -2   && error(string("Pardiso Error (", code, "): ", "Not enough memory."))
        code == -3   && error(string("Pardiso Error (", code, "): ", "Reordering problem."))
        code == -4   && error(string("Pardiso Error (", code, "): ", "Zero pivot, numerical fact. or iterative refinement problem."))
        code == -5   && error(string("Pardiso Error (", code, "): ", "Unclassified (internal) error."))
        code == -6   && error(string("Pardiso Error (", code, "): ", "Teordering failed (matrix types 11, 13 only)."))
        code == -7   && error(string("Pardiso Error (", code, "): ", "Diagonal matrix is singular."))
        code == -8   && error(string("Pardiso Error (", code, "): ", "32-bit integer overflow problem."))
        code == -10  && error(string("Pardiso Error (", code, "): ", "Error opening OOC files."))
        code == -11  && error(string("Pardiso Error (", code, "): ", "Read/write error with OOC files."))
        code == -12  && error(string("Pardiso Error (", code, "): ", "pardiso_64 called from 32-bit library."))
        code == -13  && error(string("Pardiso Error (", code, "): ", "Interrupted by the (user-defined) mkl_progress function."))
        code == -15  && error(string("Pardiso Error (", code, "): ", "Internal error which can appear for iparm(24)=10 and iparm(13)=1. Try switch matching off (set iparm(13)=0 and rerun.)."))

        error(string("Pardiso Error (", code, "): ", "Unknown error code."))
    end

end

###############################################################
# Iparm parameters
#
# [I] -> input 
# [O] -> output 
# for iparm[i], where i is:
#
###############################################################

#  1:  [I]   Use default values 
const IPARM_USE_DEFAULT_VALUES                                                  = 1 
#  2:  [I]   Fill-in reducing ordering for the input matrix.
const IPARM_FILL_IN_REDUCING_ORDERING                                           = 2 
#  3:  [-]   Reserved. Set to zero.
#  4:  [I]   Preconditioned CGS/CG.
const IPARM_PRECONDITIONED_CGS/CG                                               = 4 
#  5:  [I]   User permutation.
const IPARM_USER_PERMUTATION                                                    = 5 
#  6:  [I]   Write solution on x.
const IPARM_WRITE_SOLUTION_ON_X                                                 = 6
#  7:  [O]   Number of iterative refinement steps performed.
const IPARM_NUMBER_ITERATIVE_REFINEMENT_STEPS                                   = 7 
#  8:  [I]   Iterative refinement step.
const IPARM_ITERATIVE_REFINEMENT_STEP                                           = 8 
#  9:  [-]   Reserved. Set to zero.
#  10: [I]   Pivoting perturbation.
const IPARM_PIVOTING_PERTURBATION                                               = 10 
#  11: [I]   Scaling vectors.
const IPARM_SCALING_VECTORS                                                     = 11 
#  12: [I]   Solve with transposed or conjugate transposed matrix A.
const IPARM_TRANSPOSED_OR_CONJUGATED_TRANSPOSED                                 = 12 
#  13: [I]   Improved accuracy using (non-) symmetric weighted matching.
const IPARM_NON_SYMMETRIC_WEIGHTED_MATCHING                                     = 13
#  14: [O]   Number of perturbed pivots.
const IPARM_NUMBER_OF_PERTURBED_PIVOTS                                          = 14 
#  15: [O]   Peak memory on symbolic factorization.
const IPARM_PEAK_MEMORY_ON_SYMBOLIC_FACTORIZATION                               = 15
#  16: [O]   Permanent memory on symbolic factorization.
const IPARM_PERMANENT_MEMORY_ON_SYMBOLIC_FACTORIZATION                          = 16 
#  17: [O]   Size of factors/Peak memory on numerical factorization and solution.
const IPARM_SIZE_OF_FACTORS_PEAK_MEMORY_ON_NUMERICAL_FACTORIZATION_AND_SOLUTION = 17 
#  18: [I/O] Report the number of non-zero elements in the factors.
const IPARM_REPORT_NUMBER_OF_NON_ZEROS_IN_FACTORS                               = 18 
#  19: [I/O] Report number of floating point operations (in 106 floating point operations) that are necessary to factor the matrix A.
const IPARM_REPORT_NUMBER_OF_FLOATING_POINT_OPERATIONS                          = 19
#  20: [O]   Report CG/CGS diagnostics.
const IPARM_REPORT_CG_CGS_DIAGNOSTICS                                           = 20 
#  21: [I]   Pivoting for symmetric indefinite matrices.
const IPARM_PRIVOTING_FOR_SYMMETRIC_INDEFINITE_MATRICES                         = 21 
#  22: [O]   Inertia: number of positive eigenvalues.
const IPARM_NUMBER_OF_POSITIVE_EIGENVALUES                                      = 22 
#  23: [O]   Inertia: number of negative eigenvalues.
const IPARM_NUMBER_OF_NEGATIVE_EIGENVALUES                                      = 23 
#  24: [I]   Parallel factorization control.
const IPARM_PARALLEL_FACTORIZATION_CONTROL                                      = 24
#  25: [I]   Parallel forward/backward solve control.
const IPARM_PARALLEL_FORWARD_BACKWARD_SOLVE_CONTROL                             = 25 
#  26: [-]   Reserved. Set to zero.
#  27: [I]   Matrix checker.
const IPARM_MATRIX_CHECKER                                                      = 27 
#  28: [I]   Single or double precision Intel MKL PARDISO.
const IPARM_SINGLE_OR_DOUBLE_PRECISION                                          = 28 
#  29: [-]   Reserved. Set to zero.
#  30: [O]   Number of zero or negative pivots.
const IPARM_NUMBER_OF_ZERO_OR_NEGATIVE_PIVOTS                                   = 30 
#  31: [I]   Partial solve and computing selected components of the solution vectors.
const IPARM_PARTIAL_SOLVE_AND_COMPUTING_SELECTED_COMPONENTS                     = 31 
#  32: [-]   Reserved. Set to zero.
#  33: [-]   Reserved. Set to zero.
#  34: [I]   Optimal number of OpenMP threads for conditional numerical reproducibility (CNR) mode.
const IPARM_OPTIMAL_NUMBER_OF_OPENMPI_THREADS                                   = 34 
#  35: [I]   One- or zero-based indexing of columns and rows.
const IPARM_ONE_OR_ZERO_BASED_INDEXING                                          = 35 
#  36: [I/O] Schur complement matrix computation control. 
const IPARM_SCHUR_COMPLEMENT_MATRIX                                             = 36 
#  37: [I]   Format for matrix storage.
const IPARM_FORMAT_FOR_MATRIX_STORAGE                                           = 37 
#  38: [-]   Reserved. Set to zero.
#  39: [-]   Enable low rank update to accelerate factorization for multiple matrices with identical structure and similar values.
const IPARM_ENABLE_LOW_RANK                                                     = 39 
#  40: [-]   Reserved. Set to zero.
#  41: [-]   Reserved. Set to zero.
#  42: [-]   Reserved. Set to zero.
#  43: [-]   Control parameter for the computation of the diagonal of inverse matrix.
const IPARM_CONTROL_PARAMETER_FOR_COMPUTATION_OF_THE_DIAGONAL_OF_INVERSE_MATRIX = 43 
#  44: [-]   Reserved. Set to zero.
#  45: [-]   Reserved. Set to zero.
#  46: [-]   Reserved. Set to zero.
#  47: [-]   Reserved. Set to zero.
#  48: [-]   Reserved. Set to zero.
#  49: [-]   Reserved. Set to zero.
#  50: [-]   Reserved. Set to zero.
#  51: [-]   Reserved. Set to zero.
#  52: [-]   Reserved. Set to zero.
#  53: [-]   Reserved. Set to zero.
#  54: [-]   Reserved. Set to zero.
#  55: [-]   Reserved. Set to zero.
#  56: [-]   Diagonal and pivoting control.
#  57: [-]   Reserved. Set to zero.
#  58: [-]   Reserved. Set to zero.
#  59: [-]   Reserved. Set to zero.
#  60: [I]   Intel MKL PARDISO mode.
const IPARM_INTEL_MKL_PARDISO_MODE                                              = 60 
#  61: [-]   Reserved. Set to zero.
#  62: [-]   Reserved. Set to zero.
#  63: [O]   Size of the minimum OOC memory for numerical factorization and solution.
const IPARM_SIZE_OF_THE_MINIMUM_OCC_MEMORY                                      = 63 
#  64: [-]   Reserved. Set to zero.

