
# https://software.intel.com/en-us/mkl-developer-reference-fortran-intel-mkl-pardiso-parameters-in-tabular-form

# MTYPE: Pardiso matrix type
# This scalar value defines the matrix type. PARDISO supports the following matrices

# Real matrices
MatrixType = Base.ImmutableDict(            "RealStructurallySymmetric"        => 1)
MatrixType = Base.ImmutableDict(MatrixType, "RealSymmetricPositiveDefinite"    => 2)
MatrixType = Base.ImmutableDict(MatrixType, "RealSymmetricIndefinite"          => -2)
MatrixType = Base.ImmutableDict(MatrixType, "RealNonSymmetric"                 => 11)
# Complex matrices
MatrixType = Base.ImmutableDict(MatrixType, "ComplexStructurallySymmetric"     => 3)
MatrixType = Base.ImmutableDict(MatrixType, "ComplexHermitianPositiveDefinite" => 4)
MatrixType = Base.ImmutableDict(MatrixType, "ComplexHermitianIndefinite"       => -4)
MatrixType = Base.ImmutableDict(MatrixType, "ComplexSymmetric"                 => 6)
MatrixType = Base.ImmutableDict(MatrixType, "ComplexNonSymmetric"              => 13)


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

Phase = Base.ImmutableDict(       "Analysis"                                               => 11)
Phase = Base.ImmutableDict(Phase, "AnalysisNumericalFactorization"                         => 12)
Phase = Base.ImmutableDict(Phase, "AnalysisNumericalFactorizationSolveIterativeRefinement" => 13)
Phase = Base.ImmutableDict(Phase, "NumericalFactorization"                                 => 22)
Phase = Base.ImmutableDict(Phase, "SelectedInversion"                                      => -22)
Phase = Base.ImmutableDict(Phase, "NumericalFactorizationSolveIterativeRefinement"         => 23)
Phase = Base.ImmutableDict(Phase, "SolveIterativeRefinement"                               => 33)
Phase = Base.ImmutableDict(Phase, "SolveIterativeRefinement_ForwardSubstitution"           => 331)
Phase = Base.ImmutableDict(Phase, "SolveIterativeRefinement_DiagonalSubstitution"          => 332)
Phase = Base.ImmutableDict(Phase, "SolveIterativeRefinement_BackwardSubstitution"          => 333)
Phase = Base.ImmutableDict(Phase, "ReleaseInternalMemory"                                  => 0)
Phase = Base.ImmutableDict(Phase, "ReleaseAllInternalMemory"                               => -1)


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

# Iparm parameters
#
# [I] -> input 
# [O] -> output 
# for iparm[i], where i is:
#
#  1:  [I]   Use default values
#  2:  [I]   Fill-in reducing ordering for the input matrix.
#  3:  [-]   Reserved. Set to zero.
#  4:  [I]   Preconditioned CGS/CG.
#  5:  [I]   User permutation.
#  6:  [I]   Write solution on x.
#  7:  [O]   Number of iterative refinement steps performed.
#  8:  [I]   Iterative refinement step.
#  9:  [-]   Reserved. Set to zero.
#  10: [I]   Pivoting perturbation.
#  11: [I]   Scaling vectors.
#  12: [I]   Solve with transposed or conjugate transposed matrix A.
#  13: [I]   Improved accuracy using (non-) symmetric weighted matching.
#  14: [O]   Number of perturbed pivots.
#  15: [O]   Peak memory on symbolic factorization.
#  16: [O]   Permanent memory on symbolic factorization.
#  17: [O]   Size of factors/Peak memory on numerical factorization and solution.
#  18: [I/O] Report the number of non-zero elements in the factors.
#  19: [I/O] Report number of floating point operations (in 106 floating point operations) that are necessary to factor the matrix A.
#  20: [O]   Report CG/CGS diagnostics.
#  21: [I]   Pivoting for symmetric indefinite matrices.
#  22: [O]   Inertia: number of positive eigenvalues.
#  23: [O]   Inertia: number of negative eigenvalues.
#  24: [I]   Parallel factorization control.
#  25: [I]   Parallel forward/backward solve control.
#  26: [-]   Reserved. Set to zero.
#  27: [I]   Matrix checker.
#  28: [I]   Single or double precision Intel MKL PARDISO.
#  29: [-]   Reserved. Set to zero.
#  30: [O]   Number of zero or negative pivots.
#  31: [I]   Partial solve and computing selected components of the solution vectors.
#  32: [-]   Reserved. Set to zero.
#  33: [-]   Reserved. Set to zero.
#  34: [I]   Optimal number of OpenMP threads for conditional numerical reproducibility (CNR) mode.
#  35: [I]   One- or zero-based indexing of columns and rows.
#  36: [I/O] Schur complement matrix computation control. 
#  37: [I]   Format for matrix storage.
#  38: [-]   Reserved. Set to zero.
#  39: [-]   Enable low rank update to accelerate factorization for multiple matrices with identical structure and similar values.
#  40: [-]   Reserved. Set to zero.
#  41: [-]   Reserved. Set to zero.
#  42: [-]   Reserved. Set to zero.
#  43: [-]   Control parameter for the computation of the diagonal of inverse matrix.
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
#  61: [-]   Reserved. Set to zero.
#  62: [-]   Reserved. Set to zero.
#  63: [O]   Size of the minimum OOC memory for numerical factorization and solution.
#  64: [-]   Reserved. Set to zero.

