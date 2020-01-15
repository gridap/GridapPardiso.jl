module GridapPardiso

using Libdl
using SparseArrays
using Gridap.Algebra

import Gridap.Algebra: LinearSolver
import Gridap.Algebra: SymbolicSetup, NumericalSetup
import Gridap.Algebra: symbolic_setup, numerical_setup, numerical_setup!
import Gridap.Algebra: solve, solve!

export PardisoSolver

export new_pardiso_handle
export new_iparm
export pardiso_data_type
export pardisoinit!
export pardiso!
export pardiso_64!
export pardiso_getdiag!

include("load_mkl.jl")

deps_jl = joinpath(@__DIR__, "..", "deps", "deps.jl")

if !isfile(deps_jl)
  s = """
  Package GridapPardiso not installed properly.
  """
  error(s)
end

include(deps_jl)

const pardisoinit_sym = Ref{Ptr}()
const pardiso_sym = Ref{Ptr}()
const pardiso_64_sym = Ref{Ptr}()
#const pardiso_getenv_sym = Ref{Ptr}()
#const pardiso_setenv_sym = Ref{Ptr}()
const pardiso_getdiag_sym = Ref{Ptr}()
#const pardiso_export_sym = Ref{Ptr}()
#const pardiso_handle_store_sym = Ref{Ptr}()
#const pardiso_handle_restore_sym = Ref{Ptr}()
#const pardiso_handle_delete_sym = Ref{Ptr}()
const MKL_PARDISO_LOADED = Ref(false)

function __init__()

  if MKL_FOUND 
    libmkl = load_mkl_gcc(mkllibdir,gcclibdir)

    pardisoinit_sym[] = Libdl.dlsym(libmkl,:pardisoinit)
    pardiso_sym[] = Libdl.dlsym(libmkl,:pardiso )
    pardiso_64_sym[] = Libdl.dlsym(libmkl,:pardiso_64 )
    #pardiso_getenv_sym[] = Libdl.dlsym(libmkl,:pardiso_getenv)
    #pardiso_setenv_sym[] = Libdl.dlsym(libmkl,:pardiso_setenv)
    pardiso_getdiag_sym[] = Libdl.dlsym(libmkl,:pardiso_getdiag)
    #pardiso_export_sym[] = Libdl.dlsym(libmkl,:pardiso_export)
    #pardiso_handle_store_sym[] = Libdl.dlsym(libmkl,:pardiso_handle_store)
    #pardiso_handle_restore_sym[] = Libdl.dlsym(libmkl,:pardiso_handle_restore)
    #pardiso_handle_delete_sym[] = Libdl.dlsym(libmkl,:pardiso_handle_delete)

    MKL_PARDISO_LOADED[] = true
  end
end

include("constructors.jl")

include("bindings.jl")

include("PardisoParameters.jl")

include("LinearSolver.jl")


end # module
