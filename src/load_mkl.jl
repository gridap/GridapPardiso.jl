
function load_mkl_gcc(mkllibdir,gcclibdir)

  lmkl_intel_lp64 = joinpath(mkllibdir,"libmkl_intel_lp64")
  lmkl_gnu_thread = joinpath(mkllibdir,"libmkl_gnu_thread")
  lmkl_core       = joinpath(mkllibdir,"libmkl_core")
  lgomp           = joinpath(gcclibdir,"libgomp")

  flags = Libdl.RTLD_LAZY | Libdl.RTLD_DEEPBIND | Libdl.RTLD_GLOBAL
  
  Libdl.dlopen(lgomp, flags)
  Libdl.dlopen(lmkl_core, flags)
  Libdl.dlopen(lmkl_gnu_thread, flags)
  libmkl = Libdl.dlopen(lmkl_intel_lp64, flags)

  libmkl

end

