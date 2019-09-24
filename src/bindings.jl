
macro check_if_loaded()
  quote
    if ! MKL_PARDISO_LOADED[]
      error("MKL pardiso is not properly loaded")
    end
  end
end

function pardisoinit!(
  pt::Vector{Int},
  mtype::Integer,
  iparm::Vector{Int32})

  @check_if_loaded

  ccall(
    pardisoinit_sym[],
    Cvoid, (
      Ptr{Int},
      Ptr{Int32},
      Ptr{Int32}),
    pt,
    Ref(Int32(mtype)),
    iparm)

end

function pardiso!(
  pt::Vector{Int},
  maxfct::Integer,
  mnum::Integer,
  mtype::Integer,
  phase::Integer,
  n::Integer,
  a::Vector{T},
  ia::Vector{Int32},
  ja::Vector{Int32},
  perm::Vector{Int32},
  nrhs::Integer,
  iparm::Vector{Int32},
  msglvl::Integer,
  b::Vector{T},
  x::Vector{T}) where T

  @check_if_loaded

  @assert T == pardiso_data_type(mtype,iparm)

  err = Ref(zero(Int32))

  ccall(
    pardiso_sym[],
    Cvoid, (
      Ptr{Int},
      Ptr{Int32},
      Ptr{Int32},
      Ptr{Int32},
      Ptr{Int32},
      Ptr{Int32},
      Ptr{Cvoid},
      Ptr{Int32},
      Ptr{Int32},
      Ptr{Int32},
      Ptr{Int32},
      Ptr{Int32},
      Ptr{Int32},
      Ptr{Cvoid},
      Ptr{Cvoid},
      Ptr{Int32}),
    pt,
    Ref(Int32(maxfct)),
    Ref(Int32(mnum)),
    Ref(Int32(mtype)),
    Ref(Int32(phase)),
    Ref(Int32(n)),
    a,
    ia,
    ja,
    perm,
    Ref(Int32(nrhs)),
    iparm,
    Ref(Int32(msglvl)),
    b,
    x,
    err)

  return Int(err[])

end

function pardiso_64!(
  pt::Vector{Int},
  maxfct::Integer,
  mnum::Integer,
  mtype::Integer,
  phase::Integer,
  n::Integer,
  a::Vector{T},
  ia::Vector{Int64},
  ja::Vector{Int64},
  perm::Vector{Int64},
  nrhs::Integer,
  iparm::Vector{Int64},
  msglvl::Integer,
  b::Vector{T},
  x::Vector{T}) where T

  @check_if_loaded

  @assert T == pardiso_data_type(mtype,iparm)

  err = Ref(zero(Int64))

  ccall(
    pardiso_64_sym[],
    Cvoid, (
      Ptr{Int},
      Ptr{Int64},
      Ptr{Int64},
      Ptr{Int64},
      Ptr{Int64},
      Ptr{Int64},
      Ptr{Cvoid},
      Ptr{Int64},
      Ptr{Int64},
      Ptr{Int64},
      Ptr{Int64},
      Ptr{Int64},
      Ptr{Int64},
      Ptr{Cvoid},
      Ptr{Cvoid},
      Ptr{Int64}),
    pt,
    Ref(Int64(maxfct)),
    Ref(Int64(mnum)),
    Ref(Int64(mtype)),
    Ref(Int64(phase)),
    Ref(Int64(n)),
    a,
    ia,
    ja,
    perm,
    Ref(Int64(nrhs)),
    iparm,
    Ref(Int64(msglvl)),
    b,
    x,
    err)

  return Int(err[])

end

function pardiso_getdiag!(
  pt::Vector{Int},
  df::Vector{T},
  da::Vector{T},
  mnum::Integer,
  mtype::Integer,
  iparm::Vector{<:Integer}) where T

  @assert T == pardiso_data_type(mtype,iparm)

  pardiso_getdiag!(pt,df,da,mnum)

end

function pardiso_getdiag!(
  pt::Vector{Int},
  df::Vector{T},
  da::Vector{T},
  mnum::Integer) where T

  @check_if_loaded

  err = Ref(zero(Int32))

  ccall(
   pardiso_getdiag_sym[],
   Cvoid,(
     Ptr{Int},
     Ptr{Cvoid},
     Ptr{Cvoid},
     Ptr{Int32},
     Ptr{Int32}),
   pt,
   df,
   da,
   Ref(Int32(mnum)),
   err)

  return Int(err[])

end

function pardiso_data_type(mtype::Integer,iparm::Vector{<:Integer})

  # Rules taken from
  # https://software.intel.com/en-us/mkl-developer-reference-fortran-pardiso-data-type

  T::DataType = Any

  if mtype in (1,2,-2,11)
    if iparm[28] == 0
      T = Float64
    else
      T = Float32
    end
  elseif mtype in (3,6,13,4,-4)
    if iparm[28] == 0
      T = Complex{Float64}
    else
      T = Complex{Float32}
    end
  else
    error("Unknown matrix type: mtype = $mtype")
  end

  T

end

