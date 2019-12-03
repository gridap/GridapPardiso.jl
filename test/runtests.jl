module GridapPardisoTests

using GridapPardiso
using Test

if GridapPardiso.MKL_PARDISO_LOADED[]
  @testset "Pardiso bindings" begin include("bindings.jl") end
  @testset "Linear solver" begin include("LinearSolver.jl") end
  @testset "FEM driver" begin include("femdriver.jl") end
  @testset "Non-Symmetric FEM driver" begin include("nonsymfemdriver.jl") end
end

end

