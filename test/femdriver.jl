module FEMDriver

using Test
using Gridap
using GridapPardiso

model = CartesianDiscreteModel(
  domain=(0,1,0,1,0,1), partition=(10,10,10))

fespace = FESpace(
  reffe=:Lagrangian, order=1, valuetype=Float64,
  conformity=:H1, model=model, diritags="boundary")

V = TestFESpace(fespace)
U = TrialFESpace(fespace,0.0)

trian = Triangulation(model)
quad = CellQuadrature(trian,degree=2)

t_Ω = AffineFETerm(
  (v,u) -> inner(∇(v),∇(u)),
  (v) -> inner(v, (x) -> x[1]*x[2] ),
  trian, quad)

op = LinearFEOperator(V,U,t_Ω)

ls = PardisoSolver()
solver = LinearFESolver(ls)

uh = solve(solver,op)

x = free_dofs(uh)
A = op.mat
b = op.vec

r = A*x - b

tol = 1e-10
@test maximum(abs.(r)) < tol


end #module
