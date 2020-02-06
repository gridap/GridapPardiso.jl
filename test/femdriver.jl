module FEMDriver

using Test
using Gridap
using GridapPardiso

tol = 1e-10

domain = (0,1,0,1,0,1)
partition = (10,10,10)

# Simple 2D data for debugging. TODO: remove when fixed.
domain = (0,1,0,1)
partition = (3,3)
model = CartesianDiscreteModel(domain,partition)

V = TestFESpace(reffe=:Lagrangian, order=1, valuetype=Float64,
  conformity=:H1, model=model, dirichlet_tags="boundary")

U = TrialFESpace(V)

trian = get_triangulation(model)
quad = CellQuadrature(trian,2)

t_Ω = AffineFETerm(
  (v,u) -> inner(∇(v),∇(u)),
  (v) -> inner(v, (x) -> x[1]*x[2] ),
  trian, quad)

# With non-symmetric storage

op = AffineFEOperator(SparseMatrixCSR{1,Float64,Int},V,U,t_Ω)

ls = PardisoSolver()
solver = LinearFESolver(ls)

uh = solve(solver,op)

x = get_free_values(uh)
A = get_matrix(op)
b = get_vector(op)

r = A*x - b
@test maximum(abs.(r)) < tol

# With symmetric storage

op = AffineFEOperator(SymSparseMatrixCSR{1,Float64,Int},V,U,t_Ω)

ls = PardisoSolver(GridapPardiso.MTYPE_REAL_SYMMETRIC_INDEFINITE)
solver = LinearFESolver(ls)

uh = solve(solver,op)

x = get_free_values(uh)
A = get_matrix(op)
b = get_vector(op)

r = A*x - b
@test maximum(abs.(r)) < tol

end #module
