module FEMDriver

using Test
using Gridap
using GridapPardiso
using SparseMatricesCSR

tol = 1e-10

domain = (0,1,0,1,0,1)
partition = (10,10,10)

# Simple 2D data for debugging. TODO: remove when fixed.
domain = (0,1,0,1)
partition = (3,3)
model = CartesianDiscreteModel(domain,partition)

order=1
reffe = ReferenceFE(lagrangian,Float64,order)
V = FESpace(model,
            reffe,
            conformity=:H1,
            dirichlet_tags="boundary")
U = TrialFESpace(V)

trian = get_triangulation(model)
dΩ    = Measure(trian,2)

a(u,v)=∫(∇(v)⋅∇(u))dΩ
f(x)=x[1]*x[2]
l(v)=∫(v*f)dΩ

# With non-symmetric storage
assem = SparseMatrixAssembler(SparseMatrixCSR{1,Float64,Int},Vector{Float64},U,V)
op = AffineFEOperator(a,l,U,V,assem)

ls = PardisoSolver(op)
solver = LinearFESolver(ls)

uh = solve(solver,op)

x = get_free_dof_values(uh)
A = get_matrix(op)
b = get_vector(op)

r = A*x - b
@test maximum(abs.(r)) < tol

# With symmetric storage
assem = SparseMatrixAssembler(SymSparseMatrixCSR{1,Float64,Int},Vector{Float64},U,V)
op = AffineFEOperator(a,l,U,V,assem)

ls = PardisoSolver(op)
solver = LinearFESolver(ls)

uh = solve(solver,op)

x = get_free_dof_values(uh)
A = get_matrix(op)
b = get_vector(op)

r = A*x - b
@test maximum(abs.(r)) < tol

end #module
