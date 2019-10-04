module LinearSolverTests

using Gridap
using GridapPardiso
using Test
using SparseArrays

tol = 1.0e-13

# pardiso! (solving the transpose of the system above)
A = sparse(Vector{Int32}([1,2,3,4,5]),Vector{Int32}([1,2,3,4,5]),Vector{Float64}([1.0,2.0,3.0,4.0,5.0]))
b = ones(A.n)
x = similar(b)
ps = PardisoSolver(GridapPardiso.MTYPE_REAL_NON_SYMMETRIC, new_iparm(A), GridapPardiso.MSGLVL_VERBOSE)
ss = symbolic_setup(ps, A)
ns = numerical_setup(ss, A)
solve!(x, ns, b)
@test maximum(abs.(A'*x-b)) < tol
test_linear_solver(ps, A, b, x)

if Int == Int64
    # pardiso_64! (solving the transpose of the system above)
    A = sparse([1,2,3,4,5],[1,2,3,4,5],[1.0,2.0,3.0,4.0,5.0])
    b = ones(A.n)
    x = similar(b)
    ps = PardisoSolver(GridapPardiso.MTYPE_REAL_NON_SYMMETRIC, new_iparm(A), GridapPardiso.MSGLVL_VERBOSE)
    ss = symbolic_setup(ps, A)
    ns = numerical_setup(ss, A)
    solve!(x, ns, b)
    @test maximum(abs.(A'*x-b)) < tol
    test_linear_solver(ps, A, b, x)
end

end
