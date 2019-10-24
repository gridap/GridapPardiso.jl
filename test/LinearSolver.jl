module LinearSolverTests

using Gridap
using GridapPardiso
using Test
using SparseArrays
using SparseMatricesCSR

tol = 1.0e-13
maxnz=20
maxrows=5
maxcols=5

#####################################################
# SparseMatrixCSC
# pardiso! (solving the transpose of the system above)
I = Vector{Int32}()
J = Vector{Int32}()
V = Vector{Float64}()
for (ik, jk, vk) in zip(rand(1:maxrows, maxnz), rand(1:maxcols, maxnz), rand(1:Float64(maxnz), maxnz))
    push_coo!(SparseMatrixCSC,I,J,V,ik,jk,vk)
end
finalize_coo!(SparseMatrixCSC,I,J,V,maxrows, maxcols)
A = sparse(I,J,V,maxrows, maxcols)
b = ones(size(A)[2])
x = similar(b)
ps = PardisoSolver(GridapPardiso.MTYPE_REAL_NON_SYMMETRIC, new_iparm(A), GridapPardiso.MSGLVL_VERBOSE)
ss = symbolic_setup(ps, A)
ns = numerical_setup(ss, A)
solve!(x, ns, b)
@test maximum(abs.(A*x-b)) < tol
test_linear_solver(ps, A, b, x)

if Int == Int64
    # pardiso_64! (solving the transpose of the system above)
    I = Vector{Int64}()
    J = Vector{Int64}()
    V = Vector{Float64}()
    for (ik, jk, vk) in zip(rand(1:maxrows, maxnz), rand(1:maxcols, maxnz), rand(1:Float64(maxnz), maxnz))
        push_coo!(SparseMatrixCSC,I,J,V,ik,jk,vk)
    end
    finalize_coo!(SparseMatrixCSC,I,J,V,maxrows, maxcols)
    A = sparse(I,J,V,maxrows, maxcols)
    b = ones(size(A)[2])
    x = similar(b)
    ps = PardisoSolver(GridapPardiso.MTYPE_REAL_NON_SYMMETRIC, new_iparm(A), GridapPardiso.MSGLVL_VERBOSE)
    ss = symbolic_setup(ps, A)
    ns = numerical_setup(ss, A)
    solve!(x, ns, b)
    @test maximum(abs.(A*x-b)) < tol
    test_linear_solver(ps, A, b, x)
end

#####################################################
# SparseMatrixCSR
# pardiso! 
I = Vector{Int32}()
J = Vector{Int32}()
V = Vector{Float64}()
for (ik, jk, vk) in zip(rand(1:maxrows, maxnz), rand(1:maxcols, maxnz), rand(1:Float64(maxnz), maxnz))
    push_coo!(SparseMatrixCSR,I,J,V,ik,jk,vk)
end
finalize_coo!(SparseMatrixCSR,I,J,V,maxrows, maxcols)
A = sparsecsr(I,J,V,maxrows, maxcols)
b = ones(size(A)[2])
x = similar(b)
ps = PardisoSolver(GridapPardiso.MTYPE_REAL_NON_SYMMETRIC, new_iparm(A), GridapPardiso.MSGLVL_VERBOSE)
ss = symbolic_setup(ps, A)
ns = numerical_setup(ss, A)
solve!(x, ns, b)
@test maximum(abs.(A*x-b)) < tol
test_linear_solver(ps, A, b, x)

if Int == Int64
    # pardiso_64!
    I = Vector{Int64}()
    J = Vector{Int64}()
    V = Vector{Float64}()
    for (ik, jk, vk) in zip(rand(1:maxrows, maxnz), rand(1:maxcols, maxnz), rand(1:Float64(maxnz), maxnz))
        push_coo!(SparseMatrixCSR,I,J,V,ik,jk,vk)
    end
    finalize_coo!(SparseMatrixCSR,I,J,V,maxrows, maxcols)
    A = sparsecsr(I,J,V,maxrows, maxcols)
    b = ones(size(A)[2])
    x = similar(b)
    ps = PardisoSolver(GridapPardiso.MTYPE_REAL_NON_SYMMETRIC, new_iparm(A), GridapPardiso.MSGLVL_VERBOSE)
    ss = symbolic_setup(ps, A)
    ns = numerical_setup(ss, A)
    solve!(x, ns, b)
    @test maximum(abs.(A*x-b)) < tol
    test_linear_solver(ps, A, b, x)
end

#####################################################
# SymSparseMatrixCSR
# pardiso! 
I = Vector{Int32}()
J = Vector{Int32}()
V = Vector{Float64}()
for (ik, jk, vk) in zip(rand(1:maxrows, maxnz), rand(1:maxcols, maxnz), rand(1:Float64(maxnz), maxnz))
    push_coo!(SymSparseMatrixCSR,I,J,V,ik,jk,vk)
end
finalize_coo!(SymSparseMatrixCSR,I,J,V,maxrows, maxcols)
A = symsparsecsr(I,J,V,maxrows, maxcols)
b = ones(size(A)[2])
x = similar(b)
ps = PardisoSolver(GridapPardiso.MTYPE_REAL_SYMMETRIC_INDEFINITE, new_iparm(A), GridapPardiso.MSGLVL_VERBOSE)
ss = symbolic_setup(ps, A)
ns = numerical_setup(ss, A)
solve!(x, ns, b)
@test maximum(abs.(A*x-b)) < tol
test_linear_solver(ps, A, b, x)

if Int == Int64
    # pardiso_64! (solving the transpose of the system above)
    I = Vector{Int64}()
    J = Vector{Int64}()
    V = Vector{Float64}()
    for (ik, jk, vk) in zip(rand(1:maxrows, maxnz), rand(1:maxcols, maxnz), rand(1:Float64(maxnz), maxnz))
        push_coo!(SymSparseMatrixCSR,I,J,V,ik,jk,vk)
    end
    finalize_coo!(SymSparseMatrixCSR,I,J,V,maxrows, maxcols)
    A = symsparsecsr(I,J,V,maxrows, maxcols)
    b = ones(size(A)[2])
    x = similar(b)
    ps = PardisoSolver(GridapPardiso.MTYPE_REAL_SYMMETRIC_INDEFINITE, new_iparm(A), GridapPardiso.MSGLVL_VERBOSE)
    ss = symbolic_setup(ps, A)
    ns = numerical_setup(ss, A)
    solve!(x, ns, b)
    @test maximum(abs.(A*x-b)) < tol
    test_linear_solver(ps, A, b, x)
end

end
