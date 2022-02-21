module LinearSolverTests

using Gridap.Algebra
using GridapPardiso
using Test
using SparseArrays
using SparseMatricesCSR


tol = 1.0e-13

#####################################################
# SparseMatrixCSC
#####################################################
#
# Matrix from Intel MKL Pardiso examples
#
#        DATA ia /1,4,6,9,12,14/
#        DATA ja
#     1  /1,2,  4,
#     2   1,2,
#     3       3,4,5,
#     4   1,  3,4,
#     5     2,    5/
#        DATA a
#     1  / 1.d0,-1.d0,      -3.d0,
#     2   -2.d0, 5.d0,
#     3                4.d0, 6.d0, 4.d0,
#     4   -4.d0,       2.d0, 7.d0,
#     5          8.d0,            -5.d0/
#####################################################
I_ = [1,1,1,2,2,3,3,3,4,4,4,5,5]
J_ = [1,2,4,1,2,3,4,5,1,3,4,2,5]
V_ = [1,-1,-3,-2,5,4,6,4,-4,2,7,8,-5]
rows = 5
cols = 5

# pardiso!
I = Vector{Int32}(); J = Vector{Int32}(); V = Vector{Float64}()
for (ik, jk, vk) in zip(I_,J_,V_)
    push_coo!(SparseMatrixCSC,I,J,V,ik,jk,vk)
end
finalize_coo!(SparseMatrixCSC,I,J,V,rows, cols)
A = sparse(I,J,V,rows,cols)
b = ones(size(A)[2])
x = similar(b)
ps = PardisoSolver(GridapPardiso.MTYPE_REAL_NON_SYMMETRIC, new_iparm(), GridapPardiso.MSGLVL_VERBOSE)
ss = symbolic_setup(ps, A)
ns = numerical_setup(ss, A)
solve!(x, ns, b)
@test maximum(abs.(A*x-b)) < tol
test_linear_solver(ps, A, b, x)

if Int == Int64
    # pardiso_64!
    I = Vector{Int64}(); J = Vector{Int64}(); V = Vector{Float64}()
    for (ik, jk, vk) in zip(I_,J_,V_)
        push_coo!(SparseMatrixCSC,I,J,V,ik,jk,vk)
    end
    finalize_coo!(SparseMatrixCSC,I,J,V,rows, cols)
    A = sparse(I,J,V,rows, cols)
    b = ones(size(A)[2])
    x = similar(b)
    ps = PardisoSolver(GridapPardiso.MTYPE_REAL_NON_SYMMETRIC, new_iparm(), GridapPardiso.MSGLVL_VERBOSE)
    ss = symbolic_setup(ps, A)
    ns = numerical_setup(ss, A)
    solve!(x, ns, b)
    @test maximum(abs.(A*x-b)) < tol
    test_linear_solver(ps, A, b, x)
end

I = Vector{Int32}(); J = Vector{Int32}(); V = Vector{Complex{Float64}}()
for (ik, jk, vk) in zip(I_,J_,V_)
    push_coo!(SparseMatrixCSC,I,J,V,ik,jk,vk)
end
finalize_coo!(SparseMatrixCSC,I,J,V,rows, cols)
A = sparse(I,J,V,rows,cols)
b = ones(Complex{Float64},size(A)[2])
x = similar(b)
ps = PardisoSolver(msglvl=GridapPardiso.MSGLVL_VERBOSE)
ss = symbolic_setup(ps, A)
ns = numerical_setup(ss, A)
solve!(x, ns, b)
@test maximum(abs.(A*x-b)) < tol
test_linear_solver(ps, A, b, x)

#####################################################
# SparseMatrixCSR
#####################################################
#
# Matrix from Intel MKL Pardiso examples
#
#        DATA ia /1,4,6,9,12,14/
#        DATA ja
#     1  /1,2,  4,
#     2   1,2,
#     3       3,4,5,
#     4   1,  3,4,
#     5     2,    5/
#        DATA a
#     1  / 1.d0,-1.d0,      -3.d0,
#     2   -2.d0, 5.d0,
#     3                4.d0, 6.d0, 4.d0,
#     4   -4.d0,       2.d0, 7.d0,
#     5          8.d0,            -5.d0/
#####################################################
I_ = [1,1,1,2,2,3,3,3,4,4,4,5,5]
J_ = [1,2,4,1,2,3,4,5,1,3,4,2,5]
V_ = [1,-1,-3,-2,5,4,6,4,-4,2,7,8,-5]
rows = 5
cols = 5

# pardiso!

for Bi in (0,1)
    I = Vector{Int32}(); J = Vector{Int32}(); V = Vector{Float64}()
    for (ik, jk, vk) in zip(I_,J_,V_)
        push_coo!(SparseMatrixCSR,I,J,V,ik,jk,vk)
    end
    finalize_coo!(SparseMatrixCSR,I,J,V,rows,cols)
    A = sparsecsr(Val{Bi}(),I,J,V,rows,cols)
    b = ones(size(A)[2])
    x = similar(b)
    ps = PardisoSolver(GridapPardiso.MTYPE_REAL_NON_SYMMETRIC, new_iparm(), GridapPardiso.MSGLVL_VERBOSE)
    ss = symbolic_setup(ps, A)
    ns = numerical_setup(ss, A)
    solve!(x, ns, b)
    @test maximum(abs.(A*x-b)) < tol
    test_linear_solver(ps, A, b, x)

    if Int == Int64
        # pardiso_64!
        I = Vector{Int64}(); J = Vector{Int64}(); V = Vector{Float64}()
        for (ik, jk, vk) in zip(I_,J_,V_)
            push_coo!(SparseMatrixCSR,I,J,V,ik,jk,vk)
        end
        finalize_coo!(SparseMatrixCSR,I,J,V,rows,cols)
        A = sparsecsr(Val{Bi}(),I,J,V,rows,cols)
        b = ones(size(A)[2])
        x = similar(b)
        ps = PardisoSolver(GridapPardiso.MTYPE_REAL_NON_SYMMETRIC, new_iparm(), GridapPardiso.MSGLVL_VERBOSE)
        ss = symbolic_setup(ps, A)
        ns = numerical_setup(ss, A)
        solve!(x, ns, b)
        @test maximum(abs.(A*x-b)) < tol
        test_linear_solver(ps, A, b, x)
    end

    I = Vector{Int32}(); J = Vector{Int32}(); V = Vector{Complex{Float64}}()
    for (ik, jk, vk) in zip(I_,J_,V_)
        push_coo!(SparseMatrixCSR,I,J,V,ik,jk,vk)
    end
    finalize_coo!(SparseMatrixCSR,I,J,V,rows,cols)
    A = sparsecsr(Val{Bi}(),I,J,V,rows,cols)
    b = ones(Complex{Float64},size(A)[2])
    x = similar(b)
    ps = PardisoSolver(msglvl=GridapPardiso.MSGLVL_VERBOSE)
    ss = symbolic_setup(ps, A)
    ns = numerical_setup(ss, A)
    solve!(x, ns, b)
    @test maximum(abs.(A*x-b)) < tol
    test_linear_solver(ps, A, b, x)
end

#####################################################
# SymSparseMatrixCSR
#####################################################
#
# Matrix from Intel MKL Pardiso examples
#
# ia = (/ 1, 5, 8, 10, 12, 15, 17, 18, 19 /)
# ja = (/ 1,    3,       6, 7,    &
#            2, 3,    5,          &
#               3,             8, &
#                  4,       7,    &
#                     5, 6, 7,    &
#                        6,    8, &
#                           7,    &
#                              8 /)
# a = (/ 7.d0,        1.d0,             2.d0, 7.d0,        &
#              -4.d0, 8.d0,       2.d0,                    &
#                     1.d0,                         5.d0,  &
#                           7.d0,             9.d0,        &
#                                 5.d0, 1.d0, 5.d0,        &
#                                      -1.d0,       5.d0,  &
#                                            11.d0,        &
#                                                   5.d0 /)
#####################################################
I_ = [1,1,1,1,2,2,2,3,3,4,4,5,5,5,6,6,7,8]
J_ = [1,3,6,7,2,3,5,3,8,4,7,5,6,7,6,8,7,8]
V_ = [7,1,2,7,-4,8,2,1,5,7,9,5,1,5,-1,5,11,5]
rows = 8
cols = 8

# pardiso!

for Bi in (0,1)
    I = Vector{Int32}(); J = Vector{Int32}(); V = Vector{Float64}()
    for (ik, jk, vk) in zip(I_,J_,V_)
        push_coo!(SymSparseMatrixCSR,I,J,V,ik,jk,vk)
    end
    finalize_coo!(SymSparseMatrixCSR,I,J,V,rows,cols)
    A = symsparsecsr(Val{Bi}(),I,J,V,rows,cols)
    b = ones(size(A)[2])
    x = similar(b)
    ps = PardisoSolver(GridapPardiso.MTYPE_REAL_SYMMETRIC_INDEFINITE, new_iparm(), GridapPardiso.MSGLVL_VERBOSE)
    ss = symbolic_setup(ps, A)
    ns = numerical_setup(ss, A)
    solve!(x, ns, b)
    @test maximum(abs.(A*x-b)) < tol
    test_linear_solver(ps, A, b, x)

    if Int == Int64
        # pardiso_64!
        I = Vector{Int64}(); J = Vector{Int64}(); V = Vector{Float64}()
        for (ik, jk, vk) in zip(I_,J_,V_)
            push_coo!(SymSparseMatrixCSR,I,J,V,ik,jk,vk)
        end
        finalize_coo!(SymSparseMatrixCSR,I,J,V,rows,cols)
        A = symsparsecsr(Val{Bi}(),I,J,V,rows,cols)
        b = ones(size(A)[2])
        x = similar(b)
        ps = PardisoSolver(GridapPardiso.MTYPE_REAL_SYMMETRIC_INDEFINITE, new_iparm(), GridapPardiso.MSGLVL_VERBOSE)
        ss = symbolic_setup(ps, A)
        ns = numerical_setup(ss, A)
        solve!(x, ns, b)
        @test maximum(abs.(A*x-b)) < tol
        test_linear_solver(ps, A, b, x)
    end

    I = Vector{Int32}(); J = Vector{Int32}(); V = Vector{Complex{Float64}}()
    for (ik, jk, vk) in zip(I_,J_,V_)
        push_coo!(SymSparseMatrixCSR,I,J,V,ik,jk,vk)
    end
    finalize_coo!(SymSparseMatrixCSR,I,J,V,rows,cols)
    A = symsparsecsr(Val{Bi}(),I,J,V,rows,cols)
    b = ones(Complex{Float64},size(A)[2])
    x = similar(b)
    ps = PardisoSolver(msglvl=GridapPardiso.MSGLVL_VERBOSE)
    ss = symbolic_setup(ps, A)
    ns = numerical_setup(ss, A)
    solve!(x, ns, b)
    @test maximum(abs.(A*x-b)) < tol
    test_linear_solver(ps, A, b, x)

end

end
