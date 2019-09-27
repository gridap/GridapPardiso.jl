using GridapPardiso
using Test
using SparseArrays

# Define linear system

mtype = 11

A = sparse([
  0. -2  3 0
  -2  4 -4 1
  -3  5  1 1
  1 -3 0 2 ])

n = A.n

b = Float64[1, 3, 2, 5]

x = zeros(Float64,n)

# Create the pardiso internal handler

pt = new_pardiso_handle()

# pardisoinit!

iparm = new_iparm()

pardisoinit!(pt,mtype,iparm)

# pardiso! (solving the transpose of the system above)

maxfct = 1
mnum = 1
phase = 13
a = A.nzval
ia = Vector{Int32}(A.colptr)
ja = Vector{Int32}(A.rowval)
perm = zeros(Int32,n)
nrhs = 1
msglvl = 0

err = pardiso!(
  pt,maxfct,mnum,mtype,phase,n,a,ia,ja,perm,nrhs,iparm,msglvl,b,x) 

tol = 1.0e-13

@test err == 0

@test maximum(abs.(A'*x-b)) < tol

# pardiso! (solving the system above)

iparm[12] = 2

err = pardiso!(
  pt,maxfct,mnum,mtype,phase,n,a,ia,ja,perm,nrhs,iparm,msglvl,b,x) 

@test err == 0

@test maximum(abs.(A*x-b)) < tol

@test pardiso_data_type(mtype,iparm) == Float64

# pardiso_getdiag!

iparm[56] = 1

pt = new_pardiso_handle()

err = pardiso!(
  pt,maxfct,mnum,mtype,phase,n,a,ia,ja,perm,nrhs,iparm,msglvl,b,x) 

df = zeros(Float64,n)
da = zeros(Float64,n)

err = pardiso_getdiag!(pt,df,da,mnum,mtype,iparm)

@test err == 0

err = pardiso_getdiag!(pt,df,da,mnum)

@test err == 0

# pardiso_64! (solving the transpose of the system above)

pt = new_pardiso_handle()

a = A.nzval
ia = A.colptr
ja = A.rowval
perm = zeros(Int64,n)
iparm = new_iparm_64()

err = pardiso_64!(
  pt,maxfct,mnum,mtype,phase,n,a,ia,ja,perm,nrhs,iparm,msglvl,b,x) 

@test err == 0

# pardiso_64! (solving the transpose of the system above)

@test maximum(abs.(A'*x-b)) < tol


# pardiso_64! (solving the transpose of the system above)
A = sparse([1,2,3,4,5],[1,2,3,4,5],[1.0,2.0,3.0,4.0,5.0])
b = ones(A.n)
x = similar(b)
ps = PardisoSolver(11, new_iparm(), 1)
ss = symbolic_setup(ps, A)
ns = numerical_setup(ss, A)
solve!(x, ns, b)
@test maximum(abs.(A'*x-b)) < tol

