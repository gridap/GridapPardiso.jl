
new_pardiso_handle() = zeros(Int, 64)

new_iparm(::AbstractSparseMatrix{T, Ti}) where {T, Ti} = zeros(Ti, 64)

new_iparm() = zeros(Int, 64)

