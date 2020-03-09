# store current variables
const stored_init_rho = [0.5, 0.5]
const stored_init_lambda = [[0.5,], [0.5,]]


# univariate LBFGSB optimizer
const optimizer_lbfgs = L_BFGS_B(1, 17)
const stored_bounds = zeros(3, 1)
stored_bounds[1,1] = 1  # represents the type of bounds imposed on the variables:
                 #  0->unbounded, 1->only lower bound, 2-> both lower and upper bounds, 3->only upper bound
stored_bounds[2,1] = 0.0
stored_bounds[3,1] = Inf

# LBFGSB optimizer for marginal_projection_with_constraint
const optimizer_lbfgs_cs = L_BFGS_B(1024, 17)


function solve_p_given_abk(a::AbstractVector, b, k::Integer, ids::AbstractVector{<:Integer})
    
    n = length(a)
    T = eltype(a)
   
    raw = a .- b

    sorted_raw = raw[ids]
    neg_cut_id = findfirst(x -> x < 0, sorted_raw)      # first negative
    if isnothing(neg_cut_id)
        neg_cut_id = n + 1
        sraw_neg = T[]
        n_neg = 0
    else
        sraw_neg = @views sorted_raw[neg_cut_id:end]
        n_neg = length(sraw_neg)
    end

    sum_p = sum(@views sorted_raw[1:neg_cut_id-1])
    if  sorted_raw[1] <= sum_p / k
        return max.(0, raw)
    end

    # starting
    i = 1                   # #x in sum_p_else
    j = neg_cut_id - 2      # #x in sum_p_edge
    l = 0                   # #x in sum_p_neg

    sum_p_edge = sorted_raw[1]
    sum_p_else = sum_p - sorted_raw[1]
    sum_p_neg = zero(T)

    # solution
    sol_r = zero(T)
    p_add = zero(T)

    while true

        next_r = (sum_p_else + sum_p_neg + sum_p_edge * (j + l) / (k - i)) / (k - i + (i * (j + l)) / (k - i)) |> T
        p_else_add = (sum_p_edge - i * next_r) / (k - i) |> T

        if l < n_neg && p_else_add + sraw_neg[l + 1] > 0 
            sum_p_neg += sraw_neg[l + 1]
            l += 1
            continue
        end

        if  i == k-1  || next_r + 1e-6 >= sorted_raw[i + 1] + p_else_add     # 1e-6 for floating point error
            sol_r = next_r
            p_add = p_else_add

            break
        end
    
        sum_p_edge += sorted_raw[i + 1]
        sum_p_else -= sorted_raw[i + 1]
        i += 1
        j -= 1

    end 

    res = min.(max.(raw .+ p_add, 0), sol_r)
    return res
    
end

function solve_p_given_abk(a::AbstractVector, b, k::Integer)
    raw = a .- b
    ids = sortperm(raw, rev = true)

    return solve_p_given_abk(a, b, k, ids)
end


function obj_rho(rho_arr::AbstractVector, A::AbstractMatrix, IDS::AbstractMatrix)
    n = size(A, 1)
    T = eltype(rho_arr)
    rho = rho_arr[1]

    # to store best P
    best_P = zeros(T, n, n)

    obj = zero(T)
    grad = zero(T)

    for k = 1:n
        a = A[:, k]
        b = rho / k
        ids = IDS[:, k]

        opt_p = solve_p_given_abk(a, b, k, ids)
        best_P[:, k] = opt_p

        obj += sum(opt_p) * rho / k + sum((opt_p - a) .^ 2) / 2
        grad += sum(opt_p) / k
    end

    obj -= rho
    grad -= 1

    return -obj, [-grad], best_P
end

function marginal_projection(A::AbstractMatrix; init_storage_id = 1)

    n = size(A, 1)
    T = eltype(A)

    IDS = zeros(Int, n, n)
    for i = 1:n
        IDS[:, i] = sortperm(A[:,i], rev = true)
    end

    init_rho = stored_init_rho[init_storage_id]

    _, xout = optimizer_lbfgs(rho -> obj_rho(rho, A, IDS)[1:2], [init_rho], stored_bounds, maxiter=20, factr=1e8, pgtol=1e-3)

    opt_P = obj_rho(xout, A, IDS)[3]
    stored_init_rho[init_storage_id] = xout[1]

    opt_P = min.(max.(opt_P, zero(T)), one(T))

    return opt_P
end


## marginal projection with constraints

function obj_rho_lambda(rho_lambda::AbstractVector, A::AbstractMatrix, 
    B_list::AbstractVector{<:AbstractMatrix}, c_list::AbstractVector{<:Number}, tau_list::AbstractVector{<:Number})

    n = size(A, 1)
    ncs = length(B_list)
    T = eltype(rho_lambda)

    rho = rho_lambda[1]
    lda = rho_lambda[2:end]

    B_wsum = zeros(n, n)
    for j = 1:ncs
        B_wsum += lda[j] * B_list[j]
    end

    # to store best P
    best_P = zeros(T, n, n)

    obj = zero(T)
    grad = zeros(T, length(rho_lambda))

    for k = 1:n
        a = A[:, k]
        b = (rho / k) .- B_wsum[:, k]

        opt_p = solve_p_given_abk(a, b, k)
        best_P[:, k] = opt_p

        obj += sum(opt_p) * rho / k + sum((opt_p - a) .^ 2) / 2
        grad[1] += sum(opt_p) / k
    end

    obj -= rho
    for j = 1:ncs
        obj += lda[j] * (tau_list[j] - c_list[j])
    end

    grad[1] -= 1
    for j = 1:ncs
        grad[j+1] += tau_list[j] - c_list[j] - sum(B_list[j] .* best_P)
    end

    return -obj, -grad, best_P
end

function marginal_projection_with_constraint(A::AbstractMatrix, B_list::AbstractVector{<:AbstractMatrix}, 
    c_list::AbstractVector{<:Number}, tau_list::AbstractVector{<:Number}, alg = :lbfgs; args...)

    if alg == :lbfgs
        return lbfgs_projection_with_constraint(A, B_list, c_list, tau_list; args...)
    elseif alg == :dykstra
        return dykstra_projection_with_constraint(A, B_list, c_list, tau_list; args...)
    else
        return lbfgs_projection_with_constraint(A, B_list, c_list, tau_list; args...)
    end
end

function lbfgs_projection_with_constraint(A::AbstractMatrix, B_list::AbstractVector{<:AbstractMatrix}, 
    c_list::AbstractVector{<:Number}, tau_list::AbstractVector{<:Number}; init_storage_id = 1)

    n = size(A, 1)
    T = eltype(A)
    ncs = length(B_list)

    # pull stored parameters
    init_rho = stored_init_rho[init_storage_id]
    if length(stored_init_lambda[init_storage_id]) != ncs
        stored_init_lambda[init_storage_id] = ones(ncs) * 0.5
    end
    init_lambda = stored_init_lambda[init_storage_id]
    init_pars = [init_rho; init_lambda]
    
    # bounds
    bounds = zeros(3, ncs + 1)
    bounds[1,:] .= 1  # represents the type of bounds imposed on the variables:
                      #  0->unbounded, 1->only lower bound, 2-> both lower and upper bounds, 3->only upper bound
    bounds[2,:] .= 0.0
    bounds[3,:] .= Inf

    _, xout = optimizer_lbfgs_cs(rho_lambda -> obj_rho_lambda(rho_lambda, A, B_list, c_list, tau_list)[1:2], 
                    init_pars, bounds, maxiter=20, factr=1e8, pgtol=1e-3)

    opt_P = obj_rho_lambda(xout, A, B_list, c_list, tau_list)[3]
    opt_P = min.(max.(opt_P, zero(T)), one(T))

    # store pars
    stored_init_rho[init_storage_id] = xout[1]
    stored_init_lambda[init_storage_id] = xout[2:end]

    return opt_P
end


function dykstra_projection_with_constraint(A::AbstractMatrix, B_list::AbstractVector{<:AbstractMatrix}, 
    c_list::AbstractVector{<:Number}, tau_list::AbstractVector{<:Number}; args...)

    n = size(A, 1)
    T = eltype(A)
    ncs = length(B_list)

    Zs = [zeros(T, n, n) for _ = 1:ncs+1]
    P = copy(A)

    niter = 10

    for iter = 1:niter
        for ic = 1:ncs+1
            if ic == ncs+1
                P_next = marginal_projection(P + Zs[ic]; args...)
            else
                P_next = exact_with_constraint(P + Zs[ic], B_list[ic], c_list[ic], tau_list[ic])
            end
            Zs[ic] = P + Zs[ic] - P_next
            P = P_next
        end
    end

    return P
end


## max sum k largest, non neg
# compute value
function sumlargest(a::AbstractVector, k::Integer)
    sa = sort(a, rev=true)
    return sum(sa[1:k])
end

function sumlargest(A::AbstractMatrix)
    n = size(A, 1)
    sl = zeros(n)
    for i = 1:n
        sl[i] = sumlargest(A[:,i], i)
    end
    return sl
end

function max_sumlargest(A::AbstractMatrix, non_neg = true)
    sl = sumlargest(A)
    m = maximum(sl)
    if non_neg
        m = max(0, m)
    end
    return m
end



## Proximal functions
function prox_max_sumlargest(A::AbstractMatrix; args...)
    return A - marginal_projection(A; args...)
end

function prox_max_sumlargest(A::AbstractMatrix, rho::Real; args...)
    return A - marginal_projection(A .* rho; args...) ./ rho
end

function prox_max_sumlargest_with_constraint(A::AbstractMatrix, B_list::AbstractVector{<:AbstractMatrix}, 
    c_list::AbstractVector{<:Number}, tau_list::AbstractVector{<:Number}; args...)

    return A - marginal_projection_with_constraint(A, B_list, c_list, tau_list; args...)
end

function prox_max_sumlargest_with_constraint(A::AbstractMatrix, rho::Real, B_list::AbstractVector{<:AbstractMatrix}, 
    c_list::AbstractVector{<:Number}, tau_list::AbstractVector{<:Number}; args...)

    return A - marginal_projection_with_constraint(A .* rho, B_list, c_list, tau_list; args...) ./ rho
end


# Best P: min_P <B, P>,   rev =true for max_P
function find_opt_p(B, rev = false)
    n = size(B, 1)
    T = eltype(B)

    SB = zeros(Int, n, n)
    for i = 1:n
        SB[:, i] = sortperm(B[:,i], rev = rev)
    end

    best_i = 0
    best_sum = zero(T)
    for i = 1:n
        sum_i = sum(B[SB[1:i, i], i])
        if rev && sum_i > best_sum
            best_i = i
            best_sum = sum_i
        elseif !rev && sum_i < best_sum
            best_i = i
            best_sum = sum_i
        end
    end
        
    P = zeros(T, n, n)
    if best_i > 0
        P[SB[1:best_i, best_i], best_i] .= one(T)
    end

    return P, best_sum
end


## reset storage
function reset_projection_storage()
    stored_init_rho[1] = 0.5
    stored_init_rho[2] = 0.5
    stored_init_lambda[1] = [0.5,]
    stored_init_lambda[2] = [0.5,]
end