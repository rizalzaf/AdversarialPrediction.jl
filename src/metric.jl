# type for metric data
mutable struct MetricData{TE <: CM_Expression, TT <: Tuple}
    metric_expr::TE                 # expression defined by the metric
    constraint_list::TT    # list of constraints expression
    multiplier_pq::ConstantOverPQ
    
    B_list::Vector{Matrix{Float32}}
    c_list::Vector{Float32}
    tau_list::Vector{Float32}

    # default values
    MetricData() = new{EXPR_UnaryIdentity,Tuple{}}(
        EXPR_UnaryIdentity(0f0), tuple(), ConstantOverPQ(), 
        Vector{Matrix{Float32}}(), Vector{Float32}(), Vector{Float32}())

    MetricData(metric_expr, constraint_list) = new{typeof(metric_expr),typeof(constraint_list)}(
        metric_expr, constraint_list, ConstantOverPQ(), 
        Vector{Matrix{Float32}}(), Vector{Float32}(), Vector{Float32}())

    MetricData(metric_expr, constraint_list, multiplier_pq) = new{typeof(metric_expr),typeof(constraint_list)}(
        metric_expr, constraint_list, multiplier_pq, 
        Vector{Matrix{Float32}}(), Vector{Float32}(), Vector{Float32}())
end

# data for ADMM optimization
mutable struct OptData
    # n sample
    n::Int

    # f(AQB + QC +D) + <Q,E> + ct     
    A::Matrix{Float32}                 
    B::Matrix{Float32}
    C::Matrix{Float32}
    D::Matrix{Float32}                 
    E::Matrix{Float32}
    ct::Float32             # constant

    # stored eigen decomposition matrix
    CIinv::Matrix{Float32} 
    BC_Cinv::Matrix{Float32}
    UA::Matrix{Float32}     
    UBC::Matrix{Float32}
    UAinv::Matrix{Float32}
    UBCinv::Matrix{Float32}
    sabc1::Matrix{Float32}

    function OptData()
        MO = zeros(Float32, 0, 0)
        new(0, MO, MO, MO, MO, MO, 0f0, MO, MO, MO, MO, MO, MO, MO)
    end 

    function OptData(n, A, B, C, D, E, ct, CIinv, BC_Cinv, UA, UBC, UAInv, UBCinv, sabc1)
        new(n, A, B, C, D, E, ct, CIinv, BC_Cinv, UA, UBC, UAInv, UBCinv, sabc1)
    end
end


##### METRIC ####
# abstract typr for performance metric
abstract type PerformanceMetric end

## functions that needed to be implemented
function define(::Type{<:PerformanceMetric}, C::ConfusionMatrix, args...)
    return EXPR_UnaryIdentity()
end

# optional function to overload
function constraint(::Type{<:PerformanceMetric}, C::ConfusionMatrix, args...)
    return nothing
end

# macro for defining metric
macro metric(name)
    return quote
        mutable struct $name{T <: MetricData} <: PerformanceMetric
            info::MetricInfo
            data::T
            opt_data::OptData

            function $name() 
                info, data = initialize($name)
                new{typeof(data)}(info, data, OptData())
            end
        end
    end
end

Base.show(io::IO, x::PerformanceMetric) = print(io, string(typeof(x).name))

# macro for defining metric with arguments
macro metric(name, args...)
    return quote
        mutable struct $name{T <: MetricData} <: PerformanceMetric
            info::MetricInfo
            data::T
            opt_data::OptData
            $([arg for arg in args]...)

            function $name($([arg for arg in args]...)) 
                info, data = initialize($name, $([arg for arg in args]...))
                new{typeof(data)}(info, data, OptData(), $([arg for arg in args]...))
            end
        end

        function Base.show(io::IO, x::$(esc(name))) 
            print(io, string(typeof(x).name, "(", $(:(x.$(args[1]))),
                $([:("," * string(x.$(args[2]))) for i = 2:length(args)]...)),
                ")" )
        end
    end
end


### METRIC: Internal functions ###

# enforcing special cases
function special_case_positive!(pm::PerformanceMetric) 
    pm.info.special_case_positive = true
end

function special_case_negative!(pm::PerformanceMetric) 
    pm.info.special_case_negative = true
end

function cs_special_case_positive!(pm::PerformanceMetric, val::Vector{<:Bool})
    pm.info.cs_special_case_positive_list = val
end

function cs_special_case_positive!(pm::PerformanceMetric, val::Bool)
    pm.info.cs_special_case_positive_list = repeat([val], pm.info.n_constraints)
end

function cs_special_case_negative!(pm::PerformanceMetric, val::Vector{<:Bool})
    pm.info.cs_special_case_negative_list = val
end

function cs_special_case_negative!(pm::PerformanceMetric, val::Bool)
    pm.info.cs_special_case_negative_list = repeat([val], pm.info.n_constraints)
end


# list the constraints
function list_constraints(constraint_expr::Tuple)
    return constraint_expr
end

function list_constraints(constraint_expr::Vector{EXPR_Constraint})
    return tuple(constraint_expr...)
end

function list_constraints(constraint_expr::EXPR_Constraint)
    return tuple(constraint_expr)
end

function list_constraints(::Nothing)
    return tuple()
end


# initialize metrics based on the definition 
function initialize(pm_type::Type{<:PerformanceMetric}, args...)
    # parse definition
    CM = ConfusionMatrix()
    metric_expr = define(pm_type, CM, args...)

    # gather infos
    info = MetricInfo()
    info.valid = metric_expr.info.is_linear_tp_tn && metric_expr.info.depends_cell_cm
    info.needs_adv_sum_marg = metric_expr.info.needs_adv_sum_marg
    info.needs_pred_sum_marg = metric_expr.info.needs_pred_sum_marg

    if !info.valid
        throw(ErrorException("The metric and/or the constraints are not supported"))
    end

    # check constraints
    constraint_expr = constraint(pm_type, CM, args...)
    constraint_list = list_constraints(constraint_expr)

    info.n_constraints = length(constraint_list)
    info.cs_special_case_positive_list = repeat([false], info.n_constraints)
    info.cs_special_case_negative_list = repeat([false], info.n_constraints)

    # update metric
    data = MetricData(metric_expr, constraint_list)

    return info, data
end


# compute the metric values given prediction and actual
function compute_metric(pm::PerformanceMetric, yhat::AbstractVector{<:Number}, y::AbstractVector{<:Number})
    # check for special cases
    if pm.info.special_case_positive
        if sum(y .== 0) == length(y) && sum(yhat .== 0) == length(yhat)
            return 1f0
        elseif sum(y .== 0) == length(y)
            return 0f0
        elseif sum(yhat .== 0) == length(yhat)
            return 0f0
        end
    end
    
    if pm.info.special_case_negative
        if sum(y .== 1) == length(y) && sum(yhat .== 1) == length(yhat)
            return 1f0
        elseif sum(y .== 1) == length(y)
            return 0f0
        elseif sum(yhat .== 1) == length(yhat)
            return 0f0
        end
    end

    Cval = CM_Value(yhat, y)
    val = compute_value(pm.data.metric_expr, Cval) ::Float32

    return val
end


# compute the constraints lhs values given prediction and actual
function compute_constraints(pm::PerformanceMetric, yhat::AbstractVector{<:Number}, y::AbstractVector{<:Number})
    n_constraints = pm.info.n_constraints
    vals = zeros(Float32, n_constraints)

    Cval = CM_Value(yhat, y)

    for ics = 1:n_constraints

        # check for special cases
        if pm.info.cs_special_case_positive_list[ics]
            if sum(y .== 0) == length(y) && sum(yhat .== 0) == length(yhat)
                vals[ics] = 1f0
            elseif sum(y .== 0) == length(y)
                vals[ics] = 0f0
            elseif sum(yhat .== 0) == length(yhat)
                vals[ics] = 0f0
            end
        end
        
        if pm.info.cs_special_case_negative_list[ics]
            if sum(y .== 1) == length(y) && sum(yhat .== 1) == length(yhat)
                vals[ics] = 1f0
            elseif sum(y .== 1) == length(y)
                vals[ics] = 0f0
            elseif sum(yhat .== 1) == length(yhat)
                vals[ics] = 0f0
            end
        end

        cs_expr = pm.data.constraint_list[ics]
        lhs_expr = cs_expr.expr     # left hand side of the expr >= tau
        
        vals[ics] = compute_value(lhs_expr, Cval) ::Float32

    end

    return vals
end


# compute & store multipliers (i.e.) constants that controls the relation between P and Q
function compute_multipliers!(pm::PerformanceMetric, n::Integer)
    # compute multipliers
    m = n + 1

    multiplier_pq = compute_scaling(pm.data.metric_expr, m, pm.info) :: ConstantOverPQ
    pm.data.multiplier_pq = multiplier_pq

    return nothing
end


function compute_admm_matrices!(pm::PerformanceMetric, n)
    multiplier_pq = pm.data.multiplier_pq
    info = pm.info 

    ### special case positive only, for now
    ks = 1:n
    IK = diagm(1 ./ ks)

    idx = 2:n+1
    idn = 1:n

    ## <P,  A Q B + Q C + D> + <Q, E>
    ## A = ones(n,n)

    A = ones(Float32, n,n)
    B = zeros(Float32, n,n)
    C = zeros(Float32, n,n)
    D = zeros(Float32, n,n)
    E = zeros(Float32, n,n)

    mult_u0v0 = 0f0

    # special case negative modify M
    if info.special_case_negative
        if !isnothing(multiplier_pq.cPQ) 
            multiplier_pq.cPQ[n+1, :] .= 0f0
            multiplier_pq.cPQ[:, n+1] .= 0f0
            multiplier_pq.cPQ[n+1, n+1] = 1f0
        end
        if !isnothing(multiplier_pq.c)
            multiplier_pq.c[n+1, :] .= 0f0
            multiplier_pq.c[:, n+1] .= 0f0
            multiplier_pq.c[n+1, n+1] = 1f0
        end
    end


    # compute matrices 
    if !isnothing(multiplier_pq.cPQ) 
        C += multiplier_pq.cPQ[idx, idx]'
    end
    if !isnothing(multiplier_pq.cPQ0) 
        C -= multiplier_pq.cPQ0[idx, idx]'
        B += IK * multiplier_pq.cPQ0[idx, idx]'
    end
   
    if !isnothing(multiplier_pq.cP0Q) 
        C -= multiplier_pq.cP0Q[idx, idx]'
        B += multiplier_pq.cPQ0[idx, idx]' * IK
    end
    if !isnothing(multiplier_pq.cP0Q0) 
        MP0Q0 = multiplier_pq.cP0Q0
        C += MP0Q0[idx, idx]'
        B += n * IK * MP0Q0[idx, idx]' * IK  -  MP0Q0[idx, idx]' * IK  -  IK * MP0Q0[idx, idx]'

        if !info.special_case_positive
            B += MP0Q0[1, idx] * ones(n)' * IK  -  n * IK * MP0Q0[1, idx] * ones(n)' * IK
            B += IK * ones(n) * MP0Q0[idx, 1]'  -  n * IK * ones(n) * MP0Q0[idx, 1]' * IK

            D += n * ones(n) * MP0Q0[idx, 1]' * IK  -  ones(n) * MP0Q0[idx, 1]'
            E += n * ones(n) * MP0Q0[1, idx]' * IK  -  ones(n) * MP0Q0[1, idx]'

            mult_u0v0 += MP0Q0[1,1]
        end
    end

    if !isnothing(multiplier_pq.c)
        B += IK * multiplier_pq.c[idx, idx]' * IK
        
        if !info.special_case_positive
            B -= IK * ones(n) * multiplier_pq.c[idx, 1]' * IK  +  IK * multiplier_pq.c[1, idx] * ones(n)' * IK
            D += ones(n) * multiplier_pq.c[idx, 1]' * IK
            E += ones(n) * multiplier_pq.c[1, idx]' * IK

            mult_u0v0 += multiplier_pq.c[1, 1]
        end
    end
    

    if info.special_case_positive
        B += IK * ones(n,n) * IK
        D -= ones(n,n) * IK
        E -= ones(n,n) * IK
        ct = 1f0    
    else
        B += IK * ones(n,n) * IK * mult_u0v0
        D -= ones(n,n) * IK * mult_u0v0
        E -= ones(n,n) * IK * mult_u0v0
        ct = mult_u0v0    
    end


    # eigen decomposition
    # precompute Matrices
    BC = n * B * B' + C * B' + B * C'
    CIinv = inv(C * C' + I)
    BC_Cinv = BC * CIinv

    # find eigen decomposition of BC * CIinv using CIinv^{0.5} * BC * CIinv^{0.5}
    # CIinv^{0.5} * BC * CIinv^{0.5} is symmetric, so we get a nicer eigendecomposition
    # CIinv^{0.5} * BC * CIinv^{0.5} and BC * CIinv have the same eigen values
    sqC = real(sqrt(CIinv))   # matrix sqrt: i.e  sqC * sqC = CIinv; always real since CIinv is posdef
    sqC_BC_sqC = sqC * BC * sqC     # it's symmetric

    sz, UZ = eigen(Symmetric(sqC_BC_sqC))

    ## convert to eigen decomposition over BC * CIinv, say:
    ## CIinv^{0.5} * BC * CIinv^{0.5} = U S U^-1, and
    ## BC * CIinv = V S V^-1, therefore:
    ## CIinv^{-0.5} CIinv^{0.5} * BC * CIinv^{0.5} CIinv^{0.5} = CIinv^{-0.5} U S U^-1 CIinv^{0.5}
    ## BC * CIinv = CIinv^{-0.5} U S U^-1 CIinv^{0.5}
    ## Hence: V = CIinv^{-0.5} U
    sbc = sz
    UBC = inv(sqC) * UZ

    # eugen dec for A
    sa, UA = eigen(Symmetric(A))
    
    # sa * sbc' + 1
    sabc1 = (sa * sbc' .+ 1)

    # inverses
    UAinv = inv(UA)
    UBCinv = inv(UBC)

    # store matrices in opt_data
    opt_data = OptData(n, A, B, C, D, E, ct, CIinv, BC_Cinv, UA, UBC, UAinv, UBCinv, sabc1)
    pm.opt_data = opt_data

    return nothing
end

# compute grad p from scratch
function compute_grad_p(Q::AbstractMatrix, multiplier_pq::ConstantOverPQ, info::MetricInfo) 
    n = size(Q,1)
    
    # compute Q(zerovec)
    ks = 1:n
    qsum_zero = (1 - sum(Q ./ ks'))
    q_zerovec = qsum_zero .* ones(Float32, n)
   
    # compute Q0
    qsum = vec(sum(Q ./ ks', dims = 1))
    Q0 = sum(Q ./ ks', dims = 1) .- Q

    Q_0k = [zeros(Float32, n) Q]
    Q0_0k = [q_zerovec Q0]
    qsum_0k = [qsum_zero; qsum]

    # storing gradients
    # zeros(Float32, ), to match the type of Q
    dP = Q * 0f0
    dP0 = Q0 * 0f0
    dpsum = qsum * 0f0
    dp_zerovec = q_zerovec * 0f0

    # regular idx
    if info.special_case_negative && info.special_case_positive
        idx = 2:n
    elseif info.special_case_positive
        idx = 2:n+1
    elseif info.special_case_negative
        idx = 1:n
    else
        idx = 1:n+1
    end
    idn = 1:n

    # 0 to k
    # zeros(Float32, ), to match the type of Q
    dP_0k = Q_0k * 0f0
    dP0_0k = Q0_0k * 0f0
    dpsum_0k = qsum_0k * 0f0

    # collect for regular idx
    if !isnothing(multiplier_pq.cPQ) 
        dP_0k[idn, idx] += Q_0k[idn, idx] * multiplier_pq.cPQ[idx, idx]'
    end
    if !isnothing(multiplier_pq.cPQ0) 
        dP_0k[idn, idx] += Q0_0k[idn, idx] * multiplier_pq.cPQ0[idx, idx]'
    end
   
    if !isnothing(multiplier_pq.cP0Q) 
        dP0_0k[idn, idx] += Q_0k[idn, idx] * multiplier_pq.cP0Q[idx, idx]'
    end
    if !isnothing(multiplier_pq.cP0Q0) 
        dP0_0k[idn, idx] += Q0_0k[idn, idx] * multiplier_pq.cP0Q0[idx, idx]'
    end

    if !isnothing(multiplier_pq.c)
        dpsum_0k[idx] += multiplier_pq.c[idx, idx] * qsum_0k[idx]
    end

    # put it id dP
    dP += dP_0k[:,2:n+1]
    dP0 += dP0_0k[:,2:n+1]
    dpsum += dpsum_0k[2:n+1]

    ## transform dP0 and dQ0 as dP and dQ
    ## Use array broadcasting
    dP += sum(dP0 ./ ks', dims = 1) .- dP0

    # transform dpsum as dp
    dP .+= (dpsum ./ ks)' 

    # zerovec cases
    if info.special_case_positive
        dp_zerovec += q_zerovec / n
    else
        dp_zerovec += dP0_0k[:,1]
        dp_zerovec .+= dpsum_0k[1] / n
    end

    # onevec cases
    if info.special_case_negative
        dP[:,n] += Q[:,n] / n
    end

    ## transform dp_zerovec and dq_zerovec as dP and dQ
    dP .+= (-(sum(dp_zerovec)) ./ ks')

    # gather constants from (1 - P(sum not 0)) (1 - Q(sum not 0)) if special case positive or
    # gather constants from (1 - P(sum not 0)) 
    const_p = sum(dp_zerovec)

    return dP, const_p
end


# use the stored matric calculation instead
function compute_grad_p(pm::PerformanceMetric, Q::AbstractMatrix) 
    # get stored matrices
    od = pm.opt_data
    A = od.A; B = od.B; C = od.C; D = od.D; E = od.E; ct = od.ct; 
    dP = A * Q * B + Q * C + D
    const_p = sum(Q .* E) + ct 

    return dP, const_p
end

# generate constraints
function generate_constraints_on_p!(pm::PerformanceMetric, y::AbstractVector)
    n = length(y)
    n_constraints = pm.info.n_constraints

    y_int = Int.(y)
    k = sum(y_int .== 1)
    Q = zeros(Float32, n, n)
    if k > 0 
        Q[:, k] = y
    end

    # for storing constraints
    B_list = Vector{Matrix{Float32}}(undef, n_constraints)
    c_list = zeros(Float32, n_constraints)
    tau_list = zeros(Float32, n_constraints)

    for ics = 1:n_constraints
        # get expr
        cs_expr = pm.data.constraint_list[ics]
        lhs_expr = cs_expr.expr     # left hand side of the expr >= tau
        tau = cs_expr.threshold     # right hand side of the expr >= tau

        expr_info = lhs_expr.info :: ExpressionInfo

        # construct info
        info = MetricInfo()
        info.valid = expr_info.is_linear_tp_tn && expr_info.depends_cell_cm
        info.needs_adv_sum_marg = expr_info.needs_adv_sum_marg
        info.needs_pred_sum_marg = expr_info.needs_pred_sum_marg
        info.special_case_positive = pm.info.cs_special_case_positive_list[ics]
        info.special_case_negative = pm.info.cs_special_case_negative_list[ics]

        # compute multiplier
        m = n + 1
        multiplier_pq = compute_scaling(lhs_expr, m, info) :: ConstantOverPQ

        # compute grad p
        dP, const_p = compute_grad_p(Q, multiplier_pq, info)

        # store
        B_list[ics] = dP
        c_list[ics] = const_p
        tau_list[ics] = tau
    end

    pm.data.B_list = B_list
    pm.data.c_list = c_list
    pm.data.tau_list = tau_list

    return nothing
end


function obj_admm_q(Q, A, B, C, D, E, ct)
    obj = max_sumlargest(A * Q * B + Q * C + D) + sum(Q .* E) + ct
    return obj
end


function solve_admm_q(pm::PerformanceMetric, PSI::AbstractMatrix, prox_function::Function; 
    rho = 1.0, max_iter = 100) 

    n = size(PSI, 1)
    ks = 1:n

    # get stored matrices
    od = pm.opt_data
    A = od.A; B = od.B; C = od.C; D = od.D; E = od.E; ct = od.ct; 
    CIinv = od.CIinv; BC_Cinv = od.BC_Cinv
    UA = od.UA; UBC = od.UBC; UAinv = od.UAinv; UBCinv = od.UBCinv; sabc1 = od.sabc1

    EP = E - PSI

    Q = zeros(n,n)
    X = zeros(n,n)
    Z = zeros(n,n)

    U = zeros(n,n)
    W = zeros(n,n)

    # resetting the storage in the LBFGS for marginal projection
    reset_projection_storage()
    
    it = 1
    while true

        # opt Q
        Q = marginal_projection( (rho * (X + W) - EP) / rho, init_storage_id = 1 )

        # opt Z
        Z = prox_function(A * X * B  +  X * C + D + U, rho, init_storage_id = 2)

        # opt X
        DZU = D - Z + U
        F = A * DZU * B' + DZU * C' + W - Q

        UFU = UAinv * (-F * CIinv) * UBC
        X1 = UFU ./ sabc1
        X = UA * X1 * UBCinv

        # opt dual
        U = U + A * X * B + X * C + D - Z
        W = W + X - Q

        if it >= max_iter
            break
        end

        it += 1
    end

    obj = obj_admm_q(Q, A, B, C, D, EP, ct)
   
    return Float32.(Q), Float32(obj), it
end

function solve_admm_q(pm::PerformanceMetric, psi::AbstractVector, prox_function::Function; args...)
    n = length(psi)
    PSI = repeat(psi, 1, n)

    Q, obj, it = solve_admm_q(pm, PSI, prox_function; args...)
    q = vec(sum(Q, dims = 2))
    q = min.(q, 1)

    return q, obj, it
end


function objective(pm::PerformanceMetric, psi::AbstractVector, y::AbstractVector; args...)

    n = length(psi)

    # compute stored multipliers and matrices 
    if all_nothing(pm.data.multiplier_pq) || pm.opt_data.n != n
        compute_multipliers!(pm, n)
    end
    if pm.opt_data.n != n
        compute_admm_matrices!(pm, n)
    end    


    if pm.info.n_constraints == 0
        prox_function = prox_max_sumlargest
    else
        # update metric constraints based on y
        generate_constraints_on_p!(pm, y)
        prox_function(A, rho; args...) = prox_max_sumlargest_with_constraint(
            A, rho, pm.data.B_list, pm.data.c_list, pm.data.tau_list; args...)   
    end

    q, obj = solve_admm_q(pm, psi, prox_function; args...)

    return obj, q
end


