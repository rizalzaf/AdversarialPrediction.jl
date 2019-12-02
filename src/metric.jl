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
            lp_model::JuMP.Model
            lp_solver::JuMP.OptimizerFactory

            function $name() 
                info, data = initialize($name)
                if gurobi_available()
                    solver = JuMP.with_optimizer(Gurobi.Optimizer, GUROBI_ENV, Method=0, OutputFlag=0)
                else
                    solver = JuMP.with_optimizer(ECOS.Optimizer, verbose = false)
                end
                new{typeof(data)}(info, data, JuMP.Model(), solver)
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
            lp_model::JuMP.Model
            lp_solver::JuMP.OptimizerFactory
            $([arg for arg in args]...)

            function $name($([arg for arg in args]...)) 
                info, data = initialize($name, $([arg for arg in args]...))
                if gurobi_available()
                    solver = JuMP.with_optimizer(Gurobi.Optimizer, GUROBI_ENV, Method=0, OutputFlag=0)
                else
                    solver = JuMP.with_optimizer(ECOS.Optimizer, verbose = false)
                end
                new{typeof(data)}(info, data, JuMP.Model(), solver, $([arg for arg in args]...))
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


## set solver
function set_lp_solver!(pm::PerformanceMetric, solver::OptimizerFactory)
    pm.lp_solver = solver
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


# hepler functions for calculating gradients
function collect_grad_p(cPQ, Q)
    return Q * cPQ'
end

function collect_grad_q(cPQ, P)
    return P * cPQ
end


# computing gradients     
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
        dP_0k[idn, idx] += collect_grad_p(multiplier_pq.cPQ[idx, idx], Q_0k[idn, idx])
    end
    if !isnothing(multiplier_pq.cPQ0) 
        dP_0k[idn, idx] += collect_grad_p(multiplier_pq.cPQ0[idx, idx], Q0_0k[idn, idx])
    end
   
    if !isnothing(multiplier_pq.cP0Q) 
        dP0_0k[idn, idx] += collect_grad_p(multiplier_pq.cP0Q[idx, idx], Q_0k[idn, idx])
    end
    if !isnothing(multiplier_pq.cP0Q0) 
        dP0_0k[idn, idx] += collect_grad_p(multiplier_pq.cP0Q0[idx, idx], Q0_0k[idn, idx])
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

    # gather constants from (1 - P(sum not 0)) (1 - Q(sum not 0))
    const_p = 0.0
    if info.special_case_positive
        const_p += sum(dp_zerovec)
    end

    return dP, const_p
end

# computing gradients     
function compute_grad_p(pm::PerformanceMetric, Q::AbstractMatrix) 
    return compute_grad_p(Q, pm.data.multiplier_pq, pm.info)
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


# formulate an lp problem     
function formulate_lp_prob!(pm::PerformanceMetric, n::Integer)

    if all_nothing(pm.data.multiplier_pq) || size(pm.lp_model.obj_dict[:S])[1] != n
        compute_multipliers!(pm, n)
    end

    # optimize over R and S, which is P/k and Q/k instead
    ks = 1:n

    model = Model(pm.lp_solver)
    @variable(model, S[1:n,1:n] >= 0)
    @variable(model, D[1:n,1:n] >= 0) 
    @variable(model, v >= 0) 
    
    @constraint(model, [i = 1:n, k = 1:n], S[i,k] <= 1/k * sum(S[:,k]))
    @constraint(model, sum(S) <= 1)

    dP, const_p = compute_grad_p(pm, S .* ks')
    dR = dP .* ks'

    Dk = [sum(D[:,j]) / j for i=1:n, j=1:n]
    Z = dR - D + Dk

    @constraint(model, v >= const_p)
    @constraint(model, [i = 1:n, k = 1:n], v >= Z[i,k] + const_p)

    pm.lp_model = model

    return model
end

# solve lp problem     
function solve_lp_q(pm::PerformanceMetric, PSI::AbstractMatrix)
    n = size(PSI, 1)
    ks = 1:n

    # create LP probs
    if length(all_variables(pm.lp_model)) == 0 || size(pm.lp_model.obj_dict[:S])[1] != n
        formulate_lp_prob!(pm, n)
    end

    # modify objective
    S = pm.lp_model.obj_dict[:S] ::Matrix{VariableRef}
    v = pm.lp_model.obj_dict[:v] ::VariableRef
    @objective(pm.lp_model, Min, v - sum(PSI .* S .* ks'))

    # use last solution as initial (warm start)
    if termination_status(pm.lp_model) != MOI.OPTIMIZE_NOT_CALLED
        set_start_value.(all_variables(pm.lp_model), value.(all_variables(pm.lp_model)))
    end

    # optimize
    optimize!(pm.lp_model)

    # get solution
    Sv = value.(S) 
    Q = Sv .* ks'
    obj = objective_value(pm.lp_model)

    return Float32.(Q), Float32(obj)
end

function solve_lp_q(pm::PerformanceMetric, psi::AbstractVector)
    n = length(psi)
    PSI = repeat(psi, 1, n)

    Q, obj = solve_lp_q(pm, PSI)
    q = vec(sum(Q, dims = 2))

    return q, obj
end

# formulate an lp problem for metric with oonstraints  
function formulate_lp_prob_cs!(pm::PerformanceMetric, y::AbstractVector)
    n = length(y)
    ks = 1:n

    if all_nothing(pm.data.multiplier_pq) || size(pm.lp_model.obj_dict[:S])[1] != n
        compute_multipliers!(pm, n)
    end

    # generate constraints
    generate_constraints_on_p!(pm, y)

    n_constraints = pm.info.n_constraints
    B_list = pm.data.B_list
    
    # optimize over R and S, which is P/k and Q/k instead

    model = Model(pm.lp_solver)
    @variable(model, S[1:n,1:n] >= 0)
    @variable(model, D[1:n,1:n] >= 0)
    @variable(model, alpha[1:n_constraints] >= 0)
    @variable(model, v >= 0)
    
    @constraint(model, [i = 1:n, k = 1:n], S[i,k] <= 1/k * sum(S[:,k]))
    @constraint(model, sum(S) <= 1)

    dP, const_p = compute_grad_p(pm, S .* ks')
    dR = dP .* ks'

    Dk = [sum(D[:,j]) / j for i=1:n, j=1:n]

    @constraint(model, const_p <= v)
    @constraint(model, cs_v[i = 1:n, k = 1:n], 
        dR[i,k] - D[i,k] + Dk[i,k] + sum(alpha[l] * B_list[l][i,k] * k for l = 1:n_constraints)  + const_p <= v)

    pm.lp_model = model

    return model
end

# solve lp problem for metric with oonstraints  
function solve_lp_q_cs(pm::PerformanceMetric, PSI::AbstractMatrix, y::AbstractVector)
    n = size(PSI, 1)
    ks = 1:n

    # create LP probs
    if length(all_variables(pm.lp_model)) == 0 || size(pm.lp_model.obj_dict[:S])[1] != n
        formulate_lp_prob_cs!(pm, y)
    end

    # update metric constraints based on y
    generate_constraints_on_p!(pm, y)

    n_constraints = pm.info.n_constraints
    B_list = pm.data.B_list
    c_list = pm.data.c_list
    tau_list = pm.data.tau_list

    # modify constraints
    alpha = pm.lp_model.obj_dict[:alpha] ::Vector{VariableRef}
    cs_v = pm.lp_model.obj_dict[:cs_v] ::Matrix{<:ConstraintRef}
    for i = 1:n
        for k = 1:n
            for l = 1:n_constraints
                set_normalized_coefficient(cs_v[i,k], alpha[l], B_list[l][i,k] * k)
            end
        end
    end

    # modify objective
    S = pm.lp_model.obj_dict[:S] ::Matrix{VariableRef}
    v = pm.lp_model.obj_dict[:v] ::VariableRef
    
    ac = alpha[1] * c_list[1]
    atau = alpha[1] * tau_list[1]
    for i = 2:n_constraints
        ac += alpha[i] * c_list[i]
        atau += alpha[i] * tau_list[i]
    end

    @objective(pm.lp_model, Min, v - sum(PSI .* S .* ks') + ac - atau)

    # use last solution as initial (warm start)
    if termination_status(pm.lp_model) != MOI.OPTIMIZE_NOT_CALLED
        set_start_value.(all_variables(pm.lp_model), value.(all_variables(pm.lp_model)))
    end

    # optimize
    optimize!(pm.lp_model)

    # get solution
    Sv = value.(S)
    Q = Sv .* ks'
    obj = objective_value(pm.lp_model)

    return Float32.(Q), Float32(obj)
end

function solve_lp_q_cs(pm::PerformanceMetric, psi::AbstractVector, y::AbstractVector)
    n = length(psi)
    PSI = repeat(psi, 1, n)

    Q, obj = solve_lp_q_cs(pm, PSI, y)
    q = vec(sum(Q, dims = 2))

    return q, obj
end

# inner objective an gradients w.r.t. the potentials = theta * features 
function objective(pm::PerformanceMetric, psi::AbstractVector, y::AbstractVector)
    if pm.info.n_constraints == 0
        q, obj = solve_lp_q(pm, psi)
    else
        q, obj = solve_lp_q_cs(pm, psi, y)
    end
    return obj, q
end
