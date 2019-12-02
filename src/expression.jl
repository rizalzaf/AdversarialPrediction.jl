## types for CM 
# abstract
abstract type CM_Entity end
abstract type CM_Cell <: CM_Entity end
abstract type CM_Actual_Sum <: CM_Entity end
abstract type CM_Prediction_Sum <: CM_Entity end
abstract type CM_All_Sum <: CM_Entity end

# concrete
struct CM_TP <: CM_Cell end
struct CM_FP <: CM_Cell end
struct CM_FN <: CM_Cell end
struct CM_TN <: CM_Cell end

struct CM_AP <: CM_Actual_Sum end
struct CM_AN <: CM_Actual_Sum end
struct CM_PP <: CM_Prediction_Sum end
struct CM_PN <: CM_Prediction_Sum end
struct CM_ALL <: CM_All_Sum end

## Confusion Matrix
struct ConfusionMatrix
    tp::CM_TP 
    fp::CM_FP 
    fn::CM_FN 
    tn::CM_TN 
    ap::CM_AP 
    an::CM_AN 
    pp::CM_PP
    pn::CM_PN
    all::CM_ALL

    ConfusionMatrix() = new(CM_TP(), CM_FP(), CM_FN(), CM_TN(), 
        CM_AP(), CM_AN(), CM_PP(), CM_PN(), CM_ALL())
end

struct CM_Value{T<:Number}
    tp::T 
    fp::T 
    fn::T 
    tn::T 
    ap::T 
    an::T 
    pp::T
    pn::T
    all::T

    function CM_Value(yhat::AbstractVector, y::AbstractVector)
        all = length(y)
        tp = yhat' * y
        ap = sum(y .== 1)
        pp = sum(yhat .== 1)

        an = all - ap
        pn = all - pp

        fp = pp - tp
        fn = ap - tp
        tn = an - fp

        return new{eltype(yhat)}(tp, fp, fn, tn, ap, an, pp, pn, all)
    end
end


# for storing constants
mutable struct ConstantOverPQ 
    cPQ::Union{Matrix{Float32},Nothing}
    cPQ0::Union{Matrix{Float32},Nothing}
    cP0Q::Union{Matrix{Float32},Nothing}
    cP0Q0::Union{Matrix{Float32},Nothing}
    c::Union{Matrix{Float32},Nothing}

    ConstantOverPQ() = new(
        nothing, nothing, nothing, nothing, nothing)
end

function is_constant(co::ConstantOverPQ)
    return (isnothing(co.cPQ) && isnothing(co.cPQ0) && isnothing(co.cP0Q) && 
        isnothing(co.cP0Q0) && (!isnothing(co.c)))
end

function all_nothing(co::ConstantOverPQ)
    return (isnothing(co.cPQ) && isnothing(co.cPQ0) && isnothing(co.cP0Q) && 
        isnothing(co.cP0Q0) && isnothing(co.c))
end

function apply(func::Function, co::ConstantOverPQ, other)
    result = ConstantOverPQ()
    result.cPQ = isnothing(co.cPQ) ? nothing : func(co.cPQ, other)
    result.cPQ0 = isnothing(co.cPQ0) ? nothing : func(co.cPQ0, other)
    result.cP0Q = isnothing(co.cP0Q) ? nothing : func(co.cP0Q, other)
    result.cP0Q0 = isnothing(co.cP0Q0) ? nothing : func(co.cP0Q0, other)
    result.c = isnothing(co.c) ? nothing : func(co.c, other)
    return result
end


element_mult(x, y) = x .* y
element_div(x, y) = x ./ y
Base.:*(co::ConstantOverPQ, other) = apply(element_mult, co, other)
Base.:*(other, co::ConstantOverPQ) = apply(element_mult, co, other)
Base.:/(co::ConstantOverPQ, other) = apply(element_div, co, other)

function add_nothing(x, y)
    if isnothing(x) && isnothing(y)
        return nothing
    elseif isnothing(x)
        return y
    elseif isnothing(y)
        return x
    else
        return x + y
    end        
end

function Base.:+(co::ConstantOverPQ, other::ConstantOverPQ)
    result = ConstantOverPQ()
    result.cPQ = add_nothing(co.cPQ, other.cPQ)
    result.cPQ0 = add_nothing(co.cPQ0, other.cPQ0)
    result.cP0Q = add_nothing(co.cP0Q, other.cP0Q)
    result.cP0Q0 = add_nothing(co.cP0Q0, other.cP0Q0)
    result.c = add_nothing(co.c, other.c)
    return result
end


## types for expression
abstract type CM_Expression end

# type for expr info
mutable struct ExpressionInfo
    is_linear_tp_tn::Bool       # is it linear w.r.t TP and TN
    depends_cell_cm::Bool       # does it depend on the cell of confusion matrix: tp, tn, fp, fn 
    depends_actual_sum::Bool    # does it depend on actual sum statistics: AP & AN
    depends_predicted_sum::Bool # does it depend on predicted sum statistics: PP & PN
    is_constant::Bool           # does it contain only numbers or {ALL} entity
    needs_adv_sum_marg::Bool    # does it need 'sum'-marginal of the adversary to compute 
    needs_pred_sum_marg::Bool   # does it need 'sum'-marginal of the predictor to compute
    is_constraint::Bool         # is it a constraint (contains '>=')

    # default values
    ExpressionInfo() = new(true, false, false, false, false, false, false, false) 
end

# type for metric info
mutable struct MetricInfo
    valid::Bool                 # is it a valid metric based on our definition
    needs_adv_sum_marg::Bool    # does it need 'sum'-marginal of the adversary to compute 
    needs_pred_sum_marg::Bool   # does it need 'sum'-marginal of the predictor to compute
    special_case_positive::Bool    # does it enforce special case for yhat = \zerovec && y = \zerovec 
    special_case_negative::Bool   # does it enforce special case for yhat = \onevec && y = \onevec 
    n_constraints::Int         # how many constraints does it have

    cs_special_case_positive_list::Vector{Bool}     # special cases for constraints
    cs_special_case_negative_list::Vector{Bool}     # special cases for constraints

    # default values
    MetricInfo() = new(false, false, false, false, false, 0, Bool[], Bool[]) 
end

# concrete expr
struct EXPR_UnaryEntity{TE <: CM_Entity} <: CM_Expression
    entity::TE
    multiplier::Float32
    info::ExpressionInfo

    function EXPR_UnaryEntity(entity::CM_Cell, multiplier = 1f0) 
        info = ExpressionInfo()
        info.depends_cell_cm = true
        return new{typeof(entity)}(entity, multiplier, info)
    end

    function EXPR_UnaryEntity(entity::CM_Actual_Sum, multiplier = 1f0) 
        info = ExpressionInfo()
        info.depends_actual_sum = true
        return new{typeof(entity)}(entity, multiplier, info)
    end

    function EXPR_UnaryEntity(entity::CM_Prediction_Sum, multiplier = 1f0) 
        info = ExpressionInfo()
        info.depends_predicted_sum = true
        return new{typeof(entity)}(entity, multiplier, info)
    end

    function EXPR_UnaryEntity(entity::CM_All_Sum, multiplier = 1f0) 
        info = ExpressionInfo()
        info.is_constant = true
        return new{typeof(entity)}(entity, multiplier, info)
    end
end

struct EXPR_UnaryIdentity <: CM_Expression
    multiplier::Float32
    info::ExpressionInfo

    function EXPR_UnaryIdentity(multiplier = 1f0) 
        info = ExpressionInfo()
        info.is_constant = true
        return new(multiplier, info)
    end
end

struct EXPR_UnaryExpr{TE <: CM_Expression} <: CM_Expression
    expr::TE
    multiplier::Float32
    info::ExpressionInfo

    function EXPR_UnaryExpr(expr, multiplier = 1f0) 
        return new{typeof(expr)}(expr, multiplier, expr.info)
    end
end

struct EXPR_Fraction{T1 <: CM_Expression, T2 <: CM_Expression} <: CM_Expression
    numerator::T1
    denominator::T2
    info::ExpressionInfo

    function EXPR_Fraction(numerator, denominator) 
        info = ExpressionInfo()
        
        if denominator.info.is_constant
            info.is_linear_tp_tn = numerator.info.is_linear_tp_tn
            info.depends_cell_cm = numerator.info.depends_cell_cm
            info.depends_actual_sum = numerator.info.depends_actual_sum
            info.depends_predicted_sum = numerator.info.depends_predicted_sum
            info.is_constant = numerator.info.is_constant
        else
            if denominator.info.depends_cell_cm
                info.is_linear_tp_tn = false
            else
                info.is_linear_tp_tn = numerator.info.is_linear_tp_tn
            end
            info.depends_cell_cm = numerator.info.depends_cell_cm || denominator.info.depends_cell_cm
            info.depends_actual_sum = numerator.info.depends_actual_sum || denominator.info.depends_actual_sum
            info.depends_predicted_sum = numerator.info.depends_predicted_sum || denominator.info.depends_predicted_sum
            info.is_constant = false
        end

        info.needs_adv_sum_marg = denominator.info.depends_actual_sum ? true : numerator.info.needs_adv_sum_marg
        info.needs_pred_sum_marg = denominator.info.depends_predicted_sum ? true : numerator.info.needs_pred_sum_marg     
        info.is_constraint = numerator.info.is_constraint || denominator.info.is_constraint

        return new{typeof(numerator), typeof(denominator)}(numerator, denominator, info)
    end
end

struct EXPR_Multiplication{T1 <: CM_Expression, T2 <: CM_Expression} <: CM_Expression
    expr1::T1
    expr2::T2
    info::ExpressionInfo

    function EXPR_Multiplication(expr1, expr2) 
        info = ExpressionInfo()

        if !expr1.info.depends_cell_cm
            info.is_linear_tp_tn = expr2.info.is_linear_tp_tn
        elseif !expr2.info.depends_cell_cm
            info.is_linear_tp_tn = expr1.info.is_linear_tp_tn
        else
            info.is_linear_tp_tn = false
        end
        info.depends_cell_cm = expr1.info.depends_cell_cm || expr2.info.depends_cell_cm
        info.depends_actual_sum = expr1.info.depends_actual_sum || expr2.info.depends_actual_sum
        info.depends_predicted_sum = expr1.info.depends_predicted_sum || expr2.info.depends_predicted_sum
        info.is_constant = expr1.info.is_constant && expr2.info.is_constant

        if expr1.info.depends_actual_sum
            if expr2.info.is_constant
                info.needs_adv_sum_marg = expr1.info.needs_adv_sum_marg
            else
                info.needs_adv_sum_marg = true
            end
        elseif expr2.info.depends_actual_sum
            if expr1.info.is_constant
                info.needs_adv_sum_marg = expr2.info.needs_adv_sum_marg
            else
                info.needs_adv_sum_marg = true
            end
        else
            info.needs_adv_sum_marg = false
        end

        if expr1.info.depends_predicted_sum
            if expr2.info.is_constant
                info.needs_pred_sum_marg = expr1.info.needs_pred_sum_marg
            else
                info.needs_pred_sum_marg = true
            end
        elseif expr2.info.depends_predicted_sum
            if expr1.info.is_constant
                info.needs_pred_sum_marg = expr2.info.needs_pred_sum_marg
            else
                info.needs_pred_sum_marg = true
            end
        else
            info.needs_pred_sum_marg = false
        end

        info.is_constraint = expr1.info.is_constraint || expr2.info.is_constraint

        return new{typeof(expr1), typeof(expr2)}(expr1, expr2, info)
    end
end

struct EXPR_Power{TE <: CM_Expression} <: CM_Expression
    expr::TE
    power::Float32
    info::ExpressionInfo

    function EXPR_Power(expr, power) 
        info = ExpressionInfo()

        if expr.info.depends_cell_cm
            info.is_linear_tp_tn = false
        else
            info.is_linear_tp_tn = expr.info.is_linear_tp_tn
        end
        info.depends_cell_cm = expr.info.depends_cell_cm
        info.depends_actual_sum = expr.info.depends_actual_sum
        info.depends_predicted_sum = expr.info.depends_predicted_sum
        info.is_constant = expr.info.is_constant
        info.needs_adv_sum_marg = expr.info.depends_actual_sum ? true : expr.info.needs_adv_sum_marg
        info.needs_pred_sum_marg = expr.info.depends_predicted_sum ? true : expr.info.needs_pred_sum_marg     
        info.is_constraint = expr.info.is_constraint

        return new{typeof(expr)}(expr, power, info)
    end
end

struct EXPR_FunctionCall{TE <: CM_Expression} <: CM_Expression
    func::Function
    expr::TE
    info::ExpressionInfo

    function EXPR_FunctionCall(func, expr) 
        info = ExpressionInfo()

        if expr.info.depends_cell_cm
            info.is_linear_tp_tn = false
        else
            info.is_linear_tp_tn = expr.info.is_linear_tp_tn
        end
        info.depends_cell_cm = expr.info.depends_cell_cm
        info.depends_actual_sum = expr.info.depends_actual_sum
        info.depends_predicted_sum = expr.info.depends_predicted_sum
        info.is_constant = expr.info.is_constant
        info.needs_adv_sum_marg = expr.info.depends_actual_sum ? true : expr.info.needs_adv_sum_marg
        info.needs_pred_sum_marg = expr.info.depends_predicted_sum ? true : expr.info.needs_pred_sum_marg     
        info.is_constraint = expr.info.is_constraint

        return new{typeof(expr)}(func, expr, info)
    end
end

struct EXPR_Addition{T1 <: CM_Expression, T2 <: CM_Expression} <: CM_Expression
    expr1::T1
    expr2::T2
    info::ExpressionInfo

    function EXPR_Addition(expr1, expr2) 
        info = ExpressionInfo()
        info.is_linear_tp_tn = expr1.info.is_linear_tp_tn && expr2.info.is_linear_tp_tn
        info.depends_cell_cm = expr1.info.depends_cell_cm || expr2.info.depends_cell_cm
        info.depends_actual_sum = expr1.info.depends_actual_sum || expr2.info.depends_actual_sum
        info.depends_predicted_sum = expr1.info.depends_predicted_sum || expr2.info.depends_predicted_sum
        info.is_constant = expr1.info.is_constant && expr2.info.is_constant      
        info.needs_adv_sum_marg = expr1.info.needs_adv_sum_marg || expr2.info.needs_adv_sum_marg
        info.needs_pred_sum_marg = expr1.info.needs_pred_sum_marg || expr2.info.needs_pred_sum_marg  
        info.is_constraint = expr1.info.is_constraint || expr2.info.is_constraint

        return new{typeof(expr1), typeof(expr2)}(expr1, expr2, info)
    end
end

struct EXPR_Constraint{TE <: CM_Expression} <: CM_Expression
    expr::TE
    threshold::Float32
    info::ExpressionInfo

    function EXPR_Constraint(expr, threshold) 
        info = ExpressionInfo()

        info.is_linear_tp_tn = false
        info.depends_cell_cm = expr.info.depends_cell_cm
        info.depends_actual_sum = expr.info.depends_actual_sum
        info.depends_predicted_sum = expr.info.depends_predicted_sum
        info.is_constant = expr.info.is_constant
        info.needs_adv_sum_marg = expr.info.depends_actual_sum ? true : expr.info.needs_adv_sum_marg
        info.needs_pred_sum_marg = expr.info.depends_predicted_sum ? true : expr.info.needs_pred_sum_marg    
        info.is_constraint = true

        return new{typeof(expr)}(expr, threshold, info)
    end
end


### functions on entities & expressions

## Pos and neg
Base.:+(e::CM_Entity) = EXPR_UnaryEntity(e, 1f0)
Base.:-(e::CM_Entity) = EXPR_UnaryEntity(e, -1f0)

Base.:+(e::CM_Expression) = e
Base.:-(e::CM_Expression) = EXPR_UnaryExpr(e, -1f0)

Base.:-(e::EXPR_UnaryEntity) = EXPR_UnaryEntity(e.entity, -e.multiplier)
Base.:-(e::EXPR_UnaryIdentity) = EXPR_UnaryIdentity(-e.multiplier)
Base.:-(e::EXPR_UnaryExpr) = EXPR_UnaryExpr(e.expr, -e.multiplier)


## Addition
# special cases entity
Base.:+(e1::CM_TP, e2::CM_FP) = CM_PP()
Base.:+(e1::CM_FP, e2::CM_TP) = CM_PP()
Base.:+(e1::CM_TP, e2::CM_FN) = CM_AP()
Base.:+(e1::CM_FN, e2::CM_TP) = CM_AP()

Base.:+(e1::CM_TN, e2::CM_FN) = CM_PN()
Base.:+(e1::CM_FN, e2::CM_TN) = CM_PN()
Base.:+(e1::CM_TN, e2::CM_FP) = CM_AN()
Base.:+(e1::CM_FP, e2::CM_TN) = CM_AN()

Base.:+(e1::CM_AP, e2::CM_AN) = CM_ALL()
Base.:+(e1::CM_AN, e2::CM_AP) = CM_ALL()
Base.:+(e1::CM_PP, e2::CM_PN) = CM_ALL()
Base.:+(e1::CM_PN, e2::CM_PP) = CM_ALL()

# general
Base.:+(e1::CM_Expression, e2::CM_Expression) = EXPR_Addition(e1, e2)
Base.:+(e1::CM_Expression, e2::CM_Entity) = e1 + EXPR_UnaryEntity(e2)
Base.:+(e1::CM_Entity, e2::CM_Expression) = EXPR_UnaryEntity(e1) + e2
Base.:+(e1::CM_Entity, e2::CM_Entity) = EXPR_UnaryEntity(e1) + EXPR_UnaryEntity(e2)

Base.:+(e1::CM_Expression, x::Number) = e1 + EXPR_UnaryIdentity(x)
Base.:+(x::Number, e1::CM_Expression) = EXPR_UnaryIdentity(x) + e1
Base.:+(e::CM_Entity, x::Number) = EXPR_UnaryEntity(e) + EXPR_UnaryIdentity(x)
Base.:+(x::Number, e::CM_Entity) = EXPR_UnaryIdentity(x) + EXPR_UnaryEntity(e)

## Subtraction
Base.:-(e1::CM_Expression, e2::CM_Expression) = e1 + (-e2)
Base.:-(e1::CM_Expression, e2::CM_Entity) = e1 + (-EXPR_UnaryEntity(e2))
Base.:-(e1::CM_Entity, e2::CM_Expression) = EXPR_UnaryEntity(e1) + (-e2)
Base.:-(e1::CM_Entity, e2::CM_Entity) = EXPR_UnaryEntity(e1) + (-e2)

Base.:-(e1::CM_Expression, e2) = e1 + (-e2)
Base.:-(e1::CM_Entity, e2) = EXPR_UnaryEntity(e1) + (-e2)
Base.:-(e1, e2::CM_Expression) = e1 + (-e2)
Base.:-(e1, e2::CM_Entity) = e1 + (-EXPR_UnaryEntity(e2))


## Multiplication
Base.:*(e1::CM_Expression, e2::CM_Expression) = EXPR_Multiplication(e1, e2)
Base.:*(e1::CM_Expression, e2::CM_Entity) = EXPR_Multiplication(e1, EXPR_UnaryEntity(e2))
Base.:*(e1::CM_Entity, e2::CM_Expression) = EXPR_Multiplication(EXPR_UnaryEntity(e1), e2)
Base.:*(e1::CM_Entity, e2::CM_Entity) = EXPR_Multiplication(EXPR_UnaryEntity(e1), EXPR_UnaryEntity(e2))

Base.:*(e::CM_Expression, x::Number) = EXPR_UnaryExpr(e, x)
Base.:*(x::Number, e::CM_Expression) = e * x
Base.:*(e::CM_Entity, x::Number) = EXPR_UnaryEntity(e, x)
Base.:*(x::Number, e::CM_Entity) = e * x

Base.:*(e::EXPR_UnaryEntity, x::Number) = EXPR_UnaryExpr(e.entity, x * e.multiplier)
Base.:*(e::EXPR_UnaryIdentity, x::Number) = EXPR_UnaryIdentity(x * e.multiplier)
Base.:*(e::EXPR_UnaryExpr, x::Number) = EXPR_UnaryExpr(e.expr, x * e.multiplier)


## Division
Base.:/(e1::CM_Expression, e2::CM_Expression) = EXPR_Fraction(e1, e2)
Base.:/(e1::CM_Expression, e2::CM_Entity) = EXPR_Fraction(e1, EXPR_UnaryEntity(e2))
Base.:/(e1::CM_Entity, e2::CM_Expression) = EXPR_Fraction(EXPR_UnaryEntity(e1), e2)
Base.:/(e1::CM_Entity, e2::CM_Entity) = EXPR_Fraction(EXPR_UnaryEntity(e1), EXPR_UnaryEntity(e2))

Base.:/(e::CM_Expression, x::Number) = EXPR_UnaryExpr(e, 1/x)
Base.:/(x::Number, e::CM_Expression) = EXPR_UnaryIdentity(x) / e
Base.:/(e::CM_Entity, x::Number) = EXPR_UnaryEntity(e, 1/x)
Base.:/(x::Number, e::CM_Entity) = EXPR_UnaryIdentity(x) / e

Base.:/(e::EXPR_UnaryEntity, x::Number) = EXPR_UnaryExpr(e.entity, e.multiplier / x)
Base.:/(e::EXPR_UnaryExpr, x::Number) = EXPR_UnaryExpr(e.expr, e.multiplier / x)

Base.:/(e::EXPR_UnaryIdentity, x::Number) = EXPR_UnaryIdentity(e.multiplier / x)
Base.:/(x::Number, e::EXPR_UnaryIdentity) = EXPR_UnaryIdentity(x / e.multiplier)


## Power
Base.:^(e::CM_Expression, x::Number) = EXPR_Power(e, x)
Base.:^(e::CM_Entity, x::Number) = EXPR_Power(EXPR_UnaryEntity(e), x)

## Greater than equal (for metric constraints)
Base.:>=(e::CM_Expression, x::Number) = EXPR_Constraint(e, x)
Base.:>=(e::CM_Entity, x::Number) = EXPR_Constraint(EXPR_UnaryEntity(e), x)

## Function call
Base.sqrt(e::CM_Expression) = EXPR_FunctionCall(Base.sqrt, e)
Base.sqrt(e::CM_Entity) = EXPR_FunctionCall(Base.sqrt, EXPR_UnaryEntity(e))
Base.log(e::CM_Expression) = EXPR_FunctionCall(Base.log, e)
Base.log(e::CM_Entity) = EXPR_FunctionCall(Base.log, EXPR_UnaryEntity(e))
Base.exp(e::CM_Expression) = EXPR_FunctionCall(Base.exp, e)
Base.exp(e::CM_Entity) = EXPR_FunctionCall(Base.exp, EXPR_UnaryEntity(e))


## Printing
Base.show(io::IO, x::CM_TP) =  print(io, "TP")
Base.show(io::IO, x::CM_FP) =  print(io, "FP")
Base.show(io::IO, x::CM_FN) =  print(io, "FN")
Base.show(io::IO, x::CM_TN) =  print(io, "TN")

Base.show(io::IO, x::CM_AP) =  print(io, "AP")
Base.show(io::IO, x::CM_AN) =  print(io, "AN")
Base.show(io::IO, x::CM_PP) =  print(io, "PP")
Base.show(io::IO, x::CM_PN) =  print(io, "PN")
Base.show(io::IO, x::CM_ALL) =  print(io, "ALL")

Base.show(io::IO, x::CM_Entity) = print(io, SubString(string(typeof(x)), 4))
Base.show(io::IO, x::CM_Expression) = print(io, "Expression::" * typeof(x))
Base.show(io::IO, x::EXPR_UnaryIdentity) = print(io, x.multiplier)

function Base.show(io::IO, x::EXPR_UnaryEntity)
    if x.multiplier == 1
        print(io, x.entity)
    elseif x.multiplier == -1
        print(io, "-" * string(x.entity))
    else
        print(io, string(x.multiplier) * " " * string(x.entity))
    end
end

function Base.show(io::IO, x::EXPR_UnaryExpr)
    if x.multiplier == 1
        print(io, x.expr)
    elseif x.multiplier == -1
        print(io, "-" * "(" * string(x.expr) * ")")
    else
        print(io, string(x.multiplier) * " " * "(" * string(x.expr) * ")")
    end
end

Base.show(io::IO, x::EXPR_Fraction) = print(
    io, "(" * string(x.numerator) * ")" * " / " * "(" * string(x.denominator) * ")")
Base.show(io::IO, x::EXPR_Multiplication) = print(
        io, "(" * string(x.expr1) * ")" * " * " * "(" * string(x.expr2) * ")")    
Base.show(io::IO, x::EXPR_Addition) = print(
    io, "(" * string(x.expr1) * ")" * " + " * "(" * string(x.expr2) * ")")    

Base.show(io::IO, x::EXPR_Power) = print(io, "(" * string(x.expr) * ")^" * string(x.power))
Base.show(io::IO, x::EXPR_FunctionCall) = print(io, string(x.func) * "(" * string(x.expr) * ")" )
Base.show(io::IO, x::EXPR_Constraint) = print(io, "(" * string(x.expr) * " >= " * string(x.threshold))


## Compute value
compute_value(::CM_TP, Cval::CM_Value) = Cval.tp
compute_value(::CM_FP, Cval::CM_Value) = Cval.fp
compute_value(::CM_FN, Cval::CM_Value) = Cval.fn
compute_value(::CM_TN, Cval::CM_Value) = Cval.tn

compute_value(::CM_AP, Cval::CM_Value) = Cval.ap
compute_value(::CM_AN, Cval::CM_Value) = Cval.an
compute_value(::CM_PP, Cval::CM_Value) = Cval.pp
compute_value(::CM_PN, Cval::CM_Value) = Cval.pn
compute_value(::CM_ALL, Cval::CM_Value) = Cval.all

compute_value(e::EXPR_UnaryEntity, Cval::CM_Value) = e.multiplier * compute_value(e.entity, Cval)
compute_value(e::EXPR_UnaryIdentity, Cval::CM_Value) = e.multiplier
compute_value(e::EXPR_UnaryExpr, Cval::CM_Value) = e.multiplier * compute_value(e.expr, Cval)

compute_value(e::EXPR_Fraction, Cval::CM_Value) = 
    compute_value(e.numerator, Cval) / compute_value(e.denominator, Cval)
compute_value(e::EXPR_Multiplication, Cval::CM_Value) = 
    compute_value(e.expr1, Cval) * compute_value(e.expr2, Cval)
compute_value(e::EXPR_Addition, Cval::CM_Value) = 
    compute_value(e.expr1, Cval) + compute_value(e.expr2, Cval)

compute_value(e::EXPR_Power, Cval::CM_Value) = compute_value(e.expr, Cval) ^ e.power
compute_value(e::EXPR_FunctionCall, Cval::CM_Value) = e.func(compute_value(e.expr, Cval))

compute_value(e::EXPR_Constraint, Cval::CM_Value) = compute_value(e.expr, Cval) >= e.threshold

compute_value(::Nothing, ::CM_Value) = 0f0

## Compute value, m = n + 1, for \sum y_i = 0 to n
function compute_scaling(::CM_TP, m::Integer, info::MetricInfo) 
    res = ConstantOverPQ()
    res.cPQ = ones(Float32, m,m)
    return res
end
function compute_scaling(::CM_FP, m::Integer, info::MetricInfo) 
    res = ConstantOverPQ()
    res.cPQ0 = ones(Float32, m,m)
    return res
end
function compute_scaling(::CM_FN, m::Integer, info::MetricInfo) 
    res = ConstantOverPQ()
    res.cP0Q = ones(Float32, m,m)
    return res
end
function compute_scaling(::CM_TN, m::Integer, info::MetricInfo) 
    res = ConstantOverPQ()
    res.cP0Q0 = ones(Float32, m,m)
    return res
end

function compute_scaling(::CM_AP, m::Integer, info::MetricInfo) 
    res = ConstantOverPQ()
    n = m - 1
    res.c = Float32.(repeat(0:n, 1, m)')
    return res
end
function compute_scaling(::CM_AN, m::Integer, info::MetricInfo) 
    res = ConstantOverPQ()
    n = m - 1
    res.c = Float32.(repeat(n:-1:0, 1, m)')
    return res
end
function compute_scaling(::CM_PP, m::Integer, info::MetricInfo) 
    res = ConstantOverPQ()
    n = m - 1
    res.c = Float32.(repeat(0:n, 1, m))
    return res
end
function compute_scaling(::CM_PN, m::Integer, info::MetricInfo) 
    res = ConstantOverPQ()
    n = m - 1
    res.c = Float32.(repeat(n:-1:0, 1, m))
    return res
end
function compute_scaling(::CM_ALL, m::Integer, info::MetricInfo) 
    res = ConstantOverPQ()
    n = m - 1
    res.c = n * ones(Float32, m,m)
    return res
end

function compute_scaling(e::EXPR_UnaryEntity, m::Integer, info::MetricInfo) 
    return e.multiplier * compute_scaling(e.entity, m, info)
end
function compute_scaling(e::EXPR_UnaryIdentity, m::Integer, info::MetricInfo) 
    res = ConstantOverPQ()
    res.c = e.multiplier * ones(Float32, m,m)
    return res
end
function compute_scaling(e::EXPR_UnaryExpr, m::Integer, info::MetricInfo) 
    return e.multiplier * compute_scaling(e.expr, m, info)
end

function compute_scaling(e::EXPR_Fraction, m::Integer, info::MetricInfo) 
    res_num = compute_scaling(e.numerator, m, info)
    res_den = compute_scaling(e.denominator, m, info)

    if is_constant(res_den)
        C = res_den.c ::Matrix{Float32}
        invC = zeros(Float32, m,m)

        if info.special_case_positive && info.special_case_negative
            invC[2:m-1, 2:m-1] = 1 ./ C[2:m-1, 2:m-1]
        elseif info.special_case_positive
            invC[2:m, 2:m] = 1 ./ C[2:m, 2:m]
        elseif info.special_case_negative
            invC[1:m-1,1:m-1] = 1 ./ C[1:m-1,1:m-1]
        else
            invC = 1 ./ C
        end

        # handle diviion by zero, when special cases not set => set it to zero
        invC[isinf.(invC)] .= 0f0

        return res_num * invC
    else
        throw(ErrorException("The metric contains unsupported operations"))
    end
end
function compute_scaling(e::EXPR_Multiplication, m::Integer, info::MetricInfo) 
    res_ex1 = compute_scaling(e.expr1, m, info)
    res_ex2 = compute_scaling(e.expr2, m, info)

    if is_constant(res_ex1)
        return res_ex2 * res_ex1.c
    elseif is_constant(res_ex2)
        return res_ex1 * res_ex2.c
    else
        throw(ErrorException("The metric contains unsupported operations"))
    end
end

function compute_scaling(e::EXPR_Addition, m::Integer, info::MetricInfo) 
    return compute_scaling(e.expr1, m, info) + compute_scaling(e.expr2, m, info)
end

function compute_scaling(e::EXPR_Power, m::Integer, info::MetricInfo) 
    res_expr = compute_scaling(e.expr, m, info)
    if is_constant(res_expr)
        res_expr.c = res_expr.c .^ e.power
        return res_expr
    else
        throw(ErrorException("The metric contains unsupported operations"))
    end
end
function compute_scaling(e::EXPR_FunctionCall, m::Integer, info::MetricInfo) 
    res_expr = compute_scaling(e.expr, m, info)
    if is_constant(res_expr)
        res_expr.c = e.func.(res_expr.c)
        return res_expr
    else
        throw(ErrorException("The metric contains unsupported operations"))
    end
end

compute_scaling(::Nothing, ::Integer, ::MetricInfo) = ConstantOverPQ()