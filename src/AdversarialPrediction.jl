module AdversarialPrediction

# import for expression.jl
import Base.sqrt, Base.log, Base.exp

# packages used in metric.jl
using LinearAlgebra
using JuMP
using ECOS

# packages used in nn.jl
using Flux.Tracker
using Flux.Tracker: TrackedArray, track, data, @grad
using Requires

# default
gurobi_available() = false

include("expression.jl")
include("metric.jl")
include("nn.jl")

# if Gurobi or CuArrays are loaded
function __init__()
    @require CuArrays="3a865a2d-5b23-5a0f-bc46-62713ec82fae" include("nncuda.jl")

    @require Gurobi="2e9cd046-0924-5485-92f1-d5272153d98b" begin
        gurobi_available() = true
        const GUROBI_ENV = Gurobi.Env()
    end
end

export ConfusionMatrix, CM_Value, PerformanceMetric
export @metric, define, constraint
export set_lp_solver!, special_case_positive!, special_case_negative!, cs_special_case_positive!, cs_special_case_negative!
export compute_metric, compute_constraints, objective
export ap_objective, ap_obj_grad

end # module
