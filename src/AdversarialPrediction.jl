module AdversarialPrediction

# import for expression.jl
import Base.sqrt, Base.log, Base.exp

# packages used in metric.jl
using LinearAlgebra
using LBFGSB

# packages used in nn.jl
using Flux.Tracker
using Flux.Tracker: TrackedArray, track, data, @grad
using Requires

include("expression.jl")
include("projection.jl")
include("metric.jl")
include("nn.jl")

# common metrics
include("common_metrics/CommonMetrics.jl")

# if Gurobi or CuArrays are loaded
function __init__()
    @require CuArrays="3a865a2d-5b23-5a0f-bc46-62713ec82fae" include("nncuda.jl")
end

export ConfusionMatrix, CM_Value, PerformanceMetric
export @metric, define, constraint
export special_case_positive!, special_case_negative!, cs_special_case_positive!, cs_special_case_negative!
export compute_metric, compute_constraints, objective
export ap_objective, ap_obj_grad

end # module
