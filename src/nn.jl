# negative of obj, since the original formulation is max_\theta
# whereas the optimizer only support minimization

# all the opt in the metric needs to be done in cpu

### objective ###
function ap_objective(ps::AbstractVector, y::AbstractVector, pm::PerformanceMetric; args...)
    obj, _ = objective(pm, ps, y; args...)
    obj = obj + dot(ps, y)
    return -obj
end

# custom gradient
@adjoint function ap_objective(ps::AbstractVector, y::AbstractVector, pm::PerformanceMetric; args...)
    obj, q = objective(pm, ps, y; args...)
    obj = obj + dot(ps, y)
    grad = (q - y) 
    return -obj, Δ -> (Δ * grad, nothing, nothing)
end
