import AdversarialPrediction: define, constraint, generate_constraints_on_p!
using LinearAlgebra

# sample metrics
@metric Prec
function define(::Type{Prec}, C::ConfusionMatrix)
    return C.tp / C.pp   
end

@metric F1_score
function define(::Type{F1_score}, C::ConfusionMatrix)
    return (2 * C.tp) / (C.ap + C.pp)   
end

@metric Kappa
function define(::Type{Kappa}, C::ConfusionMatrix)
    num = (C.tp + C.tn) / C.all - (C.ap * C.pp + C.an * C.pn) / C.all^2
    den = 1 - (C.ap * C.pp + C.an * C.pn) / C.all^2
    return num / den   
end

@metric Prec_F1_score
function define(::Type{Prec_F1_score}, C::ConfusionMatrix)
    return C.tp / C.pp   
end
function constraint(::Type{Prec_F1_score}, C::ConfusionMatrix)
    return (2 * C.tp) / (C.ap + C.pp) >= 0.6
end

@testset "compute metric" begin
        
    # hardcoded functions
    prec_fn(yhat, y) = (dot(yhat, y) / sum(yhat))
    f1_fn(yhat, y) = (2 * dot(yhat, y) / (sum(yhat) + sum(y)))

    function kappa_fn(yhat, y)
        n = length(y)
        num = ((dot(yhat, y) + dot(1 .- yhat, 1 .- y)) / n - 
            (sum(y) * sum(yhat) + sum(1 .- y) * sum(1 .- yhat)) / n^2 )
        den = 1 - (sum(y) * sum(yhat) + sum(1 .- y) * sum(1 .- yhat)) / n^2
        return num / den
    end

    y = rand(0:1, 100)
    yhat = rand(0:1, 100)

    prec = Prec()
    @test compute_metric(prec, yhat, y) ≈ prec_fn(yhat, y)

    f1 = F1_score()
    @test compute_metric(f1, yhat, y) ≈ f1_fn(yhat, y)

    kappa = Kappa()
    @test compute_metric(kappa, yhat, y) ≈ kappa_fn(yhat, y)
end



@testset "constraints metric" begin
        
    # hardcoded functions
    function f1_fn(y)
        n = length(y)
        iq = sum(y .== 1)

        ks = 1:n
        B = zeros(n,n)
        c = 0.0
        
        for ip = 1:n
            B[:,ip] = ( 2 * y ) / (ip + iq)
        end

        if iq == 0        
            B .+= (-1 ./ ks')
            c += 1.0
        end

        return B, c
    end

    prec_f1 = Prec_F1_score()
    special_case_positive!(prec_f1)
    cs_special_case_positive!(prec_f1, true)

    y = rand(0:1, 20)
    generate_constraints_on_p!(prec_f1, y)

    B, c = f1_fn(y)
    @test B ≈ prec_f1.data.B_list[1]
    @test isapprox(c, prec_f1.data.c_list[1], atol = 1e-6)

    # if y == zeros
    y = zeros(20)
    generate_constraints_on_p!(prec_f1, y)

    B, c = f1_fn(y)
    @test B ≈ prec_f1.data.B_list[1]
    @test isapprox(c, prec_f1.data.c_list[1], atol = 1e-6)

    # if y == ones
    y = ones(20)
    generate_constraints_on_p!(prec_f1, y)

    B, c = f1_fn(y)
    @test B ≈ prec_f1.data.B_list[1]
    @test isapprox(c, prec_f1.data.c_list[1], atol = 1e-6)

end

