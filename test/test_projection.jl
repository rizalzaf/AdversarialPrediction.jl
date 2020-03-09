import AdversarialPrediction: solve_p_given_abk, marginal_projection, marginal_projection_with_constraint,
    prox_max_sumlargest, prox_max_sumlargest_with_constraint
using LinearAlgebra

tol = 1e-3

@testset "test solve_p_given_rho" begin
    n = 30
    a = rand(n)
    rho = 3 * rand()
    k = rand(1:n)
    b = ones(n) * rho / k

    p = solve_p_given_abk(a, b, k)

    @test all(p .>= 0)
    @test all(p .<= (sum(p) / k + tol))

end


@testset "test projection" begin
    n = 30
    A = (rand(n,n) .- 0.4) ./ 0.01

    P = marginal_projection(A)
    ks = 1:n

    @test all(P .>= 0)
    @test all(sum(P, dims = 2) .<= 1 + tol)
    @test all(P .<= sum(P, dims = 1) ./ ks' .+ tol)
    @test sum(P ./ ks') <= 1 + tol
end

@testset "test projection2" begin

    A = [0.3 0.2; 0.4 0.21]
    X = [0.3 0.205; 0.4 0.205]
    
    P = marginal_projection(A)

    @test P ≈ X
end

