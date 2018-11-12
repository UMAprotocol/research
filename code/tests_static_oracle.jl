using Test

include("static_oracle_problem.jl")


p = Params(20, 1.0, Normal(0, 0.5), 0.75, 0.85)

@testset "constructor" begin

    test_p = Params(20, 1.0, Normal(0, 0.5), 0.75, 0.85)

    @test test_p.K == 20
    @test isapprox(test_p.μ_l, 1.0)
    @test isapprox(mean(test_p.l_dist), 0)
    @test isapprox(var(test_p.l_dist), 0.25)
    @test isapprox(test_p.ξm, 0.75)
    @test isapprox(test_p.ξo, 0.85)

end

@testset "voter_payoffs" begin

    γ = rand(p.K+1, 2)
    γtilde = rand(p.K+1)

    # Liar payoffs
    l = p.μ_l
    for _X in 0:p.K
        x = 0  # x = 0 corresponds to lying since S=1
        @test isapprox(voter_payoffs(p, x, _X, l, γ, γtilde), γtilde[_X+1] - p.μ_l)
    end

    # Honest payoffs
    for _X in 0:p.K

        x = 1  # x = 1 corresponds to honest since S=1
        truth = _X <= round(Int, p.K / 2)

        @test isapprox(voter_payoffs(p, x, _X, l, γ, γtilde), truth*γ[_X+1, x+1])
    end

end


@testset "voter optimal choices and distribution" begin
    γ = ones(p.K+1, 2)
    γtilde = 1.55 .* ones(p.K + 1)

    # Make sure distributions collapse to degenerate when payoffs are dumb
    @test isapprox(find_πX(p, γ, 100 .* γtilde)[end], 1.0)
    @test isapprox(find_πX(p, γ, 100 .* γtilde)[1], 0.0)
    @test isapprox(find_πX(p, γ, -100.0 .* γtilde)[1], 1.0)
    @test isapprox(find_πX(p, γ, -100.0 .* γtilde)[end], 0.0)

    πX = find_πX(p, γ, γtilde)
    @test isapprox(πX[1], 0.0)
    @test isapprox(πX[end], 0.0)
    @test isapprox(sum(πX), 1.0)

    # Make sure we're finding optimal policies (TODO: Add more tests here)
    @test find_xstar(p, 1.5, γ, γtilde, πX) == 1
    @test find_xstar(p, 0.0, γ, γtilde, πX) == 0
end
