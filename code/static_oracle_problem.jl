#=
See notes for description of environment
=#
using Base.Cartesian
using Distributions
using LinearAlgebra
using NLopt
using Parameters
using Random

struct Params{D<:Distribution}
    # Number of voters
    K::Int

    # Parameters for distribution
    μ_l::Float64
    l_dist::D

    # Malevolent/Oracle parameters
    T::Float64  # Tax revenue
    r::Float64  # Interest rate
    ξm::Float64
    ξo::Float64
end


function voter_payoffs(ap::Params, x, X, l, γ, γtilde)
    @unpack K = ap

    dividend = ap.T / ap.K
    p_prime = ifelse(X > round(Int, K/2), 0.0, (1.0 + ap.r)/ap.r * dividend)

    # Value of telling truth γ[X, x]
    truth_value = (p_prime + dividend)*γ[X+1, x+1]

    # Value of lying γtilde[X] (bc γtilde(x=S) = 0 and S is known)
    lie_value = γtilde[X+1] - l

    return x*truth_value + (1 - x)*lie_value
end


function find_xstar(ap::Params, l, γ, γtilde, π)

    @unpack K = ap

    # Set x0/x1 to 0
    x0, x1 = (0.0, 0.0)
    for X in 0:K
        x0 += voter_payoffs(ap, 0, X, l, γ, γtilde) * π[X+1]
        x1 += voter_payoffs(ap, 1, X, l, γ, γtilde) * π[X+1]
    end

    return ifelse(x0 > x1, 0, 1)
end


function find_πX(ap::Params, γ, γtilde, nSamples=10_000)
    @unpack K, μ_l, l_dist = ap

    π_0 = ones(K+1) ./ (K+1)
    π_update = Array{Float64}(undef, K+1)

    dist, iter = 1.0, 0
    while (dist > 1e-3) & (iter < 1e2)
        iter += 1

        X_count = zeros(Int, K+1)
        Random.seed!(42)
        for nsample in 1:nSamples
            n_x0, n_x1 = 0, 0

            l_vals = μ_l .+ rand(l_dist, K)
            for _l in l_vals

                # If telling truth, increment truth
                # TODO: Can read ibar from formulas... Maybe can compute this
                #       faster
                if find_xstar(ap, _l, γ, γtilde, π_0) == 1
                    n_x1 += 1
                else
                    n_x0 += 1
                end
            end
            X_count[n_x0+1] += 1
        end

        π_update = X_count ./ nSamples
        dist = maximum(abs, π_update - π_0)

        δ = 0.35
        copyto!(π_0, δ .* π_update + (1 - δ) .* π_0)

    end
    println("Took $iter iterations and ended with $dist distance")
    @show π_update

    return π_update
end


function malevolent_cost(p::Params, γ, γtilde)
    # Unpack parameters
    @unpack μ_l, l_dist, K, ξm = p

    # Given new εtilde, determine distribution of z
    πX = find_πX(p, γ, γtilde)

    E_cost = 0.0
    for X in 0:K
        E_cost += X * γtilde[X+1] * πX[X+1]
    end

    return E_cost
end


function lie_constraint(p::Params, γ, γtilde)
    @unpack K, ξm = p

    # Compute the pmf and cmf for Xs
    πX = find_πX(p, γ, γtilde)
    FX = cumsum(πX)

    # Figure out where to evaluate cmf and return
    # (1 - F(1/2)) > ξm  <-- Lie constraint
    return ξm - (1 - FX[ceil(Int, K/2) + 1])
end


function optimize_γtilde_1(p::Params, γ)

    # Set up optimization
    opt = Opt(:LN_COBYLA, p.K+1)

    let p=p, γ=γ

        function myobj(γtilde::Vector, grad::Vector)
            @show γtilde
            return malevolent_cost(p, γ, γtilde)
        end

        function mycons(γtilde::Vector, grad::Vector)
            return lie_constraint(p, γ, γtilde)
        end
        xtol_abs!(opt, 1e-4)
        lower_bounds!(opt, zeros(p.K + 1))
        min_objective!(opt, myobj)
        inequality_constraint!(opt, mycons, 1e-8)
    end

    minf, minx, ret = optimize(opt, 2 .* rand(p.K + 1) .+ ones(p.K + 1))

    return opt, minf, minx, ret
end


function optimize_γtilde(p::Params, γ)

    # Set up optimization
    opt = Opt(:LN_AUGLAG, p.K + 1)

    let p=p, γ=γ

        function myobj(γtilde::Vector, grad::Vector)
            @show γtilde
            return malevolent_cost(p, γ, γtilde)
        end

        function mycons(γtilde::Vector, grad::Vector)
            return lie_constraint(p, γ, γtilde)
        end

        opt2 = Opt(:LN_NELDERMEAD, p.K+1)
        xtol_abs!(opt2, 1e-4)

        lower_bounds!(opt, zeros(p.K + 1))
        min_objective!(opt, myobj)
        inequality_constraint!(opt, mycons, 1e-8)
        xtol_abs!(opt, 1e-4)

        local_optimizer!(opt, opt2)

    end

    minf, minx, ret = optimize(opt, ones(p.K + 1))

    return opt, minf, minx, ret
end


function optimize_γtilde_bff(p::Params, γ)

    # Start with wide bounds but make adjacent elements the same
    nvals = 4
    γtilde_vals = range(0.0, stop=4.0, length=nvals)

    γtilde = zeros(p.K+1)
    γtilde_star = copy(γtilde)
    cost_star = 1e5

    # Instead of search over all γtilde_vals -- Only look at ones adjacent to
    # the other values... This drastically limits what we need to search and
    # adds a degree of continuity
    @nloops 6 i d->1:nvals begin
        # Determine all of the indices
        indices = @ntuple 6 i

        @nexprs 3 j-> γtilde[2*(j-1) + 1] = γtilde_vals[indices[j]]
        @nexprs 3 j-> γtilde[2*j] = γtilde_vals[indices[j]]

        # Feasible
        if lie_constraint(p, γ, γtilde) < 0
            cost = malevolent_cost(p, γ, γtilde)
            if cost < cost_star
                cost_star = cost
                copyto!(γtilde_star, γtilde)
            end
        end
    end

    return γtilde_star
end


#=
p = Params(9, 1.0, Normal(0, 0.5), 0.05, 0.025, 0.65, 0.65)

γ = ones(p.K+1, 2)
γtilde = ones(p.K+1)

πX = find_πX(p, γ, γtilde)
dG_dγtilde = Array{Float64}(undef, 10, 10)
for i in 1:10
    ε = zeros(10)
    ε[i] = 0.01
    ΔG = find_πX(p, γ, γtilde + ε)
    dG_dγtilde[:, i] = (ΔG - πX) / 0.01
end


out = optimize_γtilde_1(p, γ)  # Use COBYLA
out = optimize_γtilde(p, γ)  # Use Langrangian w Nelder-Mead
=#
