#=
See notes for description of environment
=#
using Distributions
using LinearAlgebra
using NLopt
using Parameters
using Random

struct Params
    # Number of voters
    K::Int

    # Parameters for distribution
    μ_i::Float64
    i::Distribution

    # Malevolent/Oracle parameters
    ξm::Float64
    ξo::Float64
end


function voter_payoffs(ap::Params, x, Z, S, i, ε, εtilde)
    @unpack K = ap

    p_prime = ifelse(Z > round(Int, K/2), 0.0, 1.0)
    tt_bonus = ifelse(x==S, i, 0.0)

    return p_prime*ε[x+1, Z+1] + εtilde[x+1, Z+1, S+1] + tt_bonus
end


function find_xstar(ap::Params, S, i, ε, εtilde, π)

    @unpack K = ap

    # Set x0/x1 to 0
    x0, x1 = (0.0, 0.0)
    for z in 0:K
        x0 += voter_payoffs(ap, 0, z, S, i, ε, εtilde) * π[z+1]
        x1 += voter_payoffs(ap, 1, z, S, i, ε, εtilde) * π[z+1]
    end

    return ifelse(x0 > x1, (0, x0), (1, x1))
end


function find_πz(ap::Params, S, ε, εtilde, nSamples=10_000)
    @unpack K, μ_i, i = ap

    π_0 = ones(K+1) ./ (K+1)
    π_update = Array{Float64}(undef, K+1)

    dist, iter = 1.0, 0
    while dist > 1e-6

        iter += 1

        z_count = zeros(Int, K+1)
        Random.seed!(42)
        for nsample in 1:nSamples
            n_x0, n_x1 = 0, 0

            i_vals = μ_i .+ rand(i, K)
            for _i in i_vals

                # If telling truth, increment truth
                if find_xstar(ap, S, _i, ε, εtilde, π_0)[1] == 1
                    n_x1 += 1
                else
                    n_x0 += 1
                end
            end
            z_count[n_x0+1] += 1
        end

        π_update = z_count / nSamples
        dist = maximum(abs, π_update - π_0)
        copyto!(π_0, π_update)

        if mod(iter, 5) == 0
            println("$iter Iterations and $dist Distance")
            println("$(π_update[1:10])")
        end

    end

    return π_update
end


function malevolent_cost(p::Params, S, ε, εtildeV)
    # Unpack parameters
    @unpack μ_i, i, K, ξm = p

    # Given new εtilde, determine distribution of z
    πz = find_πz(p, S, ε, εtilde)

    E_cost = 0.0
    for z in 0:K
        E_cost += z * εtilde[1-S + 1, z+1, S+1] * πz[z+1]
    end

    return E_cost
end


function lie_constraint(p::Params, S, ε, εtildeV)
    @unpack K, ξm = p

    # Compute the pmf and cmf for Zs
    πz = find_πz(p, S, ε, εtilde)
    Fz = cumsum(πz)

    # Figure out where to evaluate cmf and return
    # (1 - F(1/2)) > ξm  <-- Lie constraint
    return ξm - (1 - Fz[ceil(Int, K/2) + 1])
end


function optimize_εtilde(p::Params, ε)
    let p=p, ε=ε
        myobj(εtilde::Vector, grad::Vector) = malevolent_cost(p, 1, ε, εtilde)
        mycons(εtilde::Vector, grad::Vector) = lie_constraint(p, 1, ε, εtilde)
    end

    # Set up optimization
    opt = Opt(:LN_NELDERMEAD, ?)

    lower_bounds!(opt, zeros())
    min_objective!(opt, myobj)
    inequality_constraint!(opt, mycons, 1e-8)

    minf, minx, ret = optimize(opt, ?)
    return opt, minf, minx, ret
end

#=
ε = ones(2, 101)
εtilde = zeros(2, 101, 2)
εtilde[1, :, :] = 3*zeros(101, 2)

p = Params(100, 1.0, Normal(0, 0.5), 0.75, 0.85)

πz = find_πz(p, 1, ε, εtilde)

mp = MalevolentParams(0.75)

=#
