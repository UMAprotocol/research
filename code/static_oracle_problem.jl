#=
See notes for description of environment
=#
using Distributions
using LinearAlgebra
using NLopt
using Parameters
using Random

struct Params{D<:Distribution}
    # Number of voters
    K::Int

    # Parameters for distribution
    μ_i::Float64
    i::D

    # Malevolent/Oracle parameters
    ξm::Float64
    ξo::Float64
end


function voter_payoffs(ap::Params, x, Z, i, ε, εtilde)
    @unpack K = ap

    p_prime = ifelse(Z > round(Int, K/2), 0.0, 1.0)
    tt_bonus = ifelse(x==1, i, 0.0)

    return p_prime*ε[Z+1, x+1] + εtilde[Z+1, x+1] + tt_bonus
end


function find_xstar(ap::Params, i, ε, εtilde, π)

    @unpack K = ap

    # Set x0/x1 to 0
    x0, x1 = (0.0, 0.0)
    for z in 0:K
        x0 += voter_payoffs(ap, 0, z, i, ε, εtilde) * π[z+1]
        x1 += voter_payoffs(ap, 1, z, i, ε, εtilde) * π[z+1]
    end

    return ifelse(x0 > x1, (0, x0), (1, x1))
end


function find_πz(ap::Params, ε, εtilde, nSamples=10_000)
    @unpack K, μ_i, i = ap

    π_0 = ones(K+1) ./ (K+1)
    π_update = Array{Float64}(undef, K+1)

    dist, iter = 1.0, 0
    while dist > 1e-3
        iter += 1

        z_count = zeros(Int, K+1)
        Random.seed!(42)
        for nsample in 1:nSamples
            n_x0, n_x1 = 0, 0

            i_vals = μ_i .+ rand(i, K)
            for _i in i_vals

                # If telling truth, increment truth
                # TODO: Can read ibar from formulas... Maybe can compute this
                #       faster
                if find_xstar(ap, _i, ε, εtilde, π_0)[1] == 1
                    n_x1 += 1
                else
                    n_x0 += 1
                end
            end
            z_count[n_x0+1] += 1
        end

        π_update = z_count ./ nSamples
        dist = maximum(abs, π_update - π_0)
        copyto!(π_0, π_update)

        if mod(iter, 5) == 0
            println("$iter Iterations and $dist Distance")
            println("$(π_update[1:10])")
        end

    end

    return π_update
end


function malevolent_cost(p::Params, ε, εtilde)
    # Unpack parameters
    @unpack μ_i, i, K, ξm = p

    # Given new εtilde, determine distribution of z
    πz = find_πz(p, ε, εtilde)

    E_cost = 0.0
    for z in 0:K
        E_cost += z * εtilde[z+1, 0+1] * πz[z+1]
    end

    return E_cost
end


function lie_constraint(p::Params, ε, εtilde)
    @unpack K, ξm = p

    # Compute the pmf and cmf for Zs
    πz = find_πz(p, ε, εtilde)
    Fz = cumsum(πz)

    # Figure out where to evaluate cmf and return
    # (1 - F(1/2)) > ξm  <-- Lie constraint
    return ξm - (1 - Fz[ceil(Int, K/2) + 1])
end


function optimize_εtilde(p::Params, ε)

    # Set up optimization
    opt = Opt(:LN_COBYLA, p.K + 1)

    let p=p, ε=ε

        function myobj(εtilde::Vector, grad::Vector)
            εtildeM = hcat(εtilde, zeros(size(εtilde)))
            return malevolent_cost(p, ε, εtildeM)
        end

        function mycons(εtilde::Vector, grad::Vector)
            εtildeM = hcat(εtilde, zeros(size(εtilde)))
            return lie_constraint(p, ε, εtildeM)
        end


        lower_bounds!(opt, zeros(p.K + 1))
        min_objective!(opt, myobj)
        inequality_constraint!(opt, mycons, 1e-8)
    end

    minf, minx, ret = optimize(opt, zeros(p.K + 1))

    return opt, minf, minx, ret
end


#=
ε = ones(101, 2)
εtilde = zeros(101, 2)
εtilde[1, :] = ones(101)

p = Params(100, 1.0, Normal(0, 0.5), 0.75, 0.85)

πz = find_πz(p, ε, εtilde)

=#
