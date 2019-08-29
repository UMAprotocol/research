using LinearAlgebra
using Parameters
using QuantEcon


@with_kw struct TaxModel{T<:AbstractArray}
    # Parameters
    γ::Float64 = 0.5  # Fraction of margin that can be seized
    r::Float64 = 6.5e-5  # About 2.5% annual return paid daily

    # Margin process
    M::MarkovChain{Float64,Array{Float64,2},T} = MarkovChain([0.5 0.5; 0.5 0.5], [0.98, 1.02])
end


function compute_state_dividend_payments(model::TaxModel)
    # Unpack parameters
    @unpack γ, r, M = model
    nM = length(M.state_values)

    # Can compute what dividend payments are for each state using
    # geometric sums...
    X = 2*γ * ((I - (1 / (1+r))*M.p) * M.state_values)

    return X
end


function find_tax_policy(
        model::TaxModel, Xstar=compute_state_dividend_payments(model);
        nD::Int=500, tol::Float64=1e-10, maxiter::Int=10_000, n_howard_steps::Int=25
    )
    # Unpack parameters
    @unpack γ, r, M = model
    nM = length(M.state_values)

    # Create a savings grid
    D = range(0.0, maximum(M.state_values), length=nD)

    # Allocate space for policy and value functions
    iDstar = zeros(Int, nD, nM)
    Tstar = zeros(nD, nM)
    Vstar = zeros(nD, nM)
    Vhoward = copy(Vstar)
    Vold = copy(Vstar)

    dist, iter = 10.0, 0
    while (dist > tol) && (iter < maxiter)

        # Iterate over states and find new new policies
        for iM in 1:nM
            for iD in 1:nD
                # Probability of each new state
                p_iM = M.p[iM, :]

                # Choose the best savings for tomorrow
                _iDstar = 0
                _Tstar = 0.0
                _Vstar = 1e8
                for iDtp1 in 1:nD
                    # Budget constraint tells us what taxes must be today
                    _T = max(0.0, D[iDtp1] + Xstar[iM] - (1 + r)*D[iD])

                    # Compute expected value of being at tomorrow with the
                    # given savings and each possible margin level
                    EV = 0.0
                    for iMtp1 in 1:nM
                        EV += p_iM[iMtp1] * Vold[iDtp1, iMtp1]
                    end
                    _V = _T*_T + (1 / (1+r))*EV

                    # Update if it is better than previous -- We are minimizing
                    # the tax squared
                    if _V < _Vstar
                        _iDstar = iDtp1
                        _Tstar = _T
                        _Vstar = _V
                    end
                end

                # Update functions
                iDstar[iD, iM] = _iDstar
                Tstar[iD, iM] = _Tstar
                Vstar[iD, iM] = _Vstar
            end
        end

        # Iterate over states and apply Howard steps
        for _hstep in 1:n_howard_steps
            copy!(Vhoward, Vstar)

            for iM in 1:nM, iD in 1:nD
                # Probability of each new state
                p_iM = M.p[iM, :]

                # Get policies
                iDtp1 = iDstar[iD, iM]
                _T = max(0.0, D[iDtp1] + Xstar[iM] - (1 + r)*D[iD])

                # Compute expected value of being at tomorrow with the
                # given savings and each possible margin level
                EV = 0.0
                for iMtp1 in 1:nM
                    EV += p_iM[iMtp1] * Vhoward[iDtp1, iMtp1]
                end
                Vstar[iD, iM] = _T*_T + (1 / (1+r))*EV

            end
        end

        # Check distance etc...
        iter += 1
        dist = maximum(abs, Vstar - Vold)
        copy!(Vold, Vstar)
        if mod(iter, 25) == 0
            println("Distance is $dist at iteration $iter")
        end
    end

    return iDstar, Tstar, Xstar, D, Vstar
end


# Mtp1 = Mt + g Mt (1 - Mt/Mbar)
function create_scurve_mc(M0=1.0, Mbar=1000.0, g=0.5, σ=2.5, nM=499)
    # Create evenly spaced points between M0 and Mbar
    M_values = collect(range(M0, Mbar*(1 + σ), length=nM))

    # Allocate space for transition matrix and fill
    # by determining transition probabilities
    P = zeros(nM, nM)
    for iM_t in 1:nM
        # Pull out today's margin
        M_t = M_values[iM_t]

        # Compute expected value and std deviation of margin
        # tomorrow and create distribution over t+1 values
        E_M_tp1 = M_t * (1 + g*(1.0 - M_t/Mbar))
        std_dev = M_t*σ
        d_M_tp1 = Normal(E_M_tp1, std_dev)

        for iM_tp1 in 1:nM
            # How much did we deviate from our expectation
            lb = iM_tp1 == 1 ? -Inf : M_values[iM_tp1 - 1]
            ub = iM_tp1 == nM ? Inf : M_values[iM_tp1]

            # Fill in probability
            P[iM_t, iM_tp1] = cdf(d_M_tp1, ub) - cdf(d_M_tp1, lb)
        end
    end

    return MarkovChain(P, M_values)
end


function create_scurve_mc(M0=1.0, Mbar=1000.0, g=0.5, σ=2.5, nM=499)
    # Create evenly spaced points between M0 and Mbar
    M_values = collect(range(M0, Mbar*(1 + σ), length=nM))

    # Allocate space for transition matrix and fill
    # by determining transition probabilities
    P = zeros(nM, nM)
    for iM_t in 1:nM
        # Pull out today's margin
        M_t = M_values[iM_t]

        # Compute expected value and std deviation of margin
        # tomorrow and create distribution over t+1 values
        E_M_tp1 = M_t * (1 + g*(1.0 - M_t/Mbar))
        std_dev = σ
        d_M_tp1 = Normal(E_M_tp1, std_dev)

        for iM_tp1 in 1:nM
            # How much did we deviate from our expectation
            lb = iM_tp1 == 1 ? -Inf : M_values[iM_tp1 - 1]
            ub = iM_tp1 == nM ? Inf : M_values[iM_tp1]

            # Fill in probability
            P[iM_t, iM_tp1] = cdf(d_M_tp1, ub) - cdf(d_M_tp1, lb)
        end
    end

    return MarkovChain(P, M_values)
end
#=
# Toy example
μ = 1000.0
σ = 50.0
mc_p = MarkovChain([0.95 0.05 0.0; 0.025 0.95 0.025; 0.025 0.025 0.95], [μ - σ, μ, μ + σ])
model_p = TaxModel(;r=0.00206, M=mc_p)
iDstar, Tstar, Xstar, D, Vstar = find_tax_policy(model_p; nD=1000, tol=1e-3, n_howard_steps=10)

# Persistent model
μ = 1000.0
σ = 15.0
mc_p = rouwenhorst(7, 0.999, sqrt(σ * (1.0 - 0.999^2)), μ*0.001)
model_p = TaxModel(;M=mc_p)
iDstar, Tstar, Xstar, D, Vstar = find_tax_policy(model_p; nD=2500, tol=1e-3, n_howard_steps=10)

# Persistent model
μ = 1000.0
σ = 250.0
mc_p = rouwenhorst(5, 0.95, sqrt(σ * (1.0 - 0.95^2)), μ*0.05)
model_p = TaxModel(;r=0.00206, M=mc_p)
iDstar, Tstar, Xstar, D, Vstar = find_tax_policy(model_p; nD=2500, tol=1e-3, n_howard_steps=10)

mc_np = rouwenhorst(7, 0.05, sqrt(0.01 * (1.0 - 0.05^2)), 1.0*0.95)
model_np = TaxModel(;M=mc_np)

# Growth model
mc_states = [
  0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.
]
=#
