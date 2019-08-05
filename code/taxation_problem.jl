using LinearAlgebra
using Parameters
using QuantEcon


@with_kw struct TaxModel
    # Parameters
    γ::Float64 = 0.5  # Fraction of margin that can be seized
    r::Float64 = 6.5e-5  # About 2.5% annual return paid daily

    # Margin process
    M::MarkovChain{Float64,Array{Float64,2},Array{Float64,1}} = MarkovChain([0.5 0.5; 0.5 0.5], [0.98, 1.02])
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


function find_tax_policy(model::TaxModel, Xstar=compute_state_dividend_payments(model))
    # Unpack parameters
    @unpack γ, r, M = model
    nM = length(M.state_values)

    # Create a savings grid
    nD = 500
    D = range(0.0, maximum(M.state_values), length=nD)

    # Allocate space for policy and value functions
    Tstar = zeros(nD, nM)
    Vstar = zeros(nD, nM)
    Vold = copy(Vstar)

    dist, iter = 10.0, 0
    while (dist > 1e-8) && (iter < 5000)

        # Iterate over states
        for iM in 1:nM
            for iD in 1:nD
                # Probability of each new state
                p_iM = M.p[iM, :]

                # Choose the best savings for tomorrow
                _Tstar = 0.0
                _Vstar = 1e8
                for iDtp1 in 1:nD
                    # Budget constraint tells us what taxes must be today
                    _T = max(0.0, D[iDtp1] + Xstar[iM] - (1 + r)*D[iD])

                    # Compute expected value of being at tomorrow with the
                    # given savings and each possible margin level
                    EV = 0.0
                    for iMtp1 in 1:nM
                        EV += Vold[iDtp1, iMtp1]
                    end
                    _V = _T*_T + (1 / (1+r))*EV

                    # Update if it is better than previous -- We are minimizing
                    # the tax squared
                    if _V < _Vstar
                        _Tstar = _T
                        _Vstar = _V
                    end
                end

                # Update functions
                Tstar[iD, iM] = _Tstar
                Vstar[iD, iM] = _Vstar
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

    return Tstar, Xstar, Vstar
end
