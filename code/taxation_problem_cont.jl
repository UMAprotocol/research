#=
Note, this file is unused and is only being kept as a reference --- At one
point, we considered working on a continuous version of the problem, but it
added little in terms of value and much in terms of complication.
=#
using Distributions
using GLPK
using Interpolations
using JuMP
using LinearAlgebra
using Optim
using Parameters
using PlotlyJS
using QuantEcon

const itp_type = (BSpline(Linear()), NoInterp())

@with_kw struct TaxModel{T<:AbstractArray}
    # Parameters
    γ::Float64 = 0.5  # Fraction of margin that can be seized
    r::Float64 = 6.5e-5  # About 2.5% annual return paid daily

    # Margin process
    M::MarkovChain{Float64,Array{Float64,2},T} = MarkovChain([0.5 0.5; 0.5 0.5], [0.98, 1.02])
end


function constrained_dividend_payments(model::TaxModel)
    # Unpack parameters
    @unpack γ, r, M = model
    nM = length(M.state_values)

    # Create JuMP optimizer type
    opt = Model(with_optimizer(GLPK.Optimizer))

    # Create state by state dividend variables
    @variable(opt, buybacks[1:nM] >= 0)

    # Objective is to minimize costs
    geosum = inv(I - (1.0/(1 + r) .* M.p))
    @objective(opt, Min, sum(geosum*buybacks))

    # Constraint is that we need to ensure the system is secure
    @constraint(opt, 2*γ.*M.state_values .<= geosum*buybacks)

    JuMP.optimize!(opt)

    if JuMP.termination_status(opt) != JuMP.MOI.OPTIMAL
        error("Non-optimal solution... Investigate further")
    end

    return JuMP.value.(buybacks)
end


function unconstrained_dividend_payments(model::TaxModel)
    # Unpack parameters
    @unpack γ, r, M = model
    nM = length(M.state_values)

    # Can compute what dividend payments are for each state using
    # geometric sums...
    X = 2*γ * ((I - (1 / (1+r))*M.p) * M.state_values)

    return X
end


function evaluate_policy(model::TaxModel, V_itp, D_t, iM_t, X_t, D_tp1)

    # Unpack parameters
    @unpack γ, r, M = model
    nM = length(M.state_values)

    # Compute implied tax
    T_t = D_tp1 + X_t - (1 + model.r)*D_t

    # Probability of each new state
    p_iM = M.p[iM_t, :]

    _EV = 0.0
    for iM_tp1 in 1:nM
        _EV += p_iM[iM_tp1] * V_itp(D_tp1, iM_tp1)
    end

    return T_t^2 + (1.0 / (1 + model.r))*_EV
end


# function d_evaluate_policy(model::TaxModel, V_itp, D_t, iM_t, X_t, D_tp1)
#
#     # Unpack parameters
#     @unpack γ, r, M = model
#     nM = length(M.state_values)
#
#     # Compute implied tax
#     d_first_term = 2*(D_tp1 + X_t - (1 + model.r)*D_t)
#
#     # Probability of each new state
#     p_iM = M.p[iM_t, :]
#
#     _dEV = 0.0
#     for iM_tp1 in 1:nM
#         _dEV += p_iM[iM_tp1] * Interpolations.gradient(V_itp, D_tp1, iM_tp1)[1]
#     end
#
#     return d_first_term + (1.0 / (1 + model.r))*_dEV
# end
#
#
# function dd_evaluate_policy(model::TaxModel, V_itp, D_t, iM_t, X_t, D_tp1)
#
#     # Unpack parameters
#     @unpack γ, r, M = model
#     nM = length(M.state_values)
#
#     # Compute implied tax
#     dd_first_term = 2*D_tp1
#
#     # Probability of each new state
#     p_iM = M.p[iM_t, :]
#
#     _ddEV = 0.0
#     for iM_tp1 in 1:nM
#         _ddEV += p_iM[iM_tp1] * Interpolations.hessian(V_itp, D_tp1, iM_tp1)[1]
#     end
#
#     return dd_first_term + (1.0 / (1 + model.r))*_ddEV
# end


function find_tax_policy(
        model::TaxModel, Xstar=constrained_dividend_payments(model);
        nD::Int=250, tol::Float64=1e-10, maxiter::Int=10_000, n_howard_steps::Int=25
    )
    # Unpack parameters
    @unpack γ, r, M = model
    nM = length(M.state_values)

    # Create a savings grid
    D = range(0.0, maximum(M.state_values), length=nD)

    # Allocate space for policy and value functions
    Dstar = zeros(nD, nM)
    Tstar = zeros(nD, nM)
    Vstar = zeros(nD, nM)
    Vhoward = copy(Vstar)
    Vold = copy(Vstar)

    dist, iter = 10.0, 0
    while (dist > tol) && (iter < maxiter)
        # Create an interpolator object
        V_itp = extrapolate(
            Interpolations.scale(interpolate(Vold, itp_type), D, 1:nM), Line()
        )

        # Iterate over states and find new new policies
        for iM_t in 1:nM
            for iD_t in 1:nD

                # Pull out the current D and X
                D_t = D[iD_t]
                X_t = Xstar[iM_t]

                let model=model, V_itp=V_itp, D_t=D_t, iM_t=iM_t, X_t=X_t
                    optme(D_tp1) = evaluate_policy(model, V_itp, D_t, iM_t, X_t, D_tp1)
                    opt = optimize(optme, D[1], D[end])

                    _Dstar = opt.minimizer
                    _Tstar = _Dstar + X_t - (1 + r)*D_t
                    _Vstar = opt.minimum

                    # Update functions
                    Dstar[iD_t, iM_t] = _Dstar
                    Tstar[iD_t, iM_t] = _Tstar
                    Vstar[iD_t, iM_t] = _Vstar
                end
            end
        end

        # Iterate over states and apply Howard steps
        for _hstep in 1:n_howard_steps
            copy!(Vhoward, Vstar)
            V_itp = extrapolate(
                Interpolations.scale(interpolate(Vhoward, itp_type), D, 1:nM), Line()
            )

            for iM_t in 1:nM, iD_t in 1:nD
                # Unpack curren state
                D_t = D[iD_t]
                X_t = Xstar[iM_t]

                # Evaluate policy
                _Dstar = Dstar[iD_t, iM_t]
                Vstar[iD_t, iM_t] = evaluate_policy(
                    model, V_itp, D_t, iM_t, X_t, _Dstar
                )
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

    return Dstar, Tstar, Xstar, D, Vstar
end


# Mtp1 = Mt + g Mt (1 - Mt/Mbar)
function create_scurve_mc(M0=1.0, Mbar=1000.0, g=0.005, σ=0.15, nM=499)
    # Create evenly spaced points between M0 and Mbar
    M_values = collect(range(M0, Mbar, length=nM))

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
mc = create_scurve_mc(1.0, 1000.0, 0.025, 15.0, 500)
model = TaxModel(0.5, 2.1e-3, mc)

cdp = constrained_dividend_payments(model)
ucdp = unconstrained_dividend_payments(model)

geosum = inv(I - (1.0/(1 + model.r) .* model.M.p))

[model.M.state_values cdp ucdp]
[model.M.state_values geosum*[cdp ucdp]]

plot([bar(;y=_x) for _x in [cdp, ucdp]])  # Plots buybacks under constrained/unconstrained

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
