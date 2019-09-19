using Distributions
using GLPK
using Interpolations
using JuMP
using LinearAlgebra
using Parameters
using PlotlyJS
using Random
using QuantEcon


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


function find_tax_policy(
        model::TaxModel, Xstar=constrained_dividend_payments(model);
        nD::Int=600, tol::Float64=1e-6, maxiter::Int=10_000, n_howard_steps::Int=25
    )
    # Unpack parameters
    @unpack γ, r, M = model
    nM = length(M.state_values)

    # Create a savings grid
    D = range(0.0, maximum(M.state_values)/3.0, length=nD)

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
            M_t = M.state_values[iM]

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
                    _V = (_T - 0.002*M_t)*(_T - 0.002*M_t) + (1 / (1+r))*EV

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
function create_scurve_mc(;M0=1.0, Mbar=1000.0, g=0.005, σ0=0.15, σ=0.02, nM=499)

    # Create evenly spaced points between M0 and Mbar with a value at 0
    M_values = [0.0; collect(range(M0, Mbar, length=nM))]
    # Create P and add absorbing disaster state
    P = zeros(nM+1, nM+1)
    P[1, 1] = 1.0
    p_disaster = [0.0; 0.01 .* 1 ./ M_values[2:end]]

    # Allocate space for transition matrix and fill
    # by determining transition probabilities
    for iM_t in 2:nM+1
        # Pull out today's margin
        M_t = M_values[iM_t]

        # Compute expected value and std deviation of margin
        # tomorrow and create distribution over t+1 values
        E_M_tp1 = M_t * (1 + g*(1.0 - M_t/Mbar))
        std_dev = σ0 + σ*M_t
        d_M_tp1 = Normal(E_M_tp1, std_dev)

        # Add disaster risk if system is smaller than 50
        P[iM_t, 1] = p_disaster[iM_t] ./ (1.0 + p_disaster[iM_t])

        for iM_tp1 in 2:nM+1
            # How much did we deviate from our expectation
            lb = iM_tp1 == 2 ? -Inf : M_values[iM_tp1 - 1]
            ub = iM_tp1 == nM+1 ? Inf : M_values[iM_tp1]

            # Fill in probability
            P[iM_t, iM_tp1] = (cdf(d_M_tp1, ub) - cdf(d_M_tp1, lb)) / (1.0 + p_disaster[iM_t])
        end
    end

    return MarkovChain(P, M_values)
end


function simulate_logistic_growth(;M0=1.0, Mbar=1000.0, g=0.005, σ0=0.10, σ=0.05, N=1, T=2500)
    # Allocate memory and set initial value to M0
    out = Array{Float64}(undef, N, T)
    out[:, 1] .= M0

    for t in 2:T, n in 1:N
        # Mtp1 = Mt + g Mt (1 - Mt/Mbar)
        Mt = out[n, t-1]
        σt = σ0 + σ*Mt
        out[n, t] = min(max(Mt + g*Mt*(1 - Mt/Mbar) + σt*randn(), M0), Mbar)
    end

    return out
end


function create_comparison_table(nMs=[500, 1000, 2500, 5000], Ts)
    # Create space to store data
    out = Array{Float64}(undef, length(nMs)+1, length(Ts)*2)

    println("Starting true logistic growth simulations")
    lg_sims = simulate_logistic_growth(;M0=1.0, Mbar=1000.0, g=0.005, σ=0.15, N=500, T=2000)
    for (i, t) in enumerate(Ts)
        out[1, 2*i - 1] = mean(lg_sims[:, t])
        out[1, 2*i] = std(lg_sims[:, t])
    end
    out[1, :]

    for (j, nM) in enumerate(nMs)
        println("Starting simulations for $nM approximation points")
        mc = create_scurve_mc(;M0=1.0, Mbar=1000.0, g=0.005, σ=0.15, nM=nM)
        mc_sims = hcat([simulate(mc, 2000; init=1) for _ in 1:500]...)'
        for (i, t) in enumerate(Ts)
            out[j+1, 2*i - 1] = mean(mc_sims[:, t])
            out[j+1, 2*i] = std(mc_sims[:, t])
        end
    end
end

#=

mc = create_scurve_mc(;g=0.030, σ0=0.35, σ=0.025, nM=1500)
model = TaxModel(0.5, 2.1e-3, mc)

# Create margin growth figure
nsims = 150
Random.seed!(61089)
mc_sims = [simulate(mc, 12*25; init=2) for i in 1:nsims]
scatters = [
    scatter(;
        x=range(0, 25, length=12*25), y=mc_sims[i],
        opacity=ifelse(mc_sims[i][end] > 1.0, 0.10, 0.35),
        marker=attr(color="black"),
        showlegend=false
    )
    for i in 1:nsims
]
smg_plot = plot(
    scatters,
    Layout(
        title="Margin Growth Process", xaxis_title="Years from Launch",
        yaxis_title="Millions of Dollars"
    )
)
savefig(smg_plot, "../notes/TaxationPlanImages/StochasticMarginGrowth.png")

# Compute buyback policy
cdp = constrained_dividend_payments(model)

mc_ind_sims = [simulate_indices(mc, 12*25; init=2) for i in 1:10]
annualized_buybacks = ((1.0 .+ (cdp ./ model.M.state_values)).^12 .- 1.0) .* 100

abb_bar = [
    bar(;
        x=model.M.state_values, y=annualized_buybacks,
        marker=attr(color="rgb(255, 74, 74)"), showlegend=false
    )
]

abb_scatter = [
    scatter(;
        x=range(0, 25, length=12*25), y=annualized_buybacks[mc_ind_sims[i]],
        opacity=0.35, marker=attr(color="black"), showlegend=false
    )
    for i in 1:10
]

abb_bar_plot = plot(
    abb_bar,
    Layout(
        title="Annualized Buyback Percents (as % of Margin)",
        xaxis_title="System Margin (millions of dollars)",
        yaxis_range=[0, 20]
    )
);

abb_scatter_plot = plot(
    abb_scatter,
    Layout(
        title="Simulated Buyback Percent (as % of Margin)",
        xaxis_title="Years since launch",
        yaxis_range=[0, 20]
    )
);
abb_plot = [abb_scatter_plot, abb_bar_plot]

savefig(abb_plot, "../notes/TaxationPlanImages/StochasticBuybacks.png")

iDstar, Tstar, Xstar, D, Vstar = find_tax_policy(
    model, cdp; nD=500, maxiter=1, n_howard_steps=25
)

=#

#=
mc = create_scurve_mc(1.0, 1000.0, 0.025, 15.0, 1000)
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
