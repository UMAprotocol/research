module UMATaxModel

using Distributions
using GLPK
using Interpolations
using JLD
using JuMP
using LinearAlgebra
using Optim
using ORCA
using Parameters
using PlotlyJS
using Random
using QuantEcon


const itp_type = BSpline(Quadratic(Natural(OnGrid())))


@with_kw struct TaxModel{T<:AbstractArray}
    # Parameters
    γ::Float64 = 0.5  # Fraction of margin that can be seized
    r::Float64 = 2e-3  # About 2.5% annual return paid daily
    τ_target::Float64=0.00165  # About 2% annula paid monthly

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


function period_penalty(M_t, T_t, τ_target=0.00165)
    # Compute implied tax rate
    τ = ifelse(M_t < 1e-8, 0.0, T_t / M_t)

    # If tax is below that, then 0 penalty
    out = (τ - τ_target)^2

    return out
end


function T_from_BC(model, Dt, Dtp1, Xt)
    Tt = Dtp1 .+ Xt .- (1.0 + model.r)*Dt
    return Tt
end


function find_tax_policy(
        model::TaxModel, Xstar=constrained_dividend_payments(model);
        nD::Int=600, tol::Float64=1e-7, maxiter::Int=1_000, n_howard_steps::Int=25
    )
    # Unpack parameters
    @unpack γ, r, τ_target, M = model
    nM = length(M.state_values)

    D = range(0.0, maximum(M.state_values)/3.0, length=nD)

    # Allocate space for policy and value functions
    Dstar = zeros(nD, nM)  # (D_t x M_t)
    Tstar = zeros(nD, nM)  # (D_t x M_t)
    Vstar = zeros(nD, nM)  # (D_t x M_t)
    Vold = copy(Vstar)  # (D_t x M_t)
    EV = Vold * model.M.p'  # (D_tp1 x M_t)

    dist, iter = 10.0, 0
    while (dist > tol) && (iter < maxiter)

        @show extrema(Vstar)

        # Compute expected values up front (D_tp1 x M_t)
        EV .= Vold * model.M.p'

        # Iterate over states and find new new policies
        Threads.@threads for iM in 2:nM
            # Get M and X
            M_t = model.M.state_values[iM]
            X_t = Xstar[iM]

            # Create interpolator
            EV_itp = extrapolate(
                Interpolations.scale(interpolate(EV[:, iM], itp_type), D), Interpolations.Flat()
            )

            for (iD, D_t) in enumerate(D)
                optme(Dtp1) = (
                    period_penalty(M_t, T_from_BC(model, D_t, Dtp1, X_t), τ_target) +
                    (1.0 / (1.0+model.r))*EV_itp(Dtp1)
                )

                opt = Optim.optimize(optme, 0.0, 1.05*D[end])
                _Dstar = opt.minimizer

                Dstar[iD, iM] = _Dstar
                Tstar[iD, iM] = T_from_BC(model, D_t, _Dstar, X_t)
                Vstar[iD, iM] = opt.minimum
            end
        end

        # Fill in data for the disaster state
        Dstar[:, 1] .= 0.0
        Tstar[:, 1] .= 0.0
        Vstar[:, 1] .= Vstar[:, 2]

        # Iterate over states and apply Howard steps
        for _hstep in 1:n_howard_steps

            # Compute expected values up front (D_tp1 x M_t)
            EV .= Vstar * model.M.p'

            # Iterate over Ms and Ds
            Threads.@threads for iM in 1:nM
                # Pull out buyback
                M_t = model.M.state_values[iM]
                X_t = Xstar[iM]

                # Create interpolator
                EV_itp = extrapolate(
                    Interpolations.scale(interpolate(EV[:, iM], itp_type), D), Interpolations.Flat()
                )

                for (iD, D_t) in enumerate(D)

                    # Optimal policy
                    _Dtp1 = Dstar[iD, iM]

                    _T = T_from_BC(model, D_t, _Dtp1, X_t)
                    _V = period_penalty(M_t, _T, τ_target) + (1.0 / (1.0 + r))*EV_itp(_Dtp1)
                    Vstar[iD, iM] = _V
                end
            end
        end

        # Check distance etc...
        iter += 1
        dist = maximum(abs, Vstar - Vold)
        copy!(Vold, Vstar)
        if mod(iter, 5) == 0
            println("Distance is $dist at iteration $iter")
        end
    end

    return Dstar, Tstar, Xstar, D, Vstar
end


function simulate_rainy_day(model, D, Dstar, Tstar, Xstar; nyears=25)
    # Set useful vars
    Random.seed!(61089)
    nSim = 12*nyears

    # Simulate margin states
    iM_sim = simulate_indices(model.M, nSim; init=2)
    M_sim = model.M.state_values[iM_sim]
    X_sim = Xstar[iM_sim]

    # Allocate memory
    Dstar_itp = Interpolations.scale(
        interpolate(Dstar, (BSpline(Quadratic(Natural(OnGrid()))), NoInterp())),
        D, 1:length(model.M.state_values)
    )
    D_sim = zeros(nSim)
    T_sim = zeros(nSim)

    for t in 1:nSim

        D_tp1 = Dstar_itp(D_sim[t], iM_sim[t])
        T_sim[t] = T_from_BC(model, D_sim[t], D_tp1, X_sim[t])

        # Only store D_tp1 if not end of simulation
        if t < nSim
            D_sim[t+1] = D_tp1
        end
    end

    return D_sim, M_sim, T_sim, X_sim
end


# Mtp1 = Mt + g Mt (1 - Mt/Mbar)
function create_scurve_mc(;M0=1.0, Mbar=1000.0, g=0.005, σ0=0.15, σ=0.02, nM=499)

    # Create evenly spaced points between M0 and (1 + 3σ)*Mbar with a value at 0
    M_values = [0.0; collect(range(M0, (1 + 25*σ)*Mbar, length=nM))]

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
        out[n, t] = max(Mt + g*Mt*(1 - Mt/Mbar) + σt*randn(), M0)
    end

    return out
end


function create_comparison_table(nMs, Ts)
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

struct TaxModelSolution{AA<:AbstractArray}
    Dstar::Matrix{Float64}
    Tstar::Matrix{Float64}
    Xstar::Vector{Float64}
    D::AA
    Vstar::Matrix{Float64}
end

function compute_solution(model::TaxModel)
    # Find solution
    Dstar, Tstar, Xstar, D, Vstar = find_tax_policy(model; tol=1e-6)

    return TaxModelSolution(Dstar, Tstar, Xstar, D, Vstar)
end

function get_model_name(model::TaxModel)
    _thehash = hash(
        hash(model.τ_target, hash(model.γ, hash(model.r))),
        hash(
            reduce(hash, hash.(model.M.state_values)),
            reduce(hash, hash.(model.M.p))
        )
    )

    return string(_thehash)
end


function save_solution(model::TaxModel, sol::TaxModelSolution)
    fname = "./solutions/model_$(get_model_name(model)).jld"

    jldopen(fname, "w") do file
        write(file, "Dstar", sol.Dstar)
        write(file, "Tstar", sol.Tstar)
        write(file, "Xstar", sol.Xstar)
        write(file, "D", sol.D)
        write(file, "Vstar", sol.Vstar)

        write(file, "params/r", model.r)
        write(file, "params/gamma", model.γ)
        write(file, "params/tau_target", model.τ_target)
        write(file, "params/M_statevalues", model.M.state_values)
        write(file, "params/M_p", model.M.p)
    end

    return 1
end


function load_solution(model::TaxModel)
    fname = "./solutions/model_$(get_model_name(model)).jld"

    d = load(fname)

    Dstar = d["Dstar"]
    Tstar = d["Tstar"]
    Xstar = d["Xstar"]
    D = d["D"]
    Vstar = d["Vstar"]
    sol = UMATaxModel.TaxModelSolution(Dstar, Tstar, Xstar, D, Vstar)

    jldopen(fname, "r") do file
        r = read(file, "params/r")
        gamma = read(file, "params/gamma")
        tau_target = read(file, "params/tau_target")
        M_statevalues = read(file, "params/M_statevalues")
        M_p = read(file, "params/M_p")
        modtest = UMATaxModel.TaxModel(gamma, r, tau_target, MarkovChain(M_p, M_statevalues))

        if !(get_model_name(modtest) == get_model_name(model))
            warn("Model read in is not the same as passed in")
        end
    end

    return model, sol
end

end
#=
umared = "rgb(255, 74, 74)"

mc = create_scurve_mc(;g=0.030, σ0=0.35, σ=0.025, nM=1501)
model = TaxModel(0.5, 2.1e-3, 0.00165, mc)

model_high_γ = TaxModel(0.75, 2.1e-3, 1.65e-3, mc)
model_higher_γ = TaxModel(0.9, 2.1e-3, 1.65e-3, mc)

model_lowest_τtarget = TaxModel(0.5, 2.1e-3, 1.65e-5, mc)
model_low_τtarget = TaxModel(0.5, 2.1e-3, 1.65e-4, mc)

# Compute buyback policy
cdp = constrained_dividend_payments(model)
# iDstar, Tstar, Xstar, D, Vstar = find_tax_policy(model, cdp; maxiter=1)

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
        yaxis_title="Millions of Dollars",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
    )
)
savefig(smg_plot, "../notes/TaxationPlanImages/StochasticMarginGrowth.png")

mc_ind_sims = [simulate_indices(mc, 12*25; init=2) for i in 1:5]
annualized_buybacks = ((1.0 .+ (cdp ./ model.M.state_values)).^12 .- 1.0) .* 100

abb_bar = [
    bar(;
        x=model.M.state_values, y=annualized_buybacks,
        marker=attr(color=umared, showlegend=false)
    )
]

abb_scatter = [
    scatter(;
        x=range(0, 25, length=12*25), y=annualized_buybacks[mc_ind_sims[i]],
        opacity=0.35, marker=attr(color="black"), showlegend=false
    )
    for i in 1:5
]

abb_bar_plot = plot(
    abb_bar,
    Layout(
        title="Annualized Buyback Percents (as % of Margin)",
        xaxis_title="System Margin (millions of dollars)",
        yaxis_range=[0, 45]
    )
);

abb_scatter_plot = plot(
    abb_scatter,
    Layout(
        title="Simulated Buyback Percent (as % of Margin)",
        xaxis_title="Years since launch",
        yaxis_range=[0, 45]
    )
);
abb_plot = [abb_bar_plot, abb_scatter_plot];
relayout!(
    abb_plot,
    Dict(
        :showlegend=>false,
        :paper_bgcolor=>"rgba(0,0,0,0)", :plot_bgcolor=>"rgba(0,0,0,0)"
    )
)
abb_plot
savefig(abb_plot, "../notes/TaxationPlanImages/StochasticBuybacks.png")

Dstar, Tstar, Xstar, D, Vstar = find_tax_policy(
    model, cdp; nD=500,
)

τstar = (1.0 .+ Tstar ./ model.M.state_values').^12 .- 1.0
τstar[:, 1] .= 0.0

τ_heat_plot = plot(
    heatmap(;z=τstar[1:100, 501:end], x=D[1:100], y=model.M.state_values[501:end]),
    Layout(
        title="Annualized Tax Rates",
        xaxis_title="Rainy Day Fund (Millions of USD)",
        yaxis_title="Margin (Millions of USD)",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
    )
)
savefig(τ_heat_plot, "../notes/TaxationPlanImages/AnnualizedTaxRates.png")

@unpack Dstar, Tstar, Xstar, D, Vstar = sol
D_sim, M_sim, T_sim, X_sim = simulate_rainy_day(model, D, Dstar, Tstar, Xstar)

@unpack Dstar, Tstar, Xstar, D, Vstar = sol_high_γ
D_sim, M_sim, T_sim, X_sim = simulate_rainy_day(model_high_γ, D, Dstar, Tstar, Xstar)

@unpack Dstar, Tstar, Xstar, D, Vstar = sol_lowτtarget
D_sim, M_sim, T_sim, X_sim = simulate_rainy_day(model_low_τtarget, D, Dstar, Tstar, Xstar)

τ_sim = (1.0 .+ T_sim ./ M_sim).^12 .- 1.0
bbr_sim = (1.0 .+ X_sim ./ M_sim).^12 .- 1.0

nSim = length(D_sim)
t_vals = range(0, nSim/12; length=nSim)

M_sim_plot = plot(
    scatter(;x=t_vals, y=M_sim, marker=attr(color="black"), opacity=0.35, showlegend=false),
    Layout(;yaxis_title="Margin (Millions of USD)", autosize=false, height=300, width=500)
);
τ_sim_plot = plot(
    scatter(;x=t_vals, y=τ_sim, marker=attr(color=umared), showlegend=false),
    Layout(;yaxis_title="Annualized Tax Rates", autosize=false, yaxis_range=[0.0, 0.25], height=300, width=500)
);
D_sim_plot = plot(
    scatter(;x=t_vals, y=D_sim, marker=attr(color="black"), opacity=0.35, showlegend=false),
    Layout(;yaxis_title="Rainy Day Fund (Millions of USD)", autosize=false, height=300, width=500)
);
bbr_sim_plot = plot(
    scatter(;x=t_vals, y=bbr_sim, marker=attr(color="black"), opacity=0.35, showlegend=false),
    Layout(;yaxis_title="Buyback Rates", autosize=false, yaxis_range=[0.0, 0.25], height=300, width=500)
);

sim_plot = [M_sim_plot bbr_sim_plot; D_sim_plot τ_sim_plot];
relayout!(
    sim_plot,
    Dict(
        :autosize=>false, :height=>600, :width=>1000,
        :paper_bgcolor=>"rgba(0,0,0,0)", :plot_bgcolor=>"rgba(0,0,0,0)"
    )
)
sim_plot

savefig(sim_plot, "../notes/TaxationPlanImages/SimulationPlots.png")
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
