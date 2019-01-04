using Distributions
using Interpolations
using Optim
using Parameters
using QuantEcon
# using NLsolve


ITPTYPE = BSpline(Quadratic(Interpolations.Line(OnGrid())))


struct PortfolioChoiceModel{AU<:AbstractUtility,T}
    # Preference parameters
    β::Float64  # Discount factor
    u::AU  # Utility function

    # Asset parameters
    r::Float64  # Risk-free interest rate
    ybar::Float64  # Period by period income
    nodes::Vector{Float64}  # Integration nodes for wealth shocks
    weights::Vector{Float64}  # Integration weights for wealth shocks

    # Voting parameters
    η::Float64  # Forgetful
    π::Float64  # Trembling hand
    κ::Float64  # Percent of margin that can be seized
    α::Float64  # What multiple of outstanding tokens needed to attack system
    xbar::Float64  # Number of vote tokens

    # Oracle parameters
    μ::Float64  # Margin
    τ::Float64  # Tax rate
    γ::Float64  # Redistribution rate

    # Grids
    logw_grid::T
end


# One model unit is ~$10,000 --> ybar ~ $2,500 per week, μ -> $1,000,000 margin
function PortfolioChoiceModel(;
        β=0.99925, u=LogUtility(), r=0.0008, ybar=0.25, σ=0.1, η=0.30, π=0.05,
        κ=0.40, α=2.0, xbar=1.0, μ=100.0, τ=0.001, γ=0.025, nw=750, wmin=1e-8,
        wmax=150.0
    )

    nodes, weights = qnwnorm(13, 0.0, σ^2)
    logw_grid = range(log(wmin), stop=log(wmax), length=nw)

    return PortfolioChoiceModel(β, u, r, ybar, nodes, weights, η, π, κ, α, xbar, μ, τ, γ, logw_grid)
end


function compute_risk_neutral_q(m::PortfolioChoiceModel, lb, ub)
    @unpack η, π, γ, μ, τ, r = m
    lhs = (1.0 + r)

    solve_me(q) =
        # Risk-free - (forget  + don't forget (mistake + no mistake)
        lhs - (η*q + (1-η)*(π*(1-γ)*(μ*τ + q) + (1-π)*(1 + γ*(π / (1-π)))*(μ*τ + q))) / q

    return bisect(solve_me, lb, ub)
end


function compute_risk_neutral_q(μ, τ, r, γ, η, π, ntokens, lb, ub)
    @unpack η, π, γ, μ, τ, r = m
    lhs = (1.0 + r)

    solve_me(q) =
        # Risk-free - (forget  + don't forget (mistake + no mistake)
        lhs - (η*q + (1-η)*(π*(1-γ)*(μ*τ/ntokens + q) + (1-π)*(1 + γ*(π / (1-π)))*(μ*τ/ntokens + q))) / q

    bisect(solve_me, lb, ub)
end


function compute_EV(m, V_itp, wealth)
    cont = 0.0

    for (n, w) in zip(m.nodes, m.weights)
        after_shock_wealth = max(1e-8, wealth + n)
        cont += w*V_itp(log(after_shock_wealth))
    end

    return cont
end


function solve_postcorruption(m::PortfolioChoiceModel)

    # Unpack various parameters
    @unpack β, u, r, ybar, logw_grid = m
    nw = length(logw_grid)

    # Allocate memory
    V = zeros(nw)
    V_upd = zeros(nw)
    bstar = zeros(nw)

    dist, iter = 10.0, 0
    while (dist > 1e-3) & (iter < 25000)
        if mod(iter, 25) == 0
            @show iter, dist
        end
        V_itp = extrapolate(
            Interpolations.scale(
                interpolate(V, ITPTYPE), logw_grid
            ), Interpolations.Flat()
        )

        for iw in 1:nw
            wt = exp(logw_grid[iw]) + ybar

            let β=β, u=u, r=r, iw=iw, wt=wt, V_itp=V_itp, bstar=bstar, V_upd=V_upd

                function objective(log_btp1)
                    btp1 = exp(log_btp1)
                    flow = u(wt - btp1)
                    wtp1 = (1 + r) * btp1
                    cont = compute_EV(m, V_itp, wtp1)

                    return -(flow + β*cont)
                end
                sol = optimize(objective, -25.0, log(wt-1e-5))

                bstar[iw] = exp(sol.minimizer)
                V_upd[iw] = -sol.minimum
            end

        end  # End for

        # Make updates between iterations
        dist = maximum(abs, (V .- V_upd) ./ V)
        copyto!(V, V_upd)
        iter += 1
    end  # End while

    return bstar, V
end


function solve_given_price(m::PortfolioChoiceModel, q, V_aut)

    # Unpack various parameters
    @unpack β, u, r, ybar = m
    @unpack ybar, η, π, κ, α, xbar, μ, τ, γ = m
    @unpack logw_grid = m
    nw = length(logw_grid)

    # Create autarky interpolator
    V_aut_itp = extrapolate(
        Interpolations.scale(
            interpolate(V_aut, ITPTYPE), logw_grid
        ), Interpolations.Flat()
    )

    # Allocate memory for truth-teller value functions
    V = copy(V_aut)
    V_upd = copy(V)

    # Allocate memory for policy functions and prices
    bstar = fill(-1.0, nw)
    xstar = fill(-1.0, nw)
    sstar = fill(-1, nw)

    dist, iter = 10.0, 0
    while (dist > 1e-3) & (iter < 25000)
        # Occasionally show progress
        @show iter, dist

        # Create interpolator
        V_itp = extrapolate(
            Interpolations.scale(
                interpolate(V, ITPTYPE), logw_grid
            ), Interpolations.Flat()
        )

        for iw in 1:nw
            wt = exp(logw_grid[iw]) + ybar
            bgrid = range(0, stop=wt - 1e-4, length=100)
            xgrid = range(0, stop=wt/q - 1e-4, length=200)

            # Maximize over choices
            vstar = -1e8
            for btp1 in bgrid, xtp1 in xgrid
                ct = wt - btp1 - q*xtp1

                if ct < 1e-8
                    continue
                end

                # Compute whether oracle is vulnerable -- Assumes everyone else
                # votes truthfully. Oracle corrupted if the attacker holds more
                # than α xbar tokens
                vulnerable = (xtp1 >= α*xbar)

                # If tells truth then may forget to vote
                wtp1_truth_forget = (1 + r)*btp1 + xtp1*q

                # If tells truth and doesn't forget to vote then may make mistake
                wtp1_truth_nomistake = (1 + r)*btp1 + (1 + γ*(π / (1 - π)))*xtp1*(τ*μ + q)
                wtp1_truth_mistake = (1 + r)*btp1 + (1 - γ)*xtp1*(τ*μ + q)
                value_truth = u(ct) + β*(
                    η*compute_EV(m, V_itp, wtp1_truth_forget) +  # Forget
                    (1 - η) * (  # Don't forget
                        π*compute_EV(m, V_itp, wtp1_truth_mistake) +  # Mistake
                        (1 - π)*compute_EV(m, V_itp, wtp1_truth_nomistake)  # No mistake
                    )
                )

                # Liars don't make mistakes or forget
                wtp1_lie = ((1 + r)*btp1 +
                    vulnerable*((1 + γ*(1 - π)/π)*xtp1*(τ*μ + 0.0) + κ*μ) +
                    (1 - vulnerable)*((1 - γ)*xtp1*(τ*μ + q))
                )
                value_lie = u(ct) + β*(
                    vulnerable*compute_EV(m, V_aut_itp, wtp1_lie) +
                    (1-vulnerable)*compute_EV(m, V_itp, wtp1_lie)
                )

                # No lying if hold too small a share
                report_truth = (value_truth >= value_lie) | (xtp1 < 1e-4)
                value = report_truth*value_truth + (1-report_truth)*value_lie

                if value > vstar
                    vstar = value
                    bstar[iw] = btp1
                    xstar[iw] = xtp1
                    sstar[iw] = Int(report_truth)
                end

            end  # End for over btp1/xtp1
            V_upd[iw] = vstar

        end  # End for w

        # Make updates between iterations
        dist = maximum(abs, (V - V_upd) ./ V)
        copyto!(V, V_upd)
        iter += 1
    end  # End while

    return bstar, xstar, sstar, V
end


function token_market_clearing(m::PortfolioChoiceModel, xstar, logwvals, wealth_dist)
    token_supply = m.xbar

    x_itp = extrapolate(
        Interpolations.scale(
            interpolate(xstar, ITPTYPE), m.logw_grid
        ), Interpolations.Flat()
    )

    token_demand = 0.0
    for (iw, w) in enumerate(logwvals)
        token_demand += x_itp(w) * wealth_dist[iw]
    end

    return token_supply - token_demand
end


function compute_wealth_distrbution(m, bstar, xstar, sstar, q)
    @unpack logw_grid, r, γ, π, η, μ, τ, κ, α, xbar = m
    nw = length(logw_grid)

    # Create a more fine wealth grid
    logw_grid_fine = range(logw_grid[1], stop=logw_grid[end], length=2*nw)

    # Create relevant interpolators between states

    b_itp = extrapolate(
        Interpolations.scale(
            interpolate(bstar, ITPTYPE), logw_grid
        ), Interpolations.Flat()
    )

    x_itp = extrapolate(
        Interpolations.scale(
            interpolate(xstar, ITPTYPE), logw_grid
        ), Interpolations.Flat()
    )

    # TODO: Rather than interpolate over s, try
    s_itp = extrapolate(
        Interpolations.scale(
            interpolate(sstar, BSpline(Interpolations.Constant())), logw_grid
        ), Interpolations.Flat()
    )

    Πw = zeros(2*nw, 2*nw)
    for (i_t, logw_t) in enumerate(logw_grid_fine)
        # Lie or tell truth
        report_truth = s_itp(logw_t)

        # Portfolio decisions
        btp1 = b_itp(logw_t)
        xtp1 = x_itp(logw_t)

        if report_truth == 1
            w_tp1_forget = (1 + r)*btp1 + xtp1*q
            w_tp1_mistake = (1 + r)*btp1 + (1 - γ)*xtp1*(τ*μ + q)
            w_tp1_nomistake = (1 + r)*btp1 + (1 + γ*(π/(1-π)))*xtp1*(τ*μ + q)

            w_tp1_vals = [w_tp1_forget, w_tp1_mistake, w_tp1_nomistake]
            probs = [η, (1-η)*π, (1-η)*(1-π)]
        else
            vulnerable = Int(xtp1 >= α*xbar)
            w_tp1 = (
                (1+r)*btp1 +
                vulnerable*((1 + γ*((1-π)/π))*xtp1*τ*μ + κ*μ) +
                (1-vulnerable)*(1 - γ)*xtp1*(τ*μ + q)
            )
            w_tp1_vals = [w_tp1]
            probs = [1.0]
        end

        nnodes, nvals = length(m.nodes), length(w_tp1_vals)
        shocked_wtp1_vals = zeros(nnodes*nvals)
        shocked_probs = zeros(nnodes*nvals)
        counter = 1
        for (_prob, _val) in zip(probs, w_tp1_vals)
            for (_pshock, _shock) in zip(m.weights, m.nodes)
                shocked_wtp1_vals[counter] = _val + _shock
                shocked_probs[counter] = _prob*_pshock
                counter += 1
            end
        end

        for (i, w_tp1) in enumerate(shocked_wtp1_vals)

            logw_tp1 = log(max.(w_tp1, 1e-8))
            i_tp1 = searchsortedfirst(logw_grid_fine, logw_tp1)

            # If value is less than anything on grid then put all mass on lowest
            # wealth in vector
            if i_tp1 <= 1
                Πw[i_t, 1] += shocked_probs[i]
            # If value is larger than anything on grid then put all mass on
            # largest wealth in vector
            elseif i_tp1 >= 2*nw
                Πw[i_t, end] += shocked_probs[i]
            else
                spread = exp(logw_grid_fine[i_tp1]) - exp(logw_grid_fine[i_tp1 - 1])
                weight_to_ub = (exp(logw_tp1) - exp(logw_grid_fine[i_tp1 - 1])) / spread
                weight_to_lb = 1.0 - weight_to_ub
                Πw[i_t, i_tp1 - 1] += weight_to_lb*shocked_probs[i]
                Πw[i_t, i_tp1] += weight_to_ub*shocked_probs[i]
            end
        end
    end

    mc = MarkovChain(Πw)
    sd = stationary_distributions(mc)[1]

    return logw_grid_fine, sd
end


function solve_PCM(m::PortfolioChoiceModel, q0=5.0, q1=250.0)

    # Solve autarkic problem
    b_aut, V_aut = solve_postcorruption(m)

    token_price = brent(q0, q1) do q
        #= General outline:
        - guess prices
        - get policies
        - compute implied dist
        - check market clearing
        - propose new prices
        =#

        # Get policies
        bstar, xstar, sstar, V = solve_given_price(m, q, V_aut)

        # Compute wealth distribution
        logwvals, wealth_dist = compute_wealth_distrbution(m, bstar, xstar, sstar, q)

        # Check market clearing
        residual = token_market_clearing(m, xstar, logwvals, wealth_dist)

        @show q, residual, extrema(bstar), extrema(xstar)
        return residual
    end

    bstar, xstar, sstar, V = solve_given_price(m, token_price, V_aut)

    return V, bstar, xstar, sstar, token_price
end

#=
m = PortfolioChoiceModel(wmax=42.5, nw=425)
# solve(m)

q = 3.5420894234786453

b_aut, V_aut = solve_postcorruption(m)
bstar, xstar, sstar, V = solve_given_price(m, q, V_aut)
logwvals, wealth_dist = compute_wealth_distrbution(m, bstar, xstar, sstar, q)
residual = token_market_clearing(m, xstar, logwvals, wealth_dist)
=#

#=
Ideas for overcoming wealth distribution affecting decisions:

* Impose an ad-hoc rule where attack is successul if individual holds
  certain number of tokens
* Choose a wealth distribution then compute the price that generates the wealth
  distribution and


\sum_i f^w_i x_{i, t+1} = \bar{x}
=#

    # guess = [1.0001; ones(length(logw_grid)-1) ./ length(logw_grid)]
    # log_guess = log.(guess)
    #
    # function market_clearing(log_q)
    #     q = exp(log_q)
    #
    #     # Compute optimal policy by everyone if price is q
    #     bstar, xstar, sstar, V = solve_given_price(m, q, wdist, V_aut)
    #
    #     # Compute the weath distribution
    #     wealth_dist = compute_wealth_distrbution(bstar, xstar, sstar)
    #
    #     # Check market clearing
    #     check_market_clearning = market_clearing_condition(..., q, wdist)
    #
    #     return out
    # end
