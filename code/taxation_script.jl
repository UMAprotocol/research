using ORCA
using PlotlyJS
using QuantEcon
using Random

include("./taxation_problem.jl")


const umared = "rgb(255, 74, 74)"


function create_solution(model)
    # Check whether solution exists
    fname = "./solutions/model_$(UMATaxModel.get_model_name(model)).jld"
    if isfile(fname)
        model, sol = UMATaxModel.load_solution(model)
    else
        sol = UMATaxModel.compute_solution(model)
        UMATaxModel.save_solution(model, sol)
    end

    return model, sol
end


function save_img(model, plot, fname)
    foldername = "../notes/TaxationPlanImages/model_$(UMATaxModel.get_model_name(model))"
    if !ispath(foldername)
        mkpath(foldername)
    end
    savefig(plot, foldername*"/"*fname)

    return 1
end


function plot_margin_process(model; nsims=250, nyears=25)

    # Margin simulations
    Random.seed!(61089)
    mc_sims = [simulate(mc, 12*nyears) for i in 1:nsims]

    # Create a scatter of each margin simulation
    scatters = [
        scatter(;
            x=range(0, nyears, length=12*nyears), y=mc_sims[i],
            opacity=ifelse(mc_sims[i][end] > 1.0, 0.10, 0.35),
            marker=attr(color="black"),
            showlegend=false
        )
        for i in 1:nsims
    ]
    scatters = vcat(scatters, [
        scatter(;
            x=range(0, nyears, length=12*nyears), y=mean(mc_sims),
            market=attr(color="black"), showlegend=false
        )
    ])

    # Create actual plot
    smg_plot = Plot(
        scatters,
        Layout(
            title="Margin Growth Process", xaxis_title="Years from Launch",
            yaxis_title="Millions of Dollars",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
        )
    )

    save_img(model, smg_plot, "StochasticMarginGrowth.png")

    return 1
end


function plot_buyback_percentage(model, sol)

    # Compute annualized buybacks
    Xstar = sol.Xstar
    annualized_buybacks = ((1.0 .+ (Xstar ./ model.M.state_values)).^12 .- 1.0) .* 100

    abb_bar = [
        bar(;
            x=model.M.state_values, y=annualized_buybacks,
            marker=attr(color=umared, showlegend=false)
        )
    ]

    abb_bar_plot = Plot(
        abb_bar,
        Layout(
            title="Annualized Buyback Percents (as % of Margin)",
            xaxis_title="System Margin (millions of dollars)",
            yaxis_range=[0, 45], showlegend=false,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
        )
    );
    save_img(model, abb_bar_plot, "AnnualizedBuybacks.png")

    return 1
end


function plot_heatmap(model, sol)
    # Compute "tax rates"
    Tstar = sol.Tstar
    τstar = (1.0 .+ Tstar ./ model.M.state_values').^12 .- 1.0
    τstar[:, 1] .= 0.0

    τ_heat_plot = Plot(
        heatmap(;z=τstar[1:100, 501:end], x=sol.D[1:100], y=model.M.state_values[501:end]),
        Layout(
            title="Annualized Tax Rates",
            xaxis_title="Rainy Day Fund (Millions of USD)",
            yaxis_title="Margin (Millions of USD)",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
        )
    )
    save_img(model, τ_heat_plot, "AnnualizedTaxRates.png")

    return 1
end


function plot_simulation_outcomes(model, sol; nyears=25)
    # Simulate model and compute tax rates and buyback rates
    D_sim, M_sim, T_sim, X_sim = UMATaxModel.simulate_rainy_day(
        model, sol.D, sol.Dstar, sol.Tstar, sol.Xstar; nyears=nyears
    )
    τ_sim = (1.0 .+ T_sim ./ M_sim).^12 .- 1.0
    bbr_sim = (1.0 .+ X_sim ./ M_sim).^12 .- 1.0

    # Create an x for plots
    t_vals = range(0, nyears; length=12*nyears)

    # Create simulation plots
    M_sim_plot = Plot(
        scatter(;x=t_vals, y=M_sim, marker=attr(color="black"), opacity=0.35, showlegend=false),
        Layout(;yaxis_title="Margin (Millions of USD)", autosize=false, height=300, width=500)
    );
    τ_sim_plot = Plot(
        scatter(;x=t_vals, y=τ_sim, marker=attr(color=umared), showlegend=false),
        Layout(;yaxis_title="Annualized Tax Rates", autosize=false, yaxis_range=[0.0, 0.25], height=300, width=500)
    );
    D_sim_plot = Plot(
        scatter(;x=t_vals, y=D_sim, marker=attr(color="black"), opacity=0.35, showlegend=false),
        Layout(;yaxis_title="Rainy Day Fund (Millions of USD)", autosize=false, height=300, width=500)
    );
    bbr_sim_plot = Plot(
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

    save_img(model, sim_plot, "SimulationPlots.png")

    return 1
end


function make_all_plots(model, sol)

    plot_margin_process(model)
    plot_buyback_percentage(model, sol)
    plot_heatmap(model, sol)
    plot_simulation_outcomes(model, sol)

    return nothing
end


function iterate_on_models(models)
    for model in models
        _model, _sol = create_solution(model)
        make_all_plots(_model, _sol)
    end

    return 1
end


#
# Create all models that we want to run
#
mc = UMATaxModel.create_scurve_mc(;g=0.030, σ0=0.35, σ=0.025, nM=1501)
baseline_model = UMATaxModel.TaxModel(0.5, 2.1e-3, 1.65e-4, mc)

# Vary gamma
models_vary_gamma = [
    UMATaxModel.TaxModel(_γ, 2.1e-3, 1.65e-4, mc) for _γ in [0.75, 0.9]
]

# Vary tau
models_vary_tautarget = [
    UMATaxModel.TaxModel(0.5, 2.1e-3, _τ_target, mc)
    for _τ_target in [1.65e-3, 1.65e-4, 3.84e-4, 1.65e-5]
]

# Vary r
models_vary_r = [
    UMATaxModel.TaxModel(0.5, _r, 1.65e-4, mc)
    for _r in [1e-3, 2.1e-3, 4.1e-3, 8e-3]
]

all_models = vcat([baseline_model], models_vary_gamma, models_vary_tautarget, models_vary_r)


iterate_on_models(all_models)
