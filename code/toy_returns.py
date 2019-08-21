import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from collections import namedtuple
from numba import jit


params = [
    "r_annual", "pi_annual", "g_annual", "S0", "M0", "Mbar", "PfC_div_M",
    "gamma", "eta"
]

BoEModel = namedtuple("BoEModel", params)


@jit(nopython=True)
def oracle_corruption_token_price(m, S, M):
    return (2*m.gamma)/(1 - m.eta) * (m.PfC_div_M*M)/S


@jit(nopython=True)
def required_tax_rate_margin_ss(m, votes_per_day):

    # Convert to rates per vote from annual
    r = (1.0 + m.r_annual)**(1 / (365 * votes_per_day)) - 1.0

    return (2*r*m.gamma)/((1 - m.eta)*(1 + r)) * m.PfC_div_M


@jit(nopython=True)
def required_annualized_tax_rate_margin_ss(m, votes_per_day):

    tau = required_tax_rate_margin_ss(m, votes_per_day)
    tau_annual = (1.0 + tau)**(365*votes_per_day) - 1.0

    return tau_annual


@jit(nopython=True)
def simulate_margin_growth(m, nyears=25, votes_per_day=3):

    # Unpack relevant information
    M0, Mbar = m.M0, m.Mbar
    g = m.g_annual / (365 * votes_per_day)

    # Explicit solution to logistic growth differential equation -- Almost same answer as simulating
    # period-by-period
    out = M0*Mbar / (M0 + (Mbar - M0)*np.exp(-g*np.arange(nyears*votes_per_day*365)))

    return out


@jit(nopython=True)
def simulate_buyback_payments(M_sim, vote_tax_rate, cutoff_M, votes_per_day=3):
    # Compute number of years margin is simulated for
    nvotes = M_sim.shape[0]
    nyears = int(nvotes / (votes_per_day*365))

    payments = np.zeros(nvotes)
    for t in range(nvotes):

        # Determine whether tax is imposed today based on margin
        _tau = vote_tax_rate if M_sim[t] > cutoff_M else 0.0
        payments[t] = _tau * M_sim[t]

    return payments


@jit(nopython=True)
def simulate_system(m, nyears, votes_per_day, tax_rate, cutoff_M):

    # Unpack relevant information and convert things to vote period units
    nvotes = nyears*365*votes_per_day
    pi = (1.0 + m.pi_annual)**(1 / (365 * votes_per_day)) - 1.0
    r = (1.0 + m.r_annual)**(1 / (365 * votes_per_day)) - 1.0
    vote_tax_rate = (1.0 + tax_rate)**(1 / (365 * votes_per_day)) - 1.0

    # Can simulate margin and buy backs without anything else...
    M_sim = simulate_margin_growth(m, nyears, votes_per_day)
    x_sim = simulate_buyback_payments(M_sim, vote_tax_rate, cutoff_M, votes_per_day)
    payment_discounter = (1 / (1 + r))**np.arange(nvotes)

    # Compute what the pdv of payments are after reaching margin upper bound and compute when
    # the margin is within 1% of steady state
    ss_pdv_payments = (1 + r)/r * vote_tax_rate * m.Mbar
    t_reach_ss = np.searchsorted((m.Mbar - M_sim) / m.Mbar < 1e-2, True)

    # Allocate for token and price simulation
    p_sim = np.zeros(nvotes)
    p_sim[0] = (
        np.sum(payment_discounter[:t_reach_ss]*x_sim[:t_reach_ss]) +
        (1.0 / (1.0 + r))**t_reach_ss * ss_pdv_payments
    )
    S_sim = np.zeros(nvotes)
    S_sim[0] = m.S0

    for t in range(1, nvotes):

        # Once within 1% of ss, just assume in ss
        if t < t_reach_ss:
            pdv_x = (
                np.sum((1.0 + r)**t * payment_discounter[t:t_reach_ss]*x_sim[t:t_reach_ss]) +
                (1.0 / (1.0 + r))**(t_reach_ss - t) * ss_pdv_payments
            )
        else:
            pdv_x = ss_pdv_payments

        buyback_tm1 = x_sim[t-1] / p_sim[t-1]

        S_sim[t] = (1 + pi)*(S_sim[t-1] - buyback_tm1)
        p_sim[t] = pdv_x / S_sim[t]

    return M_sim, x_sim, p_sim, S_sim


@jit(nopython=True)
def compute_dynamic_tax_rates_backwards(m, nyears, votes_per_day):

    # Unpack relevant information and convert things to vote period units
    nvotes = nyears*365*votes_per_day
    pi = (1.0 + m.pi_annual)**(1 / (365 * votes_per_day)) - 1.0
    r = (1.0 + m.r_annual)**(1 / (365 * votes_per_day)) - 1.0
    vote_tax_rate = (1.0 + tax_rate)**(1 / (365 * votes_per_day)) - 1.0

    # Can simulate margin and buy backs without anything else...
    M_sim = simulate_margin_growth(m, nyears, votes_per_day)
    PfC_sim = M_sim * m.PfC_div_M
    payment_discounter = (1 / (1 + r))**np.arange(nvotes)

    # Compute what the steady state tax rate is once the system has reached its maximum level of
    # margin
    τ_ss = required_tax_rate_margin_ss(m, votes_per_day)
    t_reach_ss = np.searchsorted((m.Mbar - M_sim) / m.Mbar < 1e-3, True)

    # Allocate for token and price simulation
    τ_sim = np.zeros(nvotes)

    for t in range(nvotes):

        # Once within 0.1% of ss, just assume in ss
        if t < t_reach_ss:
            τ_sim[t] = 2*(PfC_sim[t] - PfC_sim[t+1]/(1 + r)) / M_sim[t]
        else:
            τ_sim[t] = τ_ss

    return M_sim, τ_sim


#            r     pi    g     S0   M0   Mbar    PfC/M gamma eta
m = BoEModel(0.08, 0.10, 0.50, 1.0, 1.0, 1000.0, 0.50, 1.00, 0.0)

nyears = 30
votes_per_day = 6
nvotes = nyears*365*votes_per_day
tax_rate = required_annualized_tax_rate_margin_ss(m, votes_per_day)
cutoff = 940.0

M_sim, x_sim, p_sim, S_sim = simulate_system(m, nyears, votes_per_day, tax_rate, cutoff)
p_corrupt_sim = oracle_corruption_token_price(m, S_sim, M_sim)

umared = (255/255, 74/255, 74/255)

M_sim_700, _, p_sim_700, S_sim_700 = simulate_system(m, nyears, votes_per_day, tax_rate, 700)
M_sim_800, _, p_sim_800, S_sim_800 = simulate_system(m, nyears, votes_per_day, tax_rate, 800)
M_sim_900, _, p_sim_900, S_sim_900 = simulate_system(m, nyears, votes_per_day, tax_rate, 900)
M_sim_990, _, p_sim_990, S_sim_990 = simulate_system(m, nyears, votes_per_day, tax_rate, 990)

p_corrupt_sim_700 = oracle_corruption_token_price(m, S_sim_700, M_sim_700)
p_corrupt_sim_800 = oracle_corruption_token_price(m, S_sim_800, M_sim_800)
p_corrupt_sim_900 = oracle_corruption_token_price(m, S_sim_900, M_sim_900)
p_corrupt_sim_990 = oracle_corruption_token_price(m, S_sim_990, M_sim_990)


def plot_margin_growth(m, nyears=40, votes_per_day=6):

    # Simulate margin growth
    t = np.linspace(0, nyears, nyears*votes_per_day*365)
    M_sim = simulate_margin_growth(m, nyears, votes_per_day)

    fig, ax = plt.subplots()

    ax.plot(t, M_sim, color=umared, linewidth=1.5)
    ax.set_xlabel("Years")
    ax.set_ylabel("Millions of Dollars")
    ax.set_title("Total Margin in Oracle System")

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    fig.savefig("MarginGrowth.png", dpi=300)

    return fig


def create_market_cap_figure(m, nyears=30, votes_per_day=6, cutoff=900):

    M_sim, x_sim, p_sim, S_sim = simulate_system(m, nyears, votes_per_day, tax_rate, cutoff)
    p_corrupt_sim = oracle_corruption_token_price(m, S_sim, M_sim)

    fig, ax = plt.subplots(figsize=(14, 8))

    t_array = np.linspace(0, nyears, nvotes)

    # Determine when tax goes into place
    t_taxation_begins = np.searchsorted(M_sim, cutoff)
    ax.plot(
        t_array, (p_sim*S_sim),
        color=umared, linewidth=1.5
    )
    ax.annotate("Market cap", (8.5, 650), color=umared)

    ax.plot(t_array, p_corrupt_sim*S_sim, color="k", linewidth=1.5, linestyle="--")
    ax.annotate("Market cap required to prevent corruption", (18.5, 850), color="k")

    ax.set_title("System Market Cap (millions USD)")
    ax.set_xlabel("Years Since Launch")

    ax.yaxis.set_ticks([])
    ax.yaxis.set_ticklabels([])

    x_ticks = [0, 10, t_taxation_begins / (365*votes_per_day), 20, 30]
    ax.xaxis.set_ticks(x_ticks)
    ax.xaxis.set_ticklabels(["0", "10", "", "20", "30"])
    ax.annotate(
        "Taxation begins",
        xy=(t_taxation_begins / (365*votes_per_day), 0.0),
        xytext=(t_taxation_begins / (365*votes_per_day) + 2, 50.0),
        arrowprops={"arrowstyle": "->"}
    )
    ax.vlines(
        t_taxation_begins / (365*votes_per_day), 0.0, (S_sim*p_sim)[int(t_taxation_begins)],
        linewidth=0.5, linestyle="--"
    )

    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    fig.savefig("MarketCap.png", dpi=300)

    return fig


def plot_dynamic_deterministic_tax_rates(m, nyears=40, votes_per_day=6):

    # Get margin and tax rates
    t_array = np.linspace(0, nyears, nyears*votes_per_day*365)
    M_sim, τ_sim = compute_dynamic_tax_rates_backwards(m, nyears, votes_per_day)
    τ_sim_annualized = 100 * ((τ_sim + 1)**(votes_per_day*365) - 1.0)

    fig, ax = plt.subplots()

    ax.plot(t_array, τ_sim_annualized, color="k", linestyle="--")
    ax.plot(t_array, np.maximum(0.0, τ_sim_annualized), color=umared, linewidth=1.5)

    ax.set_xlabel("Years")
    ax.set_ylabel("Annualized Tax Rate")
    ax.set_title("Minimum Sustainable Dynamic Tax Rates on Margin")
    ax.set_ylim(-np.max(np.abs(τ_sim_annualized)), np.max(np.abs(τ_sim_annualized)))

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    fig.savefig("TaxRates.png", dpi=300)

    return None


plot_margin_growth(m)
create_market_cap_figure(m)
plot_dynamic_deterministic_tax_rates(m)
