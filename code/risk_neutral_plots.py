import numpy as np
import matplotlib.pyplot as plt

from collections import namedtuple
from numba import jit


PortfolioChoiceModel = namedtuple(
    "PortfolioChoiceModel",
    ["r", "mu", "tau", "alpha", "eta", "pi", "gamma", "n"]
)


@jit(nopython=True)
def expected_payment_token(pcm, npers, nsims):
    r, mu, tau = pcm.r, pcm.mu, pcm.tau
    eta, pi, gamma = pcm.eta, pcm.pi, pcm.gamma
    n = pcm.n
    dividend = mu*tau

    eps = np.zeros(npers)
    for n in range(nsims):
        curr_x = 1
        for t in range(npers):
            rv_eta = np.random.rand()
            if rv_eta >= eta:
                rv_pi = np.random.rand()
                if rv_pi < pi:
                    curr_x = (1 - gamma)*curr_x
                else:
                    curr_x = (1 + gamma*(pi / (1-pi)))*curr_x

                # Increment current periods counter
                eps[t] = eps[t] + ((1 / (1+r))**t * curr_x*dividend / nsims)

    return eps

# pcm_25bp = PortfolioChoiceModel(0.00075, 1000000, 0.0025, 0.50, 0.05, 0.01, 0.025, 1)
# pcm_20bp = PortfolioChoiceModel(0.00075, 1000000, 0.002, 0.50, 0.05, 0.01, 0.025, 1)
pcm_15bp = PortfolioChoiceModel(0.00075, 1000000, 0.0015, 0.50, 0.05, 0.01, 0.025, 1)
pcm_10bp = PortfolioChoiceModel(0.00075, 1000000, 0.001, 0.50, 0.05, 0.01, 0.025, 1)
pcm_5bp = PortfolioChoiceModel(0.00075, 1000000, 0.0005, 0.50, 0.05, 0.01, 0.025, 1)
pcm_1bp = PortfolioChoiceModel(0.00075, 1000000, 0.0001, 0.50, 0.05, 0.01, 0.025, 1)


eps_25bp = expected_payment_token(pcm_25bp, 150*52, 5_000)
eps_20bp = expected_payment_token(pcm_20bp, 150*52, 5_000)
eps_15bp = expected_payment_token(pcm_15bp, 150*52, 5_000)
eps_10bp = expected_payment_token(pcm_10bp, 150*52, 5_000)
eps_5bp = expected_payment_token(pcm_5bp, 150*52, 5_000)
eps_1bp = expected_payment_token(pcm_1bp, 150*52, 5_000)

fig, ax = plt.subplots()

ax.plot(
    np.arange(150*52) / 52, np.cumsum(eps_15bp), color=(255/255, 74/255, 74/255)
)
ax.annotate(r"$\tau =$15 bp", (25, 1275000), color=(255/255, 74/255, 74/255))

ax.plot(
    np.arange(150*52) / 52, np.cumsum(eps_10bp), color=(255/255, 74/255, 74/255)
)
ax.annotate(r"$\tau =$10 bp", (45, 935000), color=(255/255, 74/255, 74/255))
ax.plot(
    np.arange(150*52) / 52, np.cumsum(eps_5bp), color=(255/255, 74/255, 74/255)
)
ax.annotate(r"$\tau =$5 bp", (65, 567000), color=(255/255, 74/255, 74/255))
ax.plot(
    np.arange(150*52) / 52, np.cumsum(eps_1bp), color=(255/255, 74/255, 74/255)
)
ax.annotate(r"$\tau =$1 bp", (40, 105000), color=(255/255, 74/255, 74/255))

ax.hlines(500000, 0, 140, color="black", linestyle="--")
ax.annotate("Cost of Corruption", (110, 375000), color="black")

ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.set_xlabel("Years of Dividends")
ax.set_title("Net Present Value of Dividends Over Time")

fig.savefig("../slides/VaryTau.pgf", transparent=True)

epss = [eps_25bp, eps_20bp, eps_15bp, eps_10bp, eps_5bp, eps_1bp]
pcms = [pcm_25bp, pcm_20bp, pcm_15bp, pcm_10bp, pcm_5bp, pcm_1bp]

for (_pcm, _eps) in zip(pcms, epss):
    print("For tau = {}".format(_pcm.tau))
    print("\tYearly rate is {}".format((1 + _pcm.tau)**52))
    weeks = np.searchsorted(np.cumsum(_eps), _pcm.mu/2)
    if weeks == len(_eps):
        weeks = np.inf

    print("\tYears needed is {}".format(weeks/52))


# def foo():
#     expected_payment = (
#         eta * q + (1 - eta)*(
#             pi*(1-gamma) + (1-pi)*(1 + gamma*(pi/(1-pi))) * (mu*tau/n + q)
#         )
#     )
#
#     return expected_payment
