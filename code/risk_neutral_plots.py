import numpy as np
import matplotlib.pyplot as plt

from collections import namedtuple


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

pcm_10bp = PortfolioChoiceModel(0.00075, 1000000, 0.001, 0.50, 0.25, 0.01, 0.025, 1)
pcm_5bp = PortfolioChoiceModel(0.00075, 1000000, 0.00055, 0.50, 0.25, 0.01, 0.025, 1)
pcm_1bp = PortfolioChoiceModel(0.00075, 1000000, 0.0001, 0.50, 0.25, 0.01, 0.025, 1)


eps_10bp = expected_payment_token(pcm_10bp, 150*52, 5_000)
eps_5bp = expected_payment_token(pcm_5bp, 150*52, 5_000)
eps_1bp = expected_payment_token(pcm_1bp, 150*52, 5_000)

fig, ax = plt.subplots()

ax.plot(
    np.arange(150*52) / 52, np.cumsum(eps_10bp), color=(255/255, 74/255, 74/255)
)
ax.annotate(r"$\tau =$10 bp", (45, 935000), color=(255/255, 74/255, 74/255))
ax.plot(
    np.arange(150*52) / 52, np.cumsum(eps_5bp), color=(255/255, 74/255, 74/255)
)
ax.annotate(r"$\tau =$5.5 bp", (65, 567000), color=(255/255, 74/255, 74/255))
ax.plot(
    np.arange(150*52) / 52, np.cumsum(eps_1bp), color=(255/255, 74/255, 74/255)
)
ax.annotate(r"$\tau =$1 bp", (40, 105000), color=(255/255, 74/255, 74/255))

ax.hlines(500000, 0, 140, color="black", linestyle="--")

ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.set_xlabel("Years of Dividends")
ax.set_title("Net Present Value of Dividends Over Time")

fig.savefig("../slides/VaryTau.pgf", transparent=True)



# def foo():
#     expected_payment = (
#         eta * q + (1 - eta)*(
#             pi*(1-gamma) + (1-pi)*(1 + gamma*(pi/(1-pi))) * (mu*tau/n + q)
#         )
#     )
#
#     return expected_payment
