import numpy as np
import matplotlib.pyplot as plt

from collections import namedtuple


PortfolioChoiceModel = namedtuple(
    "PortfolioChoiceModel",
    ["r", "mu", "tau", "eta", "pi", "gamma", "n"]
)

def expected_payment_token(pcm, q):
    r, mu, tau = pcm.r, pcm.mu, pcm.tau
    eta, pi, gamma = pcm.eta, pcm.pi, pcm.gamma
    n = pcm.n

    expected_payment = (
        eta * q + (1 - eta)*(
            pi*(1-gamma) + (1-pi)*(1 + gamma*(pi/(1-pi))) * (mu*tau/n + q)
        )
    )

    return expected_payment
