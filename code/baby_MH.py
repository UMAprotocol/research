import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt


pi_h = 10
pi_l = 0

p_hh = 0.66
p_lh = 0.34
p_hl = 0.34
p_ll = 0.66


g_h = 5/3
g_l = 4/3

ubar = 0.0

wbar = (g_h + ubar)**2


def u(w):
    return math.sqrt(w)


def du(w):
    return 1 / math.sqrt(w)


def obj(ws):
    w_h, w_l = ws
    return -(p_hh*(pi_h - w_h) + p_lh*(pi_l - w_l))


def participation_constraint(ws):
    w_h, w_l = ws
    return u(w_h)*p_hh + u(w_l)*p_lh - g_h - ubar


def dparticipation_constraint(ws):
    w_h, w_l = ws
    der = np.array([
        du(w_h)*p_hh,
        du(w_l)*p_lh
    ])
    return der


def incentive_constraint(ws):
    w_h, w_l = ws
    lhs = u(w_h)*p_hh + u(w_l)*p_lh - g_h
    rhs = u(w_h)*p_hl + u(w_l)*p_ll - g_l
    return lhs - rhs


def dincentive_constraint(ws):
    w_h, w_l = ws
    der = np.array([
        du(w_h)*(p_hh - p_hl),
        du(w_l)*(p_lh - p_ll)
    ])
    return der


eq_constraint = {
    "type": "ineq",
    "fun": lambda x: participation_constraint(x),
    "jac": lambda x: dparticipation_constraint(x)
}

ineq_constraint = {
    "type": "ineq",
    "fun": lambda x: incentive_constraint(x),
    "jac": lambda x: dincentive_constraint(x)
}


opt.minimize(obj, np.array([wbar+2.5, wbar-1.5]), method="SLSQP",
             constraints=[eq_constraint, ineq_constraint])
