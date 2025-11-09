from xenonnt_plot_style import XENONPlotStyle as xps
xps.use('xenonnt')

from functools import partial
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
from utils import k0, beta, a, V
from utils import te_denominator, tm_denominator
from utils import te_right, tm_right


def plot_kf(lam, nf, ns, nc, h, te=True, ylim=[-10, 10]):
    max_kf = beta(k0(lam) * ns, lam, nf)
    if te:
        denominator = te_denominator
        right = te_right
    else:
        denominator = tm_denominator
        right = tm_right
    result = optimize.root_scalar(
        partial(denominator, lam=lam, nf=nf, ns=ns, nc=nc),
        bracket=[1e-6, max_kf],
        method="brentq",
    )
    assert result.converged

    atol = 1e-6

    n = np.ceil(max_kf * h / np.pi).astype(int)
    singularity = np.pi / 2 * (2 * np.arange(n) + 1) / h
    kf = np.linspace(0, max_kf, 10001)
    kf = kf[~np.any(np.abs(kf - singularity[:, None]) < atol, axis=0)]
    singularity = np.sort(np.append(singularity, result.root))
    kf = np.sort(np.append(kf, singularity))
    kf = kf[kf < max_kf]
    is_root = np.isin(kf, singularity)

    fig, ax = plt.subplots(1, 1, figsize=(4, 3))

    ax.plot(kf, np.where(is_root, np.nan, np.tan(h * kf)))
    ax.plot(kf, np.where(is_root, np.nan, right(kf, lam, nf, ns, nc)))
    ax.axhline(0, linestyle='dashed', color=xps.colors['grey'])

    ax.set_xlim(kf[0], kf[-1])
    ax.set_ylim(*ylim)

    ax.set_xlabel('$\kappa$')

    plt.show()

    roots = []
    for i in range(len(singularity) - 1):
        max_kf = beta(k0(lam) * ns, lam, nf)
        result = optimize.root_scalar(
            lambda kf: np.tan(h * kf) - right(kf, lam, nf, ns, nc),
            bracket=[singularity[i] + atol, singularity[i + 1] - atol],
            method="brentq",
        )
        assert result.converged
        roots.append(result.root)
    return roots


def plot_V(lam, nf, ns, nc, h, te=True, numax=3, xlim=[0, 20]):
    b = np.linspace(0, 1, 10001)

    # b = (neff ** 2 - ns ** 2) / (nf ** 2 - ns ** 2)
    
    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    
    V_lam = k0(lam) * h * (nf ** 2 - ns ** 2) ** 0.5
    ax.axvline(V_lam, linestyle='dashed', color=xps.colors['grey'])
    
    atol = 1e-6
    
    roots = []
    a_t = a(nf, ns, nc)
    for nu in range(0, numax + 1):
        ax.plot(V(nu, a_t, b, nf, ns, nc, te=te), b)
        func = lambda b: V(nu, a_t, b, nf, ns, nc, te=te) - V_lam
        if func(atol) * func(1.0 - atol) >= 0:
            continue
        result = optimize.root_scalar(
            func,
            bracket=[atol, 1.0 - atol],
            method='brentq',
        )
        assert result.converged
        roots.append(result.root)

    ax.set_xlim(*xlim)
    ax.set_ylim(0, 1)
    
    ax.set_xlabel('$V$')
    ax.set_ylabel('$b$')
    
    plt.show()
    return roots
