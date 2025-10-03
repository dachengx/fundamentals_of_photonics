import numpy as np


def k0(lam):
    return 2 * np.pi / lam


def lam(k0):
    return 2 * np.pi / k0


def beta(kf, lam, nf):
    return ((k0(lam) * nf) ** 2 - kf ** 2) ** 0.5


def gamma(kf, lam, nf, n):
    return (beta(kf, lam, nf) ** 2 - (k0(lam) * n) ** 2) ** 0.5


def denominator(kf, lam, nf, ns, nc):
    gamma_s = gamma(kf, lam, nf, ns)
    gamma_c = gamma(kf, lam, nf, nc)
    return kf ** 2 - nf ** 4 / ns ** 2 / nc ** 2 * gamma_s * gamma_c


def right(kf, lam, nf, ns, nc):
    gamma_s = gamma(kf, lam, nf, ns)
    gamma_c = gamma(kf, lam, nf, nc)
    r = kf * ((nf ** 2 / ns ** 2) * gamma_s + (nf ** 2 / nc ** 2) * gamma_c)
    r /= denominator(kf, lam, nf, ns, nc)
    return r
