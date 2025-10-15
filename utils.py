import numpy as np


def k0(lam):
    return 2 * np.pi / lam


def lam(k0):
    return 2 * np.pi / k0


def beta(kf, lam, nf):
    return ((k0(lam) * nf) ** 2 - kf ** 2) ** 0.5


def gamma(kf, lam, nf, n):
    return (beta(kf, lam, nf) ** 2 - (k0(lam) * n) ** 2) ** 0.5


def te_denominator(kf, lam, nf, ns, nc):
    gamma_s = gamma(kf, lam, nf, ns)
    gamma_c = gamma(kf, lam, nf, nc)
    return kf * (1 - gamma_s * gamma_c / kf ** 2)


def tm_denominator(kf, lam, nf, ns, nc):
    gamma_s = gamma(kf, lam, nf, ns)
    gamma_c = gamma(kf, lam, nf, nc)
    return kf ** 2 - nf ** 4 / ns ** 2 / nc ** 2 * gamma_s * gamma_c


def te_right(kf, lam, nf, ns, nc):
    gamma_s = gamma(kf, lam, nf, ns)
    gamma_c = gamma(kf, lam, nf, nc)
    r = gamma_s + gamma_c
    r /= te_denominator(kf, lam, nf, ns, nc)
    return r


def tm_right(kf, lam, nf, ns, nc):
    gamma_s = gamma(kf, lam, nf, ns)
    gamma_c = gamma(kf, lam, nf, nc)
    r = kf * ((nf ** 2 / ns ** 2) * gamma_s + (nf ** 2 / nc ** 2) * gamma_c)
    r /= tm_denominator(kf, lam, nf, ns, nc)
    return r


def te_amplitude(x, kf, lam, nf, ns, nc, h):
    gamma_s = gamma(kf, lam, nf, ns)
    gamma_c = gamma(kf, lam, nf, nc)
    E_c = np.exp(-gamma_c * x)
    E_f = np.cos(kf * x) - gamma_c / kf * np.sin(kf * x)
    E_s = (np.cos(kf * h) + gamma_c / kf * np.sin(kf * h)) * np.exp(gamma_s * (x + h))
    E = np.where(x > 0, E_c, E_f)
    E = np.where(x > -h, E, E_s)
    return E


# https://www.overleaf.com/project/68e115b428ec5e0666550028
def tm_amplitude(x, kf, lam, nf, ns, nc, h):
    gamma_s = gamma(kf, lam, nf, ns)
    gamma_c = gamma(kf, lam, nf, nc)
    E_c = np.exp(-gamma_c * x)
    E_f = np.cos(kf * x) - nf**2 / nc**2 * gamma_c / kf * np.sin(kf * x)
    E_s = (np.cos(kf * h) + nf**2 / nc**2 * gamma_c / kf * np.sin(kf * h)) * np.exp(gamma_s * (x + h))
    E = np.where(x > 0, E_c, E_f)
    E = np.where(x > -h, E, E_s)
    return E


def te_phi(theta, n1, n2):
    return np.arctan(-(n1 ** 2 * np.sin(theta) ** 2 - n2 ** 2) ** 0.5 / (n1 * np.cos(theta)))


def tm_phi(theta, n1, n2):
    return np.arctan(-n1 ** 2 / n2 ** 2 * (n1 ** 2 * np.sin(theta) ** 2 - n2 ** 2) ** 0.5 / (n1 * np.cos(theta)))
