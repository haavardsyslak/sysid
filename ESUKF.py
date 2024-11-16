import numpy as np
import scipy
import utils.attitude as attitude

from filterpy.kalman import MerweScaledSigmaPoints


class SigmaPoints:
    def __init__(self, n, alpha, beta, kappa):
        self.n = n
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa

    def get_sigma_points(self, x, P):
        n = self.n
        lamb = self.alpha**2 * (n + self.kappa) - n

        sigmas = np.zeros((2 * n + 1, n))

        sigmas[0] = x
        cho = scipy.linalg.cholesky((n + lamb) * P)
        for k in range(n):
            sigmas[k + 1] = x + cho[k]
            sigmas[k + n + 1] = x - cho[k]
        self.Wm, self.Wc = self._compute_weights()

    def num_sigmas(self):
        return self.n * 2 + 1

    def _compute_weights(self):
        n = self.n
        lamb = self.alpha**2 * (n + self.kappa) - n
        c = 1 / (2 * (n + lamb))
        Wc = np.full(2 * n + 1, c)
        Wm = np.full(2 * n + 1, c)
        Wm[0] = lamb / (n + lamb)
        Wc[0] = (lamb / (n + lamb)) + (1 - self.alpha**2 + self.beta)

        return Wm, Wc


def WtoX(x, W):
    n = len(x)
    res = np.zeros(2 * n + 1, n)
    for i, w in enumerate(W):
        res[:6, i] = w[:6] + x[:6]
        qw = attitude.vec_to_quat(w[6:9])
        res[6:10, i] = attitude.quaternion_prod(x[6:10], qw)

    return res


class ESUKF:
    def __init__(self, dim_x, dim_z, fx, hx, Q, R, points, dt=0.05):
        self.x_err = np.zeros(dim_x)
        self.x = np.zeros(dim_x + 1)
        self.P = np.zeros((dim_x, dim_x))  # Covariance
        self._dim_x = dim_x
        self._dim_z = dim_z

        self.hx = hx
        self.fx = fx

        self.dt = dt

        self.Q = Q
        self.R = R

        self.points = points

    def ukf_cylce(self, u, z):
        W = WtoX(self.x, W)
