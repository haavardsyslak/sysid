import numpy as np
import matplotlib.pyplot as plt
import scipy
import utils.attitude as attitude
from scipy.spatial.transform import Rotation
from utils.data import load_data
import plotting
from filterpy.kalman import unscented_transform

import models


class SigmaPoints:
    def __init__(self, n, alpha, beta, kappa):
        self.n = n
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.Wm, self.Wc = self._compute_weights()

    def get_sigma_points(self, x, P):
        n = self.n
        lamb = self.alpha**2 * (n + self.kappa) - n

        sigmas = np.zeros((2 * n + 1, n))

        sigmas[0] = x
        np.linalg.cholesky(P)
        a = (n + lamb * P)
        cho = np.linalg.cholesky((n + lamb) * P)
        for k in range(n):
            sigmas[k + 1] = x + cho[k]
            sigmas[k + n + 1] = x - cho[k]
        return sigmas

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
    n = len(x) - 1
    res = np.zeros((2 * n + 1, n+1))
    for i, w in enumerate(W):
        res[i, :6] = w[:6] + x[:6]
        rot1 = Rotation.from_rotvec(w[6:9])
        rot2 = Rotation.from_quat(x[6:10], scalar_first=True)
        res[i, 6:10] = (rot2 * rot1).as_quat(scalar_first=True)
        res[i, 6:10] /= np.linalg.norm(res[i, 6:10])

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

    def compute_process_sigmas(self):
        W = self.points.get_sigma_points(self.x_err, self.P)

        return W

    def ukf_cylce(self, u, z):
        n = self.points.n
        W = self.compute_process_sigmas()

        X = WtoX(self.x, W)
        Y = np.zeros((n * 2 + 1, self._dim_x + 1))
        Z = np.zeros((2*n + 1, self._dim_z))
        for i, x in enumerate(X):
            y = self.fx(x, self.dt, u)
            y[6:10]  # /= np.linalg.norm(y[6:10])
            Y[i, :] = y
            a = self.hx(y)
            Z[i, :] = a

        self.x[:6] = np.dot(self.points.Wm, Y[:, :6])
        self.x[6:10] = attitude.mean_quat(Y[:, 6:10])

        W_prime = np.zeros((self.points.n * 2 + 1, self._dim_x))
        rot_bar = Rotation.from_quat(self.x[6:10], scalar_first=True)
        for i, y in enumerate(Y):
            W_prime[i, :6] = y[:6] - self.x[:6]
            rot = Rotation.from_quat(y[6:10], scalar_first=True)
            W_prime[i, 6:9] = (rot * rot_bar.inv()).as_rotvec()

        Wc_diag = np.diag(self.points.Wc)
        self.P = np.dot(W_prime.T, np.dot(Wc_diag, W_prime))
        # x_err, self.P = unscented_transform(W_prime, self.points.Wm, self.points.Wc)



        z_pred = np.dot(self.points.Wm, Z)
        y = z - z_pred
        print(y)
        # innov = np.atleast_2d(Z) - z_pred[np.newaxis, :]
        innov = z_pred[np.newaxis, :] - np.atleast_2d(Z)
        S = np.dot(innov.T, np.dot(Wc_diag, innov)) + self.R

        # Cross covariance
        Pxz = np.zeros((self._dim_x, self._dim_z))
        for i in range(2*n + 1):
            dz = Z[i] - z_pred
            Pxz += self.points.Wc[i] * np.outer(W_prime[i], dz)
            # Pxz += W_prime[i, :] @ y.T
        # Pxz /= (2*n + 1)

        K = Pxz @ np.linalg.inv(S)

        a = K @ y
        self.x[:6] += a[:6] 

        rot = Rotation.from_quat(
            self.x[6:10], scalar_first=True) * Rotation.from_rotvec(a[6:9])
        self.x[6:10] = rot.as_quat(scalar_first=True)
        self.x[6:10] /= np.linalg.norm(self.x[6:10])

        self.P -= K @ S @ K.T
        self.P += self.Q


def is_positive_def(M):
    try:
        # Attempt Cholesky decomposition
        np.linalg.cholesky(M)
        return True
    except np.linalg.LinAlgError:
        return False


def fx(x, dt, u):
    dx = model.fx(x, dt, u)
    res = x + dt * dx
    res[6:10] /= np.linalg.norm(res[6:10])
    # w = x[3:6]
    # norm = np.linalg.norm(w)
    # rot_ang = (dt * norm) / 2
    #
    # q_prev = Rotation.from_quat(x[6:10], scalar_first=True)
    # dq = Rotation.from_quat(np.hstack([np.cos(rot_ang),
    #                 w * np.sin(rot_ang)
    #                 ]), scalar_first=True)
    # res[6:10] = (q_prev * dq).as_quat(scalar_first=True)

    return res


def hx(x):
    return x[:6]

if __name__ == "__main__":

    filename = "data/BlueRov2Heavy_mau_11-05-2024_11-10-46.h5"
    # filename = "data/BlueRov2Heavy_no_rollover_in_attitude_11-12-2024_16-45-23.h5"
    # filename = "data/BlueRov2Heavy_test_11-16-2024_18-09-39.h5"
    # filename = "data/BlueRov2Heavy_no_input_11-16-2024_18-46-55.h5"
    # filename = "data/BlueRov2Heavy_yaw_only_11-16-2024_18-55-19.h5"
    data = load_data(filename)
    state = data["x"]
    inputs = data["u"]
    t_vec = data["t"]
    model = models.BlueRov2Heavy()
    points = SigmaPoints(9, alpha=1e-5, beta=2, kappa=0)
    plotting.plot_state(t_vec, state)
    plt.show()

    measurements = state.copy()
    measurements += np.random.normal(0, 0.05, state.shape)

    Q = np.eye(9) * 1e-5
    R = np.eye(6) * 1e-3
    P = np.eye(9) * 1e-4
    kf = ESUKF(9, 6, fx=fx, hx=hx, Q=Q,
               R=Q, points=points)
    x0 = np.zeros(10)
    x0[6:10] = attitude.euler_to_quat([0, 0, 0])
    a = Rotation.from_quat(x0[6:10]).as_rotvec()
    b = Rotation.from_rotvec(a).as_quat()

    kf.x = x0
    kf.P = P
    kf.Q = Q
    kf.R = R

    x_bar = np.zeros((len(t_vec), 10))
    for i, (x, u, z) in enumerate(zip(state, inputs, measurements)):
        print(i)
        # kf.predict(u)
        kf.ukf_cylce(u, z[:6])
        x_bar[i, :]= kf.x.copy()
        # x_bar[i, 6:10] = x_bar[i, 6:10] / np.linalg.norm(x_bar[i, 6:10])

    plotting.plot_state_est(x_bar[:, :10], state, measurements, t_vec)
    plt.show()
