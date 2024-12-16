import numpy as np
from filterpy.kalman import unscented_transform, MerweScaledSigmaPoints
from utils.attitude import quaternion_error, quaternion_prod, euler_to_quat, Tq
import models
from utils.data import load_data
import plotting
from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy


class USQUE:
    def __init__(self, dim_x, dim_z, fx_err, fx_nom, hx, Q, R, points,
                 dt=0.05, residual_x=None, residual_z=None):

        self.x_err = np.zeros(dim_x)
        self.x = np.zeros(dim_x + 1)
        self.P = np.zeros((dim_x, dim_x))
        self._dim_x = dim_x
        self._dim_z = dim_z

        self.hx = hx
        self.fx_nom = fx_nom
        self.fx_err = fx_err

        self.dt = dt
        self.Q = Q
        self.R = R
        self.points = points

        if residual_x is None:
            self.residual_x = np.subtract
        if residual_z is None:
            self.residual_z = np.subtract

        self._num_sigmas = points.num_sigmas()
        self.sigmas_f_nom = np.zeros((self._num_sigmas, self._dim_x + 1))
        self.sigmas_f = np.zeros((self._num_sigmas, self._dim_x))
        self.sigmas_h = np.zeros((self._num_sigmas, self._dim_x))

        self.K = np.zeros((dim_x, dim_z))
        self.y = np.zeros(dim_x)
        self.S = np.zeros((dim_z, dim_z))

    def compute_process_sigmas(self):
        sigmas = self.points.sigma_points(self.x_err, self.P)
        for i, s in enumerate(sigmas):
            self.sigmas_f[i] = s

    def update(self, z, u):
        self.compute_process_sigmas()
        # The propagated sigmas
        self.x = self.fx_nom(self.x, self.dt, u)
        sigmas_x = np.zeros((self._num_sigmas, self._dim_x + 1))
        for i, s in enumerate(self.sigmas_f):
            sigmas_x[i, :6] = self.x[:6] + s[:6]
            delta_q = (1 / np.sqrt(4 + s[6:9].T @ s[6:9])) * \
                np.array([2, s[6], s[7], s[8]])
            # delta_q /= np.linalg.norm(delta_q)
            sigmas_x[i, 6:10] = quaternion_prod(self.x[6:], delta_q)

        x_err_prop = np.zeros_like(self.sigmas_f_nom)
        x_err_prop[0, :6] = self.sigmas_f[0, :6]
        x_err_prop[0, 6:10] = np.array([1, 0, 0, 0])
        self.sigmas_f[0, :6] = x_err_prop[0, :6]
        self.sigmas_f[0, 6:9] = 2 * (x_err_prop[i, 7:10] / x_err_prop[0, 6])
        # Find the propagated sigma points, and propagated error sigma sigma_points
        self.sigmas_f_nom[0] = self.fx_nom(sigmas_x[0], self.dt, u)
        for i, s in enumerate(sigmas_x[1:]):
            self.sigmas_f_nom[i] = self.fx_nom(s, self.dt, u)
            x_err_prop[i, :6] = self.sigmas_f_nom[i, :6] - \
                self.sigmas_f_nom[0, :6]
            x_err_prop[i, 6:10] = quaternion_error(self.sigmas_f_nom[i, 6:10],
                                                   self.sigmas_f_nom[0, 6:10])
            self.sigmas_f[i, :6] = x_err_prop[i, :6]
            self.sigmas_f[i, 6:9] = 2 * \
                (x_err_prop[i, 7:10] / x_err_prop[i, 6])

        self.x_err, self.P = unscented_transform(self.sigmas_f,
                                                 self.points.Wm,
                                                 self.points.Wc,
                                                 self.Q,
                                                 None,
                                                 None)

        for i, s in enumerate(self.sigmas_f_nom):
            self.sigmas_h[i] = self.sigmas_f[i]
        z_pred, S = unscented_transform(self.sigmas_h,
                                        self.points.Wm,
                                        self.points.Wc,
                                        self.R,
                                        None,
                                        None)

        print(z_pred)
        print(S)
        print(np.linalg.det(S))
        input("det")
        Pxz = np.zeros((self._dim_x, self._dim_z))
        for i in range(self._num_sigmas):
            dx = self.sigmas_f[i] - self.x_err
            dz = self.sigmas_h[i] - z_pred
            Pxz += self.points.Wc[i] * np.outer(dx, dz)

        self.K = Pxz @ np.linalg.inv(S)
        z_err = np.zeros_like(self.x_err)
        tmp = residual_z(z, self.sigmas_f_nom[0])
        z_err[:6] = tmp[:6]
        z_err[6:9] = 2 * (tmp[7:10] / tmp[6])

        self.y = z_err - z_pred
        # input()

        self.x_err += self.K @ self.y

        self.P -= self.K @ S @ self.K.T
        print(np.linalg.det(self.P))

        self.x[:6] = self.x_err[:6] + self.sigmas_f_nom[0, :6]
        d_alpha = self.x_err[6:9]
        delta_q = (1 / np.sqrt(4 + d_alpha.T @ d_alpha)) * np.array([2, d_alpha[0], d_alpha[1],
                                                                     d_alpha[2]])
        self.x[6:10] = quaternion_prod(self.sigmas_f_nom[0, 6:10], delta_q)
        self.x_err = np.zeros(self._dim_x)


class ESUKF:
    def __init__(self, dim_x, dim_z, fx_err, fx_nom, hx, Q, R, points,
                 dt=0.05, residual_x=None, residual_z=None):
        self.x_err = np.zeros(dim_x)  # The error state
        self.x = np.zeros(dim_x + 1)  # The nominal state
        self.P = np.zeros((dim_x, dim_x))  # Covariance
        self._dim_x = dim_x
        self._dim_z = dim_z

        self.hx = hx
        self.fx_err = fx_err
        self.fx_nom = fx_nom

        self.dt = dt

        self.Q = Q
        self.R = R

        self.points = points

        if residual_x is None:
            self.residual_x = np.subtract
        else:
            self.residual_x = residual_x
        if residual_z is None:
            self.residual_z = np.subtract
        else:
            self.residual_z = residual_z

        self._num_sigmas = points.num_sigmas()
        self.sigmas_f = np.zeros((self._num_sigmas, self._dim_x))
        self.sigmas_h = np.zeros((self._num_sigmas, self._dim_x))

        self.K = np.zeros((dim_x, dim_z))   # Kalman gain
        self.y = np.zeros((dim_z))          # Residuals
        # self.z = np.zeros([None] * dim_z).T  # measurement
        self.S = np.zeros((dim_z, dim_z))   # system uncertainty

        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()

        self.x_post = self.x.copy()

    def compute_process_sigmas(self, dt, u):
        sigmas = self.points.sigma_points(self.x_err, self.P)

        for i, s in enumerate(sigmas):
            self.sigmas_f[i] = self.fx_err(s, dt, u, self.x)
        self.P_post = self.P.copy()

    def predict(self, u):
        self.x = self.fx_nom(self.x, self.dt, u)
        self.compute_process_sigmas(self.dt, u)
        self.x_err, self.P = unscented_transform(self.sigmas_f,
                                                 self.points.Wm,
                                                 self.points.Wc,
                                                 self.Q,
                                                 None,
                                                 self.residual_x)

    def update(self, z):
        # sigmas = self.points.sigma_points(self.x_err, self.P)
        q_err = np.ones(4)
        xt = np.zeros(self._dim_x+1)
        q = np.ones(4)
        for i, s in enumerate(self.sigmas_f):
            xt[:6] = s[:6]
            q[1:] = 1/2 * s[6:9]
            xt[6:10] = quaternion_prod(self.x[6:10], q)
            self.sigmas_h[i] = self.hx(s)

        z_pred, S = unscented_transform(self.sigmas_h,
                                        self.points.Wm,
                                        self.points.Wc,
                                        self.R,
                                        None,
                                        None)

        Pxz = np.zeros((self._dim_x, self._dim_z))
        for i in range(self._num_sigmas):
            dx = self.sigmas_f[i] - self.x_err
            dz = self.sigmas_h[i] - z_pred
            Pxz += self.points.Wc[i] * np.outer(dx, dz)

        self.K = Pxz @ np.linalg.inv(S)
        tmp = residual_z(z, self.x)
        z_err = np.zeros_like(self.x_err)
        z_err[:6] = tmp[:6]
        ang = 2 * np.arccos(tmp[6])
        if not np.isclose(ang, 0):
            z_err[6:9] = (ang / np.sin(ang / 2)) * (tmp[7:10])

        self.y = z_err - z_pred
        # self.residual_z(z, z_pred)
        # self.y = self.residual_z(z_err, z_pred)

        self.x_err = self.K @ self.y
        self.P -= self.K @ S @ self.K.T
        # self.P = (np.eye(self._dim_x) - self.K @ )

        self.x[:6] += self.x_err[:6]
        q = self.x[6:10]
        theta = self.x_err[6:9]
        q_err = np.ones(4)
        q_err[1:] = 1/2 * theta

        self.x[6:10] = quaternion_prod(q, q_err)
        self.x[6:10] /= np.linalg.norm(self.x[6:10])
        # self.x[10:] += self.x_err[9:]
        self.x_err = np.zeros(self._dim_x)

    def incjet_and_reset(self, z):
        pass


def hx(z):
    return z


def fx_nom(x, dt, u):
    dx = model.fx(x, dt, u)
    res = x + dt * dx
    res[6:10] /= np.linalg.norm(res[6:10])
    return res


def fx(x, dt, u, x_nom):
    dx = model.fx_err_state(x, dt, u, x_nom)
    res = x + dt * dx
    res[6:10] /= np.linalg.norm()
    res_err = np.zeros_like(x)
    res_err[:6] = res
    res_err[6:9] = 2 * (res[7:10] / res[6])
    return res


def residual_z(z, z_pred):
    """Residual for measurements with quaternion handling."""
    # Compute residual for velocity/angular velocity components
    residual = z[:6] - z_pred[:6]

    # Compute quaternion residual using quaternion multiplication
    q_meas = z[6:10]
    q_pred = z_pred[6:10]
    # q_pred = np.ones(4)
    # q_pred[1:] = z_pred[7:10]

    # Quaternion error between measurement and prediction
    q_err = quaternion_error(q_meas, q_pred)
    # Convert small-angle quaternion error to vector form
    # q_err_vec = 2 * q_err[1:]
    res = np.hstack((residual, q_err))
    return res


def sqrt(M):
    eval, evec = np.linalg.eig(M)
    sqrt_eigval = np.sqrt(eval)
    diag = np.diag(sqrt_eigval)
    return evec @ diag @ evec.T


if __name__ == "__main__":
    filename = "data/BlueRov2Heavy_mau_11-05-2024_11-10-46.h5"
    # filename = "data/BlueRov2Heavy_no_rollover_in_attitude_11-12-2024_16-45-23.h5"
    data = load_data(filename)
    state = data["x"]
    inputs = data["u"]
    t_vec = data["t"]
    model = models.BlueRov2Heavy()
    points = MerweScaledSigmaPoints(
        9, alpha=1e-4, beta=2.0, kappa=0)

    measurements = state.copy()

    Q = np.eye(9) * 1e-4
    R = np.eye(9) * 1
    P = np.eye(9) * 1
    kf = USQUE(9, 9, fx_err=fx, fx_nom=fx_nom, hx=hx, Q=Q,
               R=Q, points=points, residual_z=residual_z)
    x0 = np.zeros(10)
    x0[6:10] = euler_to_quat([0, 0, 0])
    kf.x = x0
    kf.P = P
    kf.Q = Q
    kf.R = R

    x_bar = np.zeros((len(t_vec), 10))
    for i, (x, u, z) in enumerate(zip(state, inputs, measurements)):
        print(i)
        # kf.predict(u)
        kf.update(z, u)
        x_bar[i, :] = kf.x.copy()
    # x_bar[i, 6:10] = x_bar[i, 6:10] / np.linalg.norm(x_bar[i, 6:10])

    plotting.plot_state_est(x_bar[:, :10], state, state, t_vec)
    plt.show()


def mau():
    filename = "data/BlueRov2Heavy_mau_11-05-2024_11-10-46.h5"
    filename = "data/BlueRov2Heavy_no_rollover_in_attitude_11-12-2024_16-45-23.h5"
    filenam = "data/BlueRov2Heavy_test_11-16-2024_18-09-39.h5"
    data = load_data(filename)
    state = data["x"]
    inputs = data["u"]
    t_vec = data["t"]
    model = models.BlueRov2Heavy()
    points = MerweScaledSigmaPoints(9, alpha=1e-3, beta=2.0, kappa=0)

    measurements = state.copy()

    Q = np.eye(9) * 1
    R = np.eye(9) * 1e-9
    P = np.eye(9) * 1e-4
    kf = ESUKF(9, 9, fx_err=fx, fx_nom=fx_nom, hx=hx, Q=Q,
               R=Q, points=points, residual_z=residual_z)
    x0 = np.zeros(10)
    x0[6:10] = euler_to_quat([0, 0, 0])
    kf.x = x0
    kf.P = P
    kf.Q = Q
    kf.R = R

    x_bar = np.zeros((len(t_vec), 10))
    for i, (x, u, z) in enumerate(zip(state, inputs, measurements)):
        print(i)
        kf.predict(u)
        kf.update(z)
        x_bar[i, :] = kf.x.copy()
    # x_bar[i, 6:10] = x_bar[i, 6:10] / np.linalg.norm(x_bar[i, 6:10])

    plotting.plot_state_est(x_bar[:, :10], state, state, t_vec)
    plt.show()
