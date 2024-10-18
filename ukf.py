from filterpy.kalman import (
    UnscentedKalmanFilter,
    JulierSigmaPoints,
    MerweScaledSigmaPoints,
)
import numpy as np
import models
import h5py
import matplotlib.pyplot as plt
import scipy
import plotting


def load_data(filename: str) -> (np.ndarray, np.ndarray, np.ndarray):
    with h5py.File(filename, "r") as f:
        x = np.array(f["x"])
        u = np.array(f["u"])
        t = np.array(f["t"])

    return x, u, t


class AugmentedModel:
    def __init__(self, model):
        self.model = model
        self._ground_truth = [
            self.model.Xu,
            self.model.Yv,
            self.model.Zw,
            self.model.Kp,
            self.model.Mq,
            self.model.Nr,
            self.model.Xdu,
            self.model.Ydv,
            self.model.Zdw,
            self.model.Kdp,
            self.model.Mdq,
            self.model.Ndr,
        ]

    def fx(self, x, dt, fx_args):
        self.update_hydrodynamic_params(x)
        dx = self.model.fx_q(x[:10], 0, fx_args)
        x[:10] = x[:10] + dx * dt

        return x

    def hx(self, z):
        a = np.block([[np.eye(10), np.zeros((10, len(self.ground_truth)))]])
        return a @ z

    def update_hydrodynamic_params(self, x):
        self.model.Xu = x[10]
        self.model.Yv = x[11]
        self.model.Zw = x[12]
        self.model.Kp = x[13]
        self.model.Mq = x[14]
        self.model.Nr = x[15]
        self.model.Xdu = x[16]
        self.model.Ydv = x[17]
        self.model.Zdw = x[18]
        self.model.Kdp = x[19]
        self.model.Mdq = x[20]
        self.model.Ndr = x[21]

    @property
    def ground_truth(self):
        return self._ground_truth


class AugmentedModelQuadratic:
    def __init__(self, model):
        self.model = model
        self.Zw = self.model.Zw
        self.Xu = self.model.Xu
        self.Zabs_w = self.model.Xabs_u

    def fx(self, x, dt, fx_args):
        x = x.copy()
        # y = scipy.integrate.odeint(self.model.fx, x[:9].copy(), [dt], args=(fx_args,))
        # y = solve_ivp(self.model.fx, [0, dt], x[:9], args=(fx_args,))
        dx = self.model.fx_q(x[:10], 0, fx_args)
        # return dx
        # x[:9] = y[0].copy()
        x[:10] = x[:10] + dx * dt
        x[6:10] /= np.linalg.norm(x[6:10])
        return x

    def hx(self, z):
        a = np.block([[np.eye(9), np.zeros((9, 1))]])
        return a @ z


def no_quadratic():
    model = AugmentedModel(models.BlueRov2Heavy())
    dim = 10 + len(model.ground_truth)
    points = MerweScaledSigmaPoints(dim, alpha=1e-3, beta=2.0, kappa=3.0 - dim)
    # points = JulierSigmaPoints(dim, kappa=3.-dim)
    filename = "data/BlueRov2Heavy_20_xyz__10-12-2024_13-01-19.h5"
    filename = "data/BlueRov2Heavy_20_xyz_t50_yawjump_10-12-2024_13-28-50.h5"
    filename = "data/BlueRov2Heavy_50_xyz__10-12-2024_14-47-56.h5"
    filename = "data/BlueRov2Heavy_x_const_balanced_10-12-2024_15-04-16.h5"
    filename = "data/BlueRov2Heavy_quickstep_all_10-12-2024_15-11-03.h5"
    # filename = "data/BlueRov2Heavy_yaw_only_10-12-2024_15-13-02.h5"
    # filename = "data/BlueRov2Heavy_const_roll_10-12-2024_15-17-47.h5"
    filename = "data/BlueRov2Heavy_big_all_10-15-2024_15-32-53.h5"
    filename = "data/BlueRov2Heavy_big_all_10-15-2024_15-34-50.h5"
    filename = "data/BlueRov2Heavy_x_const_balanced_10-12-2024_15-04-16.h5"

    state, inputs, t_vec = load_data(filename)
    measurements = state.copy()
    measurements[:, :3] += np.random.normal(0, 0.01, state[:, :3].shape)
    measurements[:, 3:6] += np.random.normal(0, 0.001, state[:, 3:6].shape)
    measurements[:, 6:10] += np.random.normal(0, 0.00087, state[:, 6:10].shape)
    # measurements = state + np.random.normal(0, 0.01, state.shape)
    Q = np.eye(dim) * 1e-4
    Q[10, 10] = 1e-1  # Xu
    Q[11, 11] = 1e-1  # Yv
    Q[12, 12] = 1e-1  # Zw
    Q[13, 13] = 1e-3  # Kp
    Q[14, 14] = 1e-3  # Mq
    Q[15, 15] = 1e-2  # Nr
    Q[16, 16] = 1e-2  # Xdu
    Q[17, 17] = 1e-2  # Ydv
    Q[18, 18] = 1e-2  # Zdw
    Q[19, 19] = 1e-3  # Kdp
    Q[20, 20] = 1e-3  # Mdq
    Q[21, 21] = 1e-3  # Ndr

    R = np.eye(10) * 0.01
    P = np.eye(dim)  # * 1e-4
    # P[9, 9] = 10
    # P[10, 10] = 0.0001
    dt = 0.05

    ukf = UnscentedKalmanFilter(
        dim_x=dim, dim_z=10, fx=model.fx, hx=model.hx, dt=0.05, points=points
    )
    ukf.Q = Q
    ukf.R = R
    ukf.P = P
    x_bar = np.zeros((len(t_vec), dim))
    x0 = np.zeros(dim)
    x0[:10] = state[0, :10]

    # x0[10:] = model.ground_truth
    ukf.x = x0
    for i, (x, u, z) in enumerate(zip(state, inputs, measurements)):
        ukf.predict(fx_args=u)
        ukf.update(z)
        x_bar[i, :] = ukf.x.copy()

    plotting.plot_state_kf(x_bar[:, :9], state, measurements, t_vec)
    labels = (r"$X_u$", r"$Y_v$", r"$Z_w$", r"$K_p$", r"$M_q$", r"$N_r$")
    plotting.plot_hydro_params(
        x_bar[:, 10:16], model.ground_truth[:6], t_vec, labels, title="one"
    )
    labels = (
        r"$X_{\dot u}$",
        r"$Y_{\dot v}$",
        r"$Z_{\dot w}$",
        r"$K_{\dot p}$",
        r"$M_{\dot q}$",
        r"$N_{\dot r}$",
    )
    plotting.plot_hydro_params(
        x_bar[:, 16:], model.ground_truth[6:], t_vec, labels, title="two"
    )
    plt.show()


def old_main():
    # points = JulierSigmaPoints(9)
    points = MerweScaledSigmaPoints(10, alpha=0.1, beta=2.0, kappa=-7.0)
    # filename = "data/BlueRov2Heavy_uniform_all_10-08-2024_11-08-19.h5"
    filename = "data/BlueRov2Heavy_random_10-09-2024_10-02-37.h5"
    # filename = "data/BlueRov2Heavy_testing_10-08-2024_11-02-06.h5"
    state, inputs, t_vec = load_data(filename)
    measurements = state + np.random.normal(0, 0.01, state.shape)
    Q = np.eye(10) * 0.01
    Q[-1, -1] = 0.00
    R = np.eye(9) * 0.01
    P = np.eye(10)
    dt = 0.05
    model = AugmentedModelQuadratic(models.BlueRov2Heavy())

    ukf = UnscentedKalmanFilter(
        dim_x=10, dim_z=9, fx=model.fx, hx=model.hx, dt=dt, points=points
    )
    ukf.Q = Q
    ukf.R = R
    ukf.P = P

    x_bar = np.zeros((len(t_vec), 10))

    for i, (x, u, z) in enumerate(zip(state, inputs, measurements)):
        ukf.predict(fx_args=u)
        ukf.update(z)
        x_bar[i, :] = ukf.x.copy()

    plotting.plot_state_kf(x_bar[:, :9], state, measurements, t_vec)
    # plt.figure()
    # plt.plot(t_vec, x_bar[:, 9])
    # plt.axhline(Zw, linestyle="--")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    no_quadratic()
    # main()
