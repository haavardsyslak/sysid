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
from tqdm import tqdm
from utils.data import load_data
import sys
from dataclasses import dataclass


class AugmentedQuadratic:
    def __init__(self, model):
        self.model = model
        self._ground_truth = [
            self.model.Xabs_u,
            self.model.Yabs_v,
            self.model.Zabs_w,
            self.model.Kabs_p,
            self.model.Mabs_q,
            self.model.Nabs_r,
        ]

        self.H = np.block([[np.eye(10), np.zeros((10, len(self._ground_truth)))]])


    def fx(self, x, dt, fx_args):
        self.update_hydrodynamic_params(x)
        dx = self.model.fx(x[:10], 0, fx_args)
        dx = self.model.fx(x[:10], 0, fx_args)
        # RK4 integration steps
        k1 = self.model.fx(x[:10], 0, fx_args)
        k2 = self.model.fx(x[:10] + 0.5 * k1 * dt, 0, fx_args)
        k3 = self.model.fx(x[:10] + 0.5 * k2 * dt, 0, fx_args)
        k4 = self.model.fx(x[:10] + k3 * dt, 0, fx_args)

        # Update state using RK4 formula
        x[:10] = x[:10] + (k1 + 2 * k2 + 2 * k3 + k4) * (dt / 6)

        # return x
        # x[:10] = x[:10] + dx * dt

        return x

    def hx(self, z):
        return self.H @ z

    def update_hydrodynamic_params(self, x):
        self.model.Xabs_u = x[10]
        self.model.Yabs_v = x[11]
        self.model.Zabs_w = x[12]
        self.model.Kabs_p = x[13]
        self.model.Mabs_q = x[14]
        self.model.Nabs_r = x[15]


    @property
    def ground_truth(self):
        return self._ground_truth


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

        self.H = np.block([[np.eye(10), np.zeros((10, len(self.ground_truth)))]])

    def fx(self, x, dt, fx_args):
        self.update_hydrodynamic_params(x)
        dx = self.model.fx(x[:10], 0, fx_args)
        # RK4 integration steps
        k1 = self.model.fx(x[:10], 0, fx_args)
        k2 = self.model.fx(x[:10] + 0.5 * k1 * dt, 0, fx_args)
        k3 = self.model.fx(x[:10] + 0.5 * k2 * dt, 0, fx_args)
        k4 = self.model.fx(x[:10] + k3 * dt, 0, fx_args)

        # Update state using RK4 formula
        x[:10] = x[:10] + (k1 + 2 * k2 + 2 * k3 + k4) * (dt / 6)

        # return x
        # x[:10] = x[:10] + dx * dt

        return x

    def hx(self, z):
        return self.H @ z

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


@dataclass
class UKFResults:
    x: np.ndarray
    x_bar: np.ndarray
    meas: np.ndarray
    input: np.ndarray
    P: np.ndarray
    innovation: np.ndarray
    S: np.ndarray
    t_vec: np.ndarray


def run_ukf(filename):
    model = AugmentedModel(models.BlueRov2Heavy())
    dim = 10 + len(model.ground_truth)
    points = MerweScaledSigmaPoints(dim, alpha=1e-3, beta=2.0, kappa=3.0 - dim)

    data = load_data(filename)
    state = data["x"]
    inputs = data["u"]
    t_vec = data["t"]
    measurements = state.copy()
    measurements[:, :3] += np.random.normal(0, 0.005, state[:, :3].shape)
    measurements[:, 3:6] += np.random.normal(0, 2.5e-3, state[:, 3:6].shape)
    measurements[:, 6:10] += np.random.normal(0, 0.00087, state[:, 6:10].shape)
    # measurements = state + np.random.normal(0, 0.01, state.shape)
    Q = np.eye(dim) * 1e-5

    Q[10, 10] = 5e-1  # Xu
    Q[11, 11] = 8e-1  # Yv
    Q[12, 12] = 5e-1  # Zw
    Q[13, 13] = 4e-2  # Kp
    Q[14, 14] = 1e-3  # Mq
    Q[15, 15] = 1e-5  # Nr
    Q[16, 16] = 6e-2  # Xdu
    Q[17, 17] = 6e-2  # Ydv
    Q[18, 18] = 4e-2  # Zdw
    Q[19, 19] = 1e-4  # Kdp
    Q[20, 20] = 1e-4  # Mdq
    Q[21, 21] = 1e-4  # Ndr

    R = np.eye(10)  # * 0.001
    R[:3, :3] *= 0.008
    R[3:6, 3:6] *= 1e-4
    R[6:, 6:] *= 0.0008
    P = np.eye(dim) * 1e-4
    # P[9, 9] = 11
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
    # x0[:10] = state[0, :10]

    # x0[10:] = model.ground_truth.copy()
    ukf.x = x0
    P_post = np.zeros((len(t_vec), dim, dim))
    innovation = np.zeros((len(t_vec), 10))
    S = np.zeros((len(t_vec), 10, 10))
    for i, (x, u, z) in tqdm(
        enumerate(zip(state, inputs, measurements)),
        total=len(t_vec),
        ncols=80,
        desc="Running UKF",
    ):
        ukf.predict(fx_args=u)
        ukf.update(z)
        x_bar[i, :] = ukf.x
        P_post[i, :] = ukf.P_post
        S[i, :] = ukf.S
        innovation[i, :] = ukf.y

    # return UKFResults(state, x_bar, measurements, inputs, P_post, innovation, t_vec)
    plotting.plot_state_est(x_bar[:, :10], state, measurements, t_vec)
    labels = (r"$X_u$", r"$Y_v$", r"$Z_w$", r"$K_p$", r"$M_q$", r"$N_r$")
    plotting.plot_hydro_params(
        x_bar[:, 10:16], model.ground_truth[:6], t_vec, labels, title="Linear damping"
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
        x_bar[:, 16:],
        model.ground_truth[6:],
        t_vec,
        labels,
        title="Hydrodynamic added mass",
    )

    plotting.plot_input(t_vec, inputs)
    plt.show()


if __name__ == "__main__":
    filename = "data/BlueRov2Heavy__10-26-2024_15-40-22.h5"
    # filename = "data/BlueRov2Heavy_Mp__1dot3_10-27-2024_12-16-43.h5"
    # filename = "data/BlueRov2Heavy_test_10-27-2024_16-04-25.h5"
    # filename = "data/BlueRov2Heavy_test2222222222222_10-27-2024_16-16-32.h5"

    # filename= "data/BlueRov2Heavy_ttttttttttttttttttttttttt_10-27-2024_16-22-48.h5"
    filename = "data/BlueRov2Heavy_some_teset_10-27-2024_16-47-27.h5"
    filename = "data/BlueRov2Heavy_input_10-27-2024_16-54-39.h5"
    args = sys.argv[1:]
    if args:
        filename = args[0]

    if filename:
        run_ukf(filename)
        # run_quadratic(filename)
    else:
        print(
            "No file provided",
            "Set filename in ukf.py or provide it as an arg:",
            "python3 ukf.py <filepath>",
        )
