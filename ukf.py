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
    measurements[:, 3:6] += np.random.normal(0, 0.0005, state[:, 3:6].shape)
    measurements[:, 6:10] += np.random.normal(0, 0.0087, state[:, 6:10].shape)
    # measurements = state + np.random.normal(0, 0.01, state.shape)
    Q = np.eye(dim) * 1e-4
    Q[10, 10] = 3e-2  # Xu
    Q[11, 11] = 1e-2  # Yv
    Q[12, 12] = 1e-2  # Zw
    Q[13, 13] = 1e-5  # Kp
    Q[14, 14] = 1e-5  # Mq
    Q[15, 15] = 1e-5  # Nr
    Q[16, 16] = 1e-2  # Xdu
    Q[17, 17] = 1e-2  # Ydv
    Q[18, 18] = 1e-2  # Zdw
    Q[19, 19] = 6e-6  # Kdp
    Q[20, 20] = 1e-6  # Mdq
    Q[21, 21] = 1e-6  # Ndr

    R = np.eye(10) * 0.0001
    P = np.eye(dim) * 1e-4
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
    # x0[:10] = state[0, :10]

    # x0[10:] = model.ground_truth
    ukf.x = x0
    for i, (x, u, z) in tqdm(
        enumerate(zip(state, inputs, measurements)),
        total=len(t_vec),
        ncols=80,
        desc="Running UKF",
    ):
        ukf.predict(fx_args=u)
        ukf.update(z)
        x_bar[i, :] = ukf.x

    plotting.plot_state_est(x_bar[:, :9], state, measurements, t_vec)
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
    plt.show()


if __name__ == "__main__":
    filename = ""

    args = sys.argv[1:]
    if args:
        filename = args[0]

    if filename:
        run_ukf(filename)
    else:
        print(
            "No file provided",
            "Set filename in ukf.py or provide it as an arg:",
            "python3 ukf.py <filepath>",
        )
