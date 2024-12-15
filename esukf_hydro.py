import numpy as np
from scipy.spatial.transform import Rotation
from utils.data import load_data
import utils.attitude as attitude
import models
from ESUKF import ESUKF, SigmaPoints
from ukf import AugmentedModel
import plotting
import matplotlib.pyplot as plt


def fx():
    pass


def hx():
    pass


if __name__ == "__main__":
    # filename = "data/BlueRov2Heavy_mau_11-05-2024_11-10-46.h5"
    filename = "data/BlueRov2Heavy_Zdw__25_10-27-2024_17-06-18.h5"
    data = load_data(filename)
    data = load_data(filename)
    state = data["x"]
    inputs = data["u"]
    t_vec = data["t"]
    # model = models.BlueRov2Heavy()
    model = AugmentedModel(models.BlueRov2Heavy())
    dim = 10 + len(model.ground_truth)
    points = SigmaPoints(9 + len(model.ground_truth), alpha=1e-3, beta=4.5, kappa=0)
    # plotting.plot_state(t_vec, state)
    # plt.show()

    measurements = state.copy()
    # measurements += np.random.normal(0, 0.05, state.shape)

    Q = np.eye(dim - 1) * 1e-6
    Q[10:, 10:] = np.eye(dim - 1 - 10) * 1e-10
    # Q[10, 10] = 5e-3  # Xu
    # Q[11, 11] = 8e-3  # Yv
    # Q[12, 12] = 5e-3  # Zw
    # Q[13, 13] = 4e-3  # Kp
    # Q[14, 14] = 1e-3  # Mq
    # Q[15, 15] = 1e-3  # Nr
    # Q[16, 16] = 6e-3  # Xdu
    # Q[17, 17] = 6e-3  # Ydv
    Q[18, 18] = 4e-3  # Zdw
    # Q[19, 19] = 1e-3  # Kdp
    # Q[20, 20] = 1e-3  # Mdq
    # Q[21, 21] = 1e-3  # Ndr
    R = np.eye(6) * 1e-4
    P = np.eye(dim - 1) * 1e-4
    kf = ESUKF(dim - 1, 6, fx=model.fx, hx=model.hx_qukf, Q=Q,
               R=R, points=points)
    x0 = np.zeros(dim)
    x0[6:10] = attitude.euler_to_quat([0, 0, 0])
    x0[10:] = model.ground_truth

    kf.x = x0
    kf.P = P
    kf.Q = Q
    kf.R = R
    x_bar = np.zeros((len(t_vec), dim))

    for i, (x, u , z) in enumerate(zip(state, inputs, measurements)):
        print(i)
        kf.ukf_cylce(u, z[:6])
        x_bar[i,:] = kf.x.copy()


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


