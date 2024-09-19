import numpy as np
import models
import h5py
import matplotlib.pyplot as plt


class EKF:
    def __init__(self, model: models.Model, Q: np.ndarray, R: np.ndarray, Pk: np.ndarray = None):
        self.model = model
        self.t = 0
        self.Q = Q
        self.R = R

        if Pk is None:
            self.Pk = np.eye(Q.shape[0])
        else:
            self.Pk = Pk

    def predict(self, x: np.ndarray, u: np.ndarray, t: float) -> np.ndarray:
        dt = t - self.t
        x = dt * self.model.f(x, u)
        F = np.eye(len(x)) + dt * self.model.F(x, u)
        self.Pk = F @ self.Pk @ F.T + self.Q

        return x

    def update(self, x: np.ndarray, z: np.ndarray, t: float) -> np.ndarray:
        H = self.model.H(x)
        # innovation
        y = z - H @ x
        # Innovation cov
        S = H @ self.Pk @ H.T @ + self.R
        # Kalman gain
        K = self.Pk @ H.T @ np.linalg.inv(S)

        x = x + K @ y
        Imat = np.eye(len(x))
        self.Pk = (Imat - K @ H) @ self.Pk

        return x


def load_data(filename):
    with h5py.File(filename, "r+") as f:
        v = np.array(f["v"])
        eta = np.array(f["eta"])
        u = np.array(f["u"])
        t = np.array(f["t"])

    x = np.zeros((len(t), 9))
    x[:, :6] = v
    x[:, 6:] = eta[:, 3:]

    return x, v, eta, u, t


def main():
    x, v, eta, inputs, t_vec = load_data(
        "./data/BlueRov2Heavy_uniform_xyz_yaw_09-15-2024_15-30-15.h5"
    )
    meas = x + np.random.normal(0, 0.1, x.shape)

    x0 = x[0, :]
    Q = np.eye(9)
    R = np.eye(9)
    ekf = EKF(models.BlueRov2Heavy(), Q, R)
    x_hat = np.zeros((len(t_vec), 9))
    x_bar = ekf.predict(x0, inputs[0, :], 0.01)

    for i, (t, states, u, z) in enumerate(zip(t_vec[1:], x[1:, :], inputs[1:, :], meas[1:, :])):
        x_hat[i, :] = ekf.update(x_bar, z, t)
        x_bar = ekf.predict(x_hat[i, :], u, t)
    plot_ekf_stats(t_vec, x_hat, v, eta)


def plot_ekf_stats(t_vec, x_hat, v, eta):
    # Plot the velocity states (v)
    plt.figure(figsize=(10, 8))
    for i in range(6):
        plt.subplot(3, 2, i + 1)
        plt.plot(t_vec, v[:, i], label="True v[{}]".format(i))
        plt.plot(t_vec, x_hat[:, i], label="Estimated v[{}]".format(i), linestyle='--')
        plt.xlabel("Time [s]")
        plt.ylabel("v[{}]".format(i))
        plt.legend()
        plt.grid()

    plt.suptitle("Velocity States Comparison")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    # Plot the eta[3:] states
    plt.figure(figsize=(10, 4))
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.plot(t_vec, eta[:, i + 3], label="True eta[{}]".format(i + 3))
        plt.plot(t_vec, x_hat[:, i + 6], label="Estimated eta[{}]".format(i + 3), linestyle='--')
        plt.xlabel("Time [s]")
        plt.ylabel("eta[{}]".format(i + 3))
        plt.legend()
        plt.grid()

    plt.suptitle("Attitude States (eta[3:]) Comparison")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


if __name__ == "__main__":
    main()
