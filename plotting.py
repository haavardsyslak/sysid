import numpy as np
import matplotlib.pyplot as plt
import utils.attitude


def plot_state(t_vec, x):
    xq = x.copy()
    q = xq[:, 6:]
    # euler = scipy.spatial.transform.Rotation.from_quat(q).as_euler("zyx", degrees=True)
    fig, axes = plt.subplots(3, 3, figsize=(10, 15), sharex=True)
    axes = axes.flatten()
    labels = ("u", "v", "w", r"$\dot \phi$", r"$\dot \theta$",
              r"$\dot \psi$", r"$\phi$", r"$\theta$", r"$\psi$")
    x = np.zeros((len(t_vec), 9))
    x[:, :6] = xq[:, :6].copy()
    euler = utils.attitude.quats_to_euler(q)
    x[:, 6:] = euler
    for i in range(9):
        axes[i].plot(t_vec, x[:, i])
        axes[i].set_ylabel(labels[i], rotation=0)
        axes[i].grid(True)


def plot_state_est(x_bar, x, meas, t_vec):
    x_bar = x_bar.copy()
    x = x.copy()
    fix, axes = plt.subplots(3, 3, figsize=(10, 15), sharex=True)
    axes = axes.flatten()
    labels = ("u", "v", "w", r"$\dot \phi$", r"$\dot \theta$",
              r"$\dot \psi$", r"$\phi$", r"$\theta$", r"$\psi$")

    for i in range(9):
        axes[i].plot(t_vec, x[:, i])
        axes[i].plot(t_vec, x_bar[:, i], linestyle="--")
        axes[i].scatter(t_vec, meas[:, i], s=10)
        axes[i].set_ylabel(labels[i], rotation=0)
        axes[i].grid(True)

    axes[-1].set_xlabel("Time [s]")
    plt.tight_layout()


def plot_hydro_params(x_bar, ground_truth, t_vec, labels, title=""):
    fig, axes = plt.subplots(3, 2, figsize=(10, 15), sharex=True)
    axes = axes.flatten()

    for i, (ax, label) in enumerate(zip(axes, labels)):
        ax.plot(t_vec, x_bar[:, i])
        ax.axhline(ground_truth[i], linestyle="--")
        ax.set_ylabel(labels[i], rotation=0)
        ax.grid(True)

    if title is not None:
        fig.canvas.manager.set_window_title(title)

    plt.tight_layout()


def plot_input(t_vec, u):
    fig, axes = plt.subplots(2, 3, sharex=True)
    labels = ("X", "Y", "Z", "K", "M", "N")
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        ax.plot(t_vec, u[:, i])
        ax.set_ylabel(labels[i])
        ax.grid(True)

    axes[-1].set_xlabel("Time")
    plt.tight_layout()

