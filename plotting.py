import numpy as np
import matplotlib.pyplot as plt


def plot_state_kf(x_bar, x, meas, t_vec):
    x_bar = x_bar.copy()
    x = x.copy()
    fix, axes = plt.subplots(3, 3, figsize=(10, 15), sharex=True)
    axes = axes.flatten()
    labels = ("u", "v", "w", r"$\dot \phi$", r"$\dot \theta$",
              r"$\dot \psi$", r"$\phi$", r"$\theta$", r"$\psi$")
    # for i in range(6, 9):
    #     x[:, i] = np.rad2deg(x[:, i])
    #     x_bar[:, i] = np.rad2deg(x_bar[:, i])
    #     meas[:, i] = np.rad2deg(meas[:, i])

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
 

