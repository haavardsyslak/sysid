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

    fig.suptitle("States")


def plot_state_est(x_bar, x, meas, t_vec):
    plt.rcParams.update({"font.size": 14})
    x_bar = x_bar.copy()
    x = x.copy()
    meas = meas.copy()
    # fig, axes = plt.subplots(3, 3, figsize=(10, 15), sharex=True)
    fig_trans, ax1 = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
    fig_ang, ax2 = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
    fig_euler, ax3 = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
    axes = np.array((ax1, ax2, ax3)).flatten()
    labels = ("u [m/s]", "v [m/s]", "w [m/s]", r"$\dot \phi [rad/s]$", r"$\dot \theta$ [rad/s]",
              r"$\dot \psi$ [rad/s]", r"$\phi$ [rad/s]", r"$\theta$ [rad]", r"$\psi$ [rad]")


    # x[:, :6] = x[:, :6].copy()
    # x_bar[:, :6] = x_bar[:, :6].copy()
    euler = utils.attitude.quats_to_euler(x[:, 6:10].copy())
    euler_est = utils.attitude.quats_to_euler(x_bar[:, 6:10].copy())
    euler_meas = utils.attitude.quats_to_euler(meas[:, 6:].copy())
    x[:, 6:9] = euler
    x_bar[:, 6:9] = euler_est
    meas[:, 6:9] = euler_meas

    for i in range(9):
        axes[i].plot(t_vec, x[:, i], label="True value")
        axes[i].plot(t_vec, x_bar[:, i], linestyle="--", label="Estimate")
        if i < 6:
            axes[i].scatter(t_vec, meas[:, i], s=10, color="gray", alpha=0.5, label="Measurement")
        axes[i].set_ylabel(labels[i], rotation=0)
        axes[i].grid(True)
        if not ((i + 1) % 3):
            axes[i].set_label("Time [s]")
        axes[i].legend()

    axes[-1].set_xlabel("Time [s]")
    fig_trans.suptitle("Translational Velcoity")
    fig_trans.tight_layout()
    fig_ang.suptitle("Angluar Velocity")
    fig_ang.tight_layout()
    fig_euler.suptitle("Euler Angles")
    fig_euler.tight_layout()
    fig_trans.savefig("translational_velocity_qukf.pdf")
    fig_ang.savefig("angular_velocity_qukf.pdf")
    fig_euler.savefig("euler_angles_qukf.pdf")

def plot_hydro_params(x_bar, ground_truth, t_vec, labels, title=""):
    fig, axes = plt.subplots(3, 2, figsize=(10, 15), sharex=True)
    axes = axes.flatten()

    for i, (ax, label) in enumerate(zip(axes, labels)):
        ax.plot(t_vec, x_bar[:, i])
        ax.axhline(ground_truth[i], linestyle="--")
        ax.set_ylabel(labels[i], rotation=0)
        ax.grid(True)

    if title is not None:
        # fig.canvas.manager.set_window_title(title)
        fig.suptitle(title)

    # plt.tight_layout()


def plot_input(t_vec, u):
    fig, axes = plt.subplots(2, 3, sharex=True)
    labels = ("X", "Y", "Z", "K", "M", "N")
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        ax.plot(t_vec, u[:, i])
        ax.set_ylabel(labels[i])
        ax.grid(True)

    axes[-1].set_xlabel("Time")
    fig.suptitle("Inputs")
    # plt.tight_layout()

def plot_nis(innovation, S, labels, title=""):
    nis = get_nis(innovation, S)

def plot_ukf_error(x_bar, x, t_vec):
    plt.rcParams.update({"font.size": 14})
    x_bar = x_bar.copy()
    x = x.copy()
    # meas = meas.copy()
    # fig, axes = plt.subplots(3, 3, figsize=(10, 15), sharex=True)
    fig_trans, ax1 = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
    fig_ang, ax2 = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
    fig_euler, ax3 = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
    axes = np.array((ax1, ax2, ax3)).flatten()
    labels = ("u [m/s]", "v [m/s]", "w [m/s]", r"$\dot \phi [rad/s]$", r"$\dot \theta$ [rad/s]",
              r"$\dot \psi$ [rad/s]", r"$\phi$ [rad/s]", r"$\theta$ [rad]", r"$\psi$ [rad]")


    # x[:, :6] = x[:, :6].copy()
    # x_bar[:, :6] = x_bar[:, :6].copy()
    euler = utils.attitude.quats_to_euler(x[:, 6:10].copy())
    euler_est = utils.attitude.quats_to_euler(x_bar[:, 6:10].copy())
    # euler_meas = utils.attitude.quats_to_euler(meas[:, 6:].copy())
    x[:, 6:9] = euler
    x_bar[:, 6:9] = euler_est
    # meas[:, 6:9] = euler_meas
    err = np.zeros_like(x)
    err[:, :6] = x[:, :6] - x_bar[:, :6]
    err[:, 6] = angle_diff(x[:, 6], x_bar[:, 6])
    err[:, 7] = angle_diff(x[:, 7], x_bar[:, 7])
    err[:, 8] = angle_diff(x[:, 8], x_bar[:, 8])

    for i in range(9):
        axes[i].plot(t_vec, err[:, i], label="Error")
        axes[i].set_ylabel(labels[i])
        axes[i].grid(True)
        if not ((i + 1) % 3):
            axes[i].set_label("Time [s]")
        axes[i].legend()

    fig_trans.suptitle("Translational Velcoity")
    fig_trans.tight_layout()
    fig_ang.suptitle("Angluar Velocity")
    fig_ang.tight_layout()
    fig_euler.suptitle("Euler Angles")
    fig_euler.tight_layout()
    fig_trans.savefig("translational_velocity_qukf_err.pdf")
    fig_ang.savefig("angular_velocity_qukf_err.pdf")
    fig_euler.savefig("euler_angles_qukf_err.pdf")

def angle_diff(arr1, arr2):

    diff = np.array(arr1) - np.array(arr2)
    
    # Normalize the difference to be within [-pi, pi]
    diff = (diff + np.pi) % (2 * np.pi) - np.pi
    
    return diff


