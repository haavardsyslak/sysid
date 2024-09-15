import numpy as np
import matplotlib.pyplot as plt


def plot_input(t, u):
    labels = ["x", "y", "z", "roll", "pitch", "yaw"]
    num_subplots = len(labels)

    fig, axes = plt.subplots(3, 2, figsize=(12, 10))

    axes = axes.flatten()

    for i in range(num_subplots):
        axes[i].plot(t, u[:, i])
        axes[i].set_title(f"{labels[i]} vs Time")
        axes[i].set_xlabel("Time")
        axes[i].set_ylabel(labels[i])
        axes[i].grid(True)

    plt.tight_layout()


def plot_ned_state(timesteps, eta):
    """
    Plots the state vector components over time.

    Args:
        timesteps (list or array): The time steps.
        state_vector (2D list or array): The state vectors [x, y, z, roll, pitch, yaw].
                                         Each row corresponds to one time step.
    """
    labels = ["x", "y", "z", "roll", "pitch", "yaw"]
    num_subplots = len(labels)

    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    eta[:, 3] = np.rad2deg(eta[:, 3])
    eta[:, 4] = np.rad2deg(eta[:, 4])
    eta[:, 5] = np.rad2deg(eta[:, 5])

    axes = axes.flatten()

    print(axes.shape)
    for i in range(num_subplots):
        axes[i].plot(timesteps, eta[:, i])
        axes[i].set_title(f"{labels[i]} vs Time")
        axes[i].set_xlabel("Time")
        axes[i].set_ylabel(labels[i])
        axes[i].grid(True)

    plt.tight_layout()
