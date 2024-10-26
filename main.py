import numpy as np
import matplotlib.pyplot as plt
import models
import scipy
from input_generator import MultiDOFStep
import utils.attitude as attitude
from utils.data import load_data, save_data
from dataclasses import dataclass
import plotting


def heave_step(t):
    u = np.zeros(6)
    if t >= 5:
        u[2] = 20
    return u


@dataclass
class SimulationResults:
    state: np.ndarray
    input: np.ndarray


class OdeSolver:
    def __init__(self, model, fx=None):
        self.model = model
        self.t_vec = None
        self.input = None  # np.zeros((len(t_vec), 6))
        self.idx = 0
        self.fx = fx if fx is not None else model.fx

    def dx_euler(self, t: float, x: np.ndarray) -> np.ndarray:
        u = self.get_input(t)
        idx = np.searchsorted(self.t_vec, t)
        self.input_array[self.idx:(idx - 1), :] = u
        self.idx = idx - 1
        return self.model.fx(x, t, u)

    def dx(self, t: float, x: np.ndarray) -> np.ndarray:
        idx = np.searchsorted(self.t_vec, t)
        if callable(self.input):
            u = self.input(t)
        else:
            u = self.input[idx - 1]

        self.input_array[self.idx:(idx - 1), :] = u
        self.idx = idx - 1
        return self.fx(x, t, u)

    def solve(self, x0, u, t_vec, dt=None):
        if dt is None:
            dt = self.t_vec[1] - self.t_vec[0]
        ode_solver = scipy.integrate.ode(self.dx)
        # ode_solver.set_integrator("dopri5")
        ode_solver.set_initial_value(x0)
        self.input_array = np.zeros((len(t_vec), len(self.model.dtpye_input)))
        self.input = u
        self.t_vec = t_vec

        res = []
        while ode_solver.successful() and ode_solver.t < t_vec[-1]:
            ode_solver.integrate(ode_solver.t + dt)
            norm = np.linalg.norm(ode_solver.y[6:])

            ode_solver.y[6:] /= norm
            res.append(ode_solver.y.copy())

        res = np.array(res)
        return SimulationResults(res, self.input_array)


def get_input_range(sim_time):
    return (
        [(0, min(5, sim_time)), (-8, 8)],
        [(0, min(5, sim_time)), (-8, 8)],
        [(0, min(5, sim_time)), (-8, 8)],
        [(2, min(5, sim_time)), (-1, 1)],
        [(2, min(5, sim_time)), (-1, 1)],
        [(2, min(5, sim_time)), (-1, 1)],
    )


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


def plot(t_vec, x):
    x = x.copy()
    fig, axes = plt.subplots(3, 3, figsize=(10, 15), sharex=True)
    labels = ("u", "v", "w", r"$\dot \phi$", r"$\dot \theta$",
              r"$\dot \psi$", r"$\phi$", r"$\theta$", r"$\psi$")
    axes = axes.flatten()
    for i in range(6, 9):
        x[:, i] = np.rad2deg(x[:, i])

    for i in range(9):
        axes[i].plot(t_vec, x[:, i])
        axes[i].set_ylabel(labels[i], rotation=0)
        axes[i].grid(True)

    axes[-1].set_xlabel("Time")
    plt.tight_layout()


def plot_quat(t_vec, x):
    xq = x.copy()
    q = xq[:, 6:]
    # euler = scipy.spatial.transform.Rotation.from_quat(q).as_euler("zyx", degrees=True)
    x = np.zeros((len(t_vec), 9))
    x[:, :6] = xq[:, :6].copy()
    euler = attitude.quats_to_euler(q)
    x[:, 6:] = euler
    plot(t_vec, x)


def compute_input_vec(t_vec, fn):
    input = np.zeros((len(t_vec), 6))
    for i, t in enumerate(t_vec):
        input[i, :] = fn(t)

    return input


def quat_test():
    rov = models.BlueRov2Heavy()
    sim_time = 100
    dt = 0.05
    t_vec = np.arange(0, sim_time, dt)
    u_fn = heave_step
    u_fn = MultiDOFStep(get_input_range(sim_time))
    x0 = np.zeros(10)
    x0[6:] = attitude.euler_to_quat([0, 0, 0])
    u = compute_input_vec(t_vec, u_fn.get_input_vec)
    solver = OdeSolver(rov)
    res = solver.solve(x0, u, t_vec, dt)
    x = res.state
    u = res.input

    # plot(t_vec, x)
    plotting.plot_state(t_vec, x)
    plotting.plot_input(t_vec, u)
    plt.show()

    inn = input("Save [y/N] ")
    if inn.lower() == "y":
        filename = input("filename? ")
        save_data(rov, x, solver.input, t_vec, filename)


if __name__ == "__main__":
    quat_test()
