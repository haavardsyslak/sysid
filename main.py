import numpy as np
import matplotlib.pyplot as plt
import models
import scipy
from input_generator import MultiDOFStep
from datetime import datetime
import h5py
import utils


def heave_step(t):
    u = np.zeros(6)
    if t >= 5:
        u[2] = 20
    return u


class OdeSolver:
    def __init__(self, model, input_func, t_vec, fx=None):
        self.model = model
        self.get_input = input_func
        self.t_vec = t_vec
        self.input = np.zeros((len(t_vec), 6))
        self.idx = 0
        self.fx = fx if fx is None else model.fx

    def dx_euler(self, x, t):
        u = self.get_input(t)
        idx = np.searchsorted(self.t_vec, t)
        self.input[self.idx:(idx - 1), :] = u
        self.idx = idx - 1
        return self.model.fx(x, t, u)

    def dx(self, t, x):
        u = self.get_input(t)
        idx = np.searchsorted(self.t_vec, t)
        self.input[self.idx:(idx - 1), :] = u
        self.idx = idx - 1
        return self.model.fx_q(x, t, u)

    def solve(self, x0, dt=None):
        if dt is None:
            dt = self.t_vec[1] - self.t_vec[0]
        ode_solver = scipy.integrate.ode(self.dx)
        ode_solver.set_integrator("dopri5")
        ode_solver.set_initial_value(x0)

        res = []
        while ode_solver.successful() and ode_solver.t < self.t_vec[-1]:
            ode_solver.integrate(ode_solver.t + dt)
            norm = np.linalg.norm(ode_solver.y[6:])

            # ode_solver.y[6:] /= norm
            res.append(ode_solver.y.copy())

        res = np.array(res)
        return res


def get_input_range(sim_time):
    return (
        [(0, min(5, sim_time)), (0, 0)],
        [(0, min(5, sim_time)), (0, 0)],
        [(0, min(5, sim_time)), (0, 0)],
        [(0, min(5, sim_time)), (10, 10)],
        [(0, min(5, sim_time)), (0, 0)],
        [(0, min(5, sim_time)), (0, 0)],
    )

def save_data_quat(rov, x, u, t_vec, filename=""):
    time = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")

    filename = f"./data/{type(rov).__name__}_{filename}_{time}.h5"

    with h5py.File(filename, "w") as f:
        f.create_dataset("x", data=x)
        f.create_dataset("t", data=t_vec)
        f.create_dataset("u", data=u)

def save_data_euler(rov, x, u, t_vec, filename=""):
    time = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")

    filename = f"./data/{type(rov).__name__}_{filename}_{time}.h5"

    with h5py.File(filename, "w") as f:
        f.create_dataset("x", data=x)
        f.create_dataset("t", data=t_vec)
        f.create_dataset("u", data=u)


def main():
    rov = models.BlueRov2Heavy()
    sim_time = 20
    dt = 0.05
    t_vec = np.arange(0, sim_time, dt)
    u_fn = heave_step
    u_fn = MultiDOFStep(get_input_range(sim_time))
    solver = OdeSolver(rov, u_fn.get_input_vec, t_vec)
    x0 = np.zeros(9)

    x = np.zeros((len(t_vec), 9))
    x = scipy.integrate.odeint(solver.dx, x0, t_vec)

    plot(t_vec, x)
    plot_input(t_vec, solver.input)
    plt.show()

    inn = input("Save [y/N] ")
    if inn.lower() == "y":
        filename = input("filename? ")
        save_data(rov, x, solver.input, t_vec, filename)


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
    euler = utils.quats_to_euler(q)
    print(euler.shape)
    x[:, 6:] = euler
    plot(t_vec, x)


def quat_test():
    rov = models.BlueRov2Heavy()
    sim_time = 50
    dt = 0.05
    t_vec = np.arange(0, sim_time, dt)
    u_fn = heave_step
    u_fn = MultiDOFStep(get_input_range(sim_time))
    x0 = np.zeros(10)
    x0[6:] = utils.euler_to_quat([0, 0, 0])
    # xq = np.zeros((len(t_vec), 10))
    solver = OdeSolver(rov, u_fn.get_input_vec, t_vec)
    x = solver.solve(x0, dt)

    # plot(t_vec, x)
    plot_quat(t_vec, x)
    plot_input(t_vec, solver.input)
    plt.show()

    inn = input("Save [y/N] ")
    if inn.lower() == "y":
        filename = input("filename? ")
        save_data_quat(rov, x, solver.input, t_vec, filename)


if __name__ == "__main__":
    quat_test()
