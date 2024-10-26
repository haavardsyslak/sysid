import numpy as np
import matplotlib.pyplot as plt
import models
from input_generator import MultiDOFStep
import utils.attitude as attitude
import plotting
from ode_solver import OdeSolver
from utils.data import save_data


def heave_step(t):
    u = np.zeros(6)
    if t >= 5:
        u[2] = 20
    return u


def get_input_range(sim_time):
    return (
        [(0, min(5, sim_time)), (-8, 8)],
        [(0, min(5, sim_time)), (-8, 8)],
        [(0, min(5, sim_time)), (-8, 8)],
        [(2, min(5, sim_time)), (-1, 1)],
        [(2, min(5, sim_time)), (-1, 1)],
        [(2, min(5, sim_time)), (-1, 1)],
    )


def compute_input_vec(t_vec, fn):
    input = np.zeros((len(t_vec), 6))
    for i, t in enumerate(t_vec):
        input[i, :] = fn(t)

    return input


def run_simulation():
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
    run_simulation()
