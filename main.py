import numpy as np
import matplotlib.pyplot as plt
import models
from input_generator import MultiDOFStep
import utils.attitude as attitude
import plotting
from ode_solver import OdeSolver
from utils.data import save_data, load_data
from scipy.spatial.transform import Rotation as Rot


def heave_step(t):
    u = np.zeros(6)
    if t >= 5:
        u[2] = 20
    return u


def get_input_range(sim_time):
    return (
        [(2, min(8, sim_time)), (-15, 15)],
        [(2, min(8, sim_time)), (-15, 15)],
        [(2, min(8, sim_time)), (-15, 15)],
        [(2, min(5, sim_time)), (-5, 5)],
        [(2, min(5, sim_time)), (-5, 5)],
        [(2, min(5, sim_time)), (-5, 5)],
    )


def compute_input_vec(t_vec, fn):
    input = np.zeros((len(t_vec), 6))
    for i, t in enumerate(t_vec):
        input[i, :] = fn(t)

    return input


def get_input() -> np.ndarray:
    file = "data/BlueRov2Heavy__10-26-2024_15-40-22.h5"
    data = load_data(file)
    return data["u"]

def zero(t):
    return np.zeros(6)

def run_simulation():
    rov = models.BlueRov2Heavy()
    sim_time = 100
    dt = 0.05
    t_vec = np.arange(0, sim_time, dt)
    print(len(t_vec))
    u_fn = heave_step
    u_fn = MultiDOFStep(get_input_range(sim_time))
    x0 = np.zeros(10)
    x0[6:] = attitude.euler_to_quat([0, 0, 0])
    # u = compute_input_vec(t_vec, u_fn.get_input_vec)
    # u = get_input()
    solver = OdeSolver(rov)
    res = solver.solve(x0, u_fn.get_input_vec, t_vec, dt)
    # res = solver.solve(x0, zero, t_vec, dt)
    x = res.state
    u = res.input

    
    plotting.plot_state(t_vec, x)
    plotting.plot_input(t_vec, u)
    plt.show()

    inn = input("Save [y/N] ")
    if inn.lower() == "y":
        filename = input("filename? ")
        save_data(rov, x, u, t_vec, filename)


if __name__ == "__main__":
    run_simulation()
    # _run()
