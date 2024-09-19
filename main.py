import models
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import h5py

import utils
import plotting


class RandomStepGenrator:
    def __init__(self, duration_range, amplitude_range):
        self.amplitude_range = amplitude_range
        self.duration_range = duration_range
        self.current_amplitude = 0
        self.next_change_time = 0

    def get_next_step(self):
        self.current_amplitude = np.random.uniform(*self.amplitude_range)
        self.next_change_time += np.random.uniform(*self.duration_range)

    def get_amplitude(self, t):
        if t > self.next_change_time:
            self.get_next_step()

        return self.current_amplitude


class MultiDOFStep:
    def __init__(self, ranges):
        # ranges = [(min_on_time, max_on_time), (min_off_time, max_off_time), (min_amp, max_amp)]

        self.generators = []

        for i, val in enumerate(ranges):
            self.generators.append(RandomStepGenrator(val[0], val[1]))

    def get_input_vec(self, t):
        vec = np.zeros(6)
        for i in range(6):
            vec[i] = self.generators[i].get_amplitude(t)

        return vec


def save_data(rov, v, eta, t_vec, u, filename=""):
    time = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")

    filename = f"./data/{type(rov).__name__}_{filename}_{time}.h5"

    with h5py.File(filename, "w") as f:
        f.create_dataset("v", data=v)
        f.create_dataset("eta", data=eta)
        f.create_dataset("t", data=t_vec)
        f.create_dataset("u", data=u)


def heave_input_step(t):
    thrust_array = np.zeros(6, float)

    thrust_array[4] = 0
    if t < 4:
        thrust_array[2] = 120
    if t > 10:
        thrust_array[2] = 0
    return thrust_array


def get_input_range(sim_time):
    return [
        [(0, min(10, sim_time)), (0, 0)],
        [(0, min(10, sim_time)), (0, 0)],
        [(0, min(10, sim_time)), (0, 0)],
        [(0, min(1, sim_time)), (-1, 1)],
        [(0, min(1, sim_time)), (-1, 1)],
        [(0, min(1, sim_time)), (-1, 1)],
    ]


def ramp_xyz(t):
    arr = np.zeros(6)
    arr[0:3] = t * 1
    return arr


def main():
    timestep = 0.01
    sim_time = 50
    t_vec = np.arange(0, sim_time, timestep)

    rov = models.BlueRov2Heavy()

    n = len(t_vec)
    v = np.zeros((n, 6), float)
    eta = np.zeros((n, 6), float)
    v[0, :] = [0, 0, 0, 0, 0, 0]
    eta[0, :] = [0, 0, 0, 0, 0, 0]

    inputs = np.zeros((n, 6), float)

    inn = MultiDOFStep(get_input_range(sim_time))

    for i, t in enumerate(t_vec[1:], start=1):
        inputs[i, :] = inn.get_input_vec(t)
        v[i, :] = rov.solve(v[i - 1, :], eta[i - 1, :], inputs[i, :], timestep)
        eta[i, :] = utils.attitudeEuler(eta[i - 1, :], v[i, :], timestep)

    plotting.plot_ned_state(t_vec, eta)
    plotting.plot_ned_state(t_vec, v)
    plotting.plot_input(t_vec, inputs)
    plt.show()

    inn = input("Save [y/N]")
    if inn.lower() == "y":
        filename = input("filename?")
        save_data(rov, v, eta, t_vec, inputs, filename=filename)


if __name__ == "__main__":
    main()
