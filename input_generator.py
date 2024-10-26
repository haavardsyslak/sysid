import numpy as np


class RandomStepGenerator:
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
        # ranges = [(min_on_time, max_on_time), (min_amp, max_amp)]

        self.generators = []

        for i, val in enumerate(ranges):
            self.generators.append(RandomStepGenerator(val[0], val[1]))

    def get_input_vec(self, t):
        vec = np.zeros(6)
        for i in range(6):
            vec[i] = self.generators[i].get_amplitude(t)

        return vec
