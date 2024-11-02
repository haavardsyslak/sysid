import numpy as np
from dataclasses import dataclass
import scipy
from tqdm import tqdm


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
        self.hydro_params = None

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

        self.input_array[self.idx:(idx), :] = u
        self.idx = idx
        return self.fx(x, t, u)

    def solve(self, x0, u, t_vec, hydro_params, dt=None):
        if dt is None:
            dt = t_vec[1] - t_vec[0]
        ode_solver = scipy.integrate.ode(self.dx)
        # ode_solver.set_integrator("dopri5")
        ode_solver.set_initial_value(x0)
        self.input_array = np.zeros((len(t_vec), len(self.model.dtpye_input)))
        self.hydro_param_array = np.zeros((len(t_vec), len(self.model.hydro_params)))
        self.input = u
        self.t_vec = t_vec

        steps = len(t_vec)
        res = []
        updated = False
        with tqdm(total=steps, desc="Generating data", unit="step", ncols=80) as progress:
            while ode_solver.successful() and ode_solver.t < t_vec[-1]:
                ode_solver.integrate(ode_solver.t + dt)
                norm = np.linalg.norm(ode_solver.y[6:])

                ode_solver.y[6:] /= norm
                res.append(ode_solver.y.copy())
                progress.update(1)
                if not updated and ode_solver.t >= 40 and not updated:
                    # self.model.Nr = -1
                    updated = True

        res = np.array(res)
        return SimulationResults(res, self.input_array)
