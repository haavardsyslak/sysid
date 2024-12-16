import numpy as np
from utils.fossen import gvect_quat, gvect, m2c
import utils.attitude as attitude
from utils.fossen import Smtrx


class BlueRov2Heavy:
    def __init__(self):
        self.dtype_state = [
            ("u", "f8"),
            ("v", "f8"),
            ("w", "f8"),
            ("p", "f8"),
            ("q", "f8"),
            ("r", "f8"),
            ("r", "f8"),
            ("n", "f8"),
            ("e1", "f8"),
            ("e2", "f8"),
            ("e3", "f8"),
        ]

        self.dtpye_input = [
            ("X", "f8"),
            ("Y", "f8"),
            ("Z", "f8"),
            ("K", "f8"),
            ("M", "f8"),
            ("N", "f8"),
        ]

        self.m = 13.5
        self.rho = 1000
        self.displaced_volume = 0.0134
        self.W = self.m * 9.81
        # self.W  # self.m * self.rho * self.displaced_volume
        self.B = self.m * self.rho * self.displaced_volume  # self.W
        self.rg = np.array([0, 0, 0])
        self.rb = np.array([0, 0, -0.02])

        self.Ix = 0.26
        self.Iy = 0.23
        self.Iz = 0.37
        self.Imat = np.diag([self.Ix, self.Iy, self.Iz])

        self.hydro_param_idx = {
            "Xdu": 0,
            "Ydv": 1,
            "Zdw": 2,
            "Kdp": 3,
            "Mdq": 4,
            "Ndr": 5,
            "Xu": 6,
            "Yv": 7,
            "Zw": 8,
            "Kp": 9,
            "Mq": 10,
            "Nr": 11,
            "Xabs_u": 12,
            "Yabs_v": 13,
            "Zabs_w": 14,
            "Kabs_p": 15,
            "Mabs_q": 16,
            "Nabs_r": 17,
        }

        self.hydro_params = np.array(
            [
                -6.36,
                -7.12,
                -18.68,
                -0.189,
                -0.135,
                -0.222,
                -13.7,
                -6.0,
                -33.0,
                -0.9,
                -0.8,
                0,
                -141.0,
                -217.0,
                -190.0,
                -1.19,
                -0.47,
                -1.5,
            ]
        )

        self._M = None

    def fx_err_state(self, x, t, u):
        v = x[:6]
        g = gvect_quat(self.B, self.W, x[6:10], self.rg, self.rb)
        C = m2c(self.M, x[:6]) @ v
        D = self.get_D(x[:6]) @ v
        # Minv = np.linalg.inv(self.M)
        dv = np.linalg.solve(self.M, (u - g - C - D))
        # d_theta = -Smtrx(x[3:6]) @ theta
        #d_theta = attitude.Tq(x_nom[6:10]) @ v[3:6]
        q = np.zeros(4)
        omega = v[3:6]
        norm = np.linalg.norm(omega)
        if norm < 1e-6:
            print(norm)
            q[0] = 1.0
            q[1:] = 0.0
        else:
            q[0] = np.cos(norm * (0.05 / 2))
            q[1:] = (omega / norm) * np.sin(norm * (0.05 / 2))


        # d_theta = 2 * (attitude.Tq(q) @ omega)[1:]

        return np.hstack((dv, q))

    def fx(self, x, t, u):
        """State update using quaternions"""
        v = x[:6]
        q = x[6:10]
        g = gvect_quat(self.B, self.W, q, self.rg, self.rb)
        C = m2c(self.M, v) @ v
        D = self.get_D(v) @ v
        # Minv = np.linalg.inv(self.M)
        dv = np.linalg.solve(self.M, (u - g - C - D))
        dn = attitude.Tq(q) @ v[3:6]
        # dn /= np.linalg.norm(dn)

        return np.hstack((dv, dn))

    def fx_euler(self, x, t, u):
        """State update using euler angles"""
        v = x[:6]
        phi, theta, psi = x[6:]
        g = gvect(self.W, self.B, theta, phi, self.rg, self.rb)
        C = m2c(self.M, v) @ v
        D = self.get_D(v) @ v
        Minv = np.linalg.inv(self.M)
        dv = Minv @ (u - g - C - D)
        d_eta = attitude.Tzyx(phi, theta) @ v[3:6]

        return np.hstack((dv, d_eta))

    @property
    def M(self):
        MRB = np.array(
            [
                [self.m, 0, 0, 0, 0, 0],
                [0, self.m, 0, 0, 0, 0],
                [0, 0, self.m, 0, 0, 0],
                [0, 0, 0, self.Ix, 0, 0],
                [0, 0, 0, 0, self.Iy, 0],
                [0, 0, 0, 0, 0, self.Iz],
            ]
        )
        MA = -np.diag(
            [
                self.hydro_params[0],
                self.hydro_params[1],
                self.hydro_params[2],
                self.hydro_params[3],
                self.hydro_params[4],
                self.hydro_params[5],
            ]
        )
        self._M = MRB + MA
        return self._M

    def get_D(self, v):
        D = -np.diag(
            (
                self.hydro_params[6],
                self.hydro_params[7],
                self.hydro_params[8],
                self.hydro_params[9],
                self.hydro_params[10],
                self.hydro_params[11],
            )
        )

        Dn = -np.diag(
            [
                self.hydro_params[12] * abs(v[0]),
                self.hydro_params[13] * abs(v[1]),
                self.hydro_params[14] * abs(v[2]),
                self.hydro_params[15] * abs(v[3]),
                self.hydro_params[16] * abs(v[4]),
                self.hydro_params[17] * abs(v[5]),
            ]
        )

        return D + Dn

    def set_hdyroparam(self, params):
        for key, value in params.items():
            if key in self.hydro_params:
                self.hydro_params[key] = value
            else:
                raise KeyError(f"{key} is not a valid hydrodynamic parameter.")
