import numpy as np
from utils.fossen import gvect_quat, gvect, m2c
import utils.attitude as attitude


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

        self.Xdu = -6.36
        self.Ydv = -7.12
        self.Zdw = -18.68
        self.Kdp = -0.189
        self.Mdq = -0.135
        self.Ndr = -0.222

        self.Xu = -13.7
        self.Yv = -6.0  # self.Yv = -0.0
        self.Zw = -33.0
        self.Kp = -0.9  # -0.9  # self.Kp = -0.0
        self.Mq = -0.8
        self.Nr = 0

        self.Xabs_u = -141.0
        self.Yabs_v = -217.0
        self.Zabs_w = -190.0
        self.Kabs_p = -1.19
        self.Mabs_q = -0.47
        self.Nabs_r = -1.5

        self._M = None

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
            [[self.m, 0, 0, 0, 0, 0],
             [0, self.m, 0, 0, 0, 0],
             [0, 0, self.m, 0, 0, 0],
             [0, 0, 0, self.Ix, 0, 0],
             [0, 0, 0, 0, self.Iy, 0],
             [0, 0, 0, 0, 0, self.Iz]]
        )
        MA = -np.diag([self.Xdu, self.Ydv, self.Zdw, self.Kdp, self.Mdq, self.Ndr])
        self._M = MRB + MA
        return self._M

    def get_D(self, v):
        D = -np.diag((self.Xu, self.Yv, self.Zw, self.Kp, self.Mq, self.Nr))
        Dn = -np.diag([self.Xabs_u * abs(v[0]),
                       self.Yabs_v * abs(v[1]),
                       self.Zabs_w * abs(v[2]),
                       self.Kabs_p * abs(v[3]),
                       self.Mabs_q * abs(v[4]),
                       self.Nabs_r * abs(v[5])])

        return D + Dn


