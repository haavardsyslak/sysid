import numpy as np
from numpy import sin, cos
import scipy

import utils


class BlueRov2:
    def __init__(self):
        m = 10
        self.B = 100.06
        self.W = 98.1
        self.rg = np.array([0, 0, 0])  # center of gravity
        self.rb = np.array([0, 0, 0.02])  # center of buoyancy
        self.Imat = np.diag([0.16, 0.16, 0.16])

        self.Xdu = -5.5
        self.Ydv = -12.7
        self.Zdw = -14.57
        self.Kdp = -0.12
        self.Mdq = -0.12
        self.Ndr = -0.12

        self.Xu = -4.03
        self.Yv = -6.22
        self.Zw = -5.18
        self.Kp = -0.07
        self.Mq = -0.07
        self.Nr = -0.07

        self.Xabs_u = -18.18
        self.Yabs_v = -21.66
        self.Zabs_w = -36.99
        self.Kabs_p = -1.55
        self.Mabs_q = -1.55
        self.Nabs_r = -1.55

        MA = -np.diag([self.Xdu, self.Ydv, self.Zdw, self.Kdp, self.Mdq, self.Ndr])
        MRB = np.block([
            [m * np.eye(3), -m * utils.Smtrx(self.rg)],
            [m * utils.Smtrx(self.rb), self.Imat]
        ])

        self.M = MA + MRB
        self.Minv = np.linalg.inv(self.M)

        self.D = -np.diag([self.Xu, self.Yv, self.Zw, self.Kp, self.Mq, self.Nr])

        self.Thurst_alloc = np.array([[0.7071, 0.7071, -0.7071, -0.7071, 0, 0],
                                      [-0.7071, 0.7071, -0.7071, 0.7071, 0, 0],
                                      [0, 0, 0, 0, -1, -1],
                                      [0, 0, 0, 0, 0.115, -0.115],
                                      [0, 0, 0, 0, 0, 0],
                                      [-0.1773, 0.1773, -0.1773, 0.1773, 0, 0]])

    def solve(self, v, eta, u, timestep):
        C = utils.m2c(self.M, v)
        D = self.D - np.diag([self.Xabs_u * abs(v[0]),
                              self.Yabs_v * abs(v[1]),
                              self.Zabs_w * abs(v[2]),
                              self.Kabs_p * abs(v[3]),
                              self.Mabs_q * abs(v[4]),
                              self.Nabs_r * abs(v[5])])
        g = utils.gvect(self.W, self.B, eta[4], eta[3], self.rg, self.rb)
        # u = self.Thurst_alloc @ u

        dv = np.matmul(self.Minv, u - C @ v - D @ v - g)

        return v + dv * timestep


class BlueRov2Heavy:

    def __init__(self):
        self.m = 13.5
        self.rho = 1000
        self.W = self.m * 9.81
        # self.W  # self.m * self.rho * self.displaced_volume
        self.B = self.W

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
        self.Xabs_u = -141.0
        self.Yv = 0  # self.Yv = -0.0
        self.Yabs_v = -217.0
        self.Zw = -33.0
        self.Zabs_w = -190.0
        self.Kp = -0.9  # self.Kp = -0.0
        self.Kabs_p = -1.19
        self.Mq = -0.8
        self.Mabs_q = -0.47
        self.Nr = 0
        self.Nabs_r = -1.5

        self.r_cb = np.array([0, 0, -0.01], float)
        self.r_cg = np.array([0.0, 0.0, 0.0], float)

        self.displaced_volume = 0.0134

        MRB = np.block([[np.eye(3) * self.m, np.zeros((3, 3))],
                        [np.zeros((3, 3)), self.Imat]])
        MRB = np.array(
            [[self.m, 0, 0, 0, 0, 0],
             [0, self.m, 0, 0, 0, 0],
             [0, 0, self.m, 0, 0, 0],
             [0, 0, 0, self.Ix, 0, 0],
             [0, 0, 0, 0, self.Iy, 0],
             [0, 0, 0, 0, 0, self.Iz]]
        )

        # self.MA = -np.diag([6.36, 7.12, 18.68, 0.189, 0.135, 0.222])
        MA = -np.diag([self.Xdu, self.Ydv, self.Zdw, self.Kdp, self.Mdq, self.Ndr])
        self.M = MRB + MA
        self.Minv = np.linalg.inv(self.M)

        self.D = -np.diag([self.Xu, self.Yv, self.Zw, self.Kp, self.Mq, self.Nr])

    def solve(self, v, eta, u, timestep):
        # Currently u is set to be the foces for each DOF. For future this should be an array of
        # thruster voltages, then trasnformed to forces in body frame.

        g = utils.gvect(self.W, self.B, eta[4], eta[3], self.r_cg, self.r_cb)
        C = utils.m2c(self.M, v)
        D = self.D - np.diag([self.Xabs_u * abs(v[0]),
                              self.Yabs_v * abs(v[1]),
                              self.Zabs_w * abs(v[2]),
                              self.Kabs_p * abs(v[3]),
                              self.Mabs_q * abs(v[4]),
                              self.Nabs_r * abs(v[5])])

        Minv = np.linalg.inv(self.M)
        D = np.matmul(D, v)
        C = np.matmul(C, v)
        # u = self.get_thrust_vec(u)
        dv = np.matmul(Minv, u - D - C - g)

        v = v + (timestep * dv)

        return v

        def get_thrust_vec(self, V):
            f = ThrusterModel.get_thrust
            F = np.array(
                [f(V[0]), f(V[1]), f(V[2]), f(V[3]), f(V[4]), f(V[5]), f(V[6]), f(V[7])]
            )

            def J(x):
                return np.array([[cos(x), -sin(x), 0], [sin(x), cos(x), 0], [0, 0, 1]])

            T = np.zeros((6, 8), float)

            alpha = [0, 5.05, 1.91, np.pi]
            beta = [0, np.pi / 2, (3 * np.pi) / 4, np.pi]
            gamma = [0, 4.15, 1.01, np.pi]

            v1 = np.array([0.156, 0.111, 0.085])
            v2 = np.array([1 / np.sqrt(2), -1 / np.sqrt(2), 0])
            v3 = np.array([0.120, 0.218, 0])

            for i, (a, b) in enumerate(zip(alpha, beta)):
                e = J(a) @ v2
                r = J(a) @ v1

                T[:3, i] = e
                T[3:, i] = np.cross(e, r)

            e = np.array([0, 0, -1])
            for i, g in enumerate(gamma):
                r = J(g) @ v3
                T[:3, i + 4] = e
                T[3:, i + 4] = np.cross(e, r)

            # print(F)
            tf = scipy.signal.TransferFunction([6136, 108700], [1, 89, 9258, 108700])
            K = np.diag([tf, tf, tf, tf, tf, tf, tf, tf])

            return T @ F


class ThrusterModel:
    @staticmethod
    def get_thrust(V):
        return -140.3 * V**9 + 389.9 * V**7 - 404.1 * V**5 + 176.0 * V**3 + 8.9 * V



