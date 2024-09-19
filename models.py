import numpy as np
from numpy import sin, cos, tan
import scipy

from abc import ABC, abstractmethod

import utils


class Model(ABC):
    @abstractmethod
    def f(self, x, u):
        pass

    @abstractmethod
    def F(self, x, u):
        pass

    @abstractmethod
    def H(self, x):
        pass


class BlueRov2Heavy(Model):
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

        D = np.matmul(D, v)
        C = np.matmul(C, v)
        # u = self.get_thrust_vec(u)
        dv = np.matmul(self.Minv, u - D - C - g)

        v = v + (timestep * dv)

        return v

    def f(self, x, u):
        v = x[:6]
        phi, theta, psi = x[6:]
        g = utils.gvect(self.W, self.B, theta, phi, self.r_cg, self.r_cb)
        C = utils.m2c(self.M, v)
        D = self.D - np.diag([self.Xabs_u * abs(v[0]),
                              self.Yabs_v * abs(v[1]),
                              self.Zabs_w * abs(v[2]),
                              self.Kabs_p * abs(v[3]),
                              self.Mabs_q * abs(v[4]),
                              self.Nabs_r * abs(v[5])])
        D = np.matmul(D, v)
        C = np.matmul(C, v)
        dv = np.matmul(self.Minv, u - D - C - g)
        # d_eta = np.matmul(utils.J(phi, theta, psi), v[3:])
        d_eta = np.matmul(utils.Tzyx(phi, theta), v[3:])
        return np.hstack((dv, d_eta))

    def F(self, x, u):
        u, v, w, p, q, r, phi, theta, psi = x
        xb, yb, zb = self.r_cb
        xg, yg, zg = self.r_cg

        F = np.zeros((9, 9))
        F[0, 0] = self.Xu + 2 * self.Xabs_u * abs(u)
        F[0, 1] = (self.m - self.Yv) * r
        F[0, 2] = (self.Zdw - self.m) * q
        F[0, 3] = 0
        F[0, 4] = (self.Zdw - self.m) * w
        F[0, 5] = (self.m - self.Ydv) * w
        F[0, 6] = (self.B - self.W) * cos(phi)
        F[0, 7] = 0
        F[0, 8] = 0

        F[1, 0] = (self.Xdu - self.m) * r
        F[1, 1] = self.Yv + 2 * self.Yabs_v * abs(v)
        F[1, 2] = (self.m - self.Zdw) * p
        F[1, 3] = (self.m - self.Zdw) * w
        F[1, 4] = 0
        F[1, 5] = (self.Xdu - self.m) * u
        F[1, 6] = (self.B - self.W) * sin(phi) * sin(theta)
        F[1, 7] = (self.W - self.B) * cos(phi) * cos(theta)
        F[1, 8] = 0

        F[2, 0] = (self.m - self.Xdu) * q
        F[2, 1] = (self.Ydv - self.m) * p
        F[2, 2] = self.Zw + 2 * self.Zabs_w * abs(w)
        F[2, 3] = (self.Ydv - self.m) * v
        F[2, 4] = (self.m - self.Xdu) * u
        F[2, 5] = 0
        F[2, 6] = (self.B - self.W) * sin(phi) * cos(theta)
        F[2, 7] = (self.B - self.W) * sin(theta) * cos(phi)
        F[2, 8] = 0

        F[3, 0] = 0
        F[3, 1] = (self.Zdw - self.Ydv) * w
        F[3, 2] = (self.Zdw - self.Ydv) * v
        F[3, 3] = self.Kp * 2 * self.Kabs_p * abs(p)
        F[3, 4] = (self.Iy - self.Iz - self.Mdq + self.Ndr) * r
        F[3, 5] = (self.Iy - self.Iz - self.Mdq + self.Ndr) * q
        F[3, 6] = self.B * (yb * cos(theta) - zb * sin(theta)) * sin(theta)
        F[3, 7] = self.B * (yb * sin(theta) + zb * cos(theta)) * cos(phi)
        F[3, 8] = 0

        F[4, 0] = (self.Xdu - self.Zdw) * w
        F[4, 1] = 0
        F[4, 2] = (self.Xdu - self.Zdw) * u
        F[4, 3] = (self.Iz - self.Ix + self.Kdp - self.Ndr) * r
        F[4, 4] = self.Mq + 2 * self.Mabs_q * abs(q)
        F[4, 5] = (self.Iz - self.Ix + self.Kdp - self.Ndr) * p
        F[4, 6] = self.B * zb * cos(phi) - xb * sin(phi) * cos(theta)
        F[4, 7] = -xb * sin(theta) * cos(phi)
        F[4, 8] = 0

        F[5, 0] = (self.Ydv - self.Xdu) * v
        F[5, 1] = (self.Ydv - self.Xdu) * u
        F[5, 2] = 0
        F[5, 3] = (self.Ix - self.Iy - self.Kdp - self.Mdq) * q
        F[5, 4] = (self.Ix - self.Iy - self.Kdp + self.Mdq) * p
        F[5, 5] = self.Nr + 2 * self.Nabs_r * abs(r)
        F[5, 6] = self.B * (-2 * xb * cos(theta)**2 + xb - yb * cos(phi))
        F[5, 7] = 0
        F[5, 8] = 0

        F[6, 0:3] = 0
        F[6, 3] = 1
        F[6, 4] = sin(theta) * tan(theta)
        F[6, 5] = cos(phi) * tan(theta)
        F[6, 6] = (q * cos(theta) - r * sin(theta)) * tan(theta)
        F[6, 7] = (q * sin(theta) + r * cos(phi)) / cos(theta)**2
        F[6, 8] = 0

        F[7, 0:4] = 0
        F[7, 4] = cos(phi)
        F[7, 5] = -sin(phi)
        F[7, 6] = -q * sin(phi) - r * cos(phi)
        F[7, 7] = 0
        F[7, 8] = 0

        F[8, 0:4] = 0
        F[8, 4] = sin(phi) / cos(theta)
        F[8, 5] = cos(phi) / cos(theta)
        F[8, 6] = (q * cos(phi) - r * sin(phi)) / cos(theta)
        F[8, 7] = (q * sin(theta) + r * cos(phi) * sin(theta)) / cos(theta) ** 2
        F[8, 8] = 0

        return F

    def H(self, x):
        return np.eye(9)

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



