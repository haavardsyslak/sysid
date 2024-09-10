from typing import List
import numpy as np
from numpy import sin, cos, tan
import scipy


class BlueRov2:

    def __init__(self):
        self.input = lambda x: np.zeros(6)
        self.m = 13.5
        self.rho = 1000
        self.Ix = 0.26
        self.Iy = 0.23
        self.Iz = 0.37

        self.Xdu = 6.36
        self.Ydv = 7.12
        self.Zdw = 18.68
        self.Kdp = 0.189
        self.Mdq = 0.135
        self.Ndr = 0.222

        self.Xu = 13.7
        self.Xabs_u = 141.0
        self.Yv = 0
        self.Yabs_v = 217.0
        self.Zw = 33.0
        self.Zabs_w = 190
        self.Kp = 0
        self.Kabs_p = 1.19
        self.Mq = 0.8
        self.Mabs_q = 0.4
        self.Nr = 0
        self.Nabs_r = 1.5

        self.displaced_volume = 0.0134

        self.MRB = np.array(
            [[self.m, 0, 0, 0, 0, 0],
             [0, self.m, 0, 0, 0, 0],
             [0, 0, self.m, 0, 0, 0],
             [0, 0, 0, self.Ix, 0, 0],
             [0, 0, 0, 0, self.Iy, 0],
             [0, 0, 0, 0, 0, self.Iz]]
        )

        self.MA = -np.diag([6.36, 7.12, 18.68, 0.189, 0.135, 0.222])

        self.M = self.MA + self.MRB
        self.i = 0

    def system(self, x, t, u):
        print(t)
        v = x[:6]
        eta = x[6:]

        C = self.get_C(v)
        D = self.get_D(v)
        g = self.get_g(v)
        J = self.get_J(eta)
        tau = u[self.i]
        d_v = np.linalg.inv(self.M) @ (-C @ v - D @ v - g + tau)
        d_eta = J @ v

        return np.concatenate([d_v, d_eta])

    def solve(self, x0, t, u):
        self.i = 0
        x = scipy.integrate.odeint(self.system, x0, t, args=tuple([u]))
        v = x[:, :6]
        eta = x[:, 6:]

        return v, eta

    def get_J(self, vec):
        phi, theta, psi = vec[3:]
        J = np.zeros((6, 6))
        J[0, 0] = cos(psi) * cos(theta)
        J[0, 1] = -sin(psi) * cos(phi) + cos(psi) * sin(theta) * sin(psi)
        J[0, 2] = sin(psi) * sin(phi) + cos(psi) * cos(theta) * sin(theta)

        J[1, 0] = sin(psi) * cos(theta)
        J[1, 1] = cos(psi) * cos(phi) + sin(psi) * sin(theta) * sin(psi)
        J[1, 2] = -cos(psi) * sin(phi) + sin(theta) * sin(psi) * cos(theta)

        J[2, 0] = -sin(theta)
        J[2, 1] = cos(theta) * sin(psi)
        J[2, 2] = cos(theta) * cos(psi)

        J[3, 3] = 1
        J[3, 4] = sin(phi) * tan(theta)
        J[3, 5] = cos(phi) * tan(theta)

        J[4, 3] = 0
        J[4, 4] = cos(phi)
        J[4, 5] = -sin(phi)

        J[5, 3] = 0
        J[5, 4] = sin(phi) / cos(theta)
        J[5, 5] = cos(phi) / cos(theta)

        return J

    def get_C(self, vec):
        u, v, w, p, q, r = vec

        CRB = np.array(
            [
                [0, 0, 0, 0, self.m * w, -self.m * v],
                [0, 0, 0, -self.m * w, 0, self.m * u],
                [0, 0, 0, self.m * v, -self.m * u, 0],
                [0, self.m * w, -self.m * v, 0, -self.Iz * r, self.Iy * q],
                [-self.m * w, 0, self.m * u, self.Iz * r, 0, -self.Ix * p],
                [self.m * v, -self.m * u, 0, -self.Iy * q, self.Ix * p, 0],
            ]
        )

        CA = np.array(
            [
                [0, 0, 0, 0, self.Zdw * w, self.Ydv * v],
                [0, 0, 0, self.Zdw * w, 0, self.Xdu * u],
                [0, 0, 0, -self.Ydv * v, self.Xdu * u, 0],
                [0, -self.Zdw * w, self.Ydv * v, 0, -self.Ndr * r, self.Mdq * q],
                [self.Zdw * w, 0, -self.Xdu * u, self.Ndr * r, 0, -self.Kdp * p],
                [-self.Ydv * v, self.Xdu * u, 0, -self.Mdq * q, self.Kdp * p, 0],
            ]
        )

        return CRB + CA

    def get_D(self, vec):
        u, v, w, p, q, r = vec
        D = -np.diag([self.Xu, self.Yv, self.Zw, self.Kp, self.Mq, self.Nr])
        Dn = -np.diag(
            [
                self.Xabs_u * abs(u),
                self.Yabs_v * abs(v),
                self.Zabs_w * abs(w),
                self.Kabs_p * abs(p),
                self.Mabs_q * abs(q),
                self.Nabs_r * abs(r),
            ]
        )

        return D + Dn

    def get_g(self, eta):
        x, y, z, phi, theta, _ = eta
        W = self.m * 9.81
        B = self.rho * 9.81 * self.displaced_volume

        return np.array(
            [
                (W - B) * sin(theta),
                -(W - B) * cos(theta) * sin(phi),
                -(W - B) * cos(theta) * cos(phi),
                y * B * cos(theta) * cos(phi) - z * B * cos(theta) * sin(phi),
                -z * B * sin(theta) - x * B * cos(theta) * cos(phi),
                x * B * cos(theta) * sin(phi) + y * B * sin(theta),
            ]
        )

    def get_thrust_vec(self, V):
        f = ThrusterModel.get_thrust
        F = np.array(
            [f(V[0]), f(V[1]), f(V[2]), f(V[3]), f(V[4]), f(V[5]), f(V[6]), f(V[7])]
        )

        def J(x):
            return np.array([[cos(x), -sin(x), 0], [sin(x), cos(x), 0], [0, 0, 1]])

        T = np.zeros((6, 8))

        alpha = [0, 5.05, 1.91, np.pi]
        beta = [0, np.pi / 2, (3 * np.pi) / 4, np.pi]
        gamma = [0, 4.15, 1.01, np.pi]

        v1 = np.array([0.156, 0.111, 0.085])
        v2 = np.array([1 / np.sqrt(2), -1 / np.sqrt(2), 0])
        v3 = np.array([0.120, 0.218, 0])

        for i, (a, b) in enumerate(zip(alpha, beta)):
            e = J(a) @ v2
            r = J(a) @ v1
            # The shape of 3 is (3,), and e should become the first 3 rows of the column i in the
            # T matrix. How ??

            T[:3, i] = e
            T[3:, i] = np.cross(e, r)
            print(i)

        e = np.array([0, 0, -1])
        for i, g in enumerate(gamma):
            r = J(g) @ v3
            T[:3, i + 4] = e
            T[3:, i + 4] = np.cross(e, r)

        print(T)
        print(T.T)
        tf = scipy.signal.TransferFunction([6136, 108700], [1, 89, 9258, 108700])
        K = np.diag([tf, tf, tf, tf, tf, tf, tf, tf])

        # return T.T @ K #@ F


class ThrusterModel:
    @staticmethod
    def get_thrust(V):
        return -140.3 * V**9 + 389.9 * V**7 - 404.1 * V**5 + 176.0 * V**3 + 8.9 * V



