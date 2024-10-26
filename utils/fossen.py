import numpy as np
import math
from .attitude import Smtrx

# Several of the functions here are copied from the Python vehicle simulator
# Github: https://github.com/cybergalactic/PythonVehicleSimulator


def gvect_quat(B, W, q, rg, rb):
    """
    g = gvect(W,B,q,r_bg,r_bb) computes the 6x1 vector of restoring
    forces about an arbitrarily poin CO for a submerged body where q is
    the unit quaternions representing the attitude
    """

    n, e1, e2, e3 = q
    xg, yg, zg = rg
    xb, yb, zb = rb
    s1 = 2 * e1**2 + 2 * e2**2 - 1
    s2 = 2 * e1 * e3 - 2 * e2 * n
    s3 = 2 * e2 * e3 + 2 * e1 * n

    return np.array([
        (B - W) * s2,
        (B - W) * s3,
        -(B - W) * s1,
        W * yg * s1 - B * zb * s3 - B * yb * s1 + W * zg * s3,
        B * xg * s1 + B * zb * s2 - W * xg * s1 - W * zg * s2,
        B * xb * s3 - B * yb * s2 - W * xg * s3 + W * yg * s2,
    ])


def gvect(W, B, theta, phi, r_bg, r_bb):
    """
    g = gvect(W,B,theta,phi,r_bg,r_bb) computes the 6x1 vector of restoring
    forces about an arbitrarily point CO for a submerged body.

    Inputs:
        W, B: weight and buoyancy (kg)
        phi,theta: roll and pitch angles (rad)
        r_bg = [x_g y_g z_g]: location of the CG with respect to the CO (m)
        r_bb = [x_b y_b z_b]: location of the CB with respect to th CO (m)

    Returns:
        g: 6x1 vector of restoring forces about CO
    """

    sth = math.sin(theta)
    cth = math.cos(theta)
    sphi = math.sin(phi)
    cphi = math.cos(phi)
    xb, yb, zb = r_bb

    g = np.array([
        (W - B) * sth,
        -(W - B) * cth * sphi,
        -(W - B) * cth * cphi,
        yb * B * cth * cphi - zb * B * cth * sphi,
        -zb * B * sth - xb * cth * cphi,
        xb * B * cth * sth + yb * B * sth,
    ])

    return g


def m2c(M, nu):
    """
    C = m2c(M,nu) computes the Coriolis and centripetal matrix C from the
    mass matrix M and generalized velocity vector nu (Fossen 2021, Ch. 3)
    """

    M = 0.5 * (M + M.T)     # systematization of the inertia matrix

    M11 = M[0:3, 0:3]
    M12 = M[0:3, 3:6]
    M21 = M12.T
    M22 = M[3:6, 3:6]

    nu1 = nu[0:3]
    nu2 = nu[3:6]
    dt_dnu1 = np.matmul(M11, nu1) + np.matmul(M12, nu2)
    dt_dnu2 = np.matmul(M21, nu1) + np.matmul(M22, nu2)

    # C  = [  zeros(3,3)      -Smtrx(dt_dnu1)
    #      -Smtrx(dt_dnu1)  -Smtrx(dt_dnu2) ]
    C = np.zeros((6, 6))
    C[0:3, 3:6] = -Smtrx(dt_dnu1)
    C[3:6, 0:3] = -Smtrx(dt_dnu1)
    C[3:6, 3:6] = -Smtrx(dt_dnu2)

    return C
