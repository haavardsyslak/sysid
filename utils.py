import numpy as np
import math
import scipy

# Several of the functions here are copied from the Python vehicle simulator
# Github: https://github.com/cybergalactic/PythonVehicleSimulator


def attitudeEuler(eta, nu, sampleTime):
    """
    eta = attitudeEuler(eta,nu,sampleTime) computes the generalized
    position/Euler angles eta[k+1]
    """

    p_dot = np.matmul(Rzyx(eta[3], eta[4], eta[5]), nu[0:3])
    v_dot = np.matmul(Tzyx(eta[3], eta[4]), nu[3:6])

    # Forward Euler integration
    eta[0:3] = eta[0:3] + sampleTime * p_dot
    eta[3:6] = eta[3:6] + sampleTime * v_dot

    return eta


def euler_to_quat(euler):
    rot = scipy.spatial.transform.Rotation.from_euler("zyx", euler)
    quaternion = rot.as_quat()
    return np.hstack((quaternion[3], quaternion[:3]))


def J(phi, theta, psi):
    return np.block([[Rzyx(phi, theta, psi), np.zeros(((3, 3)))],
                     [np.zeros((3, 3)), Tzyx(phi, theta)]])


def Rq(q):
    n, e1, e2, e3 = q

    return np.array([[1 - 2 * (e2**2 + e3**2), 2 * (e1 * e2 - e3 * n), 2 * (e1 * e3 + e2 * n)],
                     [2 * (e1 * e2 + e3 * n), 1 - 2 * (e1**2 + e3**2), 2 * (e2 * e3 - e1 * n)],
                     [2 * (e1 * e3 - e2 * n), 2 * (e2 * e3 + e1 * n), 1 - 2 * (e1**2 + e2**2)]])


def Tq(q):
    n, e1, e2, e3 = q

    return 1 / 2 * np.array([[-e1, -e2, -e3],
                             [n, -e3, e2],
                             [e3, n, -e1],
                             [-e2, e1, n]])


def Rzyx(phi, theta, psi):
    """
    R = Rzyx(phi,theta,psi) computes the Euler angle rotation matrix R in SO(3)
    using the zyx convention
    """

    cphi = math.cos(phi)
    sphi = math.sin(phi)
    cth = math.cos(theta)
    sth = math.sin(theta)
    cpsi = math.cos(psi)
    spsi = math.sin(psi)

    R = np.array([
        [cpsi * cth, -spsi * cphi + cpsi * sth * sphi, spsi * sphi + cpsi * cphi * sth],
        [spsi * cth, cpsi * cphi + sphi * sth * spsi, -cpsi * sphi + sth * spsi * cphi],
        [-sth, cth * sphi, cth * cphi]])

    return R


def Tzyx(phi, theta):
    """
    T = Tzyx(phi,theta) computes the Euler angle attitude
    transformation matrix T using the zyx convention
    """

    cphi = math.cos(phi)
    sphi = math.sin(phi)
    cth = math.cos(theta)
    sth = math.sin(theta)

    try:
        T = np.array([
            [1, sphi * sth / cth, cphi * sth / cth],
            [0, cphi, -sphi],
            [0, sphi / cth, cphi / cth]])

    except ZeroDivisionError:
        print("Tzyx is singular for theta = +-90 degrees.")

    return T


def gvect_quat(B, W, q, rg, rb):
    """
    g = gvect(W,B,q,r_bg,r_bb) computes the 6x1 vector of restoring
    forces about an arbitrarily point CO for a submerged body where q is
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

    if (len(nu) == 6):  # 6-DOF model 

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

    else:   # 3-DOF model (surge, sway and yaw)
        # C = [ 0             0            -M(2,2)*nu(2)-M(2,3)*nu(3)
        #      0             0             M(1,1)*nu(1)
        #      M(2,2)*nu(2)+M(2,3)*nu(3)  -M(1,1)*nu(1)          0  ]
        C = np.zeros((3, 3))
        C[0, 2] = -M[1, 1] * nu[1] - M[1, 2] * nu[2]
        C[1, 2] = M[0, 0] * nu[0]
        C[2, 0] = -C[0, 2]
        C[2, 1] = -C[1, 2]

    return C


def Smtrx(a):
    """
    S = Smtrx(a) computes the 3x3 vector skew-symmetric matrix S(a) = -S(a)'.
    The cross product satisfies: a x b = S(a)b.
    """

    S = np.array([
        [0, -a[2], a[1]],
        [a[2], 0, -a[0]],
        [-a[1], a[0], 0]])

    return S


def quats_to_euler(q_array):
    euler = np.zeros((len(q_array), 3))
    for i, q in enumerate(q_array):
        euler[i, :] = quat_to_euler(q)

    return euler


def quat_to_euler(q):
    n, e1, e2, e3 = q
    phi = np.arctan2(2 * (e2 * e3 + e1 * n), 1 - 2 * (e1**2 + e2**2))
    theta = - np.arcsin(2 * (e1 * e3 - e2 * n))
    psi = np.arctan2(2 * (e1 * e2 + e3 * n), 1 - 2 * (e2**2 + e3**2))

    return np.array([phi, theta, psi])

