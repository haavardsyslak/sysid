import numpy as np
import math
import scipy
from scipy.spatial.transform import Rotation



def quaternion_prod(q1, q2):
    """
    Computes the product of two quaternions q1 and q2.

    Parameters:
    q1, q2 : array-like, shape (4,)
        Input quaternions represented as [w, x, y, z], 
        where w is the scalar part and x, y, z are the vector parts.

    Returns:
    product : ndarray, shape (4,)
        The resulting quaternion from the multiplication q1 * q2.
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return np.array([w, x, y, z])


def quaternion_error(q1, q2):
    """Returns a new quaternion representing
    the attitude error between the two quaternion"""
    q2_conj = np.zeros_like(q2)
    q2_conj[0] = q2[0]
    q2_conj[1:] = -q2[1:]
    return quaternion_prod(q1, q2_conj)


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
                     [2 * (e1 * e2 + e3 * n), 1 - 2
                      * (e1**2 + e3**2), 2 * (e2 * e3 - e1 * n)],
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
        [cpsi * cth, -spsi * cphi + cpsi * sth
            * sphi, spsi * sphi + cpsi * cphi * sth],
        [spsi * cth, cpsi * cphi + sphi * sth
            * spsi, -cpsi * sphi + sth * spsi * cphi],
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

def to_scipy_quat(q):
    return np.array([q[1], q[2], q[3], q[0]])

def vec_to_quat(v):
    # ang = np.linalg.norm(v) / 2
    # e = v / np.linalg.norm(v)
    #
    # q = np.zeros(4)
    # q[0] = np.cos(ang)
    # q[1:] = e * np.sin(ang)
    #
    # return q
    
    q = scipy.spatial.transform.Rotation.from_rotvec(v).as_quat(scalar_first=True)
    return q


def mean_quat(quats):
    quats = np.array([q / np.linalg.norm(q) for q in quats])

    # Compute the covariance matrix
    C = np.zeros((4, 4))
    for q in quats:
        C += np.outer(q, q)
    C /= len(quats)

    # Perform eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(C)

    # The eigenvector with the largest eigenvalue is the mean quaternion
    mean_q = eigenvectors[:, np.argmax(eigenvalues)]

    # Normalize the resulting quaternion to ensure it is unit
    mean_q /= np.linalg.norm(mean_q)
    return mean_q
