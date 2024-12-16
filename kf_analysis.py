import numpy as np
import utils.attitude as attitude

def get_nis(innovation, S):
    return innovation.T @ np.linalg.inv(S) @ innovation


def get_nees(x, x_true, P):
    error = x - x_true
    return error.T @ np.linalg.inv(P) @ error

