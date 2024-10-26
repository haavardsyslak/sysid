from datetime import datetime
import h5py
import numpy as np
import os


def save_data(rov, x, u, t_vec, filename="", path="./data", hyrdodyn_coefficients=None) -> None:
    time = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
    if not os.path.exists(path):
        os.makedirs(path)

    filename = f"./data/{type(rov).__name__}_{filename}_{time}.h5"

    with h5py.File(filename, "w") as f:
        f.create_dataset("x", data=x)
        f.create_dataset("t", data=t_vec)
        f.create_dataset("u", data=u)
        if hyrdodyn_coefficients is not None:
            f.create_dataset("gt", data=hyrdodyn_coefficients)


def load_data(filepath: str):
    data = {}

    with h5py.File(filepath, "r") as f:
        for key in f.keys():
            data[key] = np.array(f[key])

    return data

