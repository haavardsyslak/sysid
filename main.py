import models
import numpy as np
import matplotlib.pyplot as plt


def heave_input_step(t):
    thrust_array = np.zeros(8)
    if t >= 5 and t <= 10:
        thrust_array[4] = 0
        thrust_array[5] = 0
        thrust_array[6] = 0
        thrust_array[7] = 0

    return thrust_array


def main():
    t = np.arange(0, 15, 0.01)
    print(t)
    tau = np.genfromtxt("tau.csv", delimiter=",")
    x0 = np.zeros(12)
    rov = models.BlueRov2()

    v, eta = rov.solve(x0, t, tau)

    print(np.shape(eta[:, 2]))
    plt.plot(eta[:, 2])
    plt.show()


if __name__ == "__main__":
    main()
