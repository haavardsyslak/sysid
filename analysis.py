import matplotlib.pyplot as plt
import numpy as np
import h5py
from ukf import UKFResults
import pandas as pd

files = {
    "one_res/BlueRov2Heavy_Xu": (-16., -13.7, 10),
    "one_res/BlueRov2Heavy_Yv": (-10., -6.0, 11),
    "one_res/BlueRov2Heavy_Zw": (-39., -33.0, 12),
    "one_res/BlueRov2Heavy_Kp": (-1.5, -0.9, 13),
    "one_res/BlueRov2Heavy_Mq": (-1.5, -0.8, 14),
    "one_res/BlueRov2Heavy_Nr": (-1., 0, 15),
    "one_res/BlueRov2Heavy_Xdu": (-10., -6.36, 16),
    "one_res/BlueRov2Heavy_Ydv": (-10., -6.0, 17),
    "one_res/BlueRov2Heavy_Zdw": (-25., -18.68, 18),
    "one_res/BlueRov2Heavy_Kdp": (-0.3, -0.189, 19),
    "one_res/BlueRov2Heavy_Mdq": (-0.3, -0.135, 20),
    "one_res/BlueRov2Heavy_Ndr": (-0.3, -0.222, 21),

}


def load_data(file):
    data = {}
    with h5py.File(file, "r") as f:
        for key in f.keys():
            data[key] = np.array(f[key])

    return data


f = "BlueRov2Heavy_Zw"


def get_y_label_from_filename(filename):
    label = filename.split("_")[-1]
    if "d" in label:
        label = label.replace("d", "_\\dot{") + "}"
    else:
        label = label[0] + "_" + label[1]
    return "$\\mathbf{" + label + "}$"


def plot_hydro_param_settling(x_bar, x, t_vec, label="", filename=""):
    gt = np.zeros((len(t_vec)))
    for i, t in enumerate(t_vec):
        if t < 40:
            gt[i] = x[1]
        else:
            gt[i] = x[0]

    # Calculate 2% bounds
    final_value = gt[-1]
    inital_value = gt[0]
    print(f"final: {final_value}, init: {inital_value}")
    lower_bound = final_value - (final_value - inital_value) * 0.05
    upper_bound = final_value + (final_value - inital_value) * 0.05

    fig, axes = plt.subplots(2, 1, sharex=False)

    axes = axes.flatten()
    # Plot the true value and the estimated value
    axes[0].plot(t_vec, gt, linestyle="--", label="True value")
    axes[0].plot(t_vec, x_bar, label="Estimated")
    # Add 2% bounds as horizontal lines
    axes[0].axhline(y=lower_bound, color='r', linestyle='--', label="2% Bounds")
    axes[0].axhline(y=upper_bound, color='r', linestyle='--')
    axes[0].legend()
    axes[0].grid(True)
    axes[0].set_ylabel(label, rotation=0)

    # Plot the error
    axes[1].grid(True)
    axes[1].plot(t_vec, x_bar - gt, label="Error")
    axes[1].set_ylabel(label, rotation=0)
    axes[1].set_xlabel("Time [s]")
    axes[1].legend()

    if filename:
        if ".pdf" not in filename:
            filename += ".pdf"
        plt.savefig(f"one_res/plot/{filename}", format="pdf")


def plot():
    for key, val in files.items():
        data = load_data(key)
        x_bar = data["x_bar"][:, val[-1]]
        label = get_y_label_from_filename(key)
        filename = key.split("/")[-1]
        plot_hydro_param_settling(x_bar, val[:2], data["t"], label=label, filename=filename)

    plt.show()


def compute_average_error_after_settling(x_bar, x, t_vec):
    gt = np.zeros((len(t_vec)))
    for i, t in enumerate(t_vec):
        if t < 40:
            gt[i] = x[1]
        else:
            gt[i] = x[0]

    # Calculate 5% bounds
    final_value = gt[-1]
    inital_value = gt[0]
    lower_bound = final_value + (final_value - inital_value) * 0.05
    upper_bound = final_value - (final_value - inital_value) * 0.05
    print(f"lower: {lower_bound}, upper: {upper_bound}")

    # Find the first index where the value is within the 5% bounds
    settling_index = np.argmax((x_bar[100:] <= upper_bound)) +100 # & (x_bar <= upper_bound))

    print(settling_index)

    # Calculate average error from the settling point onward
    if settling_index > 0:  # Ensure we found a valid settling point
        errors = np.abs(x_bar[settling_index:] - gt[settling_index:])
        average_error = np.mean(errors)
        settling_time = t_vec[settling_index]
    else:
        average_error = None  # No valid settling point
        settling_time = None

    return average_error, settling_time


# Loop through files and compute the average error for each parameter
def averages():
    results = []
    for key, val in files.items():
        data = load_data(key)
        x_bar = data["x_bar"][:, val[-1]]
        t_vec = data["t"]
        average_error, settling_time = compute_average_error_after_settling(x_bar, val[:2], t_vec)
        label = get_y_label_from_filename(key)

        results.append({
            "Parameter": label,
            "Average Error": average_error,
            "Settling Time (s)": settling_time - 40
        })

# Display the results
    import pandas as pd
    results_df = pd.DataFrame(results)
    print(results_df)

if __name__ == "__main__":
    averages()
    plot()
