import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# === Load data ===
def answer(filepath):
    file_path = "/home/chorus/pullpointkostis_inclusions27e-08.npy"
    pull = np.load(file_path, allow_pickle=True)  # shape: (nt, ny, nx) if array, or list of 2D arrays

    print(f"Loaded pull data: {len(pull)} frames")

    # === Compute statistics ===
    means = []
    variances = []

    for frame in pull:
        means.append(np.mean(frame))
        variances.append(np.var(frame))

    means = np.array(means)
    variances = np.array(variances)

    # === Save as dataframe ===
    df = pd.DataFrame({
        "frame": np.arange(len(pull)),
        "mean": means,
        "variance": variances
    })

    frames = df["frame"].values
    variances = df["variance"].values

    df.to_csv("pull_statistics.csv", index=False)
    print("Saved results to pull_statistics.csv")

    # === Plot evolution ===
    plt.figure()
    plt.plot(df["frame"], df["mean"], label="Mean")
    plt.plot(df["frame"], df["variance"], label="Variance")
    plt.xlabel("Frame index (time step)")
    plt.ylabel("Value")
    plt.legend()
    plt.title("Evolution of Mean and Variance over time")
    plt.grid(True)
    plt.show()

    # === Optional: log-log plot of variance ===
    plt.figure()
    plt.loglog(df["frame"][1:], df["variance"][1:], marker="o", linestyle="-")
    ref_x = frames[1:]
    ref_y = variances[1] * (ref_x / ref_x[0]) ** (-1)  # normalize to start at same point
    plt.loglog(ref_x, ref_y, "k--", label="Slope -1")
    plt.xlabel("Frame index")
    plt.ylabel("Variance")
    

    plt.title("Variance (log-log scale)")
    plt.grid(True, which="both")
    plt.show()