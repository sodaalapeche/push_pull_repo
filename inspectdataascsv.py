import numpy as np
import pandas as pd

DATA_PATH = ("/home/chorus/Downloads/data2(4).npy")
CSV_OUT = "/home/chorus/results_inspect.csv"

results = np.load(DATA_PATH, allow_pickle=True)

rows = []

for run_idx, res in enumerate(results):
    label = res.get("label", "")
    root  = res.get("root", "")
    Ta    = res.get("Ta", np.nan)
    Sigma_max = res.get("Sigma_max", np.nan)
    xc = res.get("xc_pix", np.nan)
    yc = res.get("yc_pix", np.nan)

    time = res["time"]
    mean = res["mean"]
    var  = res["var"]
    sigma_m = res.get("sigma_m", np.full_like(time, np.nan))

    for i, t in enumerate(time):
        rows.append({
            "run_idx": run_idx,
            "root": root,
            "label": label,
            "Ta": Ta,
            "time": t,
            "mean": mean[i] if i < len(mean) else np.nan,
            "var": var[i] if i < len(var) else np.nan,
            "sigma_m": sigma_m[i] if i < len(sigma_m) else np.nan,
            "Sigma_max": Sigma_max,
            "xc_pix": xc,
            "yc_pix": yc,
        })

# Convert to DataFramew
df = pd.DataFrame(rows)

# Save to CSV
df.to_csv(CSV_OUT, index=False)
print(f"Saved results to {CSV_OUT}")