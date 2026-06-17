import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv("/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/python_scripts/weekly_means/val_fkcrps_metric_sfc_2t_1.csv")
df_daily
#Rename run
run_names = {
    "aifs-subs-debug-finetuning": r"lr = $2.5 \times 10^{-5}$",
    "aifs-subs-weekly-lr-2.5e-4": r"lr = $1 \times 10^{-3}$",
    "aifs-subs-weekly-finetuning-freeze-model.processor-lr-0.625e-6": r"lr = $2.5 \times 10^{-6}$, processor frozen",
    "aifs-subs-weekly-finetuning-freeze-model.processor-encoder-lr-0.625e-6": r"lr = $2.5 \times 10^{-6}$, processor & encoder frozen",
    "aifs-subs-weekly-finetuning-freeze-model.encoder-decoder-lr-0.625e-6": r"lr = $2.5 \times 10^{-6}$, encoder & decoder frozen",
    "aifs-subs-weekly-from-scratch-lr-0.625e-6": r"lr = $2.5 \times 10^{-6}$, from scratch",
    "aifs-subs-weekly-finetuning-lr-0.625e-rolling-average-27-0": r"lr = $2.5 \times 10^{-6}$",
    "lr = 6.25e-8": r"lr = $2.5 \times 10^{-7}$",
}

df["plot_group"] = df["Run"]

fig, ax = plt.subplots(figsize=(10, 6))

# Plot one line per run
for group_name, run_df in df.groupby("plot_group"):
    run_df = run_df.sort_values("step")
    
    original_name = run_df["Run"].iloc[0]
    label = run_names.get(original_name, original_name)

    ax.plot(
        run_df["step"],
        run_df["value"],
        marker="o",
        linewidth=2,
        label=label,
    )

ax.set_xlabel("Training Step")
ax.set_ylabel("fCRPS")
ax.set_title("Validation Loss (fCRPS) on 2 meters Temperature per Weekly Mean Training experiments")
ax.grid(True, alpha=0.3)

# Put legend outside if run names are long
ax.legend(
    loc="upper right",
    fontsize=8
)

plt.tight_layout()
plt.savefig("/ec/res4/hpcperm/nld4584/Anemoi_S2S_experiment/python_scripts/weekly_means/images_MLFlow/val_fkcrps_metric_sfc_2t_1.png", dpi=200, bbox_inches='tight')