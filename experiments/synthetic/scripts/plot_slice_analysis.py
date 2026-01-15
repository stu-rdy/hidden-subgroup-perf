import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set aesthetics
sns.set_theme(style="whitegrid")
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "figure.titlesize": 18,
    }
)


def main():
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "../../../"))
    csv_path = os.path.join(
        project_root, "results/eval_domino_bias-08/Sheet 3-Slice Analysis Summary.csv"
    )
    plot_dir = os.path.join(project_root, "results/plots")
    os.makedirs(plot_dir, exist_ok=True)

    # Load data
    df = pd.read_csv(csv_path)

    # Strip potential whitespace from columns and split
    df.columns = [c.strip() for c in df.columns]
    df["split"] = df["split"].str.strip()

    # Filter for test split
    test_df = df[df["split"] == "test"].copy()

    # Sort by accuracy
    test_df = test_df.sort_values("accuracy", ascending=True).reset_index(drop=True)
    test_df["slice_rank"] = range(1, len(test_df) + 1)

    # Calculate overall test accuracy (weighted average of slice accuracies)
    overall_acc = (test_df["accuracy"] * test_df["size"]).sum() / test_df["size"].sum()

    # Create plot
    fig, ax1 = plt.subplots(figsize=(12, 7))

    # Design choice: Highlight the poor performance tail
    # Slices sorted by accuracy (ASC), so index 0 is the lowest.
    bar_colors = []
    for i, acc in enumerate(test_df["accuracy"]):
        if i == 0:
            bar_colors.append("#EC7063")  # Muted Red (opaque)
        elif acc < overall_acc:
            bar_colors.append("#5D6D7E")  # Muted Dark Grey (opaque)
        else:
            bar_colors.append("#D5D8DC")  # Muted Light Grey (opaque)

    # Opaque bars to hide grid lines while maintaining muted look
    ax1.bar(
        test_df["slice_rank"],
        test_df["accuracy"],
        color=bar_colors,
        alpha=1.0,
        width=0.6,
        label="Slice Accuracy",
        zorder=3,
    )

    # Add a ghost secondary axis to match padding and layout of concentration plot
    ax2 = ax1.twinx()
    ax2.set_ylabel("", alpha=0)  # Hidden label to reserve space
    ax2.set_yticks([])  # No ticks
    ax2.grid(False)

    # Ensure grid is below artists
    ax1.set_axisbelow(True)
    ax1.grid(True, zorder=0)

    # Add horizontal line for overall accuracy
    ax1.axhline(
        y=overall_acc,
        color="#E74C3C",
        linestyle="--",
        linewidth=2.5,
        label="Overall test accuracy (aggregate)",
        zorder=5,
    )

    # Labels and title
    ax1.set_xlabel("Slice rank (DOMINO-discovered, sorted by accuracy)")
    ax1.set_ylabel("Slice Accuracy")
    plt.title("Slice Performance Discovery (Domino)", pad=20)

    # Standardize x-axis to match error concentration plot (1-15 with padding)
    ax1.set_xticks(range(1, len(test_df) + 1))
    ax1.set_xlim(0, len(test_df) + 1)
    ax1.set_ylim(0, 1.05)

    # Formatting y-axis as percentage
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f"{x:.0%}"))

    # Add legend
    ax1.legend(loc="lower right", frameon=True, shadow=True)

    # Annotate the "worst" slice to highlight the tail
    worst_slice = test_df.iloc[0]
    # Moving annotation to ax2 layer for consistency with concentration plot
    ax2.annotate(
        f"Worst slice: {worst_slice['accuracy']:.1%}\n(discovered subgroup)",
        xy=(1, worst_slice["accuracy"]),
        xycoords=ax1.transData,
        xytext=(0.05, 0.9),
        textcoords="axes fraction",
        arrowprops=dict(
            arrowstyle="->",
            connectionstyle="arc3,rad=.2",
            color="black",
            lw=2,
        ),
        fontsize=12,
        fontweight="bold",
        color="#2C3E50",
        bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=1.0),
        zorder=20,
    )

    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(plot_dir, "slice_analysis.png")
    plt.savefig(plot_path, dpi=300)
    print(f"Plot saved to: {plot_path}")


if __name__ == "__main__":
    main()
