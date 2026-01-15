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

    # Calculate errors per slice
    # errors = (1 - accuracy) * size
    test_df["errors"] = (1 - test_df["accuracy"]) * test_df["size"]
    total_errors = test_df["errors"].sum()

    # Calculate error fraction
    test_df["error_fraction"] = test_df["errors"] / total_errors

    # Sort by error fraction descending
    test_df = test_df.sort_values("error_fraction", ascending=False).reset_index(
        drop=True
    )
    test_df["cumulative_error_fraction"] = test_df["error_fraction"].cumsum()
    test_df["rank"] = range(1, len(test_df) + 1)

    # Create plot
    fig, ax1 = plt.subplots(figsize=(12, 7))

    # Define muted opaque colors: Red for top 3, Light Grey for others
    bar_colors = ["#EC7063"] * min(3, len(test_df)) + ["#D5D8DC"] * max(
        0, len(test_df) - 3
    )

    # Opaque bars to hide grid lines while maintaining muted look
    ax1.bar(
        test_df["rank"],
        test_df["error_fraction"],
        color=bar_colors,
        alpha=1.0,
        width=0.6,
        label="Slice Error Fraction",
        zorder=3,
    )

    # Overlay cumulative line (Pareto)
    ax2 = ax1.twinx()
    ax2.plot(
        test_df["rank"],
        test_df["cumulative_error_fraction"],
        color="#3498DB",
        marker="o",
        linewidth=3,
        markersize=8,
        label="Cumulative % of Total Errors",
    )

    # Set y-axis to percentage
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f"{x:.0%}"))
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f"{x:.0%}"))

    # Limits
    ax1.set_ylim(0, max(test_df["error_fraction"]) * 1.2)
    ax2.set_ylim(0, 1.1)

    # Labels and title
    ax1.set_xlabel("Slice rank (DOMINO-discovered, sorted by error mass)")
    ax1.set_ylabel("Fraction of total model errors", color="#2C3E50")
    ax2.set_ylabel("Cumulative % of total errors", color="#3498DB")
    plt.title("Error Mass Concentration (Pareto of Failures)", pad=30)

    # Ensure grid is below artists and disabled on ax2 to prevent overlap visibility
    ax1.set_axisbelow(True)
    ax1.grid(True, zorder=0)
    ax2.grid(False)

    # Explicitly set x-axis ticks and limits to match data range (1-15) with balanced padding
    ax1.set_xticks(range(1, len(test_df) + 1))
    ax1.set_xlim(0, len(test_df) + 1)

    # Annotate the top 3 error sources (the red bars)
    # Moving this to ax2 ensures it is on the top-most layer
    top_3_fraction = test_df.iloc[2]["cumulative_error_fraction"]
    ax2.annotate(
        f"Top 3 slices → {top_3_fraction:.1%} of errors",
        xy=(
            2,
            test_df.iloc[1]["error_fraction"],
        ),  # Reference coordinates (rank=2, error_fraction)
        xycoords=ax1.transData,  # Use ax1's data coordinate system
        xytext=(0.1, 0.9),
        textcoords="axes fraction",
        arrowprops=dict(
            arrowstyle="->", connectionstyle="arc3,rad=.2", color="black", lw=2
        ),
        fontsize=12,
        fontweight="bold",
        color="#2C3E50",
        bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=1.0),
        zorder=20,
    )

    # Finalize
    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(plot_dir, "error_concentration.png")
    plt.savefig(plot_path, dpi=300)
    print(f"Plot saved to: {plot_path}")

    # Print concentration stats
    print(
        f"Top 1 slice accounts for {test_df.iloc[0]['cumulative_error_fraction']:.1%} of total errors."
    )
    if len(test_df) > 2:
        print(
            f"Top 3 slices account for {test_df.iloc[2]['cumulative_error_fraction']:.1%} of total errors."
        )


if __name__ == "__main__":
    main()
