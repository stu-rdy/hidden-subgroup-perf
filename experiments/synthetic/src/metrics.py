import numpy as np
import pandas as pd


def compute_performance_gap(accuracies):
    """
    Δ(S) = max(M(s)) - min(M(s))
    """
    if len(accuracies) == 0:
        return 0.0
    return np.max(accuracies) - np.min(accuracies)


def compute_average_purity(df, slice_col="slice", ground_truth_col="gt_group", c=1):
    """
    AP(S) = (1/|A|) * Σ_a max_{s ∈ S_a} (n_{s,a} / (n_s + c))
    where S_a is the set of subgroups whose majority attribute is a.

    Args:
        df: DataFrame containing slice assignments and ground truth group labels.
        slice_col: Column name for discovered slices.
        ground_truth_col: Column name for ground truth groups/attributes.
        c: Robustness term for small subgroups.
    """
    unique_slices = df[slice_col].unique()
    unique_gt_groups = df[ground_truth_col].unique()

    # 1. For each slice, find its majority ground truth group
    slice_stats = []
    for sl in unique_slices:
        subset = df[df[slice_col] == sl]
        n_s = len(subset)
        if n_s == 0:
            continue

        # Count occurrences of each gt_group in this slice
        counts = subset[ground_truth_col].value_counts()
        majority_a = counts.index[0]
        n_sa = counts.iloc[0]

        purity = n_sa / (n_s + c)

        slice_stats.append({"slice": sl, "majority_a": majority_a, "purity": purity})

    stats_df = pd.DataFrame(slice_stats)

    # 2. For each gt_group a, find Sa (slices where majority_a == a) and take max purity
    group_purities = []
    for a in unique_gt_groups:
        Sa = stats_df[stats_df["majority_a"] == a]
        if len(Sa) > 0:
            max_purity = Sa["purity"].max()
        else:
            max_purity = 0.0
        group_purities.append(max_purity)

    # 3. Average across all gt_groups
    if len(unique_gt_groups) == 0:
        return 0.0

    return np.mean(group_purities)
