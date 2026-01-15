import numpy as np
import pandas as pd


def analyze_slices(slice_assignments, targets, predictions=None, metadata=None):
    """
    Analyze discovered slices.

    Args:
        slice_assignments: array of slice assignments (N,)
        targets: array of ground truth labels (N,)
        predictions: optional array of predicted labels (N,) for computing accuracy
        metadata: optional dict of additional boolean/numeric metadata (N,) per key

    Returns:
        DataFrame with per-slice statistics
    """
    data = {"slice": slice_assignments, "target": targets}
    if predictions is not None:
        data["prediction"] = predictions

    if metadata is not None:
        for k, v in metadata.items():
            data[k] = v

    df = pd.DataFrame(data)

    unique_slices = np.unique(slice_assignments)
    results = []

    for sl in unique_slices:
        subset = df[df["slice"] == sl]
        size = len(subset)

        # Get dominant class
        if len(subset) > 0:
            dom_class = subset["target"].mode()[0]
            dom_class_perc = (subset["target"] == dom_class).mean()
        else:
            dom_class = -1
            dom_class_perc = 0.0

        result = {
            "slice": int(sl),
            "size": size,
            "dom_class": int(dom_class),
            "dom_class_perc": float(dom_class_perc),
        }

        # Calculate means for metadata
        if metadata is not None:
            for k in metadata.keys():
                result[f"{k}_rate"] = float(subset[k].mean())

        # Add accuracy if predictions are provided
        if predictions is not None:
            if len(subset) > 0:
                accuracy = (subset["prediction"] == subset["target"]).mean()
            else:
                accuracy = 0.0
            result["accuracy"] = float(accuracy)

        results.append(result)

    return pd.DataFrame(results)
