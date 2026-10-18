"""Classification helpers for the CheMLAgent classifier pipeline.

`make_classes_df` and `clean_ions` are adapted from the user's
`$PYLOC/CafChem/Code/CafChemClassifiers.py` (functions `make_classes`, line 182
and `clean_ions`, line 167), with one deliberate change: classes are stored as
int indices 0..K-1 (what sklearn / LightGBM / chemprop / the MLP's NLLLoss all
consume directly) while the human-readable equ-count range labels ("4.1 < 5.6")
stay available as a string column and in the run manifest.

`classification_metrics` generalizes the source file's `analyze_predictions`
percent-correct idea (which re-parsed the boundary strings) to proper
classification scores that work for K classes: accuracy, balanced accuracy,
MCC, macro F1, and a row-normalized confusion matrix.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
)

# Broadcast print flag: the agent sets chemlagent.classify.print_flag =
# <args.print> at startup (same convention as chemlagent.models/tools).
print_flag = False


def clean_ions(smiles: str) -> str:
    """Strip the common counter-ions from a SMILES string.

    Direct port of CafChemClassifiers.clean_ions.

    Args:
        smiles: SMILES string possibly containing salt fragments
    Returns:
        SMILES string with [Na+], [Cl-], [K+], [Br-], [I-], [Ca2+] removed
    """
    smiles = smiles.replace("[Na+].", "").replace("[Cl-].", "") \
                   .replace(".[Cl-]", "").replace(".[Na+]", "")
    smiles = smiles.replace("[K+].", "").replace("[Br-].", "") \
                   .replace(".[K+]", "").replace(".[Br-]", "")
    smiles = smiles.replace("[I-].", "").replace(".[I-]", "") \
                   .replace("[Ca2+].", "").replace(".[Ca2+]", "")
    return smiles


def make_classes_df(df: pd.DataFrame, target_name: str,
                    num_classes: int) -> tuple[pd.DataFrame, list[str]]:
    """Divide a dataframe's target column into num_classes equal-count bands.

    Adapted from CafChemClassifiers.make_classes: the rows are sorted by the
    target, the sorted column is cut at multiples of ``n // num_classes`` (so
    every class gets approximately the same number of samples regardless of the
    target distribution), and the last cutoff is the bottom / top value. Each
    row is assigned to the FIRST band whose upper cutoff is >= its target (the
    source's ``<=`` rule), i.e. equal-count bands on the sorted target.

    Two changes vs the source: classes are written as int indices 0..K-1 in the
    "class" column, and the boundary strings ("lo < hi") are written to a
    "class_labels" column AND returned as a list. Rows keep their original
    order (the source sorted in place; here labels are mapped back through the
    sort).

    Args:
        df: dataframe with at least a numeric target column
        target_name: name of the target column to bin
        num_classes: requested number of classes (>= 2)
    Returns:
        (df, class_labels): df is a copy with added "class" (int64) and
        "class_labels" (str) columns; class_labels is the list of K range
        strings, index-aligned with the int classes.
    """
    num_classes = int(num_classes)
    if num_classes < 2:
        raise ValueError(f"num_classes must be >= 2, got {num_classes}")
    n = len(df)
    samples_per_class = n // num_classes
    if samples_per_class < 1:
        raise ValueError(
            f"{n} rows cannot be divided into {num_classes} classes.")
    if print_flag:
        print(f"Samples per class: {samples_per_class}, total samples: {n}")

    # Rows ascending by target; `order` holds the original positional row
    # indices so labels map back to the dataframe's original order regardless
    # of what the DataFrame index looks like.
    order = np.argsort(df[target_name].to_numpy(dtype=float), kind="stable")
    vals = df[target_name].to_numpy(dtype=float)[order]

    bottom = vals[0]
    cutoffs = [bottom]
    for i in range(samples_per_class, n - num_classes, samples_per_class):
        v = vals[i]
        if v not in cutoffs:
            cutoffs.append(v)
    top = vals[-1]
    if top not in cutoffs:
        cutoffs.append(top)

    class_labels = [f"{cutoffs[i]} < {cutoffs[i + 1]}"
                    for i in range(len(cutoffs) - 1)]
    n_classes = len(class_labels)

    # Assign per sorted row: first class whose upper cutoff >= target (the
    # source's <= rule). The last cutoff IS the top value, so every row fits;
    # clamp defensively against float-rounding edge cases.
    classes_sorted = []
    for v in vals:
        for i in range(n_classes):
            if v <= cutoffs[i + 1]:
                classes_sorted.append(i)
                break
        else:
            classes_sorted.append(n_classes - 1)

    classes = np.empty(n, dtype=np.int64)
    classes[order] = np.asarray(classes_sorted, dtype=np.int64)

    out = df.copy()
    out["class"] = classes
    out["class_labels"] = [class_labels[c] for c in classes]
    return out, class_labels


def classification_metrics(y_true, y_pred) -> dict:
    """Score predictions with the standard classification metric set.

    Generalizes CafChemClassifiers.analyze_predictions ("percent correct") to
    multi-class scoring. All inputs treated as int class indices.

    Args:
        y_true: true class labels (ints 0..K-1)
        y_pred: predicted class labels (ints 0..K-1)
    Returns:
        dict with keys: accuracy, balanced_accuracy, mcc, f1_macro,
        confusion_matrix (row-normalized list of lists), n_test
    """
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.int64).reshape(-1)
    labels = list(range(int(max(y_true.max(), y_pred.max(), 0)) + 1))
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "confusion_matrix": cm.tolist(),
        "n_test": int(len(y_true)),
    }


def class_label_strings(pred_classes, class_labels: list[str]) -> list[str]:
    """Map predicted int class indices to their human-readable range labels."""
    idx = np.asarray(pred_classes, dtype=int).reshape(-1)
    return [class_labels[int(i)] for i in idx]