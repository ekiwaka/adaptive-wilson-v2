from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd


# High-contrast base colors, adapted from colorblind-safe plotting palettes.
BASE_SAMPLER_COLORS = {
    "uniform": "#000000",
    "degree": "#D55E00",
    "wilson": "#0072B2",
    "adaptive_wilson": "#009E73",
    "none": "#999999",
}

BASE_SAMPLER_MARKERS = {
    "uniform": "o",
    "degree": "s",
    "wilson": "^",
    "adaptive_wilson": "D",
    "none": "x",
}

BASE_SAMPLER_LINESTYLES = {
    "uniform": "-",
    "degree": "-",
    "wilson": "-",
    "adaptive_wilson": "-",
    "none": "--",
}


def slugify(value: object) -> str:
    text = str(value)
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9._-]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_") or "unknown"


def infer_weighting_regime(path: Path) -> str:
    parts = set(path.parts)
    if "normalized" in parts:
        return "normalized"
    if "unweighted" in parts:
        return "unweighted"
    return "unknown"


def numericize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().sum() > 0:
            df[col] = converted
    return df


def first_existing_col(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def find_csvs(root: Path, filename: str) -> list[Path]:
    return sorted(
        path
        for path in root.rglob(filename)
        if "__MACOSX" not in path.parts and not path.name.startswith("._")
    )


def read_summary_tables(root: Path) -> pd.DataFrame:
    paths = find_csvs(root, "benchmark_summary.csv")
    frames: list[pd.DataFrame] = []

    for path in paths:
        df = pd.read_csv(path)
        df["summary_path"] = str(path)
        df["study_root"] = str(path.parent.parent)
        df["weighting_regime"] = infer_weighting_regime(path)
        frames.append(df)

    if not frames:
        raise FileNotFoundError(f"No benchmark_summary.csv files found under {root}")

    return numericize(pd.concat(frames, ignore_index=True))


def read_detailed_tables(root: Path) -> pd.DataFrame:
    paths = find_csvs(root, "benchmark_detailed_runs.csv")
    frames: list[pd.DataFrame] = []

    for path in paths:
        df = pd.read_csv(path)
        df["detailed_path"] = str(path)
        df["study_root"] = str(path.parent.parent)
        df["weighting_regime"] = infer_weighting_regime(path)
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    return numericize(pd.concat(frames, ignore_index=True))


def epoch_to_cumulative_minibatch_updates(
    epoch_series: pd.Series,
    *,
    steps_per_epoch: int,
) -> pd.Series:
    epochs = pd.to_numeric(epoch_series, errors="coerce")

    if epochs.dropna().empty:
        return epochs

    # If epochs are zero-indexed, epoch 0 usually means "after the first epoch",
    # i.e. after steps_per_epoch mini-batch updates.
    if float(epochs.min()) == 0.0:
        return (epochs + 1) * steps_per_epoch

    # If epochs are one-indexed, epoch 1 means after steps_per_epoch updates.
    return epochs * steps_per_epoch


def add_derived_columns(df: pd.DataFrame, *, steps_per_epoch: int) -> pd.DataFrame:
    df = df.copy()

    if "mean_best_epoch" in df.columns:
        df["mean_best_cumulative_minibatch_updates"] = df["mean_best_epoch"] * steps_per_epoch

        # Backward-compatible alias.
        df["mean_best_global_step"] = df["mean_best_cumulative_minibatch_updates"]

    if "std_best_epoch" in df.columns:
        df["std_best_cumulative_minibatch_updates"] = df["std_best_epoch"] * steps_per_epoch

        # Backward-compatible alias.
        df["std_best_global_step"] = df["std_best_cumulative_minibatch_updates"]

    if "mean_average_center_nodes" in df.columns and "mean_train_node_count" in df.columns:
        denom = df["mean_train_node_count"].replace(0, pd.NA)
        df["mean_center_node_fraction"] = df["mean_average_center_nodes"] / denom

    if "mean_val_error" not in df.columns and "mean_val_accuracy" in df.columns:
        df["mean_val_error"] = 1.0 - df["mean_val_accuracy"]
    if "mean_test_error" not in df.columns and "mean_test_accuracy" in df.columns:
        df["mean_test_error"] = 1.0 - df["mean_test_accuracy"]
    if "mean_train_error" not in df.columns and "mean_train_accuracy" in df.columns:
        df["mean_train_error"] = 1.0 - df["mean_train_accuracy"]
    if "std_val_error" not in df.columns and "std_val_accuracy" in df.columns:
        df["std_val_error"] = df["std_val_accuracy"]
    if "std_test_error" not in df.columns and "std_test_accuracy" in df.columns:
        df["std_test_error"] = df["std_test_accuracy"]
    if "std_train_error" not in df.columns and "std_train_accuracy" in df.columns:
        df["std_train_error"] = df["std_train_accuracy"]

    return df


def choose_actual_budget_x(df: pd.DataFrame) -> str:
    candidates = [
        "mean_average_center_nodes",
        "average_center_nodes",
        "mean_resolved_budget",
        "resolved_budget",
        "budget_value",
    ]
    col = first_existing_col(df, candidates)
    if col is None:
        raise ValueError("Could not find a usable x-axis budget column.")
    return col


def sampler_label(row: pd.Series, *, include_alpha: bool = False) -> str:
    sampler = str(row.get("sampler_name", "unknown"))

    if include_alpha and sampler == "adaptive_wilson":
        alpha = row.get("adaptive_alpha")
        if pd.notna(alpha):
            return f"adaptive_wilson α={float(alpha):g}"

    return sampler


def parse_adaptive_alpha_from_label(label: object) -> float | None:
    text = str(label)
    match = re.search(r"adaptive_wilson\s+α=([0-9.eE+-]+)", text)
    if not match:
        return None

    try:
        return float(match.group(1))
    except ValueError:
        return None


def sort_plot_labels(labels: Iterable[object]) -> list[str]:
    labels_as_text = [str(label) for label in labels]

    def key(label: str) -> tuple[int, float, str]:
        if label == "uniform":
            return (0, 0.0, label)
        if label == "degree":
            return (1, 0.0, label)
        if label == "wilson":
            return (2, 0.0, label)

        alpha = parse_adaptive_alpha_from_label(label)
        if alpha is not None:
            return (3, alpha, label)

        if label == "adaptive_wilson":
            return (3, 0.0, label)

        return (4, 0.0, label)

    return sorted(labels_as_text, key=key)


def base_style_for_sampler(sampler_name: object, *, regime: object | None = None) -> dict[str, object]:
    sampler = str(sampler_name)
    regime_text = "" if regime is None or pd.isna(regime) else str(regime)

    linestyle = BASE_SAMPLER_LINESTYLES.get(sampler, "-")
    if regime_text == "normalized":
        linestyle = "--"
    elif regime_text == "unweighted":
        linestyle = "-"

    return {
        "color": BASE_SAMPLER_COLORS.get(sampler, "#444444"),
        "marker": BASE_SAMPLER_MARKERS.get(sampler, "o"),
        "linestyle": linestyle,
        "linewidth": 2.1,
        "markersize": 4.5,
    }


def build_convergence_style_map(labels: Iterable[object]) -> dict[str, dict[str, object]]:
    labels_sorted = sort_plot_labels(labels)
    style_map: dict[str, dict[str, object]] = {}

    adaptive_labels = [
        label
        for label in labels_sorted
        if parse_adaptive_alpha_from_label(label) is not None
    ]

    # Use a perceptually separated colormap range for the Adaptive Wilson alpha sweep.
    # This avoids the default matplotlib cycle where many lines look too similar.
    cmap = plt.get_cmap("viridis")
    if adaptive_labels:
        denom = max(1, len(adaptive_labels) - 1)
        for idx, label in enumerate(adaptive_labels):
            position = 0.05 + 0.90 * (idx / denom)
            style_map[label] = {
                "color": cmap(position),
                "marker": "o",
                "linestyle": "-",
                "linewidth": 1.9,
                "markersize": 3.5,
            }

    # Strong fixed colors for baselines.
    for label in labels_sorted:
        if label in style_map:
            continue

        sampler = label.split(" | ")[0]
        if label in BASE_SAMPLER_COLORS:
            sampler = label

        style_map[label] = base_style_for_sampler(sampler)

    return style_map


def annotate_adaptive_points(subset: pd.DataFrame, x_col: str, y_col: str) -> None:
    if "adaptive_alpha" not in subset.columns:
        return

    for _, row in subset.iterrows():
        alpha = row.get("adaptive_alpha")
        x = row.get(x_col)
        y = row.get(y_col)

        if pd.notna(alpha) and pd.notna(x) and pd.notna(y):
            plt.annotate(
                f"α={float(alpha):g}",
                (float(x), float(y)),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
            )


def plot_metric_by_sampler(
    df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    y_std_col: str | None,
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: Path,
    annotate_alpha: bool = True,
    log_x: bool = False,
    log_y: bool = False,
) -> None:
    if x_col not in df.columns or y_col not in df.columns:
        return

    data = df.copy()
    data = data[data[y_col].notna()]
    data = data[data[x_col].notna()]

    if "sampler_name" in data.columns:
        data = data[~data["sampler_name"].astype(str).str.startswith("baseline_")]

    if data.empty:
        return

    plt.figure(figsize=(8.8, 5.4))

    for sampler in sorted(data["sampler_name"].astype(str).unique()):
        subset = data[data["sampler_name"].astype(str) == sampler].copy()
        sort_cols = [x_col, "adaptive_alpha"] if "adaptive_alpha" in subset.columns else [x_col]
        subset = subset.sort_values(sort_cols)

        xs = subset[x_col].astype(float)
        ys = subset[y_col].astype(float)
        style = base_style_for_sampler(sampler)

        if y_std_col and y_std_col in subset.columns and subset[y_std_col].notna().any():
            yerr = subset[y_std_col].astype(float).clip(lower=0)
            plt.errorbar(
                xs,
                ys,
                yerr=yerr,
                capsize=3,
                label=sampler,
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=style["linewidth"],
                markersize=style["markersize"],
            )
        else:
            plt.plot(
                xs,
                ys,
                label=sampler,
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=style["linewidth"],
                markersize=style["markersize"],
            )

        if annotate_alpha and sampler == "adaptive_wilson":
            annotate_adaptive_points(subset, x_col, y_col)

    if log_x:
        plt.xscale("log")
    if log_y:
        plt.yscale("log")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_summary_metrics(summary: pd.DataFrame, out_root: Path, *, steps_per_epoch: int) -> None:
    summary = add_derived_columns(summary, steps_per_epoch=steps_per_epoch)
    x_col = choose_actual_budget_x(summary)

    plots = [
        (
            "test_macro_f1_vs_actual_centers.png",
            "mean_test_macro_f1",
            "std_test_macro_f1",
            "Test macro-F1 vs actual sampled center nodes",
            "Test macro-F1",
        ),
        (
            "test_accuracy_vs_actual_centers.png",
            "mean_test_accuracy",
            "std_test_accuracy",
            "Test accuracy vs actual sampled center nodes",
            "Test accuracy",
        ),
        (
            "val_macro_f1_vs_actual_centers.png",
            "mean_val_macro_f1",
            "std_val_macro_f1",
            "Validation macro-F1 vs actual sampled center nodes",
            "Validation macro-F1",
        ),
        (
            "val_error_vs_actual_centers.png",
            "mean_val_error",
            "std_val_error",
            "Validation error vs actual sampled center nodes",
            "Validation error",
        ),
        (
            "test_error_vs_actual_centers.png",
            "mean_test_error",
            "std_test_error",
            "Test error vs actual sampled center nodes",
            "Test error",
        ),
        (
            "best_epoch_vs_actual_centers.png",
            "mean_best_epoch",
            "std_best_epoch",
            "Best epoch vs actual sampled center nodes",
            "Best epoch",
        ),
        (
            "best_cumulative_minibatch_updates_vs_actual_centers.png",
            "mean_best_cumulative_minibatch_updates",
            "std_best_cumulative_minibatch_updates",
            "Best cumulative mini-batch updates vs actual sampled center nodes",
            "Best cumulative mini-batch updates",
        ),
        (
            "expected_minibatch_loss_vs_actual_centers.png",
            "mean_mean_epoch_batch_loss_mean",
            "std_mean_epoch_batch_loss_mean",
            "Estimated expected mini-batch loss vs actual sampled center nodes",
            "Mean mini-batch loss estimate",
        ),
        (
            "loss_estimator_variance_vs_actual_centers.png",
            "mean_mean_epoch_batch_loss_variance",
            "std_mean_epoch_batch_loss_variance",
            "Mini-batch loss-estimator variance vs actual sampled center nodes",
            "Mini-batch loss variance",
        ),
        (
            "batch_loss_std_vs_actual_centers.png",
            "mean_std_batch_loss",
            "std_std_batch_loss",
            "Batch-loss standard deviation vs actual sampled center nodes",
            "Batch-loss std",
        ),
        (
            "loss_variance_vs_test_macro_f1.png",
            "mean_mean_epoch_batch_loss_variance",
            None,
            "Test macro-F1 vs loss-estimator variance",
            "Mini-batch loss variance",
        ),
        (
            "center_class_coverage_vs_actual_centers.png",
            "mean_mean_center_class_coverage_ratio",
            "std_mean_center_class_coverage_ratio",
            "Center-label class coverage vs actual sampled center nodes",
            "Center class coverage ratio",
        ),
        (
            "center_label_entropy_vs_actual_centers.png",
            "mean_mean_center_label_entropy_ratio",
            "std_mean_center_label_entropy_ratio",
            "Center-label entropy ratio vs actual sampled center nodes",
            "Center label entropy ratio",
        ),
        (
            "time_per_epoch_vs_actual_centers.png",
            "mean_training_time_per_epoch",
            "std_training_time_per_epoch",
            "Training time per epoch vs actual sampled center nodes",
            "Training time per epoch",
        ),
        (
            "total_training_time_vs_actual_centers.png",
            "mean_total_training_time",
            "std_total_training_time",
            "Total training time vs actual sampled center nodes",
            "Total training time",
        ),
        (
            "mean_total_batch_time_vs_actual_centers.png",
            "mean_mean_total_batch_time",
            "std_mean_total_batch_time",
            "Mean sampled-batch time vs actual sampled center nodes",
            "Mean batch time (s)",
        ),
        (
            "mean_sampling_time_vs_actual_centers.png",
            "mean_mean_sampling_time",
            "std_mean_sampling_time",
            "Mean sampler time vs actual sampled center nodes",
            "Mean sampling time (s)",
        ),
        (
            "mean_subgraph_time_vs_actual_centers.png",
            "mean_mean_subgraph_construction_time",
            "std_mean_subgraph_construction_time",
            "Mean subgraph construction time vs actual sampled center nodes",
            "Mean subgraph time (s)",
        ),
        (
            "mean_model_compute_time_vs_actual_centers.png",
            "mean_mean_model_compute_time",
            "std_mean_model_compute_time",
            "Mean model compute time vs actual sampled center nodes",
            "Mean model compute time (s)",
        ),
        (
            "peak_memory_mb_vs_actual_centers.png",
            "mean_mean_peak_memory_mb",
            "std_mean_peak_memory_mb",
            "Peak memory snapshot vs actual sampled center nodes",
            "Peak memory snapshot (MB)",
        ),
        (
            "sampled_edges_vs_actual_centers.png",
            "mean_average_sampled_edges",
            "std_average_sampled_edges",
            "Sampled edges vs actual sampled center nodes",
            "Mean sampled edges",
        ),
        (
            "online_normalized_weighting_enabled_vs_actual_centers.png",
            "mean_mean_online_normalized_weighting_enabled",
            "std_mean_online_normalized_weighting_enabled",
            "Online normalized-weighting status vs actual sampled center nodes",
            "Online normalized weighting enabled",
        ),
        (
            "online_weight_min_vs_actual_centers.png",
            "mean_mean_online_weight_min",
            "std_mean_online_weight_min",
            "Online weight minimum vs actual sampled center nodes",
            "Mean online weight minimum",
        ),
        (
            "online_weight_max_vs_actual_centers.png",
            "mean_mean_online_weight_max",
            "std_mean_online_weight_max",
            "Online weight maximum vs actual sampled center nodes",
            "Mean online weight maximum",
        ),
        (
            "online_weight_mean_vs_actual_centers.png",
            "mean_mean_online_weight_mean",
            "std_mean_online_weight_mean",
            "Online weight mean vs actual sampled center nodes",
            "Mean online weight mean",
        ),
    ]

    for (dataset, regime), group in summary.groupby(["dataset_name", "weighting_regime"], dropna=False):
        group_out = out_root / slugify(dataset) / slugify(regime) / "summary"

        for filename, y_col, y_std_col, title, ylabel in plots:
            if y_col not in group.columns:
                continue

            local_x = x_col
            xlabel = "Mean sampled center nodes"

            if filename == "loss_variance_vs_test_macro_f1.png":
                if "mean_test_macro_f1" not in group.columns:
                    continue
                local_x = "mean_mean_epoch_batch_loss_variance"
                xlabel = "Mini-batch loss variance"
                y_col = "mean_test_macro_f1"
                y_std_col = "std_test_macro_f1"
                ylabel = "Test macro-F1"

            plot_metric_by_sampler(
                group,
                x_col=local_x,
                y_col=y_col,
                y_std_col=y_std_col,
                title=f"{dataset} | {regime} | {title}",
                xlabel=xlabel,
                ylabel=ylabel,
                out_path=group_out / filename,
                annotate_alpha=True,
            )


def plot_adaptive_alpha_metrics(summary: pd.DataFrame, out_root: Path) -> None:
    if "adaptive_alpha" not in summary.columns:
        return

    adaptive = summary[summary["sampler_name"].astype(str) == "adaptive_wilson"].copy()
    adaptive = adaptive[adaptive["adaptive_alpha"].notna()]
    adaptive = adaptive[adaptive["adaptive_alpha"] > 0]

    if adaptive.empty:
        return

    metrics = [
        (
            "adaptive_alpha_vs_actual_centers.png",
            choose_actual_budget_x(adaptive),
            "Actual sampled center nodes",
        ),
        (
            "adaptive_alpha_vs_test_macro_f1.png",
            "mean_test_macro_f1",
            "Test macro-F1",
        ),
        (
            "adaptive_alpha_vs_val_macro_f1.png",
            "mean_val_macro_f1",
            "Validation macro-F1",
        ),
        (
            "adaptive_alpha_vs_loss_estimator_variance.png",
            "mean_mean_epoch_batch_loss_variance",
            "Mini-batch loss variance",
        ),
        (
            "adaptive_alpha_vs_expected_minibatch_loss.png",
            "mean_mean_epoch_batch_loss_mean",
            "Mean mini-batch loss estimate",
        ),
        (
            "adaptive_alpha_vs_training_time_per_epoch.png",
            "mean_training_time_per_epoch",
            "Training time per epoch",
        ),
        (
            "adaptive_alpha_vs_center_class_coverage.png",
            "mean_mean_center_class_coverage_ratio",
            "Center class coverage ratio",
        ),
        (
            "adaptive_alpha_vs_sampled_edges.png",
            "mean_average_sampled_edges",
            "Mean sampled edges",
        ),
    ]

    for (dataset, regime), group in adaptive.groupby(["dataset_name", "weighting_regime"], dropna=False):
        group = group.sort_values("adaptive_alpha")
        group_out = out_root / slugify(dataset) / slugify(regime) / "adaptive_alpha"

        for filename, y_col, ylabel in metrics:
            if y_col not in group.columns:
                continue

            plot_metric_by_sampler(
                group,
                x_col="adaptive_alpha",
                y_col=y_col,
                y_std_col=None,
                title=f"{dataset} | {regime} | {ylabel} vs Adaptive Wilson alpha",
                xlabel="Adaptive Wilson alpha",
                ylabel=ylabel,
                out_path=group_out / filename,
                annotate_alpha=False,
                log_x=True,
            )


def plot_normalized_vs_unweighted(summary: pd.DataFrame, out_root: Path) -> None:
    if "weighting_regime" not in summary.columns:
        return

    x_col = choose_actual_budget_x(summary)

    metrics = [
        ("test_macro_f1_normalized_vs_unweighted.png", "mean_test_macro_f1", "Test macro-F1"),
        ("test_accuracy_normalized_vs_unweighted.png", "mean_test_accuracy", "Test accuracy"),
        (
            "loss_variance_normalized_vs_unweighted.png",
            "mean_mean_epoch_batch_loss_variance",
            "Mini-batch loss variance",
        ),
        (
            "expected_loss_normalized_vs_unweighted.png",
            "mean_mean_epoch_batch_loss_mean",
            "Mean mini-batch loss estimate",
        ),
    ]

    for dataset, group in summary.groupby("dataset_name", dropna=False):
        group = group.copy()
        group = group[group[x_col].notna()]
        group_out = out_root / slugify(dataset) / "normalized_vs_unweighted"

        for filename, y_col, ylabel in metrics:
            if y_col not in group.columns:
                continue

            data = group[group[y_col].notna()].copy()
            if data.empty:
                continue

            plt.figure(figsize=(8.8, 5.4))

            for (regime, sampler), subset in data.groupby(["weighting_regime", "sampler_name"], dropna=False):
                sort_cols = [x_col, "adaptive_alpha"] if "adaptive_alpha" in subset.columns else [x_col]
                subset = subset.sort_values(sort_cols)

                style = base_style_for_sampler(sampler, regime=regime)
                label = f"{sampler} | {regime}"

                plt.plot(
                    subset[x_col].astype(float),
                    subset[y_col].astype(float),
                    label=label,
                    color=style["color"],
                    marker=style["marker"],
                    linestyle=style["linestyle"],
                    linewidth=style["linewidth"],
                    markersize=style["markersize"],
                )

                if str(sampler) == "adaptive_wilson":
                    annotate_adaptive_points(subset, x_col, y_col)

            plt.xlabel("Mean sampled center nodes")
            plt.ylabel(ylabel)
            plt.title(f"{dataset} | {ylabel}: normalized vs unweighted")
            plt.grid(True, alpha=0.25)
            plt.legend(fontsize=8)
            plt.tight_layout()

            group_out.mkdir(parents=True, exist_ok=True)
            plt.savefig(group_out / filename, dpi=180, bbox_inches="tight")
            plt.close()


def make_config_key(row: pd.Series) -> str:
    fields = [
        "sampler_name",
        "adaptive_alpha",
        "budget_value",
        "resolved_budget",
        "batch_size",
        "num_hops",
        "q",
        "sampling_mode",
        "online_schedule",
        "split_idx",
    ]

    pieces = []
    for field in fields:
        if field in row.index:
            value = row.get(field)
            if pd.isna(value):
                value = "NA"
            pieces.append(f"{field}={value}")

    return "|".join(pieces)


def history_path_from_row(row: pd.Series, repo_root: Path) -> Path | None:
    raw = row.get("history_path")
    if pd.isna(raw) or not str(raw).strip():
        return None

    path = Path(str(raw))
    if path.is_absolute() and path.exists():
        return path

    candidate = repo_root / path
    if candidate.exists():
        return candidate

    if path.exists():
        return path

    return None


def select_best_history_rows(group: pd.DataFrame) -> pd.DataFrame:
    score_col = first_existing_col(
        group,
        [
            "best_monitor_value",
            "test_macro_f1",
            "val_macro_f1",
            "test_accuracy",
            "val_accuracy",
        ],
    )

    group = group.copy()
    group["config_key"] = group.apply(make_config_key, axis=1)

    selected_parts: list[pd.DataFrame] = []

    for sampler, sampler_group in group.groupby("sampler_name", dropna=False):
        sampler_name = str(sampler)

        # For Adaptive Wilson, keep every alpha configuration.
        # This lets convergence plots show α=0.001, α=0.003, ..., α=1, etc.
        if sampler_name == "adaptive_wilson":
            selected_parts.append(sampler_group)
            continue

        # For non-adaptive samplers, keep one representative/best config.
        if score_col is None:
            selected_parts.append(sampler_group)
            continue

        scores = (
            sampler_group.groupby("config_key", dropna=False)[score_col]
            .mean()
            .sort_values(ascending=False)
        )

        if scores.empty:
            continue

        best_key = scores.index[0]
        selected_parts.append(sampler_group[sampler_group["config_key"] == best_key])

    if not selected_parts:
        return pd.DataFrame()

    return pd.concat(selected_parts, ignore_index=True)


def load_selected_histories(
    detailed: pd.DataFrame,
    *,
    repo_root: Path,
    steps_per_epoch: int,
) -> pd.DataFrame:
    if detailed.empty or "history_path" not in detailed.columns:
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []

    for (dataset, regime), group in detailed.groupby(["dataset_name", "weighting_regime"], dropna=False):
        selected = select_best_history_rows(group)

        for _, row in selected.iterrows():
            path = history_path_from_row(row, repo_root)
            if path is None:
                continue

            hist = pd.read_csv(path)
            hist = numericize(hist)

            hist["dataset_name"] = dataset
            hist["weighting_regime"] = regime
            hist["sampler_name"] = row.get("sampler_name")
            hist["adaptive_alpha"] = row.get("adaptive_alpha")
            hist["budget_value"] = row.get("budget_value")
            hist["resolved_budget"] = row.get("resolved_budget")
            hist["seed"] = row.get("seed")
            hist["history_path"] = str(path)

            if "epoch" in hist.columns:
                hist["cumulative_minibatch_updates"] = epoch_to_cumulative_minibatch_updates(
                    hist["epoch"],
                    steps_per_epoch=steps_per_epoch,
                )

            if "val_error" not in hist.columns and "val_accuracy" in hist.columns:
                hist["val_error"] = 1.0 - hist["val_accuracy"].astype(float)
            if "test_error" not in hist.columns and "test_accuracy" in hist.columns:
                hist["test_error"] = 1.0 - hist["test_accuracy"].astype(float)
            if "train_error" not in hist.columns and "train_accuracy" in hist.columns:
                hist["train_error"] = 1.0 - hist["train_accuracy"].astype(float)

            hist["plot_label"] = sampler_label(row, include_alpha=True)

            frames.append(hist)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def plot_history_curves(histories: pd.DataFrame, out_root: Path) -> None:
    if histories.empty:
        return

    x_col = (
        "cumulative_minibatch_updates"
        if "cumulative_minibatch_updates" in histories.columns
        else "epoch"
    )
    xlabel = "Cumulative mini-batch updates" if x_col == "cumulative_minibatch_updates" else "Epoch"

    metrics = [
        ("val_macro_f1_convergence.png", "val_macro_f1", "Validation macro-F1"),
        ("test_macro_f1_convergence.png", "test_macro_f1", "Test macro-F1"),
        ("val_accuracy_convergence.png", "val_accuracy", "Validation accuracy"),
        ("test_accuracy_convergence.png", "test_accuracy", "Test accuracy"),
        ("val_error_convergence.png", "val_error", "Validation error"),
        ("test_error_convergence.png", "test_error", "Test error"),
        ("train_loss_convergence.png", "train_loss", "Train loss"),
        ("val_loss_convergence.png", "val_loss", "Validation loss"),
        ("epoch_minibatch_loss_variance_convergence.png", "epoch_minibatch_loss_variance", "Mini-batch loss-estimator variance"),
        ("epoch_minibatch_loss_std_convergence.png", "epoch_minibatch_loss_std", "Mini-batch loss std"),
        ("loss_residual_squared_convergence.png", "minibatch_loss_minus_expectation_squared_mean", "Mean squared mini-batch loss residual"),
        ("mean_total_batch_time_convergence.png", "mean_total_batch_time", "Mean time per sampled batch (s)"),
        ("mean_sampling_time_convergence.png", "mean_sampling_time", "Mean sampling time per batch (s)"),
        ("mean_subgraph_construction_time_convergence.png", "mean_subgraph_construction_time", "Mean subgraph construction time per batch (s)"),
        ("mean_model_compute_time_convergence.png", "mean_model_compute_time", "Mean model compute time per batch (s)"),
        ("peak_memory_mb_convergence.png", "max_peak_memory_mb", "Peak memory snapshot (MB)"),
    ]

    for (dataset, regime), group in histories.groupby(["dataset_name", "weighting_regime"], dropna=False):
        group_out = out_root / slugify(dataset) / slugify(regime) / "convergence"

        for filename, metric, ylabel in metrics:
            if metric not in group.columns:
                continue

            data = group[group[metric].notna()].copy()
            if data.empty:
                continue

            agg = (
                data.groupby(["plot_label", x_col], dropna=False)[metric]
                .mean()
                .reset_index()
            )

            labels = sort_plot_labels(agg["plot_label"].unique())
            style_map = build_convergence_style_map(labels)

            plt.figure(figsize=(9.6, 5.8))

            for label in labels:
                subset = agg[agg["plot_label"].astype(str) == label].copy()
                if subset.empty:
                    continue

                subset = subset.sort_values(x_col)
                style = style_map[label]

                plt.plot(
                    subset[x_col].astype(float),
                    subset[metric].astype(float),
                    label=str(label),
                    color=style["color"],
                    marker=style["marker"],
                    linestyle=style["linestyle"],
                    linewidth=style["linewidth"],
                    markersize=style["markersize"],
                )

            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.title(f"{dataset} | {regime} | {ylabel} over training iterations")
            plt.grid(True, alpha=0.25)

            # Move legend outside for crowded alpha-sweep convergence plots.
            plt.legend(
                fontsize=8,
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                borderaxespad=0.0,
            )

            plt.tight_layout()
            group_out.mkdir(parents=True, exist_ok=True)
            plt.savefig(group_out / filename, dpi=180, bbox_inches="tight")
            plt.close()


def print_outputs(out_root: Path) -> None:
    pngs = sorted(out_root.rglob("*.png"))
    print(f"Wrote {len(pngs)} plot files under: {out_root}")
    for path in pngs[:20]:
        print(f"  {path}")
    if len(pngs) > 20:
        print(f"  ... and {len(pngs) - 20} more")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate plots from online benchmark outputs."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/active/local/online"),
        help="Root directory containing benchmark outputs.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/active/local/online/extra_plots"),
        help="Directory where plots will be written.",
    )
    parser.add_argument(
        "--steps-per-epoch",
        type=int,
        default=50,
        help="Used to convert epoch-level histories into cumulative mini-batch updates.",
    )
    parser.add_argument(
        "--skip-history",
        action="store_true",
        help="Only make summary-level plots; skip history.csv convergence curves.",
    )
    args = parser.parse_args()

    repo_root = Path.cwd()

    summary = read_summary_tables(args.root)
    plot_summary_metrics(summary, args.out, steps_per_epoch=args.steps_per_epoch)
    plot_adaptive_alpha_metrics(summary, args.out)
    plot_normalized_vs_unweighted(summary, args.out)

    if not args.skip_history:
        detailed = read_detailed_tables(args.root)
        histories = load_selected_histories(
            detailed,
            repo_root=repo_root,
            steps_per_epoch=args.steps_per_epoch,
        )
        plot_history_curves(histories, args.out)

    print_outputs(args.out)


if __name__ == "__main__":
    main()