#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd


METRICS: list[tuple[str, str, str, bool]] = [
    ("test_error_convergence", "test_error", "Test error", False),
    ("val_error_convergence", "val_error", "Validation error", False),
    ("test_accuracy_convergence", "test_accuracy", "Test accuracy", True),
    ("val_accuracy_convergence", "val_accuracy", "Validation accuracy", True),
    ("test_macro_f1_convergence", "test_macro_f1", "Test macro-F1", True),
    ("val_macro_f1_convergence", "val_macro_f1", "Validation macro-F1", True),
    ("train_loss_convergence", "train_loss", "Train loss", False),
    ("val_loss_convergence", "val_loss", "Validation loss", False),
    ("epoch_minibatch_loss_variance_convergence", "epoch_minibatch_loss_variance", "Mini-batch loss-estimator variance", False),
    ("loss_residual_squared_convergence", "minibatch_loss_minus_expectation_squared_mean", "Mean squared mini-batch loss residual", False),
    ("mean_total_batch_time_convergence", "mean_total_batch_time", "Mean time per sampled batch (s)", False),
    ("mean_sampling_time_convergence", "mean_sampling_time", "Mean sampling time per batch (s)", False),
    ("mean_subgraph_construction_time_convergence", "mean_subgraph_construction_time", "Mean subgraph construction time per batch (s)", False),
    ("mean_model_compute_time_convergence", "mean_model_compute_time", "Mean model compute time per batch (s)", False),
    ("peak_memory_mb_convergence", "max_peak_memory_mb", "Peak memory snapshot (MB)", False),
]

SAMPLER_ORDER = {
    "uniform": 0,
    "degree": 1,
    "wilson": 2,
    "adaptive_wilson": 3,
}


def slugify(value: object) -> str:
    text = str(value).strip().lower().replace("/", "_").replace("-", "_")
    text = re.sub(r"[^a-z0-9._-]+", "_", text)
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


def find_detailed_csvs(root: Path) -> list[Path]:
    return sorted(
        p for p in root.rglob("benchmark_detailed_runs.csv")
        if "_combined_for_plots" not in p.parts
        and "__MACOSX" not in p.parts
        and not p.name.startswith("._")
    )


def resolve_history_path(raw: object, repo_root: Path) -> Path | None:
    if raw is None or pd.isna(raw):
        return None
    text = str(raw).strip()
    if not text:
        return None

    path = Path(text)

    if path.is_absolute() and path.exists():
        return path

    candidate = repo_root / path
    if candidate.exists():
        return candidate

    if path.exists():
        return path

    # Handles Grid absolute paths copied to local repo, e.g. /home/.../outputs/...
    marker = "outputs/"
    if marker in text:
        suffix = text[text.index(marker):]
        candidate = repo_root / suffix
        if candidate.exists():
            return candidate

    marker = "./outputs/"
    if marker in text:
        suffix = text[text.index(marker) + 2:]
        candidate = repo_root / suffix
        if candidate.exists():
            return candidate

    return None


def read_detailed(root: Path, dataset_filter: str | None, repo_root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    dataset_filter_slug = slugify(dataset_filter) if dataset_filter else None

    for path in find_detailed_csvs(root):
        df = pd.read_csv(path)
        if df.empty:
            continue
        df = numericize(df)
        df["source_detailed_path"] = str(path)
        df["weighting_regime"] = infer_weighting_regime(path)
        if dataset_filter_slug is not None and "dataset_name" in df.columns:
            df = df[df["dataset_name"].map(slugify) == dataset_filter_slug]
        if not df.empty:
            frames.append(df)

    if not frames:
        raise FileNotFoundError(f"No benchmark_detailed_runs.csv rows found under {root}")

    detailed = pd.concat(frames, ignore_index=True)
    if "history_path" not in detailed.columns:
        raise ValueError("benchmark_detailed_runs.csv does not contain history_path; per-budget convergence cannot be made.")
    return detailed


def epoch_to_updates(epoch: pd.Series, steps_per_epoch: int) -> pd.Series:
    epochs = pd.to_numeric(epoch, errors="coerce")
    if epochs.dropna().empty:
        return epochs
    if float(epochs.min()) == 0.0:
        return (epochs + 1) * steps_per_epoch
    return epochs * steps_per_epoch


def normalized_budget_value(value: object) -> int | None:
    if value is None or pd.isna(value):
        return None
    try:
        return int(round(float(value)))
    except Exception:
        return None


def sampler_sort_key(label: str) -> tuple[int, str]:
    base = label.split(" b=")[0]
    return (SAMPLER_ORDER.get(base, 99), label)


def load_histories(detailed: pd.DataFrame, repo_root: Path, steps_per_epoch: int) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    unresolved = 0
    unresolved_examples: list[str] = []

    for _, row in detailed.iterrows():
        path = resolve_history_path(row.get("history_path"), repo_root)
        if path is None:
            unresolved += 1
            if len(unresolved_examples) < 10:
                unresolved_examples.append(str(row.get("history_path")))
            continue

        try:
            hist = pd.read_csv(path)
        except Exception:
            unresolved += 1
            if len(unresolved_examples) < 10:
                unresolved_examples.append(str(path))
            continue

        if hist.empty:
            continue

        hist = numericize(hist)
        hist["dataset_name"] = row.get("dataset_name", "unknown_dataset")
        hist["weighting_regime"] = row.get("weighting_regime", "unknown")
        hist["sampler_name"] = row.get("sampler_name", "unknown")
        hist["budget_value"] = normalized_budget_value(row.get("budget_value"))
        hist["resolved_budget"] = normalized_budget_value(row.get("resolved_budget"))
        hist["seed"] = row.get("seed")

        if "epoch" in hist.columns:
            hist["cumulative_minibatch_updates"] = epoch_to_updates(hist["epoch"], steps_per_epoch)

        if "test_error" not in hist.columns and "test_accuracy" in hist.columns:
            hist["test_error"] = 1.0 - pd.to_numeric(hist["test_accuracy"], errors="coerce")
        if "val_error" not in hist.columns and "val_accuracy" in hist.columns:
            hist["val_error"] = 1.0 - pd.to_numeric(hist["val_accuracy"], errors="coerce")
        if "train_error" not in hist.columns and "train_accuracy" in hist.columns:
            hist["train_error"] = 1.0 - pd.to_numeric(hist["train_accuracy"], errors="coerce")

        frames.append(hist)

    if unresolved:
        print(f"WARNING: unresolved history paths: {unresolved}")
        for example in unresolved_examples:
            print(f"  unresolved example: {example}")

    if not frames:
        raise FileNotFoundError("No history.csv files could be resolved from history_path.")

    histories = pd.concat(frames, ignore_index=True)
    if "budget_value" not in histories.columns or histories["budget_value"].dropna().empty:
        raise ValueError("No budget_value was found in histories; cannot make per-budget convergence plots.")
    return histories


def aggregate(histories: pd.DataFrame, metric: str, keys: list[str], x_col: str) -> pd.DataFrame:
    data = histories[histories[metric].notna()].copy()
    if data.empty:
        return data
    return (
        data.groupby(keys + [x_col], dropna=False)[metric]
        .agg(mean="mean", std="std", n="count")
        .reset_index()
    )


def plot_per_budget(histories: pd.DataFrame, out_root: Path, x_col: str) -> None:
    for (dataset, regime), group in histories.groupby(["dataset_name", "weighting_regime"], dropna=False):
        dataset_slug = slugify(dataset)
        regime_slug = slugify(regime)

        for budget in sorted(group["budget_value"].dropna().unique(), key=float):
            budget_int = int(round(float(budget)))
            budget_group = group[group["budget_value"] == budget]
            group_out = out_root / dataset_slug / regime_slug / "per_budget" / f"budget_{budget_int}"
            group_out.mkdir(parents=True, exist_ok=True)

            for basename, metric, ylabel, higher_is_better in METRICS:
                if metric not in budget_group.columns:
                    continue
                agg = aggregate(budget_group, metric, ["sampler_name"], x_col)
                if agg.empty:
                    continue

                plt.figure(figsize=(9.4, 5.6))
                labels = sorted(agg["sampler_name"].dropna().astype(str).unique(), key=sampler_sort_key)
                for label in labels:
                    sub = agg[agg["sampler_name"].astype(str) == label].sort_values(x_col)
                    x = sub[x_col].astype(float)
                    y = sub["mean"].astype(float)
                    plt.plot(x, y, marker="o", linewidth=2.0, markersize=3.5, label=label)
                    if "std" in sub.columns and sub["std"].notna().any():
                        std = sub["std"].fillna(0).astype(float)
                        plt.fill_between(x, y - std, y + std, alpha=0.12)

                plt.xlabel("Cumulative mini-batch updates")
                plt.ylabel(ylabel)
                direction = "higher is better" if higher_is_better else "lower is better"
                plt.title(f"{dataset} | {regime} | budget={budget_int} | {ylabel} ({direction})")
                plt.grid(True, alpha=0.25)
                plt.legend(fontsize=8)
                plt.tight_layout()
                plt.savefig(group_out / f"{basename}_budget_{budget_int}.png", dpi=180, bbox_inches="tight")
                plt.close()


def plot_all_budgets_same_plot(histories: pd.DataFrame, out_root: Path, x_col: str) -> None:
    for (dataset, regime), group in histories.groupby(["dataset_name", "weighting_regime"], dropna=False):
        dataset_slug = slugify(dataset)
        regime_slug = slugify(regime)
        group = group.copy()
        group["sampler_budget_label"] = group.apply(
            lambda r: f"{r.get('sampler_name')} b={int(round(float(r.get('budget_value'))))}"
            if pd.notna(r.get("budget_value")) else str(r.get("sampler_name")),
            axis=1,
        )
        group_out = out_root / dataset_slug / regime_slug / "all_budgets_same_plot"
        group_out.mkdir(parents=True, exist_ok=True)

        for basename, metric, ylabel, higher_is_better in METRICS:
            if metric not in group.columns:
                continue
            agg = aggregate(group, metric, ["sampler_budget_label"], x_col)
            if agg.empty:
                continue

            plt.figure(figsize=(11.5, 6.4))
            labels = sorted(agg["sampler_budget_label"].dropna().astype(str).unique(), key=sampler_sort_key)
            for label in labels:
                sub = agg[agg["sampler_budget_label"].astype(str) == label].sort_values(x_col)
                plt.plot(
                    sub[x_col].astype(float),
                    sub["mean"].astype(float),
                    linewidth=1.5,
                    alpha=0.9,
                    label=label,
                )

            plt.xlabel("Cumulative mini-batch updates")
            plt.ylabel(ylabel)
            direction = "higher is better" if higher_is_better else "lower is better"
            plt.title(f"{dataset} | {regime} | all budgets | {ylabel} ({direction})")
            plt.grid(True, alpha=0.25)
            plt.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)
            plt.tight_layout()
            plt.savefig(group_out / f"{basename}_all_budgets.png", dpi=180, bbox_inches="tight")
            plt.close()


def plot_by_sampler_budget_sweep(histories: pd.DataFrame, out_root: Path, x_col: str) -> None:
    for (dataset, regime, sampler), group in histories.groupby(["dataset_name", "weighting_regime", "sampler_name"], dropna=False):
        dataset_slug = slugify(dataset)
        regime_slug = slugify(regime)
        sampler_slug = slugify(sampler)
        group_out = out_root / dataset_slug / regime_slug / "by_sampler_budget_sweep" / sampler_slug
        group_out.mkdir(parents=True, exist_ok=True)

        for basename, metric, ylabel, higher_is_better in METRICS:
            if metric not in group.columns:
                continue
            agg = aggregate(group, metric, ["budget_value"], x_col)
            if agg.empty:
                continue

            plt.figure(figsize=(9.4, 5.6))
            budgets = sorted(agg["budget_value"].dropna().unique(), key=float)
            for budget in budgets:
                sub = agg[agg["budget_value"] == budget].sort_values(x_col)
                b = int(round(float(budget)))
                plt.plot(sub[x_col].astype(float), sub["mean"].astype(float), marker="o", linewidth=1.8, markersize=3.2, label=f"budget={b}")

            plt.xlabel("Cumulative mini-batch updates")
            plt.ylabel(ylabel)
            direction = "higher is better" if higher_is_better else "lower is better"
            plt.title(f"{dataset} | {regime} | {sampler} | budget sweep | {ylabel} ({direction})")
            plt.grid(True, alpha=0.25)
            plt.legend(fontsize=8)
            plt.tight_layout()
            plt.savefig(group_out / f"{basename}_budget_sweep.png", dpi=180, bbox_inches="tight")
            plt.close()


def print_outputs(out_root: Path) -> None:
    pngs = sorted(out_root.rglob("*.png"))
    print(f"Wrote {len(pngs)} PNG files under {out_root}")
    for p in pngs[:40]:
        print(" ", p)
    if len(pngs) > 40:
        print(f"  ... and {len(pngs) - 40} more")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create per-budget convergence plots from chunked online benchmark outputs.")
    parser.add_argument("--root", type=Path, required=True, help="Root containing chunked outputs, e.g. outputs/active/grid/chunked/online_expected_q")
    parser.add_argument("--out", type=Path, required=True, help="Output directory for per-budget convergence plots")
    parser.add_argument("--dataset", default=None, help="Optional dataset filter, e.g. roman-empire")
    parser.add_argument("--steps-per-epoch", type=int, default=50, help="Epoch-to-cumulative-mini-batch-update multiplier")
    parser.add_argument("--skip-all-budgets", action="store_true", help="Skip all-budgets-in-one-plot outputs")
    parser.add_argument("--skip-by-sampler", action="store_true", help="Skip per-sampler budget-sweep outputs")
    args = parser.parse_args()

    repo_root = Path.cwd()
    root = args.root if args.root.is_absolute() else repo_root / args.root
    out_root = args.out if args.out.is_absolute() else repo_root / args.out

    detailed = read_detailed(root, args.dataset, repo_root)
    histories = load_histories(detailed, repo_root, args.steps_per_epoch)

    x_col = "cumulative_minibatch_updates" if "cumulative_minibatch_updates" in histories.columns else "epoch"
    print(f"Detailed rows: {len(detailed)}")
    print(f"History rows: {len(histories)}")
    print(f"Datasets: {sorted(map(str, histories['dataset_name'].dropna().unique()))}")
    print(f"Regimes: {sorted(map(str, histories['weighting_regime'].dropna().unique()))}")
    print(f"Budgets: {sorted(int(round(float(x))) for x in histories['budget_value'].dropna().unique())}")
    print(f"x-axis: {x_col}")

    plot_per_budget(histories, out_root, x_col)
    if not args.skip_all_budgets:
        plot_all_budgets_same_plot(histories, out_root, x_col)
    if not args.skip_by_sampler:
        plot_by_sampler_budget_sweep(histories, out_root, x_col)

    print_outputs(out_root)


if __name__ == "__main__":
    main()
