from __future__ import annotations
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt

METRICS = [
    ("train_accuracy", "Training accuracy"),
    ("test_accuracy", "Testing accuracy"),
]


def _read_csv(path: Path):
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _safe_float(value):
    if value in (None, "", "None"):
        return None
    return float(value)


def _slug(name: str) -> str:
    return name.replace("/", "_")


def _mean_std(values):
    if not values:
        return None, None
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return mean, var ** 0.5


def _gather_rows(manifest_path: Path):
    manifest = json.loads(manifest_path.read_text())
    rows = []
    for root_str in manifest.get("benchmark_roots", []):
        detailed = Path(root_str) / "tables" / "benchmark_detailed_runs.csv"
        if detailed.exists():
            rows.extend(_read_csv(detailed))
    return rows


def _plot_dataset(rows, out_dir: Path, dataset_name: str):
    ds_rows = [row for row in rows if row.get("dataset_name") == dataset_name and row.get("sampling_mode") == "static"]
    if not ds_rows:
        return

    sampled = [row for row in ds_rows if row.get("sampler_name") in {"uniform", "degree", "wilson"}]
    baseline = [row for row in ds_rows if row.get("sampler_name") == "none"]
    if not sampled:
        return

    ds_out = out_dir / _slug(dataset_name)
    ds_out.mkdir(parents=True, exist_ok=True)

    for metric_key, ylabel in METRICS:
        plt.figure(figsize=(8, 5.2))

        for sampler_name in sorted({row["sampler_name"] for row in sampled}):
            subset = [row for row in sampled if row.get("sampler_name") == sampler_name]
            grouped = defaultdict(list)
            for row in subset:
                budget = _safe_float(row.get("budget_value") or row.get("batch_size"))
                value = _safe_float(row.get(metric_key))
                if budget is None or value is None:
                    continue
                grouped[budget].append(value)
            xs, ys, es = [], [], []
            for budget in sorted(grouped):
                mean, std = _mean_std(grouped[budget])
                xs.append(budget)
                ys.append(mean)
                es.append(std)
            if xs:
                plt.plot(xs, ys, marker="o", label=sampler_name)
                lower = [y - e for y, e in zip(ys, es)]
                upper = [y + e for y, e in zip(ys, es)]
                plt.fill_between(xs, lower, upper, alpha=0.15)

        if baseline:
            vals = [_safe_float(row.get(metric_key)) for row in baseline]
            vals = [v for v in vals if v is not None]
            if vals:
                mean, std = _mean_std(vals)
                plt.axhline(mean, linestyle="--", linewidth=1.8, label="no sampler")
                plt.axhspan(mean - std, mean + std, alpha=0.12)

        plt.xlabel("Sampling budget")
        plt.ylabel(ylabel)
        plt.title(f"{dataset_name} | {ylabel} vs sampling budget")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(ds_out / f"{metric_key}_vs_budget.png", dpi=180, bbox_inches="tight")
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot static accuracy-vs-budget curves averaged over seeds and splits.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()

    rows = _gather_rows(args.manifest)
    datasets = sorted({row.get("dataset_name") for row in rows if row.get("dataset_name")})
    args.outdir.mkdir(parents=True, exist_ok=True)
    for dataset_name in datasets:
        _plot_dataset(rows, args.outdir, dataset_name)


if __name__ == "__main__":
    main()
