from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from gnn_graph_sampling.experiments.benchmark import write_benchmark_plots_from_rows  # noqa: E402
from gnn_graph_sampling.experiments.common import aggregate_rows  # noqa: E402
from gnn_graph_sampling.utils.io import write_csv, write_json  # noqa: E402


def slugify(value: object) -> str:
    text = str(value).strip().lower().replace('/', '_').replace('-', '_')
    text = re.sub(r'[^a-z0-9._-]+', '_', text)
    return text.strip('_') or 'unknown'


def infer_weighting_regime(path: Path) -> str:
    parts = set(path.parts)
    if 'normalized' in parts:
        return 'normalized'
    if 'unweighted' in parts:
        return 'unweighted'
    return 'unknown'


def _resolve_existing_path(raw: object, *, repo_root: Path) -> Path | None:
    if raw is None or pd.isna(raw):
        return None
    text = str(raw).strip()
    if not text:
        return None
    path = Path(text)
    candidates = [path] if path.is_absolute() else [repo_root / path, path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def infer_weighting_regime_from_yaml(path: Path | None) -> str:
    if path is None or not path.exists():
        return 'unknown'
    try:
        payload = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
    except Exception:
        return 'unknown'
    if not isinstance(payload, dict):
        return 'unknown'

    optimization = payload.get('optimization')
    if isinstance(optimization, dict):
        if bool(optimization.get('use_normalized_weights', False)) or bool(
            optimization.get('use_static_weight_normalization', False)
        ):
            return 'normalized'
        return 'unweighted'
    return 'unknown'


def infer_weighting_regime_for_frame(path: Path, df: pd.DataFrame, *, repo_root: Path) -> str:
    path_regime = infer_weighting_regime(path)
    if path_regime != 'unknown':
        return path_regime

    for col in ('resolved_config_path', 'config_path'):
        if col not in df.columns:
            continue
        for raw in df[col].dropna().unique():
            regime = infer_weighting_regime_from_yaml(_resolve_existing_path(raw, repo_root=repo_root))
            if regime != 'unknown':
                return regime

    normalized_signal_cols = [
        'mean_online_normalized_weighting_enabled',
        'mean_used_train_sample_weight',
        'mean_static_weight_normalization_enabled',
    ]
    for col in normalized_signal_cols:
        if col in df.columns and df[col].notna().any():
            numeric = pd.to_numeric(df[col], errors='coerce')
            if numeric.notna().any():
                return 'normalized' if float(numeric.max()) > 0.5 else 'unweighted'

    return 'unknown'


def numericize_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        converted = pd.to_numeric(df[col], errors='coerce')
        if converted.notna().sum() > 0:
            df[col] = converted
    return df


def parse_float_list(value: str | None) -> set[float] | None:
    if value is None or not value.strip():
        return None

    return {
        float(item.strip())
        for item in value.split(',')
        if item.strip()
    }


def filter_adaptive_alpha_rows(
    df: pd.DataFrame,
    *,
    min_alpha: float | None = None,
    max_alpha: float | None = None,
    alpha_list: set[float] | None = None,
) -> pd.DataFrame:
    if df.empty or 'adaptive_alpha' not in df.columns or 'sampler_name' not in df.columns:
        return df

    data = df.copy()

    is_adaptive = data['sampler_name'].astype(str) == 'adaptive_wilson'
    alpha = pd.to_numeric(data['adaptive_alpha'], errors='coerce')

    keep = pd.Series(True, index=data.index)

    if min_alpha is not None:
        keep &= ~(is_adaptive & alpha.notna() & (alpha < float(min_alpha)))

    if max_alpha is not None:
        keep &= ~(is_adaptive & alpha.notna() & (alpha > float(max_alpha)))

    if alpha_list is not None:
        rounded_allowed = {round(float(value), 12) for value in alpha_list}
        rounded_alpha = alpha.round(12)
        keep &= ~(is_adaptive & alpha.notna() & ~rounded_alpha.isin(rounded_allowed))

    return data[keep].copy()


def normalize_dataset_name(value: object) -> str:
    return slugify(str(value))


def find_detailed_csvs(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob('benchmark_detailed_runs.csv')
        if '_combined_for_plots' not in path.parts
        and '__MACOSX' not in path.parts
        and not path.name.startswith('._')
    )


def read_detailed_rows(root: Path, *, dataset: str | None, repo_root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    dataset_slug = normalize_dataset_name(dataset) if dataset else None

    for path in find_detailed_csvs(root):
        df = pd.read_csv(path)
        if df.empty:
            continue
        df = numericize_frame(df)
        weighting_regime = infer_weighting_regime_for_frame(path, df, repo_root=repo_root)
        metadata = pd.DataFrame(
            {
                'source_detailed_path': str(path),
                'source_chunk_root': str(path.parent.parent),
                'weighting_regime': weighting_regime,
            },
            index=df.index,
        )
        df = pd.concat([df.copy(), metadata], axis=1)
        if dataset_slug is not None and 'dataset_name' in df.columns:
            df = df[df['dataset_name'].map(normalize_dataset_name) == dataset_slug]
        if not df.empty:
            frames.append(df)

    if not frames:
        raise FileNotFoundError(f'No benchmark_detailed_runs.csv rows found under {root} for dataset={dataset!r}.')

    return pd.concat(frames, ignore_index=True)


def rows_from_frame(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in df.to_dict(orient='records'):
        clean: dict[str, Any] = {}
        for key, value in row.items():
            if pd.isna(value):
                clean[key] = None
            elif hasattr(value, 'item'):
                clean[key] = value.item()
            else:
                clean[key] = value
        rows.append(clean)
    return rows


def summary_group_keys(rows: list[dict[str, Any]]) -> list[str]:
    preferred = [
        'weighting_regime',
        'dataset_name',
        'dataset_kind',
        'feature_mode',
        'planetoid_split',
        'split_idx',
        'split_mode',
        'training_objective',
        'model_family',
        'sampler_name',
        'adaptive_alpha',
        'sampling_mode',
        'online_schedule',
        'budget_unit',
        'budget_value',
        'batch_size',
        'num_hops',
        'hidden_dim',
        'num_layers',
        'dropout',
        'aggr',
        'jk_mode',
        'head_hidden_dim',
        'q',
    ]
    available = set().union(*(row.keys() for row in rows)) if rows else set()
    return [key for key in preferred if key in available]


def split_rows_by_regime_and_dataset(rows: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        regime = str(row.get('weighting_regime') or 'unknown')
        dataset = slugify(row.get('dataset_name') or 'unknown_dataset')
        grouped.setdefault((regime, dataset), []).append(row)
    return grouped


def write_combined_tables(
    rows: list[dict[str, Any]],
    combined_root: Path,
    *,
    write_benchmark_plots: bool,
) -> None:
    for (regime, dataset_slug), subset in split_rows_by_regime_and_dataset(rows).items():
        run_like_root = combined_root / regime / dataset_slug
        table_dir = run_like_root / 'tables'
        table_dir.mkdir(parents=True, exist_ok=True)
        keys = summary_group_keys(subset)
        summary = aggregate_rows(subset, group_keys=keys)
        write_csv(table_dir / 'benchmark_detailed_runs.csv', subset)
        write_csv(table_dir / 'benchmark_summary.csv', summary)
        write_json(table_dir / 'benchmark_detailed_runs.json', {'detailed_results': subset})
        write_json(table_dir / 'benchmark_summary.json', {'summary_rows': summary})
        if write_benchmark_plots:
            write_benchmark_plots_from_rows(flat_rows=subset, summary_rows=summary, root=run_like_root)


def run_manual_plotter(*, repo_root: Path, combined_root: Path, plot_out: Path, steps_per_epoch: int) -> None:
    command = [
        sys.executable,
        'scripts/plot_extra_metrics.py',
        '--root',
        str(combined_root),
        '--out',
        str(plot_out),
        '--steps-per-epoch',
        str(steps_per_epoch),
    ]
    subprocess.run(command, cwd=repo_root, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Combine chunked benchmark folders for a dataset and generate plots manually after experiments.'
    )
    parser.add_argument('--root', type=Path, required=True, help='Root containing chunked benchmark output folders.')
    parser.add_argument('--dataset', default=None, help='Optional dataset filter, e.g. roman-empire.')
    parser.add_argument(
        '--out',
        type=Path,
        default=Path('outputs/active/grid/manual_plots'),
        help='Output directory for combined tables and plots.',
    )
    parser.add_argument('--steps-per-epoch', type=int, default=50)
    parser.add_argument('--skip-plots', action='store_true', help='Skip scripts/plot_extra_metrics.py plots.')
    parser.add_argument(
        '--skip-benchmark-plots',
        action='store_true',
        help='Skip the original run_benchmark.py plot family on the combined tables.',
    )
    parser.add_argument(
        '--min-adaptive-alpha',
        type=float,
        default=None,
        help='Drop adaptive_wilson rows with adaptive_alpha below this value.',
    )
    parser.add_argument(
        '--max-adaptive-alpha',
        type=float,
        default=None,
        help='Drop adaptive_wilson rows with adaptive_alpha above this value.',
    )
    parser.add_argument(
        '--adaptive-alpha-list',
        type=str,
        default=None,
        help='Comma-separated adaptive_wilson alpha values to keep, e.g. 0.3,0.5,0.7,1,1.5,2,3,5,7,9,10.',
    )
    args = parser.parse_args()

    repo_root = Path.cwd()
    root = args.root if args.root.is_absolute() else repo_root / args.root
    out_root = args.out if args.out.is_absolute() else repo_root / args.out
    dataset_part = slugify(args.dataset or 'all_datasets')
    combined_root = out_root / dataset_part / '_combined_for_plots'
    plot_out = out_root / dataset_part / 'plots'

    alpha_list = parse_float_list(args.adaptive_alpha_list)

    detailed = read_detailed_rows(root, dataset=args.dataset, repo_root=repo_root)
    original_num_rows = len(detailed)

    detailed = filter_adaptive_alpha_rows(
        detailed,
        min_alpha=args.min_adaptive_alpha,
        max_alpha=args.max_adaptive_alpha,
        alpha_list=alpha_list,
    )

    if detailed.empty:
        raise ValueError(
            'No rows remain after adaptive alpha filtering. '
            'Relax --min-adaptive-alpha, --max-adaptive-alpha, or --adaptive-alpha-list.'
        )

    rows = rows_from_frame(detailed)
    write_combined_tables(
        rows,
        combined_root,
        write_benchmark_plots=not bool(args.skip_benchmark_plots),
    )

    manifest = {
        'source_root': str(root),
        'dataset': args.dataset,
        'num_detailed_rows_before_filter': int(original_num_rows),
        'num_detailed_rows_after_filter': int(len(rows)),
        'min_adaptive_alpha': args.min_adaptive_alpha,
        'max_adaptive_alpha': args.max_adaptive_alpha,
        'adaptive_alpha_list': sorted(alpha_list) if alpha_list is not None else None,
        'combined_root': str(combined_root),
        'plot_out': str(plot_out),
        'manual_plotter': 'scripts/plot_extra_metrics.py',
        'plot_extra_metrics_is_called': not bool(args.skip_plots),
        'original_run_benchmark_plots_generated': not bool(args.skip_benchmark_plots),
        'original_run_benchmark_plot_location': str(combined_root / '<weighting_regime>' / '<dataset>' / 'plots'),
    }
    write_json(out_root / dataset_part / 'combined_manifest.json', manifest)

    if not args.skip_plots:
        run_manual_plotter(
            repo_root=repo_root,
            combined_root=combined_root,
            plot_out=plot_out,
            steps_per_epoch=int(args.steps_per_epoch),
        )

    print(f'Combined rows before filter: {original_num_rows}')
    print(f'Combined rows after filter: {len(rows)}')
    print(f'Combined tables: {combined_root}')
    if not args.skip_benchmark_plots:
        print(f'Original run_benchmark-style plots: {combined_root}/<weighting_regime>/<dataset>/plots')
    if not args.skip_plots:
        print(f'Extra metric plots: {plot_out}')


if __name__ == '__main__':
    main()