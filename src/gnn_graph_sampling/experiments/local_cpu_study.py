from __future__ import annotations
import argparse
import csv
from pathlib import Path
from typing import Any
import yaml

from ..config.loader import load_config
from ..config.schema import BenchmarkConfig, DiagnosticsConfig, QSweepConfig, SearchConfig, TrainConfig
from ..utils.io import ensure_dir, write_json
from ..utils.logging import configure_logging, get_logger
from .benchmark import run_benchmark_study
from .diagnostics import run_diagnostics_study
from .q_sweep import run_q_sweep_study
from .reporting import generate_sampler_report
from .search import run_search_study
from .train import run_training_experiment

LOGGER = get_logger(__name__)


def _resolve_paths(values: list[str], *, base: Path) -> list[Path]:
    return [(base / value).resolve() if not Path(value).is_absolute() else Path(value) for value in values]


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open('r') as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f'Study config must decode to a mapping: {path}')
    return data


def _best_wilson_q_for_root(root: Path) -> tuple[str | None, float | None]:
    leaderboard = root / 'tables' / 'q_sweep_leaderboard.csv'
    if not leaderboard.exists():
        return None, None
    with leaderboard.open('r', newline='') as handle:
        for row in csv.DictReader(handle):
            if str(row.get('sampler_name')) != 'wilson':
                continue
            dataset_name = row.get('dataset_name')
            q_value = row.get('q')
            if dataset_name and q_value not in (None, '', 'None'):
                return str(dataset_name), float(q_value)
    return None, None


def run_local_cpu_study(study_config_path: Path) -> dict[str, Any]:
    study_config_path = study_config_path.resolve()
    study = _read_yaml(study_config_path)
    base = study_config_path.parent
    for parent in [study_config_path.parent, *study_config_path.parents]:
        if parent.name == 'configs':
            base = parent.parent
            break
    output_root = (base / study.get('output_root', 'outputs')).resolve()
    manifest_root = output_root / study.get('manifest_name', 'local_cpu_sampler_study')
    ensure_dir(manifest_root)
    configure_logging('INFO', manifest_root / 'study.log', log_to_stdout=True, reset_handlers=True)

    run_order = list(study.get('run_order', ['diagnostics', 'search', 'q_sweep', 'train', 'benchmark', 'report']))
    auto_use_best_q = bool(study.get('auto_use_best_q', True))

    diagnostics_configs = _resolve_paths(list(study.get('diagnostics_configs', [])), base=base)
    search_configs = _resolve_paths(list(study.get('search_configs', [])), base=base)
    q_sweep_configs = _resolve_paths(list(study.get('q_sweep_configs', [])), base=base)
    train_configs = _resolve_paths(list(study.get('train_configs', [])), base=base)
    benchmark_configs = _resolve_paths(list(study.get('benchmark_configs', [])), base=base)

    manifest: dict[str, Any] = {
        'study_config': str(study_config_path),
        'run_order': run_order,
        'diagnostics_roots': [],
        'search_roots': [],
        'q_sweep_roots': [],
        'train_roots': [],
        'benchmark_roots': [],
        'best_q_by_dataset': {},
    }

    for stage in run_order:
        if stage == 'diagnostics':
            for path in diagnostics_configs:
                LOGGER.info('Running diagnostics config: %s', path)
                root = run_diagnostics_study(load_config(DiagnosticsConfig, path))
                manifest['diagnostics_roots'].append(str(root))
        elif stage == 'search':
            for path in search_configs:
                LOGGER.info('Running search config: %s', path)
                root = run_search_study(load_config(SearchConfig, path))
                manifest['search_roots'].append(str(root))
        elif stage == 'q_sweep':
            for path in q_sweep_configs:
                LOGGER.info('Running q_sweep config: %s', path)
                root = run_q_sweep_study(load_config(QSweepConfig, path))
                manifest['q_sweep_roots'].append(str(root))
                dataset_name, q_value = _best_wilson_q_for_root(root)
                if dataset_name is not None and q_value is not None:
                    manifest['best_q_by_dataset'][dataset_name] = float(q_value)
                    LOGGER.info('Best Wilson q for dataset=%s set to %.4f from %s', dataset_name, q_value, root)
        elif stage == 'train':
            for path in train_configs:
                LOGGER.info('Running train config: %s', path)
                cfg = load_config(TrainConfig, path)
                if auto_use_best_q and cfg.sampling.name == 'wilson':
                    best_q = manifest['best_q_by_dataset'].get(cfg.dataset.name)
                    if best_q is not None:
                        cfg.sampling.wilson.q = float(best_q)
                        cfg.run_name = f"{cfg.run_name}_q{str(best_q).replace('.', 'p')}"
                _, root = run_training_experiment(cfg)
                manifest['train_roots'].append(str(root))
        elif stage == 'benchmark':
            for path in benchmark_configs:
                LOGGER.info('Running benchmark config: %s', path)
                cfg = load_config(BenchmarkConfig, path)
                if auto_use_best_q and len(cfg.data.datasets) == 1:
                    dataset_name = str(cfg.data.datasets[0])
                    best_q = manifest['best_q_by_dataset'].get(dataset_name)
                    if best_q is not None:
                        cfg.sampling.wilson.q = float(best_q)
                        cfg.study_name = f"{cfg.study_name}_q{str(best_q).replace('.', 'p')}"
                root = run_benchmark_study(cfg)
                manifest['benchmark_roots'].append(str(root))
        elif stage == 'report':
            report_root = generate_sampler_report(
                benchmark_roots=[Path(value) for value in manifest['benchmark_roots']],
                q_sweep_roots=[Path(value) for value in manifest['q_sweep_roots']],
                out_root=manifest_root / 'sampler_report',
            )
            manifest['report_root'] = str(report_root)
        else:
            raise ValueError(f'Unsupported stage in run_order: {stage}')

        write_json(manifest_root / 'manifest.json', manifest)

    LOGGER.info('Local CPU sampler study complete. Manifest written to %s', manifest_root / 'manifest.json')
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description='Run the lightweight local CPU sampler study.')
    parser.add_argument('--config', type=Path, required=True, help='Study YAML describing the ordered local workflow.')
    args = parser.parse_args()
    run_local_cpu_study(args.config)


if __name__ == '__main__':
    main()
