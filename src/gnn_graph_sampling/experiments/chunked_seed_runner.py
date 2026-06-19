from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def slugify(value: object) -> str:
    text = str(value).strip().lower()
    text = re.sub(r'[^a-z0-9._-]+', '_', text)
    text = re.sub(r'_+', '_', text)
    return text.strip('_') or 'unknown'


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open('r', encoding='utf-8') as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f'{path} must contain a YAML mapping.')
    return dict(payload)


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)


def read_status(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {'chunks': {}}
    with path.open('r', encoding='utf-8') as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        return {'chunks': {}}
    payload.setdefault('chunks', {})
    return payload


def write_status(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + '.tmp')
    with tmp_path.open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    tmp_path.replace(path)


def chunk_ranges(*, start_seed: int, total_runs: int, chunk_size: int) -> list[tuple[int, int, int]]:
    if total_runs <= 0:
        raise ValueError('--total-runs must be positive.')
    if chunk_size <= 0:
        raise ValueError('--chunk-size must be positive.')

    chunks: list[tuple[int, int, int]] = []
    remaining = int(total_runs)
    seed = int(start_seed)
    while remaining > 0:
        size = min(int(chunk_size), remaining)
        chunks.append((seed, seed + size - 1, size))
        seed += size
        remaining -= size
    return chunks


def chunk_key_for_range(seed_start: int, seed_end: int) -> str:
    return f'{int(seed_start):04d}_{int(seed_end):04d}'


def first_unfinished_chunk_key(
    chunks: list[tuple[int, int, int]],
    status: dict[str, Any],
) -> str | None:
    chunk_status = status.get('chunks', {})
    if not isinstance(chunk_status, dict):
        return None
    for seed_start, seed_end, _ in chunks:
        key = chunk_key_for_range(seed_start, seed_end)
        previous = chunk_status.get(key, {})
        if not isinstance(previous, dict) or previous.get('status') != 'done':
            return key
    return None


def infer_dataset_slug(config: dict[str, Any], fallback: str) -> str:
    datasets = config.get('data', {}).get('datasets') or []
    if len(datasets) == 1:
        return slugify(str(datasets[0]).replace('/', '_').replace('-', '_'))
    return slugify(fallback)


def infer_weighting_regime(config: dict[str, Any], config_path: Path) -> str:
    path_parts = set(config_path.parts)
    if 'normalized' in path_parts:
        return 'normalized'
    if 'unweighted' in path_parts:
        return 'unweighted'

    optimization = config.get('optimization')
    if isinstance(optimization, dict):
        if bool(optimization.get('use_normalized_weights', False)) or bool(
            optimization.get('use_static_weight_normalization', False)
        ):
            return 'normalized'
        return 'unweighted'
    return 'unknown_weighting'


def make_chunk_config(
    base_config: dict[str, Any],
    *,
    seed_start: int,
    seed_end: int,
    run_count: int,
    output_root: Path,
    experiment_prefix: str,
    parallel_workers: int | None,
    force_cpu: bool,
    save_plots: bool,
) -> dict[str, Any]:
    cfg = json.loads(json.dumps(base_config))
    chunk_label = f'seed{seed_start:04d}_{seed_end:04d}'

    cfg['study_name'] = f"{slugify(cfg.get('study_name', experiment_prefix))}_{chunk_label}"

    output = cfg.setdefault('output', {})
    output['root'] = str(output_root)
    output['experiment_name'] = f'{slugify(experiment_prefix)}_{chunk_label}'
    output['overwrite'] = False
    output['append_timestamp'] = False

    benchmark = cfg.setdefault('benchmark', {})
    benchmark['runs'] = int(run_count)
    benchmark['base_seed'] = int(seed_start)
    benchmark['save_plots'] = bool(save_plots)
    if parallel_workers is not None:
        benchmark['parallel_workers'] = int(parallel_workers)

    if force_cpu:
        runtime = cfg.setdefault('runtime', {})
        runtime['device'] = 'cpu'
        runtime['device_policy'] = 'warn_and_fallback'
        runtime['torch_num_threads'] = 1
        evaluation = cfg.setdefault('evaluation', {})
        evaluation['device'] = 'cpu'

    return cfg


def run_chunk(
    *,
    repo_root: Path,
    chunk_config_path: Path,
    log_path: Path,
) -> int:
    command = [sys.executable, 'scripts/run_benchmark.py', '--config', str(chunk_config_path)]
    env = os.environ.copy()
    env.setdefault('PYTHONPATH', str(repo_root / 'src'))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('w', encoding='utf-8') as handle:
        handle.write('$ ' + ' '.join(command) + '\n')
        handle.flush()
        completed = subprocess.run(
            command,
            cwd=repo_root,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    return int(completed.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Run a BenchmarkConfig over chunked seed ranges and track completed chunks.'
    )
    parser.add_argument('--config', type=Path, required=True, help='Base BenchmarkConfig YAML.')
    parser.add_argument('--total-runs', type=int, default=100, help='Total number of seeds/runs to execute.')
    parser.add_argument('--chunk-size', type=int, default=25, help='Number of seeds per chunk.')
    parser.add_argument('--start-seed', type=int, default=0, help='First seed in the full run.')
    parser.add_argument(
        '--output-root',
        type=Path,
        default=Path('outputs/active/grid/roazhon13/chunked'),
        help='Root where chunk outputs, generated chunk YAMLs, logs, and status are stored.',
    )
    parser.add_argument('--experiment-prefix', default=None, help='Optional prefix for per-chunk output folders.')
    parser.add_argument('--parallel-workers', type=int, default=None, help='Override benchmark.parallel_workers.')
    parser.add_argument('--status-file', type=Path, default=None, help='Optional status JSON path.')
    parser.add_argument('--force', action='store_true', help='Rerun chunks even if status says they are done.')
    parser.add_argument('--force-cpu', action='store_true', default=True, help='Force runtime/evaluation device to CPU.')
    parser.add_argument('--allow-plots-during-run', action='store_true', help='Do not disable benchmark plots.')
    args = parser.parse_args()

    repo_root = Path.cwd()
    base_config_path = args.config if args.config.is_absolute() else repo_root / args.config
    base_config = load_yaml(base_config_path)
    dataset_slug = infer_dataset_slug(base_config, base_config_path.stem)
    weighting_regime = infer_weighting_regime(base_config, base_config_path)
    experiment_prefix = args.experiment_prefix or f'{dataset_slug}_{weighting_regime}_{slugify(base_config_path.stem)}'

    output_root = args.output_root if args.output_root.is_absolute() else repo_root / args.output_root
    configs_root = output_root / '_chunk_configs' / slugify(experiment_prefix)
    logs_root = output_root / '_chunk_logs' / slugify(experiment_prefix)
    status_file = args.status_file or (output_root / '_chunk_status' / f'{slugify(experiment_prefix)}.json')
    tracker_existed_at_start = status_file.exists()
    status = read_status(status_file)
    chunks = chunk_ranges(
        start_seed=args.start_seed,
        total_runs=args.total_runs,
        chunk_size=args.chunk_size,
    )
    resume_key = first_unfinished_chunk_key(chunks, status) if tracker_existed_at_start else None
    status.update(
        {
            'base_config': str(base_config_path),
            'experiment_prefix': slugify(experiment_prefix),
            'weighting_regime': weighting_regime,
            'total_runs': int(args.total_runs),
            'chunk_size': int(args.chunk_size),
            'start_seed': int(args.start_seed),
            'updated_at': utc_now(),
            'tracker_existed_at_start': bool(tracker_existed_at_start),
            'resume_from_chunk': resume_key,
        }
    )
    status.setdefault('chunks', {})

    if tracker_existed_at_start:
        if resume_key is None:
            print(f'[resume] tracker exists and all chunks are already marked done: {status_file}')
        else:
            print(f'[resume] tracker exists: {status_file}')
            print(f'[resume] first unfinished chunk is {resume_key}; earlier done chunks will be skipped')
    else:
        print(f'[start] no tracker found; creating: {status_file}')

    failures: list[str] = []
    for seed_start, seed_end, run_count in chunks:
        chunk_key = chunk_key_for_range(seed_start, seed_end)
        previous = status['chunks'].get(chunk_key, {})
        if previous.get('status') == 'done' and not args.force:
            print(f'[skip] chunk {chunk_key} already done')
            continue

        chunk_config = make_chunk_config(
            base_config,
            seed_start=seed_start,
            seed_end=seed_end,
            run_count=run_count,
            output_root=output_root,
            experiment_prefix=experiment_prefix,
            parallel_workers=args.parallel_workers,
            force_cpu=bool(args.force_cpu),
            save_plots=bool(args.allow_plots_during_run),
        )
        chunk_config_path = configs_root / f'{chunk_key}.yaml'
        log_path = logs_root / f'{chunk_key}.log'
        write_yaml(chunk_config_path, chunk_config)

        previous_status = previous.get('status') if isinstance(previous, dict) else None
        previous_attempts = int(previous.get('attempts', 0)) if isinstance(previous, dict) else 0
        status['chunks'][chunk_key] = {
            'status': 'running',
            'previous_status': previous_status,
            'attempts': previous_attempts + 1,
            'seed_start': seed_start,
            'seed_end': seed_end,
            'runs': run_count,
            'config': str(chunk_config_path),
            'log': str(log_path),
            'started_at': utc_now(),
        }
        write_status(status_file, status)

        print(f'[run] chunk {chunk_key}: seeds {seed_start}-{seed_end}')
        return_code = run_chunk(repo_root=repo_root, chunk_config_path=chunk_config_path, log_path=log_path)
        chunk_record = status['chunks'][chunk_key]
        chunk_record['finished_at'] = utc_now()
        chunk_record['return_code'] = return_code
        if return_code == 0:
            chunk_record['status'] = 'done'
            print(f'[done] chunk {chunk_key}')
        else:
            chunk_record['status'] = 'failed'
            failures.append(chunk_key)
            print(f'[failed] chunk {chunk_key}; see {log_path}')
        status['updated_at'] = utc_now()
        write_status(status_file, status)

        if return_code != 0:
            break

    if failures:
        raise SystemExit(f'Failed chunks: {", ".join(failures)}')

    print(f'Status written to: {status_file}')


if __name__ == '__main__':
    main()
