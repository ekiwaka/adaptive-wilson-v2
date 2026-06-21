from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
import yaml

from gnn_graph_sampling.config.loader import load_config
from gnn_graph_sampling.config.schema import BenchmarkConfig, DatasetConfig
from gnn_graph_sampling.data.loading import load_dataset
from gnn_graph_sampling.utils.random import seed_everything


def _tensor_sha256(values: torch.Tensor) -> str:
    ordered = torch.sort(values.detach().cpu().to(torch.long)).values.contiguous()
    return hashlib.sha256(ordered.numpy().astype('int64', copy=False).tobytes()).hexdigest()


def _dataset_config(config: BenchmarkConfig, *, dataset_name: str, feature_mode: str, split_idx: int) -> DatasetConfig:
    return DatasetConfig(
        root=config.data.root,
        name=dataset_name,
        feature_mode=feature_mode,
        split_idx=int(split_idx),
        planetoid_split=config.data.planetoid_split,
        standardize_features=config.data.standardize_features,
        split=config.data.split,
    )


def _plain_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text()) or {}
    if not isinstance(payload, dict):
        raise ValueError(f'{path} must contain a mapping')
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Verify online split-sweep configs use q tables matching their exact train split.'
    )
    parser.add_argument('--config-root', type=Path, default=Path('configs/active/grid/online_split_sweep'))
    parser.add_argument('--config', type=Path, action='append', default=None)
    parser.add_argument('--budgets', nargs='*', default=['32', '64', '128', '256', '512'])
    parser.add_argument('--max-configs', type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.config:
        configs = [Path(p) for p in args.config]
    else:
        configs = sorted(args.config_root.rglob('*.yaml'))
    if args.max_configs is not None:
        configs = configs[: int(args.max_configs)]
    if not configs:
        raise SystemExit(f'No configs found under {args.config_root}')

    expected_budgets = {str(int(float(b))) for b in args.budgets}
    failures: list[str] = []

    for idx, config_path in enumerate(configs, start=1):
        print()
        print(f'[{idx}/{len(configs)}] {config_path}')
        raw = _plain_yaml(config_path)
        config = load_config(BenchmarkConfig, config_path)
        q_path = Path(config.sampling.wilson.q_table_path or '')
        print(f'  q_table_path: {q_path}')
        print(f'  split_file:   {config.data.split.split_file}')
        print(f'  split ratios: train={config.data.split.train_ratio} val={config.data.split.val_ratio} test={config.data.split.test_ratio}')

        if not q_path.exists():
            failures.append(f'{config_path}: missing q table {q_path}')
            print('  ERROR: missing q table')
            continue

        seed_everything(int(config.runtime.seed))
        dataset_name = str(config.data.datasets[0])
        feature_mode = str(config.data.feature_modes[0])
        split_idx = int(config.data.split_idx)
        dataset = load_dataset(
            _dataset_config(config, dataset_name=dataset_name, feature_mode=feature_mode, split_idx=split_idx)
        )

        train_nodes = dataset.train_nodes.detach().cpu().to(torch.long)
        val_nodes = dataset.val_nodes.detach().cpu().to(torch.long)
        test_nodes = dataset.test_nodes.detach().cpu().to(torch.long)
        actual = {
            'train_node_count': int(train_nodes.numel()),
            'val_node_count': int(val_nodes.numel()),
            'test_node_count': int(test_nodes.numel()),
            'train_node_sha256': _tensor_sha256(train_nodes),
            'val_node_sha256': _tensor_sha256(val_nodes),
            'test_node_sha256': _tensor_sha256(test_nodes),
        }

        payload = json.loads(q_path.read_text())
        datasets = payload.get('datasets')
        if not isinstance(datasets, dict) or not datasets:
            failures.append(f'{config_path}: q table missing datasets object')
            continue
        if dataset_name in datasets:
            dataset_key = dataset_name
        elif len(datasets) == 1:
            dataset_key = next(iter(datasets))
        else:
            failures.append(f'{config_path}: dataset {dataset_name!r} not found in q table keys={list(datasets)}')
            continue
        entry = datasets[dataset_key]

        qtab = entry.get('q_by_budget_count')
        if not isinstance(qtab, dict):
            failures.append(f'{config_path}: q table missing q_by_budget_count')
            continue
        budgets = set(str(k) for k in qtab.keys())
        missing = expected_budgets - budgets
        extra = budgets - expected_budgets
        if missing:
            failures.append(f'{config_path}: missing q budgets {sorted(missing, key=int)}')
        if extra:
            failures.append(f'{config_path}: unexpected q budgets {sorted(extra, key=int)}')

        for key, value in actual.items():
            expected = entry.get(key)
            if expected is None:
                failures.append(f'{config_path}: q table missing metadata field {key}')
                continue
            if key.endswith('_count'):
                ok = int(expected) == int(value)
            else:
                ok = str(expected) == str(value)
            if not ok:
                failures.append(f'{config_path}: {key} mismatch q_table={expected} current={value}')

        print(f"  train nodes: {actual['train_node_count']} hash={actual['train_node_sha256'][:16]}...")
        print(f"  val nodes:   {actual['val_node_count']} hash={actual['val_node_sha256'][:16]}...")
        print(f"  test nodes:  {actual['test_node_count']} hash={actual['test_node_sha256'][:16]}...")
        print(f"  q budgets:   {sorted(budgets, key=lambda x: int(float(x)))}")

    print()
    if failures:
        print('FAILURES:')
        for failure in failures:
            print(' -', failure)
        raise SystemExit(1)
    print(f'All {len(configs)} configs have matching q-table split metadata.')


if __name__ == '__main__':
    main()
