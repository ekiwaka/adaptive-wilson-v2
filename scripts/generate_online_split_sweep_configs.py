from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


DEFAULT_BASE_CONFIGS = [
    'configs/active/grid/online_static_informed/normalized/roman_empire__all_samplers_fixed_batch.yaml',
    'configs/active/grid/online_static_informed/unweighted/roman_empire__all_samplers_fixed_batch.yaml',
    'configs/active/grid/online_static_informed/normalized/amazon_ratings__all_samplers_fixed_batch.yaml',
    'configs/active/grid/online_static_informed/unweighted/amazon_ratings__all_samplers_fixed_batch.yaml',
    'configs/active/grid/online_static_informed/normalized/pubmed__all_samplers_fixed_batch.yaml',
    'configs/active/grid/online_static_informed/unweighted/pubmed__all_samplers_fixed_batch.yaml',
]


DATASET_ALIASES = {
    'roman_empire': 'roman-empire',
    'amazon_ratings': 'amazon-ratings',
    'pubmed': 'pubmed',
}


def ratio_label(value: float) -> str:
    return f'{float(value):.2f}'.replace('.', 'p')


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open('r', encoding='utf-8') as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f'{path} must contain a YAML mapping.')
    return payload


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def deep_copy_yaml(payload: dict[str, Any]) -> dict[str, Any]:
    return yaml.safe_load(yaml.safe_dump(payload, sort_keys=False))


def infer_weighting(config_path: Path) -> str:
    parts = set(config_path.parts)
    if 'normalized' in parts:
        return 'normalized'
    if 'unweighted' in parts:
        return 'unweighted'
    raise ValueError(f'Could not infer weighting regime from path: {config_path}')


def dataset_slug_from_config_path(config_path: Path) -> str:
    return config_path.name.split('__', 1)[0]


def build_config(
    base: dict[str, Any],
    *,
    source_path: Path,
    train_ratio: float,
    val_ratio: float,
    split_seed: int,
    output_root: str,
    q_root: str,
) -> dict[str, Any]:
    cfg = deep_copy_yaml(base)
    weighting = infer_weighting(source_path)
    dataset_slug = dataset_slug_from_config_path(source_path)
    dataset_name = DATASET_ALIASES.get(dataset_slug, dataset_slug.replace('_', '-'))
    tlabel = ratio_label(train_ratio)
    vlabel = ratio_label(val_ratio)
    split_label = f'train{tlabel}'
    test_ratio = 1.0 - float(train_ratio) - float(val_ratio)
    if test_ratio <= 0.0:
        raise ValueError(f'Invalid train/val ratios: train={train_ratio}, val={val_ratio}')

    cfg['study_name'] = f'{dataset_slug}_{weighting}_online_expected_q_{split_label}_val{vlabel}'

    output = cfg.setdefault('output', {})
    # Logical config output root. The chunk submitter will still override the
    # concrete run output root per chunk, but keeping this path split-first makes
    # direct runs and smoke runs easy to reason about.
    output['root'] = f'{output_root}/{split_label}/{dataset_slug}/{weighting}'
    output['experiment_name'] = f'{dataset_slug}_{weighting}_online_expected_q_{split_label}_val{vlabel}'
    output['overwrite'] = False
    # Chunked jobs add one timestamp parent folder. Avoid per-run timestamp
    # fragmentation inside each chunk.
    output['append_timestamp'] = False

    data = cfg.setdefault('data', {})
    data['datasets'] = [dataset_name]
    split = data.setdefault('split', {})
    split['mode'] = 'generated'
    split['train_ratio'] = float(train_ratio)
    split['val_ratio'] = float(val_ratio)
    split['test_ratio'] = float(test_ratio)
    split['seed'] = int(split_seed)
    split['split_file'] = (
        f'outputs/splits/online_split_sweep/{split_label}/{dataset_slug}/'
        f'generated_{split_label}_val{vlabel}_seed{int(split_seed)}.pt'
    )
    split['save_generated_split'] = True
    split['reuse_existing_split'] = True

    sampling = cfg.setdefault('sampling', {})
    sampling['budget_unit'] = 'count'
    sampling['budget_values'] = [32, 64, 128, 256, 512]
    # Kept as a single placeholder to avoid creating a batch-size x budget cartesian product.
    sampling['batch_sizes'] = [64]

    wilson = sampling.setdefault('wilson', {})
    wilson['auto_q_method'] = 'precomputed_table'
    wilson['q_estimation'] = None
    wilson['q_table_path'] = (
        f'{q_root}/{split_label}/{dataset_slug}/{weighting}/'
        f'{dataset_slug}_{weighting}_{split_label}_val{vlabel}_seed{int(split_seed)}_q_table.json'
    )
    wilson['q_table_missing_policy'] = 'error'
    wilson['budget_targeting'] = 'expected_budget'
    wilson['redraw_on_shortfall'] = False
    wilson['allow_uniform_fill'] = False
    wilson['avoid_padding'] = True
    wilson['avoid_capping'] = True
    wilson['max_sampling_attempts'] = 1

    adaptive = sampling.setdefault('adaptive_wilson', {})
    adaptive['q_mode'] = 'precomputed_budget_q'

    return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate online expected-q split-ratio sweep configs.')
    parser.add_argument(
        '--base-config',
        action='append',
        default=None,
        help='Base config to transform. May be supplied multiple times. Defaults to all six online_static_informed configs.',
    )
    parser.add_argument('--train-ratios', nargs='+', type=float, default=[0.10, 0.20, 0.30, 0.40])
    parser.add_argument('--val-ratio', type=float, default=0.20)
    parser.add_argument('--split-seed', type=int, default=0)
    parser.add_argument('--out-dir', default='configs/active/grid/online_split_sweep')
    parser.add_argument('--output-root', default='outputs/active/grid/chunked/online_split_sweep')
    parser.add_argument('--q-root', default='outputs/precomputed/online_expected_q/split_sweep')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_configs = [Path(p) for p in (args.base_config or DEFAULT_BASE_CONFIGS)]
    out_dir = Path(args.out_dir)

    written: list[Path] = []
    for base_path in base_configs:
        base = load_yaml(base_path)
        weighting = infer_weighting(base_path)
        dataset_slug = dataset_slug_from_config_path(base_path)
        for train_ratio in args.train_ratios:
            tlabel = ratio_label(train_ratio)
            split_label = f'train{tlabel}'
            cfg = build_config(
                base,
                source_path=base_path,
                train_ratio=float(train_ratio),
                val_ratio=float(args.val_ratio),
                split_seed=int(args.split_seed),
                output_root=str(args.output_root),
                q_root=str(args.q_root),
            )
            out_path = out_dir / split_label / weighting / f'{dataset_slug}__{split_label}_all_samplers_fixed_batch.yaml'
            write_yaml(out_path, cfg)
            written.append(out_path)

    for path in written:
        print(path)
    print(f'Wrote {len(written)} configs.')


if __name__ == '__main__':
    main()
