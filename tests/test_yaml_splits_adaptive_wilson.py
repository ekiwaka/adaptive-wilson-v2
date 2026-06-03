from __future__ import annotations

from pathlib import Path

import pytest
import torch

Data = pytest.importorskip('torch_geometric.data').Data

from gnn_graph_sampling.config.loader import load_config
from gnn_graph_sampling.config.schema import DatasetConfig, TrainConfig
from gnn_graph_sampling.data.loading import load_dataset
from gnn_graph_sampling.exceptions import ConfigurationError
from gnn_graph_sampling.sampling.wilson_sampler import AdaptiveWilsonCenterSampler


def _base_train_yaml(extra: list[str]) -> str:
    return '\n'.join(
        [
            'run_name: yaml_guard',
            'experiment:',
            '  sampling_regime: online',
            'dataset:',
            '  name: sbm_balanced',
            '  split:',
            '    mode: generated',
            '    train_ratio: 0.6',
            '    val_ratio: 0.2',
            '    test_ratio: 0.2',
            'model:',
            '  num_layers: 2',
            'sampling:',
            '  name: uniform',
            '  mode: static',
            '  num_hops: 2',
        ]
        + extra
    )


def test_yaml_experiment_sampling_regime_switches_sampling_mode(tmp_path: Path) -> None:
    path = tmp_path / 'config.yaml'
    path.write_text(_base_train_yaml([]), encoding='utf-8')
    config = load_config(TrainConfig, path)
    assert config.sampling.mode == 'online'


def test_generated_split_is_reproducible_and_reloadable(tmp_path: Path) -> None:
    split_file = tmp_path / 'split.pt'
    cfg = DatasetConfig(name='sbm_balanced')
    cfg.synthetic.sizes = [20, 20, 20]
    cfg.synthetic.num_features = 4
    cfg.split.mode = 'generated'
    cfg.split.train_ratio = 0.5
    cfg.split.val_ratio = 0.25
    cfg.split.test_ratio = 0.25
    cfg.split.seed = 123
    cfg.split.split_file = str(split_file)
    cfg.split.save_generated_split = True
    cfg.split.reuse_existing_split = True

    first = load_dataset(cfg, device=torch.device('cpu'), seed=0)
    second = load_dataset(cfg, device=torch.device('cpu'), seed=999)

    assert split_file.exists()
    assert torch.equal(first.data.train_mask, second.data.train_mask)
    assert torch.equal(first.data.val_mask, second.data.val_mask)
    assert torch.equal(first.data.test_mask, second.data.test_mask)
    assert not bool((first.data.train_mask & first.data.val_mask).any())
    assert not bool((first.data.train_mask & first.data.test_mask).any())
    assert int(first.train_nodes.numel()) > 0
    assert int(first.val_nodes.numel()) > 0
    assert int(first.test_nodes.numel()) > 0


def test_missing_custom_split_file_fails_clearly(tmp_path: Path) -> None:
    cfg = DatasetConfig(name='sbm_balanced')
    cfg.synthetic.sizes = [10, 10, 10]
    cfg.split.mode = 'custom'
    cfg.split.split_file = str(tmp_path / 'missing.pt')
    with pytest.raises(FileNotFoundError, match='Configured split file'):
        load_dataset(cfg, device=torch.device('cpu'), seed=0)


def test_invalid_generated_split_ratios_fail_in_yaml(tmp_path: Path) -> None:
    path = tmp_path / 'bad.yaml'
    path.write_text(
        '\n'.join(
            [
                'run_name: bad_ratios',
                'dataset:',
                '  name: sbm_balanced',
                '  split:',
                '    mode: generated',
                '    train_ratio: 0.5',
                '    val_ratio: 0.5',
                '    test_ratio: 0.5',
                'model:',
                '  num_layers: 2',
                'sampling:',
                '  num_hops: 2',
            ]
        ),
        encoding='utf-8',
    )
    with pytest.raises(ConfigurationError, match='sum to 1'):
        load_config(TrainConfig, path)


def test_adaptive_wilson_alpha_sampler_runs() -> None:
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    sampler = AdaptiveWilsonCenterSampler(
        edge_index=edge_index,
        num_nodes=4,
        device=torch.device('cpu'),
        backend='python',
        alpha=0.75,
        allow_uniform_fill=False,
    )
    generator = torch.Generator(device='cpu').manual_seed(7)
    out = sampler.sample(torch.arange(4), 2, generator=generator)
    assert out.center_nodes.numel() > 0
    assert out.metadata['adaptive_wilson'] == 1.0
    assert out.metadata['adaptive_alpha'] == 0.75
