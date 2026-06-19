from __future__ import annotations
from pathlib import Path
import pytest
import torch

Data = pytest.importorskip('torch_geometric.data').Data

from gnn_graph_sampling.config.loader import load_config
from gnn_graph_sampling.config.schema import DatasetConfig, SamplingConfig, TrainConfig
from gnn_graph_sampling.data.loading import DatasetBundle
from gnn_graph_sampling.data.splitting import apply_sampler_split
from gnn_graph_sampling.exceptions import ConfigurationError
from gnn_graph_sampling.sampling.center_selectors import UniformCenterSampler


def test_apply_sampler_split_uses_total_node_budget_and_partitions_remaining() -> None:
    data = Data(
        x=torch.randn(10, 4),
        y=torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.long),
        edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long),
        num_nodes=10,
    )
    data.train_mask = torch.zeros(10, dtype=torch.bool)
    data.val_mask = torch.zeros(10, dtype=torch.bool)
    data.test_mask = torch.zeros(10, dtype=torch.bool)
    bundle = DatasetBundle(
        name='toy',
        kind='synthetic',
        data=data,
        num_features=4,
        num_classes=2,
        train_nodes=torch.tensor([0, 1, 2], dtype=torch.long),
        val_nodes=torch.tensor([3, 4], dtype=torch.long),
        test_nodes=torch.tensor([5, 6, 7, 8, 9], dtype=torch.long),
        metadata={},
    )
    dataset_config = DatasetConfig(name='toy')
    dataset_config.split.mode = 'sampler'
    dataset_config.split.val_ratio_of_remaining = 0.25
    sampling = SamplingConfig(name='uniform', budget_unit='ratio', budget_value=0.3)
    sampler = UniformCenterSampler()

    updated = apply_sampler_split(bundle, dataset_config, sampler, sampling, seed=0)

    assert int(updated.train_nodes.numel()) == 3
    assert int(updated.val_nodes.numel()) == 2
    assert int(updated.test_nodes.numel()) == 5
    assert updated.metadata['split_mode'] == 'sampler'
    assert pytest.approx(updated.metadata['split_train_fraction'], rel=0, abs=1e-8) == 0.3


def test_train_config_rejects_sampler_split_without_real_sampler(tmp_path: Path) -> None:
    config_path = tmp_path / 'bad_sampler_split.yaml'
    config_path.write_text(
        '\n'.join(
            [
                'run_name: sampler_split_bad',
                'dataset:',
                '  name: cora',
                '  split:',
                '    mode: sampler',
                'sampling:',
                '  name: none',
                '  num_hops: 2',
                'model:',
                '  num_layers: 2',
            ]
        ),
        encoding='utf-8',
    )
    with pytest.raises(ConfigurationError, match='requires a real sampler'):
        load_config(TrainConfig, config_path)
