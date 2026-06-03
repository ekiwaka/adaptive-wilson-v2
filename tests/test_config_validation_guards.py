from __future__ import annotations
from pathlib import Path
import pytest

from gnn_graph_sampling.config.loader import load_config
from gnn_graph_sampling.config.schema import SearchConfig, TrainConfig
from gnn_graph_sampling.exceptions import ConfigurationError
from gnn_graph_sampling.experiments.common import validate_unique_run_roots


def test_train_config_rejects_hop_layer_mismatch(tmp_path: Path) -> None:
    config_path = tmp_path / 'bad_train.yaml'
    config_path.write_text(
        '\n'.join(
            [
                'run_name: mismatch',
                'dataset:',
                '  name: cora',
                'sampling:',
                '  name: uniform',
                '  num_hops: 1',
                'model:',
                '  num_layers: 2',
            ]
        ),
        encoding='utf-8',
    )
    with pytest.raises(ConfigurationError, match='sampling.num_hops'):
        load_config(TrainConfig, config_path)


def test_search_config_requires_matching_hop_and_layer_grid(tmp_path: Path) -> None:
    config_path = tmp_path / 'bad_search.yaml'
    config_path.write_text(
        '\n'.join(
            [
                'study_name: mismatch_search',
                'data:',
                '  datasets: [cora]',
                '  feature_modes: [enabled]',
                'model_search:',
                '  hidden_dims: [64]',
                '  num_layers: [2]',
                '  dropouts: [0.5]',
                '  aggrs: [mean]',
                '  jk_modes: [last]',
                '  head_hidden_dims: [128]',
                'sampling:',
                '  samplers: [uniform]',
                '  batch_sizes: [64]',
                '  num_hops: [3]',
            ]
        ),
        encoding='utf-8',
    )
    with pytest.raises(ConfigurationError, match='sampling.num_hops must exactly match model_search.num_layers'):
        load_config(SearchConfig, config_path)


def test_duplicate_run_roots_raise_configuration_error() -> None:
    jobs = [
        {'run_root': 'outputs/example/run_a'},
        {'run_root': 'outputs/example/run_a'},
    ]
    with pytest.raises(ConfigurationError, match='Run-root schema collision'):
        validate_unique_run_roots(jobs)
