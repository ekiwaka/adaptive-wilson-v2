from __future__ import annotations
from pathlib import Path
from gnn_graph_sampling.config.loader import load_config
from gnn_graph_sampling.config.schema import TrainConfig


def test_load_train_config_with_nested_sections(tmp_path: Path) -> None:
    config_path = tmp_path / 'config.yaml'
    config_path.write_text(
        "\n".join(
            [
                'run_name: demo',
                'runtime:',
                '  device: cpu',
                'dataset:',
                '  name: flickr',
                '  feature_mode: enabled',
                'sampling:',
                '  name: wilson',
                '  num_hops: 3',
                '  wilson:',
                '    backend: python',
            ]
        ),
        encoding='utf-8',
    )
    config = load_config(TrainConfig, config_path)
    assert config.run_name == 'demo'
    assert config.dataset.name == 'flickr'
    assert config.sampling.name == 'wilson'
    assert config.sampling.wilson.backend == 'python'
