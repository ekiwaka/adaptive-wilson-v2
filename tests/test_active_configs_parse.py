from __future__ import annotations

from pathlib import Path

from gnn_graph_sampling.config.loader import load_config
from gnn_graph_sampling.config.schema import (
    BenchmarkConfig,
    DatasetMetricsConfig,
    GraphVisualizationConfig,
    WilsonChecksConfig,
)


def test_active_yaml_configs_parse_successfully() -> None:
    count = 0
    for path in Path('configs/active').rglob('*.yaml'):
        text_path = str(path)
        if '/checks/wilson/' in text_path:
            cls = WilsonChecksConfig
        elif '/checks/dataset_metrics/' in text_path:
            cls = DatasetMetricsConfig
        elif '/checks/graph_visualization/' in text_path:
            cls = GraphVisualizationConfig
        else:
            cls = BenchmarkConfig
        load_config(cls, path)
        count += 1
    assert count >= 1
