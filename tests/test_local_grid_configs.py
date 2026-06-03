from __future__ import annotations
from gnn_graph_sampling.config.loader import load_config
from gnn_graph_sampling.config.schema import (
    BenchmarkConfig,
    DiagnosticsConfig,
    QSweepConfig,
    SearchConfig,
    SuiteConfig,
    TrainConfig,
)


def test_local_and_grid_configs_load() -> None:
    assert isinstance(load_config(BenchmarkConfig, 'configs/benchmark/local_smoke.yaml'), BenchmarkConfig)
    assert isinstance(load_config(BenchmarkConfig, 'configs/benchmark/local_all_datasets.yaml'), BenchmarkConfig)
    assert isinstance(load_config(BenchmarkConfig, 'configs/benchmark/grid_smoke.yaml'), BenchmarkConfig)
    assert isinstance(load_config(BenchmarkConfig, 'configs/benchmark/grid_all_datasets.yaml'), BenchmarkConfig)
    assert isinstance(load_config(DiagnosticsConfig, 'configs/diagnostics/local_smoke.yaml'), DiagnosticsConfig)
    assert isinstance(load_config(DiagnosticsConfig, 'configs/diagnostics/grid_smoke.yaml'), DiagnosticsConfig)
    assert isinstance(load_config(SuiteConfig, 'configs/suite/local_smoke.yaml'), SuiteConfig)
    assert isinstance(load_config(SuiteConfig, 'configs/suite/grid_all_datasets.yaml'), SuiteConfig)
    assert isinstance(load_config(TrainConfig, 'configs/train/local_flickr_wilson_smoke.yaml'), TrainConfig)
    assert isinstance(load_config(TrainConfig, 'configs/train/grid_flickr_wilson_smoke.yaml'), TrainConfig)
    assert isinstance(load_config(QSweepConfig, 'configs/q_sweep/local_smoke.yaml'), QSweepConfig)
    assert isinstance(load_config(QSweepConfig, 'configs/q_sweep/grid_smoke.yaml'), QSweepConfig)
    assert isinstance(load_config(QSweepConfig, 'configs/q_sweep/local_flickr_features_on.yaml'), QSweepConfig)
    assert isinstance(load_config(SearchConfig, 'configs/search/local_smoke.yaml'), SearchConfig)
    assert isinstance(load_config(SearchConfig, 'configs/search/grid_smoke.yaml'), SearchConfig)
    assert isinstance(load_config(SearchConfig, 'configs/search/local_flickr_features_on.yaml'), SearchConfig)
    assert isinstance(load_config(BenchmarkConfig, 'configs/validation_schemes/local_accuracy.yaml'), BenchmarkConfig)
    assert isinstance(load_config(BenchmarkConfig, 'configs/validation_schemes/grid_weighted_f1.yaml'), BenchmarkConfig)
