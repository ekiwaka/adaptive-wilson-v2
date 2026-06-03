from __future__ import annotations
from pathlib import Path
import pytest

pytest.importorskip('torch_geometric')

from gnn_graph_sampling.config.loader import load_config
from gnn_graph_sampling.config.schema import DiagnosticsConfig
from gnn_graph_sampling.experiments.diagnostics import run_diagnostics_study



def test_diagnostics_study_smoke_writes_expected_artifacts(tmp_path: Path) -> None:
    config_path = tmp_path / 'diagnostics.yaml'
    config_path.write_text(
        '\n'.join(
            [
                'study_name: diagnostics_smoke_test',
                'runtime:',
                '  device: cpu',
                '  device_policy: warn_and_fallback',
                '  seed: 0',
                '  torch_num_threads: 1',
                '  deterministic: true',
                'output:',
                f'  root: {tmp_path.as_posix()}',
                '  experiment_name: diagnostics_test',
                '  save_checkpoints: false',
                '  overwrite: true',
                'logging:',
                '  level: INFO',
                '  log_to_stdout: false',
                'data:',
                '  root: data',
                '  split_idx: 0',
                '  standardize_features: true',
                '  datasets: [sbm_balanced]',
                '  feature_modes: [enabled]',
                '  synthetic:',
                '    sizes: [20, 20, 20]',
                '    num_features: 8',
                '    feature_noise: 0.2',
                '    train_ratio: 0.6',
                '    val_ratio: 0.2',
                'model:',
                '  hidden_dim: 16',
                '  num_layers: 3',
                '  dropout: 0.2',
                '  aggr: mean',
                '  use_batch_norm: true',
                '  residual: true',
                '  jk_mode: cat',
                '  head_hidden_dim: 16',
                'optimization:',
                '  epochs: 1',
                '  lr: 0.003',
                '  weight_decay: 0.0001',
                '  use_amp: false',
                '  grad_clip_norm: 1.0',
                '  accumulation_steps: 1',
                '  use_class_weights: false',
                '  class_weight_mode: effective_num',
                '  class_weight_beta: 0.999',
                '  label_smoothing: 0.0',
                '  early_stopping_patience: 1',
                '  min_epochs: 1',
                '  scheduler_factor: 0.5',
                '  scheduler_patience: 1',
                '  scheduler_min_lr: 1.0e-05',
                '  monitor_metric: macro_f1',
                'sampling:',
                '  samplers: [uniform, degree, wilson]',
                '  batch_sizes: [8, 16]',
                '  num_hops: [3]',
                '  steps_per_epoch: 1',
                '  eval_batch_size: 32',
                '  sample_inference: false',
                '  inference_max_batches: 16',
                '  wilson:',
                '    backend: python',
                '    q: 1.0',
                '    max_sampling_attempts: 4',
                '    allow_uniform_fill: true',
                '    make_undirected: false',
                '    remove_self_loops: false',
                'diagnostics:',
                '  gradient_variance_batches: 2',
            ]
        ),
        encoding='utf-8',
    )
    config = load_config(DiagnosticsConfig, config_path)
    root = run_diagnostics_study(config)

    dataset_dir = root / 'sbm_balanced' / 'enabled'
    assert (dataset_dir / 'diagnostics.json').exists()
    assert (dataset_dir / 'degree_distribution.png').exists()
    assert (dataset_dir / 'class_distribution.png').exists()
    assert (dataset_dir / 'gradient_variance.csv').exists()
    assert (dataset_dir / 'gradient_variance_vs_batch_size.png').exists()
    assert (root / 'tables' / 'dataset_diagnostics_summary.csv').exists()
