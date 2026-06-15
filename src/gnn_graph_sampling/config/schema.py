from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, Optional

from ..exceptions import ConfigurationError

FeatureMode = Literal['enabled', 'disabled']
SamplerName = Literal['uniform', 'degree', 'wilson', 'adaptive_wilson', 'none']
WilsonBackend = Literal['auto', 'native', 'python']
WilsonAutoQMethod = Literal['mean_degree', 'empirical_binary_search', 'precomputed_table']
QTableMissingPolicy = Literal['error', 'empirical_binary_search', 'mean_degree']
WilsonWeightEstimation = Literal['exact', 'monte_carlo_forests']
WilsonBudgetTargetingMode = Literal['legacy', 'q_target_search', 'expected_budget']
BudgetTargetFailurePolicy = Literal['warn_and_continue', 'error']
MonitorMetric = Literal['accuracy', 'macro_f1', 'balanced_accuracy', 'weighted_f1']
ModelFamily = Literal['graphsage', 'gcn', 'mixhop']
SamplingMode = Literal['online', 'static']
OnlineSchedule = Literal['per_batch', 'per_epoch']
BudgetUnit = Literal['count', 'ratio']
ReweightingTarget = Literal['loss', 'gradient']
AdaptiveWilsonBudgetGridMode = Literal['legacy_alpha_cap', 'fixed_budget']
AdaptiveWilsonQMode = Literal['manual_alpha', 'precomputed_budget_q']
SplitMode = Literal['predefined', 'public', 'custom', 'generated', 'sampler']


def _validate_exact_neighborhood_depth(*, num_hops: int, num_layers: int, context: str) -> None:
    if num_hops < num_layers:
        raise ConfigurationError(
            f'{context}: sampling.num_hops ({num_hops}) must be >= model.num_layers ({num_layers}).'
        )
    if num_hops != num_layers:
        raise ConfigurationError(
            f'{context}: sampling.num_hops ({num_hops}) must equal model.num_layers ({num_layers}) '
            'because training and exact evaluation now use the same receptive-field depth.'
        )


@dataclass(slots=True)
class LoggingConfig:
    """Logging configuration for one experiment process."""

    level: str = 'INFO'
    log_to_stdout: bool = True


@dataclass(slots=True)
class ExperimentConfig:
    """High-level experiment switches.

    sampling_regime is an alias for sampling.mode in TrainConfig and for
    sampling.modes in BenchmarkConfig. It exists so YAML can switch between
    static and online without touching code.
    """

    sampling_regime: Optional[SamplingMode] = None

    def validate(self) -> None:
        if self.sampling_regime is not None and self.sampling_regime not in {'static', 'online'}:
            raise ConfigurationError('experiment.sampling_regime must be static or online when provided.')


@dataclass(slots=True)
class EvaluationConfig:
    """Evaluation/inference execution policy."""

    mode: Literal['exact_batched', 'full_graph'] = 'exact_batched'
    device: Literal['auto', 'cpu', 'cuda'] = 'auto'
    batch_size: int = 1024
    non_blocking: bool = True
    compute_train_metrics: bool = True
    track_test_metrics_during_training: bool = False

    def validate(self) -> None:
        if self.batch_size <= 0:
            raise ConfigurationError('evaluation.batch_size must be positive.')


@dataclass(slots=True)
class OutputConfig:
    """Artifact storage configuration."""

    root: str = 'outputs'
    experiment_name: str = 'default'
    save_checkpoints: bool = True
    overwrite: bool = False
    append_timestamp: bool = False
    timestamp_format: str = '%Y-%m-%d-%H-%M'


@dataclass(slots=True)
class RuntimeConfig:
    """Runtime configuration shared by all experiments."""

    device: str = 'cpu'
    device_policy: Literal['warn_and_fallback', 'error'] = 'warn_and_fallback'
    seed: int = 0
    torch_num_threads: int = 1
    deterministic: bool = True

    def validate(self) -> None:
        if self.torch_num_threads <= 0:
            raise ConfigurationError('runtime.torch_num_threads must be positive.')


@dataclass(slots=True)
class SyntheticDatasetConfig:
    """Configuration for synthetic SBM datasets."""

    sizes: list[int] = field(default_factory=lambda: [400, 400, 400])
    p_in: float = 0.10
    p_out: float = 0.01
    num_features: int = 32
    feature_noise: float = 0.4
    train_ratio: float = 0.6
    val_ratio: float = 0.2

    def validate(self) -> None:
        if any(size <= 0 for size in self.sizes):
            raise ConfigurationError('synthetic.sizes must all be positive.')
        if not (0.0 <= self.p_in <= 1.0 and 0.0 <= self.p_out <= 1.0):
            raise ConfigurationError('synthetic probabilities must lie in [0, 1].')
        if self.num_features <= 0:
            raise ConfigurationError('synthetic.num_features must be positive.')
        if not (0.0 < self.train_ratio < 1.0 and 0.0 < self.val_ratio < 1.0):
            raise ConfigurationError('synthetic split ratios must lie in (0, 1).')
        if self.train_ratio + self.val_ratio >= 1.0:
            raise ConfigurationError('synthetic train_ratio + val_ratio must be < 1.')


@dataclass(slots=True)
class SplitConfig:
    """Controls how train/val/test node splits are obtained.

    Modes:
      - predefined/public: use masks shipped by PyG or synthetic generation.
      - sampler: sample train nodes first, then split the remainder.
      - generated: create a deterministic stratified split from labels.
      - custom: load a previously saved split from split_file.
    """

    mode: SplitMode = 'predefined'
    val_ratio_of_remaining: float = 0.1
    name: Optional[str] = None
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    test_ratio: float = 0.2
    seed: int = 0
    split_file: Optional[str] = None
    save_generated_split: bool = False
    reuse_existing_split: bool = True

    def validate(self) -> None:
        if self.mode not in {'predefined', 'public', 'custom', 'generated', 'sampler'}:
            raise ConfigurationError('split.mode must be predefined, public, custom, generated, or sampler.')
        if not (0.0 < self.val_ratio_of_remaining < 1.0):
            raise ConfigurationError('split.val_ratio_of_remaining must lie in (0, 1).')
        if self.mode in {'generated'}:
            ratios = [float(self.train_ratio), float(self.val_ratio), float(self.test_ratio)]
            if any(value < 0.0 or value > 1.0 for value in ratios):
                raise ConfigurationError('generated split ratios must lie in [0, 1].')
            if abs(sum(ratios) - 1.0) > 1e-6:
                raise ConfigurationError('generated split train_ratio + val_ratio + test_ratio must sum to 1.')
            if self.train_ratio <= 0.0 or self.val_ratio <= 0.0 or self.test_ratio <= 0.0:
                raise ConfigurationError('generated split train_ratio, val_ratio, and test_ratio must all be positive.')
        if self.mode == 'custom' and not self.split_file:
            raise ConfigurationError('custom split mode requires split.split_file.')


@dataclass(slots=True)
class DatasetConfig:
    """Dataset configuration for a single training run."""

    name: str = 'flickr'
    root: str = 'data'
    split_idx: int = 0
    planetoid_split: Literal['public', 'full', 'geom-gcn', 'random'] = 'public'
    feature_mode: FeatureMode = 'enabled'
    standardize_features: bool = False
    split: SplitConfig = field(default_factory=SplitConfig)
    synthetic: SyntheticDatasetConfig = field(default_factory=SyntheticDatasetConfig)

    def validate(self) -> None:
        if self.feature_mode not in {'enabled', 'disabled'}:
            raise ConfigurationError('dataset.feature_mode must be enabled or disabled.')
        if self.split_idx < 0:
            raise ConfigurationError('dataset.split_idx must be non-negative.')
        if self.planetoid_split not in {'public', 'full', 'geom-gcn', 'random'}:
            raise ConfigurationError('dataset.planetoid_split must be one of public, full, geom-gcn, random.')
        self.split.validate()
        self.synthetic.validate()


@dataclass(slots=True)
class BenchmarkDataConfig:
    """Dataset grid configuration for benchmark and diagnostics studies."""

    root: str = 'data'
    split_idx: int = 0
    split_indices: list[int] = field(default_factory=list)
    planetoid_split: Literal['public', 'full', 'geom-gcn', 'random'] = 'public'
    standardize_features: bool = False
    split: SplitConfig = field(default_factory=SplitConfig)
    datasets: list[str] = field(default_factory=lambda: ['flickr'])
    feature_modes: list[FeatureMode] = field(default_factory=lambda: ['enabled'])
    synthetic: SyntheticDatasetConfig = field(default_factory=SyntheticDatasetConfig)

    def validate(self) -> None:
        if self.split_idx < 0:
            raise ConfigurationError('data.split_idx must be non-negative.')
        if any(index < 0 for index in self.split_indices):
            raise ConfigurationError('data.split_indices must contain only non-negative integers.')
        if self.planetoid_split not in {'public', 'full', 'geom-gcn', 'random'}:
            raise ConfigurationError('data.planetoid_split must be one of public, full, geom-gcn, random.')
        if not self.datasets:
            raise ConfigurationError('data.datasets must be non-empty.')
        self.split.validate()
        if not self.feature_modes:
            raise ConfigurationError('data.feature_modes must be non-empty.')
        for mode in self.feature_modes:
            if mode not in {'enabled', 'disabled'}:
                raise ConfigurationError('data.feature_modes must only contain enabled or disabled.')
        self.synthetic.validate()


@dataclass(slots=True)
class ModelConfig:
    """Message-passing backbone configuration."""

    family: ModelFamily = 'graphsage'
    hidden_dim: int = 128
    num_layers: int = 3
    dropout: float = 0.5
    input_dropout: float = 0.0
    aggr: Literal['mean', 'sum', 'max'] = 'mean'
    use_batch_norm: bool = True
    norm_type: Optional[Literal['batch', 'layer', 'none']] = None
    residual: bool = True
    jk_mode: Literal['last', 'cat'] = 'last'
    head_type: Literal['mlp', 'linear'] = 'mlp'
    head_hidden_dim: Optional[int] = None

    def validate(self) -> None:
        if self.family not in {'graphsage', 'gcn', 'mixhop'}:
            raise ConfigurationError("model.family must be one of graphsage, gcn, or mixhop.")
        if self.hidden_dim <= 0:
            raise ConfigurationError('model.hidden_dim must be positive.')
        if self.num_layers < 2:
            raise ConfigurationError('model.num_layers must be at least 2.')
        if not (0.0 <= self.dropout < 1.0):
            raise ConfigurationError('model.dropout must lie in [0, 1).')
        if not (0.0 <= self.input_dropout < 1.0):
            raise ConfigurationError('model.input_dropout must lie in [0, 1).')
        if self.norm_type is not None and self.norm_type not in {'batch', 'layer', 'none'}:
            raise ConfigurationError("model.norm_type must be one of batch, layer, or none when provided.")
        if self.head_type not in {'mlp', 'linear'}:
            raise ConfigurationError("model.head_type must be one of mlp or linear.")


@dataclass(slots=True)
class OptimizationConfig:
    """Optimization and early-stopping configuration."""

    epochs: int = 100
    lr: float = 3e-3
    weight_decay: float = 1e-4
    use_amp: bool = True
    grad_clip_norm: float = 1.0
    accumulation_steps: int = 1
    use_class_weights: bool = False
    auto_disable_class_weights_for_tiny_balanced_citation_splits: bool = True
    auto_disable_class_weights_for_small_balanced_splits: Optional[bool] = None
    class_weight_auto_disable_max_train_nodes: int = 200
    class_weight_auto_disable_max_imbalance_ratio: float = 2.0
    class_weight_mode: Literal['sqrt_inv', 'effective_num'] = 'sqrt_inv'
    class_weight_beta: float = 0.999
    label_smoothing: float = 0.0
    early_stopping_patience: int = 20
    min_epochs: int = 20
    scheduler_factor: float = 0.5
    scheduler_patience: int = 5
    scheduler_min_lr: float = 1e-5
    monitor_metric: MonitorMetric = 'macro_f1'
    use_static_weight_normalization: bool = False
    # General sampler-normalized weighting flag. This supersedes the legacy
    # static-only flag above while preserving backward compatibility.
    use_normalized_weights: bool = False
    normalize_weights: bool = True
    weight_normalization: Literal['mean_one'] = 'mean_one'
    static_weight_normalization_eps: float = 1e-12
    strict_one_pass_wilson_weights: bool = True
    # loss: current weighted-loss objective. gradient: gradient-equivalent
    # detached sample weighting, recorded separately for gradient-estimator studies.
    reweighting_target: ReweightingTarget = 'loss'

    def validate(self) -> None:
        if self.epochs <= 0:
            raise ConfigurationError('optimization.epochs must be positive.')
        if self.lr <= 0.0:
            raise ConfigurationError('optimization.lr must be positive.')
        if self.weight_decay < 0.0:
            raise ConfigurationError('optimization.weight_decay must be non-negative.')
        if self.grad_clip_norm <= 0.0:
            raise ConfigurationError('optimization.grad_clip_norm must be positive.')
        if self.accumulation_steps <= 0:
            raise ConfigurationError('optimization.accumulation_steps must be positive.')
        if not (0.0 <= self.label_smoothing < 1.0):
            raise ConfigurationError('optimization.label_smoothing must lie in [0, 1).')
        if self.scheduler_factor <= 0.0 or self.scheduler_factor >= 1.0:
            raise ConfigurationError('optimization.scheduler_factor must lie in (0, 1).')
        if self.scheduler_min_lr < 0.0:
            raise ConfigurationError('optimization.scheduler_min_lr must be non-negative.')
        if self.class_weight_auto_disable_max_train_nodes <= 0:
            raise ConfigurationError('optimization.class_weight_auto_disable_max_train_nodes must be positive.')
        if self.class_weight_auto_disable_max_imbalance_ratio < 1.0:
            raise ConfigurationError('optimization.class_weight_auto_disable_max_imbalance_ratio must be >= 1.0.')
        if self.static_weight_normalization_eps <= 0.0:
            raise ConfigurationError(
                'optimization.static_weight_normalization_eps must be positive.'
            )
        if self.weight_normalization not in {'mean_one'}:
            raise ConfigurationError('optimization.weight_normalization must be mean_one.')
        if self.reweighting_target not in {'loss', 'gradient'}:
            raise ConfigurationError('optimization.reweighting_target must be loss or gradient.')


@dataclass(slots=True)
class WilsonConfig:
    """Wilson backend, q-selection, weighting, and budget-targeting configuration."""

    backend: WilsonBackend = 'auto'
    q: Optional[float] = None
    redraw_on_shortfall: bool = True
    max_sampling_attempts: int = 8
    allow_uniform_fill: bool = True
    make_undirected: bool = False
    remove_self_loops: bool = False

    # q resolution when q is null. mean_degree preserves the historical behavior.
    auto_q_method: WilsonAutoQMethod = 'mean_degree'
    q_calibration_trials: int = 8
    q_calibration_max_iters: int = 16
    q_calibration_tolerance: float = 0.03
    q_min: float = 1.0e-8
    q_max: float = 100.0
    q_calibration_seed: int = 1729

    # Precomputed q-table mode.
    q_table_path: Optional[str] = None
    q_table_missing_policy: QTableMissingPolicy = 'error'

    # Static-weight estimation for Wilson sampler-split experiments.
    weight_estimation: WilsonWeightEstimation = 'exact'
    num_weight_forests: int = 256
    weight_estimation_seed: int = 314159

    # Wilson-only budget targeting without padding/capping.
    budget_targeting: WilsonBudgetTargetingMode = 'legacy'
    budget_targeting_enabled: bool = False
    avoid_padding: bool = False
    avoid_capping: bool = False
    q_estimation: Optional[WilsonAutoQMethod] = None
    target_budget_tolerance: float = 0.03
    max_budget_search_steps: int = 20
    budget_target_failure_policy: BudgetTargetFailurePolicy = 'warn_and_continue'

    def validate(self) -> None:
        if self.backend not in {'auto', 'native', 'python'}:
            raise ConfigurationError('sampling.wilson.backend must be auto, native, or python.')
        if self.q is not None and self.q <= 0.0:
            raise ConfigurationError('sampling.wilson.q must be positive when provided.')
        if self.max_sampling_attempts <= 0:
            raise ConfigurationError('sampling.wilson.max_sampling_attempts must be positive.')
        if self.auto_q_method not in {'mean_degree', 'empirical_binary_search', 'precomputed_table'}:
            raise ConfigurationError(
                'sampling.wilson.auto_q_method must be one of mean_degree, '
                'empirical_binary_search, or precomputed_table.'
            )
        if self.q_calibration_trials <= 0:
            raise ConfigurationError('sampling.wilson.q_calibration_trials must be positive.')
        if self.q_calibration_max_iters <= 0:
            raise ConfigurationError('sampling.wilson.q_calibration_max_iters must be positive.')
        if not (0.0 < self.q_calibration_tolerance < 1.0):
            raise ConfigurationError('sampling.wilson.q_calibration_tolerance must lie in (0, 1).')
        if self.q_min <= 0.0:
            raise ConfigurationError('sampling.wilson.q_min must be positive.')
        if self.q_max <= self.q_min:
            raise ConfigurationError('sampling.wilson.q_max must be greater than q_min.')
        if self.q_table_missing_policy not in {'error', 'empirical_binary_search', 'mean_degree'}:
            raise ConfigurationError(
                'sampling.wilson.q_table_missing_policy must be error, empirical_binary_search, or mean_degree.'
            )
        if self.auto_q_method == 'precomputed_table' and not self.q_table_path:
            raise ConfigurationError(
                'sampling.wilson.q_table_path must be set when auto_q_method=precomputed_table.'
            )
        if self.q_estimation is not None:
            self.auto_q_method = self.q_estimation
        if self.budget_targeting_enabled:
            self.budget_targeting = 'q_target_search'
        if self.avoid_padding:
            self.allow_uniform_fill = False
        if self.avoid_capping and self.budget_targeting == 'legacy':
            # legacy Wilson caps overshoots; avoid_capping therefore requires q_target_search.
            if self.budget_targeting_enabled:
                self.budget_targeting = 'q_target_search'
            else:
                raise ConfigurationError('sampling.wilson.avoid_capping=true requires budget_targeting_enabled=true.')
        if self.weight_estimation not in {'exact', 'monte_carlo_forests'}:
            raise ConfigurationError('sampling.wilson.weight_estimation must be exact or monte_carlo_forests.')
        if self.num_weight_forests <= 0:
            raise ConfigurationError('sampling.wilson.num_weight_forests must be positive.')
        if self.budget_targeting not in {'legacy', 'q_target_search', 'expected_budget'}:
            raise ConfigurationError(
                'sampling.wilson.budget_targeting must be legacy, q_target_search, or expected_budget.'
            )
        if not (0.0 < self.target_budget_tolerance < 1.0):
            raise ConfigurationError('sampling.wilson.target_budget_tolerance must lie in (0, 1).')
        if self.max_budget_search_steps <= 0:
            raise ConfigurationError('sampling.wilson.max_budget_search_steps must be positive.')
        if self.budget_target_failure_policy not in {'warn_and_continue', 'error'}:
            raise ConfigurationError(
                'sampling.wilson.budget_target_failure_policy must be warn_and_continue or error.'
            )


@dataclass(slots=True)
class AdaptiveWilsonConfig:
    """Online Adaptive Wilson settings.

    The current implementation exposes the supervisor-recommended baseline
    q_i = alpha, which is represented through the scalar-q Wilson backend.
    """

    q_i_mode: Literal['alpha'] = 'alpha'
    alpha: float = 1.0
    alpha_grid: list[float] = field(default_factory=list)
    # manual_alpha preserves the old q_i=alpha behaviour. precomputed_budget_q
    # makes Adaptive Wilson ignore alpha_grid/manual alpha and resolve q from
    # sampling.wilson.q_table_path for each requested mini-batch budget.
    q_mode: AdaptiveWilsonQMode = 'manual_alpha'
    # legacy_alpha_cap preserves the old behaviour: alpha sweeps use one loose
    # cap/budget and alpha determines the realised root count. fixed_budget runs
    # every alpha at every requested budget, so adaptive_wilson is comparable
    # with uniform/degree at matched center-node budgets.
    budget_grid_mode: AdaptiveWilsonBudgetGridMode = 'legacy_alpha_cap'

    def validate(self) -> None:
        if self.q_i_mode not in {'alpha'}:
            raise ConfigurationError('sampling.adaptive_wilson.q_i_mode currently supports only alpha.')
        if self.alpha <= 0.0:
            raise ConfigurationError('sampling.adaptive_wilson.alpha must be positive.')
        if any(value <= 0.0 for value in self.alpha_grid):
            raise ConfigurationError('sampling.adaptive_wilson.alpha_grid must contain positive values.')
        if self.q_mode not in {'manual_alpha', 'precomputed_budget_q'}:
            raise ConfigurationError(
                'sampling.adaptive_wilson.q_mode must be manual_alpha or precomputed_budget_q.'
            )
        if self.budget_grid_mode not in {'legacy_alpha_cap', 'fixed_budget'}:
            raise ConfigurationError(
                'sampling.adaptive_wilson.budget_grid_mode must be legacy_alpha_cap or fixed_budget.'
            )



@dataclass(slots=True)
class SamplingConfig:
    """Sampling configuration for a single training run.

    mode=online with online_schedule=per_batch reproduces the original
    behaviour of sampling fresh center nodes for every optimization step.

    mode=online with online_schedule=per_epoch samples one reduced
    training graph at the beginning of each epoch and performs one full-batch
    update on that epoch-local reduced graph.

    mode=static samples one reduced training graph before optimization,
    optionally caches it, and reuses it for all epochs.
    """

    name: SamplerName = 'uniform'
    mode: SamplingMode = 'online'
    online_schedule: OnlineSchedule = 'per_batch'
    budget_unit: BudgetUnit = 'count'
    budget_value: Optional[float] = None
    batch_size: int = 128
    steps_per_epoch: int = 24
    eval_batch_size: int = 256
    num_hops: int = 1
    sample_inference: bool = False
    inference_max_batches: int = 128
    cache_preprocessed_graph: bool = True
    reuse_cached_preprocessed_graph: bool = True
    cache_dirname: str = 'preprocessed_graphs'
    wilson: WilsonConfig = field(default_factory=WilsonConfig)
    adaptive_wilson: AdaptiveWilsonConfig = field(default_factory=AdaptiveWilsonConfig)

    def validate(self) -> None:
        if self.name not in {'uniform', 'degree', 'wilson', 'adaptive_wilson', 'none'}:
            raise ConfigurationError('sampling.name must be uniform, degree, wilson, adaptive_wilson, or none.')
        if self.mode not in {'online', 'static'}:
            raise ConfigurationError('sampling.mode must be online or static.')
        if self.online_schedule not in {'per_batch', 'per_epoch'}:
            raise ConfigurationError('sampling.online_schedule must be per_batch or per_epoch.')
        if self.budget_unit not in {'count', 'ratio'}:
            raise ConfigurationError('sampling.budget_unit must be count or ratio.')
        if self.batch_size <= 0 or self.eval_batch_size <= 0:
            raise ConfigurationError('sampling batch sizes must be positive.')
        if self.steps_per_epoch <= 0:
            raise ConfigurationError('sampling.steps_per_epoch must be positive.')
        if self.num_hops < 0:
            raise ConfigurationError('sampling.num_hops must be non-negative.')
        if self.inference_max_batches <= 0:
            raise ConfigurationError('sampling.inference_max_batches must be positive.')
        if self.budget_value is not None:
            if self.budget_unit == 'count' and self.budget_value <= 0:
                raise ConfigurationError('sampling.budget_value must be positive when budget_unit=count.')
            if self.budget_unit == 'ratio' and not (0.0 < self.budget_value <= 1.0):
                raise ConfigurationError('sampling.budget_value must lie in (0, 1] when budget_unit=ratio.')
        self.wilson.validate()
        self.adaptive_wilson.validate()



@dataclass(slots=True)
class BenchmarkSamplingConfig:
    """Sampling grid configuration for benchmark and diagnostics studies."""

    samplers: list[SamplerName] = field(default_factory=lambda: ['uniform', 'degree', 'wilson'])
    modes: list[SamplingMode] = field(default_factory=lambda: ['online'])
    online_schedules: list[OnlineSchedule] = field(default_factory=lambda: ['per_batch'])
    budget_unit: BudgetUnit = 'count'
    budget_values: list[float] = field(default_factory=list)
    batch_sizes: list[int] = field(default_factory=lambda: [128])
    num_hops: list[int] = field(default_factory=lambda: [1])
    steps_per_epoch: int = 24
    eval_batch_size: int = 256
    sample_inference: bool = False
    inference_max_batches: int = 128
    cache_preprocessed_graph: bool = True
    reuse_cached_preprocessed_graph: bool = True
    cache_dirname: str = 'preprocessed_graphs'
    wilson: WilsonConfig = field(default_factory=WilsonConfig)
    adaptive_wilson: AdaptiveWilsonConfig = field(default_factory=AdaptiveWilsonConfig)

    def validate(self) -> None:
        if not self.samplers:
            raise ConfigurationError('sampling.samplers must be non-empty.')
        for sampler in self.samplers:
            if sampler not in {'uniform', 'degree', 'wilson', 'adaptive_wilson', 'none'}:
                raise ConfigurationError(f'Unsupported sampler in sampling.samplers: {sampler}')
        if not self.modes:
            raise ConfigurationError('sampling.modes must be non-empty.')
        for mode in self.modes:
            if mode not in {'online', 'static'}:
                raise ConfigurationError(f'Unsupported sampling mode: {mode}')
        if not self.online_schedules:
            raise ConfigurationError('sampling.online_schedules must be non-empty.')
        for schedule in self.online_schedules:
            if schedule not in {'per_batch', 'per_epoch'}:
                raise ConfigurationError(f'Unsupported online schedule: {schedule}')
        if self.budget_unit not in {'count', 'ratio'}:
            raise ConfigurationError('sampling.budget_unit must be count or ratio.')
        if not self.batch_sizes or any(size <= 0 for size in self.batch_sizes):
            raise ConfigurationError('sampling.batch_sizes must contain positive integers.')
        if self.budget_values:
            if self.budget_unit == 'count' and any(value <= 0 for value in self.budget_values):
                raise ConfigurationError('sampling.budget_values must be positive when budget_unit=count.')
            if self.budget_unit == 'ratio' and any(value <= 0.0 or value > 1.0 for value in self.budget_values):
                raise ConfigurationError('sampling.budget_values must lie in (0, 1] when budget_unit=ratio.')
        if not self.num_hops or any(hops < 0 for hops in self.num_hops):
            raise ConfigurationError('sampling.num_hops must contain non-negative integers.')
        if self.steps_per_epoch <= 0 or self.eval_batch_size <= 0:
            raise ConfigurationError('sampling.steps_per_epoch and eval_batch_size must be positive.')
        self.wilson.validate()
        self.adaptive_wilson.validate()




@dataclass(slots=True)
class ModelSearchConfig:
    """Backbone hyperparameter grid for search studies."""

    families: list[ModelFamily] = field(default_factory=lambda: ['graphsage'])
    hidden_dims: list[int] = field(default_factory=lambda: [64])
    num_layers: list[int] = field(default_factory=lambda: [2])
    dropouts: list[float] = field(default_factory=lambda: [0.5])
    aggrs: list[Literal['mean', 'sum', 'max']] = field(default_factory=lambda: ['mean'])
    jk_modes: list[Literal['last', 'cat']] = field(default_factory=lambda: ['last'])
    head_hidden_dims: list[int] = field(default_factory=lambda: [128])
    use_batch_norm: bool = True
    residual: bool = True

    def validate(self) -> None:
        if not self.families:
            raise ConfigurationError('model_search.families must be non-empty.')
        for family in self.families:
            if family not in {'graphsage', 'gcn', 'mixhop'}:
                raise ConfigurationError(f'Unsupported model family in model_search.families: {family}')
        if not self.hidden_dims or any(value <= 0 for value in self.hidden_dims):
            raise ConfigurationError('model_search.hidden_dims must contain positive integers.')
        if not self.num_layers or any(value < 2 for value in self.num_layers):
            raise ConfigurationError('model_search.num_layers must contain integers >= 2.')
        if not self.dropouts or any(value < 0.0 or value >= 1.0 for value in self.dropouts):
            raise ConfigurationError('model_search.dropouts must lie in [0, 1).')
        if not self.aggrs:
            raise ConfigurationError('model_search.aggrs must be non-empty.')
        if not self.jk_modes:
            raise ConfigurationError('model_search.jk_modes must be non-empty.')
        if not self.head_hidden_dims or any(value <= 0 for value in self.head_hidden_dims):
            raise ConfigurationError('model_search.head_hidden_dims must contain positive integers.')


@dataclass(slots=True)
class SearchSamplingConfig(BenchmarkSamplingConfig):
    """Sampling search grid that also sweeps Wilson q values."""

    wilson_q_values: list[float] = field(default_factory=lambda: [0.1, 0.5, 1.0, 2.0, 4.0, 8.0])

    def validate(self) -> None:
        BenchmarkSamplingConfig.validate(self)
        if not self.wilson_q_values:
            raise ConfigurationError('sampling.wilson_q_values must be non-empty.')
        if any(value <= 0.0 for value in self.wilson_q_values):
            raise ConfigurationError('sampling.wilson_q_values must contain positive values.')


@dataclass(slots=True)
class QSweepSettings:
    """Wilson q-sweep definition and ranking policy."""

    q_values: list[float] = field(default_factory=list)
    q_start: float = 0.1
    q_stop: float = 10.0
    q_step: float = 0.1
    runs: int = 3
    base_seed: int = 0
    parallel_workers: int = 1
    selection_metric: str = 'mean_val_macro_f1'

    def validate(self) -> None:
        if self.q_values and any(value <= 0.0 for value in self.q_values):
            raise ConfigurationError('q_sweep.q_values must contain strictly positive values.')
        if self.q_step <= 0.0:
            raise ConfigurationError('q_sweep.q_step must be positive.')
        if self.q_start <= 0.0 or self.q_stop <= 0.0:
            raise ConfigurationError('q_sweep.q_start and q_sweep.q_stop must be strictly positive.')
        if self.q_stop < self.q_start:
            raise ConfigurationError('q_sweep.q_stop must be >= q_sweep.q_start.')
        if self.runs <= 0:
            raise ConfigurationError('q_sweep.runs must be positive.')
        if self.parallel_workers <= 0:
            raise ConfigurationError('q_sweep.parallel_workers must be positive.')
        if not self.selection_metric:
            raise ConfigurationError('q_sweep.selection_metric must be non-empty.')


@dataclass(slots=True)
class SearchSettings:
    """Search repetition and ranking settings."""

    runs: int = 3
    base_seed: int = 0
    parallel_workers: int = 1
    selection_metric: str = 'mean_val_macro_f1'

    def validate(self) -> None:
        if self.runs <= 0:
            raise ConfigurationError('search.runs must be positive.')
        if self.parallel_workers <= 0:
            raise ConfigurationError('search.parallel_workers must be positive.')
        if not self.selection_metric:
            raise ConfigurationError('search.selection_metric must be non-empty.')


@dataclass(slots=True)
class DatasetMetricsSettings:
    """Dataset-level metrics and plot generation settings."""

    max_exact_path_nodes: int = 2000
    spectral_num_communities: Optional[int] = None
    save_plots: bool = True

    def validate(self) -> None:
        if self.max_exact_path_nodes <= 0:
            raise ConfigurationError('metrics.max_exact_path_nodes must be positive.')
        if self.spectral_num_communities is not None and self.spectral_num_communities <= 0:
            raise ConfigurationError('metrics.spectral_num_communities must be positive when provided.')


@dataclass(slots=True)
class DatasetMetricsConfig:
    """YAML-driven dataset download/load and metrics reporting workflow."""

    study_name: str = 'dataset_metrics'
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    data: BenchmarkDataConfig = field(default_factory=BenchmarkDataConfig)
    metrics: DatasetMetricsSettings = field(default_factory=DatasetMetricsSettings)

    def validate(self) -> 'DatasetMetricsConfig':
        self.runtime.validate()
        self.data.validate()
        self.metrics.validate()
        return self


@dataclass(slots=True)
class GraphVisualizationSettings:
    """Graph visualization settings for one or more datasets."""

    max_nodes: int = 500
    subgraph_method: Literal['full_graph', 'largest_component', 'random'] = 'random'
    layout: Literal['spring', 'kamada_kawai', 'spectral', 'shell', 'circular', 'random'] = 'spring'
    color_by: Literal['label', 'split', 'degree', 'constant'] = 'label'
    node_size_by: Literal['degree', 'constant'] = 'constant'
    node_size: float = 24.0
    edge_alpha: float = 0.15
    node_alpha: float = 0.9
    label_nodes: bool = False
    label_with: Literal['label', 'node_id'] = 'label'
    output_format: Literal['png', 'pdf', 'svg'] = 'png'
    dpi: int = 180

    def validate(self) -> None:
        if self.max_nodes <= 0:
            raise ConfigurationError('visualization.max_nodes must be positive.')
        if self.subgraph_method not in {'full_graph', 'largest_component', 'random'}:
            raise ConfigurationError(
                'visualization.subgraph_method must be one of: '
                'full_graph, largest_component, random.'
            )
        if self.layout not in {'spring', 'kamada_kawai', 'spectral', 'shell', 'circular', 'random'}:
            raise ConfigurationError('visualization.layout is unsupported.')
        if self.color_by not in {'label', 'split', 'degree', 'constant'}:
            raise ConfigurationError('visualization.color_by is unsupported.')
        if self.node_size_by not in {'degree', 'constant'}:
            raise ConfigurationError('visualization.node_size_by is unsupported.')
        if self.label_with not in {'label', 'node_id'}:
            raise ConfigurationError('visualization.label_with must be label or node_id.')
        if not (0.0 <= self.edge_alpha <= 1.0):
            raise ConfigurationError('visualization.edge_alpha must lie in [0, 1].')
        if not (0.0 <= self.node_alpha <= 1.0):
            raise ConfigurationError('visualization.node_alpha must lie in [0, 1].')
        if self.output_format not in {'png', 'pdf', 'svg'}:
            raise ConfigurationError('visualization.output_format must be png, pdf, or svg.')
        if self.dpi <= 0:
            raise ConfigurationError('visualization.dpi must be positive.')


@dataclass(slots=True)
class GraphVisualizationConfig:
    """YAML-driven graph structure visualization workflow."""

    study_name: str = 'graph_visualization'
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    data: BenchmarkDataConfig = field(default_factory=BenchmarkDataConfig)
    visualization: GraphVisualizationSettings = field(default_factory=GraphVisualizationSettings)

    def validate(self) -> 'GraphVisualizationConfig':
        self.runtime.validate()
        self.data.validate()
        self.visualization.validate()
        return self


@dataclass(slots=True)
class TrainConfig:
    """Configuration for a single training run."""

    run_name: str = 'train'
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def validate(self) -> 'TrainConfig':
        self.experiment.validate()
        if self.experiment.sampling_regime is not None:
            self.sampling.mode = self.experiment.sampling_regime
        self.runtime.validate()
        self.dataset.validate()
        self.model.validate()
        self.optimization.validate()
        self.sampling.validate()
        self.evaluation.validate()
        _validate_exact_neighborhood_depth(
            num_hops=int(self.sampling.num_hops),
            num_layers=int(self.model.num_layers),
            context='train config',
        )
        if self.dataset.split.mode == 'sampler' and self.sampling.name == 'none':
            raise ConfigurationError('train config: dataset.split.mode=sampler requires a real sampler; sampling.name cannot be none.')
        return self


@dataclass(slots=True)
class BenchmarkSettings:
    """Benchmark repetition, scheduling, and artifact configuration."""

    runs: int = 3
    base_seed: int = 0
    parallel_workers: int = 1
    save_plots: bool = True

    def validate(self) -> None:
        if self.runs <= 0:
            raise ConfigurationError('benchmark.runs must be positive.')
        if self.parallel_workers <= 0:
            raise ConfigurationError('benchmark.parallel_workers must be positive.')


@dataclass(slots=True)
class BenchmarkConfig:
    """Configuration for a multi-dataset benchmark study."""

    study_name: str = 'benchmark'
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    data: BenchmarkDataConfig = field(default_factory=BenchmarkDataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    sampling: BenchmarkSamplingConfig = field(default_factory=BenchmarkSamplingConfig)
    benchmark: BenchmarkSettings = field(default_factory=BenchmarkSettings)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def validate(self) -> 'BenchmarkConfig':
        self.experiment.validate()
        if self.experiment.sampling_regime is not None:
            self.sampling.modes = [self.experiment.sampling_regime]
        self.runtime.validate()
        self.data.validate()
        self.model.validate()
        self.optimization.validate()
        self.sampling.validate()
        self.benchmark.validate()
        self.evaluation.validate()
        for num_hops in self.sampling.num_hops:
            _validate_exact_neighborhood_depth(
                num_hops=int(num_hops),
                num_layers=int(self.model.num_layers),
                context='benchmark config',
            )
        if self.data.split.mode == 'sampler' and any(sampler == 'none' for sampler in self.sampling.samplers):
            raise ConfigurationError('benchmark config: data.split.mode=sampler requires real samplers; sampling.samplers cannot include none.')
        return self


@dataclass(slots=True)
class QSweepConfig:
    """Configuration for a Wilson q-sweep study."""

    study_name: str = 'q_sweep'
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    q_sweep: QSweepSettings = field(default_factory=QSweepSettings)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def validate(self) -> 'QSweepConfig':
        self.runtime.validate()
        self.dataset.validate()
        self.model.validate()
        self.optimization.validate()
        self.sampling.validate()
        self.q_sweep.validate()
        self.evaluation.validate()
        _validate_exact_neighborhood_depth(
            num_hops=int(self.sampling.num_hops),
            num_layers=int(self.model.num_layers),
            context='q_sweep config',
        )
        if self.sampling.name != 'wilson':
            raise ConfigurationError('q_sweep experiments require sampling.name to be wilson.')
        return self


@dataclass(slots=True)
class SearchConfig:
    """Configuration for an end-to-end grid search study."""

    study_name: str = 'search'
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    data: BenchmarkDataConfig = field(default_factory=BenchmarkDataConfig)
    model_search: ModelSearchConfig = field(default_factory=ModelSearchConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    sampling: SearchSamplingConfig = field(default_factory=SearchSamplingConfig)
    search: SearchSettings = field(default_factory=SearchSettings)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def validate(self) -> 'SearchConfig':
        self.runtime.validate()
        self.data.validate()
        self.model_search.validate()
        self.optimization.validate()
        self.sampling.validate()
        self.search.validate()
        self.evaluation.validate()
        hop_values = {int(value) for value in self.sampling.num_hops}
        layer_values = {int(value) for value in self.model_search.num_layers}
        if hop_values != layer_values:
            raise ConfigurationError(
                'search config: sampling.num_hops must exactly match model_search.num_layers so '
                'every searched model is trained and evaluated with the same neighborhood depth.'
            )
        if self.data.split.mode == 'sampler' and any(sampler == 'none' for sampler in self.sampling.samplers):
            raise ConfigurationError('search config: data.split.mode=sampler requires real samplers; sampling.samplers cannot include none.')
        return self


@dataclass(slots=True)
class DiagnosticsSettings:
    """Diagnostics-specific options."""

    gradient_variance_batches: int = 8

    def validate(self) -> None:
        if self.gradient_variance_batches <= 0:
            raise ConfigurationError('diagnostics.gradient_variance_batches must be positive.')


@dataclass(slots=True)
class DiagnosticsConfig:
    """Configuration for dataset diagnostics and gradient-variance analysis."""

    study_name: str = 'diagnostics'
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    data: BenchmarkDataConfig = field(default_factory=BenchmarkDataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    sampling: BenchmarkSamplingConfig = field(default_factory=BenchmarkSamplingConfig)
    diagnostics: DiagnosticsSettings = field(default_factory=DiagnosticsSettings)

    def validate(self) -> 'DiagnosticsConfig':
        self.runtime.validate()
        self.data.validate()
        self.model.validate()
        self.optimization.validate()
        self.sampling.validate()
        self.diagnostics.validate()
        for num_hops in self.sampling.num_hops:
            _validate_exact_neighborhood_depth(
                num_hops=int(num_hops),
                num_layers=int(self.model.num_layers),
                context='diagnostics config',
            )
        return self


@dataclass(slots=True)
class WilsonChecksSettings:
    """Wilson pre-experiment diagnostic settings."""

    q: Optional[float] = 1.0
    sample_counts: list[int] = field(default_factory=lambda: [10, 20, 50, 100])
    num_forests: int = 100
    num_weight_forests: int = 100
    max_pairs: int = 64
    pair_count: int = 8
    max_exact_nodes: int = 3000
    make_undirected: bool = False
    save_plots: bool = True
    seeds: list[int] | None = None
    aggregate_seeds: bool = False

    def validate(self) -> None:
        if self.q is not None and self.q <= 0.0:
            raise ConfigurationError('wilson_checks.q must be positive when provided.')
        if not self.sample_counts or any(value <= 0 for value in self.sample_counts):
            raise ConfigurationError('wilson_checks.sample_counts must contain positive integers.')
        if self.num_forests <= 0 or self.num_weight_forests <= 0:
            raise ConfigurationError('wilson_checks forest counts must be positive.')
        if self.max_pairs <= 0 or self.pair_count <= 0:
            raise ConfigurationError('wilson_checks pair counts must be positive.')
        if self.max_exact_nodes <= 0:
            raise ConfigurationError('wilson_checks.max_exact_nodes must be positive.')
        if self.seeds is not None:
            if not self.seeds:
                raise ConfigurationError(
                    'wilson_checks.seeds must not be empty when provided.'
                )
            if any(int(seed) < 0 for seed in self.seeds):
                raise ConfigurationError(
                    'wilson_checks.seeds must contain non-negative integer seeds.'
                )
            self.seeds = [int(seed) for seed in self.seeds]

        if self.aggregate_seeds and self.seeds is not None and len(self.seeds) < 2:
            raise ConfigurationError(
                'wilson_checks.aggregate_seeds=true requires at least two seeds.'
            )



@dataclass(slots=True)
class WilsonChecksConfig:
    """YAML-driven Wilson pre-experiment checks."""

    study_name: str = 'wilson_checks'
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    data: BenchmarkDataConfig = field(default_factory=BenchmarkDataConfig)
    sampling: BenchmarkSamplingConfig = field(default_factory=BenchmarkSamplingConfig)
    wilson_checks: WilsonChecksSettings = field(default_factory=WilsonChecksSettings)

    def validate(self) -> 'WilsonChecksConfig':
        self.runtime.validate()
        self.output
        self.data.validate()
        self.sampling.validate()
        self.wilson_checks.validate()
        return self



@dataclass(slots=True)
class SuiteConfig:
    """Configuration describing which study configs to run in sequence."""

    suite_name: str = 'suite'
    benchmark_config: str = 'configs/benchmark/publication_benchmark.yaml'
    diagnostics_config: str = 'configs/diagnostics/publication_diagnostics.yaml'

    def validate(self) -> 'SuiteConfig':
        if not self.benchmark_config:
            raise ConfigurationError('suite.benchmark_config must be provided.')
        if not self.diagnostics_config:
            raise ConfigurationError('suite.diagnostics_config must be provided.')
        return self
