from __future__ import annotations
import time
from pathlib import Path
from typing import Optional

import torch

from ..analysis.utils import build_center_sampler
from ..config.loader import load_config
from ..config.schema import TrainConfig
from ..data.loading import load_dataset
from ..data.splitting import apply_sampler_split
from ..exceptions import ConfigurationError
from ..training.loop import (
    EarlyStopper,
    RunResult,
    aggregate_sampler_metadata,
    build_loss_fn,
    build_model,
    build_optimizer,
    build_scheduler,
    evaluate_model,
    train_one_epoch,
    train_one_epoch_full_graph_semisupervised,
    train_one_epoch_on_reduced_graph,
)
from ..training.preprocessing import prepare_reduced_graph, resolve_budget_count
from ..utils.device import resolve_device
from ..utils.io import assert_pristine_run_dir, dataclass_to_dict, ensure_dir, write_csv, write_json, write_yaml
from ..utils.logging import configure_logging, get_logger, remove_logging_handler
from ..utils.random import build_generator, seed_everything
from .common import parse_config_arg, prepare_experiment_root

LOGGER = get_logger(__name__)


def _normalized_weighting_enabled(config: TrainConfig) -> bool:
    return bool(config.optimization.use_normalized_weights or config.optimization.use_static_weight_normalization)


def _validate_static_weight_normalization_compatibility(config: TrainConfig) -> None:
    if config.optimization.use_static_weight_normalization and not (
        config.dataset.split.mode == 'sampler' and config.sampling.mode == 'static'
    ):
        raise ConfigurationError(
            'optimization.use_static_weight_normalization=true is the legacy static-only flag and requires '
            'dataset.split.mode=sampler and sampling.mode=static. For online batches use optimization.use_normalized_weights=true.'
        )

    if not _normalized_weighting_enabled(config):
        return

    if config.sampling.name == 'none' and config.sampling.mode == 'online':
        raise ConfigurationError('online normalized weighting requires a real sampler; sampling.name cannot be none.')

    if not (config.dataset.split.mode == 'sampler' and config.sampling.mode == 'static'):
        return

    if config.sampling.name not in {'wilson', 'adaptive_wilson'}:
        return

    if (
        config.sampling.wilson.weight_estimation == 'exact'
        and config.optimization.strict_one_pass_wilson_weights
    ):
        if config.sampling.wilson.redraw_on_shortfall:
            raise ConfigurationError(
                'Wilson exact normalized weighting with strict_one_pass_wilson_weights=true '
                'requires sampling.wilson.redraw_on_shortfall=false. '
                'Otherwise q(L+qI)^(-1) weights do not match the actual redraw-based sampler.'
            )

        if config.sampling.wilson.allow_uniform_fill:
            raise ConfigurationError(
                'Wilson exact normalized weighting with strict_one_pass_wilson_weights=true '
                'requires sampling.wilson.allow_uniform_fill=false. '
                'Uniform fill changes the final inclusion distribution.'
            )

        if config.sampling.wilson.budget_targeting != 'legacy':
            raise ConfigurationError(
                'Wilson exact normalized weighting currently supports only '
                'sampling.wilson.budget_targeting=legacy. '
                'Use weight_estimation=monte_carlo_forests for q_target_search.'
            )


def _prepare_run_root(config: TrainConfig, *, run_root: Optional[Path]) -> Path:
    if run_root is not None:
        ensure_dir(run_root)
        ensure_dir(run_root / 'logs')
        ensure_dir(run_root / 'plots')
        ensure_dir(run_root / 'tables')
        assert_pristine_run_dir(run_root)
        return run_root
    return prepare_experiment_root(
        config.output.root,
        config.output.experiment_name,
        config.run_name,
        overwrite=config.output.overwrite,
        append_timestamp=config.output.append_timestamp,
        timestamp_format=config.output.timestamp_format,
    )


def _checkpoint_payload(model: torch.nn.Module, config: TrainConfig, result: RunResult) -> dict[str, object]:
    return {
        'model_state_dict': model.state_dict(),
        'config': dataclass_to_dict(config),
        'best_epoch': result.best_epoch,
        'best_val_loss': result.best_val_loss,
        'best_monitor_value': result.best_monitor_value,
        'monitor_metric': result.monitor_metric,
        'result': result.to_dict(),
    }


def monitor_value(metrics, metric_name: str) -> float:
    value = getattr(metrics, metric_name)
    if value is None:
        raise ConfigurationError(f'Monitor metric {metric_name} was not computed by evaluation.')
    return float(value)


def _cumulative_minibatch_updates_for_epoch(config: TrainConfig, epoch: int) -> int:
    if config.sampling.mode == 'online' and config.sampling.online_schedule == 'per_batch':
        return int(epoch) * int(config.sampling.steps_per_epoch)
    return int(epoch)


def _cumulative_optimizer_updates_for_epoch(config: TrainConfig, epoch: int) -> int:
    if config.sampling.mode == 'online' and config.sampling.online_schedule == 'per_batch':
        minibatches = int(epoch) * int(config.sampling.steps_per_epoch)
        return (minibatches + int(config.optimization.accumulation_steps) - 1) // int(config.optimization.accumulation_steps)
    return int(epoch)


def _first_history_row_reaching_fraction(
    history: list[dict[str, object]],
    *,
    metric_key: str,
    best_value: float,
    fraction: float,
) -> dict[str, object] | None:
    if best_value == float('-inf'):
        return None
    threshold = float(fraction) * float(best_value)
    for row in history:
        value = row.get(metric_key)
        if isinstance(value, (int, float)) and float(value) >= threshold:
            return row
    return None


def _uses_sampler_split(config: TrainConfig) -> bool:
    return config.dataset.split.mode == 'sampler'


def _resolved_budget_reference_count(config: TrainConfig, dataset) -> int:
    if _uses_sampler_split(config):
        return int(dataset.data.num_nodes)
    return int(dataset.train_nodes.numel())


def _training_objective(config: TrainConfig) -> str:
    return 'full_graph_semi_supervised' if _uses_sampler_split(config) else 'sampled_subgraph'


def run_training_experiment(
    config: TrainConfig,
    *,
    run_root: Optional[Path] = None,
    configure_root_logging: bool = True,
) -> tuple[RunResult, Path]:
    config.validate()
    _validate_static_weight_normalization_compatibility(config)
    run_root = _prepare_run_root(config, run_root=run_root)

    run_log_handler = None
    if configure_root_logging:
        run_log_handler = configure_logging(
            config.logging.level,
            run_root / 'logs' / 'run.log',
            log_to_stdout=config.logging.log_to_stdout,
            reset_handlers=True,
        )

    try:
        resolution = resolve_device(config.runtime.device, config.runtime.device_policy)
        device = torch.device(resolution.resolved)
        seed_everything(
            config.runtime.seed,
            deterministic=config.runtime.deterministic,
            torch_num_threads=config.runtime.torch_num_threads,
        )

        resolved_config_path = run_root / 'resolved_config.yaml'
        write_yaml(resolved_config_path, dataclass_to_dict(config))

        raw_dataset = load_dataset(config.dataset, device=torch.device('cpu'), seed=config.runtime.seed)
        raw_dataset.metadata.setdefault('feature_mode', config.dataset.feature_mode)
        sampler = build_center_sampler(config.sampling, raw_dataset, device)
        dataset = apply_sampler_split(
            raw_dataset,
            config.dataset,
            sampler,
            config.sampling,
            seed=config.runtime.seed,
            use_static_weight_normalization=(
                _normalized_weighting_enabled(config)
                and config.dataset.split.mode == 'sampler'
                and config.sampling.mode == 'static'
            ),
            static_weight_normalization_eps=config.optimization.static_weight_normalization_eps,
            strict_one_pass_wilson_weights=config.optimization.strict_one_pass_wilson_weights,
        )
        generator = build_generator(torch.device('cpu'), config.runtime.seed)

        model = build_model(config.model, in_channels=dataset.num_features, out_channels=dataset.num_classes).to(device)
        loss_fn, class_weights = build_loss_fn(config.optimization, dataset, device)
        optimizer = build_optimizer(model, config.optimization)
        scheduler = build_scheduler(optimizer, config.optimization)

        evaluation_device = device
        if config.evaluation.device == 'cpu':
            evaluation_device = torch.device('cpu')
        elif config.evaluation.device == 'cuda' and resolution.resolved == 'cuda':
            evaluation_device = torch.device('cuda')

        amp_enabled = bool(config.optimization.use_amp) and device.type == 'cuda'
        scaler = torch.amp.GradScaler('cuda', enabled=amp_enabled) if device.type == 'cuda' else None

        best_state: Optional[dict[str, torch.Tensor]] = None
        best_epoch = 0
        best_val_loss = float('inf')
        best_monitor_value = float('-inf')
        stopped_early = False
        history: list[dict[str, object]] = []
        batch_history_rows: list[dict[str, object]] = []
        epoch_times: list[float] = []
        all_batch_stats = []
        total_center_node_ids_seen: set[int] = set()
        total_unique_center_nodes_seen = 0
        cumulative_minibatch_updates = 0
        cumulative_optimizer_updates = 0

        early_stopper = EarlyStopper(config.optimization.early_stopping_patience, min_delta=1e-4)
        total_start = time.perf_counter()
        training_objective = _training_objective(config)
        resolved_budget = resolve_budget_count(config.sampling, _resolved_budget_reference_count(config, dataset))

        static_reduced = None
        preprocessing_cache_path = None
        if config.sampling.mode == 'static' and training_objective != 'full_graph_semi_supervised':
            static_reduced = prepare_reduced_graph(
                dataset,
                sampler,
                config.sampling,
                generator,
                seed=config.runtime.seed,
                cache_root=Path(config.output.root),
            )
            preprocessing_cache_path = static_reduced.cache_path
            write_json(
                run_root / 'preprocessed_graph.json',
                {
                    'cache_path': preprocessing_cache_path,
                    'metadata': {
                        str(k): v
                        for k, v in static_reduced.metadata.items()
                        if isinstance(v, (int, float, str, bool, type(None)))
                    },
                },
            )

        LOGGER.info(
            'Starting run dataset=%s feature_mode=%s model=%s sampler=%s split_mode=%s objective=%s mode=%s schedule=%s budget=%s(%s) seed=%d device=%s hops=%d layers=%d normalized_weights=%s',
            dataset.name,
            config.dataset.feature_mode,
            config.model.family,
            config.sampling.name,
            config.dataset.split.mode,
            training_objective,
            config.sampling.mode,
            config.sampling.online_schedule,
            config.sampling.budget_value if config.sampling.budget_value is not None else config.sampling.batch_size,
            config.sampling.budget_unit,
            config.runtime.seed,
            resolution.resolved,
            config.sampling.num_hops,
            config.model.num_layers,
            _normalized_weighting_enabled(config),
        )
        LOGGER.info(
            'Split sizes train=%d val=%d test=%d total=%d',
            int(dataset.train_nodes.numel()),
            int(dataset.val_nodes.numel()),
            int(dataset.test_nodes.numel()),
            int(dataset.data.num_nodes),
        )

        for epoch in range(1, config.optimization.epochs + 1):
            epoch_start = time.perf_counter()
            epoch_preprocess_cache_path = preprocessing_cache_path
            if training_objective == 'full_graph_semi_supervised':
                train_loss, batch_stats, epoch_training_diagnostics, epoch_batch_history = train_one_epoch_full_graph_semisupervised(
                    model=model,
                    dataset=dataset,
                    optimizer=optimizer,
                    loss_fn=loss_fn,
                    device=device,
                    grad_clip_norm=config.optimization.grad_clip_norm,
                    scaler=scaler,
                    amp_enabled=amp_enabled,
                )
            elif config.sampling.mode == 'online' and config.sampling.online_schedule == 'per_batch':
                train_loss, batch_stats, epoch_training_diagnostics, epoch_batch_history = train_one_epoch(
                    model=model,
                    dataset=dataset,
                    sampler=sampler,
                    sampling=config.sampling,
                    optimizer=optimizer,
                    loss_fn=loss_fn,
                    generator=generator,
                    device=device,
                    grad_clip_norm=config.optimization.grad_clip_norm,
                    accumulation_steps=config.optimization.accumulation_steps,
                    scaler=scaler,
                    amp_enabled=amp_enabled,
                    use_normalized_weights=_normalized_weighting_enabled(config),
                    weight_eps=config.optimization.static_weight_normalization_eps,
                )
            else:
                reduced_graph = static_reduced
                if reduced_graph is None:
                    reduced_graph = prepare_reduced_graph(
                        dataset,
                        sampler,
                        config.sampling,
                        generator,
                        seed=config.runtime.seed + epoch,
                        cache_root=Path(config.output.root),
                        force_cache=False,
                    )
                    epoch_preprocess_cache_path = reduced_graph.cache_path
                train_loss, batch_stats, epoch_training_diagnostics, epoch_batch_history = train_one_epoch_on_reduced_graph(
                    model=model,
                    reduced_dataset=reduced_graph.dataset,
                    optimizer=optimizer,
                    loss_fn=loss_fn,
                    device=device,
                    grad_clip_norm=config.optimization.grad_clip_norm,
                    scaler=scaler,
                    amp_enabled=amp_enabled,
                    use_normalized_weights=_normalized_weighting_enabled(config),
                    weight_eps=config.optimization.static_weight_normalization_eps,
                )
            epoch_time = time.perf_counter() - epoch_start
            epoch_times.append(epoch_time)
            all_batch_stats.extend(batch_stats)

            epoch_minibatch_updates = int(epoch_training_diagnostics.get('observed_minibatches', len(batch_stats)))
            epoch_optimizer_updates = int(epoch_training_diagnostics.get('optimizer_updates', 1))
            cumulative_minibatch_updates += epoch_minibatch_updates
            cumulative_optimizer_updates += epoch_optimizer_updates
            epoch_unique_ids = epoch_training_diagnostics.get('unique_center_node_ids')
            if isinstance(epoch_unique_ids, list):
                total_center_node_ids_seen.update(int(value) for value in epoch_unique_ids)
                total_unique_center_nodes_seen = len(total_center_node_ids_seen)
            else:
                total_unique_center_nodes_seen = max(
                    total_unique_center_nodes_seen,
                    int(epoch_training_diagnostics.get('unique_center_nodes_epoch', 0)),
                )

            for batch_row in epoch_batch_history:
                step_in_epoch = int(batch_row.get('step_in_epoch', 1))
                global_minibatch_update = (
                    ((epoch - 1) * int(config.sampling.steps_per_epoch) + step_in_epoch)
                    if config.sampling.mode == 'online' and config.sampling.online_schedule == 'per_batch'
                    else epoch
                )
                batch_history_rows.append(
                    {
                        'epoch': int(epoch),
                        'cumulative_minibatch_updates': int(global_minibatch_update),
                        'cumulative_optimizer_updates': int(
                            (global_minibatch_update + int(config.optimization.accumulation_steps) - 1)
                            // int(config.optimization.accumulation_steps)
                        ),
                        'sampler_name': config.sampling.name,
                        'sampling_mode': config.sampling.mode,
                        'online_schedule': config.sampling.online_schedule,
                        'normalized_weighting_enabled': bool(_normalized_weighting_enabled(config)),
                        **batch_row,
                    }
                )

            train_metrics = evaluate_model(
                model,
                dataset,
                dataset.train_nodes,
                loss_fn=loss_fn,
                evaluation_mode=config.evaluation.mode,
                evaluation_device=evaluation_device,
                evaluation_batch_size=config.evaluation.batch_size,
                non_blocking=config.evaluation.non_blocking,
            ) if config.evaluation.compute_train_metrics else None

            val_metrics = evaluate_model(
                model,
                dataset,
                dataset.val_nodes,
                loss_fn=loss_fn,
                evaluation_mode=config.evaluation.mode,
                evaluation_device=evaluation_device,
                evaluation_batch_size=config.evaluation.batch_size,
                non_blocking=config.evaluation.non_blocking,
            )

            test_metrics = None
            if config.evaluation.track_test_metrics_during_training:
                test_metrics = evaluate_model(
                    model,
                    dataset,
                    dataset.test_nodes,
                    loss_fn=loss_fn,
                    evaluation_mode=config.evaluation.mode,
                    evaluation_device=evaluation_device,
                    evaluation_batch_size=config.evaluation.batch_size,
                    non_blocking=config.evaluation.non_blocking,
                )

            score = monitor_value(val_metrics, config.optimization.monitor_metric)
            scheduler.step(score)
            if score > best_monitor_value or (score == best_monitor_value and val_metrics.loss < best_val_loss):
                best_monitor_value = float(score)
                best_val_loss = float(val_metrics.loss)
                best_epoch = epoch
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

            history_row = {
                'epoch': epoch,
                'elapsed_runtime': float(time.perf_counter() - total_start),
                'epoch_time': float(epoch_time),
                'cumulative_minibatch_updates': int(cumulative_minibatch_updates),
                'cumulative_optimizer_updates': int(cumulative_optimizer_updates),
                'train_loss': float(train_loss),
                'expected_minibatch_loss': float(epoch_training_diagnostics.get('expected_minibatch_loss', train_loss)),
                'epoch_minibatch_loss_mean': float(epoch_training_diagnostics.get('epoch_batch_loss_mean', train_loss)),
                'epoch_minibatch_loss_variance': float(epoch_training_diagnostics.get('epoch_batch_loss_variance', 0.0)),
                'epoch_minibatch_loss_std': float(epoch_training_diagnostics.get('epoch_batch_loss_std', 0.0)),
                'minibatch_loss_minus_expectation_mean': float(
                    epoch_training_diagnostics.get('minibatch_loss_minus_expectation_mean', 0.0)
                ),
                'minibatch_loss_minus_expectation_squared_mean': float(
                    epoch_training_diagnostics.get('minibatch_loss_minus_expectation_squared_mean', 0.0)
                ),
                'epoch_unique_center_nodes': float(epoch_training_diagnostics.get('unique_center_nodes_epoch', 0.0)),
                'epoch_unique_center_node_coverage': float(
                    epoch_training_diagnostics.get('unique_center_node_coverage_epoch', 0.0)
                ),
                'total_unique_center_nodes_seen': int(total_unique_center_nodes_seen),
                'total_unique_center_node_coverage': float(
                    total_unique_center_nodes_seen / max(int(dataset.train_nodes.numel()), 1)
                ),
                'epoch_observed_minibatches': int(epoch_minibatch_updates),
                'epoch_optimizer_updates': int(epoch_optimizer_updates),
                'train_accuracy': float(train_metrics.accuracy) if train_metrics is not None else None,
                'train_macro_f1': float(train_metrics.macro_f1) if train_metrics is not None else None,
                'train_balanced_accuracy': float(train_metrics.balanced_accuracy) if train_metrics is not None else None,
                'val_loss': float(val_metrics.loss),
                'val_accuracy': float(val_metrics.accuracy),
                'val_macro_f1': float(val_metrics.macro_f1),
                'val_balanced_accuracy': float(val_metrics.balanced_accuracy),
                'lr': float(optimizer.param_groups[0]['lr']),
                'monitor_metric': config.optimization.monitor_metric,
                'monitor_value': float(score),
                'sampling_mode': config.sampling.mode,
                'online_schedule': config.sampling.online_schedule,
                'split_mode': config.dataset.split.mode,
                'training_objective': training_objective,
                'normalized_weighting_enabled': bool(_normalized_weighting_enabled(config)),
                'resolved_budget': int(resolved_budget),
                'total_nodes': int(dataset.data.num_nodes),
                'train_node_count': int(dataset.train_nodes.numel()),
                'val_node_count': int(dataset.val_nodes.numel()),
                'test_node_count': int(dataset.test_nodes.numel()),
                'preprocessed_cache_path': epoch_preprocess_cache_path,
            }
            if test_metrics is not None:
                history_row.update(
                    {
                        'test_accuracy': float(test_metrics.accuracy),
                        'test_macro_f1': float(test_metrics.macro_f1),
                        'test_balanced_accuracy': float(test_metrics.balanced_accuracy),
                    }
                )
            history.append(history_row)

            LOGGER.info(
                'epoch=%03d train_loss=%.4f val_acc=%.4f val_macro_f1=%.4f lr=%.6f time=%.3fs',
                epoch,
                train_loss,
                val_metrics.accuracy,
                val_metrics.macro_f1,
                optimizer.param_groups[0]['lr'],
                epoch_time,
            )

            if epoch >= config.optimization.min_epochs and early_stopper.step(score):
                stopped_early = True
                LOGGER.info('Early stopping triggered at epoch %d.', epoch)
                break

        if best_state is None:
            raise RuntimeError('Training finished without any checkpoint state.')
        model.load_state_dict(best_state)
        total_training_time = time.perf_counter() - total_start

        final_train_metrics = evaluate_model(
            model,
            dataset,
            dataset.train_nodes,
            loss_fn=loss_fn,
            evaluation_mode=config.evaluation.mode,
            evaluation_device=evaluation_device,
            evaluation_batch_size=config.evaluation.batch_size,
            non_blocking=config.evaluation.non_blocking,
        )
        final_val_metrics = evaluate_model(
            model,
            dataset,
            dataset.val_nodes,
            loss_fn=loss_fn,
            evaluation_mode=config.evaluation.mode,
            evaluation_device=evaluation_device,
            evaluation_batch_size=config.evaluation.batch_size,
            non_blocking=config.evaluation.non_blocking,
        )
        final_test_metrics = evaluate_model(
            model,
            dataset,
            dataset.test_nodes,
            loss_fn=loss_fn,
            evaluation_mode=config.evaluation.mode,
            evaluation_device=evaluation_device,
            evaluation_batch_size=config.evaluation.batch_size,
            non_blocking=config.evaluation.non_blocking,
        )

        average_center_nodes = float(sum(stat.center_nodes for stat in all_batch_stats) / max(len(all_batch_stats), 1))
        checkpoint_path = ''
        metrics_path = run_root / 'metrics.json'
        history_path = run_root / 'history.csv'
        batch_history_path = run_root / 'batch_history.csv'

        best_cumulative_minibatch_updates = _cumulative_minibatch_updates_for_epoch(config, best_epoch)
        best_cumulative_optimizer_updates = _cumulative_optimizer_updates_for_epoch(config, best_epoch)
        first_95pct_row = _first_history_row_reaching_fraction(
            history,
            metric_key='monitor_value',
            best_value=best_monitor_value,
            fraction=0.95,
        )
        first_epoch_to_95pct_best_monitor = (
            int(first_95pct_row['epoch']) if first_95pct_row is not None else None
        )
        first_cumulative_minibatch_updates_to_95pct_best_monitor = (
            int(first_95pct_row['cumulative_minibatch_updates'])
            if first_95pct_row is not None and first_95pct_row.get('cumulative_minibatch_updates') is not None
            else None
        )

        result = RunResult(
            seed=int(config.runtime.seed),
            dataset_name=dataset.name,
            dataset_kind=dataset.kind,
            feature_mode=config.dataset.feature_mode,
            model_family=config.model.family,
            sampler_name=config.sampling.name,
            sampling_mode=config.sampling.mode,
            online_schedule=config.sampling.online_schedule,
            budget_unit=config.sampling.budget_unit,
            budget_value=float(config.sampling.budget_value) if config.sampling.budget_value is not None else None,
            split_mode=config.dataset.split.mode,
            training_objective=training_objective,
            resolved_budget=int(resolved_budget),
            total_nodes=int(dataset.data.num_nodes),
            train_node_count=int(dataset.train_nodes.numel()),
            val_node_count=int(dataset.val_nodes.numel()),
            test_node_count=int(dataset.test_nodes.numel()),
            split_val_ratio_of_remaining=(
                float(dataset.metadata.get('split_val_ratio_of_remaining'))
                if dataset.metadata.get('split_val_ratio_of_remaining') is not None
                else None
            ),
            preprocessed_cache_path=preprocessing_cache_path,
            requested_device=config.runtime.device,
            resolved_device=resolution.resolved,
            device_warnings=resolution.warnings,
            best_epoch=best_epoch,
            best_cumulative_minibatch_updates=int(best_cumulative_minibatch_updates),
            best_cumulative_optimizer_updates=int(best_cumulative_optimizer_updates),
            first_epoch_to_95pct_best_monitor=first_epoch_to_95pct_best_monitor,
            first_cumulative_minibatch_updates_to_95pct_best_monitor=first_cumulative_minibatch_updates_to_95pct_best_monitor,
            total_minibatch_updates=int(cumulative_minibatch_updates),
            total_optimizer_updates=int(cumulative_optimizer_updates),
            stopped_early=stopped_early,
            train_metrics=final_train_metrics.to_dict(),
            val_metrics=final_val_metrics.to_dict(),
            test_metrics=final_test_metrics.to_dict(),
            training_time_per_epoch=float(sum(epoch_times) / max(len(epoch_times), 1)),
            total_training_time=float(total_training_time),
            inference_latency=float(final_test_metrics.latency),
            average_sampled_nodes=float(sum(stat.sampled_nodes for stat in all_batch_stats) / max(len(all_batch_stats), 1)),
            average_sampled_edges=float(sum(stat.sampled_edges for stat in all_batch_stats) / max(len(all_batch_stats), 1)),
            average_center_nodes=float(average_center_nodes),
            budget_match_ratio=float(average_center_nodes / max(resolved_budget, 1)),
            sampler_diagnostics=aggregate_sampler_metadata(all_batch_stats),
            checkpoint_path=checkpoint_path,
            best_val_loss=float(best_val_loss),
            best_monitor_value=float(best_monitor_value),
            monitor_metric=config.optimization.monitor_metric,
            final_lr=float(optimizer.param_groups[0]['lr']),
            epoch_history=history,
            class_weights=[float(value) for value in class_weights.cpu().tolist()] if class_weights is not None else None,
            split_metadata={
                str(k): v
                for k, v in dataset.metadata.items()
                if isinstance(v, (int, float, str, bool, type(None)))
            },
            evaluation_mode=config.evaluation.mode,
            evaluation_device=str(evaluation_device),
            evaluation_batch_size=int(config.evaluation.batch_size),
            run_root=str(run_root),
            resolved_config_path=str(resolved_config_path),
            metrics_path=str(metrics_path),
            history_path=str(history_path),
            batch_history_path=str(batch_history_path),
        )

        if config.output.save_checkpoints:
            checkpoint_path = str(run_root / 'checkpoint.pt')
            torch.save(_checkpoint_payload(model, config, result), checkpoint_path)
            result.checkpoint_path = checkpoint_path

        write_json(metrics_path, result.to_dict())
        write_csv(history_path, history)
        write_csv(batch_history_path, batch_history_rows)
        LOGGER.info(
            'Finished run. Test accuracy=%.4f test macro_f1=%.4f',
            final_test_metrics.accuracy,
            final_test_metrics.macro_f1,
        )
        return result, run_root
    finally:
        if configure_root_logging and run_log_handler is not None:
            remove_logging_handler(run_log_handler)


def main() -> None:
    args = parse_config_arg('Run one sampled node-classification training experiment.')
    config = load_config(TrainConfig, args.config)
    run_training_experiment(config)


if __name__ == '__main__':  # pragma: no cover
    main()
