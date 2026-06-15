from __future__ import annotations

from statistics import mean, pstdev
from typing import Any, Optional
import json
from pathlib import Path

import torch

from .base import CenterSampler, SamplerOutput
from .wilson_backend import WilsonBackendFactory


class WilsonCenterSampler(CenterSampler):
    """Wilson-based center selector with optional q-target budget search.

    legacy mode preserves the original behaviour: draw q-forest roots, redraw on
    shortfall if enabled, optionally fill shortfalls uniformly, and cap overshoots.

    q_target_search mode keeps the final selected set as a Wilson-root set only:
    it searches q and redraws complete q-forests until the realised root count is
    within target_budget_tolerance. It does not pad or cap.
    """

    name = 'wilson'

    def __init__(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        device: torch.device,
        *,
        backend: str = 'auto',
        q: Optional[float] = None,
        redraw_on_shortfall: bool = True,
        max_sampling_attempts: int = 8,
        allow_uniform_fill: bool = True,
        make_undirected: bool = False,
        remove_self_loops: bool = False,
        redraw_q_growth: float = 1.5,
        auto_q_method: str = 'mean_degree',
        q_calibration_trials: int = 8,
        q_calibration_max_iters: int = 16,
        q_calibration_tolerance: float = 0.03,
        q_min: float = 1.0e-8,
        q_max: float = 100.0,
        q_calibration_seed: int = 1729,
        budget_targeting: str = 'legacy',
        target_budget_tolerance: float = 0.03,
        max_budget_search_steps: int = 20,
        budget_target_failure_policy: str = 'warn_and_continue',
        weight_estimation: str = 'monte_carlo_forests',
        num_weight_forests: int = 256,
        weight_estimation_seed: int = 314159,
        online_exact_weight_max_nodes: int = 2500,
        dataset_name: Optional[str] = None,
        q_table_path: Optional[str] = None,
        q_table_missing_policy: str = 'error',
    ) -> None:
        self.num_nodes = int(num_nodes)
        self.device = device
        self.q_override = q
        self.max_sampling_attempts = int(max_sampling_attempts)
        self.allow_uniform_fill = bool(allow_uniform_fill)
        self.redraw_q_growth = float(redraw_q_growth)
        self.redraw_on_shortfall = bool(redraw_on_shortfall)
        self.auto_q_method = str(auto_q_method)
        self.q_calibration_trials = int(q_calibration_trials)
        self.q_calibration_max_iters = int(q_calibration_max_iters)
        self.q_calibration_tolerance = float(q_calibration_tolerance)
        self.q_min = float(q_min)
        self.q_max = float(q_max)
        self.q_calibration_seed = int(q_calibration_seed)
        self.budget_targeting = str(budget_targeting)
        self.target_budget_tolerance = float(target_budget_tolerance)
        self.max_budget_search_steps = int(max_budget_search_steps)
        self.budget_target_failure_policy = str(budget_target_failure_policy)
        self.weight_estimation = str(weight_estimation)
        self.num_weight_forests = int(num_weight_forests)
        self.weight_estimation_seed = int(weight_estimation_seed)
        self.online_exact_weight_max_nodes = int(online_exact_weight_max_nodes)
        self.dataset_name = dataset_name
        self._online_root_marginal_cache: dict[tuple[Any, ...], tuple[torch.Tensor, dict[str, float | str]]] = {}
        self.q_table_path = q_table_path
        self.q_table_missing_policy = str(q_table_missing_policy)
        self._q_table_cache: dict[str, Any] | None = None
        self._candidate_mask = torch.zeros(self.num_nodes, device=device, dtype=torch.bool)
        self._selected_mask = torch.zeros(self.num_nodes, device=device, dtype=torch.bool)
        self.backend = WilsonBackendFactory.create(
            backend_name=backend,
            edge_index=edge_index,
            num_nodes=num_nodes,
            make_undirected=make_undirected,
            drop_self_loops=remove_self_loops,
            device=device,
        )
        mean_degree = float(self.backend.graph.outgoing_mass.float().mean().item())
        self._mean_degree = max(mean_degree, 1e-6)
        self.backend_name = backend if backend != 'auto' else type(self.backend).__name__.replace('WilsonForestBackend', '').lower()

    def _budget_to_q(self, budget: int) -> float:
        budget = max(1, min(int(budget), self.num_nodes - 1))
        ratio = float(budget) / float(self.num_nodes)
        ratio = min(max(ratio, 1e-4), 1.0 - 1e-4)
        return self._mean_degree * ratio / (1.0 - ratio)

    def _load_q_table_payload(self) -> dict[str, Any]:
        if self._q_table_cache is not None:
            return self._q_table_cache
        if not self.q_table_path:
            raise ValueError(
                'Wilson auto_q_method=precomputed_table requires sampling.wilson.q_table_path.'
            )
        path = Path(self.q_table_path)
        if not path.exists():
            raise FileNotFoundError(f'Wilson q table not found: {path}')
        with path.open('r', encoding='utf-8') as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError(f'Wilson q table must contain a JSON object: {path}')
        self._q_table_cache = payload
        return payload

    def _lookup_precomputed_q(self, budget: int) -> float:
        payload = self._load_q_table_payload()
        datasets = payload.get('datasets')
        if not isinstance(datasets, dict) or not datasets:
            raise ValueError('Wilson q table is missing a non-empty datasets object.')

        dataset_key = self.dataset_name
        if dataset_key is None:
            if len(datasets) == 1:
                dataset_key = next(iter(datasets.keys()))
            else:
                raise ValueError(
                    'Wilson q table contains multiple datasets, but dataset_name was not supplied to the sampler.'
                )
        if dataset_key not in datasets:
            available = ', '.join(sorted(str(k) for k in datasets.keys()))
            raise KeyError(f'Dataset {dataset_key!r} not found in Wilson q table. Available: {available}')

        table = datasets[dataset_key].get('q_by_budget_count')
        if not isinstance(table, dict):
            raise ValueError(f'Wilson q table for {dataset_key!r} is missing q_by_budget_count.')
        row = table.get(str(int(budget)))
        if row is None:
            available = ', '.join(sorted(str(k) for k in table.keys()))
            raise KeyError(
                f'Budget {int(budget)} not found in Wilson q table for {dataset_key!r}. Available: {available}'
            )
        if isinstance(row, dict):
            q = row.get('q')
        else:
            q = row
        if q is None:
            raise ValueError(f'Wilson q table row for budget {int(budget)} has no q value.')
        q = float(q)
        if q <= 0.0:
            raise ValueError(f'Wilson q table contains non-positive q={q} for budget {int(budget)}.')
        return q

    def _resolve_initial_q(self, budget: int) -> float:
        if self.q_override is not None:
            return float(self.q_override)
        if self.auto_q_method == 'precomputed_table':
            try:
                return self._lookup_precomputed_q(int(budget))
            except Exception:
                if self.q_table_missing_policy == 'error':
                    raise
                if self.q_table_missing_policy == 'mean_degree':
                    return self._budget_to_q(budget)
                if self.q_table_missing_policy == 'empirical_binary_search':
                    # The full empirical search needs candidate_nodes and is handled by q_target_search.
                    return self._budget_to_q(budget)
                raise
        return self._budget_to_q(budget)

    def _eligible_roots(self, roots_mask: torch.Tensor, candidate_nodes: torch.Tensor) -> torch.Tensor:
        self._candidate_mask.zero_()
        self._candidate_mask[candidate_nodes] = True
        return torch.nonzero(roots_mask & self._candidate_mask, as_tuple=False).view(-1)

    def _draw_eligible_roots(self, candidate_nodes: torch.Tensor, q: float, *, generator: torch.Generator) -> torch.Tensor:
        _, roots_mask, _, _ = self.backend.sample_forest_with_q_factor(
            q=float(q),
            generator=generator,
            trace=False,
        )
        return self._eligible_roots(roots_mask, candidate_nodes)

    def _estimate_mean_root_count(self, candidate_nodes: torch.Tensor, q: float, *, seed: int) -> float:
        counts: list[float] = []
        trial_generator = torch.Generator(device='cpu')
        for trial_idx in range(self.q_calibration_trials):
            trial_generator.manual_seed(int(seed) + trial_idx)
            roots = self._draw_eligible_roots(candidate_nodes, q, generator=trial_generator)
            counts.append(float(roots.numel()))
        return float(sum(counts) / max(len(counts), 1))

    # def _search_q_for_budget(self, candidate_nodes: torch.Tensor, budget: int, *, seed: int) -> tuple[float, dict[str, float]]:
    #     target = float(max(1, int(budget)))
    #     low = float(self.q_min)
    #     high = float(max(self.q_max, self.q_min * 2.0))

    #     best_q = self._resolve_initial_q(int(budget))
    #     best_mean = self._estimate_mean_root_count(candidate_nodes, best_q, seed=seed)
    #     best_err = abs(best_mean - target) / max(target, 1.0)

    #     for step_idx in range(self.max_budget_search_steps):
    #         mid = 0.5 * (low + high)
    #         mid_mean = self._estimate_mean_root_count(
    #             candidate_nodes,
    #             mid,
    #             seed=seed + 10_000 + step_idx * max(self.q_calibration_trials, 1),
    #         )
    #         rel_err = abs(mid_mean - target) / max(target, 1.0)
    #         if rel_err < best_err:
    #             best_q = mid
    #             best_mean = mid_mean
    #             best_err = rel_err
    #         if rel_err <= self.q_calibration_tolerance:
    #             return mid, {
    #                 'q_search_estimated_budget': mid_mean,
    #                 'q_search_steps': float(step_idx + 1),
    #                 'q_search_rel_error': rel_err,
    #             }
    #         if mid_mean < target:
    #             low = mid
    #         else:
    #             high = mid

    #     return best_q, {
    #         'q_search_estimated_budget': best_mean,
    #         'q_search_steps': float(self.max_budget_search_steps),
    #         'q_search_rel_error': best_err,
    #     }


    def _search_q_for_budget(
        self,
        candidate_nodes: torch.Tensor,
        budget: int,
        *,
        seed: int,
    ) -> tuple[float, dict[str, float]]:
        target = float(max(1, int(budget)))
        low = float(self.q_min)
        high = float(max(self.q_max, self.q_min * 2.0))

        initial_q = float(self._resolve_initial_q(int(budget)))
        best_q = initial_q
        best_mean = self._estimate_mean_root_count(candidate_nodes, best_q, seed=seed)
        best_err = abs(best_mean - target) / max(target, 1.0)

        for step_idx in range(self.max_budget_search_steps):
            mid = 0.5 * (low + high)
            mid_mean = self._estimate_mean_root_count(
                candidate_nodes,
                mid,
                seed=seed + 10_000 + step_idx * max(self.q_calibration_trials, 1),
            )
            rel_err = abs(mid_mean - target) / max(target, 1.0)

            if rel_err < best_err:
                best_q = mid
                best_mean = mid_mean
                best_err = rel_err

            if rel_err <= self.q_calibration_tolerance:
                return mid, {
                    'q_resolved_initial': float(initial_q),
                    'q_search_final': float(mid),
                    'q_search_estimated_budget': float(mid_mean),
                    'q_search_steps': float(step_idx + 1),
                    'q_search_rel_error': float(rel_err),
                }

            if mid_mean < target:
                low = mid
            else:
                high = mid

        return best_q, {
            'q_resolved_initial': float(initial_q),
            'q_search_final': float(best_q),
            'q_search_estimated_budget': float(best_mean),
            'q_search_steps': float(self.max_budget_search_steps),
            'q_search_rel_error': float(best_err),
        }



    def _candidate_cache_key(self, candidate_nodes: torch.Tensor, q: float, budget: int) -> tuple[Any, ...]:
        candidate_nodes_cpu = candidate_nodes.detach().cpu().to(torch.long)
        if candidate_nodes_cpu.numel() == 0:
            checksum = 0
            first = -1
            last = -1
        else:
            # A compact deterministic fingerprint for the common fixed train-node pool.
            checksum = int(candidate_nodes_cpu.sum().item())
            first = int(candidate_nodes_cpu[0].item())
            last = int(candidate_nodes_cpu[-1].item())
        return (
            str(self.weight_estimation),
            int(self.num_weight_forests),
            int(self.weight_estimation_seed),
            int(candidate_nodes_cpu.numel()),
            checksum,
            first,
            last,
            round(float(q), 12),
            int(budget),
        )

    def _estimate_root_marginals_by_forests(
        self,
        candidate_nodes: torch.Tensor,
        q: float,
        *,
        seed: int,
    ) -> tuple[torch.Tensor, dict[str, float | str]]:
        counts = torch.zeros(self.num_nodes, dtype=torch.float32, device=self.device)
        trials = max(1, int(self.num_weight_forests))
        trial_generator = torch.Generator(device='cpu')

        for trial_idx in range(trials):
            trial_generator.manual_seed(int(seed) + trial_idx)
            roots = self._draw_eligible_roots(candidate_nodes, float(q), generator=trial_generator)
            if roots.numel() > 0:
                counts[roots.to(device=self.device, dtype=torch.long)] += 1.0

        probabilities = (counts / float(trials)).detach().cpu().to(torch.float32)
        return probabilities, {
            'wilson_online_probability_estimation': 'monte_carlo_root_marginals',
            'wilson_online_probability_num_forests': float(trials),
            'wilson_online_probability_q': float(q),
        }

    def _exact_root_marginals_dense(self, q: float) -> tuple[torch.Tensor, dict[str, float | str]]:
        if self.num_nodes > self.online_exact_weight_max_nodes:
            raise RuntimeError(
                f'exact Wilson online weights require dense solve for {self.num_nodes} nodes, '
                f'exceeding online_exact_weight_max_nodes={self.online_exact_weight_max_nodes}.'
            )

        rowptr = self.backend.graph.rowptr.detach().cpu().to(torch.long)
        col = self.backend.graph.col.detach().cpu().to(torch.long)
        weights = self.backend.graph.weights
        if weights is None:
            weights_cpu = torch.ones(col.numel(), dtype=torch.float64)
        else:
            weights_cpu = weights.detach().cpu().to(torch.float64)

        n = int(self.num_nodes)
        adjacency = torch.zeros((n, n), dtype=torch.float64)
        for src in range(n):
            left = int(rowptr[src].item())
            right = int(rowptr[src + 1].item())
            if right > left:
                adjacency[src, col[left:right]] += weights_cpu[left:right]

        degree = adjacency.sum(dim=1)
        laplacian = torch.diag(degree) - adjacency
        system = laplacian + float(q) * torch.eye(n, dtype=torch.float64)
        inv_system = torch.linalg.inv(system)
        probabilities = (float(q) * torch.diagonal(inv_system)).to(torch.float32)
        probabilities = torch.clamp(probabilities, min=0.0, max=1.0)
        return probabilities, {
            'wilson_online_probability_estimation': 'exact_dense_root_marginals_qLqI_inverse',
            'wilson_online_probability_num_forests': 0.0,
            'wilson_online_probability_q': float(q),
        }

    def _root_marginals_for_online_weights(
        self,
        candidate_nodes: torch.Tensor,
        q: float,
        *,
        budget: int,
    ) -> tuple[torch.Tensor, dict[str, float | str]]:
        key = self._candidate_cache_key(candidate_nodes, q, budget)
        cached = self._online_root_marginal_cache.get(key)
        if cached is not None:
            probabilities, metadata = cached
            cached_metadata = dict(metadata)
            cached_metadata['wilson_online_probability_cache_hit'] = 1.0
            return probabilities, cached_metadata

        metadata: dict[str, float | str]
        try:
            if self.weight_estimation == 'exact':
                probabilities, metadata = self._exact_root_marginals_dense(float(q))
            else:
                probabilities, metadata = self._estimate_root_marginals_by_forests(
                    candidate_nodes,
                    float(q),
                    seed=self.weight_estimation_seed,
                )
        except RuntimeError:
            # Large graphs should not crash online training just because exact
            # dense marginals are impractical. Fall back to the configured Monte
            # Carlo forest estimator and record that fallback explicitly.
            probabilities, metadata = self._estimate_root_marginals_by_forests(
                candidate_nodes,
                float(q),
                seed=self.weight_estimation_seed,
            )
            metadata['wilson_online_probability_exact_fallback_to_mc'] = 1.0

        metadata['wilson_online_probability_cache_hit'] = 0.0
        self._online_root_marginal_cache[key] = (probabilities, dict(metadata))
        return probabilities, metadata

    def center_inclusion_probabilities(
        self,
        *,
        center_nodes: torch.Tensor,
        candidate_nodes: torch.Tensor,
        budget: int,
        sample_metadata: dict[str, float | str],
        eps: float,
    ) -> tuple[torch.Tensor, dict[str, float | str]]:
        q_value = sample_metadata.get('q_used_for_sample')
        if not isinstance(q_value, (int, float)):
            q_value = sample_metadata.get('q_final')
        if not isinstance(q_value, (int, float)):
            q_value = sample_metadata.get('q_initial')
        if not isinstance(q_value, (int, float)):
            q_value = self._resolve_initial_q(int(budget))

        candidate_nodes = candidate_nodes.to(device=self.device, dtype=torch.long)
        center_nodes_cpu = center_nodes.detach().cpu().to(torch.long)
        probabilities_all, metadata = self._root_marginals_for_online_weights(
            candidate_nodes,
            float(q_value),
            budget=int(budget),
        )
        selected_probabilities = probabilities_all[center_nodes_cpu].to(torch.float32)
        selected_probabilities = torch.clamp(selected_probabilities, min=float(eps), max=1.0)

        metadata = dict(metadata)
        metadata.update(
            {
                'wilson_online_probability_uses_root_marginals': 1.0,
                'wilson_online_probability_budget': float(budget),
                'wilson_online_probability_selected_min': float(selected_probabilities.min().item()),
                'wilson_online_probability_selected_max': float(selected_probabilities.max().item()),
                'wilson_online_probability_selected_mean': float(selected_probabilities.mean().item()),
            }
        )
        if self.budget_targeting == 'expected_budget':
            metadata['wilson_online_probability_matches_final_centers_exactly'] = 1.0
        elif self.redraw_on_shortfall or self.allow_uniform_fill or self.budget_targeting != 'legacy':
            metadata['wilson_online_probability_matches_final_centers_exactly'] = 0.0
        else:
            metadata['wilson_online_probability_matches_final_centers_exactly'] = 1.0
        return selected_probabilities, metadata

    def _sample_with_q_target_search(
        self,
        candidate_nodes: torch.Tensor,
        budget: int,
        *,
        generator: torch.Generator,
    ) -> SamplerOutput:
        base_seed = int(generator.initial_seed())
        current_q, search_metadata = self._search_q_for_budget(
            candidate_nodes,
            int(budget),
            seed=self.q_calibration_seed + base_seed,
        )
        target = float(budget)
        best_roots: Optional[torch.Tensor] = None
        best_q = float(current_q)
        best_rel_err = float('inf')
        raw_counts: list[int] = []

        low_q = self.q_min
        high_q = max(self.q_max, current_q)
        for attempt_idx in range(self.max_sampling_attempts):
            roots = self._draw_eligible_roots(candidate_nodes, current_q, generator=generator)
            raw_counts.append(int(roots.numel()))
            rel_err = abs(float(roots.numel()) - target) / max(target, 1.0)
            if rel_err < best_rel_err:
                best_roots = roots
                best_q = float(current_q)
                best_rel_err = rel_err
            if rel_err <= self.target_budget_tolerance:
                q_used_for_sample = float(current_q)
                return SamplerOutput(
                    center_nodes=roots,
                    metadata={
                        'requested_budget': float(budget),
                        'realized_budget': float(roots.numel()),

                        # q semantics
                        'q_initial': float(search_metadata.get('q_resolved_initial', q_used_for_sample)),
                        'q_resolved_initial': float(search_metadata.get('q_resolved_initial', q_used_for_sample)),
                        'q_search_final': float(search_metadata.get('q_search_final', q_used_for_sample)),
                        'q_used_for_sample': q_used_for_sample,
                        'q_final': q_used_for_sample,

                        # sampler diagnostics
                        'attempts_used': float(attempt_idx + 1),
                        'q_redraws_used': float(attempt_idx),
                        'redraw_on_shortfall': float(self.redraw_on_shortfall),
                        'fallback_uniform': 0.0,
                        'allow_uniform_fill': 0.0,
                        'one_pass_wilson': float(attempt_idx == 0),
                        'budget_targeting': 1.0,
                        'budget_relative_error': float(rel_err),
                        'raw_mean_roots_before_cap': float(mean(raw_counts)) if raw_counts else 0.0,
                        'raw_std_roots_before_cap': float(pstdev(raw_counts)) if len(raw_counts) > 1 else 0.0,
                        **search_metadata,
                    },
                )

            if roots.numel() < budget:
                low_q = max(low_q, current_q)
            else:
                high_q = min(high_q, current_q)
            current_q = 0.5 * (low_q + high_q)
            current_q = max(float(self.q_min), min(float(self.q_max), float(current_q)))

        if best_roots is None:
            raise RuntimeError('Wilson q_target_search failed without producing any roots.')
        if self.budget_target_failure_policy == 'error':
            raise RuntimeError(
                'Wilson q_target_search failed to meet target_budget_tolerance. '
                f'Best relative error={best_rel_err:.6f}, tolerance={self.target_budget_tolerance:.6f}.'
            )
        q_used_for_sample = float(best_q)
        return SamplerOutput(
            center_nodes=best_roots,
            metadata={
                'requested_budget': float(budget),
                'realized_budget': float(best_roots.numel()),

                # q semantics
                'q_initial': float(search_metadata.get('q_resolved_initial', q_used_for_sample)),
                'q_resolved_initial': float(search_metadata.get('q_resolved_initial', q_used_for_sample)),
                'q_search_final': float(search_metadata.get('q_search_final', q_used_for_sample)),
                'q_used_for_sample': q_used_for_sample,
                'q_final': q_used_for_sample,

                # sampler diagnostics
                'attempts_used': float(self.max_sampling_attempts),
                'q_redraws_used': float(max(self.max_sampling_attempts - 1, 0)),
                'redraw_on_shortfall': float(self.redraw_on_shortfall),
                'fallback_uniform': 0.0,
                'allow_uniform_fill': 0.0,
                'one_pass_wilson': 0.0,
                'budget_targeting': 1.0,
                'budget_relative_error': float(best_rel_err),
                'raw_mean_roots_before_cap': float(mean(raw_counts)) if raw_counts else 0.0,
                'raw_std_roots_before_cap': float(pstdev(raw_counts)) if len(raw_counts) > 1 else 0.0,
                **search_metadata,
            },
        )


    def _sample_expected_budget(self, candidate_nodes: torch.Tensor, budget: int, *, generator: torch.Generator) -> SamplerOutput:
        """Draw one q-Wilson forest using q calibrated for the target budget.

        This mode preserves the Wilson root distribution: it does not cap,
        truncate, redraw to hit the target, or fill with uniform nodes. The
        requested budget is interpreted as an expected mini-batch size used to
        resolve q, not as an exact final batch-size constraint.
        """
        q_initial = float(self._resolve_initial_q(int(budget)))
        roots = self._draw_eligible_roots(candidate_nodes, q_initial, generator=generator)
        realized = int(roots.numel())
        if realized <= 0:
            raise RuntimeError(
                'Wilson expected_budget sampling produced zero eligible roots. '
                'Increase the target budget or recalibrate the q table.'
            )
        rel_err = abs(float(realized) - float(budget)) / max(float(budget), 1.0)
        metadata = {
            'requested_budget': float(budget),
            'target_expected_budget': float(budget),
            'realized_budget': float(realized),
            'q_initial': float(q_initial),
            'q_resolved_initial': float(q_initial),
            'q_used_for_sample': float(q_initial),
            'q_final': float(q_initial),
            'attempts_used': 1.0,
            'q_redraws_used': 0.0,
            'redraw_on_shortfall': 0.0,
            'fallback_uniform': 0.0,
            'allow_uniform_fill': 0.0,
            'one_pass_wilson': 1.0,
            'budget_targeting': 1.0,
            'expected_budget_targeting': 1.0,
            'postprocessed_fixed_budget': 0.0,
            'cap_or_truncate_used': 0.0,
            'uniform_fill_used': 0.0,
            'budget_relative_error': float(rel_err),
            'raw_roots_before_postprocessing': float(realized),
            'raw_mean_roots_before_cap': float(realized),
            'raw_std_roots_before_cap': 0.0,
        }
        return SamplerOutput(center_nodes=roots, metadata=metadata)

    def _sample_legacy(self, candidate_nodes: torch.Tensor, budget: int, *, generator: torch.Generator) -> SamplerOutput:
        q_initial = self._resolve_initial_q(budget)
        current_q = q_initial

        self._selected_mask.zero_()
        raw_root_counts: list[int] = []
        attempts_used = 0
        q_redraws_used = 0
        fallback_uniform = 0.0

        attempt_limit = self.max_sampling_attempts if self.redraw_on_shortfall else 1

        for attempt_idx in range(attempt_limit):
            attempts_used += 1
            eligible_roots = self._draw_eligible_roots(candidate_nodes, current_q, generator=generator)
            raw_root_counts.append(int(eligible_roots.numel()))

            if eligible_roots.numel() > 0:
                new_roots = eligible_roots[~self._selected_mask[eligible_roots]]
                if new_roots.numel() > 0:
                    permutation = torch.randperm(
                        new_roots.numel(),
                        generator=generator,
                        device=self.device,
                    )
                    new_roots = new_roots[permutation]

                    need = budget - int(self._selected_mask[candidate_nodes].sum().item())
                    chosen_now = new_roots[:need]
                    self._selected_mask[chosen_now] = True

            if int(self._selected_mask[candidate_nodes].sum().item()) >= budget:
                break

            if self.redraw_on_shortfall and attempt_idx < attempt_limit - 1:
                current_q *= self.redraw_q_growth
                q_redraws_used += 1

        chosen = torch.nonzero(self._selected_mask[candidate_nodes], as_tuple=False).view(-1)
        chosen = candidate_nodes[chosen]
        if chosen.numel() < budget and self.allow_uniform_fill:
            remaining = candidate_nodes[~self._selected_mask[candidate_nodes]]
            if remaining.numel() > 0:
                permutation = torch.randperm(remaining.numel(), generator=generator, device=self.device)
                fill = remaining[permutation[: budget - chosen.numel()]]
                chosen = torch.cat([chosen, fill], dim=0)
                fallback_uniform = 1.0
        if chosen.numel() > budget:
            permutation = torch.randperm(chosen.numel(), generator=generator, device=self.device)
            chosen = chosen[permutation[:budget]]

        metadata = {
            'requested_budget': float(budget),
            'realized_budget': float(chosen.numel()),
            'q_initial': float(q_initial),
            'q_final': float(current_q),
            'attempts_used': float(attempts_used),
            'q_redraws_used': float(q_redraws_used),
            'redraw_on_shortfall': float(self.redraw_on_shortfall),
            'fallback_uniform': float(fallback_uniform),
            'allow_uniform_fill': float(self.allow_uniform_fill),
            'one_pass_wilson': float(
                (not self.redraw_on_shortfall)
                and attempts_used == 1
                and fallback_uniform == 0.0
            ),
            'budget_targeting': 0.0,
            'budget_relative_error': abs(float(chosen.numel()) - float(budget)) / max(float(budget), 1.0),
            'raw_mean_roots_before_cap': float(mean(raw_root_counts)) if raw_root_counts else 0.0,
            'raw_std_roots_before_cap': float(pstdev(raw_root_counts)) if len(raw_root_counts) > 1 else 0.0,
        }
        return SamplerOutput(center_nodes=chosen, metadata=metadata)

    def sample(self, candidate_nodes: torch.Tensor, budget: int, *, generator: torch.Generator) -> SamplerOutput:
        if candidate_nodes.numel() == 0:
            raise ValueError('candidate_nodes must be non-empty.')
        budget = min(int(budget), int(candidate_nodes.numel()))
        if budget <= 0:
            raise ValueError('budget must be positive.')
        candidate_nodes = candidate_nodes.to(device=self.device, dtype=torch.long)
        if self.budget_targeting == 'expected_budget':
            return self._sample_expected_budget(candidate_nodes, budget, generator=generator)
        if self.budget_targeting == 'q_target_search':
            return self._sample_with_q_target_search(candidate_nodes, budget, generator=generator)
        return self._sample_legacy(candidate_nodes, budget, generator=generator)


class AdaptiveWilsonCenterSampler(WilsonCenterSampler):
    """Baseline Adaptive Wilson sampler with q_i = alpha.

    The current Wilson backend accepts a scalar q. The supervisor baseline
    q_i = alpha for every node is exactly that scalar case, so this class
    wraps WilsonCenterSampler and records adaptive-specific metadata.
    """

    name = 'adaptive_wilson'

    def __init__(
        self,
        *args,
        alpha: float = 1.0,
        q_i_mode: str = 'alpha',
        q_mode: str = 'manual_alpha',
        **kwargs,
    ) -> None:
        if q_i_mode != 'alpha':
            raise ValueError('AdaptiveWilsonCenterSampler currently supports only q_i_mode=alpha.')
        if alpha <= 0.0:
            raise ValueError('AdaptiveWilsonCenterSampler alpha must be positive.')
        if q_mode not in {'manual_alpha', 'precomputed_budget_q'}:
            raise ValueError('AdaptiveWilsonCenterSampler q_mode must be manual_alpha or precomputed_budget_q.')
        # In manual_alpha mode, Adaptive Wilson is the scalar-q baseline q_i=alpha.
        # In precomputed_budget_q mode, q is resolved by WilsonCenterSampler from
        # sampling.wilson.auto_q_method/q_table_path for each target budget. This
        # preserves the Wilson draw and avoids cap/truncate/fill post-processing.
        if q_mode == 'manual_alpha':
            kwargs['q'] = float(alpha)
        super().__init__(*args, **kwargs)
        self.alpha = float(alpha)
        self.q_i_mode = str(q_i_mode)
        self.q_mode = str(q_mode)

    def sample(self, candidate_nodes: torch.Tensor, budget: int, *, generator: torch.Generator) -> SamplerOutput:
        out = super().sample(candidate_nodes, budget, generator=generator)
        metadata = dict(out.metadata)
        metadata.update(
            {
                'adaptive_wilson': 1.0,
                'adaptive_q_i_mode_alpha': 1.0,
                'adaptive_alpha': float(self.alpha),
                'adaptive_q_mode_precomputed_budget_q': float(self.q_mode == 'precomputed_budget_q'),
                'adaptive_q_mode_manual_alpha': float(self.q_mode == 'manual_alpha'),
            }
        )
        return SamplerOutput(center_nodes=out.center_nodes, metadata=metadata)
