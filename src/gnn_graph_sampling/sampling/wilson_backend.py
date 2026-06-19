from __future__ import annotations
from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Any, Callable, Optional, Protocol
import torch

from ..exceptions import NativeBackendUnavailableError
from ..native import load_wilson_extension
from .csr import CSRGraph
from .transitions import sample_next_node


@dataclass(slots=True)
class WilsonTrace:
    """Trace of one Wilson walk attachment."""

    start: int
    raw_walk: list[int]
    loop_erased_path: list[int]
    attached_edges: list[tuple[int, int]]
    hit_node: int


class WilsonForestBackend(Protocol):
    """Protocol for Wilson forest backends."""

    def sample_forest(
        self,
        root_nodes: torch.Tensor,
        *,
        generator: Optional[torch.Generator] = None,
        seed: Optional[int] = None,
        node_order: str = 'random',
        max_steps: Optional[int] = None,
        trace: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any], list[WilsonTrace]]:
        ...

    def sample_forest_with_q_factor(
        self,
        q: float,
        *,
        generator: Optional[torch.Generator] = None,
        seed: Optional[int] = None,
        node_order: str = 'random',
        max_steps: Optional[int] = None,
        trace: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any], list[WilsonTrace]]:
        ...


class _WilsonBackendBase:
    def __init__(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        *,
        edge_weight: Optional[torch.Tensor] = None,
        make_undirected: bool = False,
        drop_self_loops: bool = False,
        device: Optional[torch.device] = None,
    ) -> None:
        self.num_nodes = int(num_nodes)
        self.device = device or edge_index.device
        self.graph = CSRGraph.from_edge_index(
            edge_index=edge_index,
            num_nodes=self.num_nodes,
            edge_weight=edge_weight,
            make_undirected=make_undirected,
            drop_self_loops=drop_self_loops,
            device=self.device,
        )

    @staticmethod
    def _prepare_generator(device: torch.device, seed: Optional[int], generator: Optional[torch.Generator]) -> torch.Generator:
        if generator is None:
            generator = torch.Generator(device=device)
        if seed is not None:
            generator.manual_seed(int(seed))
        return generator

    def _iteration_order(self, node_order: str, generator: torch.Generator) -> torch.Tensor:
        if node_order == 'random':
            return torch.randperm(self.graph.num_nodes, generator=generator, device=self.device)
        if node_order == 'sequential':
            return torch.arange(self.graph.num_nodes, device=self.device)
        raise ValueError("node_order must be 'random' or 'sequential'.")

    @staticmethod
    def _trace(start: int, raw_walk: list[int], attached_edges: list[tuple[int, int]], hit_node: int) -> WilsonTrace:
        if attached_edges:
            loop_erased_path = [attached_edges[0][0]] + [dst for _, dst in attached_edges]
        else:
            loop_erased_path = [start]
        return WilsonTrace(
            start=int(start),
            raw_walk=raw_walk,
            loop_erased_path=loop_erased_path,
            attached_edges=attached_edges,
            hit_node=int(hit_node),
        )


class PythonWilsonForestBackend(_WilsonBackendBase):
    """Pure Python/PyTorch Wilson implementation.

    This backend exists in the repository and does not require any compiled
    extension. It is slower than the native backend on large graphs, but it is
    fully functional and suitable for correctness checks, CPU-only environments,
    and reproducible fallback behaviour.
    """

    def sample_forest(
        self,
        root_nodes: torch.Tensor,
        *,
        generator: Optional[torch.Generator] = None,
        seed: Optional[int] = None,
        node_order: str = 'random',
        max_steps: Optional[int] = None,
        trace: bool = False,
        transition_fn: Callable[[int, torch.Tensor, Optional[torch.Tensor], Optional[torch.Generator]], int] = sample_next_node,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any], list[WilsonTrace]]:
        """Sample a rooted spanning forest using loop-erased random walks."""

        generator = self._prepare_generator(self.device, seed, generator)
        root_nodes = root_nodes.to(device=self.device, dtype=torch.long).unique()
        if root_nodes.numel() == 0:
            raise ValueError('root_nodes must be non-empty.')

        order = self._iteration_order(node_order, generator)
        rowptr, col, weights_all = self.graph.rowptr, self.graph.col, self.graph.weights
        parent = torch.full((self.num_nodes,), -1, dtype=torch.long, device=self.device)
        in_tree = torch.zeros((self.num_nodes,), dtype=torch.bool, device=self.device)
        in_tree[root_nodes] = True

        stats: dict[str, Any] = {'walks': 0, 'total_steps': 0, 'total_erased': 0}
        traces: list[WilsonTrace] = []

        for start in order.tolist():
            if bool(in_tree[start].item()):
                continue
            stats['walks'] += 1
            path = [start]
            path_pos: dict[int, int] = {start: 0}
            successor_map: dict[int, int] = {}
            raw_walk = [start]
            current = start
            hit_node = -1
            steps_here = 0

            while not bool(in_tree[current].item()):
                if max_steps is not None and steps_here >= int(max_steps):
                    raise RuntimeError(f'Max steps reached for Wilson walk starting from {start}.')
                steps_here += 1
                stats['total_steps'] += 1

                left = int(rowptr[current].item())
                right = int(rowptr[current + 1].item())
                neighbours = col[left:right]
                weights = None if weights_all is None else weights_all[left:right]
                successor = transition_fn(current, neighbours, weights, generator)
                successor_map[current] = successor
                current = successor
                raw_walk.append(current)

                if bool(in_tree[current].item()):
                    hit_node = current
                    break

                if current in path_pos:
                    cut = path_pos[current]
                    removed = path[cut + 1 :]
                    stats['total_erased'] += len(removed)
                    for removed_node in removed:
                        path_pos.pop(removed_node, None)
                        successor_map.pop(removed_node, None)
                    path = path[: cut + 1]
                else:
                    path_pos[current] = len(path)
                    path.append(current)

            attached_edges: list[tuple[int, int]] = []
            node = start
            while not bool(in_tree[node].item()):
                parent_node = successor_map[node]
                parent[node] = parent_node
                in_tree[node] = True
                attached_edges.append((node, parent_node))
                node = parent_node

            if trace:
                traces.append(self._trace(start, raw_walk, attached_edges, hit_node))

        roots_mask = (parent == -1) & in_tree
        return parent, roots_mask, stats, traces

    def sample_forest_with_q_factor(
        self,
        q: float,
        *,
        generator: Optional[torch.Generator] = None,
        seed: Optional[int] = None,
        node_order: str = 'random',
        max_steps: Optional[int] = None,
        trace: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any], list[WilsonTrace]]:
        """Sample a q-forest using Wilson's algorithm with killing."""

        if q <= 0.0:
            raise ValueError('q must be positive.')
        generator = self._prepare_generator(self.device, seed, generator)
        order = self._iteration_order(node_order, generator)
        rowptr, col, weights_all = self.graph.rowptr, self.graph.col, self.graph.weights
        outgoing_mass = self.graph.outgoing_mass

        parent = torch.full((self.num_nodes,), -1, dtype=torch.long, device=self.device)
        in_tree = torch.zeros((self.num_nodes,), dtype=torch.bool, device=self.device)

        stats: dict[str, Any] = {
            'walks': 0,
            'total_steps': 0,
            'total_moves': 0,
            'total_erased': 0,
            'kills': 0,
            'tree_hits': 0,
        }
        traces: list[WilsonTrace] = []

        for start in order.tolist():
            if bool(in_tree[start].item()):
                continue

            stats['walks'] += 1
            path = [start]
            path_pos: dict[int, int] = {start: 0}
            successor_map: dict[int, int] = {}
            raw_walk = [start]
            current = start
            killed = False
            hit_node = -1
            steps_here = 0

            while not bool(in_tree[current].item()):
                if max_steps is not None and steps_here >= int(max_steps):
                    raise RuntimeError(f'Max steps reached for Wilson q-walk starting from {start}.')
                steps_here += 1
                stats['total_steps'] += 1

                total_mass = float(outgoing_mass[current].item())
                if total_mass <= 0.0:
                    in_tree[current] = True
                    parent[current] = -1
                    stats['kills'] += 1
                    killed = True
                    break

                p_kill = float(q) / (float(q) + total_mass)
                if float(torch.rand((), generator=generator, device=self.device).item()) < p_kill:
                    in_tree[current] = True
                    parent[current] = -1
                    stats['kills'] += 1
                    killed = True
                    break

                left = int(rowptr[current].item())
                right = int(rowptr[current + 1].item())
                neighbours = col[left:right]
                weights = None if weights_all is None else weights_all[left:right]
                successor = sample_next_node(current, neighbours, weights, generator)
                successor_map[current] = successor
                current = successor
                raw_walk.append(current)
                stats['total_moves'] += 1

                if bool(in_tree[current].item()):
                    hit_node = current
                    stats['tree_hits'] += 1
                    break

                if current in path_pos:
                    cut = path_pos[current]
                    removed = path[cut + 1 :]
                    stats['total_erased'] += len(removed)
                    for removed_node in removed:
                        path_pos.pop(removed_node, None)
                        successor_map.pop(removed_node, None)
                    path = path[: cut + 1]
                else:
                    path_pos[current] = len(path)
                    path.append(current)

            attached_edges: list[tuple[int, int]] = []
            node = start
            while not bool(in_tree[node].item()):
                parent_node = successor_map[node]
                parent[node] = parent_node
                in_tree[node] = True
                attached_edges.append((node, parent_node))
                node = parent_node

            if trace:
                traces.append(self._trace(start, raw_walk, attached_edges, -1 if killed else hit_node))

        roots_mask = (parent == -1) & in_tree
        stats['num_roots'] = int(roots_mask.sum().item())
        return parent, roots_mask, stats, traces


class NativeWilsonForestBackend(_WilsonBackendBase):
    """Wilson backend backed by the optional compiled extension."""

    def __init__(self, edge_index: torch.Tensor, num_nodes: int, **kwargs: Any) -> None:
        super().__init__(edge_index=edge_index, num_nodes=num_nodes, **kwargs)
        self.extension = load_wilson_extension()
        if self.extension is None:
            raise NativeBackendUnavailableError(
                'The native Wilson backend was requested but the compiled extension is unavailable.'
            )

    def _native_seed(self, generator: torch.Generator) -> int:
        return int(torch.randint(0, 2**63 - 1, (1,), generator=generator, device=self.device, dtype=torch.long).item())

    def sample_forest(
        self,
        root_nodes: torch.Tensor,
        *,
        generator: Optional[torch.Generator] = None,
        seed: Optional[int] = None,
        node_order: str = 'random',
        max_steps: Optional[int] = None,
        trace: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any], list[WilsonTrace]]:
        """Sample a rooted forest with the native extension.

        Trace mode is intentionally unsupported for the native backend.
        """

        if trace:
            raise NotImplementedError('Trace output is only supported by the Python Wilson backend.')
        generator = self._prepare_generator(self.device, seed, generator)
        order = self._iteration_order(node_order, generator).to(dtype=torch.long).contiguous()
        roots = root_nodes.to(device=self.device, dtype=torch.long).unique().contiguous()
        weights = self.graph.weights if self.graph.weights is not None else torch.empty(0, dtype=torch.float32, device=self.device)
        parent, roots_mask, stats_tensor = self.extension.sample_forest_native(
            self.graph.rowptr.contiguous(),
            self.graph.col.contiguous(),
            weights.to(torch.float32).contiguous(),
            roots,
            order,
            True,
            -1 if max_steps is None else int(max_steps),
            self._native_seed(generator),
        )
        stats_tensor = stats_tensor.to(device='cpu', dtype=torch.long)
        stats = {
            'walks': int(stats_tensor[0].item()),
            'total_steps': int(stats_tensor[1].item()),
            'total_erased': int(stats_tensor[2].item()),
            'num_roots': int(stats_tensor[3].item()),
        }
        return parent.to(self.device), roots_mask.to(self.device), stats, []

    def sample_forest_with_q_factor(
        self,
        q: float,
        *,
        generator: Optional[torch.Generator] = None,
        seed: Optional[int] = None,
        node_order: str = 'random',
        max_steps: Optional[int] = None,
        trace: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any], list[WilsonTrace]]:
        """Sample a q-forest with the native extension."""

        if trace:
            raise NotImplementedError('Trace output is only supported by the Python Wilson backend.')
        generator = self._prepare_generator(self.device, seed, generator)
        order = self._iteration_order(node_order, generator).to(dtype=torch.long).contiguous()
        weights = self.graph.weights if self.graph.weights is not None else torch.empty(0, dtype=torch.float32, device=self.device)
        parent, roots_mask, stats_tensor = self.extension.sample_forest_with_q_factor_native(
            self.graph.rowptr.contiguous(),
            self.graph.col.contiguous(),
            weights.to(torch.float32).contiguous(),
            self.graph.outgoing_mass.to(torch.float32).contiguous(),
            order,
            float(q),
            -1 if max_steps is None else int(max_steps),
            self._native_seed(generator),
        )
        stats_tensor = stats_tensor.to(device='cpu', dtype=torch.long)
        stats = {
            'walks': int(stats_tensor[0].item()),
            'total_steps': int(stats_tensor[1].item()),
            'total_moves': int(stats_tensor[2].item()),
            'total_erased': int(stats_tensor[3].item()),
            'kills': int(stats_tensor[4].item()),
            'tree_hits': int(stats_tensor[5].item()),
            'num_roots': int(stats_tensor[6].item()),
        }
        return parent.to(self.device), roots_mask.to(self.device), stats, []


class WilsonBackendFactory:
    """Factory for Wilson backends selected through configuration."""

    @staticmethod
    def create(
        backend_name: str,
        edge_index: torch.Tensor,
        num_nodes: int,
        *,
        edge_weight: Optional[torch.Tensor] = None,
        make_undirected: bool = False,
        drop_self_loops: bool = False,
        device: Optional[torch.device] = None,
    ) -> WilsonForestBackend:
        """Instantiate a Wilson backend from configuration.

        Args:
            backend_name: One of auto, native, or python.
            edge_index: Graph edges in COO format.
            num_nodes: Number of nodes.
            edge_weight: Optional edge weights.
            make_undirected: Whether to symmetrize the graph before Wilson.
            drop_self_loops: Whether to remove self-loops before Wilson.
            device: Target torch device.
        """

        common_kwargs = {
            'edge_weight': edge_weight,
            'make_undirected': make_undirected,
            'drop_self_loops': drop_self_loops,
            'device': device,
        }
        if backend_name == 'python':
            return PythonWilsonForestBackend(edge_index=edge_index, num_nodes=num_nodes, **common_kwargs)
        if backend_name == 'native':
            return NativeWilsonForestBackend(edge_index=edge_index, num_nodes=num_nodes, **common_kwargs)
        if backend_name == 'auto':
            if load_wilson_extension() is not None:
                return NativeWilsonForestBackend(edge_index=edge_index, num_nodes=num_nodes, **common_kwargs)
            return PythonWilsonForestBackend(edge_index=edge_index, num_nodes=num_nodes, **common_kwargs)
        raise ValueError(f'Unsupported Wilson backend: {backend_name}')
