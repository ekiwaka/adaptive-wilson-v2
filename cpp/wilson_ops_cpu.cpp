#include <torch/extension.h>
#include <algorithm>
#include <cstdint>
#include <random>
#include <vector>

namespace {

bool has_weights(const torch::Tensor& weights) {
    return weights.defined() && weights.numel() > 0;
}

int64_t sample_successor_cpu(
    int64_t current,
    const int64_t* rowptr,
    const int64_t* col,
    const float* weights,
    std::mt19937_64& rng
) {
    const int64_t left = rowptr[current];
    const int64_t right = rowptr[current + 1];
    const int64_t degree = right - left;
    if (degree <= 0) {
        return -1;
    }

    if (weights == nullptr) {
        std::uniform_int_distribution<int64_t> pick(0, degree - 1);
        return col[left + pick(rng)];
    }

    float total = 0.0f;
    for (int64_t edge_idx = left; edge_idx < right; ++edge_idx) {
        total += weights[edge_idx];
    }
    if (!(total > 0.0f)) {
        return -1;
    }

    std::uniform_real_distribution<float> draw_dist(0.0f, total);
    const float draw = draw_dist(rng);
    float accum = 0.0f;
    for (int64_t edge_idx = left; edge_idx < right; ++edge_idx) {
        accum += weights[edge_idx];
        if (draw <= accum || edge_idx == right - 1) {
            return col[edge_idx];
        }
    }
    return col[right - 1];
}

}  // namespace

std::vector<torch::Tensor> sample_forest_cpu(
    torch::Tensor rowptr,
    torch::Tensor col,
    torch::Tensor weights,
    torch::Tensor root_nodes,
    torch::Tensor order,
    bool treat_isolated_as_root,
    int64_t max_steps,
    uint64_t seed
) {
    TORCH_CHECK(!rowptr.is_cuda(), "sample_forest_cpu expects CPU tensors");
    rowptr = rowptr.contiguous();
    col = col.contiguous();
    root_nodes = root_nodes.contiguous();
    order = order.contiguous();
    if (weights.defined()) {
        weights = weights.contiguous();
        TORCH_CHECK(weights.scalar_type() == torch::kFloat32, "weights must be float32");
    }

    const int64_t num_nodes = rowptr.size(0) - 1;
    auto parent = torch::full({num_nodes}, -1, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
    auto roots_mask = torch::zeros({num_nodes}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU));
    auto stats = torch::zeros({4}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));

    auto* rowptr_ptr = rowptr.data_ptr<int64_t>();
    auto* col_ptr = col.data_ptr<int64_t>();
    const float* weights_ptr = has_weights(weights) ? weights.data_ptr<float>() : nullptr;
    auto* parent_ptr = parent.data_ptr<int64_t>();
    auto* roots_ptr = roots_mask.data_ptr<bool>();
    auto* stats_ptr = stats.data_ptr<int64_t>();

    std::vector<uint8_t> in_tree(static_cast<size_t>(num_nodes), 0);
    std::vector<int64_t> visit_epoch(static_cast<size_t>(num_nodes), 0);
    std::vector<int64_t> path_index(static_cast<size_t>(num_nodes), -1);
    std::vector<int64_t> next_node(static_cast<size_t>(num_nodes), -1);
    std::vector<int64_t> path_nodes(static_cast<size_t>(num_nodes), -1);

    auto* root_ptr = root_nodes.data_ptr<int64_t>();
    for (int64_t i = 0; i < root_nodes.numel(); ++i) {
        const int64_t node = root_ptr[i];
        TORCH_CHECK(node >= 0 && node < num_nodes, "root node is out of range");
        in_tree[static_cast<size_t>(node)] = 1;
    }

    auto* order_ptr = order.data_ptr<int64_t>();
    std::mt19937_64 rng(seed);
    int64_t walks = 0;
    int64_t total_steps = 0;
    int64_t total_erased = 0;
    int64_t epoch = 0;

    for (int64_t order_idx = 0; order_idx < order.numel(); ++order_idx) {
        const int64_t start = order_ptr[order_idx];
        TORCH_CHECK(start >= 0 && start < num_nodes, "order contains out-of-range index");
        if (in_tree[static_cast<size_t>(start)] != 0) {
            continue;
        }

        const int64_t left0 = rowptr_ptr[start];
        const int64_t right0 = rowptr_ptr[start + 1];
        if (left0 == right0) {
            if (treat_isolated_as_root) {
                in_tree[static_cast<size_t>(start)] = 1;
                parent_ptr[start] = -1;
                continue;
            }
            TORCH_CHECK(false, "Isolated node encountered in sample_forest_cpu");
        }

        ++walks;
        ++epoch;
        int64_t current = start;
        int64_t steps_here = 0;
        int64_t path_len = 1;
        path_nodes[0] = start;
        visit_epoch[static_cast<size_t>(start)] = epoch;
        path_index[static_cast<size_t>(start)] = 0;

        while (in_tree[static_cast<size_t>(current)] == 0) {
            if (max_steps >= 0 && steps_here >= max_steps) {
                TORCH_CHECK(false, "Max steps reached in sample_forest_cpu");
            }
            ++steps_here;
            ++total_steps;

            const int64_t successor = sample_successor_cpu(current, rowptr_ptr, col_ptr, weights_ptr, rng);
            if (successor < 0) {
                if (treat_isolated_as_root) {
                    in_tree[static_cast<size_t>(current)] = 1;
                    parent_ptr[current] = -1;
                    break;
                }
                TORCH_CHECK(false, "Dead-end encountered in sample_forest_cpu");
            }

            next_node[static_cast<size_t>(current)] = successor;
            current = successor;

            if (in_tree[static_cast<size_t>(current)] != 0) {
                break;
            }

            if (visit_epoch[static_cast<size_t>(current)] == epoch) {
                const int64_t cut = path_index[static_cast<size_t>(current)];
                total_erased += path_len - (cut + 1);
                while (path_len > (cut + 1)) {
                    const int64_t removed_node = path_nodes[static_cast<size_t>(path_len - 1)];
                    visit_epoch[static_cast<size_t>(removed_node)] = 0;
                    path_index[static_cast<size_t>(removed_node)] = -1;
                    next_node[static_cast<size_t>(removed_node)] = -1;
                    --path_len;
                }
            } else {
                path_index[static_cast<size_t>(current)] = path_len;
                visit_epoch[static_cast<size_t>(current)] = epoch;
                path_nodes[static_cast<size_t>(path_len)] = current;
                ++path_len;
            }
        }

        int64_t node = start;
        while (in_tree[static_cast<size_t>(node)] == 0) {
            const int64_t parent_node = next_node[static_cast<size_t>(node)];
            parent_ptr[node] = parent_node;
            in_tree[static_cast<size_t>(node)] = 1;
            node = parent_node;
        }
    }

    int64_t num_roots = 0;
    for (int64_t i = 0; i < num_nodes; ++i) {
        const bool is_root = parent_ptr[i] == -1 && in_tree[static_cast<size_t>(i)] != 0;
        roots_ptr[i] = is_root;
        num_roots += static_cast<int64_t>(is_root);
    }

    stats_ptr[0] = walks;
    stats_ptr[1] = total_steps;
    stats_ptr[2] = total_erased;
    stats_ptr[3] = num_roots;
    return {parent, roots_mask, stats};
}

std::vector<torch::Tensor> sample_forest_with_q_factor_cpu(
    torch::Tensor rowptr,
    torch::Tensor col,
    torch::Tensor weights,
    torch::Tensor outgoing_mass,
    torch::Tensor order,
    double q,
    int64_t max_steps,
    uint64_t seed
) {
    TORCH_CHECK(!rowptr.is_cuda(), "sample_forest_with_q_factor_cpu expects CPU tensors");
    TORCH_CHECK(q > 0.0, "q must be positive");
    rowptr = rowptr.contiguous();
    col = col.contiguous();
    outgoing_mass = outgoing_mass.contiguous();
    order = order.contiguous();
    if (weights.defined()) {
        weights = weights.contiguous();
        TORCH_CHECK(weights.scalar_type() == torch::kFloat32, "weights must be float32");
    }

    const int64_t num_nodes = rowptr.size(0) - 1;
    auto parent = torch::full({num_nodes}, -1, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
    auto roots_mask = torch::zeros({num_nodes}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU));
    auto stats = torch::zeros({7}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));

    auto* rowptr_ptr = rowptr.data_ptr<int64_t>();
    auto* col_ptr = col.data_ptr<int64_t>();
    const float* weights_ptr = has_weights(weights) ? weights.data_ptr<float>() : nullptr;
    auto* outgoing_ptr = outgoing_mass.data_ptr<float>();
    auto* parent_ptr = parent.data_ptr<int64_t>();
    auto* roots_ptr = roots_mask.data_ptr<bool>();
    auto* stats_ptr = stats.data_ptr<int64_t>();
    auto* order_ptr = order.data_ptr<int64_t>();

    std::vector<uint8_t> in_tree(static_cast<size_t>(num_nodes), 0);
    std::vector<int64_t> visit_epoch(static_cast<size_t>(num_nodes), 0);
    std::vector<int64_t> path_index(static_cast<size_t>(num_nodes), -1);
    std::vector<int64_t> next_node(static_cast<size_t>(num_nodes), -1);
    std::vector<int64_t> path_nodes(static_cast<size_t>(num_nodes), -1);

    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    int64_t walks = 0;
    int64_t total_steps = 0;
    int64_t total_moves = 0;
    int64_t total_erased = 0;
    int64_t kills = 0;
    int64_t tree_hits = 0;
    int64_t epoch = 0;

    for (int64_t order_idx = 0; order_idx < order.numel(); ++order_idx) {
        const int64_t start = order_ptr[order_idx];
        TORCH_CHECK(start >= 0 && start < num_nodes, "order contains out-of-range index");
        if (in_tree[static_cast<size_t>(start)] != 0) {
            continue;
        }

        ++walks;
        ++epoch;
        int64_t current = start;
        int64_t steps_here = 0;
        int64_t path_len = 1;
        path_nodes[0] = start;
        visit_epoch[static_cast<size_t>(start)] = epoch;
        path_index[static_cast<size_t>(start)] = 0;

        while (in_tree[static_cast<size_t>(current)] == 0) {
            if (max_steps >= 0 && steps_here >= max_steps) {
                TORCH_CHECK(false, "Max steps reached in sample_forest_with_q_factor_cpu");
            }
            ++steps_here;
            ++total_steps;

            const double total_mass = static_cast<double>(outgoing_ptr[current]);
            if (total_mass <= 0.0) {
                in_tree[static_cast<size_t>(current)] = 1;
                parent_ptr[current] = -1;
                ++kills;
                break;
            }

            const double p_kill = q / (q + total_mass);
            if (unit(rng) < p_kill) {
                in_tree[static_cast<size_t>(current)] = 1;
                parent_ptr[current] = -1;
                ++kills;
                break;
            }

            const int64_t successor = sample_successor_cpu(current, rowptr_ptr, col_ptr, weights_ptr, rng);
            if (successor < 0) {
                in_tree[static_cast<size_t>(current)] = 1;
                parent_ptr[current] = -1;
                ++kills;
                break;
            }

            next_node[static_cast<size_t>(current)] = successor;
            current = successor;
            ++total_moves;

            if (in_tree[static_cast<size_t>(current)] != 0) {
                ++tree_hits;
                break;
            }

            if (visit_epoch[static_cast<size_t>(current)] == epoch) {
                const int64_t cut = path_index[static_cast<size_t>(current)];
                total_erased += path_len - (cut + 1);
                while (path_len > (cut + 1)) {
                    const int64_t removed_node = path_nodes[static_cast<size_t>(path_len - 1)];
                    visit_epoch[static_cast<size_t>(removed_node)] = 0;
                    path_index[static_cast<size_t>(removed_node)] = -1;
                    next_node[static_cast<size_t>(removed_node)] = -1;
                    --path_len;
                }
            } else {
                path_index[static_cast<size_t>(current)] = path_len;
                visit_epoch[static_cast<size_t>(current)] = epoch;
                path_nodes[static_cast<size_t>(path_len)] = current;
                ++path_len;
            }
        }

        int64_t node = start;
        while (in_tree[static_cast<size_t>(node)] == 0) {
            const int64_t parent_node = next_node[static_cast<size_t>(node)];
            parent_ptr[node] = parent_node;
            in_tree[static_cast<size_t>(node)] = 1;
            node = parent_node;
        }
    }

    int64_t num_roots = 0;
    for (int64_t i = 0; i < num_nodes; ++i) {
        const bool is_root = parent_ptr[i] == -1 && in_tree[static_cast<size_t>(i)] != 0;
        roots_ptr[i] = is_root;
        num_roots += static_cast<int64_t>(is_root);
    }

    stats_ptr[0] = walks;
    stats_ptr[1] = total_steps;
    stats_ptr[2] = total_moves;
    stats_ptr[3] = total_erased;
    stats_ptr[4] = kills;
    stats_ptr[5] = tree_hits;
    stats_ptr[6] = num_roots;
    return {parent, roots_mask, stats};
}
