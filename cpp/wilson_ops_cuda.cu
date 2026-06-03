#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstdint>
#include <vector>

namespace {

__device__ int64_t sample_successor_cuda(
    int64_t current,
    const int64_t* rowptr,
    const int64_t* col,
    const float* weights,
    curandStatePhilox4_32_10_t* state
) {
    const int64_t left = rowptr[current];
    const int64_t right = rowptr[current + 1];
    const int64_t degree = right - left;
    if (degree <= 0) {
        return -1;
    }

    if (weights == nullptr) {
        const double draw = curand_uniform_double(state);
        int64_t offset = static_cast<int64_t>(draw * static_cast<double>(degree));
        if (offset >= degree) {
            offset = degree - 1;
        }
        return col[left + offset];
    }

    float total = 0.0f;
    for (int64_t edge_idx = left; edge_idx < right; ++edge_idx) {
        total += weights[edge_idx];
    }
    if (!(total > 0.0f)) {
        return -1;
    }

    const float draw = curand_uniform(state) * total;
    float accum = 0.0f;
    for (int64_t edge_idx = left; edge_idx < right; ++edge_idx) {
        accum += weights[edge_idx];
        if (draw <= accum || edge_idx == right - 1) {
            return col[edge_idx];
        }
    }
    return col[right - 1];
}

__global__ void sample_forest_kernel(
    const int64_t* rowptr,
    const int64_t* col,
    const float* weights,
    int64_t num_nodes,
    const int64_t* root_nodes,
    int64_t num_roots_in,
    const int64_t* order,
    bool treat_isolated_as_root,
    int64_t max_steps,
    uint64_t seed,
    int64_t* parent_out,
    bool* roots_mask_out,
    int64_t* stats_out,
    int64_t* status_out,
    uint8_t* in_tree,
    int64_t* visit_epoch,
    int64_t* path_index,
    int64_t* next_node,
    int64_t* path_nodes
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    curandStatePhilox4_32_10_t state;
    curand_init(seed, 0, 0, &state);

    for (int64_t i = 0; i < num_roots_in; ++i) {
        const int64_t node = root_nodes[i];
        if (node < 0 || node >= num_nodes) {
            status_out[0] = 1;
            return;
        }
        in_tree[node] = 1;
    }

    int64_t walks = 0;
    int64_t total_steps = 0;
    int64_t total_erased = 0;
    int64_t epoch = 0;

    for (int64_t order_idx = 0; order_idx < num_nodes; ++order_idx) {
        const int64_t start = order[order_idx];
        if (start < 0 || start >= num_nodes) {
            status_out[0] = 2;
            return;
        }
        if (in_tree[start] != 0) {
            continue;
        }

        const int64_t left0 = rowptr[start];
        const int64_t right0 = rowptr[start + 1];
        if (left0 == right0) {
            if (treat_isolated_as_root) {
                in_tree[start] = 1;
                parent_out[start] = -1;
                continue;
            }
            status_out[0] = 3;
            return;
        }

        ++walks;
        ++epoch;
        int64_t current = start;
        int64_t steps_here = 0;
        int64_t path_len = 1;
        path_nodes[0] = start;
        visit_epoch[start] = epoch;
        path_index[start] = 0;

        while (in_tree[current] == 0) {
            if (max_steps >= 0 && steps_here >= max_steps) {
                status_out[0] = 4;
                return;
            }
            ++steps_here;
            ++total_steps;

            const int64_t successor = sample_successor_cuda(current, rowptr, col, weights, &state);
            if (successor < 0) {
                if (treat_isolated_as_root) {
                    in_tree[current] = 1;
                    parent_out[current] = -1;
                    break;
                }
                status_out[0] = 5;
                return;
            }

            next_node[current] = successor;
            current = successor;

            if (in_tree[current] != 0) {
                break;
            }

            if (visit_epoch[current] == epoch) {
                const int64_t cut = path_index[current];
                total_erased += path_len - (cut + 1);
                while (path_len > (cut + 1)) {
                    const int64_t removed_node = path_nodes[path_len - 1];
                    visit_epoch[removed_node] = 0;
                    path_index[removed_node] = -1;
                    next_node[removed_node] = -1;
                    --path_len;
                }
            } else {
                path_index[current] = path_len;
                visit_epoch[current] = epoch;
                path_nodes[path_len] = current;
                ++path_len;
            }
        }

        int64_t node = start;
        while (in_tree[node] == 0) {
            const int64_t parent_node = next_node[node];
            parent_out[node] = parent_node;
            in_tree[node] = 1;
            node = parent_node;
        }
    }

    int64_t num_roots = 0;
    for (int64_t i = 0; i < num_nodes; ++i) {
        const bool is_root = parent_out[i] == -1 && in_tree[i] != 0;
        roots_mask_out[i] = is_root;
        num_roots += static_cast<int64_t>(is_root);
    }

    stats_out[0] = walks;
    stats_out[1] = total_steps;
    stats_out[2] = total_erased;
    stats_out[3] = num_roots;
}

__global__ void sample_forest_with_q_factor_kernel(
    const int64_t* rowptr,
    const int64_t* col,
    const float* weights,
    const float* outgoing_mass,
    int64_t num_nodes,
    const int64_t* order,
    double q,
    int64_t max_steps,
    uint64_t seed,
    int64_t* parent_out,
    bool* roots_mask_out,
    int64_t* stats_out,
    int64_t* status_out,
    uint8_t* in_tree,
    int64_t* visit_epoch,
    int64_t* path_index,
    int64_t* next_node,
    int64_t* path_nodes
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    curandStatePhilox4_32_10_t state;
    curand_init(seed, 0, 0, &state);

    int64_t walks = 0;
    int64_t total_steps = 0;
    int64_t total_moves = 0;
    int64_t total_erased = 0;
    int64_t kills = 0;
    int64_t tree_hits = 0;
    int64_t epoch = 0;

    for (int64_t order_idx = 0; order_idx < num_nodes; ++order_idx) {
        const int64_t start = order[order_idx];
        if (start < 0 || start >= num_nodes) {
            status_out[0] = 10;
            return;
        }
        if (in_tree[start] != 0) {
            continue;
        }

        ++walks;
        ++epoch;
        int64_t current = start;
        int64_t steps_here = 0;
        int64_t path_len = 1;
        path_nodes[0] = start;
        visit_epoch[start] = epoch;
        path_index[start] = 0;

        while (in_tree[current] == 0) {
            if (max_steps >= 0 && steps_here >= max_steps) {
                status_out[0] = 11;
                return;
            }
            ++steps_here;
            ++total_steps;

            const double total_mass = static_cast<double>(outgoing_mass[current]);
            if (total_mass <= 0.0) {
                in_tree[current] = 1;
                parent_out[current] = -1;
                ++kills;
                break;
            }

            const double p_kill = q / (q + total_mass);
            if (curand_uniform_double(&state) < p_kill) {
                in_tree[current] = 1;
                parent_out[current] = -1;
                ++kills;
                break;
            }

            const int64_t successor = sample_successor_cuda(current, rowptr, col, weights, &state);
            if (successor < 0) {
                in_tree[current] = 1;
                parent_out[current] = -1;
                ++kills;
                break;
            }

            next_node[current] = successor;
            current = successor;
            ++total_moves;

            if (in_tree[current] != 0) {
                ++tree_hits;
                break;
            }

            if (visit_epoch[current] == epoch) {
                const int64_t cut = path_index[current];
                total_erased += path_len - (cut + 1);
                while (path_len > (cut + 1)) {
                    const int64_t removed_node = path_nodes[path_len - 1];
                    visit_epoch[removed_node] = 0;
                    path_index[removed_node] = -1;
                    next_node[removed_node] = -1;
                    --path_len;
                }
            } else {
                path_index[current] = path_len;
                visit_epoch[current] = epoch;
                path_nodes[path_len] = current;
                ++path_len;
            }
        }

        int64_t node = start;
        while (in_tree[node] == 0) {
            const int64_t parent_node = next_node[node];
            parent_out[node] = parent_node;
            in_tree[node] = 1;
            node = parent_node;
        }
    }

    int64_t num_roots = 0;
    for (int64_t i = 0; i < num_nodes; ++i) {
        const bool is_root = parent_out[i] == -1 && in_tree[i] != 0;
        roots_mask_out[i] = is_root;
        num_roots += static_cast<int64_t>(is_root);
    }

    stats_out[0] = walks;
    stats_out[1] = total_steps;
    stats_out[2] = total_moves;
    stats_out[3] = total_erased;
    stats_out[4] = kills;
    stats_out[5] = tree_hits;
    stats_out[6] = num_roots;
}

void check_status(torch::Tensor status, const char* fn_name) {
    AT_CUDA_CHECK(cudaGetLastError());
    AT_CUDA_CHECK(cudaDeviceSynchronize());
    const int64_t code = status.cpu().item<int64_t>();
    TORCH_CHECK(code == 0, fn_name, " failed on CUDA with status code ", code);
}

}  // namespace

std::vector<torch::Tensor> sample_forest_cuda(
    torch::Tensor rowptr,
    torch::Tensor col,
    torch::Tensor weights,
    torch::Tensor root_nodes,
    torch::Tensor order,
    bool treat_isolated_as_root,
    int64_t max_steps,
    uint64_t seed
) {
    TORCH_CHECK(rowptr.is_cuda(), "sample_forest_cuda expects CUDA tensors");
    rowptr = rowptr.contiguous();
    col = col.contiguous();
    root_nodes = root_nodes.contiguous();
    order = order.contiguous();
    if (weights.defined() && weights.numel() > 0) {
        weights = weights.contiguous();
        TORCH_CHECK(weights.scalar_type() == torch::kFloat32, "weights must be float32");
    }

    const auto options_long = torch::TensorOptions().dtype(torch::kInt64).device(rowptr.device());
    const auto options_bool = torch::TensorOptions().dtype(torch::kBool).device(rowptr.device());
    const auto options_u8 = torch::TensorOptions().dtype(torch::kUInt8).device(rowptr.device());
    const int64_t num_nodes = rowptr.size(0) - 1;

    auto parent = torch::full({num_nodes}, -1, options_long);
    auto roots_mask = torch::zeros({num_nodes}, options_bool);
    auto stats = torch::zeros({4}, options_long);
    auto status = torch::zeros({1}, options_long);
    auto in_tree = torch::zeros({num_nodes}, options_u8);
    auto visit_epoch = torch::zeros({num_nodes}, options_long);
    auto path_index = torch::full({num_nodes}, -1, options_long);
    auto next_node = torch::full({num_nodes}, -1, options_long);
    auto path_nodes = torch::full({num_nodes}, -1, options_long);

    const float* weights_ptr = (weights.defined() && weights.numel() > 0) ? weights.data_ptr<float>() : nullptr;
    sample_forest_kernel<<<1, 1, 0, at::cuda::getDefaultCUDAStream()>>>(
        rowptr.data_ptr<int64_t>(),
        col.data_ptr<int64_t>(),
        weights_ptr,
        num_nodes,
        root_nodes.data_ptr<int64_t>(),
        root_nodes.numel(),
        order.data_ptr<int64_t>(),
        treat_isolated_as_root,
        max_steps,
        seed,
        parent.data_ptr<int64_t>(),
        roots_mask.data_ptr<bool>(),
        stats.data_ptr<int64_t>(),
        status.data_ptr<int64_t>(),
        in_tree.data_ptr<uint8_t>(),
        visit_epoch.data_ptr<int64_t>(),
        path_index.data_ptr<int64_t>(),
        next_node.data_ptr<int64_t>(),
        path_nodes.data_ptr<int64_t>()
    );
    check_status(status, "sample_forest_cuda");
    return {parent, roots_mask, stats};
}

std::vector<torch::Tensor> sample_forest_with_q_factor_cuda(
    torch::Tensor rowptr,
    torch::Tensor col,
    torch::Tensor weights,
    torch::Tensor outgoing_mass,
    torch::Tensor order,
    double q,
    int64_t max_steps,
    uint64_t seed
) {
    TORCH_CHECK(rowptr.is_cuda(), "sample_forest_with_q_factor_cuda expects CUDA tensors");
    TORCH_CHECK(q > 0.0, "q must be positive");
    rowptr = rowptr.contiguous();
    col = col.contiguous();
    outgoing_mass = outgoing_mass.contiguous();
    order = order.contiguous();
    if (weights.defined() && weights.numel() > 0) {
        weights = weights.contiguous();
        TORCH_CHECK(weights.scalar_type() == torch::kFloat32, "weights must be float32");
    }

    const auto options_long = torch::TensorOptions().dtype(torch::kInt64).device(rowptr.device());
    const auto options_bool = torch::TensorOptions().dtype(torch::kBool).device(rowptr.device());
    const auto options_u8 = torch::TensorOptions().dtype(torch::kUInt8).device(rowptr.device());
    const int64_t num_nodes = rowptr.size(0) - 1;

    auto parent = torch::full({num_nodes}, -1, options_long);
    auto roots_mask = torch::zeros({num_nodes}, options_bool);
    auto stats = torch::zeros({7}, options_long);
    auto status = torch::zeros({1}, options_long);
    auto in_tree = torch::zeros({num_nodes}, options_u8);
    auto visit_epoch = torch::zeros({num_nodes}, options_long);
    auto path_index = torch::full({num_nodes}, -1, options_long);
    auto next_node = torch::full({num_nodes}, -1, options_long);
    auto path_nodes = torch::full({num_nodes}, -1, options_long);

    const float* weights_ptr = (weights.defined() && weights.numel() > 0) ? weights.data_ptr<float>() : nullptr;
    sample_forest_with_q_factor_kernel<<<1, 1, 0, at::cuda::getDefaultCUDAStream()>>>(
        rowptr.data_ptr<int64_t>(),
        col.data_ptr<int64_t>(),
        weights_ptr,
        outgoing_mass.data_ptr<float>(),
        num_nodes,
        order.data_ptr<int64_t>(),
        q,
        max_steps,
        seed,
        parent.data_ptr<int64_t>(),
        roots_mask.data_ptr<bool>(),
        stats.data_ptr<int64_t>(),
        status.data_ptr<int64_t>(),
        in_tree.data_ptr<uint8_t>(),
        visit_epoch.data_ptr<int64_t>(),
        path_index.data_ptr<int64_t>(),
        next_node.data_ptr<int64_t>(),
        path_nodes.data_ptr<int64_t>()
    );
    check_status(status, "sample_forest_with_q_factor_cuda");
    return {parent, roots_mask, stats};
}
