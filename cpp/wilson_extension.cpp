#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> sample_forest_cpu(
    torch::Tensor rowptr,
    torch::Tensor col,
    torch::Tensor weights,
    torch::Tensor root_nodes,
    torch::Tensor order,
    bool treat_isolated_as_root,
    int64_t max_steps,
    uint64_t seed
);

std::vector<torch::Tensor> sample_forest_with_q_factor_cpu(
    torch::Tensor rowptr,
    torch::Tensor col,
    torch::Tensor weights,
    torch::Tensor outgoing_mass,
    torch::Tensor order,
    double q,
    int64_t max_steps,
    uint64_t seed
);

#ifdef WITH_CUDA
std::vector<torch::Tensor> sample_forest_cuda(
    torch::Tensor rowptr,
    torch::Tensor col,
    torch::Tensor weights,
    torch::Tensor root_nodes,
    torch::Tensor order,
    bool treat_isolated_as_root,
    int64_t max_steps,
    uint64_t seed
);

std::vector<torch::Tensor> sample_forest_with_q_factor_cuda(
    torch::Tensor rowptr,
    torch::Tensor col,
    torch::Tensor weights,
    torch::Tensor outgoing_mass,
    torch::Tensor order,
    double q,
    int64_t max_steps,
    uint64_t seed
);
#endif

std::vector<torch::Tensor> sample_forest_native(
    torch::Tensor rowptr,
    torch::Tensor col,
    torch::Tensor weights,
    torch::Tensor root_nodes,
    torch::Tensor order,
    bool treat_isolated_as_root,
    int64_t max_steps,
    uint64_t seed
) {
    TORCH_CHECK(rowptr.scalar_type() == torch::kInt64, "rowptr must be int64");
    TORCH_CHECK(col.scalar_type() == torch::kInt64, "col must be int64");
    TORCH_CHECK(root_nodes.scalar_type() == torch::kInt64, "root_nodes must be int64");
    TORCH_CHECK(order.scalar_type() == torch::kInt64, "order must be int64");
    TORCH_CHECK(rowptr.device() == col.device(), "rowptr and col must be on the same device");
    TORCH_CHECK(root_nodes.device() == rowptr.device(), "root_nodes must be on the same device as rowptr");
    TORCH_CHECK(order.device() == rowptr.device(), "order must be on the same device as rowptr");
    TORCH_CHECK(!weights.defined() || weights.device() == rowptr.device(), "weights must be on the same device as rowptr");

    if (rowptr.is_cuda()) {
#ifdef WITH_CUDA
        return sample_forest_cuda(rowptr, col, weights, root_nodes, order, treat_isolated_as_root, max_steps, seed);
#else
        TORCH_CHECK(false, "Wilson extension was built without CUDA support.");
#endif
    }
    return sample_forest_cpu(rowptr, col, weights, root_nodes, order, treat_isolated_as_root, max_steps, seed);
}

std::vector<torch::Tensor> sample_forest_with_q_factor_native(
    torch::Tensor rowptr,
    torch::Tensor col,
    torch::Tensor weights,
    torch::Tensor outgoing_mass,
    torch::Tensor order,
    double q,
    int64_t max_steps,
    uint64_t seed
) {
    TORCH_CHECK(rowptr.scalar_type() == torch::kInt64, "rowptr must be int64");
    TORCH_CHECK(col.scalar_type() == torch::kInt64, "col must be int64");
    TORCH_CHECK(order.scalar_type() == torch::kInt64, "order must be int64");
    TORCH_CHECK(outgoing_mass.scalar_type() == torch::kFloat32, "outgoing_mass must be float32");
    TORCH_CHECK(rowptr.device() == col.device(), "rowptr and col must be on the same device");
    TORCH_CHECK(order.device() == rowptr.device(), "order must be on the same device as rowptr");
    TORCH_CHECK(outgoing_mass.device() == rowptr.device(), "outgoing_mass must be on the same device as rowptr");
    TORCH_CHECK(!weights.defined() || weights.device() == rowptr.device(), "weights must be on the same device as rowptr");

    if (rowptr.is_cuda()) {
#ifdef WITH_CUDA
        return sample_forest_with_q_factor_cuda(rowptr, col, weights, outgoing_mass, order, q, max_steps, seed);
#else
        TORCH_CHECK(false, "Wilson extension was built without CUDA support.");
#endif
    }
    return sample_forest_with_q_factor_cpu(rowptr, col, weights, outgoing_mass, order, q, max_steps, seed);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sample_forest_native", &sample_forest_native, "Native Wilson forest sampler");
    m.def(
        "sample_forest_with_q_factor_native",
        &sample_forest_with_q_factor_native,
        "Native q-factor Wilson forest sampler"
    );
}
