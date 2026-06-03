#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" {

struct WilsonForestStats {
    int64_t walks;
    int64_t total_steps;
    int64_t total_erased;
    int64_t num_roots;
};

struct WilsonQForestStats {
    int64_t walks;
    int64_t total_steps;
    int64_t total_moves;
    int64_t total_erased;
    int64_t kills;
    int64_t tree_hits;
    int64_t num_roots;
};

namespace {

void write_error(const std::string& message, char* error_buf, size_t error_buf_len) {
    if (error_buf == nullptr || error_buf_len == 0) {
        return;
    }
    std::snprintf(error_buf, error_buf_len, "%s", message.c_str());
}

int fail(const std::string& message, char* error_buf, size_t error_buf_len) {
    write_error(message, error_buf, error_buf_len);
    return 1;
}

}  // namespace

int sample_forest_unweighted_cpu(
    const int64_t* rowptr,
    const int64_t* col,
    int64_t num_nodes,
    const int64_t* root_nodes,
    int64_t num_roots_in,
    const int64_t* order,
    bool treat_isolated_as_root,
    int64_t max_steps,
    uint64_t seed,
    int64_t* parent_out,
    uint8_t* roots_mask_out,
    WilsonForestStats* stats_out,
    char* error_buf,
    size_t error_buf_len
) {
    try {
        if (rowptr == nullptr || col == nullptr || root_nodes == nullptr || order == nullptr || parent_out == nullptr || roots_mask_out == nullptr || stats_out == nullptr) {
            return fail("sample_forest_unweighted_cpu received a null pointer.", error_buf, error_buf_len);
        }
        if (num_nodes < 0) {
            return fail("num_nodes must be non-negative.", error_buf, error_buf_len);
        }

        std::fill(parent_out, parent_out + num_nodes, static_cast<int64_t>(-1));
        std::fill(roots_mask_out, roots_mask_out + num_nodes, static_cast<uint8_t>(0));
        std::vector<uint8_t> in_tree(static_cast<size_t>(num_nodes), 0);
        for (int64_t i = 0; i < num_roots_in; ++i) {
            const int64_t root = root_nodes[i];
            if (root < 0 || root >= num_nodes) {
                return fail("root_nodes contains an out-of-range index.", error_buf, error_buf_len);
            }
            in_tree[static_cast<size_t>(root)] = 1;
        }

        std::vector<int64_t> visit_epoch(static_cast<size_t>(num_nodes), 0);
        std::vector<int64_t> path_index(static_cast<size_t>(num_nodes), -1);
        std::vector<int64_t> next_node(static_cast<size_t>(num_nodes), -1);
        std::vector<int64_t> path_nodes(static_cast<size_t>(num_nodes), -1);

        std::mt19937_64 rng(seed);
        int64_t walks = 0;
        int64_t total_steps = 0;
        int64_t total_erased = 0;
        int64_t epoch = 0;

        for (int64_t order_idx = 0; order_idx < num_nodes; ++order_idx) {
            const int64_t start = order[order_idx];
            if (start < 0 || start >= num_nodes) {
                return fail("order contains an out-of-range index.", error_buf, error_buf_len);
            }
            if (in_tree[static_cast<size_t>(start)] != 0) {
                continue;
            }

            const int64_t a = rowptr[start];
            const int64_t b = rowptr[start + 1];
            if (a == b) {
                if (treat_isolated_as_root) {
                    in_tree[static_cast<size_t>(start)] = 1;
                    parent_out[start] = -1;
                    continue;
                }
                return fail("Isolated node encountered outside the root set.", error_buf, error_buf_len);
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
                    return fail("Max steps reached while sampling an unweighted Wilson walk.", error_buf, error_buf_len);
                }
                ++steps_here;
                ++total_steps;

                const int64_t left = rowptr[current];
                const int64_t right = rowptr[current + 1];
                const int64_t degree = right - left;
                if (degree <= 0) {
                    if (treat_isolated_as_root) {
                        in_tree[static_cast<size_t>(current)] = 1;
                        parent_out[current] = -1;
                        break;
                    }
                    return fail("Dead-end encountered in unweighted Wilson sampling.", error_buf, error_buf_len);
                }

                std::uniform_int_distribution<int64_t> pick(0, degree - 1);
                const int64_t successor = col[left + pick(rng)];
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
                parent_out[node] = parent_node;
                in_tree[static_cast<size_t>(node)] = 1;
                node = parent_node;
            }
        }

        int64_t num_roots = 0;
        for (int64_t i = 0; i < num_nodes; ++i) {
            const bool is_root = parent_out[i] == -1 && in_tree[static_cast<size_t>(i)] != 0;
            roots_mask_out[i] = static_cast<uint8_t>(is_root ? 1 : 0);
            num_roots += static_cast<int64_t>(is_root);
        }

        stats_out->walks = walks;
        stats_out->total_steps = total_steps;
        stats_out->total_erased = total_erased;
        stats_out->num_roots = num_roots;
        write_error("", error_buf, error_buf_len);
        return 0;
    } catch (const std::exception& exc) {
        return fail(exc.what(), error_buf, error_buf_len);
    } catch (...) {
        return fail("Unknown C++ exception in sample_forest_unweighted_cpu.", error_buf, error_buf_len);
    }
}

int sample_forest_with_q_factor_unweighted_cpu(
    const int64_t* rowptr,
    const int64_t* col,
    const float* outgoing_mass,
    int64_t num_nodes,
    const int64_t* order,
    double q,
    int64_t max_steps,
    uint64_t seed,
    int64_t* parent_out,
    uint8_t* roots_mask_out,
    WilsonQForestStats* stats_out,
    char* error_buf,
    size_t error_buf_len
) {
    try {
        if (rowptr == nullptr || col == nullptr || outgoing_mass == nullptr || order == nullptr || parent_out == nullptr || roots_mask_out == nullptr || stats_out == nullptr) {
            return fail("sample_forest_with_q_factor_unweighted_cpu received a null pointer.", error_buf, error_buf_len);
        }
        if (num_nodes < 0) {
            return fail("num_nodes must be non-negative.", error_buf, error_buf_len);
        }
        if (!(q > 0.0)) {
            return fail("q must be positive.", error_buf, error_buf_len);
        }

        std::fill(parent_out, parent_out + num_nodes, static_cast<int64_t>(-1));
        std::fill(roots_mask_out, roots_mask_out + num_nodes, static_cast<uint8_t>(0));

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

        for (int64_t order_idx = 0; order_idx < num_nodes; ++order_idx) {
            const int64_t start = order[order_idx];
            if (start < 0 || start >= num_nodes) {
                return fail("order contains an out-of-range index.", error_buf, error_buf_len);
            }
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
                    return fail("Max steps reached while sampling a q-factor Wilson walk.", error_buf, error_buf_len);
                }
                ++steps_here;
                ++total_steps;

                const double total_mass = static_cast<double>(outgoing_mass[current]);
                if (total_mass <= 0.0) {
                    in_tree[static_cast<size_t>(current)] = 1;
                    parent_out[current] = -1;
                    ++kills;
                    break;
                }

                const double p_kill = q / (q + total_mass);
                if (unit(rng) < p_kill) {
                    in_tree[static_cast<size_t>(current)] = 1;
                    parent_out[current] = -1;
                    ++kills;
                    break;
                }

                const int64_t left = rowptr[current];
                const int64_t right = rowptr[current + 1];
                const int64_t degree = right - left;
                if (degree <= 0) {
                    in_tree[static_cast<size_t>(current)] = 1;
                    parent_out[current] = -1;
                    ++kills;
                    break;
                }

                std::uniform_int_distribution<int64_t> pick(0, degree - 1);
                const int64_t successor = col[left + pick(rng)];
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
                parent_out[node] = parent_node;
                in_tree[static_cast<size_t>(node)] = 1;
                node = parent_node;
            }
        }

        int64_t num_roots = 0;
        for (int64_t i = 0; i < num_nodes; ++i) {
            const bool is_root = parent_out[i] == -1 && in_tree[static_cast<size_t>(i)] != 0;
            roots_mask_out[i] = static_cast<uint8_t>(is_root ? 1 : 0);
            num_roots += static_cast<int64_t>(is_root);
        }

        stats_out->walks = walks;
        stats_out->total_steps = total_steps;
        stats_out->total_moves = total_moves;
        stats_out->total_erased = total_erased;
        stats_out->kills = kills;
        stats_out->tree_hits = tree_hits;
        stats_out->num_roots = num_roots;
        write_error("", error_buf, error_buf_len);
        return 0;
    } catch (const std::exception& exc) {
        return fail(exc.what(), error_buf, error_buf_len);
    } catch (...) {
        return fail("Unknown C++ exception in sample_forest_with_q_factor_unweighted_cpu.", error_buf, error_buf_len);
    }
}

}  // extern "C"
