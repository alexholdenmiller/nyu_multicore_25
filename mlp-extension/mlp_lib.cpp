#include <torch/extension.h>
#include <iostream>
#include <vector>
#include <omp.h>

// matrix multiply with a transpose
at::Tensor mv(
        torch::Tensor input,
        torch::Tensor weights
) {
    return weights.mv(input);
}

// matrix_vector multiply with transpose and relu
at::Tensor mv_relu(
        torch::Tensor input,
        torch::Tensor weights
) {
    return torch::relu(weights.mv(input));
}

// full mlp forward given input weights, output weights, and 3D matrix of hidden layer weights
at::Tensor mlp_forward(
        torch::Tensor input,
        torch::Tensor in_weights,
        torch::Tensor hidden_weights,
        torch::Tensor output_weights,
        int64_t num_layers
) {
    torch::Tensor state = torch::relu(in_weights.mv(input));
    for (int i = 0; i < num_layers; i++) {
        state = torch::relu(hidden_weights[i].mv(state));
    }
    return output_weights.mv(state);
}


/**
 * @brief convert tensor to csr format
 * @param matrix tensor to be converted
 * @return a triple consisting of the representation of the csr format
 */

std::tuple <at::Tensor, at::Tensor, at::Tensor> to_csr(const at::Tensor &matrix) {
    const int64_t m = matrix.size(0);
    const int64_t n = matrix.size(1);
    const int64_t nnz = matrix.nonzero().size(0);

    // initialize the three arrays
    auto f_opt = torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false);
    auto i_opt = torch::TensorOptions().dtype(torch::kInt32).requires_grad(false);
    auto V = torch::empty({nnz}, f_opt);
    auto COL_INDEX = torch::empty({nnz}, i_opt);
    auto ROW_INDEX = torch::empty({m + 1}, i_opt);

    // iterate all the data in the matrix and store them into the arrays
    int64_t num_of_value = 0; // number of values before row i
    for (int64_t i = 0; i < m; i++) {
        ROW_INDEX[i] = num_of_value;
        for (int64_t j = 0; j < n; j++) {
            const auto value = matrix[i][j];
            if (value.item<float>() != 0.0) {
                V[num_of_value] = value;
                COL_INDEX[num_of_value] = j;
                num_of_value++;
            }
        }
    }
    ROW_INDEX[m] = nnz;
    return std::make_tuple(V, COL_INDEX, ROW_INDEX);
}


/**
 * @brief convert tensor to csr format
 * @param matrix tensor to be converted
 * @return a triple consisting of the representation of the csr format
 */

std::tuple <std::vector<float_t>, std::vector<int32_t>, std::vector<int32_t>, int64_t> to_coo(const at::Tensor &matrix) {
    const int64_t m = matrix.size(0);
    const int64_t n = matrix.size(1);

    std::vector<int32_t> COL_INDEX;
    std::vector<int32_t> ROW_INDEX;
    std::vector<float_t> V;

    // iterate all the data in the matrix and store them into the arrays
    for (int64_t i = 0; i < m; i++) {
        for (int64_t j = 0; j < n; j++) {
            const auto value = matrix[i][j];
            if (value.item<float>() != 0.0) {
                ROW_INDEX.push_back(i);
                COL_INDEX.push_back(j);
                V.push_back(value.item<float>());
            }
        }
    }

    return std::make_tuple(V, COL_INDEX, ROW_INDEX, m);
}


/**
 * @brief multiple a csr_matrix with a vector
 * @param csr_matrix csr format matrix
 * @param x tensor
 * @return a resulting tensor
 */
torch::Tensor csr_sparse_mv(
        const std::tuple <at::Tensor, at::Tensor, at::Tensor> csr_matrix,
        const at::Tensor &x
) {
    const auto A_V = std::get<0>(csr_matrix);
    const auto A_COL_INDEX = std::get<1>(csr_matrix);
    const auto A_ROW_INDEX = std::get<2>(csr_matrix);
    const int64_t m = A_ROW_INDEX.size(0) - 1;

    auto opt = torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false);
    torch::Tensor result = torch::zeros(m, opt);

    // sparse matrix multiply vector
    for (int i = 0; i < m; i++) {
        for (int j = A_ROW_INDEX[i].item<int>(); j < A_ROW_INDEX[i+1].item<int>(); j++) {
            result[i] += A_V[j] * x[A_COL_INDEX[j]];
        }
    }

    return result;
}


/**
 * @brief multiple a coo_matrix with a vector
 * @param coo_matrix coo format matrix
 * @param x tensor
 * @return a resulting tensor
 */
torch::Tensor coo_sparse_mv(
        const std::tuple <std::vector<float_t>, std::vector<int32_t>, std::vector<int32_t>, int64_t> coo_matrix,
        const at::Tensor &x
) {
    const auto A_V = std::get<0>(coo_matrix);
    const auto A_COL_INDEX = std::get<1>(coo_matrix);
    const auto A_ROW_INDEX = std::get<2>(coo_matrix);
    const int nnz = (int) A_V.size();
    const auto m = std::get<3>(coo_matrix);

    // initialize arrays of result
    auto opt = torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false);
    torch::Tensor result = torch::zeros(m, opt);

    // sparse matrix multiply vector
    // THIS IS THE SINGLE-THREADED ONE: NO OMP
    for (int ind = 0; ind < nnz; ind++) {
        int32_t i = A_ROW_INDEX[ind];
        int32_t j = A_COL_INDEX[ind];
        int32_t v = A_V[ind];
        result[i] += v * x[j];
    }

    return result;
}

/**
 * uses omp to do multithreaded processing
 * @brief multiple a csr_matrix with a vector
 * @param csr_matrix csr format matrix
 * @param x tensor
 * @return a resulting tensor
 */
torch::Tensor csr_sparse_mv_mt(
        const std::tuple <at::Tensor, at::Tensor, at::Tensor> csr_matrix,
        const at::Tensor &x,
        int8_t num_threads
) {
    const auto A_V = std::get<0>(csr_matrix);
    const auto A_COL_INDEX = std::get<1>(csr_matrix);
    const auto A_ROW_INDEX = std::get<2>(csr_matrix);
    const int64_t m = A_ROW_INDEX.size(0) - 1;  // same as x.size(0)

    // initialize arrays of result
    auto opt = torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false);
    torch::Tensor result = torch::zeros(m, x.options());

    const int64_t step = ((m + num_threads - 2) / num_threads) + 1;

    // sparse matrix multiply vector
    # pragma omp parallel num_threads(num_threads) shared(result)
    {
        int16_t rank = omp_get_thread_num();
        for (int i = rank*step; i < std::min((rank+1) * step, m); i++) {
            for (int j = A_ROW_INDEX[i].item<int>(); j < A_ROW_INDEX[i+1].item<int>(); j++) {
                result[i] += A_V[j] * x[A_COL_INDEX[j]];
            }
        }
    }

    
    return result;
}

/**
 * uses omp to do multithreaded processing
 * @brief multiple a coo_matrix with a vector
 * @param coo_matrix coo format matrix
 * @param x tensor
 * @return a resulting tensor
*/

torch::Tensor coo_sparse_mv_mt(
        const std::tuple<std::vector<float_t>, std::vector<int32_t>, std::vector<int32_t>, int64_t> coo_matrix,
        const at::Tensor &x,
        int8_t num_threads
) {
    const auto A_V = std::get<0>(coo_matrix);
    const auto A_COL_INDEX = std::get<1>(coo_matrix);
    const auto A_ROW_INDEX = std::get<2>(coo_matrix);
    const int64_t nnz = A_V.size();
    const auto m = std::get<3>(coo_matrix);

    // initialize arrays of result
    torch::Tensor result = torch::zeros(m, x.options());

    const int64_t step = ((nnz + num_threads - 2) / num_threads) + 1;

    // sparse matrix multiply vector
    # pragma omp parallel num_threads(num_threads) shared(result)
    {
        int16_t rank = omp_get_thread_num();
        for (int ind = rank*step; ind < std::min((rank+1) * step, nnz); ind++) {
            int32_t i = A_ROW_INDEX[ind];
            int32_t j = A_COL_INDEX[ind];
            int32_t v = A_V[ind];
            result[i] += v * x[j];
        }
    }

    return result;
}


/**
 * mlp with tensor in sparse matrix format
 * @param input
 * @param in_weights
 * @param hidden_weights
 * @param output_weights
 * @param num_layers
 * @return
 */
at::Tensor mlp_sparse_forward_csr(
        torch::Tensor input,
        std::tuple <at::Tensor, at::Tensor, at::Tensor> in_weights,
        std::vector<std::tuple <at::Tensor, at::Tensor, at::Tensor>> hidden_weights,
        std::tuple <at::Tensor, at::Tensor, at::Tensor> output_weights,
        int16_t num_layers
) {
    torch::Tensor state = torch::relu(csr_sparse_mv(in_weights, input));
    for (int i = 0; i < num_layers; i++) {
        state = torch::relu(csr_sparse_mv(hidden_weights[i], state));
    }
    return csr_sparse_mv(output_weights, state);
}

/**
 * mlp with tensor in sparse matrix format
 * @param input
 * @param in_weights
 * @param hidden_weights
 * @param output_weights
 * @param num_layers
 * @return
 */
at::Tensor mlp_sparse_forward_coo(
        torch::Tensor input,
        std::tuple <std::vector<float_t>, std::vector<int32_t>, std::vector<int32_t>, int64_t> in_weights,
        std::vector<std::tuple <std::vector<float_t>, std::vector<int32_t>, std::vector<int32_t>, int64_t>> hidden_weights,
        std::tuple <std::vector<float_t>, std::vector<int32_t>, std::vector<int32_t>, int64_t> output_weights,
        int16_t num_layers
) {
    torch::Tensor state = torch::relu(coo_sparse_mv(in_weights, input));
    for (int i = 0; i < num_layers; i++) {
        state = torch::relu(coo_sparse_mv(hidden_weights[i], state));
    }
    return coo_sparse_mv(output_weights, state);
}


/**
 * mlp with tensor in sparse matrix format
 * @param input
 * @param in_weights
 * @param hidden_weights
 * @param output_weights
 * @param num_layers
 * @return
*/
at::Tensor mlp_sparse_forward_mt_csr(
        torch::Tensor input,
        std::tuple <at::Tensor, at::Tensor, at::Tensor> in_weights,
        std::vector<std::tuple <at::Tensor, at::Tensor, at::Tensor>> hidden_weights,
        std::tuple <at::Tensor, at::Tensor, at::Tensor> output_weights,
        int16_t num_layers,
        int8_t num_threads
) {
    torch::Tensor state = torch::relu(csr_sparse_mv_mt(in_weights, input, num_threads));
    for (int i = 0; i < num_layers; i++) {
        state = torch::relu(csr_sparse_mv_mt(hidden_weights[i], state, num_threads));
    }
    return csr_sparse_mv_mt(output_weights, state, num_threads);
}

/**
 * mlp with tensor in sparse matrix format
 * @param input
 * @param in_weights
 * @param hidden_weights
 * @param output_weights
 * @param num_layers
 * @return
*/
at::Tensor mlp_sparse_forward_mt_coo(
        torch::Tensor input,
        std::tuple <std::vector<float_t>, std::vector<int32_t>, std::vector<int32_t>, int64_t> in_weights,
        std::vector<std::tuple <std::vector<float_t>, std::vector<int32_t>, std::vector<int32_t>, int64_t>> hidden_weights,
        std::tuple <std::vector<float_t>, std::vector<int32_t>, std::vector<int32_t>, int64_t> output_weights,
        int16_t num_layers,
        int8_t num_threads
) {
    torch::Tensor state = torch::relu(coo_sparse_mv_mt(in_weights, input, num_threads));
    for (int i = 0; i < num_layers; i++) {
        state = torch::relu(coo_sparse_mv_mt(hidden_weights[i], state, num_threads));
    }
    return coo_sparse_mv_mt(output_weights, state, num_threads);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m
) {
m.def("mm_t", &mv, "matrix-vector multiply with transpose");
m.def("mm_t_relu", &mv_relu, "matrix-vector multiply with transpose and relu");
m.def("mlp_forward", &mlp_forward, "mlp forward");
m.def("to_csr", &to_csr, "convert tensor to csv format");
m.def("to_coo", &to_coo, "convert tensor to coo format");
m.def("mlp_sparse_forward_csr", &mlp_sparse_forward_csr, "mlp forward with csr format");
m.def("mlp_sparse_forward_coo", &mlp_sparse_forward_coo, "mlp forward with coo format");
m.def("mlp_sparse_forward_mt_csr", &mlp_sparse_forward_mt_csr, "mlp forward with csr format and multithreading");
m.def("mlp_sparse_forward_mt_coo", &mlp_sparse_forward_mt_coo, "mlp forward with csr format and multithreading");
}