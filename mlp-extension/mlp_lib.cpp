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
    auto V = torch::empty({nnz}, matrix.options());
    auto COL_INDEX = torch::empty({nnz}, matrix.options());
    auto ROW_INDEX = torch::empty({m + 1}, matrix.options());

    // iterate all the data in the matrix and store them into the arrays
    int64_t num_of_value = 0; // number of values before row i
    for (int64_t i = 0; i < m; i++) {
        ROW_INDEX[i] = num_of_value;
        for (int64_t j = 0; j < n; j++) {
            const auto value = matrix[i][j].item<float>();
            if (value != 0.0) {
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
 * @brief multiple a csr_matrix with a vector
 * @param csr_matrix csr format matrix
 * @param x tensor
 * @return a resulting tensor
 */
torch::Tensor csr_sparse_mv(
        const std::tuple <at::Tensor, at::Tensor, at::Tensor> csr_matrix,
        const at::Tensor &x) {

    // Get the three arrays of matrix A
    const auto A_V = std::get<0>(csr_matrix);
    const auto A_COL_INDEX = std::get<1>(csr_matrix);
    const auto A_ROW_INDEX = std::get<2>(csr_matrix);
    const int64_t m = A_ROW_INDEX.size(0) - 1;
//    const int64_t nnz = A_V.size(0);

    // initialize arrays of result
    torch::Tensor result = torch::zeros(m, A_V.options());

    // sparse matrix multiply vector
    for (int i = 0; i < m; i++) {
        float sum = 0;
        for (int j = A_ROW_INDEX[i].item<int>(); j < A_ROW_INDEX[i+1].item<int>(); j++) {
            sum += A_V[j].item<float>() * x[A_COL_INDEX[j].item<int>()].item<float>();
        }
        result[i] = sum;
    }

    return result;
}


torch::Tensor to_csr_and_mv(const at::Tensor &matrix, const at::Tensor &x) {
    const int64_t m = matrix.size(0);
    const int64_t n = matrix.size(1);
    const int64_t nnz = matrix.nonzero().size(0);

    // initialize arrays of result
    torch::Tensor result = torch::zeros(m, matrix.options());
    #pragma omp parallel
    {
        // initialize the three arrays
        std::vector<double> V(nnz);
        std::vector<int> COL_INDEX(nnz);
        std::vector<int> ROW_INDEX(m+1);

        // iterate all the data in the matrix and store them into the arrays
        int64_t num_of_value = 0; // number of values before row i
        #pragma omp for
        for (int64_t i = 0; i < m; i++) {
            ROW_INDEX[i] = num_of_value;
            for (int64_t j = 0; j < n; j++) {
                const auto value = matrix[i][j].item<float >();
                if (value != 0.0) {
                    V[num_of_value] = value;
                    COL_INDEX[num_of_value] = j;
                    num_of_value++;
                }
            }
        }
        ROW_INDEX[m] = nnz;

        // sparse matrix multiply vector
        #pragma omp for
        for (int i = 0; i < m; i++) {
            double sum = 0;
            for (int j = ROW_INDEX[i]; j < ROW_INDEX[i+1]; j++) {
                int col = COL_INDEX[j];
                sum += V[j] * x[col].item<float>();
            }
            result[i] = sum;
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
at::Tensor mlp_sparse_forward(
        torch::Tensor input,
        torch::Tensor in_weights,
        torch::Tensor hidden_weights,
        torch::Tensor output_weights,
        int64_t num_layers
) {
    torch::Tensor state = torch::relu(to_csr_and_mv(in_weights,input));
    for (int i = 0; i < num_layers; i++) {
        state = torch::relu(to_csr_and_mv(hidden_weights[i],state));
    }
    return to_csr_and_mv(output_weights,state);
//    torch::Tensor state = torch::relu(csr_sparse_mv(to_csr(in_weights),input));
//    for (int i = 0; i < num_layers; i++) {
//        state = torch::relu(csr_sparse_mv(to_csr(hidden_weights[i]),state));
//    }
//    return csr_sparse_mv(to_csr(output_weights),state);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m
) {
m.def("mm_t", &mv, "matrix-vector multiply with transpose");
m.def("mm_t_relu", &mv_relu, "matrix-vector multiply with transpose and relu");
m.def("mlp_forward", &mlp_forward, "mlp forward");
m.def("to_csr", &to_csr, "convert tensor to csv format");
m.def("csr_sparse_mv", &csr_sparse_mv, "matrix-vector multiply by csr format");
m.def("to_csr_and_mv", &to_csr_and_mv, "convert to csr format and matrix-vector multiply");
m.def("mlp_sparse_forward", &mlp_sparse_forward, "mlp forward with csr format");
}