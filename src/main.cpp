#include <ATen/ATen.h>
#include <fast_matrix_market/fast_matrix_market.hpp>
#include <iostream>
#include <fstream>

#include "matrix.hpp"

namespace fmm = fast_matrix_market;

/**
 * @brief Forward pass of a graph neural network
 * @param X Feature tensor, shape (n, d_in)
 * @param W Weight tensor, shape (d_out, d_in)
 * @param A Adjacency matrix, shape (n, n)
 * @return Output features, shape (n, d_out)
 */
at::Tensor forward_pass(at::Tensor &X, at::Tensor &W, at::Tensor &A)
{
    // Message aggregation Z = A @ X, Z has shape (n, d_in)
    auto Z = A.mm(X);

    // Linear transformation H = Z @ W.T, has shape (n, d_out)
    auto H = Z.mm(W.t());

    // <optional> Activation function
    // (e.g. ReLU, Sigmoid, Tanh, etc.)
    // auto H = torch::relu(Z.mm(W));

    return H;
}

/**
 * @brief Compute the gradient of the loss with respect to the input features
 * @param X Feature tensor, shape (n, d_in)
 * @param W Weight tensor, shape (d_out, d_in)
 * @param A Adjacency matrix, shape (n, n)
 * @param dL_dH Gradient of the loss with respect to the output features, shape (n, d_out)
 * @return Gradient of the loss with respect to the input features, shape (n, d_in)
 */
at::Tensor compute_grad(at::Tensor &X, at::Tensor &W, at::Tensor &A, at::Tensor dL_dH)
{
    // Compute the gradient of the loss with respect to the weights
    auto grad_intermediate = A.t().mm(dL_dH); // has shape (n, d_out)
    auto dL_dW = grad_intermediate.t().mm(X); // has shape (d_out, d_in)
    return dL_dW;
}

/**
 * @brief Update the weights using the gradient descent algorithm
 * @param W Weight tensor, shape (d_out, d_in)
 * @param dL_dW Gradient of the loss with respect to the weights, shape (d_out, d_in)
 * @param learning_rate Learning rate
 */
void update_weights(at::Tensor &W, at::Tensor &dL_dW, double learning_rate)
{
    W -= learning_rate * dL_dW;
}

int main(int argc, char *argv[])
{
    // Load a matrix from a file
    std::ifstream file("../data/gre_1107.mtx");
    COOMatrix<double> coo_matrix;
    fmm::matrix_market_header header;
    fmm::read_matrix_market_triplet(file, header, coo_matrix.rows, coo_matrix.cols, coo_matrix.vals);
    coo_matrix.ncols = header.nrows;
    coo_matrix.nrows = header.ncols;

    // Concatenate the row and column indices
    int64_t nnz = coo_matrix.vals.size();
    std::vector<int64_t> concatenated_indices(2 * nnz);
    for (int64_t i = 0; i < nnz; i++)
    {
        concatenated_indices[2 * i] = coo_matrix.rows[i];
        concatenated_indices[2 * i + 1] = coo_matrix.cols[i];
    }

    // Dimensions
    int64_t n = coo_matrix.nrows; // Number of nodes
    int64_t d_in = 100;           // Number of input features
    int64_t d_out = 50;           // Number of output features

    // Create tensors

    // Create random feature vectors
    // Feature tensor X has shape (n, d_in)
    auto X = at::randn({n, d_in}, at::kDouble);

    // Create random weights
    // Weight tensor W has shape (d_in, d_out)
    auto W = at::randn({d_out, d_in}, at::kDouble);

    // Create a adjacency matrix
    auto indices = at::from_blob(concatenated_indices.data(), {nnz, 2}, at::kLong);
    auto values = at::from_blob(coo_matrix.vals.data(), {nnz}, at::kDouble);
    auto A = at::sparse_coo_tensor(indices.t(), values, {coo_matrix.nrows, coo_matrix.ncols});

    // Add self-loops to the adjacency matrix
    auto I = at::eye(n).to_sparse();
    A = I + A;

    // Check if A is sparse
    std::cout << "Is A sparse? " << A.is_sparse() << std::endl;

    std::cout << "Tensors created!" << std::endl;

    // Forward pass
    auto H = forward_pass(X, W, A); // Output features, shape (n, d_out)
    std::cout << "Forward pass done!" << std::endl;

    // Backward pass
    auto dL_dH = at::randn({n, d_out}, at::kDouble); // Dummy gradient of the loss with respect to the output features
    auto dL_dW = compute_grad(X, W, A, dL_dH);

    std::cout << "Backward pass done!" << std::endl;

    // Optimizer
    double learning_rate = 0.01;
    update_weights(W, dL_dW, learning_rate);

    std::cout << "Weights updated!" << std::endl;

    return 0;
}
