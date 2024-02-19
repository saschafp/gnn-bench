#include <string>

#include "matrix.hpp"

template <typename T>
void readMatrixMarketCOO(const std::string &filename, COOMatrix<T> &cooMatrix)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    // Read the header
    int nrows, ncols, nnz;
    file >> nrows >> ncols >> nnz;

    // Allocate memory for the COO matrix
    cooMatrix.nrows = nrows;
    cooMatrix.ncols = ncols;
    cooMatrix.vals = new T[nnz];
    cooMatrix.rows = new int64_t[nnz];
    cooMatrix.cols = new int64_t[nnz];

    // Read the data
    for (int i = 0; i < nnz; i++)
    {
        file >> cooMatrix.rows[i] >> cooMatrix.cols[i] >> cooMatrix.vals[i];
    }

    file.close();
    return;
}