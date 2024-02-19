#ifndef MATRIX_HPP
#define MATRIX_HPP


/**
 * @brief A struct representing a COO (Coordinate) matrix.
 * 
 * This struct stores the number of rows and columns of the matrix,
 * as well as arrays of non-zero values, row indices, and column indices.
 * 
 * @tparam T The type of the values in the matrix.
 */
template <typename T>
struct COOMatrix
{
    int nrows;  // number of rows
    int ncols;  // number of columns
    std::vector<T> vals;  // array of non-zero values
    std::vector<int64_t> rows;  // array of row indices
    std::vector<int64_t> cols;  // array of column indices
};

/**
 * Reads a matrix in Matrix Market COO format from the specified file.
 *
 * @param filename The path to the file containing the matrix data.
 * @param cooMatrix The COOMatrix object to store the matrix data.
 */
template <typename T>
bool readMatrixMarketCOO(const std::string& filename, COOMatrix<T>& cooMatrix);


#endif // MATRIX_HPP