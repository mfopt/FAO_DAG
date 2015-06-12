//    This file is part of CVXcanon.
//
//    CVXcanon is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    CVXcanon is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with CVXcanon.  If not, see <http://www.gnu.org/licenses/>.

#ifndef FAO_H
#define FAO_H

#include <vector>
#include <cassert>
#include <iostream>
#include "gsl/cblas.h"
#include "gsl/gsl_blas.h"
#include "gsl/gsl_vector.h"
#include "gsl/gsl_matrix.h"
#include "gsl/gsl_spmat.h"
#include "gsl/gsl_spblas.h"

// /* ID for all coefficient matrices associated with linOps of CONSTANT_TYPE */
// static const int CONSTANT_ID = -1;

// /* TYPE of each LinOP */
// enum operatortype {
//     VARIABLE,
//     PROMOTE,
//     MUL,
//     RMUL,
//     MUL_ELEM,
//     DIV,
//     SUM,
//     NEG,
//     INDEX,
//     TRANSPOSE,
//     SUM_ENTRIES,
//     TRACE,
//     RESHAPE,
//     DIAG_VEC,
//     DIAG_MAT,
//     UPPER_TRI,
//     CONV,
//     HSTACK,
//     VSTACK,
//     SCALAR_CONST,
//     DENSE_CONST,
//     SPARSE_CONST,
//     NO_OP,
//     SPLIT,
//     COPY
// };

// /* linOp TYPE */
// typedef operatortype OperatorType;

/* LinOp Class mirrors the CVXPY linOp class. Data fields are determined
      by the TYPE of LinOp. No error checking is performed on the data fields,
      and the semantics of SIZE, ARGS, and DATA depends on the linop TYPE. */
class FAO {
public:
    /* Input FAOs in the DAG */
    std::vector<FAO*> input_nodes;
    /* Output FAOs in the DAG */
    std::vector<FAO*> output_nodes;
    /* Dimensions of inputs and outputs. */
    std::vector<std::vector<size_t> > input_sizes;
    std::vector<std::vector<size_t> > output_sizes;
    /* Input and output data arrays. */
    gsl::vector<double> input_data;
    gsl::vector<double> output_data;

    /* Does the FAO operate in-place?

	   Default is no.
    */
    bool is_inplace() {
        return false;
    }

    /* Functions for forward and adjoint evaluation.  */
    void forward_eval(gsl::vector<double>* input, gsl::vector<double> *output) {
        return;
    }

    void adjoint_eval(gsl::vector<double>* input, gsl::vector<double> *output) {
        return;
    }

    /* Allocate the input and output data arrays.
       By default allocates separate arrays.

       If inplace is true, only allocates one array.
     */
    void alloc_data() {
        size_t input_len = get_length(input_sizes);
        input_data = gsl::vector_calloc<double>(input_len);
        size_t output_len = get_length(output_sizes);
        if (is_inplace()) {
            assert(input_len == output_len);
            output_data = input_data;
        } else {
            output_data = gsl::vector_calloc<double>(output_len);
        }
    }

    void free_data() {
        gsl::vector_free<double>(&input_data);
        if (!is_inplace()) {
        	gsl::vector_free<double>(&output_data);
        }
    }

private:
	/* Returns the length of an input/output element. */
	size_t get_elem_length(std::vector<size_t> elem_dims) {
		size_t len = 0;
		for (auto dim_len : elem_dims) {
            len += dim_len;
        }
        return len;
	}

    /* Returns the total length of an array of dimensions. */
    size_t get_length(std::vector<std::vector<size_t> > sizes) {
        size_t len = 0;
        for (auto elem_dims : sizes) {
            len += get_elem_length(elem_dims);
        }
        return len;
    }
};

/* TODO not technically an FAO. */
class Variable : public FAO {
public:
    int var_id;
};

/* TODO not technically an FAO. */
class Constant : public FAO {
};

class NoOp : public FAO {
};

class DenseMatMul : public FAO {
public:
    gsl::matrix<double, CblasRowMajor> matrix;
    // TODO should I store the transpose separately?
    // gsl::matrix<T, CblasRowMajor> matrix_trans;

    void set_matrix_data(double* data, int rows, int cols) {
        matrix = gsl::matrix_alloc<double, CblasRowMajor>(rows, cols);
        matrix.data = data;
    }

    /* Standard dense matrix multiplication. */
    void forward_eval(gsl::vector<double>* input, gsl::vector<double> *output) {
        return gsl::blas_gemv<double, CblasRowMajor>(CblasNoTrans, 1, &matrix, input, 0, output);
    }

    void adjoint_eval(gsl::vector<double>* input, gsl::vector<double> *output) {
        return gsl::blas_gemv<double, CblasRowMajor>(CblasTrans, 1, &matrix, input, 0, output);
    }

};

class SparseMatMul : public FAO {
public:
    gsl::spmat<double, size_t, CblasRowMajor> spmatrix;
    // TODO should I store the transpose separately?
    // gsl::spmat<T, CblasRowMajor> spmatrix_trans;

    void set_spmatrix_data(double *data, size_t data_len, size_t *ptrs,
                         size_t ptrs_len, size_t *indices, size_t idx_len,
                         size_t rows, size_t cols) {

        assert(rows_len == data_len && cols_len == data_len);
        spmatrix = gsl::spmat_alloc<double, size_t, CblasRowMajor>(rows, cols, data_len);
        spmatrix.val = data;
        spmatrix.ind = indices;
        spmatrix.ptr = ptrs;
    }

    /* Standard sparse matrix multiplication. */
    void forward_eval(gsl::vector<double>* input, gsl::vector<double> *output) {
        return gsl::spblas_gemv<double, size_t, CblasRowMajor>(CblasNoTrans, 1, &spmatrix, input, 0, output);
    }

    void adjoint_eval(gsl::vector<double>* input, gsl::vector<double> *output) {
        return gsl::spblas_gemv<double, size_t, CblasRowMajor>(CblasTrans, 1, &spmatrix, input, 0, output);
    }

};

class ScalarMul : public FAO {
public:
    double scalar;

    /* Operation is in-place. */
    bool is_inplace() {
        return true;
    }

    /* Scale the input/output. */
    void forward_eval(gsl::vector<double>* input, gsl::vector<double> *output) {
        return gsl::blas_scal<double>(scalar, output);
    }

    void adjoint_eval(gsl::vector<double>* input, gsl::vector<double> *output) {
        forward_eval(input, output);
    }
};

class Neg : public ScalarMul {
public:
    double scalar = -1;
};

class Sum: public FAO {
public:
    /* Sum the inputs. */
    void forward_eval(gsl::vector<double>* input, gsl::vector<double> *output) {
 		size_t elem_size = output->size;
 		auto subvec = gsl::vector_subvector<double>(input, 0, elem_size);
 		gsl::vector_memcpy<double>(&subvec, output);
 		for (size_t i=1; i < input_sizes.size(); ++i) {
 			auto subvec = gsl::vector_subvector<double>(input, i*elem_size, elem_size);
 			gsl::blas_axpy<double>(1, &subvec, output);
 		}
    }

    /* Copy the input. */
    void adjoint_eval(gsl::vector<double>* input, gsl::vector<double> *output) {
        size_t elem_size = input->size;
        for (size_t i=0; i < output_sizes.size(); ++i) {
        	auto subvec = gsl::vector_subvector<double>(output, i*elem_size, elem_size);
        	gsl::vector_memcpy<double>(input, &subvec);
        }
    }
};


class Copy : public Sum {
/* Adjoint of Sum. */
public:
    /* Copy the inputs. */
    void forward_eval(gsl::vector<double>* input, gsl::vector<double> *output) {
    	Sum::adjoint_eval(input, output);
    }

    /* Sum the inputs. */
    void adjoint_eval(gsl::vector<double>* input, gsl::vector<double> *output) {
    	Sum::forward_eval(input, output);
    }
};


class Vstack : public FAO {
public:
	/* Operation is in-place. */
	bool is_inplace() {
	    return true;
	}
};

class Split : public Vstack {
	/* Adjoint of vstack. */
};


//     /* Initializes DENSE_DATA. MATRIX is a pointer to the data of a 2D
//      * numpy array, ROWS and COLS are the size of the ARRAY.
//      *
//      * MATRIX must be a contiguous array of doubles aligned in fortran
//      * order.
//      *
//      * NOTE: The function prototype must match the type-map in CVXCanon.i
//      * exactly to compile and run properly.
//      */
//     void set_dense_data(double* matrix, int rows, int cols) {
//         dense_data = Eigen::Map<Eigen::MatrixXd> (matrix, rows, cols);
//     }

//      Initializes SPARSE_DATA from a sparse matrix in COO format.
//      * DATA, ROW_IDXS, COL_IDXS are assumed to be contiguous 1D numpy arrays
//      * where (DATA[i], ROW_IDXS[i], COLS_IDXS[i]) is a (V, I, J) triplet in
//      * the matrix. ROWS and COLS should refer to the size of the matrix.
//      *
//      * NOTE: The function prototype must match the type-map in CVXCanon.i
//      * exactly to compile and run properly.

    // void set_sparse_data(double *data, int data_len, double *row_idxs,
    //                      int rows_len, double *col_idxs, int cols_len,
    //                      int rows, int cols) {

    //     assert(rows_len == data_len && cols_len == data_len);
    //     sparse = true;
    //     Matrix sparse_coeffs(rows, cols);
    //     std::vector<Triplet> tripletList;
    //     tripletList.reserve(data_len);
    //     for (int idx = 0; idx < data_len; idx++) {
    //         tripletList.push_back(Triplet(int(row_idxs[idx]), int(col_idxs[idx]),
    //                                       data[idx]));
    //     }
    //     sparse_coeffs.setFromTriplets(tripletList.begin(), tripletList.end());
    //     sparse_coeffs.makeCompressed();
    //     sparse_data = sparse_coeffs;
    // }
// };
#endif