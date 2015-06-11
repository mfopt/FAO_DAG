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

#ifndef LINOP_H
#define LINOP_H

#include <vector>
#include <cassert>
#include <iostream>
#include "gsl/cblas.h"
#include "gsl/gsl_blas.h"
#include "gsl/gsl_vector.h"
#include "gsl/gsl_matrix.h"
#include "gsl/gsl_spmat.h"
#include "gsl/gsl_spblas.h"

/* ID for all coefficient matrices associated with linOps of CONSTANT_TYPE */
static const int CONSTANT_ID = -1;

/* TYPE of each LinOP */
enum operatortype {
    VARIABLE,
    PROMOTE,
    MUL,
    RMUL,
    MUL_ELEM,
    DIV,
    SUM,
    NEG,
    INDEX,
    TRANSPOSE,
    SUM_ENTRIES,
    TRACE,
    RESHAPE,
    DIAG_VEC,
    DIAG_MAT,
    UPPER_TRI,
    CONV,
    HSTACK,
    VSTACK,
    SCALAR_CONST,
    DENSE_CONST,
    SPARSE_CONST,
    NO_OP,
    SPLIT,
    COPY
};

/* linOp TYPE */
typedef operatortype OperatorType;

/* LinOp Class mirrors the CVXPY linOp class. Data fields are determined
      by the TYPE of LinOp. No error checking is performed on the data fields,
      and the semantics of SIZE, ARGS, and DATA depends on the linop TYPE. */
class FAO<T, S> {
public:
    /* Input FAOs in the DAG */
    std::vector<FAO*> input_nodes;
    /* Output FAOs in the DAG */
    std::vector<FAO*> output_nodes;
    /* Dimensions of inputs and outputs. */
    std::vector<std::vector<size_t> > input_sizes;
    std::vector<std::vector<size_t> > output_sizes;
    /* Input and output data arrays. */
    gsl::vector<T> input_data;
    gsl::vector<S> output_data;

    /* Does the FAO operate in-place?

	   Default is no.
    */
    virtual bool is_inplace() {
        return false;
    }

    /* Functions for forward and adjoint evaluation.
       Default is to do nothing. */
    virtual void forward_eval(const gsl::vector<T>& input, gsl::vector<S> *output) {
        return;
    }

    virtual void adjoint_eval(const gsl::vector<S>& input, gsl::vector<T> *output) {
        return;
    }

    /* Allocate the input and output data arrays.
       By default allocates separate arrays.

       If inplace is true, only allocates one array.
     */
    void alloc_data() {
        size_t input_len = get_length(input_sizes);
        input_data = gsl::vector_calloc(input_len);
        size_t output_len = get_length(output_sizes);
        if (kInPlace) {
            assert input_len == output_len;
            output_data = input_data;
        } else {
            output_data = gsl::vector_calloc(output_len);
        }
    }

    void free_data() {
        gsl::vector_free(input_data);
        if (!kInPlace) {
        	gsl::vector_free(output_data);
        }
    }

private:
	/* Returns the length of an input/output element. */
	size_t get_elem_length(std::vector<size_t> elem_dims) {
		size_t len = 0;
		for (auto dim_len : dims) {
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
class Variable<T> : public FAO<T, T> {
public:
    int var_id;
};

/* TODO not technically an FAO. */
class Constant : public FAO<T, T> {
};

class NoOp<T,S> : public FAO<T, S> {
};

class DenseMatMul<T> : public FAO<T, T> {
public:
    gsl::matrix<T, gsl::CblasRowMajor> matrix;
    // TODO should I store the transpose separately?
    // gsl::matrix<T, CblasRowMajor> matrix_trans;

    void set_matrix_data(T* data, size_t rows, size_t cols) {
        matrix = gsl::matrix_alloc(rows, cols);
        matrix.data = data;
    }

    /* Standard dense matrix multiplication. */
    void forward_eval(const gsl::vector<T>& input, gsl::vector<S> *output) {
        return gsl::blas_gemv(gsl::CblasNoTrans, 1, &matrix, &input, 0, output);
    }

    void adjoint_eval(const gsl::vector<S>& input, gsl::vector<T> *output) {
        return gsl::blas_gemv(gsl::CblasTrans, 1, &matrix, &input, 0, output);
    }

};

class SparseMatMul<T> : public FAO<T, T> {
public:
    gsl::spmat<T, size_t, gsl::CblasRowMajor> spmatrix;
    // TODO should I store the transpose separately?
    // gsl::spmat<T, CblasRowMajor> spmatrix_trans;

    void set_spmatrix_data(T *data, size_t data_len, T *ptrs,
                         size_t ptrs_len, size_t *indices, size_t idx_len,
                         size_t rows, size_t cols) {

        assert(rows_len == data_len && cols_len == data_len);
        spmatrix = gsl::spmat_alloc(rows, cols, data_len);
        spmatrix.val = data;
        spmatrix.ind = indices;
        spmatrix.ptr = ptrs;
    }

    /* Standard sparse matrix multiplication. */
    void forward_eval(const gsl::vector<T>& input, gsl::vector<S> *output) {
        return gsl::spblas_gemv(gsl::CblasNoTrans, 1, &spmatrix, &input, 0, output);
    }

    void adjoint_eval(const gsl::vector<S>& input, gsl::vector<T> *output) {
        return gsl::blas_gemv(gsl::CblasTrans, 1, &spmatrix, &input, 0, output);
    }

};

class ScalarMul<T> : public FAO<T, T> {
public:
    T scalar;

    /* Operation is in-place. */
    bool is_inplace() {
        return true;
    }

    /* Scale the input/output. */
    void forward_eval(const gsl::vector<T>& input, gsl::vector<S> *output) {
        return gsl::blas_scal(scalar, output);
    }

    void adjoint_eval(const gsl::vector<T>& input, gsl::vector<S> *output) {
        forward_eval(input, output);
    }
};

class Sum<T> : public FAO<T, T> {
public:
    /* Sum the inputs. */
    void forward_eval(const gsl::vector<T>& input, gsl::vector<S> *output) {
 		size_t elem_size = output->size;
 		auto subvec = vector_subvector(&input, 0, elem_size);
 		gsl::vector_memcpy(&subvec, output);
 		for (size_t i=1; i < input_sizes.length; ++i) {
 			auto subvec = vector_subvector(&input, i*elem_size, elem_size);
 			gsl::blas_axpy(1, &subvec, output);
 		}
    }

    /* Copy the input. */
    void adjoint_eval(const gsl::vector<T>& input, gsl::vector<S> *output) {
        size_t elem_size = input.size;
        for (size_t i=0; i < output_sizes.length; ++i) {
        	auto subvec = vector_subvector(output, i*elem_size, elem_size);
        	gsl::vector_memcpy(&input, subvec);
        }
    }
};


class Copy<T> : public Sum<T> {
/* Adjoint of Sum. */
public:
    /* Copy the inputs. */
    void forward_eval(const gsl::vector<T>& input, gsl::vector<S> *output) {
    	Sum<T>::adjoint_eval(input, output);
    }

    /* Sum the inputs. */
    void adjoint_eval(const gsl::vector<T>& input, gsl::vector<S> *output) {
    	Sum<T>::forward_eval(input, output);
    }
};


class Vstack<T> : public FAO<T, T> {
public:
	/* Operation is in-place. */
	bool is_inplace() {
	    return true;
	}
};

class Split<T> : public Vstack<T> {
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