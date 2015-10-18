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
#include <map>
#include <cassert>
#include <iostream>
#include <algorithm>
#include "gsl/cblas.h"
#include "gsl/gsl_blas.h"
#include "gsl/gsl_vector.h"
#include "gsl/gsl_matrix.h"
#include "gsl/gsl_spmat.h"
#include "gsl/gsl_spblas.h"
#include <fftw3.h>
#include "pogs_fork/src/include/timer.h"

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


/* FAO Class mirrors the CVXPY FAO class.  */
class FAO {
public:
	virtual ~FAO() {};
    /* Input edges in the DAG */
    std::vector<int> input_edges;
    /* Output edges in the DAG */
    std::vector<int> output_edges;
    /* Dimensions of inputs and outputs. */
    std::vector<std::vector<size_t> > input_sizes;
    std::vector<std::vector<size_t> > output_sizes;
    /* Input and output data arrays. */
    gsl::vector<double> input_data;
    gsl::vector<double> output_data;
    /* Map from edge index to offset in input_data. */
    std::map<int, size_t> input_offsets;
    /* Map from edge index to offset in output_data. */
    std::map<int, size_t> output_offsets;

    /* Does the FAO operate in-place?

	   Default is no.
    */
    virtual bool is_inplace() {
        return false;
    }

    /* Functions for forward and adjoint evaluation.

	   Default does nothing.
    */
    virtual void forward_eval() {
        return;
    }

    virtual void adjoint_eval() {
        return;
    }

    /* Allocate the input and output data arrays.
       By default allocates separate arrays.

       If inplace is true, only allocates one array.
     */
    virtual void alloc_data() {
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

    /* Initialize the input and output offset maps. */
    void init_offset_maps() {
    	size_t offset = 0;
    	for (size_t i=0; i < input_edges.size(); ++i) {
    		input_offsets[input_edges[i]] = offset;
    		offset += get_elem_length(input_sizes[i]);
    	}
    	offset = 0;
    	for (size_t i=0; i < output_edges.size(); ++i) {
    		output_offsets[output_edges[i]] = offset;
    		offset += get_elem_length(output_sizes[i]);
    	}
    }

    virtual void free_data() {
        gsl::vector_free<double>(&input_data);
        if (!is_inplace()) {
        	gsl::vector_free<double>(&output_data);
        }
    }

	/* Returns the length of an input/output element. */
	size_t get_elem_length(std::vector<size_t> elem_dims) {
		size_t len = 1;
		for (auto dim_len : elem_dims) {
            len *= dim_len;
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

class NoOp : public FAO {
	/* Zero out the output. */
	void forward_eval() {
	    memset(output_data.data, 0, output_data.size*sizeof(double));
	}

	/* Zero out the input. */
	void adjoint_eval() {
	    memset(input_data.data, 0, input_data.size*sizeof(double));
	}
};

class DenseMatVecMul : public FAO {
public:
    gsl::matrix<double, CblasColMajor> matrix;
    // TODO should I store the transpose separately?
    // gsl::matrix<T, CblasRowMajor> matrix_trans;

    void set_matrix_data(double* data, int rows, int cols) {
        // Reverse rows and cols because data is transpose.
        // Needed because SWIG alwasy passes in in row major order.
        matrix = gsl::matrix_alloc<double, CblasColMajor>(cols, rows);
        gsl::matrix_memcpy<double, CblasColMajor>(&matrix, data);
    }

    /* Standard dense matrix multiplication. */
    void forward_eval() {
        return gsl::blas_gemv<double, CblasColMajor>(CblasNoTrans, 1, &matrix,
        	&input_data, 0, &output_data);
    }

    void adjoint_eval() {
        return gsl::blas_gemv<double, CblasColMajor>(CblasTrans, 1, &matrix,
        	&output_data, 0, &input_data);
    }

};

class DenseMatMatMul : public DenseMatVecMul {
public:

    /* Standard dense matrix matrix multiplication AX = Y.
       A in R^{M x K}, X in R^{K x N}, Y in R^{M x N}
    */
    void forward_eval() {
        int M = static_cast<int>(output_sizes[0][0]);
        int N = static_cast<int>(output_sizes[0][1]);
        int K = static_cast<int>(input_sizes[0][0]);
        cblas_dgemm(CblasColMajor, CblasNoTrans,
                    CblasNoTrans, M, N,
                    K, 1, matrix.data,
                    M, input_data.data, K,
                    0, output_data.data, M);

    }

    /* Standard dense matrix matrix multiplication A^TY = X.
       A^T in R^{K x M}, Y in R^{M x N}, X in R^{K x N}
    */
    void adjoint_eval() {
        int M = static_cast<int>(output_sizes[0][0]);
        int N = static_cast<int>(output_sizes[0][1]);
        int K = static_cast<int>(input_sizes[0][0]);
        cblas_dgemm(CblasColMajor, CblasTrans,
                    CblasNoTrans, K, N,
                    M, 1, matrix.data,
                    M, output_data.data, M,
                    0, input_data.data, K);
    }

};

class DenseMatMatRMul : public DenseMatVecMul {
public:

    /* Standard dense matrix matrix multiplication XA = Y.
       X in R^{M x K}, A in R^{K x N}, Y in R^{M x N}
    */
    void forward_eval() {
        int M = static_cast<int>(output_sizes[0][0]);
        int N = static_cast<int>(output_sizes[0][1]);
        int K = static_cast<int>(input_sizes[0][1]);
        cblas_dgemm(CblasColMajor, CblasNoTrans,
                    CblasNoTrans, M, N,
                    K, 1, input_data.data,
                    M, matrix.data, K,
                    0, output_data.data, M);

    }

    /* Standard dense matrix matrix multiplication YA^T = X.
       Y in R^{M x N}, A^T in R^{N x K}, X in R^{M x K}

       We don't transpose A b/c actually in Row major order.
    */
    void adjoint_eval() {
        int M = static_cast<int>(output_sizes[0][0]);
        int N = static_cast<int>(output_sizes[0][1]);
        int K = static_cast<int>(input_sizes[0][1]);
        cblas_dgemm(CblasColMajor, CblasNoTrans,
                    CblasTrans, M, K,
                    N, 1, output_data.data,
                    M, matrix.data, K,
                    0, input_data.data, M);
    }

};

class SparseMatVecMul : public FAO {
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
    void forward_eval() {
        return gsl::spblas_gemv<double, size_t, CblasRowMajor>(CblasNoTrans, 1,
        	&spmatrix, &input_data, 0, &output_data);
    }

    void adjoint_eval() {
        return gsl::spblas_gemv<double, size_t, CblasRowMajor>(CblasTrans, 1,
        	&spmatrix, &output_data, 0, &input_data);
    }

};


class SparseMatMatMul : public SparseMatVecMul {
};

class ScalarMul : public FAO {
public:
    double scalar;

    /* Get the scalar value. */
    virtual double get_scalar() {
    	return scalar;
    }

    /* Operation is in-place. */
    bool is_inplace() {
        return true;
    }

    /* Scale the input/output. */
    void forward_eval() {
        gsl::blas_scal<double>(get_scalar(), &input_data);
    }

    void adjoint_eval() {
        forward_eval();
    }
};

class Neg : public ScalarMul {
public:
    double get_scalar() {
    	return -1;
    }
};

class Sum: public FAO {
public:
    /* Sum the inputs. */
    void forward_eval() {
    	forward_eval_base(input_data, output_data, input_sizes);
    }

    /* Factored out so usable by Sum and Copy. */
    void forward_eval_base(gsl::vector<double> input_data,
    					   gsl::vector<double> output_data,
    					   std::vector<std::vector<size_t> > input_sizes) {
    	size_t elem_size = output_data.size;
    	gsl::vector_subvec_memcpy<double>(&output_data, 0, &input_data, 0, elem_size);
    	for (size_t i=1; i < input_sizes.size(); ++i) {
    		auto subvec = gsl::vector_subvector<double>(&input_data,
    			i*elem_size, elem_size);
    		gsl::blas_axpy<double>(1, &subvec, &output_data);
    	}
    }

    /* Copy the input. */
    void adjoint_eval() {
        adjoint_eval_base(output_data, input_data, input_sizes);
    }

    /* Factored out so usable by Sum and Copy. */
    void adjoint_eval_base(gsl::vector<double> input_data,
    					   gsl::vector<double> output_data,
    					   std::vector<std::vector<size_t> > output_sizes) {
        size_t elem_size = input_data.size;
        for (size_t i=0; i < output_sizes.size(); ++i) {
        	gsl::vector_subvec_memcpy<double>(&output_data, i*elem_size,
        									  &input_data, 0, elem_size);
        }
    }
};


class Copy : public Sum {
/* Adjoint of Sum. */
public:
    /* Copy the inputs. */
    void forward_eval() {
    	Sum::adjoint_eval_base(input_data, output_data, output_sizes);
    }

    /* Sum the inputs. */
    void adjoint_eval() {
    	Sum::forward_eval_base(output_data, input_data, output_sizes);
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


class Reshape : public FAO {
public:
    /* Operation is in-place. */
    bool is_inplace() {
        return true;
    }
};

class Conv : public FAO {
public:

	double *kernel;
	size_t input_len;
	size_t kernel_len;
	size_t padded_len;
	fftw_complex *kernel_fft;
	fftw_complex *rev_kernel_fft;
	fftw_complex *r2c_out;
	fftw_plan forward_fft_plan;
	fftw_plan forward_ifft_plan;
	fftw_plan adjoint_fft_plan;
	fftw_plan adjoint_ifft_plan;

    // Timing info.
    int forward_evals = 0;
    int adjoint_evals = 0;
    double total_forward_r2c_time = 0;
    double total_adjoint_r2c_time = 0;

	void alloc_data() {
		input_len = get_length(input_sizes);
        padded_len = get_length(output_sizes);
        // TODO could use fftw_alloc here.
        input_data = gsl::vector_calloc<double>(padded_len);
       	output_data = gsl::vector_calloc<double>(padded_len);
        kernel_fft = fftw_alloc_complex(padded_len);
        rev_kernel_fft = fftw_alloc_complex(padded_len);
        r2c_out = fftw_alloc_complex(padded_len);
        /* kernel_fft is DFT(padded kernel). */
        /* Must copy because FFTW destroys input array. */
        // TODO alignment of kernel_fft!
        memcpy(input_data.data, kernel, kernel_len*sizeof(double));
        fftw_plan plan = fftw_plan_dft_r2c_1d(padded_len, input_data.data,
        									  kernel_fft, FFTW_ESTIMATE);
     	fftw_execute(plan);
     	fftw_destroy_plan(plan);
     	/* rev_kernel_fft is conj(DFT(padded kernel))=IDFT(padded kernel). */
     	// TODO parallelize.
     	for (size_t i=0; i < padded_len; ++i) {
     		rev_kernel_fft[i][0] = kernel_fft[i][0];
     		rev_kernel_fft[i][1] = -kernel_fft[i][1];
     	}
     	/* Initialize the plans for forward_eval. */
     	// TODO also FFTW_MEASURE for faster planning, worse performance.
     	forward_fft_plan = fftw_plan_dft_r2c_1d(padded_len, input_data.data,
     		r2c_out, FFTW_MEASURE);
     	forward_ifft_plan = fftw_plan_dft_c2r_1d(padded_len, r2c_out,
     		output_data.data, FFTW_MEASURE);
     	adjoint_fft_plan = fftw_plan_dft_r2c_1d(padded_len, output_data.data,
     		r2c_out, FFTW_MEASURE);
     	adjoint_ifft_plan = fftw_plan_dft_c2r_1d(padded_len, r2c_out,
     		input_data.data, FFTW_MEASURE);
    }

    void free_data() {
        printf("n=%u, avg_forward_r2c=%e\n", input_len,
            total_forward_r2c_time/forward_evals);
        printf("n=%u, avg_adjoint_r2c=%e\n", input_len,
            total_adjoint_r2c_time/adjoint_evals);
    	fftw_destroy_plan(forward_fft_plan);
    	fftw_destroy_plan(forward_ifft_plan);
    	fftw_destroy_plan(adjoint_fft_plan);
    	fftw_destroy_plan(adjoint_ifft_plan);
    	fftw_free(kernel_fft);
    	fftw_free(rev_kernel_fft);
    	fftw_free(r2c_out);
    	fftw_cleanup();
        FAO::free_data();
    }

	void set_conv_data(double *kernel, int kernel_len) {
		this->kernel = kernel;
		this->kernel_len = kernel_len;
	}

	/* Multiply kernel_fft and output.
	   Divide by n because fftw doesn't.
	   Writes to output.
	*/
	// TODO parallelize.
	void multiply_fft(fftw_complex *kernel_fft, fftw_complex *output) {
		double len = (double) padded_len;
		double tmp;
    	for (size_t i=0; i < padded_len; ++i) {
    		tmp = (kernel_fft[i][0]*output[i][0] -
    			   kernel_fft[i][1]*output[i][1])/len;
    		output[i][1] = (kernel_fft[i][0]*output[i][1] +
    						kernel_fft[i][1]*output[i][0])/len;
    		output[i][0] = tmp;
    	}
	}

	/* Fill out the input padding with zeros. */
	void zero_pad_input() {
		/* Zero out extra part of input. */
		memset(input_data.data + input_len, 0,
			   (padded_len - input_len)*sizeof(double));
	}

	/* Column convolution. */
    void forward_eval() {
        forward_evals++;
        double t = timer<double>();
    	zero_pad_input();
        // printf("T_exec_zero_pad = %e\n", timer<double>() - t);
        t = timer<double>();
    	fftw_execute(forward_fft_plan);
        double r2c_time = timer<double>() - t;
        total_forward_r2c_time += r2c_time;
        // printf("T_exec_r2c = %e\n", r2c_time);
        t = timer<double>();
    	multiply_fft(kernel_fft, r2c_out);
        // printf("T_multiply_fft = %e\n", timer<double>() - t);
        t = timer<double>();
    	fftw_execute(forward_ifft_plan);
        // printf("T_exec_c2r = %e\n", timer<double>() - t);
    }

    /* Row convolution. */
    void adjoint_eval() {
        adjoint_evals++;
        double t = timer<double>();
    	fftw_execute(adjoint_fft_plan);
        double r2c_time = timer<double>() - t;
        total_adjoint_r2c_time += r2c_time;
        // printf("T_exec_r2c = %e\n", r2c_time);
        t = timer<double>();
    	multiply_fft(rev_kernel_fft, r2c_out);
        // printf("T_multiply_fft = %e\n", timer<double>() - t);
        t = timer<double>();
		fftw_execute(adjoint_ifft_plan);
        // printf("T_exec_c2r = %e\n", timer<double>() - t);
		// TODO do this? zero_pad_input();
    }
};
#endif