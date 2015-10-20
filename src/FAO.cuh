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
#include "cml/cml_blas.cuh"
#include "cml/cml_vector.cuh"
#include "cml/cml_matrix.cuh"
#include "cml/cml_spmat.cuh"
#include "cml/cml_spblas.cuh"
#include "cml/cml_utils.cuh"
#include "pogs_fork/src/include/util.h"
#include "pogs_fork/src/include/timer.h"
#include <cufft.h>
#include <thrust/device_ptr.h>
#include <thrust/transform.h>
#include <thrust/complex.h>
#include <thrust/iterator/constant_iterator.h>


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
template <class T>
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
    cml::vector<T> input_data;
    cml::vector<T> output_data;
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
        input_data = cml::vector_calloc<T>(input_len);
        size_t output_len = get_length(output_sizes);
        if (is_inplace()) {
            assert(input_len == output_len);
            output_data = input_data;
        } else {
            output_data = cml::vector_calloc<T>(output_len);
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
        cml::vector_free<T>(&input_data);
        if (!is_inplace()) {
            cml::vector_free<T>(&output_data);
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

template <class T>
class NoOp : public FAO<T> {
    /* Zero out the output. */
    void forward_eval() {
        cml::vector_scale<T>(&this->output_data, 0.0);
        cudaDeviceSynchronize();
        CUDA_CHECK_ERR();
    }

    /* Zero out the input. */
    void adjoint_eval() {
        cml::vector_scale<T>(&this->input_data, 0.0);
        cudaDeviceSynchronize();
        CUDA_CHECK_ERR();
    }
};

template <class T>
class DenseMatVecMul : public FAO<T> {
public:
    cml::matrix<T, CblasColMajor> matrix;
    cublasHandle_t hdl;
    // TODO should I store the transpose separately?
    // gsl::matrix<T, CblasRowMajor> matrix_trans;

    DenseMatVecMul() {
        cublasCreate(&hdl);
        CUDA_CHECK_ERR();
    }

    ~DenseMatVecMul() {
        cublasDestroy(hdl);
        CUDA_CHECK_ERR();
    }

    void free_data() {
        cml::matrix_free(&matrix);
        FAO<T>::free_data();
    }

    void set_matrix_data(T* data, int rows, int cols) {
        // Reverse rows and cols because data is transpose.
        // Needed because SWIG alwasy passes in in row major order.
        matrix = cml::matrix_alloc<T, CblasColMajor>(cols, rows);
        cml::matrix_memcpy(&matrix, data);
        cudaDeviceSynchronize();
        CUDA_CHECK_ERR();
    }

    /* Standard dense matrix multiplication. */
    void forward_eval() {
        cml::blas_gemv(hdl, CUBLAS_OP_N, 1., &matrix,
            &this->input_data, 0., &this->output_data);
        cudaDeviceSynchronize();
        CUDA_CHECK_ERR();
    }

    void adjoint_eval() {
        cml::blas_gemv(hdl, CUBLAS_OP_T, 1., &matrix,
            &this->output_data, 0., &this->input_data);
        cudaDeviceSynchronize();
        CUDA_CHECK_ERR();
    }

};

template <class T>
class DenseMatMatMul : public DenseMatVecMul<T> {
public:
    cml::matrix<T, CblasColMajor> input_matrix;
    cml::matrix<T, CblasColMajor> output_matrix;
    int M, N, K;

    int forward_evals = 0;
    int adjoint_evals = 0;
    double total_forward_mul_time = 0;
    double total_adjoint_mul_time = 0;

    void alloc_data() {
        FAO<T>::alloc_data();
        M = static_cast<int>(this->output_sizes[0][0]);
        N = static_cast<int>(this->output_sizes[0][1]);
        K = static_cast<int>(this->input_sizes[0][0]);
        input_matrix = cml::matrix_init<T, CblasColMajor>(K, N,
            this->input_data.data);
        output_matrix = cml::matrix_init<T, CblasColMajor>(M, N,
            this->output_data.data);
    }

    void free_data() {
        printf("n=%u, avg_forward_mul=%e\n", N*K,
            total_forward_mul_time/forward_evals);
        printf("n=%u, avg_adjoint_mul=%e\n", M*N,
            total_adjoint_mul_time/adjoint_evals);
        DenseMatVecMul<T>::free_data();
    }

    /* Standard dense matrix matrix multiplication AX = Y.
       A in R^{M x K}, X in R^{K x N}, Y in R^{M x N}
    */
    void forward_eval() {
        // cblas_dgemm(CblasColMajor, CblasTrans,
        //             CblasNoTrans, M, N,
        //             K, 1, matrix.data,
        //             K, input_data.data, K,
        //             0, output_data.data, M);
        forward_evals++;
        double t = timer<double>();
        cml::blas_gemm(this->hdl, CUBLAS_OP_N, CUBLAS_OP_N,
                       static_cast<T>(1.0), &this->matrix, &input_matrix,
                       static_cast<T>(0.0), &output_matrix);
        cudaDeviceSynchronize();
        CUDA_CHECK_ERR();
        total_forward_mul_time += timer<double>() - t;
    }

    void adjoint_eval() {
        adjoint_evals++;
        double t = timer<double>();
        // cblas_dgemm(CblasColMajor, CblasNoTrans,
        //             CblasNoTrans, M, N,
        //             K, 1, matrix.data,
        //             M, output_data.data, K,
        //             0, input_data.data, M);
        cml::blas_gemm(this->hdl, CUBLAS_OP_T, CUBLAS_OP_N,
                       static_cast<T>(1.0), &this->matrix, &output_matrix,
                       static_cast<T>(0.0), &input_matrix);
        cudaDeviceSynchronize();
        CUDA_CHECK_ERR();
        total_adjoint_mul_time += timer<double>() - t;
    }

};

template <class T>
class DenseMatMatRMul : public DenseMatMatMul<T> {
public:

    void alloc_data() {
        FAO<T>::alloc_data();
        this->M = static_cast<int>(this->output_sizes[0][0]);
        this->N = static_cast<int>(this->output_sizes[0][1]);
        this->K = static_cast<int>(this->input_sizes[0][1]);
        this->input_matrix = cml::matrix_init<T, CblasColMajor>(this->M, this->K,
            this->input_data.data);
        this->output_matrix = cml::matrix_init<T, CblasColMajor>(this->M, this->N,
            this->output_data.data);
    }

    void free_data() {
        printf("n=%u, avg_forward_rmul=%e\n", this->M*this->K,
            this->total_forward_mul_time/this->forward_evals);
        printf("n=%u, avg_adjoint_rmul=%e\n", this->M*this->N,
            this->total_adjoint_mul_time/this->adjoint_evals);
        DenseMatVecMul<T>::free_data();
    }

    /* Standard dense matrix matrix multiplication XA = Y.
       X in R^{M x K}, A in R^{K x N}, Y in R^{M x N}
    */
    void forward_eval() {
        this->forward_evals++;
        double t = timer<double>();
        cml::blas_gemm(this->hdl, CUBLAS_OP_N, CUBLAS_OP_N,
                       static_cast<T>(1.0), &this->input_matrix, &this->matrix,
                       static_cast<T>(0.0), &this->output_matrix);
        cudaDeviceSynchronize();
        CUDA_CHECK_ERR();
        this->total_forward_mul_time += timer<double>() - t;
    }

    /* Standard dense matrix matrix multiplication YA^T = X.
       Y in R^{M x N}, A^T in R^{N x K}, X in R^{M x K}
    */
    void adjoint_eval() {
        this->adjoint_evals++;
        double t = timer<double>();
        cml::blas_gemm(this->hdl, CUBLAS_OP_N, CUBLAS_OP_T,
                       static_cast<T>(1.0), &this->output_matrix, &this->matrix,
                       static_cast<T>(0.0), &this->input_matrix);
        cudaDeviceSynchronize();
        CUDA_CHECK_ERR();
        this->total_adjoint_mul_time += timer<double>() - t;
    }
};

cusparseOperation_t OpToCusparseOp(char trans) {
  ASSERT(trans == 'n' || trans == 'N' || trans == 't' || trans == 'T');
  return (trans == 'n' || trans == 'N')
      ? CUSPARSE_OPERATION_NON_TRANSPOSE : CUSPARSE_OPERATION_TRANSPOSE;
}

template <class T>
class SparseMatVecMul : public FAO<T> {
public:
    cml::spmat<T, int, CblasRowMajor> spmatrix;
    cusparseHandle_t hdl;
    cusparseMatDescr_t descr;
    // TODO should I store the transpose separately?
    // gsl::spmat<T, CblasRowMajor> spmatrix_trans;

    SparseMatVecMul() {
        cusparseCreate(&hdl);
        cusparseCreateMatDescr(&descr);
        CUDA_CHECK_ERR();
    }

    ~SparseMatVecMul() {
        cusparseDestroy(hdl);
        cusparseDestroyMatDescr(descr);
        CUDA_CHECK_ERR();
    }

    void set_spmatrix_data(T *data, int data_len, int *ptrs,
                           int ptrs_len, int *indices, int idx_len,
                           int rows, int cols) {

        assert(rows_len == data_len && cols_len == data_len);
        spmatrix = cml::spmat_alloc<T, int, CblasRowMajor>(rows, cols, data_len);
        cml::spmat_memcpy(hdl, &spmatrix, data, indices, ptrs);
        cudaDeviceSynchronize();
        CUDA_CHECK_ERR();
    }

    /* Standard sparse matrix multiplication. */
    void forward_eval() {
        cml::spblas_gemv(hdl, OpToCusparseOp('n'), descr, 1.,
              &spmatrix, &this->input_data, 0., &this->output_data);
        cudaDeviceSynchronize();
        CUDA_CHECK_ERR();
    }

    void adjoint_eval() {
        cml::spblas_gemv(hdl, OpToCusparseOp('t'), descr, 1.,
              &spmatrix, &this->output_data, 0., &this->input_data);
        cudaDeviceSynchronize();
        CUDA_CHECK_ERR();
    }

};

// TODO
// class SparseMatMatMul : public SparseMatVecMul {
// };

template <class T>
class ScalarMul : public FAO<T> {
public:
    T scalar;
      cublasHandle_t hdl;

    ScalarMul() {
        cublasCreate(&hdl);
    }

    ~ScalarMul() {
        cublasDestroy(hdl);
        CUDA_CHECK_ERR();
    }

    /* Get the scalar value. */
    virtual T get_scalar() {
        return scalar;
    }

    /* Operation is in-place. */
    bool is_inplace() {
        return true;
    }

    /* Scale the input/output. */
    void forward_eval() {
        cml::blas_scal(hdl, get_scalar(), &this->input_data);
        cudaDeviceSynchronize();
        CUDA_CHECK_ERR();
    }

    void adjoint_eval() {
        forward_eval();
    }
};

template <class T>
class Neg : public ScalarMul<T> {
public:
    T get_scalar() {
        return -1;
    }
};

template <class T>
class Sum: public FAO<T> {
public:
     cublasHandle_t hdl;

    Sum() {
        cublasCreate(&hdl);
    }

    ~Sum() {
        cublasDestroy(hdl);
        CUDA_CHECK_ERR();
    }

    /* Sum the inputs. */
    void forward_eval() {
        // printf("Sum forward eval\n");
        forward_eval_base(this->input_data, this->output_data,
            this->input_sizes);
    }

    /* Factored out so usable by Sum and Copy. */
    void forward_eval_base(cml::vector<T> input_data,
                           cml::vector<T> output_data,
                           std::vector<std::vector<size_t> > input_sizes) {
        size_t elem_size = output_data.size;
        cml::vector_subvec_memcpy(&output_data, 0, &input_data, 0, elem_size);
        cudaDeviceSynchronize();
        for (size_t i=1; i < input_sizes.size(); ++i) {
            auto subvec = cml::vector_subvector(&input_data,
                i*elem_size, elem_size);
            cml::blas_axpy(hdl, 1, &subvec, &output_data);
            // cudaDeviceSynchronize();
            // CUDA_CHECK_ERR();
        }
        cudaDeviceSynchronize();
        CUDA_CHECK_ERR();
    }

    /* Copy the input. */
    void adjoint_eval() {
        adjoint_eval_base(this->output_data, this->input_data,
            this->input_sizes);
    }

    /* Factored out so usable by Sum and Copy. */
    void adjoint_eval_base(cml::vector<T> input_data,
                           cml::vector<T> output_data,
                           std::vector<std::vector<size_t> > output_sizes) {
        size_t elem_size = input_data.size;
        for (size_t i=0; i < output_sizes.size(); ++i) {
            cml::vector_subvec_memcpy<T>(&output_data, i*elem_size,
                                              &input_data, 0, elem_size);
        }
        cudaDeviceSynchronize();
        CUDA_CHECK_ERR();
    }
};

template <class T>
class Copy : public Sum<T> {
/* Adjoint of Sum. */
public:
    /* Copy the inputs. */
    void forward_eval() {
        Sum<T>::adjoint_eval_base(this->input_data, this->output_data,
            this->output_sizes);
    }

    /* Sum the inputs. */
    void adjoint_eval() {
        Sum<T>::forward_eval_base(this->output_data, this->input_data,
            this->output_sizes);
    }
};

template <class T>
class Vstack : public FAO<T> {
public:
    /* Operation is in-place. */
    bool is_inplace() {
        return true;
    }
};

template <class T>
class Split : public Vstack<T> {
    /* Adjoint of vstack. */
};

template <class T>
class Reshape : public FAO<T> {
public:
    /* Operation is in-place. */
    bool is_inplace() {
        return true;
    }
};

#ifndef SWIG
// Operator for multiplying complex numbers in convolution.
template <typename T>
struct MulComplexF {
  T divisor, conj_mul;
  MulComplexF(T divisor, T conj_mul) :
    divisor(divisor), conj_mul(conj_mul) { }
  __host__ __device__ thrust::complex<T> operator()(
        thrust::complex<T> x,
        thrust::complex<T> y
    ) {
    T real = x.real()*y.real() - conj_mul*x.imag()*y.imag();
    T imag = x.real()*y.imag() + conj_mul*x.imag()*y.real();
    return thrust::complex<T>(real/divisor, imag/divisor);
  }
};
#endif

template <class T>
class ConvBase : public FAO<T> {
public:

    cml::vector<T> kernel;
    size_t input_len;
    size_t kernel_len;
    size_t padded_len;
    size_t cplx_len;
    // Actually fftw_complex.
    cml::vector<T> kernel_fft;
    cml::vector<T> r2c_out;

    cml::vector<T> input_padding;
    cufftHandle forward_fft_plan;
    cufftHandle forward_ifft_plan;
    cufftHandle adjoint_fft_plan;
    cufftHandle adjoint_ifft_plan;

    /* Operation is in-place. */
    bool is_inplace() {
        return true;
    }

    void alloc_data_base() {
        input_len = this->get_length(this->input_sizes);
        padded_len = this->get_length(this->output_sizes);
        // R2C padded_len transform has this size output.
        cplx_len = 2*(padded_len/2 + 1);
        this->input_data = cml::vector_calloc<T>(padded_len);
        // Input and output can be same array.
        this->output_data = this->input_data;
        /* Isolate extra part of input. */
        input_padding = cml::vector_view_array<T>(
        this->input_data.data + input_len, padded_len - input_len);
        // Actually complex.
        kernel_fft = cml::vector_calloc<T>(2*cplx_len);
        r2c_out = cml::vector_calloc<T>(2*cplx_len);
    }

    void free_data() {
        cufftDestroy(forward_fft_plan);
        cufftDestroy(forward_ifft_plan);
        cufftDestroy(adjoint_fft_plan);
        cufftDestroy(adjoint_ifft_plan);
        vector_free(&kernel_fft);
        vector_free(&r2c_out);
        // fftw_cleanup();
        FAO<T>::free_data();
    }

    void set_conv_data(T *kernel, int kernel_len) {
        this->kernel = cml::vector_alloc<T>(kernel_len);
        this->kernel_len = kernel_len;
        cml::vector_memcpy<T>(&this->kernel, kernel);
        cudaDeviceSynchronize();
        CUDA_CHECK_ERR();
    }


    // /* Functor for multiplying two complex numbers and dividing by n. */
    // struct complex_mul
    // {
    //   const double n;

    //   complex_mul(double _n) : n(_n) {}

    //   __host__ __device__
    //   fftw_complex operator()(const fftw_complex& x, const fftw_complex& y) const
    //   {
    //     fftw_complex result;
    //     result[0] = (x[0]*y[0] - x[1]*y[1])/n;
    //     result[1] = (x[0]*y[1] + x[1]*y[0])/n;
    //     return result;
    //   }
    // };

    /* Multiply kernel_fft and output.
       Divide by n because fftw doesn't.
       Writes to output.
    */
    void multiply_fft(cml::vector<T>& kernel_fft, cml::vector<T>& output,
        bool forward) {
        T divisor = static_cast<T>(padded_len);
        T conj_mul = forward ? 1.0 : -1.0;
        cml::strided_range<thrust::device_ptr<thrust::complex<T> > > idx_a(
            thrust::device_pointer_cast((thrust::complex<T> *) kernel_fft.data),
            thrust::device_pointer_cast((thrust::complex<T> *) kernel_fft.data + cplx_len), 1);
        cml::strided_range<thrust::device_ptr<thrust::complex<T> > > idx_b(
            thrust::device_pointer_cast((thrust::complex<T> *) output.data),
            thrust::device_pointer_cast((thrust::complex<T> *) output.data + cplx_len), 1);
        thrust::transform(idx_a.begin(), idx_a.end(), idx_b.begin(), idx_b.begin(),
            MulComplexF<T>(divisor, conj_mul));
    }

    /* Fill out the input padding with zeros. */
    void zero_pad_input() {
        cml::vector_scale<T>(&input_padding, static_cast<T>(0.0));
    }
};

#ifdef SWIG
%template(FAOd) FAO<double>;
%template(FAOf) FAO<float>;
%template(ConvBased) ConvBase<double>;
%template(ConvBasef) ConvBase<float>;
#endif

// TODO TODO zero pad input so is power of 2. Much slower if weird number.

class Convd: public ConvBase<double> {
public:
    void alloc_data() {
        this->alloc_data_base();
        /* kernel_fft is DFT(padded kernel). */
        /* Must copy because FFTW destroys input array. */
        // TODO alignment of kernel_fft!
        cml::vector<double> input_start = cml::vector_view_array<double>(
            this->input_data.data, kernel_len);
        cml::vector_memcpy<double>(&input_start, &kernel);
        cudaDeviceSynchronize();
        // Done with kernel.
        vector_free(&kernel);

        cufftHandle plan;
        cufftPlan1d(&plan, padded_len, CUFFT_D2Z, 1);
        cufftExecD2Z(plan,
            (cufftDoubleReal *) this->input_data.data,
            (cufftDoubleComplex *) kernel_fft.data);
        cufftDestroy(plan);
         /* Initialize the plans for forward_eval. */
        cufftPlan1d(&forward_fft_plan, padded_len, CUFFT_D2Z, 1);
        cufftPlan1d(&forward_ifft_plan, padded_len, CUFFT_Z2D, 1);
        cufftPlan1d(&adjoint_fft_plan, padded_len, CUFFT_D2Z, 1);
        cufftPlan1d(&adjoint_ifft_plan, padded_len, CUFFT_Z2D, 1);
        cudaDeviceSynchronize();
        CUDA_CHECK_ERR();
    }

    /* Column convolution. */
    void forward_eval() {
        this->zero_pad_input();
        cufftExecD2Z(forward_fft_plan,
           (cufftDoubleReal *) this->input_data.data,
           (cufftDoubleComplex *) r2c_out.data);
        this->multiply_fft(kernel_fft, r2c_out, true);
        cufftExecZ2D(forward_ifft_plan,
           (cufftDoubleComplex *) r2c_out.data,
           (cufftDoubleReal *) this->output_data.data);
        cudaDeviceSynchronize();
        CUDA_CHECK_ERR();
    }

    /* Row convolution. */
    void adjoint_eval() {
        cufftExecD2Z(adjoint_fft_plan,
           (cufftDoubleReal *) this->output_data.data,
           (cufftDoubleComplex *) r2c_out.data);
        this->multiply_fft(kernel_fft, r2c_out, false);
        cufftExecZ2D(adjoint_ifft_plan,
           (cufftDoubleComplex *) r2c_out.data,
           (cufftDoubleReal *) this->input_data.data);
        // TODO do this? zero_pad_input();
        cudaDeviceSynchronize();
        CUDA_CHECK_ERR();
    }
};

class Convf: public ConvBase<float> {
public:
    // Timing info.
    int forward_evals = 0;
    int adjoint_evals = 0;
    double total_forward_r2c_time = 0;
    double total_adjoint_r2c_time = 0;
    double total_forward_mul_time = 0;
    double total_adjoint_mul_time = 0;

    void alloc_data() {
        this->alloc_data_base();
        /* kernel_fft is DFT(padded kernel). */
        /* Must copy because FFTW destroys input array. */
        // TODO alignment of kernel_fft!
        cml::vector<float> input_start = cml::vector_view_array<float>(
            this->input_data.data, kernel_len);
        cml::vector_memcpy<float>(&input_start, &kernel);
        cudaDeviceSynchronize();
        // Done with kernel.
        vector_free(&kernel);

        cufftHandle plan;
        cufftPlan1d(&plan, padded_len, CUFFT_R2C, 1);
        cufftExecR2C(plan,
            (cufftReal *) this->input_data.data,
            (cufftComplex *) kernel_fft.data);
        cufftDestroy(plan);

         /* Initialize the plans for forward_eval. */
        cufftPlan1d(&forward_fft_plan, padded_len, CUFFT_R2C, 1);
        cufftPlan1d(&forward_ifft_plan, padded_len, CUFFT_C2R, 1);
        cufftPlan1d(&adjoint_fft_plan, padded_len, CUFFT_R2C, 1);
        cufftPlan1d(&adjoint_ifft_plan, padded_len, CUFFT_C2R, 1);
        cudaDeviceSynchronize();
        CUDA_CHECK_ERR();
    }

    // For timing purposes.
    void free_data() {
        printf("n=%u, avg_forward_r2c=%e\n", input_len,
            total_forward_r2c_time/forward_evals);
        printf("n=%u, avg_adjoint_r2c=%e\n", input_len,
            total_adjoint_r2c_time/adjoint_evals);
        printf("n=%u, avg_forward_mul=%e\n", input_len,
            total_forward_mul_time/forward_evals);
        printf("n=%u, avg_adjoint_mul=%e\n", input_len,
            total_adjoint_mul_time/adjoint_evals);
        ConvBase<float>::free_data();
    }

    /* Column convolution. */
    void forward_eval() {
        forward_evals++;
        double t = timer<double>();
        this->zero_pad_input();
        cudaDeviceSynchronize();
        // printf("T_zero_pad = %e\n", timer<double>() - t);
        t = timer<double>();
        cufftExecR2C(forward_fft_plan,
           (cufftReal *) this->input_data.data,
           (cufftComplex *) r2c_out.data);
        cudaDeviceSynchronize();
        double r2c_time = timer<double>() - t;
        total_forward_r2c_time += r2c_time;
        // printf("T_exec_r2c = %e\n", r2c_time);
        t = timer<double>();
        this->multiply_fft(kernel_fft, r2c_out, true);
        cudaDeviceSynchronize();
        double mul_time = timer<double>() - t;
        total_forward_mul_time += mul_time;
        // printf("T_multiply_fft = %e\n", mul_time);
        t = timer<double>();
        cufftExecC2R(forward_ifft_plan,
           (cufftComplex *) r2c_out.data,
           (cufftReal *) this->output_data.data);
        cudaDeviceSynchronize();
        // printf("T_exec_c2r = %e\n", timer<double>() - t);
        CUDA_CHECK_ERR();
    }

    /* Row convolution. */
    void adjoint_eval() {
        adjoint_evals++;
        double t = timer<double>();
        cufftExecR2C(adjoint_fft_plan,
           (cufftReal *) this->output_data.data,
           (cufftComplex *) r2c_out.data);
        cudaDeviceSynchronize();
        double r2c_time = timer<double>() - t;
        total_adjoint_r2c_time += r2c_time;
        // printf("T_exec_r2c = %e\n", r2c_time);
        t = timer<double>();
        this->multiply_fft(kernel_fft, r2c_out, false);
        cudaDeviceSynchronize();
        double mul_time = timer<double>() - t;
        total_adjoint_mul_time += mul_time;
        // printf("T_multiply_fft = %e\n", mul_time);
        t = timer<double>();
        cufftExecC2R(adjoint_ifft_plan,
           (cufftComplex *) r2c_out.data,
           (cufftReal *) this->input_data.data);
        cudaDeviceSynchronize();
        // printf("T_exec_c2r = %e\n", timer<double>() - t);
        // TODO do this? zero_pad_input();
        CUDA_CHECK_ERR();
    }
};

template <class T>
class Conv2DBase : public FAO<T> {
public:

    size_t input_rows;
    size_t input_cols;
    size_t kernel_rows;
    size_t kernel_cols;
    size_t padded_rows;
    size_t padded_cols;
    size_t cplx_rows;

    cml::matrix<T, CblasColMajor> kernel;
    cml::matrix<T, CblasColMajor> input_matrix;
    cml::matrix<T, CblasColMajor> output_matrix;
    // The section of the output matrix used to store the input
    // for zero-padding reasons.
    cml::matrix<T, CblasColMajor> output_submatrix;
    // Actually fftw_complex.
    cml::matrix<T, CblasColMajor> kernel_fft;
    cml::matrix<T, CblasColMajor> r2c_out;

    cufftHandle forward_fft_plan;
    cufftHandle forward_ifft_plan;
    cufftHandle adjoint_fft_plan;
    cufftHandle adjoint_ifft_plan;

    /* Operation is in-place. */
    bool is_inplace() {
        return true;
    }

    // cuFFT assumes row major.
    // but we can get around this by reversing order of dimensions in plan.
    void alloc_data_base() {
        CUDA_CHECK_ERR();
        input_rows = this->input_sizes[0][0];
        input_cols = this->input_sizes[0][1];
        padded_rows = this->output_sizes[0][0];
        padded_cols = this->output_sizes[0][1];
        printf("input_rows = %d, input_cols = %d\n", input_rows, input_cols);
        printf("padded_rows = %d, padded_cols = %d\n", padded_rows, padded_cols);
        input_matrix = cml::matrix_calloc<T, CblasColMajor>(input_rows,
                                                            input_cols);
        this->input_data = cml::vector_view_array<T>(input_matrix.data,
                                                     input_rows*input_cols);
        // Input and output can be same array, except FAO_DAG would
        // write to the wrong part of the input.
        output_matrix = cml::matrix_calloc<T, CblasColMajor>(padded_rows,
                                                             padded_cols);
        this->output_data = cml::vector_view_array<T>(output_matrix.data,
                                                      padded_rows*padded_cols);
        output_submatrix = cml::matrix_submatrix<T, CblasColMajor>(
            &output_matrix, 0u, 0u, input_rows, input_cols);
        cplx_rows = 2*(padded_rows/2 + 1);
        // Actually complex.
        kernel_fft = cml::matrix_calloc<T, CblasColMajor>(cplx_rows, padded_cols);
        r2c_out = cml::matrix_calloc<T, CblasColMajor>(cplx_rows, padded_cols);
    }

    void free_data() {
        cufftDestroy(forward_fft_plan);
        cufftDestroy(forward_ifft_plan);
        cufftDestroy(adjoint_fft_plan);
        cufftDestroy(adjoint_ifft_plan);
        matrix_free(&kernel_fft);
        matrix_free(&r2c_out);
        // fftw_cleanup();
        FAO<T>::free_data();
    }

    void set_conv_data(T *kernel, int kernel_rows, int kernel_cols) {
        this->kernel = cml::matrix_alloc<T, CblasColMajor>(kernel_rows,
                                                           kernel_cols);
        this->kernel_rows = kernel_rows;
        this->kernel_cols = kernel_cols;
        cml::matrix_memcpy(&this->kernel, kernel);
        cudaDeviceSynchronize();
        CUDA_CHECK_ERR();
    }

    /* Multiply kernel_fft and output.
       Divide by n because fftw doesn't.
       Writes to output.
    */
    void multiply_fft(cml::matrix<T, CblasColMajor>& kernel_fft,
                      cml::matrix<T, CblasColMajor>& output,
                      bool forward) {
        T divisor = static_cast<T>(padded_rows*padded_cols);
        size_t padded_len = cplx_rows*padded_cols;
        T conj_mul = forward ? 1.0 : -1.0;
        cml::strided_range<thrust::device_ptr<thrust::complex<T> > > idx_a(
            thrust::device_pointer_cast((thrust::complex<T> *) kernel_fft.data),
            thrust::device_pointer_cast((thrust::complex<T> *) kernel_fft.data + padded_len), 1);
        cml::strided_range<thrust::device_ptr<thrust::complex<T> > > idx_b(
            thrust::device_pointer_cast((thrust::complex<T> *) output.data),
            thrust::device_pointer_cast((thrust::complex<T> *) output.data + padded_len), 1);
        thrust::transform(idx_a.begin(), idx_a.end(), idx_b.begin(), idx_b.begin(),
            MulComplexF<T>(divisor, conj_mul));
    }

    /* Zero out the output matrix, then copy in input. */
    void zero_pad_and_copy_input() {
        cml::matrix_scale<T, CblasColMajor>(&output_matrix,
                                            static_cast<T>(0.0));
        cml::matrix_memcpy<T, CblasColMajor>(&output_submatrix,
                                             &input_matrix);
    }
};

#ifdef SWIG
%template(Conv2DBased) Conv2DBase<double>;
%template(Conv2DBasef) Conv2DBase<float>;
#endif

class Conv2Df: public Conv2DBase<float> {
public:
    // Timing info.
    int forward_evals = 0;
    int adjoint_evals = 0;
    double total_forward_r2c_time = 0;
    double total_adjoint_r2c_time = 0;
    double total_forward_mul_time = 0;
    double total_adjoint_mul_time = 0;

    void alloc_data() {
        this->alloc_data_base();
        /* kernel_fft is DFT(padded kernel). */
        /* Must copy because FFTW destroys input array. */
        // TODO alignment of kernel_fft!
        cml::matrix<float, CblasColMajor> output_start =
            cml::matrix_submatrix<float, CblasColMajor>(&output_matrix,
                0u, 0u, kernel_rows, kernel_cols);
        cml::matrix_memcpy<float, CblasColMajor>(&output_start, &kernel);
        cudaDeviceSynchronize();
        CUDA_CHECK_ERR();
        // Done with kernel.
        matrix_free(&kernel);

        cufftHandle plan;
        cufftPlan2d(&plan, padded_cols, padded_rows, CUFFT_R2C);
        cufftExecR2C(plan,
            (cufftReal *) output_matrix.data,
            (cufftComplex *) kernel_fft.data);
        cufftDestroy(plan);

        // cml::vector<float> kernel_fft_vec = cml::vector_view_array(kernel_fft.data,
        //     cplx_rows*padded_cols);
        // printf("-1. kernel_fft is nan = %i\n", cml::vector_any_isnan(&kernel_fft_vec));

         /* Initialize the plans for forward_eval. */
        cufftPlan2d(&forward_fft_plan, padded_cols, padded_rows, CUFFT_R2C);
        cufftPlan2d(&forward_ifft_plan, padded_cols, padded_rows, CUFFT_C2R);
        cufftPlan2d(&adjoint_fft_plan, padded_cols, padded_rows, CUFFT_R2C);
        cufftPlan2d(&adjoint_ifft_plan, padded_cols, padded_rows, CUFFT_C2R);
        cudaDeviceSynchronize();
        CUDA_CHECK_ERR();
    }

    // For timing purposes.
    void free_data() {
        printf("n=%u, avg_forward_r2c=%e\n", padded_rows*padded_cols,
            total_forward_r2c_time/forward_evals);
        printf("n=%u, avg_adjoint_r2c=%e\n", padded_rows*padded_cols,
            total_adjoint_r2c_time/adjoint_evals);
        printf("n=%u, avg_forward_mul=%e\n", padded_rows*padded_cols,
            total_forward_mul_time/forward_evals);
        printf("n=%u, avg_adjoint_mul=%e\n", padded_rows*padded_cols,
            total_adjoint_mul_time/adjoint_evals);
        Conv2DBase<float>::free_data();
    }

    /* Column convolution. */
    void forward_eval() {
        forward_evals++;
        double t = timer<double>();
        this->zero_pad_and_copy_input();
        cudaDeviceSynchronize();
        // printf("1. input is nan = %i\n", cml::vector_any_isnan(&input_data));
        // printf("1. output is nan = %i\n", cml::vector_any_isnan(&output_data));
        // // printf("T_zero_pad = %e\n", timer<double>() - t);
        t = timer<double>();
        cufftExecR2C(forward_fft_plan,
           (cufftReal *) this->output_data.data,
           (cufftComplex *) r2c_out.data);
        cudaDeviceSynchronize();
        double r2c_time = timer<double>() - t;
        total_forward_r2c_time += r2c_time;

        // cml::vector<float> r2c_vec = cml::vector_view_array(r2c_out.data,
        //     cplx_rows*padded_cols);
        // cml::vector<float> kernel_fft_vec = cml::vector_view_array(kernel_fft.data,
        //     cplx_rows*padded_cols);
        // printf("2. kernel_fft is nan = %i\n", cml::vector_any_isnan(&kernel_fft_vec));
        // printf("2. r2c_out is nan = %i\n", cml::vector_any_isnan(&r2c_vec));
        // printf("2. output is nan = %i\n", cml::vector_any_isnan(&output_data));
        // printf("T_exec_r2c = %e\n", r2c_time);
        t = timer<double>();
        this->multiply_fft(kernel_fft, r2c_out, true);
        cudaDeviceSynchronize();
        double mul_time = timer<double>() - t;
        total_forward_mul_time += mul_time;
        // printf("3. r2c_out is nan = %i\n", cml::vector_any_isnan(&r2c_vec));
        // printf("T_multiply_fft = %e\n", mul_time);
        t = timer<double>();
        cufftExecC2R(forward_ifft_plan,
           (cufftComplex *) r2c_out.data,
           (cufftReal *) this->output_data.data);
        cudaDeviceSynchronize();
        CUDA_CHECK_ERR();
        // printf("4. r2c_out is nan = %i\n", cml::vector_any_isnan(&r2c_vec));
        // printf("4. output is nan = %i\n", cml::vector_any_isnan(&output_data));
        // printf("T_exec_c2r = %e\n", timer<double>() - t);
    }

    /* Row convolution. */
    void adjoint_eval() {
        adjoint_evals++;
        double t = timer<double>();
        cufftExecR2C(adjoint_fft_plan,
           (cufftReal *) output_data.data,
           (cufftComplex *) r2c_out.data);
        cudaDeviceSynchronize();
        double r2c_time = timer<double>() - t;
        total_adjoint_r2c_time += r2c_time;
        // printf("T_exec_r2c = %e\n", r2c_time);
        t = timer<double>();
        this->multiply_fft(kernel_fft, r2c_out, false);
        cudaDeviceSynchronize();
        double mul_time = timer<double>() - t;
        total_adjoint_mul_time += mul_time;
        // printf("T_multiply_fft = %e\n", mul_time);
        t = timer<double>();
        cufftExecC2R(adjoint_ifft_plan,
           (cufftComplex *) r2c_out.data,
           (cufftReal *) output_data.data);
        cudaDeviceSynchronize();
        // printf("T_exec_c2r = %e\n", timer<double>() - t);
        // Copy output submatrix to input.
        cml::matrix_memcpy<float, CblasColMajor>(&input_matrix,
                                                 &output_submatrix);
        cudaDeviceSynchronize();
        CUDA_CHECK_ERR();
    }
};

// TODO Conv2Dd

#ifdef SWIG
%template(NoOpd) NoOp<double>;
%template(NoOpf) NoOp<float>;
%template(DenseMatVecMuld) DenseMatVecMul<double>;
%template(DenseMatVecMulf) DenseMatVecMul<float>;
%template(SparseMatVecMuld) SparseMatVecMul<double>;
%template(SparseMatVecMulf) SparseMatVecMul<float>;
%template(DenseMatMatMuld) DenseMatMatMul<double>;
%template(DenseMatMatMulf) DenseMatMatMul<float>;
%template(DenseMatMatRMuld) DenseMatMatRMul<double>;
%template(DenseMatMatRMulf) DenseMatMatRMul<float>;
%template(ScalarMuld) ScalarMul<double>;
%template(ScalarMulf) ScalarMul<float>;
%template(Negd) Neg<double>;
%template(Negf) Neg<float>;
%template(Sumd) Sum<double>;
%template(Sumf) Sum<float>;
%template(Copyd) Copy<double>;
%template(Copyf) Copy<float>;
%template(Vstackd) Vstack<double>;
%template(Vstackf) Vstack<float>;
%template(Splitd) Split<double>;
%template(Splitf) Split<float>;
%template(Reshaped) Reshape<double>;
%template(Reshapef) Reshape<float>;
#endif

#endif
