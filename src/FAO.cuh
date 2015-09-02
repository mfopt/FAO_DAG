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
    cml::matrix<T, CblasRowMajor> matrix;
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

    void set_matrix_data(T* data, int rows, int cols) {
        matrix = cml::matrix_alloc<T, CblasRowMajor>(rows, cols);
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

// class DenseMatMatMul : public DenseMatVecMul {
// public:
//
//     /* Standard dense matrix matrix multiplication. */
//     void forward_eval() {
//         int M = static_cast<int>(output_sizes[0][0]);
//         int N = static_cast<int>(input_sizes[0][1]);
//         int K = static_cast<int>(input_sizes[0][0]);
//         cblas_dgemm(CblasColMajor, CblasTrans,
//                     CblasNoTrans, M, N,
//                     K, 1, matrix.data,
//                     K, input_data.data, K,
//                     0, output_data.data, M);
//     }
//
//     void adjoint_eval() {
//         int M = static_cast<int>(input_sizes[0][0]);
//         int N = static_cast<int>(output_sizes[0][1]);
//         int K = static_cast<int>(output_sizes[0][0]);
//         cblas_dgemm(CblasColMajor, CblasNoTrans,
//                     CblasNoTrans, M, N,
//                     K, 1, matrix.data,
//                     M, output_data.data, K,
//                     0, input_data.data, M);
//     }
//
// };

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
        // double *input = new double[input_data.size];
        // cudaMemcpy(input, input_data.data, input_data.size * sizeof(double), cudaMemcpyDeviceToHost);
        // printf("input_data.size = %zu\n", input_data.size);
        // for (int i = 0; i < input_data.size; ++i)
        //    printf("input[%i] = %e\n", i, input[i]);
        // cudaDeviceSynchronize();
        // CUDA_CHECK_ERR();
        // delete [] input;
        // cudaDeviceSynchronize();
        // CUDA_CHECK_ERR();
        // double *output = new double[output_data.size];
        // cudaMemcpy(output, output_data.data, output_data.size * sizeof(double), cudaMemcpyDeviceToHost);
        // cudaDeviceSynchronize();
        // CUDA_CHECK_ERR();
        // for (int i = 0; i < output_data.size; ++i)
        //    printf("output[%i] = %e\n", i, output[i]);
        // delete [] output;
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

// TODO use cuFFT HERE
template <class T>
class Conv : public FAO<T> {
public:

    cml::vector<T> kernel;
    size_t input_len;
    size_t kernel_len;
    size_t padded_len;
    // Actually fftw_complex.
    cml::vector<T> kernel_fft;
    cml::vector<T> rev_kernel_fft;
    cml::vector<T> r2c_out;

    cml::vector<T> input_padding;
    cufftHandle forward_fft_plan;
    cufftHandle forward_ifft_plan;
    cufftHandle adjoint_fft_plan;
    cufftHandle adjoint_ifft_plan;

    cublasHandle_t hdl;

    void alloc_data() {
        input_len = this->get_length(this->input_sizes);
        padded_len = this->get_length(this->output_sizes);
        // TODO could use fftw_alloc here.
        this->input_data = cml::vector_calloc<T>(padded_len);
           this->output_data = cml::vector_calloc<T>(padded_len);
        /* Isolate extra part of input. */
        input_padding = cml::vector_view_array<T>(
            this->input_data.data + input_len, padded_len - input_len);
        // Actually complex.
        kernel_fft = cml::vector_calloc<T>(2*padded_len);
        rev_kernel_fft = cml::vector_calloc<T>(2*padded_len);
        r2c_out = cml::vector_calloc<T>(2*padded_len);

        /* kernel_fft is DFT(padded kernel). */
        /* Must copy because FFTW destroys input array. */
        // TODO alignment of kernel_fft!
        cml::vector<T> input_start = cml::vector_view_array<T>(
            this->input_data.data, kernel_len);
        cml::vector_memcpy<T>(&input_start, &kernel);
        cudaDeviceSynchronize();
        cufftHandle plan;
        cufftPlan1d(&plan, padded_len, CUFFT_D2Z, 1);
        cufftExecD2Z(plan,
            (cufftDoubleReal *) this->input_data.data,
            (cufftDoubleComplex *) kernel_fft.data);
        cufftDestroy(plan);
        T nrm = cml::blas_nrm2<T>(hdl, &kernel_fft);
        printf("kernel_fft nrm = %e\n", nrm);
         /* rev_kernel_fft is conj(DFT(padded kernel))=IDFT(padded kernel). */
         // TODO parallelize.
        cml::vector<T> imag_part(rev_kernel_fft.data + 1, padded_len, 2);
        cml::vector_memcpy<T>(&rev_kernel_fft, &kernel_fft);
        cudaDeviceSynchronize();
        CUDA_CHECK_ERR();
        cml::vector_scale(&imag_part, static_cast<T>(-1.0));
        // for (size_t i=0; i < padded_len; ++i) {
         //       rev_kernel_fft[i][0] = kernel_fft[i][0];
         //       rev_kernel_fft[i][1] = -kernel_fft[i][1];
         // }

         /* Initialize the plans for forward_eval. */
         // TODO also FFTW_MEASURE for faster planning, worse performance.
        cufftPlan1d(&forward_fft_plan, padded_len, CUFFT_D2Z, 1);
        cufftPlan1d(&forward_ifft_plan, padded_len, CUFFT_Z2D, 1);
        cufftPlan1d(&adjoint_fft_plan, padded_len, CUFFT_D2Z, 1);
        cufftPlan1d(&adjoint_ifft_plan, padded_len, CUFFT_Z2D, 1);
         // forward_fft_plan = fftw_plan_dft_r2c_1d(padded_len,
      //       this->input_data.data,
         //     (fftw_complex *) r2c_out.data,
      //       FFTW_MEASURE);
         // forward_ifft_plan = fftw_plan_dft_c2r_1d(padded_len,
      //       (fftw_complex *) r2c_out.data, this->output_data.data,
      //       FFTW_MEASURE);
         // adjoint_fft_plan = fftw_plan_dft_r2c_1d(padded_len, this->output_data.data,
         //     (fftw_complex *) r2c_out.data, FFTW_MEASURE);
         // adjoint_ifft_plan = fftw_plan_dft_c2r_1d(padded_len,
      //       (fftw_complex *) r2c_out.data,
         //     this->input_data.data, FFTW_MEASURE);
        cudaDeviceSynchronize();
        CUDA_CHECK_ERR();
    }

    void free_data() {
        cufftDestroy(forward_fft_plan);
        cufftDestroy(forward_ifft_plan);
        cufftDestroy(adjoint_fft_plan);
        cufftDestroy(adjoint_ifft_plan);
        vector_free(&kernel);
        vector_free(&kernel_fft);
        vector_free(&rev_kernel_fft);
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
        T nrm = cml::blas_nrm2<T>(hdl, &this->kernel);
        printf("kernel data nrm = %e\n", nrm);
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
    void multiply_fft(cml::vector<T>& kernel_fft, cml::vector<T>& output) {
        thrust::complex<T> len((T) padded_len, 0.0);
        cml::strided_range<thrust::device_ptr<thrust::complex<T> > > idx_a(
            thrust::device_pointer_cast((thrust::complex<T> *) kernel_fft.data),
            thrust::device_pointer_cast((thrust::complex<T> *) kernel_fft.data + padded_len), 1);
        cml::strided_range<thrust::device_ptr<thrust::complex<T> > > idx_b(
            thrust::device_pointer_cast((thrust::complex<T> *) output.data),
            thrust::device_pointer_cast((thrust::complex<T> *) output.data + padded_len), 1);
        thrust::transform(idx_a.begin(), idx_a.end(), idx_b.begin(), idx_b.begin(),
            thrust::multiplies<thrust::complex<T> >());
        thrust::transform(idx_b.begin(), idx_b.end(),
            thrust::constant_iterator<thrust::complex<T> >(len), idx_b.begin(),
            thrust::divides<thrust::complex<T> >());
    }

    /* Fill out the input padding with zeros. */
    void zero_pad_input() {
        cml::vector_scale<T>(&input_padding, static_cast<T>(0.0));
    }

    /* Column convolution. */
    void forward_eval() {
        zero_pad_input();
        T nrm = cml::blas_nrm2<T>(hdl, &this->input_data);
        printf("input data nrm = %e\n", nrm);
        cufftExecD2Z(forward_fft_plan,
           (cufftDoubleReal *) this->input_data.data,
           (cufftDoubleComplex *) r2c_out.data);
        nrm = cml::blas_nrm2<T>(hdl, &r2c_out);
        printf("r2c_out data pre multiply fft nrm = %e\n", nrm);
        multiply_fft(kernel_fft, r2c_out);
        nrm = cml::blas_nrm2<T>(hdl, &r2c_out);
        printf("r2c_out data post multiply fft nrm = %e\n", nrm);
        cufftExecZ2D(forward_ifft_plan,
           (cufftDoubleComplex *) r2c_out.data,
           (cufftDoubleReal *) this->output_data.data);
        cudaDeviceSynchronize();
        CUDA_CHECK_ERR();
        nrm = cml::blas_nrm2<T>(hdl, &this->output_data);
        printf("output data nrm = %e\n", nrm);
    }

    /* Row convolution. */
    void adjoint_eval() {
        T nrm = cml::blas_nrm2<T>(hdl, &this->output_data);
        printf("output data nrm = %e\n", nrm);
        cufftExecD2Z(adjoint_fft_plan,
           (cufftDoubleReal *) this->output_data.data,
           (cufftDoubleComplex *) r2c_out.data);
        multiply_fft(rev_kernel_fft, r2c_out);
        cufftExecZ2D(adjoint_ifft_plan,
           (cufftDoubleComplex *) r2c_out.data,
           (cufftDoubleReal *) this->input_data.data);
        // TODO do this? zero_pad_input();
        cudaDeviceSynchronize();
        CUDA_CHECK_ERR();
        nrm = cml::blas_nrm2<T>(hdl, &this->input_data);
        printf("input data nrm = %e\n", nrm);
    }
};
#endif

#ifdef SWIG
%template(FAOd) FAO<double>;
%template(FAOf) FAO<float>;
%template(NoOpd) NoOp<double>;
%template(NoOpf) NoOp<float>;
%template(DenseMatVecMuld) DenseMatVecMul<double>;
%template(DenseMatVecMulf) DenseMatVecMul<float>;
%template(SparseMatVecMuld) SparseMatVecMul<double>;
%template(SparseMatVecMulf) SparseMatVecMul<float>;
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
%template(Convd) Conv<double>;
// %template(Convf) Conv<float>;
#endif
