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

%module FAO_DAG
%{
	#define SWIG_FILE_WITH_INIT
	#include "FAO_DAG.hpp"
%}

%include "numpy.i"
%include "std_vector.i"
%include "std_map.i"

/* Must call this before using NUMPY-C API */
%init %{
	import_array();
%}

/* Typemap for the addDenseData C++ routine in LinOp.hpp */
%apply (double* IN_FARRAY2, size_t DIM1, size_t DIM2) {(double* matrix, size_t rows, size_t cols)};

/* Typemap for the addSparseData C++ routine in LinOp.hpp */
%apply (double* INPLACE_ARRAY1, size_t DIM1) {(double *data, size_t data_len)}

%apply (size_t* INPLACE_ARRAY1, size_t DIM1) {(size_t *indices, size_t idx_len), (size_t *ptrs, size_t ptr_len)};
%include "FAO.hpp"


/* Useful wrappers for the FAO class */
namespace std {
   %template(IntVector) vector<int>;
   %template(DoubleVector) vector<double>;
   %template(IntVector2D) vector< vector<int> >;
   %template(DoubleVector2D) vector< vector<double> >;
   %template(IntIntMap) map<int, int>;
   %template(ConeConstraint) pair<int, vector<int> >;
   %template(ConeConstraintVector) vector< pair<int, vector<int> > >;
}

/*namespace FAO {
   %template(DoubleFAO) FAO<double,double>;
   %template(DoubleVariable) Variable<double>;
   %template(DoubleConstant) Constant<double>;
   %template(DoubleNoOp) NoOp<double,double>;
   %template(DoubleDenseMatMul) DenseMatMul<double>;
   %template(DoubleSparseMatMul) SparseMatMul<double>;
   %template(DoubleScalarMul) ScalarMul<double>;
   %template(DoubleSum) Sum<double>;
   %template(DoubleCopy) Copy<double>;
   %template(DoubleVstack) Vstack<double>;
   %template(DoubleSplit) Split<double>;
}*/

