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

#include "FAO.hpp"
#include "gsl/gsl_vector"

class FAO_DAG<T> {
// TODO only expose start_node input type and end_node output type.
/* Represents an FAO DAG. Used to evaluate FAO DAG and its adjoint. */
	FAO<T,T> start_node;
	FAO<T,T> end_node;

	/* Returns a pointer to the input vector for forward evaluation. */
	gsl::vector<T>* get_forward_input() {
		return &start_node.input_data;
	}

	/* Returns a pointer to the output vector for forward evaluation. */
	gsl::vector<T>* get_forward_output() {
		return &start_node.input_data;
	}

	/* Returns a pointer to the input vector for forward evaluation. */
	gsl::vector<T>* get_adjoint_input() {
		return get_forward_output();
	}

	/* Returns a pointer to the output vector for forward evaluation. */
	gsl::vector<T>* get_adjoint_output() {
		return get_forward_input();
	}


	/* Evaluate the FAO DAG. */
	void eval_dag() {
		// TODO.
		return;
	}

	/* Evaluate the adjoint DAG. */
	void eval_adjoint_dag() {
		// TODO.
		return;
	}
};
