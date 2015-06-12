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

#ifndef FAO_DAG_H
#define FAO_DAG_H

#include "FAO.hpp"
#include <vector>
#include "gsl/gsl_vector.h"

class FAO_DAG {
public:
// TODO only expose start_node input type and end_node output type.
/* Represents an FAO DAG. Used to evaluate FAO DAG and its adjoint. */
	FAO* start_node;
	FAO* end_node;

	FAO_DAG(FAO* start, FAO* end) {
		start_node = start;
		end_node = end;
		// TODO initialize all the input and output arrays.
	}

	~FAO_DAG() {
		// TODO deallocate all the input and output arrays.
	}

	/* For interacting with Python. */
	void copy_input(std::vector<double> input) {
		auto input_vec = get_forward_input();
		assert(input.size() == input_vec->size);
		gsl::vector_memcpy<double>(input.data(), input_vec);
	}

	void copy_output(std::vector<double> output) {
		auto output_vec = get_forward_output();
		assert(output.size() == output_vec->size);
		gsl::vector_memcpy<double>(output_vec, output.data());
	}

	/* Returns a pointer to the input vector for forward evaluation. */
	gsl::vector<double>* get_forward_input() {
		return &start_node->input_data;
	}

	/* Returns a pointer to the output vector for forward evaluation. */
	gsl::vector<double>* get_forward_output() {
		return &end_node->output_data;
	}

	/* Returns a pointer to the input vector for forward evaluation. */
	gsl::vector<double>* get_adjoint_input() {
		return get_forward_output();
	}

	/* Returns a pointer to the output vector for forward evaluation. */
	gsl::vector<double>* get_adjoint_output() {
		return get_forward_input();
	}


	/* Evaluate the FAO DAG. */
	void forward_eval() {
		// TODO.
		return;
	}

	/* Evaluate the adjoint DAG. */
	void adjoint_eval() {
		// TODO.
		return;
	}
};
#endif
