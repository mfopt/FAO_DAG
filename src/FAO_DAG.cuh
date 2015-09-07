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

#include "FAO.cuh"
#include <algorithm>
#include <vector>
#include <utility>
#include <map>
#include <queue>
#include <functional>
#include "cml/cml_vector.cuh"
#include <typeinfo>
#include "pogs_fork/src/include/timer.h"

template <class T>
class FAO_DAG {
public:
// TODO only expose start_node input type and end_node output type.
/* Represents an FAO DAG. Used to evaluate FAO DAG and its adjoint. */
	FAO<T>* start_node;
	FAO<T>* end_node;
	std::map<int, std::pair<FAO<T> *, FAO<T> *> > edges;
	std::queue<FAO<T> *> ready_queue;
	std::map<FAO<T> *, size_t> eval_map;

	// Timing info.
	int forward_evals = 0;
	int adjoint_evals = 0;
	double total_forward_eval_time = 0;
	double total_adjoint_eval_time = 0;

	FAO_DAG(FAO<T>* start_node, FAO<T>* end_node,
			std::map<int, std::pair<FAO<T> *, FAO<T> *> > edges) {
		this->start_node = start_node;
		this->end_node = end_node;
		this->edges = edges;
		/* Allocate input and output arrays on each node. */
		auto node_fn = [](FAO<T>* node) {
			node->alloc_data();
			node->init_offset_maps();
		};
		traverse_graph(node_fn, true);
	}

	~FAO_DAG() {
		// For timing.
		printf("forward_evals=%i, avg_forward_eval_time=%e\n", forward_evals,
			total_forward_eval_time/forward_evals);
		printf("adjoint_evals=%i, avg_adjoint_eval_time=%e\n", adjoint_evals,
			total_adjoint_eval_time/adjoint_evals);
		/* Deallocate input and output arrays on each node. */
		auto node_fn = [](FAO<T>* node) {node->free_data();};
		traverse_graph(node_fn, true);
	}

#ifndef SWIG
	void traverse_graph (std::function<void(FAO<T>*)> node_fn, bool forward) {
		/* Traverse the graph and apply the given function at each node.

		   forward: Traverse in standard or reverse order?
		   node_fn: Function to evaluate on each node.
		*/
		FAO<T> *start;
		if (forward) {
			start = start_node;
		} else {
			start = end_node;
		}
		ready_queue.push(start);
		while (!ready_queue.empty()) {
			FAO<T>* curr = ready_queue.front();
			ready_queue.pop();
			// Evaluate the given function on curr.
			node_fn(curr);
			eval_map[curr]++;
			std::vector<int> child_edges;
			if (forward) {
				child_edges = curr->output_edges;
			} else {
				child_edges = curr->input_edges;
			}
			/* If each input has visited the node, it is ready. */
			for (auto edge_idx : child_edges) {
				auto edge = edges[edge_idx];
				FAO<T> *node;
				if (forward) {
					node = edge.second;
				} else {
					node = edge.first;
				}
				eval_map[node]++;
				size_t node_inputs_count;
				if (forward) {
					node_inputs_count = node->input_edges.size();
				} else {
					node_inputs_count = node->output_edges.size();
				}
				if (eval_map[node] == node_inputs_count)
					ready_queue.push(node);
			}
		}
		eval_map.clear();
	}
#endif
	/* For interacting with Python. */
	void copy_input(std::vector<T>& input, bool forward) {
		cml::vector<T>* input_vec;
		if (forward) {
			input_vec = get_forward_input();
		} else {
			input_vec = get_adjoint_input();
		}
		assert(input.size() == input_vec->size);
		cml::vector_memcpy<T>(input_vec, input.data());
	}

	void copy_output(std::vector<T>& output, bool forward) {
		cml::vector<T>* output_vec;
		if (forward) {
			output_vec = get_forward_output();
		} else {
			output_vec = get_adjoint_output();
		}
		assert(output.size() == output_vec->size);
		cml::vector_memcpy<T>(output.data(), output_vec);
	}

	/* Returns a pointer to the input vector for forward evaluation. */
	cml::vector<T>* get_forward_input() {
		return &start_node->input_data;
	}

	/* Returns a pointer to the output vector for forward evaluation. */
	cml::vector<T>* get_forward_output() {
		return &end_node->output_data;
	}

	/* Returns a pointer to the input vector for forward evaluation. */
	cml::vector<T>* get_adjoint_input() {
		return get_forward_output();
	}

	/* Returns a pointer to the output vector for forward evaluation. */
	cml::vector<T>* get_adjoint_output() {
		return get_forward_input();
	}

	/* Evaluate the FAO DAG. */
	void forward_eval() {
		auto node_eval = [this](FAO<T> *node){
			node->forward_eval();
			// Copy data from node to children.
			for (size_t i=0; i < node->output_edges.size(); ++i) {
				size_t edge_idx = node->output_edges[i];
				FAO<T>* target = this->edges[edge_idx].second;
				size_t len = node->get_elem_length(node->output_sizes[i]);
				size_t node_offset = node->output_offsets[edge_idx];
				size_t target_offset = target->input_offsets[edge_idx];
				// printf("edge = %d\n", edge_idx);
				// printf("target->input_data.stride = %d\n", target->input_data.stride);
				// printf("node->output_data.stride = %d\n", node->output_data.stride);
				// Copy len elements from node_start to target_start.
				cml::vector_subvec_memcpy<T>(&target->input_data, target_offset,
							  &node->output_data, node_offset, len);
			}
			cudaDeviceSynchronize();
			CUDA_CHECK_ERR();
		};
		double t = timer<double>();
		traverse_graph(node_eval, true);
		cudaDeviceSynchronize();
		forward_evals++;
		double eval_time = timer<double>() - t;
		total_forward_eval_time += eval_time;
        // printf("T_forward_eval = %e\n", eval_time);
	}

	/* Evaluate the adjoint DAG. */
	void adjoint_eval() {
		auto node_eval = [this](FAO<T> *node){
			node->adjoint_eval();
			// Copy data from node to children.
			for (size_t i=0; i < node->input_edges.size(); ++i) {
				size_t edge_idx = node->input_edges[i];
				FAO<T>* target = this->edges[edge_idx].first;
				size_t len = node->get_elem_length(node->input_sizes[i]);
				size_t node_offset = node->input_offsets[edge_idx];
				size_t target_offset = target->output_offsets[edge_idx];
				// Copy len elements from node_start to target_start.
				cml::vector_subvec_memcpy<T>(&target->output_data, target_offset,
							  &node->input_data, node_offset, len);
			}
			cudaDeviceSynchronize();
			CUDA_CHECK_ERR();
		};
		double t = timer<double>();
		traverse_graph(node_eval, false);
		cudaDeviceSynchronize();
		adjoint_evals++;
		double eval_time = timer<double>() - t;
		total_adjoint_eval_time += eval_time;
        // printf("T_adjoint_eval = %e\n", eval_time);
	}

	static void static_forward_eval(void *ptr) {
        ((FAO_DAG<T> *) ptr)->forward_eval();
    }

    static void static_adjoint_eval(void *ptr) {
        ((FAO_DAG<T> *) ptr)->adjoint_eval();
    }

};

#ifdef SWIG
%template(FAO_DAGd) FAO_DAG<double>;
%template(FAO_DAGf) FAO_DAG<float>;
#endif

#endif
