#    This file is part of FAO_DAG.
#
#    FAO_DAG is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    FAO_DAG is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with FAO_DAG.  If not, see <http:#www.gnu.org/licenses/>.

import FAO_DAG
import numpy as np
from cvxpy.lin_ops.lin_op import *
from cvxpy.lin_ops.fao_utils import (SCALAR_MUL, DENSE_MAT_VEC_MUL,
DENSE_MAT_MAT_MUL, SPARSE_MAT_VEC_MUL, SPARSE_MAT_MAT_MUL,
COPY, SPLIT)

import scipy.sparse
from collections import deque

# def pogs_solve(data, cones, rho=1e-3, verbose=False,
#                abs_tol=1e-3, rel_tol=1e-3, max_iter=2500):
#     '''
#     Solves the cone program with POGS.

#     Parameters
#     ----------
#         c: A NumPy array.
#         b: A NumPy array.
#         A: A SciPy sparse matrix.
#         cones: A list of (enum, list of indices) tuples.

#     Returns
#     -------
#         ???
#     '''
#     c = data['c']
#     b = data['b']
#     A = data['A']
#     m, n = A.shape
#     c = convert_to_vec(True, c)
#     b = convert_to_vec(True, b)
#     Adata = convert_to_vec(True, A.data)
#     Aindices = convert_to_vec(False, A.indices)
#     Aindptr = convert_to_vec(False, A.indptr)
#     cones = python_cones_to_pogs_cones(cones)

#     primal_result = FAO_DAG.DoubleVector()
#     dual_result = FAO_DAG.DoubleVector()
#     pogs = FAO_DAG.PogsData()
#     opt_val = pogs.solve(m, n, c, b,
#                          A.nnz, Adata, Aindices,
#                          Aindptr, cones,
#                          primal_result, dual_result,
#                          rho, verbose,
#                          abs_tol, rel_tol, max_iter)
#     x = np.array([primal_result[i] for i in range(n)])
#     mu = np.array([dual_result[i] for i in range(m)])
#     return {'info':{'status': "Solved", 'pobj': opt_val},
#             'x': x, 'mu':mu}

def convert_to_vec(is_double, iterable, div=1):
    if is_double:
        vec = FAO_DAG.DoubleVector()
        cast = float
    else:
        vec = FAO_DAG.IntVector()
        cast = int
    for i in range(len(iterable)):
        vec.push_back(cast(iterable[i])/div)
    return vec

# def python_cones_to_pogs_cones(cones):
#     cone_vect = FAO_DAG.ConeConstraintVector()
#     for cone_key, indices in cones:
#         constr = FAO_DAG.ConeConstraint()
#         constr.first = int(cone_key);
#         constr.second = FAO_DAG.IntVector()
#         for idx in indices:
#             constr.second.push_back(idx)
#         cone_vect.push_back(constr)
#     return cone_vect

# def mat_free_pogs_solve(c, b, constr_root, var_sizes, cones):
#     '''
#     Solves the cone program with matrix-free POGS.

#     Parameters
#     ----------
#         c: A NumPy array.
#         b: A NumPy array.
#         constr_root: The root of a LinOp tree for the vstacked constraints.
#         var_sizes: Map of variable id to size.
#         cones: A list of (enum, list of indices) tuples.

#     Returns
#     -------
#         ???
#     '''
#     c = format_matrix(c, 'dense')
#     b = format_matrix(b, 'dense')
#     tmp = []
#     tree = build_lin_op_tree(constr_root, tmp)
#     tmp.append(tree)
#     start_node, end_node = tree_to_dag(tree, var_sizes, c.size[0])
#     return None

def scs_solve(py_dag, data, dims, solver_opts):
    """Solve using SCS with FAO DAGs.

    py_dag: The Python FAO DAG.
    data: A map with all the information needed by SCS.
    """
    tmp = []
    # print py_dag
    print py_dag.start_node
    print py_dag.end_node
    start_node, end_node, edges = python_to_swig(py_dag, tmp)
    dag = FAO_DAG.FAO_DAG(start_node, end_node, edges)
    scs_data = FAO_DAG.SCS_Data();
    scs_data.load_c(data['c'].flatten())
    scs_data.load_b(data['b'].A.flatten())
    # Pass in solution arrays.
    x = np.zeros(data['c'].size)
    y = np.zeros(data['b'].size)
    scs_data.load_x(x)
    scs_data.load_y(y)

    q_vec = convert_to_vec(False, dims['q'])
    s_vec = convert_to_vec(False, dims['s'])
    scs_data.solve(dag, dims['f'], dims['l'], q_vec, s_vec, dims['ep'],
                            solver_opts['max_iters'])
    info = {
        "statusVal": scs_data.statusVal,
        "iter": scs_data.iter,
        "cgIter": scs_data.cgIter,
        "pobj": scs_data.pobj,
        "dobj": scs_data.dobj,
        "resPri": scs_data.resPri,
        "resDual": scs_data.resDual,
        "relGap": scs_data.relGap,
        "solveTime": (scs_data.solveTime / 1e3),
        "setupTime": (scs_data.setupTime / 1e3),
        "status": ''.join(scs_data.status),
    }
    # Must destroy FAO DAG before calling FAO destructors.
    del dag
    return {'info':info, 'x':x, 'y':y}

def eval_FAO_DAG(py_dag, input_arr, output_arr, forward=True):
    """ordered_vars: list of (id, size) tuples.
    """
    tmp = []
    start_node, end_node, edges = python_to_swig(py_dag, tmp)
    dag = FAO_DAG.FAO_DAG(start_node, end_node, edges)
    input_vec = convert_to_vec(True, input_arr)
    dag.copy_input(input_vec, forward)
    if forward:
        dag.forward_eval()
    else:
        dag.adjoint_eval()
    output_vec = convert_to_vec(True, output_arr)
    dag.copy_output(output_vec, forward)
    output_arr[:] = output_vec[:]
    # Must destroy FAO DAG before calling FAO destructors.
    del dag

def set_dense_data(node_c, node_py):
    """Stores dense matrix data on the Swig FAO.
    """
    matrix = node_py.data.astype(float, order='F')
    node_c.set_matrix_data(matrix)

def set_sparse_data(node_c, node_py):
    """Stores dense matrix data on the Swig FAO.
    """
    csr = scipy.sparse.csr_matrix(node_py.data)
    node_c.set_spmatrix_data(csr.data, csr.indptr.astype(int),
                             csr.indices.astype(int), csr.shape[0], csr.shape[1])

def set_slice_data(linC, linPy):
    '''
    Loads the slice data, start, stop, and step into our C++ linOp.
    The semantics of the slice operator is treated exactly the same as in Python.
    Note that the 'None' cases had to be handled at the wrapper level, since we must load
    integers into our vector.
    '''
    for i, sl in enumerate(linPy.data):
        vec = FAO_DAG.IntVector()
        if (sl.start is None):
            vec.push_back(0)
        else:
            vec.push_back(sl.start)
        if(sl.stop is None):
            vec.push_back(linPy.args[0].size[i])
        else:
            vec.push_back(sl.stop)
        if sl.step is None:
            vec.push_back(1)
        else:
            vec.push_back(sl.step)
        linC.slice.push_back(vec)

type_map = {
    # "PROMOTE": FAO_DAG.PROMOTE,
    # "MUL": FAO_DAG.MUL,
    # "RMUL": FAO_DAG.RMUL,
    # "MUL_ELEM": FAO_DAG.MUL_ELEM,
    # "DIV": FAO_DAG.DIV,
    SUM: FAO_DAG.Sum,
    NEG: FAO_DAG.Neg,
    # "INDEX": FAO_DAG.INDEX,
    # "TRANSPOSE": FAO_DAG.TRANSPOSE,
    # "SUM_ENTRIES": FAO_DAG.SUM_ENTRIES,
    # "TRACE": FAO_DAG.TRACE,
    # "RESHAPE": FAO_DAG.RESHAPE,
    # "DIAG_VEC": FAO_DAG.DIAG_VEC,
    # "DIAG_MAT": FAO_DAG.DIAG_MAT,
    # "UPPER_TRI": FAO_DAG.UPPER_TRI,
    CONV: FAO_DAG.Conv,
    # "HSTACK": FAO_DAG.HSTACK,
    VSTACK: FAO_DAG.Vstack,
    SCALAR_MUL: FAO_DAG.ScalarMul,
    DENSE_MAT_VEC_MUL: FAO_DAG.DenseMatVecMul,
    SPARSE_MAT_VEC_MUL: FAO_DAG.SparseMatVecMul,
    DENSE_MAT_MAT_MUL: FAO_DAG.DenseMatMatMul,
    SPARSE_MAT_MAT_MUL: FAO_DAG.SparseMatMatMul,
    COPY: FAO_DAG.Copy,
    SPLIT: FAO_DAG.Split,
    # SCALAR_CONST: FAO_DAG.Constant,
    # DENSE_CONST: FAO_DAG.Constant,
    # SPARSE_CONST: FAO_DAG.Constant,
    NO_OP: FAO_DAG.NoOp,
}

def get_FAO(node):
    if node.type in type_map:
        # Make input and output sizes.
        input_sizes = get_dims_vec(node.input_sizes)
        output_sizes = get_dims_vec(node.output_sizes)
        swig_fao = type_map[node.type]()
        swig_fao.input_sizes = input_sizes
        swig_fao.output_sizes = output_sizes
        swig_fao.input_edges = get_edge_vec(node.input_edges)
        swig_fao.output_edges = get_edge_vec(node.output_edges)
        return swig_fao
    else:
        print node.type
        raise NotImplementedError()

def get_edge_vec(edges):
    """Returns an FAO vec full of Null pointers.
    """
    edge_vec = FAO_DAG.IntVector()
    for edge_id in edges:
        edge_vec.push_back(edge_id)
    return edge_vec

def get_dims_vec(sizes):
    """Returns the vector for a FAO input/output sizes.
    """
    dims_vec = FAO_DAG.SizetVector2D()
    for dims in sizes:
        dims_vec.push_back(get_dims(dims))
    return dims_vec

def get_dims(size):
    """Returns the vector for a FAO input/output size.
    """
    size_pair = FAO_DAG.SizetVector()
    size_pair.push_back(int(size[0]))
    size_pair.push_back(int(size[1]))
    return size_pair

def python_to_swig(py_dag, tmp):
    """Convert an FAO DAG in Python into an FAO DAG in C++.

    Parameters
    ----------
    dag: A Python FAO DAG.
    tmp: A list to keep data from going out of scope.

    Returns
    --------
    tuple
        A (start_node, end_node, nodes, edges) tuple for the C++ FAO DAG.
    """
    start_node = py_dag.start_node
    ready_queue = deque()
    start_swig = get_FAO(start_node)
    ready_queue.append(start_node)
    tmp.append(start_swig)
    id_to_swig = {id(start_node): start_swig}
    py_faos = [start_node]
    # Populate id_to_swig and py_faos.
    while len(ready_queue) > 0:
        cur_py = ready_queue.popleft()
        cur_c = id_to_swig[id(cur_py)]
        # Updating the arguments for Swig FAO.
        for edge_id in cur_py.output_edges:
            node_py = py_dag.edges[edge_id][1]
            if id(node_py) not in id_to_swig:
                node_c = get_FAO(node_py)
                tmp.append(node_c)
                id_to_swig[id(node_py)] = node_c
                py_faos.append(node_py)
                ready_queue.append(node_py)

        # Loading the problem data into the appropriate array format
        if cur_py.type == INDEX:
            set_slice_data(cur_c, cur_py)
        elif cur_py.type in [SCALAR_MUL]:
            cur_c.scalar = float(cur_py.data)
        elif cur_py.type in [DENSE_MAT_VEC_MUL, DENSE_MAT_MAT_MUL]:
            set_dense_data(cur_c, cur_py)
        elif cur_py.type in [SPARSE_MAT_VEC_MUL, SPARSE_MAT_MAT_MUL]:
            set_sparse_data(cur_c, cur_py)
        elif cur_py.type == CONV:
            cur_c.set_conv_data(cur_py.data)

    # Now populate Swig edges.
    edges_c = FAO_DAG.EdgeMap()
    for edge_id, (start, end) in py_dag.edges.items():
        start_c = id_to_swig[id(start)]
        end_c = id_to_swig[id(end)]
        edges_c[edge_id] = FAO_DAG.Edge(start_c, end_c)

    start_c = id_to_swig[id(start_node)]
    end_c = id_to_swig[id(py_dag.end_node)]
    # TODO why doesn't this work?
    # nodes_c = FAO_DAG.NodeMap()
    # for node_id, node_c in id_to_swig.items():
    #     nodes_c[int(node_id)] = node_c
    return start_c, end_c, edges_c

