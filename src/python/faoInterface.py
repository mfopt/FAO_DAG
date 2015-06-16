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

def convert_to_vec(is_double, ndarray, div=1):
    if is_double:
        vec = FAO_DAG.DoubleVector()
        cast = float
    else:
        vec = FAO_DAG.IntVector()
        cast = int
    for i in range(ndarray.size):
        vec.push_back(cast(ndarray[i])/div)
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

def eval_FAO_DAG(root, ordered_vars, input_arr, output_arr, forward=True):
    """ordered_vars: list of (id, size) tuples.
    """
    x_length = sum([size[0]*size[1] for v,size in ordered_vars])
    tmp = []
    var_nodes = []
    no_op_nodes = []
    tree = build_lin_op_tree(root, var_nodes, no_op_nodes, tmp)
    start_node, end_node = tree_to_dag(tree, var_nodes, no_op_nodes,
                                       ordered_vars, x_length, tmp)
    dag = FAO_DAG.FAO_DAG(start_node, end_node)
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


def tree_to_dag(root, var_nodes, no_op_nodes, ordered_vars, x_length, tmp):
    '''
    Convert a LinOp tree to a LinOp DAG.
    '''
    start_node = FAO_DAG.Split()
    tmp.append(start_node)
    size_pair = FAO_DAG.SizetVector()
    size_pair.push_back(int(x_length))
    start_node.input_sizes.push_back(size_pair)
    var_copies = {}
    for var_id, size in ordered_vars:
        size_pair = get_dims(size)
        # Add copy node for that variable.
        copy_node = FAO_DAG.Copy()
        tmp.append(copy_node)
        copy_node.input_sizes.push_back(size_pair)
        copy_node.input_nodes.push_back(start_node)
        start_node.output_sizes.push_back(size_pair)
        start_node.output_nodes.push_back(copy_node)
        var_copies[var_id] = copy_node
    # Link copy nodes directly to outputs of variables.
    for var in var_nodes:
        copy_node = var_copies[var.var_id]
        output_node = var.output_nodes[0]
        copy_node.output_nodes.push_back(output_node)
        copy_node.output_sizes.push_back(output_node.input_sizes[0])
        output_node.input_nodes[0] = copy_node
    # Link a copy node to all the NO_OPs.
    copy_node = var_copies[ordered_vars[0][0]]
    var_size = ordered_vars[0][1]
    for no_op_node in no_op_nodes:
        copy_node.output_nodes.push_back(no_op_node)
        copy_node.output_sizes.push_back(var_size)
        no_op_node.input_nodes.push_back(copy_node)
        no_op_node.input_sizes.push_back(var_size)
    return start_node, root

def get_leaves(root):
    '''
    Returns the leaves of the tree.
    '''
    if len(root.input_nodes) == 0:
        return [root]
    else:
        leaves = []
        for arg in root.input_nodes:
            leaves += get_leaves(arg)
        return leaves

def get_problem_matrix(constrs, id_to_col=None):
    '''
    Builds a sparse representation of the problem data by calling FAO_DAG's
    C++ build_matrix function.

    Parameters
    ----------
        constrs: A list of python linOp trees
        id_to_col: A map from variable id to offset withoun our matrix

    Returns
    ----------
        V, I, J: numpy arrays encoding a sparse representation of our problem
        const_vec: a numpy column vector representing the constant_data in our problem
    '''
    linOps = [constr.expr for constr in constrs]
    lin_vec = FAO_DAG.LinOpVector()

    id_to_col_C = FAO_DAG.IntIntMap()
    if id_to_col is None:
        id_to_col = {}

    # Loading the variable offsets from our
    # Python map into a C++ map
    for id, col in id_to_col.items():
        id_to_col_C[id] = col

    # This array keeps variables data in scope
    # after build_lin_op_tree returns
    tmp = []
    for lin in linOps:
        tree = build_lin_op_tree(lin, tmp)
        tmp.append(tree)
        lin_vec.push_back(tree)

    problemData = FAO_DAG.build_matrix(lin_vec, id_to_col_C)

    # Unpacking
    V = problemData.getV(len(problemData.V))
    I = problemData.getI(len(problemData.I))
    J = problemData.getJ(len(problemData.J))
    const_vec = problemData.getConstVec(len(problemData.const_vec))

    return V, I, J, const_vec.reshape(-1, 1)


def format_matrix(matrix, format='dense'):
    ''' Returns the matrix in the appropriate form,
        so that it can be efficiently loaded with our swig wrapper
    '''
    if(format == 'dense'):
        # return np.asfortranarray(matrix)
        return matrix.astype(float, order='F')
    elif(format == 'sparse'):
        return scipy.sparse.csr_matrix(matrix)
    elif(format == 'scalar'):
        return np.asarray(np.matrix(matrix))
    else:
        raise NotImplementedError()


def set_matrix_data(linC, linPy):
    '''  Calls the appropriate FAO_DAG function to set the matrix data field of our C++ linOp.
    '''
    if isinstance(linPy.data, LinOp):
        if linPy.data.type is 'sparse_const':
            csr = format_matrix(linPy.data.data, 'sparse')
            linC.set_spmatrix_data(csr.data, csr.indptr.astype(int),
                                 csr.indices.astype(int), csr.shape[0], csr.shape[1])
        elif linPy.data.type is 'dense_const':
            linC.set_matrix_data(format_matrix(linPy.data.data))
        else:
            raise NotImplementedError()
    else:
        linC.set_dense_data(format_matrix(linPy.data))


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
    VARIABLE: FAO_DAG.Variable,
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
    # "CONV": FAO_DAG.CONV,
    # "HSTACK": FAO_DAG.HSTACK,
    # "VSTACK": FAO_DAG.VSTACK,
    SCALAR_CONST: FAO_DAG.Constant,
    DENSE_CONST: FAO_DAG.Constant,
    SPARSE_CONST: FAO_DAG.Constant,
    NO_OP: FAO_DAG.NoOp,
}

mul_vec_type_map = {
    SCALAR_CONST: FAO_DAG.ScalarMul,
    DENSE_CONST: FAO_DAG.DenseMatVecMul,
    SPARSE_CONST: FAO_DAG.SparseMatVecMul,
}

mul_mat_type_map = {
    SCALAR_CONST: FAO_DAG.ScalarMul,
    DENSE_CONST: FAO_DAG.DenseMatMatMul,
    SPARSE_CONST: FAO_DAG.SparseMatMatMul,
}

def get_type(ty):
    if ty in type_map:
        return type_map[ty]
    else:
        raise NotImplementedError()

def get_dims(size):
    """A python LinOp.
    """
    size_pair = FAO_DAG.SizetVector()
    size_pair.push_back(int(size[0]))
    size_pair.push_back(int(size[1]))
    return size_pair

def get_FAO(linPy, var_nodes, no_op_nodes):
    if linPy.type in type_map:
        linC = type_map[linPy.type]()
    elif linPy.type == MUL:
        if linPy.args[0].size[1] == 1:
            linC = mul_vec_type_map[linPy.data.type]()
        else:
            linC = mul_mat_type_map[linPy.data.type]()
    else:
        print linPy.type
        raise Exception("unknown LinOp.")
    # Add to var_nodes or no_op_nodes.
    if linPy.type == VARIABLE:
        var_nodes.append(linC)
    elif linPy.type == NO_OP:
        no_op_nodes.append(linC)
    return linC

def build_lin_op_tree(root_linPy, var_nodes, no_op_nodes, tmp):
    '''
    Breadth-first, pre-order traversal on the Python linOp tree
    Parameters
    -------------
    root_linPy: a Python LinOp tree
    var_nodes: a list of variable nodes.
    no_op_nodes: a list of no_op nodes.
    tmp: an array to keep data from going out of scope

    Returns
    --------
    root_linC: a C++ LinOp tree created through our swig interface
    '''
    Q = deque()
    root_linC = get_FAO(root_linPy, var_nodes, no_op_nodes)
    Q.append((root_linPy, root_linC))

    # Add the output size.
    size_pair = get_dims(root_linPy.size)
    root_linC.output_sizes.push_back(size_pair)

    while len(Q) > 0:
        linPy, linC = Q.popleft()
        size_pair = get_dims(linPy.size)
        # Updating the arguments our LinOp
        for argPy in linPy.args:
            tree = get_FAO(argPy, var_nodes, no_op_nodes)
            tmp.append(tree)
            Q.append((argPy, tree))
            tree.output_nodes.push_back(linC)
            tree.output_sizes.push_back(size_pair)
            linC.input_nodes.push_back(tree)
            arg_size_pair = get_dims(argPy.size)
            linC.input_sizes.push_back(arg_size_pair)

        # Loading the problem data into the appropriate array format
        if linPy.data is None:
            pass
        elif linPy.type == VARIABLE:
            linC.var_id = int(linPy.data)
        elif isinstance(linPy.data, tuple) and isinstance(linPy.data[0], slice):
            set_slice_data(linC, linPy)
        elif isinstance(linPy.data, LinOp) and linPy.data.type is 'scalar_const':
            linC.scalar = float(linPy.data.data)
        else:
            set_matrix_data(linC, linPy)

    return root_linC
