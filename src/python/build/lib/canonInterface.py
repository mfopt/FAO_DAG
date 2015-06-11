#    This file is part of CVXcanon.
#
#    CVXcanon is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    CVXcanon is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with CVXcanon.  If not, see <http:#www.gnu.org/licenses/>.

import CVXcanon
import numpy as np
from cvxpy.lin_ops.lin_op import *
import scipy.sparse
from collections import deque

def pogs_solve(data, cones):
    '''
    Solves the cone program with POGS.

    Parameters
    ----------
        c: A NumPy array.
        b: A NumPy array.
        A: A SciPy sparse matrix.
        cones: A list of (enum, list of indices) tuples.

    Returns
    -------
        ???
    '''
    c = data['c']
    b = data['b']
    A = data['A']
    m, n = A.shape
    c = convert_to_vec(True, c)
    b = convert_to_vec(True, b)
    Adata = convert_to_vec(True, A.data)
    Aindices = convert_to_vec(False, A.indices)
    Aindptr = convert_to_vec(False, A.indptr)
    cones = python_cones_to_pogs_cones(cones)
    pogs = CVXcanon.PogsData()
    return pogs.solve(m, n, c, b,
                      A.nnz, Adata, Aindices,
                      Aindptr, cones)

def convert_to_vec(is_double, ndarray):
    if is_double:
        vec = CVXcanon.DoubleVector()
        cast = float
    else:
        vec = CVXcanon.IntVector()
        cast = int
    for i in range(ndarray.size):
        vec.push_back(cast(ndarray[i]))
    return vec

def python_cones_to_pogs_cones(cones):
    cone_vect = CVXcanon.ConeConstraintVector()
    for cone_key, indices in cones:
        constr = CVXcanon.ConeConstraint()
        constr.first = 1;
        constr.second = CVXcanon.IntVector()
        for idx in indices:
            constr.second.push_back(idx)
    return cone_vect

def mat_free_pogs_solve(c, b, constr_root, var_sizes, cones):
    '''
    Solves the cone program with matrix-free POGS.

    Parameters
    ----------
        c: A NumPy array.
        b: A NumPy array.
        constr_root: The root of a LinOp tree for the vstacked constraints.
        var_sizes: Map of variable id to size.
        cones: A list of (enum, list of indices) tuples.

    Returns
    -------
        ???
    '''
    c = format_matrix(c, 'dense')
    b = format_matrix(b, 'dense')
    tmp = []
    tree = build_lin_op_tree(constr_root, tmp)
    tmp.append(tree)
    start_node, end_node = tree_to_dag(tree, var_sizes, c.size[0])
    return None

def tree_to_dag(root, var_sizes, x_length):
    '''
    Convert a LinOp tree to a LinOp DAG.
    '''
    # First get all the variables and NOOP nodes.
    leaves = get_leaves(root)
    variables = []
    noops = []
    for node in leaves:
        if node.type == CVXcanon.VARIABLE:
            variables.append(node)
        elif node.type == CVXcanon.NO_OP:
            noops.append(node)
    end_node = CVXcanon.LinOp()
    end_node.type = CVXcanon.SPLIT
    size_pair = CVXcanon.IntPair(int(x_length),
                                 int(1))
    end_node.input_sizes.push_back(size_pair)
    var_id_list = []
    var_copies = {}
    for var_id, size in var_sizes.items():
        var_id_list.append(var_id)
        size_pair = CVXcanon.IntPair(int(size[0]),
                                     int(size[1]))
        end_node.output_sizes.push_back(size_pair)
        # Add copy node for that variable.
        copy_node = CVXcanon.LinOp()
        copy_node.type = CVXcanon.COPY
        copy_node.input_sizes.push_back(size_pair)
        copy_node.args.push_back(end_node)
        var_copies[var_id] = copy_node
    # Link copy nodes directly to outputs of variables.
    for var in variables:
        copy_node = var_copies[var.var_id]
        output_node = var.outputs[0]
        copy_node.outputs.push_back(output_node)
        copy_node.output_sizes.push_back(output_node.input_sizes[0])
        output_node.args[0] = output_node
    # Link a copy node to all the NO_OPs.
    copy_node = var_copies[var_id_list[0]]
    var_size = copy_node.output_sizes[0]
    for noop_node in noops:
        copy_node.outputs.push_back(noop_node)
        copy_node.output_sizes.push_back(var_size)
        noop_node.args.push_back(copy_node)
        noop_node.input_sizes.push_back(var_size)
    return root, end_node

def get_leaves(root):
    '''
    Returns the leaves of the tree.
    '''
    if len(root.args) == 0:
        return [root]
    else:
        leaves = []
        for arg in root.args:
            leaves += get_leaves(arg)
        return leaves

def get_problem_matrix(constrs, id_to_col=None):
    '''
    Builds a sparse representation of the problem data by calling CVXCanon's
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
    lin_vec = CVXcanon.LinOpVector()

    id_to_col_C = CVXcanon.IntIntMap()
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

    problemData = CVXcanon.build_matrix(lin_vec, id_to_col_C)

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
        return np.asfortranarray(matrix)
    elif(format == 'sparse'):
        return scipy.sparse.coo_matrix(matrix)
    elif(format == 'scalar'):
        return np.asfortranarray(np.matrix(matrix))
    else:
        raise NotImplementedError()


def set_matrix_data(linC, linPy):
    '''  Calls the appropriate CVXCanon function to set the matrix data field of our C++ linOp.
    '''
    if isinstance(linPy.data, LinOp):
        if linPy.data.type is 'sparse_const':
            coo = format_matrix(linPy.data.data, 'sparse')
            linC.set_sparse_data(coo.data, coo.row.astype(float),
                                 coo.col.astype(float), coo.shape[0], coo.shape[1])
        elif linPy.data.type is 'dense_const':
            linC.set_dense_data(format_matrix(linPy.data.data))
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
        vec = CVXcanon.IntVector()
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


type_map = { "VARIABLE": CVXcanon.VARIABLE,
"PROMOTE": CVXcanon.PROMOTE,
"MUL": CVXcanon.MUL,
"RMUL": CVXcanon.RMUL,
"MUL_ELEM": CVXcanon.MUL_ELEM,
"DIV": CVXcanon.DIV,
"SUM": CVXcanon.SUM,
"NEG": CVXcanon.NEG,
"INDEX": CVXcanon.INDEX,
"TRANSPOSE": CVXcanon.TRANSPOSE,
"SUM_ENTRIES": CVXcanon.SUM_ENTRIES,
"TRACE": CVXcanon.TRACE,
"RESHAPE": CVXcanon.RESHAPE,
"DIAG_VEC": CVXcanon.DIAG_VEC,
"DIAG_MAT": CVXcanon.DIAG_MAT,
"UPPER_TRI": CVXcanon.UPPER_TRI,
"CONV": CVXcanon.CONV,
"HSTACK": CVXcanon.HSTACK,
"VSTACK": CVXcanon.VSTACK,
"SCALAR_CONST": CVXcanon.SCALAR_CONST,
"DENSE_CONST": CVXcanon.DENSE_CONST,
"SPARSE_CONST": CVXcanon.SPARSE_CONST,
"NO_OP": CVXcanon.NO_OP }

def get_type(ty):
    if ty in type_map:
        return type_map[ty]
    else:
        raise NotImplementedError()



def build_lin_op_tree(root_linPy, tmp):
    '''
    Breadth-first, pre-order traversal on the Python linOp tree
    Parameters
    -------------
    root_linPy: a Python LinOp tree

    tmp: an array to keep data from going out of scope

    Returns
    --------
    root_linC: a C++ LinOp tree created through our swig interface
    '''
    Q = deque()
    root_linC = CVXcanon.LinOp()
    Q.append((root_linPy, root_linC))

    while len(Q) > 0:
        linPy, linC = Q.popleft()
        size_pair = CVXcanon.IntPair(int(linPy.size[0]),
                            int(linPy.size[1]))
        # Updating the arguments our LinOp
        for argPy in linPy.args:
            tree = CVXcanon.LinOp()
            tmp.append(tree)
            Q.append((argPy, tree))
            tree.outputs.push_back(linC)
            tree.output_sizes.push_back(size_pair)
            linC.args.push_back(tree)
            arg_size_pair = CVXcanon.IntPair(int(argPy.size[0]),
                                    int(argPy.size[1]))
            linC.input_sizes.push_back(arg_size_pair)

        # Setting the type of our lin op
        linC.type = get_type(linPy.type.upper())

        # Setting size
        linC.size.push_back(int(linPy.size[0]))
        linC.size.push_back(int(linPy.size[1]))

        # Loading the problem data into the appropriate array format
        if linPy.data is None:
            pass
        elif linPy.type == VARIABLE:
            linC.var_id = int(linPy.data)
        elif isinstance(linPy.data, tuple) and isinstance(linPy.data[0], slice):
            set_slice_data(linC, linPy)
        elif isinstance(linPy.data, float) or isinstance(linPy.data, int):
            linC.set_dense_data(format_matrix(linPy.data, 'scalar'))
        elif isinstance(linPy.data, LinOp) and linPy.data.type is 'scalar_const':
            linC.set_dense_data(format_matrix(linPy.data.data, 'scalar'))
        else:
            set_matrix_data(linC, linPy)

    return root_linC
