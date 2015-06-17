from cvxpy import *
import cvxpy.settings as s
import cvxpy.lin_ops.lin_utils as lu
import cvxpy.lin_ops.fao_utils as fao
from cvxpy.lin_ops.tree_mat import mul, tmul, prune_constants
import cvxpy.problems.iterative as iterative
from cvxpy.problems.solvers.utilities import SOLVERS
from cvxpy.problems.problem_data.sym_data import SymData
import numpy as np
import numpy.linalg
import scipy.sparse as sp
import scipy.linalg as LA
import unittest
import faoInterface


# Convolution
x = Variable(3)
f = np.matrix(np.array([1, 2, 3])).T
g = np.array([0, 1, 0.5])
f_conv_g = np.array([ 0., 1., 2.5,  4., 1.5])
expr = conv(f, x).canonical_form[0]
vars_ = lu.get_expr_vars(expr)
dag = fao.tree_to_dag(expr, vars_)

input_arr = g
output_arr = np.zeros(5)
faoInterface.eval_FAO_DAG(dag, input_arr, output_arr)
# self.assertItemsAlmostEqual(output_arr, f_conv_g)

input_arr = np.array(range(5))
output_arr = np.zeros(3)
toep = LA.toeplitz(np.array([1,0,0]),
                   np.array([1, 2, 3, 0, 0]))
faoInterface.eval_FAO_DAG(dag, input_arr, output_arr, forward=False)
print output_arr
print toep