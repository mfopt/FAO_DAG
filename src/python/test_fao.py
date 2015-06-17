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


x = Variable(3,2)
expr = (-x).canonical_form[0]
vars_ = lu.get_expr_vars(expr)
dag = fao.tree_to_dag(expr, vars_)
print dag
input_arr = np.arange(6)
output_arr = np.zeros(3*2)
faoInterface.eval_FAO_DAG(dag, input_arr, output_arr)
assert (output_arr == -input_arr).all()