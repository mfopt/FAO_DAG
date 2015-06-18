from cvxpy import *
x = Variable()
prob = Problem(Maximize(x), [x <= 1])
print prob.solve(solver=SCS_MAT_FREE, max_iters=2500, verbose=True)
