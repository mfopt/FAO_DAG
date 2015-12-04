from cvxpy import *
import numpy as np

# x = Variable()
# prob = Problem(Maximize(x), [x <= 1])
# prob.solve(solver=SCS_MAT_FREE, max_iters=2500, verbose=True)
# print "result =", prob.value
# print "x value =", x.value


np.random.seed(1)
m = 1000
n = 10000
x = Variable(n)
A = np.random.randn(m, n)
b = np.random.randn(m, 1)
cost = norm(A*x - b,2) + norm(x, 1)
prob = Problem(Minimize(cost))

# print "ECOS obj", prob.solve(solver=ECOS)
print "SCS obj", prob.solve(solver=SCS, use_indirect=True, verbose=True)
# print "SCS cost", cost.value

# prob.solve(solver=OLD_SCS_MAT_FREE, max_iters=2500, verbose=True,
#            equil_steps=1, samples=200, precond=True, stoch=True)
# print "OLD MAT FREE obj", prob.value
# print "OLD MAT FREE cost", cost.value

# prob.solve(solver=MAT_FREE_SCS, max_iters=2500, verbose=True,
#            equil_steps=1, samples=200, precond=True)
# print "SCS MAT FREE obj", prob.value
# print "SCS MAT FREE cost", cost.value
