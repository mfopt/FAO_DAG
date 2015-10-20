from cvxpy import *
import numpy as np
import time
# x = Variable()
# prob = Problem(Maximize(x), [x <= 1])
# prob.solve(solver=SCS_MAT_FREE, max_iters=2500, verbose=True)
# print "result =", prob.value
# print "x value =", x.value


np.random.seed(1)
n = 500
m = n
X = Variable(n, n)
A = np.abs(np.random.randn(m, n)) + 1e-6
B = np.abs(np.random.randn(n, m)) + 1e-6
C = np.random.randn(n, n)
D = np.abs(np.random.randn(m, m)) + 1e-6

print "nnz = ", (m*n)**2

Z = Variable(m, n)
cost = trace(C.T*X) #+ norm(X, 'fro')
# prob = Problem(Minimize(cost),
#                 [X >= 0, Z*B <= 1, A*X == Z])
# start = time.time()
# prob.solve(solver=ECOS, verbose=False)
# print "ECOS obj", cost.value
# print "time to solve", time.time() - start
# true_X = X.value

# prob.solve(solver=POGS, verbose=False, max_iters=2500,
#            double=True, abs_tol=1e-4, rho=1, rel_tol=7.5e-4)
# print "POGS double cost", cost.value

prob = Problem(Minimize(cost),
                [X >= 0, A*X*B <= 1])
# print "SCS obj", prob.solve(solver=SCS, use_indirect=False, verbose=True,
#                             max_iters=10000, eps=1e-3)
# print "SCS cost", cost.value

# prob.solve(solver=OLD_SCS_MAT_FREE, max_iters=2500, verbose=False,
#            equil_steps=1, samples=200, precond=True, stoch=True, eps=1e-3)
# print "OLD MAT FREE obj", prob.value
# print "OLD MAT FREE cost", cost.value

# prob.solve(solver=MAT_FREE_SCS, max_iters=5000, verbose=False,
#            equil_steps=1, samples=200, precond=True, rand_seed=False, eps=1e-3)
# print "SCS MAT FREE obj", prob.value
# print "SCS MAT FREE cost", cost.value

# denom = norm(true_X, 'fro').value*norm(C, 'fro').value

prob.solve(solver=MAT_FREE_POGS, verbose=False, extra_verbose=False, max_iters=100,
    samples=200, equil_steps=1, rho=1, double=True, abs_tol=1e-4, rel_tol=1e-3)
print "MAT FREE POGS double cost", cost.value
# print "MAT FREE POGS double relative error", cost.value/denom

prob.solve(solver=MAT_FREE_POGS, verbose=False, max_iters=100,
    samples=200, equil_steps=1, rho=1, double=False, abs_tol=1e-4, rel_tol=1e-3)
print "MAT FREE POGS float cost", cost.value
# print "MAT FREE POGS double relative error", cost.value/denom
