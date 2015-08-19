from cvxpy import *
x = Variable()
# prob = Problem(Minimize(-x), [x <= 2])
# prob.solve(solver=POGS, verbose=True, max_iter=2500)

# prob = Problem(Minimize(-x), [x <= 2])
# prob.solve(solver=MAT_FREE_POGS, verbose=True, max_iter=2500)

# prob = Problem(Minimize(x), [abs(x) <= 2])
# prob.solve(solver=POGS, verbose=True, max_iter=500)

# prob = Problem(Minimize(x), [abs(x) <= 2])
# prob.solve(solver=MAT_FREE_SCS, verbose=True, max_iters=100,
#     samples=200, equil_steps=1)

# prob = Problem(Minimize(x), [abs(x) <= 2])
# prob.solve(solver=MAT_FREE_POGS, verbose=True, max_iter=2500,
#     samples=200, equil_steps=0)
# prob.solve(solver=MAT_FREE_SCS, verbose=True, max_iters=2500,
#     samples=200, equil_steps=0)
# # print x.value

# import numpy as np
# from cvxpy import *
# np.random.seed(1)
# m = 10
# n = 10
# x = Variable(n)
# A = np.random.randn(m,n)
# b = np.random.randn(m, 1)
# cost = max_entries(x)
# prob = Problem(Minimize(cost), [norm(A*x + b) <= 2, x >= -100])
# prob.solve(solver=POGS, rho=1e-3, verbose=True)
# print "POGS obj", prob.value
# print "POGS cost", cost.value
# prob.solve(solver=OLD_SCS_MAT_FREE, rho=1, verbose=True, abs_tol=1e-4, rel_tol=1e-3, max_iter=10000,
#     samples=200, equil_steps=1)
# print "MAT FREE SCS obj", prob.value
# print "MAT FREE SCS cost", cost.value
# print "ECOS obj", prob.solve(solver=ECOS)
# print "ECOS cost", cost.value
# print prob.solve(solver=SCS, use_indirect=True, verbose=True)


# import numpy as np
# from cvxpy import *
# np.random.seed(1)
# m = 200
# n = 100
# x = Variable(n)
# A = np.random.randn(m, n)
# b = np.random.randn(m, 1)
# cost = max_entries(A*x - b)
# prob = Problem(Minimize(cost), [x >= 0])
# prob.solve(solver=POGS, rho=1e-3, verbose=True, abs_tol=1e-4, rel_tol=1e-3, max_iters=10000)
#     # samples=200, equil_steps=0)
# print "POGS obj", prob.value
# print "POGS cost", cost.value
# print "ECOS obj", prob.solve(solver=ECOS)
# print "SCS obj", prob.solve(solver=SCS, use_indirect=True, verbose=False)
# print "SCS cost", cost.value

import numpy as np
from cvxpy import *
np.random.seed(1)
m = 200
n = 100
x = Variable(n)
A = np.random.randn(m, n)
b = np.random.randn(m, 1)
# Doesn't need sum_entries because in the objective.
cost = norm(A*x - b) + norm(x, 1)
prob = Problem(Minimize(cost))
prob.solve(solver=MAT_FREE_POGS, rho=1, verbose=True, abs_tol=1e-4, rel_tol=1e-3, max_iter=10000,
    samples=200, equil_steps=0)
print "POGS obj", prob.value
print "POGS cost", cost.value
print "POGS x norm", norm(x).value
# prob.solve(solver=MAT_FREE_SCS, verbose=True, eps=1e-3, max_iters=10000,
#     samples=200, equil_steps=1)
# print "MAT FREE SCS obj", prob.value
# print "MAT FREE SCS  cost", cost.value
# print "MAT FREE SCS x norm", norm(x).value
# print "ECOS obj", prob.solve(solver=ECOS)
# print "ECOS cost", cost.value
# print "ECOS x norm", norm(x).value
# print "SCS obj", prob.solve(solver=SCS, use_indirect=True, verbose=False)
# print "SCS cost", cost.value
# print "SCS x norm", norm(x).value

