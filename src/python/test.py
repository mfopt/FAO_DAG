from cvxpy import *
x = Variable()
# prob = Problem(Minimize(-x), [x <= 2])
# prob.solve(solver=POGS, verbose=True, max_iter=2500)

prob = Problem(Minimize(-x), [x <= 2])
prob.solve(solver=MAT_FREE_POGS, verbose=True, max_iter=2500)
print x.value
print "MAT FREE POGS double obj", prob.value
prob.solve(solver=MAT_FREE_POGS, verbose=True, max_iter=2500, double=False)
print x.value
print "MAT FREE POGS float obj", prob.value
prob.solve(solver=POGS, rho=1, verbose=True, max_iters=2499)
print "POGS obj", prob.value
# prob = Problem(Minimize(-x), [x <= 2])
# prob.solve(solver=POGS, verbose=True, max_iter=2)

# prob = Problem(Minimize(x), [abs(x) <= 2])
# prob.solve(solver=MAT_FREE_SCS, verbose=True, max_iters=100,
#     samples=200, equil_steps=1)

x = Variable(2)
y = Variable(2)
prob = Problem(Minimize(x[0]), [norm(x + y) <= 2, y == 0])
prob.solve(solver=MAT_FREE_POGS, verbose=True, max_iters=2500,
    samples=200, equil_steps=0, rho=1)
print x.value
print "MAT FREE POGS double obj", prob.value
prob.solve(solver=MAT_FREE_POGS, verbose=True, max_iters=2500,
    samples=200, equil_steps=0, rho=1, double=False)
print x.value
print "MAT FREE POGS float obj", prob.value
prob.solve(solver=POGS, rho=1, verbose=True, max_iters=2499)
print "POGS obj", prob.value
# print norm(x + y).value
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
# print prob.solve(solver=POGS, rho=1e-3, verbose=True)
# print cost.value
# print prob.solve(solver=ECOS)
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
# prob.solve(solver=POGS, rho=1, verbose=True, abs_tol=1e-4, max_iter=10000)
# print "POGS obj", prob.value
# print "POGS cost", cost.value
# print "ECOS obj", prob.solve(solver=ECOS)
# print "SCS obj", prob.solve(solver=SCS, use_indirect=True, verbose=True)
# print "SCS cost", cost.value

import numpy as np
from cvxpy import *
np.random.seed(1)
m = 200
n = 100
x = Variable(n)
A = np.random.randn(m, n)
b = np.random.randn(m, 1)
cost = norm(A*x - b) + norm(x, 1)
prob = Problem(Minimize(cost))
prob.solve(solver=MAT_FREE_POGS, rho=1, verbose=True, abs_tol=1e-3, rel_tol=1e-3, max_iters=2499, double=False)
print "MAT FREE POGS float obj", prob.value
print "MAT FREE POGS float cost", cost.value
prob.solve(solver=MAT_FREE_POGS, rho=1, verbose=True, abs_tol=1e-3, rel_tol=1e-3, max_iters=2499)
print "MAT FREE POGS double obj", prob.value
print "MAT FREE POGS double cost", cost.value
prob.solve(solver=POGS, rho=1, verbose=True, abs_tol=1e-3, rel_tol=1e-3, max_iters=2499)
print "POGS obj", prob.value
print "POGS cost", cost.value
print "ECOS obj", prob.solve(solver=ECOS)
prob.solve(solver=SCS, use_indirect=True, verbose=True)
print "SCS obj", prob.value
print "SCS cost", cost.value
