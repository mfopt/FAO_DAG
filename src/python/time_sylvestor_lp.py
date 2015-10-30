#!/usr/bin/env python

from cvxpy import *
import numpy as np
import random
from math import pi, sqrt, exp
import sys
import os

REPS = 10

script_num = int(sys.argv[1])

# x = Variable()
# prob = Problem(Maximize(x), [x <= 1])
# prob.solve(solver=SCS_MAT_FREE, max_iters=2500, verbose=True)
# print "result =", prob.value
# print "x value =", x.value

with open("sylvester_pogs_times%s.csv" % script_num, "w") as f:
    f.write("n,mat_free_pogs_float_time,mat_free_pogs_float_evals,mat_free_pogs_double_time,mat_free_pogs_double_evals\n")

    np.random.seed(5)
    random.seed(5)
    n_vals = [int(n) for n in np.logspace(1, 3, 10)]
    ecos_times = []
    scs_direct_times = []
    scs_indirect_times = []
    mat_free_pogs_times = []
    for n in n_vals:
        for r in range(REPS):
            print("n=",n)
            K = 5
            m = K*n
            X = Variable(m, n)
            A = np.abs(np.random.randn(m, m)) + 1e-6
            B = np.abs(np.random.randn(n, n)) + 1e-6
            C = np.random.randn(m, n)
            # D = np.abs(np.random.randn(m, m)) + 1e-6

            # Only solve one problem.
            if r != script_num:
                continue

            print "nnz = ", (m*n)**2

            cost = trace(C.T*X)
            prob = Problem(Minimize(cost),
                            [X >= 0, A*X*B <= 1])
            if False and n <= 5454:
                result = prob.solve(solver=ECOS, verbose=True,
                    abstol=1e-3, reltol=1e-3, feastol=1e-3)
                print "ecos result", result
                print("recovered x fit", fit.value)
                print("solve time", prob.solve_time)
                solve_time = prob.solve_time
            else:
                solve_time = 0
            ecos_times.append(solve_time)
            if False and n <= 16236:
                result = prob.solve(solver=SCS,
                                    verbose=True,
                                    max_iters=10000,
                                    eps=1e-3,
                                    use_indirect=False)
                print "scs direct result", result
                print("recovered x fit", fit.value)
                print("solve time", prob.solve_time)
                solve_time = prob.solve_time
            else:
                solve_time = 0
            scs_direct_times.append(solve_time)
            if False:
                result = prob.solve(solver=SCS,
                                    verbose=True,
                                    max_iters=10000,
                                    eps=1e-3,
                                    use_indirect=True)
                print "scs indirect result", result
                print("recovered x fit", fit.value)
                print("solve time", prob.solve_time)
                solve_time = prob.solve_time
            else:
                solve_time = 0
            scs_indirect_times.append(solve_time)
            # if True:
                # result = prob.solve(solver=MAT_FREE_SCS,
                #                     verbose=False,
                #                     max_iters=10000,
                #                     equil_steps=1,
                #                     eps=1e-3,
                #                     cg_rate=2,
                #                     precond=True,
                #                     stoch=True,
                #                     samples=200,
                #                     equil_gamma=1e-8)
                # print "mat free scs result", result
                # print("solve time", prob.solve_time)
                # print "evals", prob.A_evals + prob.AT_evals
                # evals = prob.A_evals + prob.AT_evals
                # solve_time = prob.solve_time
            double_solve_time = 0
            double_evals = 0
            if False:
                result = prob.solve(solver=MAT_FREE_POGS,
                                    adaptive_rho=True,
                                    verbose=True,
                                    max_iters=10000,
                                    equil_steps=1,
                                    abs_tol=1e-4,
                                    rel_tol=1e-4,
                                    samples=200,
                                    double=True,
                                    rho=2)
                print "MAT FREE POGS double result", result
                print "MAT FREE POGS double relative result", result/(norm(C).value*norm(X).value)
                print "MAT FREE POGS double solve time", prob.solve_time
                print "MAT FREE POGS double evals", prob.A_evals + prob.AT_evals
                double_solve_time = prob.solve_time
                double_evals = prob.A_evals + prob.AT_evals
            if True:
                result = prob.solve(solver=MAT_FREE_POGS,
                                    adaptive_rho=True,
                                    verbose=True,
                                    max_iters=10000,
                                    equil_steps=1,
                                    abs_tol=1e-4,
                                    rel_tol=1e-4,
                                    samples=200,
                                    double=False,
                                    rho=2)
                print "MAT FREE POGS float result", result
                print "MAT FREE POGS float relative result", result/(norm(C).value*norm(X).value)
                print "MAT FREE POGS float solve time", prob.solve_time
                print "MAT FREE POGS float evals", prob.A_evals + prob.AT_evals
                float_solve_time = prob.solve_time
                float_evals = prob.A_evals + prob.AT_evals
                f.write("%s,%s,%s,%s,%s\n" % (n, float_solve_time, float_evals,
                                              double_solve_time, double_evals))
                f.flush()
            else:
                solve_time = 0
            mat_free_pogs_times.append(solve_time)
            # print("recovered x fit", fit.value)

            # print("nnz =", np.sum(x.value >= 1))
            # print("max =", np.max(np.max(x.value)))
            # f.write("%s,%s,%s,%s,%s\n" % (n, ecos_times[-1], scs_direct_times[-1],
            #                             scs_indirect_times[-1], mat_free_times[-1]))
            # f.flush()
