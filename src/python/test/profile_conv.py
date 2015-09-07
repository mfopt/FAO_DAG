#!/usr/bin/env python

from cvxpy import *
import numpy as np
import random
from math import pi, sqrt, exp
import sys
import os

REPS = 10

# script_num = int(sys.argv[1])

def gauss(n=11,sigma=1, scale=1, min_val=1):
    r = range(-int(n/2),int(n/2)+1)
    return [max(scale /(sigma * sqrt(2*pi)) * exp(-float(x)**2/(2*sigma**2)), min_val) for x in r]


np.random.seed(5)
random.seed(5)
n_vals = []
for n in np.logspace(3, 7, 10):
    n = int(n)
    if n % 2 == 1:
        n -= 1
    n_vals.append(n)
    ecos_times = []
    mat_free_times = []
    scs_direct_times = []
    scs_indirect_times = []

    NUM_SPIKES = 5
    DENSITY = NUM_SPIKES/n
    x = Variable(n)
    # Create sparse signal.
    HEIGHT = n/10
    true_x = np.zeros((n,1))
    spikes = random.sample(xrange(n), NUM_SPIKES)
    for idx in spikes:
        true_x[idx] = random.uniform(0, HEIGHT)

    # Gaussian kernel.
    m = n+1
    kernel = gauss(m, m/10, 1, 1e-6)
    #kernel = np.sinc(np.linspace(-m/100, m/100, m))

    # Noisy signal.
    SNR = 20
    signal = conv(kernel, true_x)
    sigma = norm(signal,2).value/(SNR*sqrt(n+m-1))
    noise = np.random.normal(scale=sigma, size=n+m-1)
    # print("SNR", norm(signal,2).value/norm(noise,2).value)
    noisy_signal = signal + noise

    gamma = Parameter(sign="positive")
    gamma.value = 0
    fit = norm(conv(kernel, x) - noisy_signal)
    constraints = [x >= 0]
    prob = Problem(Minimize(fit),
                   constraints)
    # print "True fit =", norm(conv(kernel, true_x) - noisy_signal).value
    # print "all zero fit =", norm(conv(kernel, np.zeros(n)) - noisy_signal).value

    # # Only solve one problem.
    # if r != script_num:# or n <= 16236:
    #     continue
    if False:
        result = prob.solve(solver=ECOS, verbose=True,
            # abstol=1e-3, reltol=1e-3, feastol=1e-3)
            abstol=1e-5, reltol=1e-5, feastol=1e-5)
        print "ecos result", result
        print("recovered x fit", fit.value)
        print("nnz =", np.sum(x.value >= 1))
        print("max =", np.max(np.max(x.value)))
        print("solve time", prob.solve_time)
        solve_time = prob.solve_time
    else:
        solve_time = 0
    # ecos_times.append(solve_time)
    if False:
        result = prob.solve(solver=SCS,
                            verbose=True,
                            max_iters=10000,
                            eps=1e-5,
                            use_indirect=False)
        print "scs direct result", result
        print("solve time", prob.solve_time)
        solve_time = prob.solve_time
        print "mat free scs result", result
        print("solve time", prob.solve_time)
        print("nnz =", np.sum(x.value >= 1))
        print("max =", np.max(np.max(x.value)))
    # else:
    #     solve_time = 0
    # scs_direct_times.append(solve_time)
    # # if False:
    # #     result = prob.solve(solver=SCS,
    # #                         verbose=True,
    # #                         max_iters=10000,
    # #                         eps=1e-3,
    # #                         use_indirect=True)
    # #     print "scs indirect result", result
    # #     print("recovered x fit", fit.value)
    # #     print("solve time", prob.solve_time)
    # #     solve_time = prob.solve_time
    # # else:
    # #     solve_time = 0
    # # scs_indirect_times.append(solve_time)
    if False:
        print "OLD MAT FREE"
        result = prob.solve(solver=OLD_SCS_MAT_FREE,
                            verbose=True,
                            max_iters=1,
                            eps=1e-3,
                            equil_steps=1,
                            cg_rate=2,
                            precond=True,
                            stoch=True,
                            samples=200)
        print "old mat free result", result
        print prob.cg_iters
        print("recovered x fit", fit.value)
        print("solve time", prob.solve_time)
        print("nnz =", np.sum(x.value >= 1))
        print("max =", np.max(np.max(x.value)))
        solve_time = prob.solve_time
    else:
        solve_time = 0
    scs_indirect_times.append(solve_time)
    # print("true signal fit", fit.value)
    if True:
        print "FAO MAT FREE"
        # import cProfile
        # cProfile.run("""result = prob.solve(solver=MAT_FREE_SCS,
        #                     verbose=True,
        #                     max_iters=10000,
        #                     equil_steps=1,
        #                     eps=1e-3,
        #                     cg_rate=2,
        #                     precond=True,
        #                     stoch=True,
        #                     samples=200)
        # """)
        # import yep
        # yep.start('conv.prof')
        result = prob.solve(solver=MAT_FREE_SCS,
                            verbose=False,
                            max_iters=100,
                            equil_steps=1,
                            eps=1e-3,
                            cg_rate=2,
                            precond=True,
                            stoch=True,
                            samples=200)
        # yep.stop()
        # print prob.cg_iters
        # print "mat free scs result", result
        # print("solve time", prob.solve_time)
        # print("nnz =", np.sum(x.value >= 1))
        # print("max =", np.max(np.max(x.value)))
        # print("recovered x fit", fit.value)
        solve_time = prob.solve_time
    else:
        solve_time = 0
    mat_free_times.append(solve_time)

    if False:
        print "FAO MAT FREE"
        # import cProfile
        # cProfile.run("""result = prob.solve(solver=MAT_FREE_SCS,
        #                     verbose=True,
        #                     max_iters=10000,
        #                     equil_steps=1,
        #                     eps=1e-3,
        #                     cg_rate=2,
        #                     precond=True,
        #                     stoch=True,
        #                     samples=200)
        # """)
        # import yep
        # yep.start('conv.prof')
        result = prob.solve(solver=MAT_FREE_POGS,
                            verbose=False,
                            max_iters=200,
                            equil_steps=0,
                            # eps=1e-3,
                            # cg_rate=2,
                            # precond=True,
                            # stoch=True,
                            abs_tol=1e-4,
                            rel_tol=1e-4,
                            samples=200,
                            double=False)
        # yep.stop()
        # print "mat free pogs result", result
        # print("recovered x fit", fit.value)
        # print("nnz =", np.sum(x.value >= 1))
        # print("max =", np.max(np.max(x.value)))
