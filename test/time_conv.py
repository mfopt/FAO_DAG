#!/usr/bin/env python

from cvxpy import *
import numpy as np
import random
from math import pi, sqrt, exp
import sys
import os

REPS = 10

script_num = int(sys.argv[1])

def gauss(n=11,sigma=1, scale=1, min_val=1):
    r = range(-int(n/2),int(n/2)+1)
    return [max(scale /(sigma * sqrt(2*pi)) * exp(-float(x)**2/(2*sigma**2)), min_val) for x in r]

with open("2scale_bc_times%s.csv" % script_num, "w") as f:
    f.write("n,ecos_time,scs_direct_time,old_mat_free_time,fao_mat_free_time\n")

    np.random.seed(5)
    random.seed(5)
    n_vals = []
    for n in np.hstack([np.logspace(2, 5, 20), np.logspace(5, 7, 5)[1:]]):
        n = int(n)
        if n % 2 == 1:
            n -= 1
        n_vals.append(n)
    ecos_times = []
    mat_free_times = []
    scs_direct_times = []
    scs_indirect_times = []
    for n in n_vals:
        for r in range(REPS):
            print("n=",n)
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
            print("SNR", norm(signal,2).value/norm(noise,2).value)
            noisy_signal = signal + noise

            gamma = Parameter(sign="positive")
            gamma.value = 0
            fit = norm(conv(kernel, x) - noisy_signal)
            constraints = [x >= 0]
            prob = Problem(Minimize(fit),
                           constraints)
            # Only solve one problem.
            if r != script_num:# or n <= 16236:
                continue
            if True and n <= 5454:
                result = prob.solve(solver=ECOS, verbose=True)
                    #abstol=1e-3, reltol=1e-3, feastol=1e-3)
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
            if True:
                result = prob.solve(solver=SCS,
                                    verbose=True,
                                    max_iters=10000,
                                    eps=1e-3,
                                    use_indirect=True)
                print "scs indirect result", result
                print("recovered x fit", fit.value)
                print("solve time", prob.solve_time)
                solve_time = prob.solve_time
            # else:
            #     solve_time = 0
            # scs_indirect_times.append(solve_time)
            if True:
                print "OLD MAT FREE"
                result = prob.solve(solver=OLD_SCS_MAT_FREE,
                                    verbose=True,
                                    max_iters=10000,
                                    eps=1e-3,
                                    equil_steps=1,
                                    cg_rate=2,
                                    precond=True,
                                    stoch=True,
                                    samples=200)
                print "old scs mat free result", result
                print("recovered x fit", fit.value)
                print("solve time", prob.solve_time)
                solve_time = prob.solve_time
            else:
                solve_time = 0
            scs_indirect_times.append(solve_time)
            print("true signal fit", fit.value)
            if True:
                print "FAO MAT FREE"
                result = prob.solve(solver=MAT_FREE_SCS,
                                    verbose=True,
                                    max_iters=10000,
                                    equil_steps=1,
                                    eps=5e-4,
                                    cg_rate=2,
                                    precond=True,
                                    stoch=True,
                                    samples=200)
                print "mat free scs result", result
                print("solve time", prob.solve_time)
                solve_time = prob.solve_time
            else:
                solve_time = 0
            mat_free_times.append(solve_time)
            print("recovered x fit", fit.value)

            print("nnz =", np.sum(x.value >= 1))
            print("max =", np.max(np.max(x.value)))
            f.write("%s,%s,%s,%s,%s\n" % (n, ecos_times[-1], scs_direct_times[-1],
                                          scs_indirect_times[-1], mat_free_times[-1]))
            f.flush()

# # Plot result and fit.
# # Assumes convolution kernel is centered around m/2.
# t = range(n+m-1)
# import matplotlib.pyplot as plt
# plt.figure()
# ax1 = plt.subplot(2, 1, 1)
# true_x_padded = np.vstack([np.zeros((m/2,1)), true_x, np.zeros((m/2,1))])
# x_padded = np.vstack([np.zeros((m/2,1)), x.value[:,0], np.zeros((m/2,1))])
# lns1 = ax1.plot(t, true_x_padded, label="true x")
# ax1.set_ylabel('true x')
# ax2 = ax1.twinx()
# lns2 = ax2.plot(t, x_padded, label="recovered x", color="red")
# ax2.set_ylabel('recovered x')
# ax2.set_ylim([0, np.max(x_padded)])

# # added these three lines
# lns = lns1+lns2
# labs = [l.get_label() for l in lns]
# ax1.legend(lns, labs, loc='upper right')

# plt.subplot(2, 1, 2)
# plt.plot(t, np.asarray(noisy_signal.value[:, 0]), label="signal")
# plt.legend(loc='upper right')
# plt.show()
