import numpy as np

# generate random points in q, spin1z, spin2z, lambda1, lambda2
def random_point():
    q = 1.0/np.random.uniform(1.0, 3.0)
    chi1 = np.random.uniform(0., 0.7)
    chi2 = np.random.uniform(0., 0.7)
    Lambda1 = np.random.uniform(0., 10000.0)
    Lambda2 = np.random.uniform(0., 10000.0)
    return np.array([q, chi1, chi2, Lambda1, Lambda2])

N = 3000
g = np.array([random_point() for i in np.arange(N)])

np.savetxt('random_5D_wide_%d.txt'%N, g)

