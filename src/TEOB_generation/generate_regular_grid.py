import numpy as np

# generate regular grid in q, spin1z, spin2z, lambda1, lambda2
N = 8
qs = 1.0/np.linspace(1.0, 2.0, N)
Lambdas = np.linspace(0.0, 5000.0, N)
#chis = np.array([0.0])
chis = np.linspace(-0.5, 0.5, N)

qi, c1i, c2i, L1i, L2i = np.meshgrid(qs, chis, chis, Lambdas, Lambdas, indexing='ij')
g = np.vstack([qi.flatten(), c1i.flatten(), c2i.flatten(), L1i.flatten(), L2i.flatten()]).T

np.savetxt('regular_nonspinning_5D_%d.txt'%N, g)

