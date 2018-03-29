#!/bin/bash

source /home/mpuer/lsc/SEOBNRv4T_surrogate_lalinference_o2/etc/lalsuiterc
mpirun -np 24 python ./Compute_matches_SEOBNRv4T_SEOBNRv4T_surrogate_MPI.py > Compute_matches_SEOBNRv4T_SEOBNRv4T_surrogate_MPI.log 2>&1 &
