#!/bin/bash

source /home/mpuer/lsc/DB_NRTides/etc/lalsuiterc
mpirun -np 24 python ./Compute_matches_TEOBv4_SEOBNRv4_ROM_NRTidal_MPI.py > Compute_matches_TEOBv4_SEOBNRv4_ROM_NRTidal_MPI.log 2>&1 &
