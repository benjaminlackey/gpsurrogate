#!/bin/bash

source /home/mpuer/SEOBNRv4T/etc/lalsuite-user-env.sh

export PYTHONPATH=$PYTHONPATH:/home/mpuer/projects/

SDIR=/home/mpuer/projects/gpsurrogate/src/TEOB_generation/
RDIR=/home/mpuer/projects/gpsurrogate/src/TEOB_generation/runs/QM
CFG=${RDIR}/test_Dec5_20Hz.json
LOG=${RDIR}/test_Dec5_20Hz.log

cd ${SDIR}
mpirun -n 16 Generate_TEOB_TD_MPI.py -o ${CFG} > ${LOG} 2>&1 &

