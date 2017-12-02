#!/bin/bash

source /home/mpuer/lsc/TEOBBNS/etc/lalsuiterc

export PYTHONPATH=$PYTHONPATH:/home/mpuer/projects/

SDIR=/home/mpuer/projects/gpsurrogate/src/TEOB_generation/
RDIR=/home/mpuer/projects/gpsurrogate/src/TEOB_generation/runs/set1
CFG=${RDIR}/test_refine_iteration1_20Hz.json
LOG=${RDIR}/test_refine_iteration1_20Hz.log

cd ${SDIR}
mpirun -n 24 Generate_TEOB_TD_MPI.py -o ${CFG} > ${LOG} 2>&1 &
