#!/bin/bash

source /home/mpuer/lsc/TEOBBNS/etc/lalsuiterc

export PYTHONPATH=$PYTHONPATH:/home/mpuer/projects/

SDIR=/home/mpuer/projects/gpsurrogate/src/TEOB_generation/
RDIR=/home/mpuer/projects/gpsurrogate/src/TEOB_generation/runs
CFG=${RDIR}/test.json
LOG=${RDIR}/test.log

cd ${SDIR}
mpirun -n 24 Generate_TEOB_TD_MPI.py -o ${CFG} > ${LOG} 2>&1 &
