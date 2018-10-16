#!/bin/bash

source /home/mpuer/lsc/SEOBNRv4T_surrogate_master/etc/lalsuite-user-env.sh
export LAL_DATA_PATH=/home/mpuer/SEOBNRROMData:$LAL_DATA_PATH

export PYTHONPATH=$PYTHONPATH:/home/mpuer/projects/

LOG=timings_surrogate.log
python timings_surrogate.py > ${LOG} 2>&1 &
