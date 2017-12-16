#!/bin/bash

#source /home/mpuer/lsc/TEOBBNS/etc/lalsuiterc
source ~/pe/master.sh
export LAL_DATA_PATH=/home/mpuer/SEOBNRROMData:$LAL_DATA_PATH

export PYTHONPATH=$PYTHONPATH:/home/mpuer/projects/

LOG=timings.log
python timings.py > ${LOG} 2>&1 &
