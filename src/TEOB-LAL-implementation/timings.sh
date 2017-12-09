#!/bin/bash

source /home/mpuer/lsc/TEOBBNS/etc/lalsuiterc

export PYTHONPATH=$PYTHONPATH:/home/mpuer/projects/

LOG=timings.log
python timings.py > ${LOG} 2>&1 &
