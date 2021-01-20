#!/bin/bash
#PBS -N mnn_qy
#PBS -q workq
#PBS -l nodes=1:ppn=1
#PBS -k oe

##PBS -l walltime=00:59:59
##PBS -l mem=1500MB
##PBS -V
##PBS -e /home1/yang_qi/MNN/PBSout/
##PBS -o /home1/yang_qi/MNN/PBSout/
##PBS -J 1-3

set -x
cd $PBS_O_WORKDIR

eval "$(conda shell.bash hook)"
conda activate qy

echo "$PBS_O_WORKDIR"
echo "$PBS_ARRAY_INDEX"

EXP_ID=`echo $PBS_JOBID | sed 's/\[[^]]*\]//'`
echo $EXP_ID

sleep $PBS_ARRAY_INDEX

python3 batch_processor.py $PBS_ARRAY_INDEX $EXP_ID


