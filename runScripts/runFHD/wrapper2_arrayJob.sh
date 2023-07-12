#!/bin/sh

echo "band: ${band}"
echo "JOB START TIME" `date +"%Y-%m-%d_%H:%M:%S"`
echo "Hello Dara"
echo "Processing"

echo $IDL_PATH
source ~/.bashrc
echo $IDL_PATH
echo $SLURM_ARRAY_TASK_ID

echo "------ ARGS ------"
echo "obs_file: ${obs_file_name}"
echo "outdir: ${outdir}"
echo "version_str: ${version_str}"
echo "band: ${band}"
echo "exants file: ${exants}"
echo "IDL license: ${license}"

python -u /lustre/aoc/projects/hera/dstorer/Setup/djs_fhd_pipeline/runScripts/runFHD/run_fhd_h4c_arrayJob.py ${obs_file_name} ${outdir} ${version_str} ${SLURM_ARRAY_TASK_ID} ${band} ${exants} ${license}

echo "JOB INFO"
squeue -j $SLURM_JOBID

seff $SLURM_JOBID
echo "JOB END TIME" `date +"%Y-%m-%d_%H:%M:%S"`
