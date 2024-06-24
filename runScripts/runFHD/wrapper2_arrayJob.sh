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
echo "case_name: ${case_name}"
echo "band: ${band}"
echo "exants file: ${exants}"
echo "init_cal: ${init_cal}"
echo "IDL license: ${license}"

python -u /lustre/aoc/projects/hera/dstorer/Setup/djs_fhd_pipeline/runScripts/runFHD/run_fhd_h6c_arrayJob.py ${obs_file_name} ${outdir} ${version_str} ${case_name} ${SLURM_ARRAY_TASK_ID} ${band} ${exants} ${init_cal} ${license}

echo "JOB INFO"
squeue -j $SLURM_JOBID

seff $SLURM_JOBID
echo "JOB END TIME" `date +"%Y-%m-%d_%H:%M:%S"`
