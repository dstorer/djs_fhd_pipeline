#!/bin/sh

echo "JOB START TIME" `date +"%Y-%m-%d_%H:%M:%S"`

echo "Processing"

echo $IDL_PATH
source ~/.bashrc
echo $IDL_PATH
echo ${PBS_ARRAYID}

echo "------ ARGS ------"
echo "obs_file: ${obs_file_name}"
echo "outdir: ${outdir}"
echo "version_str: ${version_str}"
echo "band: ${band}"
echo "exants file: ${exants}"

python /lustre/aoc/projects/hera/dstorer/Projects/updatedHeraOnFHD/runScripts/runFHD/run_fhd_h4c_singleObs.py ${obs_file_name} ${outdir} ${version_str} ${PBS_ARRAYID} ${band} ${exants}

echo "JOB END TIME" `date +"%Y-%m-%d_%H:%M:%S"`
