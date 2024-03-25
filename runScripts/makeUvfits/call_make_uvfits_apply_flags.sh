#!/bin/sh

echo "JOB START TIME" `date +"%Y-%m-%d_%H:%M:%S"`

echo "There are $N_obs observations in this file"

echo "----------------------- ARGS -----------------------"
echo "raw_file: ${obs_file_name}"
echo "xants_file: ${xants}"
echo "ssins_files: ${ssins_files}"
echo "outdir: ${outdir}"
echo "per_pol: ${per_pol}"
echo "band: ${band}"
echo "N_combine: ${N_combine}"
echo "internode_only: ${internode_only}"
echo "intersnap_only: ${intersnap_only}"
echo "write_minis: ${write_minis}"
echo "num_times: ${num_times}"
echo "array_job: ${array_job}"
echo "phase: ${phase}"



python -u /lustre/aoc/projects/hera/dstorer/Setup/djs_fhd_pipeline/runScripts/makeUvfits/make_uvfits_with_flags.py -f ${obs_file_name} -s ${ssins_files} -o ${outdir} -N ${N_combine} -x ${xants} -p ${per_pol} -b ${band} -I ${internode_only} -S ${intersnap_only} -m ${write_minis} -n ${num_times} --ind ${SLURM_ARRAY_TASK_ID} -a ${array_job} -e ${phase}

echo "JOB INFO"
squeue -j $SLURM_JOBID
echo "JOB END TIME" `date +"%Y-%m-%d_%H:%M:%S"`
