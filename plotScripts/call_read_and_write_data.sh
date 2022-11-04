#!/bin/sh

echo "JOB START TIME" `date +"%Y-%m-%d_%H:%M:%S"`

echo "raw_files: ${raw_files}"
echo "fhd_files: ${fhd_files}"
echo "ssins_files: ${ssins_files}"
echo "outdir: ${outdir}"
echo "exants: ${exants}"
echo "JD: ${jd}"

python -u /lustre/aoc/projects/hera/dstorer/Setup/djs_fhd_pipeline/plotScripts/read_and_write_data.py -r ${raw_files} -f ${fhd_files} -s ${ssins_files} -o ${outdir} -x ${exants} -j ${jd} -R ${RAW} -S ${SSINS} -C ${CAL} -M ${MODEL} -J ${JDS} -B ${BLS} -L ${LSTS}

echo "JOB INFO"
squeue -j $SLURM_JOBID
echo "JOB END TIME" `date +"%Y-%m-%d_%H:%M:%S"`