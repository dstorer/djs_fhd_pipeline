#!/bin/sh
#
#SBATCH --mail-type=END
#SBATCH --mail-user=darajstorer@gmail.com

echo "JOB START TIME" `date +"%Y-%m-%d_%H:%M:%S"`

echo "There are $N_obs observations in this file"

echo "Processing"


stk_sig=20
oth_sig=5
t_aggro=0.5
internode_only=0
intersnap_only=1
clobber=1
n_combine=10

echo "streak_sig = ${stk_sig}"
echo "other_sig = ${oth_sig}"
echo "t_aggro = ${t_aggro}"
echo "xants_file: ${xants}"
echo "shape_file: ${shape_dict}"
echo "internode_only: ${internode_only}"
echo "intersnap_only: ${intersnap_only}"
echo "obs_file_name = ${obs_file_name}"
echo "clobber: ${clobber}"
echo "Calling /lustre/aoc/projects/hera/dstorer/Projects/updatedHeraOnFHD/runScripts/runSSINS/Run_HERA_SSINS.py"

python -u /lustre/aoc/projects/hera/dstorer/Projects/updatedHeraOnFHD/runScripts/runSSINS/Run_HERA_SSINS.py -f ${obs_file_name} -s ${stk_sig} -o ${oth_sig} -p ${outdir} -a ${xants} -d ${shape_dict} -t ${t_aggro} -I ${internode_only} -S ${intersnap_only} -c ${clobber} -n ${n_combine}


echo "JOB INFO"
squeue -j $SLURM_JOBID
echo "JOB END TIME" `date +"%Y-%m-%d_%H:%M:%S"`