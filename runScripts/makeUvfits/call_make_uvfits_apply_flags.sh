#!/bin/sh
#
#SBATCH --mail-type=END
#SBATCH --mail-user=darajstorer@gmail.com

echo "JOB START TIME" `date +"%Y-%m-%d_%H:%M:%S"`

echo "There are $N_obs observations in this file"

echo "xants_file: ${xants}"

echo "xants_file: ${xants_file}"
echo "ssins_files: ${ssins_files}"
echo "shape_file: ${shape_file}"
echo "band: ${band}"
echo "N_combine: ${N_combine}"
echo "internode_only: ${internode_only}"
echo "intersnap_only: ${intersnap_only}"
echo "write_minis: ${write_minis}"
echo "num_times: ${num_times}"

echo "Calling /lustre/aoc/projects/hera/dstorer/Projects/updatedHeraOnFHD/runScripts/makeUvfits/make_uvfits_with_flags.py"


python /lustre/aoc/projects/hera/dstorer/Projects/updatedHeraOnFHD/runScripts/makeUvfits/make_uvfits_with_flags.py -f ${obs_file_name} -s ${ssins_files} -o ${outdir} -N ${N_combine} -x ${xants} -b ${band} -I ${internode_only} -S ${intersnap_only} -m ${write_minis} -n ${num_times}


echo "JOB END TIME" `date +"%Y-%m-%d_%H:%M:%S"`