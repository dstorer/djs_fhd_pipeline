#!/bin/bash


while getopts ":f:s:o:x:a:b:c:" option
do
  case $option in
    # A text file where each line is a path to a raw data file
    f) obs_file_name=$OPTARG;;
    # A text file where each line is a path to a sssins flags file
    s) ssins_files=$OPTARG;;
    # The output directory for the error log
    o) outdir=$OPTARG;;
    # A yaml containing a list of antennas to exclude from the data
    x) xants_file=$OPTARG;;
    a) startind=$OPTARG;;
    b) stopind=$OPTARG;;
    c) nsim=$OPTARG;;
    \?) echo "Unknown option: Accepted flags are -f (obs_file_name),"
        exit 1;;
    :) echo "Missing option argument for input flag"
       exit 1;;
  esac
done


#Throw error if no obs_id file.
if [ -z ${obs_file_name} ]; then
   echo "Need to specify a full filepath to a list of viable datapaths."
   exit 1
fi

#Throw error if no output directory
if [ -z ${outdir} ]; then
    echo "Need to specify an output directory for the error log"
fi
N_obs=$(wc -l < $obs_file_name)

echo "calling /lustre/aoc/projects/hera/dstorer/Setup/djs_fhd_pipeline/runScripts/makeUvfits/call_make_uvfits_apply_flags.sh"

#xants_file="/lustre/aoc/projects/hera/dstorer/Projects/updatedHeraOnFHD/2459854/exants.yml"
shape_file="/lustre/aoc/projects/hera/dstorer/Setup/djs_fhd_pipeline/runScripts/runSSINS/HERA_shape_dict.yml"
band="mid"
N_combine=10
internode_only=0
intersnap_only=1
write_minis=0
num_times=1
array_job=1


echo "Submitting array job"
echo "startind: ${startind}"
echo "stopind: ${stopind}"
echo "nsim: ${nsim}"

sbatch --export=obs_file_name=${obs_file_name},ssins_files=${ssins_files},outdir=${outdir},N_combine=${N_combine},xants=${xants_file},band=${band},internode_only=${internode_only},intersnap_only=${intersnap_only},write_minis=${write_minis},num_times=${num_times},array_job=${array_job} -p hera -o ${outdir}/make_uvfits_%a.out -N 1 -n 1 --mem=128G -J make_uvfits --array=${startind}-${stopind}%${nsim} /lustre/aoc/projects/hera/dstorer/Setup/djs_fhd_pipeline/runScripts/makeUvfits/call_make_uvfits_apply_flags.sh

