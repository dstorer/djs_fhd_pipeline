#!/bin/bash


while getopts ":f:o:x:a:" option
do
  case $option in
    # A text file where each line is an obsid
    f) obs_file_name="$OPTARG";;
    # The output directory for the error log
    o) outdir=$OPTARG;;
    x) xants_file=$OPTARG;;
    a) arr_str=$OPTARG;;
    \?) echo "Unknown option: Accepted flags are -f (obs_file_name),"
        exit 1;;
    :) echo "Missing option argument for input flag"
       exit 1;;
  esac
done

#Manual shift to the next flag.
shift $(($OPTIND - 1))

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

echo "calling /lustre/aoc/projects/hera/dstorer/Setup/djs_fhd_pipeline/runScripts/runSSINS/call_Run_HERA_SSINS.sh"

# xants_file="/lustre/aoc/projects/hera/dstorer/Projects/updatedHeraOnFHD/2459855/exants.yml"
shape_file="/lustre/aoc/projects/hera/dstorer/Setup/djs_fhd_pipeline/runScripts/runSSINS/HERA_shape_dict.yml"

#echo "obs_file_name = ${obs_file_name}"

array_job=1
echo "Submitting array job"
echo "Array String: ${arr_str}"

sbatch --export=obs_file_name=${obs_file_name},N_obs=${N_obs},outdir=${outdir},xants=${xants_file},shape_dict=${shape_file} -p hera -o ${outdir}/SSINS_%a.out --mem=60G -J SSINS_Flagging_N --array=${arr_str} /lustre/aoc/projects/hera/dstorer/Setup/djs_fhd_pipeline/runScripts/runSSINS/call_Run_HERA_SSINS.sh
