#!/bin/bash

while getopts ":f:o:v:n:b:a:x:l:" option
do
  case $option in
    # A text file where each line is an obsid
    f) obs_file_name="$OPTARG";;
    # The output directory for the error log
    o) outdir="$OPTARG";;
    # The run version
    v) version_str=$OPTARG;;
    # A number or prefix to include in the name of the output log
    n) num_prefix=$OPTARG;;
    b) band=$OPTARG;;
    a) exants=$OPTARG;;
    # Array job string (start-stop%interval)
    x) x=$OPTARG;;
    # Specify IDL license to use - either NRAO or HERA
    l) license=$OPTARG;;
    \?) echo "Unknown option: Accepted flags are -f (obs_file_name),"
        exit 1;;
    :) echo "Missing option argument for input flag"
       exit 1;;
  esac
done

#  --dependency=afterany:3334729

echo "band: ${band}"
echo "Calling: /lustre/aoc/projects/hera/dstorer/Setup/djs_fhd_pipeline/runScripts/runFHD/wrapper2_arrayJob.sh"

sbatch --export=obs_file_name=${obs_file_name},outdir=${outdir},version_str=${version_str},band=${band},exants=${exants},license=${license} --array=${x} -o ${outdir}/outlogs/FHD_${band}%a.out -n 1 -N 1 --mem=120G -J FHD_${num_prefix} -p hera --begin=now --dependency=afterany:3639731 /lustre/aoc/projects/hera/dstorer/Setup/djs_fhd_pipeline/runScripts/runFHD/wrapper2_arrayJob.sh
