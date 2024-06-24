#!/bin/bash

while getopts ":f:o:v:c:n:b:a:i:h:x:l:" option
do
  case $option in
    # A text file where each line is an obsid
    f) obs_file_name="$OPTARG";;
    # The output directory for the error log
    o) outdir="$OPTARG";;
    # The run version
    v) version_str=$OPTARG;;
    c) case_name=$OPTARG;;
    # A number or prefix to include in the name of the output log
    n) num_prefix=$OPTARG;;
    b) band=$OPTARG;;
    a) exants=$OPTARG;;
    i) init_cal=$OPTARG;;
    h) hold=$OPTARG;;
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

export HEALPIX='/lustre/aoc/projects/hera/dstorer/Setup/Healpix_3.50'

# echo "hold: ${hold}"
if [ ! -n ${hold} ]; then
    hold_str="-d afterany:4070830"
else
    hold_str="-d afterany:${hold##* }"
fi
echo ${hold_str}

logdir=${outdir}/fhd_${version_str}/outlogs
if [ ! -d ${outdir}/fhd_${version_str} ]; then
    mkdir -m 777 ${outdir}/fhd_${version_str}
fi
if [ ! -d ${logdir} ]; then
    mkdir -m 777 ${logdir}
fi

sbatch --export=obs_file_name=${obs_file_name},outdir=${outdir},version_str=${version_str},case_name=${case_name},band=${band},exants=${exants},init_cal=${init_cal},license=${license} --array=${x} -o ${logdir}/FHD_${version_str}%a.out -n 1 -N 1 --mem=110G -J FHD_${num_prefix} -p hera --begin=now ${hold_str} /lustre/aoc/projects/hera/dstorer/Setup/djs_fhd_pipeline/runScripts/runFHD/wrapper2_arrayJob.sh
