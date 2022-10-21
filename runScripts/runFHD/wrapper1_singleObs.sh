#!/bin/bash

while getopts ":f:o:v:n:b:a:" option
do
  case $option in
    # A uvfits file
    f) obs_file_name="$OPTARG";;
    # The output directory for the error log
    o) outdir="$OPTARG";;
    # The run version
    v) version_str=$OPTARG;;
    # A number or prefix to include in the name of the output log
    n) num_prefix=$OPTARG;;
    b) band=$OPTARG;;
    a) exants=$OPTARG;;
    \?) echo "Unknown option: Accepted flags are -f (obs_file_name),"
        exit 1;;
    :) echo "Missing option argument for input flag"
       exit 1;;
  esac
done

sbatch --export=obs_file_name=${obs_file_name},outdir=${outdir},version_str=${version_str},band=${band},exants=${exants} -o ${outdir}/FHD_${num_prefix}_${band}.out -N 1 -n 1 --mem=64G -J FHD_cal_${band} -p hera /lustre/aoc/projects/hera/dstorer/Projects/updatedHeraOnFHD/runScripts/runFHD/wrapper2_singleObs.sh 
