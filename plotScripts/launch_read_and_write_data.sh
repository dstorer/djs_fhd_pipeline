#!/bin/bash


while getopts ":r:f:s:o:x:j:R:S:C:M:J:L:B:G:D:" option
do
    case $option in
        r) raw_files=$OPTARG;;
        f) fhd_files=$OPTARG;;
        s) ssins_files=$OPTARG;;
        o) outdir=$OPTARG;;
        x) exants=$OPTARG;;
        j) jd=$OPTARG;;
        R) RAW=$OPTARG;;
        S) SSINS=$OPTARG;;
        C) CAL=$OPTARG;;
        M) MODEL=$OPTARG;;
        J) JDS=$OPTARG;;
        L) LSTS=$OPTARG;;
        B) BLS=$OPTARG;;
        G) GAINS=$OPTARG;;
        D) DIRS=$OPTARG;;
        \?) echo "Unknown option: Accepted flags are -f (obs_file_name),"
        exit 1;;
        :) echo "Missing option argument for input flag"
           exit 1;;
  esac
done

sbatch --export=raw_files=${raw_files},fhd_files=${fhd_files},ssins_files=${ssins_files},outdir=${outdir},exants=${exants},jd=${jd},RAW=${RAW},SSINS=${SSINS},CAL=${CAL},MODEL=${MODEL},JDS=${JDS},LSTS=${LSTS},BLS=${BLS},GAINS=${GAINS},DIRS=${DIRS} -p hera -o ${outdir}/read_and_write_data_${tag}.out -N 1 -n 4 --mem=128G -J write_data /lustre/aoc/projects/hera/dstorer/Setup/djs_fhd_pipeline/plotScripts/call_read_and_write_data.sh