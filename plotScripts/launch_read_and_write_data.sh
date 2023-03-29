#!/bin/bash


while getopts ":r:f:s:o:x:j:p:R:S:C:M:J:L:B:G:D:I:V:" option
do
    case $option in
        r) raw_files=$OPTARG;;
        f) fhd_files=$OPTARG;;
        s) ssins_files=$OPTARG;;
        o) outdir=$OPTARG;;
        x) exants=$OPTARG;;
        j) jd=$OPTARG;;
        p) pol=$OPTARG;;
        R) RAW=$OPTARG;;
        S) SSINS=$OPTARG;;
        C) CAL=$OPTARG;;
        M) MODEL=$OPTARG;;
        J) JDS=$OPTARG;;
        L) LSTS=$OPTARG;;
        B) BLS=$OPTARG;;
        G) GAINS=$OPTARG;;
        D) DIRS=$OPTARG;;
        I) ITER=$OPTARG;;
        V) CONV=$OPTARG;;
        \?) echo "Unknown option: Accepted flags are -f (obs_file_name),"
        exit 1;;
        :) echo "Missing option argument for input flag"
           exit 1;;
  esac
done

sbatch --export=raw_files=${raw_files},fhd_files=${fhd_files},ssins_files=${ssins_files},outdir=${outdir},exants=${exants},jd=${jd},pol=${pol},RAW=${RAW},SSINS=${SSINS},CAL=${CAL},MODEL=${MODEL},JDS=${JDS},LSTS=${LSTS},BLS=${BLS},GAINS=${GAINS},DIRS=${DIRS},ITER=${ITER},CONV=${CONV} -p hera -o ${outdir}/read_and_write_data_${pol}.out --mem=60G -J write_data --mail-user=darajstorer@gmail.com --mail-type=END  /lustre/aoc/projects/hera/dstorer/Setup/djs_fhd_pipeline/plotScripts/call_read_and_write_data.sh