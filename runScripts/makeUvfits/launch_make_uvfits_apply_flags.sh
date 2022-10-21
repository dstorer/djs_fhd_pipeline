#!/bin/bash


while getopts ":f:s:o:x:" option
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

echo "calling /lustre/aoc/projects/hera/dstorer/Projects/updatedHeraOnFHD/runScripts/makeUvfits/call_make_uvfits_apply_flags.sh"

#xants_file="/lustre/aoc/projects/hera/dstorer/Projects/updatedHeraOnFHD/2459854/exants.yml"
shape_file="/lustre/aoc/projects/hera/dstorer/Projects/updatedHeraOnFHD/runScripts/runSSINS/HERA_shape_dict.yml"
band="mid"
N_combine=10
internode_only=0
intersnap_only=0
write_minis=1
num_times=1


sbatch --export=obs_file_name=${obs_file_name},ssins_files=${ssins_files},outdir=${outdir},N_combine=${N_combine},xants=${xants_file},band=${band},internode_only=${internode_only},intersnap_only=${intersnap_only},write_minis=${write_minis},num_times=${num_times} -p hera -o ${outdir}/make_uvfits.out -N 1 -n 1 --mem=128G -J make_uvfits /lustre/aoc/projects/hera/dstorer/Projects/updatedHeraOnFHD/runScripts/makeUvfits/call_make_uvfits_apply_flags.sh