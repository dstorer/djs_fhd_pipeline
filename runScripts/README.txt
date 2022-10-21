runSSINS contains scripts needed to run SSINS on a set of HERA files. A sample command to run would be:

bash launch_Run_HERA_SSINS.sh -f /lustre/aoc/projects/hera/dstorer/Projects/updatedHeraOnFHD/2459673/673_filenames.txt -o /lustre/aoc/projects/hera/dstorer/Projects/updatedHeraOnFHD/2459673/ssinsFlagged

Remember to edit the fields in launch_Run_HERA_SSINS.sh, specifically the xants file (which specifies a set of antenna numbers to exclude from flagging) needs to match your data set.








makeUvfits contains scripts to apply SSINS flags, phase the data, and write out uvfits files that can be fed into FHD. A sample command to run would be:

bash launch_make_uvfits_apply_flags.sh -f /lustre/aoc/projects/hera/dstorer/Projects/updatedHeraOnFHD/2459630/630_filenames.txt -o /lustre/aoc/projects/hera/dstorer/Projects/updatedHeraOnFHD/2459630/uvfitsFiles/ -x /lustre/aoc/projects/hera/dstorer/Projects/updatedHeraOnFHD/2459630/exants.yml -s /lustre/aoc/projects/hera/dstorer/Projects/updatedHeraOnFHD/2459630/630_flag_filenames.txt







runFHD contains scripts needed to run FHD on a single uvfits file containing HERA data. A sample command to run a file would be:

bash wrapper1_singleObs.sh -f /lustre/aoc/projects/hera/dstorer/Projects/updatedHeraOnFHD/2459630/uvfitsFiles/zen.2459630.301083776_mid_5Obs.uvfits -o /lustre/aoc/projects/hera/dstorer/Projects/updatedHeraOnFHD/2459630/fhdOutput -v antennaNameBranch_fiveObs -n 1 -b mid -a /lustre/aoc/projects/hera/dstorer/Projects/updatedHeraOnFHD/2459630/exants.yml



Input parameters for all scripts are defined within the wrappers. 

