Step 1: Run SSINS flagging

runSSINS contains scripts needed to run SSINS on a set of HERA files. A sample command to run would be:

bash launch_Run_HERA_SSINS_array.sh -f /lustre/aoc/projects/hera/dstorer/Projects/thesis/fhdRuns/2459893/2459893_raw_filenames.txt -o /lustre/aoc/projects/hera/dstorer/Projects/thesis/fhdRuns/2459893/ssinsFlagged_v3 -x /lustre/aoc/projects/hera/dstorer/Projects/thesis/fhdRuns/2459893/exants_MR.yml -a 0-20%10

The shape file is hardcoded as "/lustre/aoc/projects/hera/dstorer/Setup/djs_fhd_pipeline/runScripts/runSSINS/HERA_shape_dict.yml", and additional settings are hardcoded in call_Run_HERA_SSINS.sh.


#################################
Step 2: Make uvfits files

makeUvfits contains scripts to apply SSINS flags, phase the data, split into shorted files, and write out uvfits files that can be fed into FHD. To run this the full path to all desired SSINS flag files should be put into one single txt file, with one path per line. It is important that the file range in the raw file list passed in here is the same as the one used to create the SSINS flags so that time stamps match. To run a subset use the array job to select indices, do not clip/trim the file lists. A sample command to run would be:

bash launch_make_uvfits_apply_flags_array.sh -f /lustre/aoc/projects/hera/dstorer/Projects/thesis/fhdRuns/2459893/2459893_raw_filenames.txt -o /lustre/aoc/projects/hera/dstorer/Projects/thesis/fhdRuns/2459893/uvfitsFiles/thesisRun_MR -x /lustre/aoc/projects/hera/dstorer/Projects/thesis/fhdRuns/2459893/exants_MR -s /lustre/aoc/projects/hera/dstorer/Projects/thesis/fhdRuns/2459893/2459893_flag_filenames_MR.txt -p 1

Some settings are hardcoded in launch_make_uvfits_apply_flags_array.sh - review these before running. Set -p to 1 to run flagging per polarization, or set to 0 to flag polarizations the same. If "-p 1", the code will expect exant files starting with the given path and with suffixes "_X.yml" and "_Y.yml". If "-p 0", the code will expect a single exants file with the given path and just the suffix ".yml".


#################################
Step 3: Run FHD

runFHD contains scripts needed to run FHD on a single uvfits file containing HERA data. A sample command to run a file would be:

bash wrapper1_arrayJob.sh -f /lustre/aoc/projects/hera/dstorer/Projects/thesis/fhdRuns/2459893/2459893_uvfits_thesis_MR.txt -o /lustre/aoc/projects/hera/dstorer/Projects/thesis/fhdRuns/2459893/fhdOutput -v thesis_v2_updatedHpxInds -c thesis_v2_updatedHpxInds_rephaseWeights0 -n _ -b mid_clip -a None -i 0 -h 4070830 -l HERA -x 0-20%10


Some of these parameters are antiquated, hence the filler/null inputs. Only 10-12 jobs can be run in parallel given license constraints. Additional jobs can be run by setting -l to NRAO instead.

