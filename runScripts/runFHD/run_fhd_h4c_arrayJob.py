import numpy as np

import argparse
import os
import os.path
from os import path
from pyuvdata import UVData
import subprocess
import yaml
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('obs_file', help='The path to a txt file containing the path to all uvfits files to be executed on')
parser.add_argument('outdir', help='Output directory')
parser.add_argument('version_str', help='A string to include in the name of all outputs indicating the run version')
parser.add_argument('SLURM_ARRAY_TASK_ID', help='The index of the array job to run')
parser.add_argument('band', help='Options are low,mid,high,full - determines frequency band')
parser.add_argument('exants', help='A yml file containing a list of flagged antennas')
args = parser.parse_args()

print('obs_file is:')
print(str(args.obs_file))
print('outdir is:')
print(str(args.outdir))
print('Array ID is:')
print(str(args.SLURM_ARRAY_TASK_ID))

curr_path = os.path.abspath(__file__)
print(f'Running {curr_path}')
dir_path = os.path.dirname(os.path.realpath(__file__))
githash = subprocess.check_output(['git', '-C', str(dir_path), 'rev-parse', 'HEAD']).decode('ascii').strip()
print(f'githash: {githash}')

overwrite = False

if args.exants=="None":
	xants = []
else:
	with open(args.exants, 'r') as xfile:
		xants = yaml.safe_load(xfile)

if not os.path.exists(args.outdir):
    os.makedirs(args.outdir)

ind = int(args.SLURM_ARRAY_TASK_ID)
if ind > 0:
    print(f'ind is {ind}')
    f = open(args.obs_file, "r")
    file_names = f.read().split('\n')
    filepath = file_names[ind]
    print(file_names[ind])
    
    uv = UVData()
    uv.read(file_names[ind])
    ants = uv.get_ants()
    if len(xants)>0:
        use_ants = [ant for ant in ants if ant not in xants]
        uv.select(antenna_nums=use_ants)
    unique = np.unique(uv.time_array)
    version = f'{unique[0]}_{args.version_str}_{args.band}_{ind}'
    dirname = args.outdir + '/fhd_' + version + '/beams'
    sourcefile = f'{args.outdir}/fhd_{version}/output_data/{unique[0]}_{args.band}_uniform_Sources_XX.fits'
    if path.exists(sourcefile) and overwrite is False:
        print(f'{sourcefile} already exists - SKIPPING')
    elif path.exists(dirname) and overwrite is False:
        print('%s has already been run - SKIPPING' % version)
    else:
        os.system("idl -e run_h4c_vivaldibeam -args " + filepath + " " + version + " " + args.outdir)
    
    
# 	for i in range(0,2):
# 		print(i)
# 		use_times = unique[i*10:i*10+10]
# 		print(59)
# # 		uv2 = uv.select(times=use_times,inplace=False)
# 		print(61)
# # 		phaseCenter = np.median(use_times)
# 		print(63)
# # 		if args.band=='low':
# # 			# 60-85 MHz
# # 			uv2.select(frequencies=uv2.freq_array[0][108:312])
# # 		#elif args.band=='mid':
# # 		#	# 150-180 MHz
# # 		#	uv2.select(frequencies=uv2.freq_array[0][845:1090])
# # 		elif args.band=='high':
# # 			# 200-220 MHz
# # 			uv2.select(frequencies=uv2.freq_array[0][1254:1418])
# 		version = f'{use_times[0]}_{args.version_str}_{args.band}_{ind+1}_{i}'
# 		print(74)
# # 		print(phaseCenter)
# # 		uv2.phase_to_time(phaseCenter)
# 		print(76)
# # 		outname = f'{args.outdir}/uvfits_files/{use_times[0]}_{args.band}.uvfits'
# 		dirname = args.outdir + '/fhd_' + version + '/beams'
# 		sourcefile = f'{args.outdir}/fhd_{version}/output_data/{use_times[0]}_{args.band}_uniform_Sources_XX.fits'
# 		if path.exists(sourcefile) and overwrite is False:
# 			print(f'{sourcefile} already exists - SKIPPING')
# 		elif path.exists(dirname) and overwrite is False:
# 			print('%s has already been run - SKIPPING' % version)
# 		else:
# # 			uv2.write_uvfits(outname, spoof_nonessential=True)
# 			os.system("idl -e /lustre/aoc/projects/hera/dstorer/Projects/scripts/IDLscripts/standardFhdRun/run_h4c_vivaldibeam -args " + filepath + " " + version + " " + args.outdir)
