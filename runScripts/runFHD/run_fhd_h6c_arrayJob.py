import numpy as np

import argparse
import os
import os.path
from os import path
import pyuvdata
from pyuvdata import UVData
import subprocess
import yaml
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('obs_file', help='The path to a txt file containing the path to all uvfits files to be executed on')
parser.add_argument('outdir', help='Output directory')
parser.add_argument('version_str', help='A string to include in the name of all outputs indicating the run version')
parser.add_argument('case_name', help='Case name to use in versions script')
parser.add_argument('SLURM_ARRAY_TASK_ID', help='The index of the array job to run')
parser.add_argument('band', help='Options are low,mid,high,full - determines frequency band')
parser.add_argument('exants', help='A yml file containing a list of flagged antennas')
parser.add_argument('init_cal', help='Path to a cal.sav file to use as initial cal solutions. If set to 0 then no initial cal will be used')
parser.add_argument('license', help='specify which IDL license to use. Can be NRAO or HERA')
args = parser.parse_args()

print('obs_file is:')
print(str(args.obs_file))
print('outdir is:')
print(str(args.outdir))
print('Array ID is:')
print(str(args.SLURM_ARRAY_TASK_ID))
print('Case name is:')
print(str(args.case_name))
print(f'Initial cal is: {args.init_cal}')

curr_path = os.path.abspath(__file__)
print(f'Running {curr_path}')
dir_path = os.path.dirname(os.path.realpath(__file__))
githash = subprocess.check_output(['git', '-C', str(dir_path), 'rev-parse', 'HEAD']).decode('ascii').strip()
print(f'djs_fhd_pipeline githash: {githash}')
print(f'pyuvdata version: {pyuvdata.__version__}')
fhd_file = '/lustre/aoc/projects/hera/dstorer/Setup/FHD/fhd_core/calibration/calfits_read.pro'
dir_path = os.path.dirname(os.path.realpath(fhd_file))
githash = subprocess.check_output(['git', '-C', str(dir_path), 'rev-parse', 'HEAD']).decode('ascii').strip()
print(f'FHD githash: {githash}')

overwrite = True

if args.exants=="None":
	xants = []
else:
	with open(args.exants, 'r') as xfile:
		xants = yaml.safe_load(xfile)

if not os.path.exists(args.outdir):
    os.makedirs(args.outdir)
if not os.path.exists(f'{args.outdir}/outlogs'):
    os.makedirs(f'{args.outdir}/outlogs')

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
    # version = f'{unique[0]}_{args.version_str}_{args.band}_{ind}'
    version = str(args.version_str)
    dirname = args.outdir + '/fhd_' + version + '/beams'
    sourcefile = f'{args.outdir}/fhd_{version}/output_data/{unique[0]}_{args.band}_uniform_Sources_XX.fits'
    if path.exists(sourcefile) and overwrite is False:
        print(f'{sourcefile} already exists - SKIPPING')
    elif path.exists(dirname) and overwrite is False:
        print('%s has already been run - SKIPPING' % version)
    else:
        # if args.license == 'NRAO':
        #     print('Calling: idl -e run_h6c_vivaldibeam_versions -args ' + filepath + " " + args.version_str + " " + args.outdir + " " + args.case_name)
        #     os.system("idl -e run_h6c_vivaldibeam_versions -args " + filepath + " " + args.version_str + " " + args.outdir + " " + args.case_name)
        
        # elif args.license == 'HERA':
        #     print('Calling: /home/heraidl/idl/bin/idl -e run_h6c_vivaldibeam_versions -args ' + filepath + " " + args.version_str + " " + args.outdir + " " + args.case_name)
        #     os.system("/home/heraidl/idl/bin/idl -e run_h6c_vivaldibeam_versions -args " + filepath + " " + args.version_str + " " + args.outdir + " " + args.case_name)
        if args.license == 'NRAO':
            print('Calling: idl -e run_h6c_thesis_run -args ' + filepath + " " + args.version_str + " " + args.outdir + " " + args.case_name + " " + args.init_cal)
            os.system("idl -e run_h6c_thesis_run -args " + filepath + " " + args.version_str + " " + args.outdir + " " + args.case_name + " " + args.init_cal)
        
        elif args.license == 'HERA':
            print('Calling: /home/heraidl/idl/bin/idl -e run_h6c_thesis_run -args ' + filepath + " " + args.version_str + " " + args.outdir + " " + args.case_name + " " + args.init_cal)
            os.system("/home/heraidl/idl/bin/idl -e run_h6c_thesis_run -args " + filepath + " " + args.version_str + " " + args.outdir + " " + args.case_name + " " + args.init_cal)
        
        elif args.license == 'versions':
            print("Calling: /home/heraidl/idl/bin/idl -e run_h6c_vivaldibeam_versions -args " + filepath + " " + version + " " + args.outdir)
            os.system("/home/heraidl/idl/bin/idl -e run_h6c_vivaldibeam_versions -args " + filepath + " " + version + " " + args.outdir)
