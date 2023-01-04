import numpy as np

import argparse
import os
import os.path
from os import path
import pyuvdata
from pyuvdata import UVData
import subprocess
import yaml

parser = argparse.ArgumentParser()
parser.add_argument('obs_file', help='The path to a uvfits file to be executed on')
parser.add_argument('outdir', help='Output directory')
parser.add_argument('version_str', help='A string to include in the name of all outputs indicating the run version')
parser.add_argument('band', help='Options are low,mid,high,full - determines frequency band')
parser.add_argument('exants', help='A yml file containing a list of flagged antennas')
args = parser.parse_args()

print('obs_file is:')
print(str(args.obs_file))
print('outdir is:')
print(str(args.outdir))

curr_path = os.path.abspath(__file__)
print(f'Running {curr_path}')
dir_path = os.path.dirname(os.path.realpath(__file__))
githash = subprocess.check_output(['git', '-C', str(dir_path), 'rev-parse', 'HEAD']).decode('ascii').strip()
print(f'githash: {githash}')
print(f'pyuvdata version: {pyuvdata.__version__}')

overwrite = False

if args.exants=="None":
	xants = []
else:
	with open(args.exants, 'r') as xfile:
		xants = yaml.safe_load(xfile)

if not os.path.exists(args.outdir):
    os.makedirs(args.outdir)

filepath = args.obs_file

uv = UVData()
uv.read(filepath)
ants = uv.get_ants()
if len(xants)>0:
    use_ants = [ant for ant in ants if ant not in xants]
    uv.select(antenna_nums=use_ants)
unique = np.unique(uv.time_array)
version = f'{unique[0]}_{args.version_str}_{args.band}'
dirname = args.outdir + '/fhd_' + version + '/beams'
sourcefile = f'{args.outdir}/fhd_{version}/output_data/{unique[0]}_{args.band}_uniform_Sources_XX.fits'
if path.exists(sourcefile) and overwrite is False:
    print(f'{sourcefile} already exists - SKIPPING')
elif path.exists(dirname) and overwrite is False:
    print('%s has already been run - SKIPPING' % version)
else:
    os.system("idl -e run_h4c_vivaldibeam -args " + filepath + " " + version + " " + args.outdir)
