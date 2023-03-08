import numpy as np
import pyuvdata
from pyuvdata import UVData, UVFlag, utils
import argparse
import os
import os.path
from os import path
from hera_mc import cm_hookup
import yaml
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import h5py
import hdf5plugin
import subprocess
from hera_commissioning_tools import utils as com_utils

parser = argparse.ArgumentParser()
parser.add_argument('-f','--obs_files', help='The path to a txt file containing the path to all raw uvh5 files to be executed on')
parser.add_argument('-s','--ssins_files', help='The path to a txt file containing the path to all ssins files to be flagged with, typically files ending in flags.h5')
parser.add_argument('-o','--outdir', help='Output directory')
parser.add_argument('-N','--N_combine', help='Number of raw files to combine into one uvfits file. Must match the number of raw files used to create the ssins flags.')
parser.add_argument('-b','--band', help='Frequency band (low, med, or high) to write out')
parser.add_argument("-x", "--xants",
                    help="The path to a yml file containing a list of antennas to exclude")
parser.add_argument("-I", "--internode_only", type=int, default=0,
                    help="The number of baselines to read in at a time")
parser.add_argument("-S", "--intersnap_only", type=int, default=0,
                    help="The number of baselines to read in at a time")
parser.add_argument("-m", "--write_minis", help='Option to also write out single observation uvfits files.')
parser.add_argument("-n", "--num_times", default=1,
                    help='Number of times to include in mini files. Only used if write_minis is set to 1.')
parser.add_argument("-a", "--array_job", default=0,
                    help='Indicates whether the job is an array job. If so, only one file, indicated by --ind, will be exected. Otherwise, all files in obs_files will be executed')
parser.add_argument("--ind", default=0,
                    help='File index to run on')
parser.add_argument('-p', '--phase', default='perobs',
                    help='Option to phase the data to the center of each observation (perobs) or to the center of the whole set of observations (perset)')
args = parser.parse_args()

print('\n')
curr_path = os.path.abspath(__file__)
print(f'running {curr_path}')
dir_path = os.path.dirname(os.path.realpath(__file__))
githash = subprocess.check_output(['git', '-C', str(dir_path), 'rev-parse', 'HEAD']).decode('ascii').strip()
print(f'githash: {githash}')
print(f'pyuvdata version: {pyuvdata.__version__}')
print(f'numpy version: {np.__version__} \n')

if not os.path.exists(args.outdir):
    os.makedirs(args.outdir)
    
f = open(args.obs_files, "r")
file_names = f.read().split('\n')
f = open(args.ssins_files, "r")
ssins_files = f.read().split('\n')

jds = [x.split('/')[-1].split('.sum')[0][4:] for x in file_names[:-1]]
jds = [float(j) for j in jds]
mid_jd = jds[len(jds)//2]

with open(args.xants, 'r') as xfile:
    xants = yaml.safe_load(xfile)

array_job = args.array_job
print(f'array_job: {array_job}')

N = int(args.N_combine)
x = cm_hookup.get_hookup('default')
for i in range(0,len(file_names),N):
    if int(array_job) == 1:
        if int(args.ind) != i//N:
            continue
        print(f'Running index {args.ind} of array job')
    elif i==0:
        print('Running all files in obs_files')
    data = file_names[i:i+N]
#     print(data)
    fname = data[0].split('/')[-1][0:-5]
#     print(fname)
    ssins = ssins_files[i//N]
    print('SSINS file:')
    print(ssins)
    print('\nreading data \n')
    uvd = UVData()
    uvd.read(data)
    use_ants = [ant for ant in uvd.get_ants() if ant not in xants]
    uvd.select(antenna_nums=use_ants)
    print('Reading SSINS \n')
    uvf = UVFlag()
    uvf.read(ssins)
    print('Data times:')
    print(np.unique(uvd.time_array))
    print('SSINS times:')
    print(uvf.time_array)
    if (np.unique(uvd.time_array)==uvf.time_array).all() is False:
        print('!!!!!!!!!!!!!!!!!! ERROR !!!!!!!!!!!!!!!!!!!!!!!!')
        print('Data file times do not match flag file times')
        break
    if args.internode_only == 1:
        print('Selecting internode baselines')
        int_bls = []
        for a1 in use_ants:
            for a2 in use_ants:
                key1 = com_utils.get_ant_key(x,a1)
                key2 = com_utils.get_ant_key(x,a2)
                try:
                    n1 = x[key1].get_part_from_type('node')[f'N<ground'][1:]
                    n2 = x[key2].get_part_from_type('node')[f'N<ground'][1:]
                    if n1 != n2:
                        int_bls.append((a1,a2))
                except:
                    print(f'ERROR - One of {key1} or {key2} not found in database!')
                    continue
        uvd.select(bls=int_bls)
    elif args.intersnap_only == 1:
        print('Selecting intersnap baselines')
        snap_bls = []
        for num,a1 in enumerate(use_ants):
            if num%20==0:
                    print(f'Selecting on ant {num} of {len(use_ants)}')
            for a2 in use_ants:
                key1 = com_utils.get_ant_key(x,a1)
                key2 = com_utils.get_ant_key(x,a2)
                try:
                    s1 = x[key1].hookup[f'N<ground'][-1].downstream_input_port[-1]
                    s2 = x[key2].hookup[f'N<ground'][-1].downstream_input_port[-1]
                    if s1 != s2:
                        snap_bls.append((a1,a2))
                except:
                    print(f'ERROR - One of {key1} or {key2} not found in database!')
                    continue
        uvd.select(bls=snap_bls)
    print('Applying flags')
    utils.apply_uvflag(uvd,uvf)
    if args.phase == 'perobs':
        phaseCenter = np.median(np.unique(uvd.time_array))
    elif args.phase == 'perset':
        phaseCenter = mid_jd
    else:
        print(f'ERROR! phase argument must be set to either perobs or perset. Currently set to {args.phase}')
    if args.band=='low':
        # 60-85 MHz
        print(f'Selecting freqs in index range [108,312]')
        uvd.select(frequencies=uvd.freq_array[0][108:312])
    elif args.band=='mid':
        # 150-180 MHz
        print(f'Selecting freqs in index range [845,1090]')
        uvd.select(frequencies=uvd.freq_array[0][845:1090])
    elif args.band=='high':
        # 200-220 MHz
        print(f'Selecting freqs in index range [1254,1418]')
        uvd.select(frequencies=uvd.freq_array[0][1254:1418])
    version = f'{fname}_{args.band}'
    print(f'Phasing observation to time {phaseCenter}')
    uvd.phase_to_time(phaseCenter)
    print('Writing to uvfits')
#     uvd.write_uvfits(f'{args.outdir}/{version}_{len(np.unique(uvd.time_array))}obs_{args.ind}.uvfits',spoof_nonessential=True)
    if int(args.write_minis) == 1:
        print('Unique times are:')
        print(np.unique(uvd.time_array))
        nint = int(args.num_times)
        print('Writing minis')
        for t in range(0,len(np.unique(uvd.time_array)),nint):
            times = np.unique(uvd.time_array)[t:t+nint]
            uv_single = uvd.select(times=times,inplace=False)
            phaseCenter = np.median(np.unique(uv_single.time_array))
            uv_single.phase_to_time(phaseCenter)
            uv_single.write_uvfits(f'{args.outdir}/zen.{times[0]}_{args.band}_{nint}obs_{args.ind}.uvfits',spoof_nonessential=True)