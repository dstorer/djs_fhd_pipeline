import numpy as np
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

parser = argparse.ArgumentParser()
parser.add_argument('-f','--obs_files', help='The path to a txt file containing the path to all raw uvh5 files to be executed on')
parser.add_argument('-s','--ssins_files', help='The path to a txt file containing the path to all ssins files to be flagged with, typically files ending in flags.h5')
parser.add_argument('-o','--outdir', help='Output directory')
parser.add_argument('-N','--N_combine', help='Number of raw files to combine into one uvfits file')
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
args = parser.parse_args()

if not os.path.exists(args.outdir):
    os.makedirs(args.outdir)
    
f = open(args.obs_files, "r")
file_names = f.read().split('\n')
f = open(args.ssins_files, "r")
ssins_files = f.read().split('\n')

with open(args.xants, 'r') as xfile:
        xants = yaml.safe_load(xfile)

N = int(args.N_combine)
for i in range(0,len(file_names),N):
    data = file_names[i:i+N]
    print(data)
    fname = data[0].split('/')[-1][0:-3]
    print(fname)
    ssins = ssins_files[i//N]
    print(ssins)
    uvd = UVData()
    uvd.read(data)
    use_ants = [ant for ant in uvd.get_ants() if ant not in xants]
    uvd.select(antenna_nums=use_ants)
    if args.internode_only == 1:
        int_bls = []
        for a1 in use_ants:
            for a2 in use_ants:
                h = cm_hookup.Hookup()
                x = h.get_hookup('HH')
                key1 = 'HH%i:A' % (a1)
                n1 = x[key1].get_part_from_type('node')[f'N<ground'][1:]
                key2 = 'HH%i:A' % (a2)
                n2 = x[key2].get_part_from_type('node')[f'N<ground'][1:]
                if n1 != n2:
                    int_bls.append((a1,a2))
        uvd.select(bls=int_bls)
    elif args.intersnap_only == 1:
        snap_bls = []
        for a1 in use_ants:
            for a2 in use_ants:
                h = cm_hookup.Hookup()
                x = h.get_hookup('HH')
                key1 = 'HH%i:A' % (a1)
                s1 = x[key1].hookup[f'N<ground'][-1].downstream_input_port[-1]
                key2 = 'HH%i:A' % (a2)
                s2 = x[key2].hookup[f'N<ground'][-1].downstream_input_port[-1]
                if n1 != n2:
                    snap_bls.append((a1,a2))
        uvd.select(bls=snap_bls)
    uvf = UVFlag()
    uvf.read(ssins)
    utils.apply_uvflag(uvd,uvf)
    phaseCenter = np.median(np.unique(uvd.time_array))
    if args.band=='low':
        # 60-85 MHz
        uvd.select(frequencies=uvd.freq_array[0][108:312])
    elif args.band=='mid':
        # 150-180 MHz
        uvd.select(frequencies=uvd.freq_array[0][845:1090])
    elif args.band=='high':
        # 200-220 MHz
        uvd.select(frequencies=uvd.freq_array[0][1254:1418])
    version = f'{fname}_{args.band}'
    uvd.phase_to_time(phaseCenter)
    uvd.write_uvfits(f'{args.outdir}/{version}_{len(np.unique(uvd.time_array))}Obs.uvfits',spoof_nonessential=True)
    if int(args.write_minis) == 1:
        print('Unique times are:')
        print(np.unique(uvd.time_array))
        nint = int(args.num_times)
        for t in range(0,len(np.unique(uvd.time_array)),nint):
            times = np.unique(uvd.time_array)[t:t+nint]
            uv_single = uvd.select(times=times,inplace=False)
            phaseCenter = np.median(np.unique(uv_single.time_array))
            uv_single.phase_to_time(phaseCenter)
            uv_single.write_uvfits(f'{args.outdir}/zen.{times[0]}_{args.band}_{nint}Obs.uvfits',spoof_nonessential=True)