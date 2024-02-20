#! /usr/bin/env python

import SSINS
from SSINS import SS, INS, version, MF, util
from SSINS import Catalog_Plot as cp
from SSINS.data import DATA_PATH
from functools import reduce
import numpy as np
import argparse
import pyuvdata
from pyuvdata import UVData, UVFlag
import yaml
from hera_mc import cm_hookup
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import h5py
import hdf5plugin
import subprocess
from hera_commissioning_tools import utils as com_utils
from sys import exit

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--filename",
                    help="The visibility file(s) to process")
parser.add_argument("-s", "--streak_sig", type=float, default=20,
                    help="The desired streak significance threshold")
parser.add_argument("-o", "--other_sig", type=float, default=5,
                    help="The desired significance threshold for other shapes")
parser.add_argument("-p", "--prefix",
                    help="The prefix for output files")
parser.add_argument("-a", "--xants",
                    help="The path to a yml file containing a list of antennas to exclude")
parser.add_argument("-d", "--shape_dict",
                    help="The path to a yml file containing a dict of shapes to use")
parser.add_argument("-t", "--tb_aggro", type=float,
                    help="The tb_aggro parameter for the match filter.")
parser.add_argument("-c", "--clobber", default=0, type=int,
                    help="Whether to overwrite files that have already been written")
parser.add_argument("-x", "--no_diff", action='store_false',
                    help="Flag to turn off differencing. Use if files are already time-differenced.")
parser.add_argument("-N", "--num_baselines", type=int, default=0,
                    help="The number of baselines to read in at a time")
parser.add_argument("-I", "--internode_only", type=int, default=0,
                    help="The number of baselines to read in at a time")
parser.add_argument("-S", "--intersnap_only", type=int, default=0,
                    help="The number of baselines to read in at a time")
parser.add_argument("-n", "--n_combine", type=int, default=10,
                    help="The number of files to combine when running SSINS and calculating flags")
parser.add_argument("-i", "--node", default='all',
                    help="If not all, node number to exclusively use for flagging")
parser.add_argument("-j", "--ind", default=0,
                    help="Array job index")
args = parser.parse_args()

print(args.no_diff)
print('ARGS:')
print(args)
print('\n')
if args.clobber == 1:
    clobber = True
else:
    clobber = False
clobber = True
    
curr_path = os.path.abspath(__file__)
print(f'Running {curr_path}')
dir_path = os.path.dirname(os.path.realpath(__file__))
githash = subprocess.check_output(['git', '-C', str(dir_path), 'rev-parse', 'HEAD']).decode('ascii').strip()
print(f'djs_fhd_pipeline githash: {githash}')
print(f'pyuvdata version: {pyuvdata.__version__}')
print(f'SSINS version: {SSINS.__version__}')
ncomb = args.n_combine

if not os.path.isdir(args.prefix):
    os.mkdir(args.prefix)

f = open(args.filename, "r")
file_names = f.read().split('\n')

# for i,f1 in enumerate(file_names[::ncomb]):
i = int(args.ind)
# i = i*ncomb
f1 = file_names[i*ncomb]

if i*ncomb+ncomb >= len(file_names):
    print('Done')
    raise Exception("Index exceeds number of available files")
print(f'Running on {f1}')
name = f1.split('/')[-1][0:-5]
prefix = f'{args.prefix}/{name}_{i}'
print(f'Prefix: {prefix}')

# if os.path.isfile(f'{prefix}.SSINS.flags.h5'):
#     print(f'Stopping because file {prefix}.SSINS.flags.h5 already exists')
#     exit()

# version_info_list = [f'{key}: {version.version_info[key]}, ' for key in version.version_info]
# version_hist_substr = reduce(lambda x, y: x + y, version_info_list)

# Make the uvflag object for storing flags later, and grab bls for partial I/O
uvd = UVData()
curr_set = file_names[i*ncomb:i*ncomb+ncomb]
print(f'reading files: {curr_set}')
uvd.read(curr_set, read_data=False)

# Exclude flagged antennas
with open(args.xants, 'r') as xfile:
    xants = yaml.safe_load(xfile)
use_ants = [ant for ant in uvd.get_ants() if ant not in xants]
uvd.select(antenna_nums=use_ants)

x = cm_hookup.get_hookup('default')
if args.internode_only==1 or args.intersnap_only==1 or args.node != 'all':
    if args.internode_only == 1 and args.node != 'all':
        raise Exception('internode_only must be disabled when node is specified')
#     if i==0:
    int_bls = []
    snap_bls = []
    node_bls = []
    for a1 in use_ants:
        known_bad=False
        for a2 in use_ants:
            if a1<a2:
                key1 = com_utils.get_ant_key(x,a1)
                key2 = com_utils.get_ant_key(x,a2)
                try:
                    n1 = x[key1].get_part_from_type('node')[f'N<ground'][1:]
                    s1 = x[key1].hookup[f'N<ground'][-1].downstream_input_port[-1]
                    n2 = x[key2].get_part_from_type('node')[f'N<ground'][1:]
                    s2 = x[key2].hookup[f'N<ground'][-1].downstream_input_port[-1]
                    if n1 != n2:
                        int_bls.append((a1,a2))
                        snap_bls.append((a1,a2))
                    elif n1==n2 and s1!=s2:
                        snap_bls.append((a1,a2))
                    if n1 == args.node and n2 == args.node:
                        if args.intersnap_only == 1 and s1!=s2:
                            node_bls.append((a1,a2))
                        elif args.intersnap_only == 0:
                            node_bls.append((a1,a2))
                except:
                    if known_bad is True:
                        print(f'ERROR - One of {key1} or {key2} not found in database!')
                        known_bad = True
                    continue
    if args.internode_only==1:
        print('###### Excluding all intranode baselines ######')
        uvd.select(bls=int_bls)
    elif args.intersnap_only==1:
        print('###### Excluding all intrasnap baselines ######')
        uvd.select(bls=snap_bls)
    elif args.node != 'all':
        print(f'###### Selecting antennas in node {args.node}')
        if args.intersnap_only==1:
            print('###### Excluding all intrasnap baselines ######')
        uvd.select(bls=node_bls)

bls = uvd.get_antpairs()
uvf = UVFlag(uvd, waterfall=True, mode='flag')
del uvd

# Make the SS object
print('Making SS object')
ss = SS()
if args.num_baselines > 0:
    ss.read(file_names[i*ncomb:i*ncomb+ncomb], bls=bls[:args.num_baselines],
            diff=args.no_diff)
    ins = INS(ss)
    Nbls = len(bls)
    for slice_ind in range(args.num_baselines, Nbls, args.num_baselines):
        ss = SS()
        ss.read(file_names[i*ncomb:i*ncomb+ncomb], bls=bls[slice_ind:slice_ind + args.num_baselines],
                diff=args.no_diff)
        new_ins = INS(ss)
        ins = util.combine_ins(ins, new_ins)
else:
    ss.read(file_names[i*ncomb:i*ncomb+ncomb], diff=args.no_diff, antenna_nums=use_ants)
    if args.internode_only == 1:
        ss.select(bls=int_bls)
    elif args.intersnap_only == 1:
        ss.select(bls=snap_bls)
#         ss.read(file_names[i:i+ncomb], diff=args.no_diff, ant_str='cross')
    ss.select(ant_str='cross')

    ins = INS(ss)

# Clear some memory??
del ss

# Write the raw data and z-scores to h5 format
print('Writing data')
ins.write(prefix, sep='.', clobber=clobber)
ins.write(prefix, output_type='z_score', sep='.', clobber=clobber)

# Write out plots
cp.INS_plot(ins, f'{prefix}_RAW', vmin=0, vmax=20000, ms_vmin=-5, ms_vmax=5)
#     ins.write(prefix)
#     ins.write(prefix, output_type='z_score')

# Flag FM radio
where_FM = np.where(np.logical_and(ins.freq_array > 87.5e6, ins.freq_array < 108e6))
ins.metric_array[:, where_FM] = np.ma.masked
ins.metric_ms = ins.mean_subtract()
ins.history += "Manually flagged the FM band. "

# Make a filter with specified settings
with open(args.shape_dict, 'r') as shape_file:
    shape_dict = yaml.safe_load(shape_file)

sig_thresh = {shape: args.other_sig for shape in shape_dict}
sig_thresh['narrow'] = args.other_sig
sig_thresh['streak'] = args.streak_sig
mf = MF(ins.freq_array, sig_thresh, shape_dict=shape_dict, tb_aggro=args.tb_aggro)

# Do flagging
mf.apply_match_test(ins, time_broadcast=True)
# ins.history += f"Flagged using apply_match_test on SSINS {version_hist_substr}."

cp.INS_plot(ins, f'{prefix}_FLAGGED', vmin=0, vmax=20000, ms_vmin=-5, ms_vmax=5)

# Write outputs
#     ins.write(prefix, output_type='flags', uvf=uvf)
print('Writing mask')
ins.write(prefix, output_type='mask', sep='.', clobber=clobber)
uvf.history += ins.history
# "flags" are not helpful if no differencing was done
if args.no_diff:
    ins.write(prefix, output_type='flags', sep='.', uvf=uvf, clobber=clobber)
print('Writing match events')
ins.write(prefix, output_type='match_events', sep='.', clobber=clobber)

print('FINISHED FLAGGING!')