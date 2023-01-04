import numpy as np
import subprocess
import argparse
import matplotlib.pyplot as plt
import plot_fits
import plot_vis
import glob
import pyuvdata
from pyuvdata import UVData, UVFlag, UVCal
import warnings
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import h5py
import hdf5plugin
import yaml
import pickle
from hera_commissioning_tools import utils
from scipy.io import readsav

parser = argparse.ArgumentParser()
parser.add_argument('-r','--raw_files', help='The path to a txt file containing the path to all raw uvfits files to be executed on')
parser.add_argument('-f','--fhd_files', help='The path to a txt file containing the path to the fhd output directory to be executed on')
parser.add_argument('-s','--ssins_files', help='The path to a txt file containing the path to the ssins flag files to be executed on')
parser.add_argument('-o','--outdir', help='Path to write all outputs to')
parser.add_argument('-x','--xants', default=None, help='Path to yml with list of antennas to exclude')
parser.add_argument('-j','--juliandate', help='JD of observations')
parser.add_argument('-n','--nobs', default=20, help='Number of times that went into each fhd observation')
parser.add_argument('-R','--RAW', default=0, help='Boolean indicating whether to read and write raw files')
parser.add_argument('-C','--CAL', default=0, help='Boolean indicating whether to read and write calibrated data files')
parser.add_argument('-M','--MODEL', default=0, help='Boolean indicating whether to read and write model visibility files')
parser.add_argument('-S','--SSINS', default=0, help='Boolean indicating whether to read and write ssins flag files')
parser.add_argument('-B','--BLS', default=1, help='Boolean indicating whether to write baseline set')
parser.add_argument('-L','--LSTS', default=0, help='Boolean indicating whether to read and write LST values')
parser.add_argument('-J','--JDS', default=0, help='Boolean indicating whether to read and write JD values')
parser.add_argument('-D','--DIRS', default=0, help='Boolean indicating whether to write FHD dirpaths')
parser.add_argument('-G','--GAINS', default=0, help='Boolean indicating whether to read and write gain values')
parser.add_argument('-I','--ITER', default=0, help='Boolean indicating whether to read and write convergence iteration numbers.')
parser.add_argument('-V','--CONV', default=0, help='Boolean indicating whether to read and write convergence values.')

args = parser.parse_args()

polarizations=[-5]

clip_data = False
# startJD = 2459855.63237
# stopJD = 2459855.65308
startJD=0
stopJD=1385992738279837

freqs = np.arange(845,1090)
clipFreqs = False


# Print out useful metadata
print('\n')
curr_path = os.path.abspath(__file__)
print(f'running {curr_path}')
dir_path = os.path.dirname(os.path.realpath(__file__))
githash = subprocess.check_output(['git', '-C', str(dir_path), 'rev-parse', 'HEAD']).decode('ascii').strip()
print(f'githash: {githash}')
print(f'pyuvdata version: {pyuvdata.__version__}')

# Open and parse files
f = open(args.raw_files, "r")
raw_data = f.read().split('\n')[0:-1]
f = open(args.fhd_files, "r")
fhd_files = f.read().split('\n')[0:-1]
f = open(args.ssins_files, "r")
ssins_files = f.read().split('\n')[0:-1]

dirty_files = []
beam_files = []
model_files = []
obs = []
for path in fhd_files:
    dfiles = sorted(glob.glob(f'{path}output_data/*Dirty_XX.fits'))
    bfiles = sorted(glob.glob(f'{path}output_data/*Beam_XX.fits'))
    mfiles = sorted(glob.glob(f'{path}output_data/*Model_XX.fits'))
    if len(dfiles) == 0:
        obs.append(0)
        continue
    if len(dfiles)>1 or len(bfiles)>1 or len(mfiles)>1:
        raise "ERROR! More than one dirty, beam, or model file per fhd obs."
    dirty_files.append(dfiles[0])
    beam_files.append(bfiles[0])
    model_files.append(mfiles[0])
    obsname = dfiles[0].split('/')[-1][0:-22]
    obs.append(obsname)

print('\n')
print('Observations to read in:')
print(obs)
print('\n')


fhd_file_array = []
for i,fhd in enumerate(fhd_files):
    flist = []
    obsname = obs[i]
    if obsname == 0:
        continue
    flist.append(fhd + 'metadata/' + obsname + '_params.sav')
    flist.append(fhd + 'metadata/' + obsname + '_settings.txt')
    flist.append(fhd + 'metadata/' + obsname + '_layout.sav')
    vis_files = ['flags.sav','vis_XX.sav','vis_YY.sav','vis_model_XX.sav','vis_model_YY.sav']
    for f in vis_files:
        flist.append(fhd + 'vis_data/' + obsname + '_' + f)
    fhd_file_array.append(flist)

if int(args.DIRS) == 1:
    print('Writing FHD directory list \n')
    file = f'{args.outdir}/{args.juliandate}_fhd_dirlist.txt'
    with open(file, 'wb') as f:
#         for line in fhd_file_array:
#             print(line)
#             f.write(f"{line}\n")
        pickle.dump(fhd_file_array,f)
            
print('Reading raw metadata')
raw_jds = []
for f in raw_data:
    fname = f.split('/')[-1]
    try:
        raw_jds.append(float(fname[4:17]))
    except:
        continue
if clip_data is True:
    startind = np.argmin(np.abs(np.subtract(raw_jds,startJD)))
    stopind = np.argmin(np.abs(np.subtract(raw_jds,stopJD)))
    print(f'Starting with index {startind}, which is file {raw_data[startind]}')
    print(f'Stopping with index {stopind}, which is file {raw_data[stopind]}')
    raw_data = raw_data[startind:stopind]
mid_jd = raw_jds[len(raw_jds)//2]

file_read = False
for i,flist in enumerate(fhd_file_array):
    if obs[i] == 0:
        continue
    if float(obs[i][4:17]) < startJD or float(obs[i][4:17]) > stopJD:
        continue
    if file_read is False:
        calData = UVData()
        calData.read(flist,use_model=False)
        calData.select(polarizations=[-5])
        use_ants = calData.get_ants()
#         bls = np.unique(calData.baseline_array)
        bls = calData.get_antpairs()
#         print('Baselines:')
#         print(bls)
        Nbls = calData.Nbls
        print(f'\n{len(use_ants)} antennas in observation set, for a total of {Nbls} baselines \n')
        break
    
raw = UVData()
# raw.read(raw_data,read_data=False,skip_bad_files=True,axis='blt')
# print(raw_data)
raw.read(raw_data,read_data=False)
if int(args.JDS) == 1:
    print('Writing JD array')
    jds = raw.time_array
    jds = np.reshape(jds,(raw.Ntimes,-1))
    np.save(f'{args.outdir}/{args.juliandate}_jd_array',jds[:,0])
if int(args.LSTS) == 1:
    print('Writing LST array')
    lsts = raw.lst_array * 3.819719
    lsts = np.reshape(lsts,(raw.Ntimes,-1))
    np.save(f'{args.outdir}/{args.juliandate}_lst_array',lsts[:,0])
    
# if args.xants is not None:
#     with open(args.xants, 'r') as xfile:
#         xants = yaml.safe_load(xfile)
#     use_ants = [a for a in raw.get_ants() if a not in xants]
#     raw.select(antenna_nums=use_ants)
print('Performing baseline selection on raw data to match baseline set in cal and model data')
# print('Raw baselines:')
# print(raw.get_antpairs())
raw.select(bls=bls)

Ntimes = raw.Ntimes
print(f'\nData has {Ntimes} time stamps\n')
Nbls = raw.Nbls
Nfreqs = len(freqs)
Npols = raw.Npols
antpairs = np.asarray(raw.get_antpairs())
if int(args.BLS) == 1:
    print('Writing baseline array')
    file = f'{args.outdir}/{args.juliandate}_bl_array.npy'
    with open(file, 'wb') as f:
        np.save(f,antpairs)
# raw.write_uvh5('2459855_raw_metadata.uvh5',clobber=True)
del raw

if int(args.RAW) == 1:
    print('Reading raw data') 
#     raw_array = np.ones((Ntimes,Nbls,Nfreqs),dtype=np.cdouble)
#     for i,flist in enumerate(raw_data):
#         if i%50 == 0:
#             print(f'Reading {i}th file of {len(raw_data)}')
#         rawData = UVData()
#         if clipFreqs:
#             rawData.read(flist,polarizations=[-5],freq_chans=freqs,bls=bls)
#         else:
#             rawData.read(flist,polarizations=[-5],bls=bls)
#         d = rawData.data_array[:,0,:,0]
#         t = len(np.unique(rawData.time_array))
# #         print(f'ntimes: {t}')
#         if i*t+t >= np.shape(raw_array)[0]:
#             print('ERROR: raw_array already full - cant add more data')
#             break
#         d = np.reshape(d,(t,-1,rawData.Nfreqs))
#         if i==0:
#             print('data shape:')
#             print(np.shape(d))
# #         print(f'Adding obs with means {np.nanmean(np.abs(d[0,:,:]))} and {np.nanmean(np.abs(d[1,:,:]))} into rows {i*t}:{i*t+t} of raw_array')
#         raw_array[i*t:i*t+t,:,:] = d
# #         print(np.nanmean(np.abs(raw_array[:,100,:]),axis=1))
    rawData = UVData()
    if clipFreqs:
        rawData.read(raw_data,polarizations=[-5],freq_chans=freqs,bls=bls)
    else:
        rawData.read(raw_data,polarizations=[-5],bls=bls)
    
    print('Writing Raw Data Array \n')
    file = f'{args.outdir}/{args.juliandate}_fhd_raw_data.uvfits'
    rawData.write_uvfits(file)
#     with open(file, 'wb') as f:
#         np.save(f, raw_array)
#     del raw_array
    del rawData
    
if int(args.GAINS) == 1:
    print('Reading gains')
    
    calfiles = []
    obsfiles = []
    layoutfiles = []
    settingsfiles = []
    for path in fhd_files:
        cf = sorted(glob.glob(f'{path}calibration/*cal.sav'))
        of = sorted(glob.glob(f'{path}metadata/*obs.sav'))
        lf = sorted(glob.glob(f'{path}metadata/*layout.sav'))
        sf = sorted(glob.glob(f'{path}metadata/*settings.txt'))
        try:
            calfiles.append(cf[0])
            obsfiles.append(of[0])
            layoutfiles.append(lf[0])
            settingsfiles.append(sf[0])
        except:
            print(f'Skipping obs {path}')
            continue
    
    g = UVCal()
    g.read_fhd_cal(cal_file=calfiles,obs_file=obsfiles,layout_file=layoutfiles,settings_file=settingsfiles,
                run_check=False,run_check_acceptability=False)
    file = f'{args.outdir}/{args.juliandate}_fhd_gains.uvfits'
    print('Writing Gains \n')
    g.write_calfits(file)

if int(args.CAL) == 1:
    print('Reading calibrated data') 
    nobs = int(args.nobs)
#     cal_array = np.ones((Ntimes,Nbls,Nfreqs),dtype=np.cdouble)
#     fhd_read = False
#     for i,flist in enumerate(fhd_file_array):
#         if i%10 == 0:
#             print(f'Reading {i}th file of {len(fhd_file_array)}')
#         if obs[i] == 0:
#             print(f'Skipping obs {i} for having no FHD solutions')
#             continue
#         if float(obs[i][4:17]) < startJD or float(obs[i][4:17]) > stopJD:
#             if clip_data:
#                 print(f'Skipping obs {obs[i]} for being outside JD range')
#                 continue
#         if fhd_read is False:
#             calData = UVData()
#             calData.read(flist,use_model=False)
#             calData.select(polarizations=[-5])
#             use_ants = calData.get_ants()
#             Nbls = calData.Nbls
#             print('Initiating array')
#             cal_array = np.ones((Ntimes,Nbls,Nfreqs),dtype=np.cdouble)
#             fhd_read=True
#         calData = UVData()
#         calData.read(flist,use_model=False)
#         calData.select(polarizations=[-5])
#         d = calData.data_array
#         t = len(np.unique(calData.time_array))
#         d = np.reshape(d,(t,-1,calData.Nfreqs))
# #         print(f'Adding obs with mean {np.nanmean(d)} into rows {i*t}:{i*t+t} of cal_array')
#         cal_array[i*t:i*t+t,:,:] = d
# #         print(f'cal_array now has sum {np.sum(cal_array)}')
    
#     print(cal_array)
    calData = UVData()
    calData.read(fhd_file_array,use_model=False,ignore_name=True)
    calData.select(polarizations=[-5])
    print('Writing Calibrated Data Array \n')
    file = f'{args.outdir}/{args.juliandate}_fhd_calibrated_data.uvfits'
    calData.write_uvfits(file)
#     with open(file,'wb') as f:
#         np.save(f, cal_array)
#     del cal_array
    del calData

if int(args.MODEL) == 1:
    print('Reading model data') 
#     cal_array = np.ones((Ntimes,Nbls,Nfreqs),dtype=np.cdouble)
#     for i,flist in enumerate(fhd_file_array):
#         if i%10 == 0:
#             print(f'Reading {i}th file of {len(fhd_file_array)}')
#         if obs[i] == 0:
#             continue
#         if float(obs[i][4:17]) < startJD or float(obs[i][4:17]) > stopJD:
#             if clip_data:
#                 print(f'Skipping obs {obs[i]} for being outside JD range')
#                 continue
#         calData = UVData()
#         calData.read(flist,use_model=True)
#         calData.select(polarizations=[-5])
#         calData.phase_to_time(mid_jd)
#         d = calData.data_array
#         t = len(np.unique(calData.time_array))
#         d = np.reshape(d,(t,-1,calData.Nfreqs))
#         cal_array[i*t:i*t+t,:,:] = d
    calData = UVData()
    calData.read(fhd_file_array,use_model=True,ignore_name=True)
    calData.select(polarizations=[-5])
    print('Writing model Data Array \n')
    file = f'{args.outdir}/{args.juliandate}_fhd_model_data.uvfits'
    calData.write_uvfits(file)
#     with open(file, 'wb') as f:
#         np.save(f, cal_array)
#     del cal_array
    del calData


if int(args.SSINS) == 1:
    print('Reading flags')
    flags = UVFlag()
    flags.read(ssins_files)
    flags.select(freq_chans = freqs)
    flags.select(polarizations=['xx'])
#     flags = flags.flag_array[:,845:1090,0]

    print('Writing flags')
    flags.write(f'{args.outdir}/{args.juliandate}_ssins_flags.hdf5',clobber=True)
#     np.save(f'{args.outdir}/{args.juliandate}_ssins_flags', flags)


