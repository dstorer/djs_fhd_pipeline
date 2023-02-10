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
import json

parser = argparse.ArgumentParser()
parser.add_argument('-r','--raw_files', help='The path to a txt file containing the path to all raw uvfits files to be executed on')
parser.add_argument('-f','--fhd_files', help='The path to a txt file containing the path to the fhd output directory to be executed on')
parser.add_argument('-s','--ssins_files', help='The path to a txt file containing the path to the ssins flag files to be executed on')
parser.add_argument('-o','--outdir', help='Path to write all outputs to')
parser.add_argument('-x','--xants', default=None, help='Path to yml with list of antennas to exclude')
parser.add_argument('-j','--juliandate', help='JD of observations')
parser.add_argument('-p','--pol', help='Polarization to read and write, can be XX or YY')
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

if args.pol == 'XX':
    pol = [-5]
    polname = 'XX'
elif args.pol == 'YY':
    pol = [-6]
    polname = 'YY'
else:
    print('ERROR: POL PARAMETER MUST BE EITHER XX OR YY')

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
if int(args.SSINS)==1:
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
#     f = open(args.fhd_files, "r")
#     with open(file, 'wb') as f:
# #         for line in fhd_file_array:
# #             print(line)
# #             f.write(f"{line}\n")
#         pickle.dump(fhd_file_array,f)
    with open(args.fhd_files, "r") as input:
        with open(file, "w") as output:
            for line in input:
                output.write(line)
            
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

if int(args.BLS) == 1:
    file_read = False
    for i,flist in enumerate(fhd_file_array):
        if obs[i] == 0:
            continue
        if float(obs[i][4:17]) < startJD or float(obs[i][4:17]) > stopJD:
            continue
        if file_read is False:
            calData = UVData()
            print(flist)
            calData.read(flist,use_model=False)
            calData.select(polarizations=pol)
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
    
if args.xants is not None:
    with open(args.xants, 'r') as xfile:
        xants = yaml.safe_load(xfile)
    use_ants = [a for a in raw.get_ants() if a not in xants]
    raw.select(antenna_nums=use_ants)
print('Performing baseline selection on raw data to match baseline set in cal and model data')
print('Raw baselines:')
print(raw.get_antpairs())
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

if int(args.ITER) == 1 or int(args.CONV) == 1:
    print('Reading iter and conv')
    if pol[0]==-5:
        polind = 0
    elif pol[0]==-6:
        polind = 1
    iters = {}
    convs = {}
    dict_init = False
    for i,path in enumerate(fhd_files):
#         print(dict_init)
        useNone = False
        cf = sorted(glob.glob(f'{path}calibration/*cal.sav'))
        try:
            cf = cf[0]
        except:
            print(f'Using None for obs {path}')
            useNone = True
        if useNone is False:
            this_dict = readsav(cf, python_dict=True)
            cal_data = this_dict["cal"]
            darr = cal_data["conv_iter"][0][polind]
            carr = cal_data["convergence"][0][polind]
        else:
            d = np.empty(Nfreqs)
            c = np.empty(Nfreqs)
            d[:] = None
            c[:] = None
        for ant in use_ants:
            if useNone is False:
                d = darr[ant,:]
                c = carr[ant,:]
            if dict_init is False:
                iters[str(ant)] = d
                convs[str(ant)] = c
            else:
#                 if ant ==40:
#                     print('Appending for ant 40')
                iters[str(ant)] = np.append(iters[str(ant)],d)
                convs[str(ant)] = np.append(convs[str(ant)],c)
        dict_init = True
#         print(i)
#         print(np.shape(iters['40']))
    for ant in use_ants:
        iters[str(ant)] = np.reshape(iters[str(ant)],(len(fhd_files),Nfreqs)).tolist()
        convs[str(ant)] = np.reshape(convs[str(ant)],(len(fhd_files),Nfreqs)).tolist()
    file = f'{args.outdir}/{args.juliandate}'
    print('Writing iters and convs \n')
    json.dump(iters, open(f'{file}_iters_{polname}.txt','w'))
    json.dump(convs, open(f'{file}_convs_{polname}.txt','w'))
    

if int(args.RAW) == 1:
    print('Reading raw data') 
    rawData = UVData()
    if clipFreqs:
        rawData.read(raw_data,polarizations=pol,freq_chans=freqs,bls=bls)
    else:
#         rawData.read(raw_data,polarizations=pol,bls=bls)
        rawData.read(raw_data,polarizations=pol)
    
    print('Writing Raw Data Array \n')
    file = f'{args.outdir}/{args.juliandate}_fhd_raw_data_{polname}.uvfits'
    rawData.write_uvfits(file,fix_autos=True)
    del rawData
    
if int(args.GAINS) == 1:
    print('Reading gains')
    
    calfiles = []
    obsfiles = []
    layoutfiles = []
    settingsfiles = []
    ngainsets = 0
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
            print(f'Skipping obs {path} and writing gains up to this point')
            if len(calfiles)==0:
                continue
            g = UVCal()
            g.read_fhd_cal(cal_file=calfiles,obs_file=obsfiles,layout_file=layoutfiles,settings_file=settingsfiles,
                        run_check=False,run_check_acceptability=False)
            file = f'{args.outdir}/{args.juliandate}_fhd_gains_{polname}_{ngainsets}.uvfits'
            print('Writing Gains \n')
            g.write_calfits(file,clobber=True)
            ngainsets+=1
            calfiles = []
            obsfiles = []
            layoutfiles = []
            settingsfiles = []
            continue
    
    g = UVCal()
#     g.read_fhd_cal(cal_file=calfiles,obs_file=obsfiles,layout_file=layoutfiles,settings_file=settingsfiles,
#                 run_check=False,run_check_acceptability=False)
    g.read_fhd_cal(cal_file=calfiles,obs_file=obsfiles,settings_file=settingsfiles,
                run_check=False,run_check_acceptability=False)
    if ngainsets==0:
        file = f'{args.outdir}/{args.juliandate}_fhd_gains_{polname}.uvfits'
    else:
        file = f'{args.outdir}/{args.juliandate}_fhd_gains_{polname}_{ngainsets}.uvfits'
    print('Writing Gains \n')
    g.write_calfits(file,clobber=True)

if int(args.CAL) == 1:
    print('Reading calibrated data') 
    nobs = int(args.nobs)
    calData = UVData()
    calData.read(fhd_file_array,use_model=False,ignore_name=True,polarizations=pol)
#     calData.select(polarizations=pol)
    print('Writing Calibrated Data Array \n')
    file = f'{args.outdir}/{args.juliandate}_fhd_calibrated_data_{polname}.uvfits'
    calData.write_uvfits(file,fix_autos=True)
    del calData

if int(args.MODEL) == 1:
    print('Reading model data') 
    calData = UVData()
    calData.read(fhd_file_array,use_model=True,ignore_name=True)
    calData.select(polarizations=pol)
    print('Writing model Data Array \n')
    file = f'{args.outdir}/{args.juliandate}_fhd_model_data_{polname}.uvfits'
    calData.write_uvfits(file,fix_autos=True)
    del calData


if int(args.SSINS) == 1:
    print('Reading flags')
    flags = UVFlag()
    flags.read(ssins_files)
    flags.select(freq_chans = freqs)
    flags.select(polarizations=pol)

    print('Writing flags')
    flags.write(f'{args.outdir}/{args.juliandate}_ssins_flags_{polname}.hdf5',clobber=True)


