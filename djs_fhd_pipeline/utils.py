"""Licensed under the MIT License"""
"""Written by Dara Storer"""

import numpy as np
from pyuvdata import UVData, UVFlag, UVCal
from matplotlib import pyplot as plt
import os
import inspect
from hera_commissioning_tools import utils
import glob
import json
from pyuvdata import utils as pyutils
from djs_fhd_pipeline import plot_fits
import warnings
import random
import yaml

dirpath = os.path.dirname(os.path.realpath(__file__))
githash = utils.get_git_revision_hash(dirpath)
curr_file = __file__

warnings.filterwarnings("ignore", message='This figure includes Axes that are not compatible with tight_layout, so results might be incorrect.')

def read_fhd(datadir,rawfiles='',file_range='all',readRaw=True,readCal=True,readModel=True,readGains=True,
            pol=['XX','YY'],print_message=False):
    model_files, vis_files, params_files, obs_files, flag_files, layout_files, settings_files, cal_files = getFhdFilenames(datadir,data_type='all',pol=pol)
    # print(obs_files[0])
    file_inds=[]
    for c in obs_files:
        try:
            file_inds.append(int(c.split('/')[-3].split('_')[-1]))
        except:
            continue
    # print(file_inds)
    if file_range=='all':
        file_range=[0,len(model_files)]
    elif type(file_range)==int:
        file_range=[file_range-file_inds[0],file_range+1-file_inds[0]]
    cal = UVData()
    if readCal:
        if print_message:
            print('Reading cal')
        cal.read(vis_files[file_range[0]:file_range[1]],
                 params_file=params_files[file_range[0]:file_range[1]],
                 obs_file=obs_files[file_range[0]:file_range[1]],
                 flags_file=flag_files[file_range[0]:file_range[1]],
                 layout_file=layout_files[file_range[0]:file_range[1]],
                 settings_file=settings_files[file_range[0]:file_range[1]],
                 axis='blt')
    model = UVData()
    if readModel:
        if print_message:
            print('Reading model')
        model.read(model_files[file_range[0]:file_range[1]],
                 params_file=params_files[file_range[0]:file_range[1]],
                 obs_file=obs_files[file_range[0]:file_range[1]],
                 flags_file=flag_files[file_range[0]:file_range[1]],
                 layout_file=layout_files[file_range[0]:file_range[1]],
                 settings_file=settings_files[file_range[0]:file_range[1]],
                   axis='blt')
    raw = UVData()
    if readRaw:
        if print_message:
            print('Reading raw')
        with open(rawfiles, 'r') as file:
            rf = yaml.safe_load(file).split(' ')
        raw.read(rf[file_range[0]:file_range[1]], axis='blt')
    gains = UVCal()
    if readGains:
        # print(file_range)
        if print_message:
            print('Reading gains')
        gains.read_fhd_cal(cal_file=cal_files[file_range[0]:file_range[1]],
                obs_file=obs_files[file_range[0]:file_range[1]],
                layout_file=layout_files[file_range[0]:file_range[1]],
                settings_file=settings_files[file_range[0]:file_range[1]])
    return raw, cal, model, gains

def apply_per_pol_flags(uv,exants_x,exants_y,exants_b=None):
    # Get xants from files
    with open(exants_x, 'r') as xfile:
        xants_x = yaml.safe_load(xfile)
    with open(exants_y, 'r') as xfile:
        xants_y = yaml.safe_load(xfile)
    if exants_b is not None:
        with open(exants_b, 'r') as xfile:
            xants_b = yaml.safe_load(xfile)

    # Get set of antennas to fully exclude if they are in xants_x and xants_y or are in xants_b
    use_ants = []
    for a in uv.get_ants():
        if a in xants_x and a in xants_y:
            continue
        else:
            use_ants.append(a)
    # Remove ants flagged for both pols from data entirely
    uv.select(antenna_nums=use_ants)
    print(f'After flagging, {len(uv.get_ants())} antennas in data')

    # Determine if each baseline should be flagged. If either pol is flagged, both cross pols will also be flagged. I.E. if an antenna is in xants_x but not in xants_y, the polarizations XX, XY, and YX will all be flagged, and only YY will remain unflagged.
    antpairpols = uv.get_antpairpols()
    x_flag_bls = []
    y_flag_bls = []
    for a in antpairpols:
        if a[2] != 'ee':
            if a[0] in xants_x or a[1] in xants_x:
                x_flag_bls.append(a)
        if a[2] != 'nn':
            if a[0] in xants_y or a[1] in xants_y:
                y_flag_bls.append(a)
    ex_d = uv.get_data(use_ants[1],use_ants[2],'XX')
    flags = np.ones((np.shape(ex_d)[0],1,np.shape(ex_d)[1],1))

    # For each baseline that was determined to be flagged, manually set the flag array to all ones.
    for bl in y_flag_bls:
        uv.set_flags(flags,bl)
    for bl in x_flag_bls:
        uv.set_flags(flags,bl)
    print(f'Flagging {len(y_flag_bls)} additional baselines based on Y pol flags')
    print(f'Flagging {len(x_flag_bls)} additional baselines based on X pol flags')
    return uv, use_ants

def get_incomplete_FHD_run_inds(fhd_dir,i_range='all',inc_no_beam=True):
    dirs = sorted(glob.glob(f'{fhd_dir}/fhd_*'))
    inds = [int(d.split('_')[-1]) for d in dirs]

    if i_range=='all':
        i_range = [1,np.max(inds)]
    inc_i = []
    for i in np.arange(i_range[0],i_range[1]):
        if i not in inds:
            inc_i.append(i)
    if inc_no_beam:
        for d in dirs:
            if not os.path.isdir(f'{d}/beams'):
                ind = int(d.split('_')[-1])
                if ind>=i_range[0] and ind<= i_range[1]:
                    inc_i.append(ind)
    
    bashstr = ''
    for i in inc_i:
        bashstr += str(i)
        bashstr += ','
    
    return inc_i, bashstr[:-1]

def get_incomplete_FHD_run_inds_noSubdirs(fhd_dir,f_use,i_range='all',inc_no_beam=True):
    dirs = sorted(glob.glob(f'{fhd_dir}/Healpix/*'))
    inds = [int(d.split('_')[-3]) for d in dirs]
    jds = [d.split('zen.')[1] for d in dirs]
    jds = np.unique([float(j.split('_')[0]) for j in jds])
    # print(jds)
    # dfs = np.diff(jds)
    # print(np.where(dfs>0.0005))
    with open(f_use, 'r') as xfile:
        X = yaml.safe_load(xfile).split(' ')
    jds_use = [d.split('zen.')[1] for d in X]
    jds_use = np.unique([float(j.split('_')[0]) for j in jds_use])

    jds_missing = []
    for i,jd in enumerate(jds_use):
        if jd not in jds:
            print(f'missing JD {jd}, index {i}')
            jds_missing.append(jd)

    # bad_inds = []
    # if i_range=='all':
    #     i_range = [1,np.max(inds)]
    # for i in np.arange(i_range[0],i_range[1]):
    #     i_c = inds.count(i)
    #     if i_c<20:
    #         print(f'Index {i} missing outputs')
    #         bad_inds.append(i)
    
    return jds_missing

def get_incomplete_uvfits_run_inds(uvfits_dir,i_range='all'):
    dirs = sorted(glob.glob(f'{uvfits_dir}/*.uvfits'))
    inds = [int(d.split('_')[-1].split('.')[0]) for d in dirs]

    if i_range=='all':
        i_range = [1,np.max(inds)]
    inc_i = []
    for i in np.arange(i_range[0],i_range[1]):
        if i not in inds:
            inc_i.append(i)
    
    bashstr = ''
    for i in inc_i:
        bashstr += str(i)
        bashstr += ','
    
    return inc_i, bashstr[:-1]

def get_exants(write_yaml=False,outfile='',keep_ants=[],**kwargs):
    exants = []
    for key in kwargs.keys():
        for a in kwargs[key]:
            if a not in exants and a not in keep_ants:
                exants.append(a)
    exants = sorted(exants)
    if write_yaml:
        import yaml
        import os

        dirname = os.path.dirname(outfile)
        if not os.path.isdir(dirname):
            os.mkdir(dirname)
        with open(outfile, 'w') as file:
            yaml.dump(exants,file)
        return exants
    else:
        return exants

def getFhdFilenames(datadir,pol=['XX','YY'],data_type='gains', dir_range='all', group_files=False):
    import glob
    fhd_dirs = sorted(glob.glob(f'{datadir}/fhd_2*'))
    if dir_range != 'all':
        fhd_dirs = fhd_dirs[dir_range[0]:dir_range[1]]

    vis_files = []
    model_files = []
    params_files = []
    obs_files = []
    layout_files = [] 
    flag_files = []
    settings_files = []
    cal_files = []

    for i,dir in enumerate(fhd_dirs):
        try:
            params_files.append(sorted(glob.glob(f'{dir}/metadata/*_params.sav'))[0])
            obs_files.append(sorted(glob.glob(f'{dir}/metadata/*_obs.sav'))[0])
            layout_files.append(sorted(glob.glob(f'{dir}/metadata/*_layout.sav'))[0])
            flag_files.append(sorted(glob.glob(f'{dir}/vis_data/*flags.sav'))[0])
            settings_files.append(sorted(glob.glob(f'{dir}/metadata/*_settings.txt'))[0])
            if type(pol)==str:
                vis_files.append(sorted(glob.glob(f'{dir}/vis_data/*vis_{pol}.sav'))[0])
                model_files.append(sorted(glob.glob(f'{dir}/vis_data/*model_{pol}.sav'))[0])
            else:
                vx = sorted(glob.glob(f'{dir}/vis_data/*vis_{pol[0]}.sav'))[0]
                vy = sorted(glob.glob(f'{dir}/vis_data/*vis_{pol[1]}.sav'))[0]
                mx = sorted(glob.glob(f'{dir}/vis_data/*model_{pol[0]}.sav'))[0]
                my = sorted(glob.glob(f'{dir}/vis_data/*model_{pol[1]}.sav'))[0]
                vis_files.append([vx,vy])
                model_files.append([mx,my])
            cal_files.append(sorted(glob.glob(f'{dir}/calibration/*cal.sav'))[0])
        except:
            vis_files.append(None)
            model_files.append(None)
            params_files.append(None)
            obs_files.append(None)
            layout_files.append(None) 
            flag_files.append(None)
            settings_files.append(None)
            cal_files.append(None)
    if group_files:
        fhd_files = []
        for i in range(len(fhd_dirs)):
            if data_type == 'gains':
                fhd_files.append([cal_files[i], obs_files[i], layout_files[i], settings_files[i]])
            elif data_type == 'data':
                fhd_files.append([vis_files[i], params_files[i], obs_files[i], flag_files[i], layout_files[i], settings_files[i]])
            elif data_type == 'model':
                fhd_files.append([model_files[i], params_files[i], obs_files[i], flag_files[i], layout_files[i], settings_files[i]])
            elif data_type == 'all':
                fhd_files.append([model_files[i], vis_files[i], params_files[i], obs_files[i], flag_files[i], layout_files[i], settings_files[i], cal_files[i]])
        return fhd_files

    if data_type == 'gains':
        return cal_files, obs_files, layout_files, settings_files
    elif data_type == 'data':
        return vis_files, params_files, obs_files, flag_files, layout_files, settings_files
    elif data_type == 'model':
        return model_files, params_files, obs_files, flag_files, layout_files, settings_files
    elif data_type == 'all':
        return model_files, vis_files, params_files, obs_files, flag_files, layout_files, settings_files, cal_files