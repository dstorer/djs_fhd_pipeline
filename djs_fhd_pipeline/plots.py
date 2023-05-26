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

dirpath = os.path.dirname(os.path.realpath(__file__))
githash = utils.get_git_revision_hash(dirpath)
curr_file = __file__

import json

def readFiles(datadir, jd, raw=None, cal=None, model=None, gains=None, extraCal=False, incGains=True, pol='XX'):
    if raw is None:
        print('Reading raw')
        raw = UVData()
        raw_file = sorted(glob.glob(f'{datadir}/*_raw_data_{pol}_0.uvfits'))
#         raw_file = f'{datadir}/{jd}_fhd_raw_data_{pol}.uvfits'
        raw.read(raw_file)
        raw.file_name = raw_file
    else:
        raw_file=None
    if cal is None:
        print('Reading cal')
        cal = UVData()
        cal_file = f'{datadir}/{jd}_fhd_calibrated_data_{pol}.uvfits'
        cal.read(cal_file)
        cal.file_name = cal_file
    else:
        cal_file=None
    if extraCal is True and cal2 is None:
        print('Reading extraCal')
        cal2 = UVData()
        cal2_file = f'{datadir}/{jd}_fhd_calibrated_data_{pol}_1.uvfits'
        cal2.read(cal2_file)
        cal2.file_name = cal2_file
    else:
        cal2_file=None
    if model is None:
        print('Reading model')
        model = UVData()
        model_file = f'{datadir}/{jd}_fhd_model_data_{pol}.uvfits'
        model.read(model_file)
        model.file_name=model_file
    else:
        model_file=None
    if incGains is True and gains is None:
        print('Reading Gains')
        gains = UVCal()
        gains_file = f'{datadir}/{jd}_fhd_gains_{pol}_0.uvfits'
        gains.read_calfits(gains_file)
        gains.file_name=gains_file
    else:
        gains_file=None
        
    if extraCal is not False:
        return raw, cal, model, gains, extraCal
    else:
        return raw, cal, model, gains

def plot_uv(uv, freq_synthesis=True, savefig=False, outfig='', nb_path = None, write_params=True, file_ext='jpeg',
           hexbin=True, blx=[], bly=[]):
    import scipy
    import random
    args = locals()
    
    fig = plt.figure(figsize=(10,8))
    freqs = uv.freq_array[0]
    wl = scipy.constants.speed_of_light/freqs
    bls = np.unique(uv.baseline_array)
    a1s,a2s = pyutils.baseline_to_antnums(bls, uv.Nants_data)
    antpos = uv.antenna_positions + uv.telescope_location
    antpos = pyutils.ENU_from_ECEF(antpos, *uv.telescope_location_lat_lon_alt)
    allants = uv.antenna_numbers
    if len(blx)==0 or len(bly)==0:
        for i,a1 in enumerate(a1s):
            if i%200==0:
                print(f'On step {i} of {len(a1s)}')
            for a2 in a2s:
                a1ind = np.argmin(abs(np.subtract(allants,a1)))
                a2ind = np.argmin(abs(np.subtract(allants,a2)))
                a1pos = antpos[a1ind]
                a2pos = antpos[a2ind]
                bl = np.subtract(a2pos,a1pos)
                if freq_synthesis:
                    for w in wl[::10]:
                        blx.append(bl[0]/w)
                        bly.append(bl[1]/w)
                else:
                    bl = bl/wl[len(wl)//2]
                    blx.append(bl[0])
                    bly.append(bl[1])
    if hexbin:
        hb = plt.hexbin(blx, bly, gridsize=50, cmap='inferno',bins='log')
        cb = fig.colorbar(hb)
    else:
        if len(blx)>100000:
            blx,bly = zip(*random.sample(list(zip(blx, bly)), 10000))
        plt.scatter(blx,bly,alpha=0.2,c='black',s=1)
    plt.xlabel('U (wavelengths)')
    plt.ylabel('V (wavelengths)')
    plt.title(f'UV Coverage - {len(bls)} total baselines')
    if savefig:
        outname = f'{outfig}.{file_ext}'
        plt.savefig(outname,bbox_inches='tight')
        if write_params:
            curr_func = inspect.stack()[0][3]
            utils.write_params_to_text(outfig,args,curr_file=curr_file,
                                       curr_func=curr_func,githash=githash,nb_path=nb_path)
        plt.close()
    else:
        plt.show()
        plt.close()
    return blx,bly

def make_frames(fhd_path,uvfits_path,outdir,pol='XX',savefig=True,jd=2459122,ra_range=9,dec_range=9,
               nframes='all',file_ext='jpeg',outsuffix='',write_params=True,plot_range='all',**kwargs):
    # To compile frames into movie after this script runs, on terminal execute: 
    # ffmpeg 2459911_wholeImage_XX.mp4 -r 5 -i 2459911_frame_XX_wholeImage_%3d.jpeg
    print(f'{fhd_path}/fhd_*/output_data/*Dirty_{pol}.fits')
    dirty_files = sorted(glob.glob(f'{fhd_path}/fhd_*/output_data/*Dirty_{pol}.fits'))
    model_files = sorted(glob.glob(f'{fhd_path}/fhd_*/output_data/*Model_{pol}.fits'))
    residual_files = sorted(glob.glob(f'{fhd_path}/fhd_*/output_data/*Residual_{pol}.fits'))
    beam_files = sorted(glob.glob(f'{fhd_path}/fhd_*/output_data/*Beam_{pol}.fits'))
    uvfits_files = sorted(glob.glob(f'{uvfits_path}/*.uvfits'))
    params_files = sorted(glob.glob(f'{fhd_path}/fhd_*/metadata/*_params.sav'))
    obs_files = sorted(glob.glob(f'{fhd_path}/fhd_*/metadata/*_obs.sav'))
    layout_files = sorted(glob.glob(f'{fhd_path}/fhd_*/metadata/*_layout.sav'))
    dirty_vis_files = sorted(glob.glob(f'{fhd_path}/fhd_*/vis_data/*vis_{pol}.sav'))
    flag_files = sorted(glob.glob(f'{fhd_path}/fhd_*/vis_data/*flags.sav'))
    settings_files = sorted(glob.glob(f'{fhd_path}/fhd_*/metadata/*_settings.txt'))
    args = locals()
    print(f'Creating images for {len(dirty_files)} files')
    if nframes=='all':
        nframes = len(dirty_files)
    if plot_range=='all':
        plot_range = [0,nframes]
    for ind in range(plot_range[0],plot_range[1]):
        print(str(ind).zfill(3))
        uv = UVData()
        fhd_files = [layout_files[ind],obs_files[ind],params_files[ind],dirty_vis_files[ind],flag_files[ind],settings_files[ind]]
        uv.read_fhd(fhd_files,read_data=False)
        lst = uv.lst_array[0] * 3.819719 
        pos = [uv.phase_center_app_ra*57.2958,-31]
        _dec_range = [pos[1]-dec_range/2,pos[1]+dec_range/2]
        prefix = f'{fhd_path}/output_data'
        color_scale=[-1763758,1972024]
        output_path = ''
        write_pixel_coordinates=False
        log_scale=False

        fig, axes = plt.subplots(2,2,figsize=(20,20))
        data = plot_fits.load_image(dirty_files[ind])
        im = plot_fits.plot_fits_image(data, axes[0][0], color_scale, output_path, prefix, write_pixel_coordinates, log_scale,
                                 ra_range=ra_range,dec_range=_dec_range,title='Calibrated')
        data = plot_fits.load_image(model_files[ind])
        im = plot_fits.plot_fits_image(data, axes[0][1], color_scale, output_path, prefix, write_pixel_coordinates, log_scale,
                                 ra_range=ra_range,dec_range=_dec_range,title='Model')
        data = plot_fits.load_image(residual_files[ind])
        vmin = np.percentile(data.signal_arr,5)
        vmax = np.percentile(data.signal_arr,95)
        im = plot_fits.plot_fits_image(data, axes[1][0], [vmin,vmax], output_path, prefix, write_pixel_coordinates, log_scale,
                                 ra_range=ra_range,dec_range=_dec_range,title='Residual')
        sources = plot_fits.gather_source_list()
        im = plot_fits.plot_sky_map(uv,axes[1][1],dec_pad=55,ra_pad=55,clip=False,sources=sources)
        ax = plt.gca()
        plt.suptitle(f'LST {np.around(lst,2)}',fontsize=30,y=0.92)
        if savefig == True:
            if len(outsuffix) > 0:
                outfig = f'{outdir}/{jd}_frame_{pol}_{outsuffix}_{str(ind).zfill(3)}.{file_ext}'
            else:
                outfig = f'{outdir}/{jd}_frame_{pol}_{str(ind).zfill(3)}.{file_ext}'
            plt.savefig(outfig,facecolor='white')
            if write_params and ind==0:
#                 curr_func = inspect.stack()[0][3]
                utils.write_params_to_text(outfig,args,curr_func='make_frames',
                       curr_file=curr_file,**kwargs)
        else:
            plt.show()
        plt.close()

def make_frames_with_vis(fhd_path,uvfits_path,outdir,dirty_vis,model_vis,gains,raw_vis,flags,lsts,
                         pol='XX',savefig=False,ant=99,lst_mask=False,jd=2459855,write_params=True):
    dirty_files = sorted(glob.glob(f'{fhd_path}/fhd_*/output_data/*Dirty_{pol}.fits'))
    model_files = sorted(glob.glob(f'{fhd_path}/fhd_*/output_data/*Model_{pol}.fits'))
    residual_files = sorted(glob.glob(f'{fhd_path}/fhd_*/output_data/*Residual_{pol}.fits'))
    beam_files = sorted(glob.glob(f'{fhd_path}/fhd_*/output_data/*Beam_{pol}.fits'))
    source_files = sorted(glob.glob(f'{fhd_path}/fhd_*/output_data/*Sources_{pol}.fits'))
    args = locals()
#     uvfits_files = sorted(glob.glob(f'{uvfits_path}/*.uvfits'))
    
#     lst_blacklists = "0-1.3 2.5-4.3 6.5-9.1 10.6-11.5 11.9-14.3 16.3-1.3"
    lst_blacklists = [(0,1.3),(2.5,4.3),(6.5,9.1),(10.6,11.5),(11.9,14.3),(16.3,24)]
    
    if lst_mask:
        alpha_mask = np.ones(np.shape(raw_vis))
        for l,lst in enumerate(np.multiply(lsts,3.819719)):
            mask = False
            for r in lst_blacklists:
    #             print(r)
                if r[0]<=lst<=r[1]:
    #                 print(f'LST {lst} in range {r}')
                    mask = True
            if mask == True:
                alpha_mask[l,:] = np.ones((1,np.shape(raw_vis)[1]),dtype=float)*0.4
            else:
                alpha_mask[l,:] = np.ones((1,np.shape(raw_vis)[1]),dtype=float)
    else:
        alpha_mask = 1
            
    
    print(f'Creating images for {len(dirty_files)} files')
    for ind in range(0,len(dirty_files)):
#     for ind in range(10,11):
        if ind%20==0:
            print(str(ind).zfill(3))
        uv = UVData()
        uv.read(uvfits_path)
        freqs = uv.freq_array[0]*1e-6
        pos = [uv.phase_center_app_ra*57.2958,-31]
        dec_range = [pos[1]-4.5,pos[1]+4.5]
        ra_range=9
#         print(f'Pos: {pos}')
#         ra_range = [pos[0]-6,pos[0]+7.8]
# #         ra_range = np.subtract(ra_range,360)
# #         print(ra_range)
#         if ra_range[0] > 180:
#             ra_range[0] = ra_range[0]-360
#         if ra_range[1] > 180:
#             ra_range[1] = ra_range[1]-360
# #         ra_range=None

#         dec_range = [pos[1]-6.5,pos[1]+6.5]
# #         dec_range=None
# #         print('dec_range')
# #         print(dec_range)
        prefix = f'{fhd_path}/output_data'
        color_scale=[-1763758,1972024]
        output_path = ''
        write_pixel_coordinates=False
        log_scale=False
        hline_frac = (ind/len(dirty_files))

        fig = plt.figure(figsize=(20,20))
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        gs = fig.add_gridspec(5, 3,wspace=0.2,hspace=0.3)
        #Beam
        beam = plot_fits.load_image(beam_files[ind])
        ax = fig.add_subplot(gs[0, 0])
        im = plot_fits.plot_fits_image(beam, ax, [0,1], output_path, prefix, write_pixel_coordinates, log_scale,
                                 ra_range=ra_range,dec_range=dec_range,title='Beam')
#         #raw visibilities
#         ax = fig.add_subplot(gs[0,1:])
# #         raw_dat = np.ma.masked_where(flags[:,0,:,0]==True,np.abs(raw_vis))
#         raw_dat = np.abs(raw_vis)
#         im = plot_vis.plot_raw_vis(raw_dat,ax,ant,lsts,freqs,alpha_mask=alpha_mask)
#         ax.axhline(hline_frac*len(raw_vis),color='c')
        
        #dirty image
        data = plot_fits.load_image(dirty_files[ind])
        ax = fig.add_subplot(gs[1, 0])
#         print(ax)
        im = plot_fits.plot_fits_image(data, ax, color_scale, output_path, prefix, write_pixel_coordinates, log_scale,
                                 ra_range=ra_range,dec_range=dec_range,title='Dirty')
        #dirty vis
#         ax = fig.add_subplot(gs[1, 1:])
#         im = plot_vis.plot_vis(dirty_vis,ax,(ant,ant),dtype='dirty',alpha_mask=alpha_mask)
#         ax.axhline(hline_frac*len(dirty_vis['amp'][(ant,ant)]),color='c')
        #model image
        data = plot_fits.load_image(model_files[ind])
        ax = fig.add_subplot(gs[2, 0])
#         print(ax)
        im = plot_fits.plot_fits_image(data, ax, color_scale, output_path, prefix, write_pixel_coordinates, log_scale,
                                 ra_range=ra_range,dec_range=dec_range,title='Model')
        #model vis
#         ax = fig.add_subplot(gs[2, 1:])
#         im = plot_vis.plot_vis(model_vis,ax,(ant,ant),dtype='model',alpha_mask=alpha_mask)
#         ax.axhline(hline_frac*len(model_vis['amp'][(ant,ant)]),color='c')
        #residual image
        data = plot_fits.load_image(residual_files[ind])
        ax = fig.add_subplot(gs[3, 0])
#         print(ax)
        im = plot_fits.plot_fits_image(data, ax, [0,1], output_path, prefix, write_pixel_coordinates, log_scale,
                                 ra_range=ra_range,dec_range=dec_range,title='Residual')
        #gains
#         ax = fig.add_subplot(gs[3, 1:])
#         im = plot_vis.plot_gains(gains,ax,ant,set_alpha_mask=True)
#         ax.axhline(hline_frac*len(gains[ant]['cal_array']),color='c')
        #Source Image
        data = plot_fits.load_image(source_files[ind])
        ax = fig.add_subplot(gs[4, 0])
#         print(ax)
        im = plot_fits.plot_fits_image(data, ax, [0,10000000], output_path, prefix, write_pixel_coordinates, log_scale,
                                 ra_range=ra_range,dec_range=dec_range,title='Sources')
        #sky map
        sources = plot_fits.gather_source_list()
        ax = fig.add_subplot(gs[4, 1:])
        im = plot_fits.plot_sky_map(uv,ax,dec_pad=55,ra_pad=55,clip=False,sources=sources)

        
        ax = plt.gca()
        if savefig == True:
            print('saving')
            outname = f'{outdir}/{jd}_frame_{pol}_{str(ind).zfill(3)}'
            plt.savefig(f'{outname}.png',facecolor='white')
            if write_params and ind==0:
                curr_func = inspect.stack()[0][3]
                utils.write_params_to_text(outname,args,curr_file=curr_file,
                                           curr_func=curr_func,githash=githash)
        else:
            plt.show()
        plt.close()

def getCrossesPerAnt(uv,ant):
    bls = []
    crossAnts = []
    for a in uv.get_ants():
        try:
            _ = uv.get_data(ant,a)
            if a != ant:
                bls.append((ant,a))
                crossAnts.append(a)
        except:
            continue
    return np.asarray(bls), np.asarray(crossAnts)

def plotPerAntCrossesAll(datadir, jd, raw=None, pol='XX', savefig=False, write_params=True, outfig='',
                       file_ext='jpeg',NantsPlot='all',nb_path=None, visNorm=np.abs):
    if raw==None:
        raw_file = f'{datadir}/{jd}_fhd_raw_data_{pol}.uvfits'
        raw = UVData()
        raw.read(raw_file)
    args = locals()
    ants = raw.get_ants()
    Nants = len(ants)
    if NantsPlot == 'all':
        NantsPlot = Nants
    nplots = NantsPlot
    ncols = 6
    
    for n in range(nplots):
        ant = ants[n]
        bls, crossAnts = getCrossesPerAnt(raw,ant)
        
        lengths = np.asarray(getBaselineLength(raw,bls))
        leninds = lengths.argsort()
        lengths = lengths[leninds]
        crossAnts = crossAnts[leninds]
        
        nrows = np.ceil(len(crossAnts)/ncols)
        fig = plt.figure(figsize=(16,24))
        gs = fig.add_gridspec(int(nrows),int(ncols))
        for i,b in enumerate(crossAnts):
            d = visNorm(raw.get_data(ant,b,pol))
            ax1 = fig.add_subplot(gs[i//ncols,i%ncols])
            if visNorm == np.angle:
                vmin = -np.pi
                vmax = np.pi
                cmap = 'twilight'
#                 print(d)
            else:
                vmin=np.percentile(d,10)
                vmax=np.percentile(d,90)
                cmap = 'viridis'
            im1 = ax1.imshow(d,aspect='auto',interpolation='nearest',vmin=vmin,vmax=vmax,cmap=cmap)
            ax1.set_title(f'({ant},{b}) - {int(lengths[i])}m')
#             plt.colorbar(im1,cax=ax1)
            if i%ncols==0:
                ax1.set_ylabel('Time')
            else:
                ax1.set_yticks([])
            ax1.set_xticks([])
        
        if savefig:
            outname = f'{outfig}_{pol}_{ant}.{file_ext}'
            plt.savefig(outname,bbox_inches='tight')
            if write_params and n==0:
                curr_func = inspect.stack()[0][3]
                utils.write_params_to_text(f'{outfig}_{pol}',args,curr_file=curr_file,curr_func=curr_func,githash=githash,nb_path=nb_path)
            plt.close()
        else:
            plt.show()
            plt.close()

def plotPerAntCrossSums(datadir, jd, raw=None, pol='XX', antsPerPage=4, savefig=False, write_params=True, outfig='',
                       file_ext='jpeg'):
    if raw==None:
        raw_file = f'{datadir}/{jd}_fhd_raw_data_{pol}.uvfits'
        raw = UVData()
        raw.read(raw_file)
    args = locals()
    ants = raw.get_ants()
    Nants = len(ants)
    nplots = Nants//antsPerPage
    for n in range(nplots):
        fig = plt.figure(figsize=(16,8*antsPerPage))
        gs = fig.add_gridspec(antsPerPage,2,wspace=0.15,hspace=0.35)
        alist = ants[n*antsPerPage:n*antsPerPage+antsPerPage]
        for i,a in enumerate(alist):
            inc = np.empty((raw.Ntimes,raw.Nfreqs),dtype='complex128')
            coh = np.empty((raw.Ntimes,raw.Nfreqs),dtype='complex128')
            for j,c in enumerate(ants):
                if a==c:
                    continue
                try:
                    d = raw.get_data(a,c,pol)
                except:
                    continue
#                 print(np.shape(d))
#                 print(raw.Ntimes)
#                 print(raw.Nfreqs)
#                 if j==0:
#                     inc = abs(d)
#                     coh = d
#                 else:
                inc += abs(d)
                coh += d
            coh = abs(coh)
            inc = abs(inc)
            ax1 = fig.add_subplot(gs[i,0])
            im1 = ax1.imshow(coh,aspect='auto',interpolation='nearest',vmin=np.percentile(coh,10),vmax=np.percentile(coh,90))
            plt.colorbar(im1)
            ax2 = fig.add_subplot(gs[i,1])
            im2 = ax2.imshow(inc,aspect='auto',interpolation='nearest',vmin=np.percentile(inc,10),vmax=np.percentile(inc,90))
            plt.colorbar(im2)
            if i==0:
                ax1.set_title('abs(sum)')
                ax2.set_title('sum(abs)')
            ax1.set_ylabel(f'ant {a}')
        
        if savefig:
            outname = f'{outfig}_{pol}_{str(n).zfill(2)}.{file_ext}'
            plt.savefig(outname,bbox_inches='tight')
            if write_params and n==0:
                curr_func = inspect.stack()[0][3]
                utils.write_params_to_text(outfig,args,curr_file=curr_file,curr_func=curr_func,githash=githash,nb_path=nb_path)
            plt.close()
        else:
            plt.show()
            plt.close()
            
    
def plotVisAndDelay(datadir, jd, calFull = None, rawFull = None, modelFull = None, gainsFull = None,
                    gainNorm=np.abs,calNorm=np.abs,modelNorm=np.abs,rawNorm=np.abs,
                   savefig=False,outfig='',write_params=True,split_plots=True,nsplit=3,file_ext='pdf',
                    readOnly=False,NblsPlot='all',sortBy='blLength',pol='XX',percentile=90,
                   extraCal=False,cal2=None,lstRange=[],ytick_unit='lst',incGains=False,fringe=False,
                   multiplyGains=False,useFtWindow=False,normDs=False,dsLogScale=True,delayRange=None,
                   nb_path=None,fixLsts=False):
    from uvtools import dspec
    args = locals()
    
    dirlist=None
    rawFull, calFull, modelFull, gainsFull = readFiles(datadir,jd,rawFull,calFull,modelFull,gainsFull,
                                                                 extraCal=extraCal,incGains=incGains,pol=pol)
    
    if incGains:
        times = np.unique(gainsFull.time_array)
        caltimes = np.unique(calFull.time_array)
        rawtimes = np.unique(rawFull.time_array)
        if len(rawtimes)<len(caltimes):
            calFull.select(times=rawtimes)
            caltimes = np.unique(calFull.time_array)
        if caltimes[0]<times[0]:
            ind = np.argmin(abs(np.subtract(caltimes,times[0])))
            caltimes = caltimes[ind:]
        elif caltimes[-1]>times[-1]:
            ind = np.argmin(abs(np.subtract(caltimes,times[-1])))
            caltimes = caltimes[0:ind]
        calFull.select(times=caltimes)
        rawFull.select(times=caltimes)
        modelFull.select(times=caltimes)
    else:
        caltimes = np.unique(calFull.time_array)
        rawFull.select(times=caltimes)
        modelFull.select(times=caltimes)
    
    if fixLsts:
        try:
            loc = rawFull.telescope_location_lat_lon_alt_degrees
            rawLsts = pyutils.get_lst_for_time(rawFull.time_array,loc[0],loc[1],loc[2])
            fhdLsts = pyutils.get_lst_for_time(calFull.time_array,loc[0],loc[1],loc[2])
            print(f'fhdLsts: {fhdLsts}')
            rawFull.lst_array = rawLsts
            calFull.lst_array = fhdLsts
            modelFull.lst_array = rawLsts
        except:
            print('uh oh')
    if len(lstRange)>0:
        l = rawLsts  * 3.819719
        inds = np.unique(l, return_index=True)[1]
        l = [l[ind] for ind in sorted(inds)]
        imin = np.argmin(np.abs(np.subtract(l,lstRange[0])))
        imax = np.argmin(np.abs(np.subtract(l,lstRange[1])))
        caltimes = caltimes[imin:imax]
        raw = rawFull.select(times=caltimes,inplace=False)
        cal = calFull.select(times=caltimes,inplace=False)
        model = modelFull.select(times=caltimes,inplace=False)
        if incGains:
            if caltimes[0]>times[0]:
                ind = np.argmin(abs(np.subtract(times,caltimes[0])))
                times = times[ind:]
            elif caltimes[-1]<times[-1]:
                ind = np.argmin(abs(np.subtract(times,caltimes[-1])))
                times = times[0:ind]
            gains = gainsFull.select(times=times,inplace=False)
    else:
        raw = rawFull
        cal = calFull
        model = modelFull
        gains = gainsFull
    
    freqs = gains.freq_array[0]*1e-6
    lstsRaw = raw.lst_array * 3.819719
    lstsRaw = np.reshape(lstsRaw,(raw.Ntimes,-1))[:,0]
    lstsFHD = cal.lst_array * 3.819719
    lstsFHD = np.reshape(lstsFHD,(cal.Ntimes,-1))[:,0]
    lstsGains = gains.lst_array * 3.819719
    lstsGains = np.reshape(lstsGains,(gains.Ntimes,-1))[:,0]
    
    taus = np.fft.fftshift(np.fft.fftfreq(freqs.size, np.diff(freqs)[0]*1e6))*1e9
    frs = np.fft.fftshift(np.fft.fftfreq(lstsGains.size, np.diff(lstsGains)[0]*3600))*1e3
    if delayRange != None:
        dmin = np.argmin(np.abs(np.subtract(taus,delayRange[0])))
        dmax = np.argmin(np.abs(np.subtract(taus,delayRange[1])))
        taus = taus[dmin:dmax]
    
    if extraCal:
        lstsFHD = np.repeat(lstsFHD,2)
    jdsRaw = np.unique(raw.time_array)
    jdsFHD = np.unique(cal.time_array)
    jdsGains = np.unique(gains.time_array)
    
    xticks = [int(i) for i in np.linspace(0, len(freqs) - 1, 5)]
    xticklabels = [int(freqs[x]) for x in xticks]
    tauticks = [int(i) for i in np.linspace(0, len(taus) - 1, 5)]
    tauticklabels = [int(taus[x]) for x in tauticks]
    frticks = [int(i) for i in np.linspace(0, len(frs) - 1, 6)]
    frticklabels = [np.around(frs[ytick], 1) for ytick in frticks]
    yticksRaw = [int(i) for i in np.linspace(0, len(lstsRaw) - 1, 6)]
    yticksFHD = [int(i) for i in np.linspace(0, len(lstsFHD) - 1, 6)]
    yticksGains = [int(i) for i in np.linspace(0, len(lstsGains) - 1, 6)]
    
    if ytick_unit == 'lst':
        yticklabelsRaw = [np.around(lstsRaw[ytick], 1) for ytick in yticksRaw]
        yticklabelsFHD = [np.around(lstsFHD[ytick], 1) for ytick in yticksFHD]
        yticklabelsGains = [np.around(lstsFHD[ytick], 1) for ytick in yticksFHD]
    elif ytick_unit == 'jd':
        yticklabelsFHD = [np.around(jdsFHD[ytick], 4) for ytick in yticksFHD]
        yticklabelsRaw = [np.around(jdsRaw[ytick], 4) for ytick in yticksRaw]
        yticklabelsGains = [np.around(jdsGains[ytick], 4) for ytick in yticksGains]
    
    allbls = []
    allants = raw.get_ants()
    for a1 in allants:
        for a2 in allants:
            if a2>a1:
                try:
                    _ = cal.get_data(a1,a2)
                    allbls.append((a1,a2))
                except:
                    continue
    allbls = np.asarray(allbls)
    alllengths = np.asarray(getBaselineLength(raw,allbls))
    allangs = []
    alldisps = []
    pos = raw.antenna_positions + raw.telescope_location
    pos = pyutils.ENU_from_ECEF(pos, *raw.telescope_location_lat_lon_alt)
    if sortBy == 'blLength':
        leninds = alllengths.argsort()
        alllengths = alllengths[leninds]
        allbls = allbls[leninds]
    if sortBy == 'blLengthRev':
        leninds = alllengths.argsort()
        alllengths = alllengths[leninds]
        allbls = allbls[leninds]
        allbls = np.flip(allbls)
    for bl in allbls:
        p1 = pos[np.argwhere(raw.antenna_numbers == bl[0])]
        p2 = pos[np.argwhere(raw.antenna_numbers == bl[1])]
        disp = (p2 - p1)[0][0][0:2]
        alldisps.append(disp)
        allangs.append(np.arctan(disp[1]/disp[0])*57.2958)
    allangs = np.asarray(allangs)
    alldisps = np.asarray(alldisps)
    if sortBy == 'blOrientation':
        anginds = allangs.argsort()
        allangs = allangs[anginds]
        allbls = allbls[anginds]
        alldisps = alldisps[anginds]
        allbls = np.flip(allbls)
    if NblsPlot != 'all':
        if NblsPlot>=1:
            allbls = allbls[0:NblsPlot]
        else:
            inds = np.linspace(0,len(allbls)-1,int(len(allbls)*NblsPlot))
            allbls = [allbls[int(i)] for i in inds]
    nbls = len(allbls)
    nplots = nbls//nsplit
    for n in range(nplots):
        if readOnly:
            continue
        bls = allbls[n*nsplit:n*nsplit+nsplit]
        angs = allangs[n*nsplit:n*nsplit+nsplit]
        disps = alldisps[n*nsplit:n*nsplit+nsplit]
        if incGains:
            if multiplyGains:
                ncols = 5
                width_ratios=[1,1,1,1,0.05]
            else:
                ncols = 6
                width_ratios=[1,1,1,1,1,0.05]
        else:
            ncols = 4
            width_ratios=[1,1,1,0.05]
        axes = np.empty((len(bls)*2,ncols),dtype=object)
        fig = plt.figure(figsize=(16,8*nsplit))
        gs = fig.add_gridspec(nsplit*2,ncols,width_ratios=width_ratios,wspace=0.15,hspace=0.35)
        lengths = getBaselineLength(raw,bls)
        maxlength = np.max(lengths)
        for b,bl in enumerate(bls):
            try:
                r = raw.get_data((bl[0],bl[1],pol))
                c = cal.get_data((bl[0],bl[1],pol))
                m = model.get_data((bl[0],bl[1],pol))
                if incGains:
                    g1 = np.transpose(gains.get_gains(bl[0],pol))
                    g2 = np.transpose(gains.get_gains(bl[1],pol))
                    if multiplyGains:
                        gm = np.multiply(g1,np.conjugate(g2))
            except:
                print(f'Skipping baseline ({bl[0]},{bl[1]})')
                continue
            if extraCal:
                c2 = calNorm(cal2.get_data((bl[0],bl[1],pol)))
                call = np.empty((np.shape(c)[0] + np.shape(c2)[0],np.shape(c)[1]),dtype=c.dtype)
                call[0::2,:] = c
                call[1::2,:] = c2
                c = call
            if incGains:
                if multiplyGains:
                    norms = [gainNorm,rawNorm,calNorm,modelNorm]
                    lsts = [lstsGains,lstsRaw,lstsFHD,lstsFHD]
                    dats = [gm,r,c,m]
                    fhdax = [2]
                    gainax = [0]
                else:
                    norms = [gainNorm,rawNorm,calNorm,modelNorm,gainNorm]
                    lsts = [lstsGains,lstsRaw,lstsFHD,lstsFHD,lstsGains]
                    dats = [g1,r,c,m,g2]
                    fhdax = [2]
                    gainax = [0,4]
            else:
                norms = [rawNorm,calNorm,modelNorm]
                dats = [r,c,m]
                fhdax = [1]
                gainax = []
            normDats = []
            for i,d in enumerate(dats):
                nd = norms[i](d)
                normDats.append(nd)
            for i in range(len(dats)):
                ax = fig.add_subplot(gs[b*2,i])
                axd = fig.add_subplot(gs[b*2+1,i])
                axes[b*2,i] = ax
                axes[b*2+1,i] = axd
                norm = norms[i]
                if norm == np.angle:
                    vmin = -np.pi
                    vmax = np.pi
                    cmap = 'twilight'
                else:
                    vmin = np.percentile(normDats[i],100-percentile)
                    vmax = np.percentile(normDats[i],percentile)
                    cmap = 'viridis'
#                 ds = 10.*np.log10(np.sqrt(np.abs(np.fft.fftshift(np.fft.ifft(dats[i],axis=1)))))
                if useFtWindow:
                    if fringe:
                        window_t = dspec.gen_window('bh',len(lsts[i]))
                    else:
                        window_t = np.ones(len(lsts[i]))
                    window_f = dspec.gen_window('bh',len(freqs))
                    window_mat = np.outer(window_t, window_f)
                    ds = np.fft.fftshift(np.fft.ifft(dats[i]*window_mat),axes=1)
                else:
                    ds = np.fft.fftshift(np.fft.ifft(dats[i]),axes=1)
                if fringe:
                    if useFtWindow:
                        ds = np.fft.fftshift(np.fft.ifft2(dats[i]*window_mat))
                    else:
                        ds = np.fft.fftshift(np.fft.ifft2(dats[i]))
#                     vmind = np.percentile(ds,5)
#                     vmaxd = np.percentile(ds,95)
#                 else:
# #                     ds = 10.*np.log10(np.sqrt(np.abs(ds)))
# #                     ds = np.divide(ds,np.nanmean(ds))
                if normDs:
                    ds /= np.max(ds)
                if dsLogScale:
                    ds = np.log10(np.abs(ds))
                vmind = np.percentile(ds,5)
                vmaxd = np.percentile(ds,95)
#                 print(vmind)
#                 print(vmaxd)
                if delayRange:
                    ds = ds[:,dmin:dmax]
                im = ax.imshow(normDats[i],aspect='auto',interpolation='nearest',vmin=vmin,vmax=vmax,cmap=cmap)
                imd = axd.imshow(ds,aspect='auto',interpolation='nearest',vmin=vmind,vmax=vmaxd,cmap='viridis')
            cax = fig.add_subplot(gs[b*2,ncols-1])
            caxd = fig.add_subplot(gs[b*2+1,ncols-1])
            bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            height = bbox.height * fig.dpi
            cax.annotate('', xy=(-40, height/2), xycoords='axes pixels', textcoords='axes pixels', 
                         xytext=(100*np.cos(angs[b]/57.2958)-40, 100*np.sin(angs[b]/57.2958)+height/2), 
                         arrowprops=dict(arrowstyle="<-", color='r',linewidth=4))
            axes[b*2,ncols-1] = cax
            axes[b*2+1,ncols-1] = caxd
        
        for i in range(np.shape(axes)[0]):
            for j in range(np.shape(axes)[1]):
                ax = axes[i,j]
                if ax is None:
                    continue
                a1 = bls[i//2][0]
                a2 = bls[i//2][1]
                if incGains:
                    if multiplyGains:
                        plotnames = [
                            f'{a1}x{a2}* Gains',
                            'Raw Visibilities',
                            'Calibrated Visibilities',
                            'Model Visibilities',
                        ]
                    else:
                        plotnames = [
                            f'{a1} Gains',
                            'Raw Visibilities',
                            'Calibrated Visibilities',
                            'Model Visibilities',
                            f'{a2} Gains'
                        ]
                else:
                    plotnames = [
                        'Raw Visibilities',
                        'Calibrated Visibilities',
                        'Model Visibilities'
                    ]
                if j<ncols-1:
                    if j==0:
                        if j in fhdax:
                            ax.set_yticks(yticksFHD)
                            ax.set_yticklabels(yticklabelsFHD)
                        elif j in gainax:
                            if i%2==0:
                                ax.set_yticks(yticksGains)
                                ax.set_yticklabels(yticklabelsGains)
                            else:
                                ax.set_yticks(frticks)
                                ax.set_yticklabels(frticklabels)
                        else:
                            ax.set_yticks(yticksRaw)
                            ax.set_yticklabels(yticklabelsRaw)
                        if fringe and i%2==1:
                            ax.set_ylabel(f'({a1}, {a2})\n {int(lengths[i//2])}m\n Fringe Rate (MHz)',fontsize=16)
#                             ax.set_yticks([])
                        else:
                            ax.set_ylabel(f'({a1}, {a2})\n {int(lengths[i//2])}m\n Time (LST)',fontsize=16)
                    else:
                        ax.set_yticks([])
                    if i%2==0:
                        ax.set_xticks(xticks)
                        ax.set_xticklabels(xticklabels)
                        ax.set_xlabel('Frequency (MHz)')
                        ax.set_title(plotnames[j],fontsize=14)
                    elif i%2==1:
                        ax.set_xticks(tauticks)
                        ax.set_xticklabels(tauticklabels)
                        ax.set_xlabel('Delay (ns)')
                if j==ncols-1:
                    if i%2==0:
                        plt.colorbar(im,cax=ax)
                    else:
                        plt.colorbar(imd,cax=ax)
        avgLen = int(np.mean(lengths))
        avgAng = int(np.mean(angs))
        if sortBy == 'blLengthRev' or sortBy == 'blLength':
            plt.suptitle(f'Avg Length this panel: {avgLen} meters',fontsize=18)
        elif sortBy == 'blOrientation':
            plt.suptitle(f'Avg angle this panel: {avgAng} degrees from East',fontsize=18)
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        if savefig is True:
            if sortBy == 'blLength' or sortBy == 'blLengthRev':
                outname = f'{outfig}_sortBy{sortBy}_{pol}_{str(avgLen).zfill(3)}m_{str(n).zfill(2)}.{file_ext}'
            elif sortBy == 'blOrientation':
                outname = f'{outfig}_sortBy{sortBy}_{pol}_{str(n).zfill(2)}.{file_ext}'
            else:
                outname = f'{outfig}_sortBy{sortBy}_{pol}_{str(n).zfill(2)}.{file_ext}'
            plt.savefig(outname,bbox_inches='tight')
            if write_params and n==0:
                curr_func = inspect.stack()[0][3]
                utils.write_params_to_text(outfig,args,dirlist=dirlist,curr_file=curr_file,curr_func=curr_func,githash=githash,
                                           raw=raw,calibrated=cal,model=model,djs_fhd_pipeline_githash=githash,outname=outname,
                                          nb_path=nb_path,datadir=datadir)
            plt.close()
        else:
            plt.show()
            plt.close()
            
    return rawFull, calFull, modelFull, gainsFull

def plotBaselineMap(uv,bls='withData'):
    allbls = []
    if bls=='withData':
        allants = uv.get_ants()
    elif bls=='all':
        allants = uv.antenna_numbers
    for a1 in allants:
        for a2 in allants:
            if a2>=a1:
                if bls=='withData':
                    try:
                        _ = uv.get_data(a1,a2)
                        allbls.append((a1,a2))
                    except:
                        continue
                elif bls=='all':
                    allbls.append((a1,a2))
    allbls = np.asarray(allbls)
    print(len(allbls))
    allangs = []
    alldisps = []
    pos = uv.antenna_positions + uv.telescope_location
    pos = pyutils.ENU_from_ECEF(pos, *uv.telescope_location_lat_lon_alt)
    for bl in allbls:
        p1 = pos[np.argwhere(uv.antenna_numbers == bl[0])]
        p2 = pos[np.argwhere(uv.antenna_numbers == bl[1])]
        disp = (p2 - p1)[0][0][0:2]
        alldisps.append(disp)
        allangs.append(np.arctan(disp[1]/disp[0])*57.2958)
    allangs = np.asarray(allangs)
    alldisps = np.asarray(alldisps)
    fig = plt.figure(figsize=(10,10))
    for b,bl in enumerate(allbls):
        plt.scatter(alldisps[b][0],alldisps[b][1],color='blue')
    plt.xlabel('EW Separation (m)')
    plt.ylabel('NS Separation (m)')
    plt.title('HERA Baseline Map')


def plotGainsAndConv(datadir, uvc = None, raw = None, cal = None, model = None,
                    gainNorm=np.abs,rawNorm=np.abs,
                   savefig=False,outfig='',write_params=True,split_plots=True,nsplit=3,file_ext='pdf',
                   jd=2459855,readOnly=False,NblsPlot='all',sortBy='blLength',pol='XX',percentile=90,
                    convParam='conv'):
    from pyuvdata import UVCal
    args = locals()
    
    if uvc is None:
        gf = sorted(glob.glob(f'{datadir}/{jd}_fhd_gains_{pol}*'))
        if len(gf)<3:
            dirlist = f'{datadir}/{jd}_fhd_dirlist.txt'
            calfiles, obsfiles, layoutfiles, settingsfiles = getFhdGainDirs(dirlist)
            uvc = UVCal()
            uvc.read_fhd_cal(calfiles, obsfiles,layoutfiles,settingsfiles)
            uvc.file_name = dirlist
        elif len(gf)==1:
            uvc = UVCal()
            uvc.read_calfits(gf[0])
        
    else:
        dirlist=None
    if raw is None:
        raw = UVData()
        raw_file = f'{datadir}/{jd}_fhd_raw_data_{pol}.uvfits'
        raw.read(raw_file)
        raw.file_name = raw_file
    else:
        raw_file=None
    if cal is None:
        cal = UVData()
        cal_file = f'{datadir}/{jd}_fhd_calibrated_data_{pol}.uvfits'
        cal.read(cal_file)
        cal.file_name = cal_file
    else:
        cal_file=None
#     if model is None:
#         model = UVData()
#         model_file = f'{datadir}/{jd}_fhd_model_data_{pol}.uvfits'
#         model.read(model_file)
#         model.file_name=model_file
#     else:
#         model_file=None
    
    if convParam == 'conv':
        conv = json.load(open(f"{jd}_convs_{pol}.txt","r"))
    elif convParam == 'iter':
        conv = json.load(open(f"{jd}_iters_{pol}.txt","r"))
    
    times = np.unique(cal.time_array)
    raw.select(times=times)
    
    freqs = raw.freq_array[0]*1e-6
    lstsRaw = raw.lst_array * 3.819719
    lstsRaw = np.reshape(lstsRaw,(raw.Ntimes,-1))[:,0]
    lstsFHD = uvc.lst_array * 3.819719
    lstsFHD = np.reshape(lstsFHD,(uvc.Ntimes,-1))[:,0]
    
    xticks = [int(i) for i in np.linspace(0, len(freqs) - 1, 5)]
    xticklabels = [int(freqs[x]) for x in xticks]
    yticksRaw = [int(i) for i in np.linspace(0, len(lstsRaw) - 1, 6)]
    yticklabelsRaw = [np.around(lstsRaw[ytick], 1) for ytick in yticksRaw]
    yticksFHD = [int(i) for i in np.linspace(0, len(lstsFHD) - 1, 6)]
    yticklabelsFHD = [np.around(lstsFHD[ytick], 1) for ytick in yticksFHD]
    
    allbls = np.asarray(np.load(f'{datadir}/{jd}_bl_array.npy'))  
    alllengths = np.asarray(getBaselineLength(raw,allbls))
    if sortBy == 'blLength':
        leninds = alllengths.argsort()
        alllengths = alllengths[leninds]
        allbls = allbls[leninds]
    if sortBy == 'blLengthRev':
        leninds = alllengths.argsort()
        alllengths = alllengths[leninds]
        allbls = allbls[leninds]
        allbls = np.flip(allbls)
    if NblsPlot != 'all':
        allbls = allbls[0:NblsPlot] 
    nbls = len(allbls)
#     print(nbls)
    nplots = nbls//nsplit
#     print(nplots)
    for n in range(nplots):
        print(n)
        if readOnly:
            continue
        bls = allbls[n*nsplit:n*nsplit+nsplit]
        axes = np.empty((len(bls),5),dtype=object)
        fig = plt.figure(figsize=(16,4*nsplit))
        gs = fig.add_gridspec(nsplit,5,width_ratios=[1,1,1,1,1])
        lengths = getBaselineLength(raw,bls)
        maxlength = np.max(lengths)
        for b,bl in enumerate(bls):
            if bl[0]==183 or bl[1]==183 or bl[0]==5 or bl[1]==5:
                continue
            g0 = uvc.get_gains(bl[0])
            g1 = uvc.get_gains(bl[1])
            g0 = np.transpose(gainNorm(g0[:,:,0]))
            g1 = np.transpose(gainNorm(g1[:,:,0]))
            gains = [g0,g1]
            c0 = conv[str(bl[0])]
            c1 = conv[str(bl[1])]
            axinds = [0,4]
            for i in [0,1]:
                if gainNorm == np.angle:
                    vmin = -np.pi
                    vmax = np.pi
                    cmap = 'twilight'
                else:
                    vmin = np.nanpercentile(gains[i],100-percentile)
                    vmax = np.nanpercentile(gains[i],percentile)
                    cmap = 'viridis'
                ax = fig.add_subplot(gs[b,axinds[i]])
                axes[b,axinds[i]] = ax
                im = ax.imshow(gains[i],aspect='auto',cmap=cmap,vmin=vmin,vmax=vmax,
                                 interpolation='nearest')
                ax.set_title(f'ant {bl[i]}',fontsize=14)
            r = rawNorm(raw.get_data((bl[0],bl[1],-5)))
#             c = calNorm(cal.get_data((bl[0],bl[1],-5)))
#             m = modelNorm(model.get_data((bl[0],bl[1],-5)))
#             norms = [convNorm,rawNorm,convNorm]
            dats = [c0,r,c1]
            for i in [0,1,2]:
                ax = fig.add_subplot(gs[b,i+1])
                axes[b,i+1] = ax
                if i==1 and rawNorm == np.angle:
                    vmin = -np.pi
                    vmax = np.pi
                    cmap = 'twilight'
                else:
                    vmin = np.nanpercentile(dats[i],100-percentile)
                    vmax = np.nanpercentile(dats[i],percentile)
                    cmap = 'viridis'
                im = ax.imshow(dats[i],aspect='auto',interpolation='nearest',vmin=vmin,vmax=vmax,cmap=cmap)
                if i==0:
                    plt.colorbar(im,ax=ax)
        
        for i in range(np.shape(axes)[0]):
            for j in range(np.shape(axes)[1]):
                ax = axes[i,j]
                ax.set_xticks(xticks)
                ax.set_xticklabels(xticklabels)
                a1 = bls[i][0]
                a2 = bls[i][1]
                plotnames = [f'ant {a1} gain - {gainNorm.__name__}',
                     f'ant {a1} conv level',
                     f'raw vis - {rawNorm.__name__}',
                     f'ant {a2} conv level',
                     f'ant {a2} gain - {gainNorm.__name__}']
                if j!=2:
                    ax.set_yticks(yticksFHD)
                    ax.set_yticklabels(yticklabelsFHD)
                else:
                    ax.set_yticks(yticksRaw)
                    ax.set_yticklabels(yticklabelsRaw)
                if i==(np.shape(axes)[0]-1):
                    ax.set_xlabel('Freq (MHz)')
                if j==0:
                    ax.set_ylabel(f'({a1}, {a2})\n LST',fontsize=16)
                if i==0:
                    ax.set_title(plotnames[j],fontsize=14)
        avgLen = int(np.mean(lengths))
        plt.suptitle(f'Avg Length this panel: {avgLen} meters',fontsize=18)
#         plt.subplots_adjust(top=0.65)
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        if savefig is True:
            if sortBy == 'blLength':
                outname = f'{outfig}_sortBy{sortBy}_{pol}_{str(avgLen).zfill(3)}m_{str(n).zfill(2)}.{file_ext}'
            else:
                outname = f'{outfig}_sortBy{sortBy}_{pol}_{str(n).zfill(2)}.{file_ext}'
            plt.savefig(outname,bbox_inches='tight')
            if write_params and n==0:
                curr_func = inspect.stack()[0][3]
                utils.write_params_to_text(outfig,args,curr_func=curr_func,curr_file=curr_file,githash=githash,dirlist=dirlist,
                                          raw_file=raw_file,cal_file=cal_file,model_file=model_file,
                                          gains=uvc,raw=raw,calibrated=cal,model=model)
            plt.close()
        else:
            plt.show()
            plt.close()
            
    return uvc, raw, cal, model


def plotVisAndGains(datadir, uvc = None, raw = None, cal = None, model = None,
                    gainNorm=np.abs,calNorm=np.abs,modelNorm=np.abs,rawNorm=np.abs,
                   savefig=False,outfig='',write_params=True,split_plots=True,nsplit=3,file_ext='pdf',
                   jd=2459855,readOnly=False,NblsPlot='all',sortBy='blLength',pol='XX',percentile=90):
    from pyuvdata import UVCal
    args = locals()
    
    if uvc is None:
        gf = sorted(glob.glob(f'{datadir}/{jd}_fhd_gains_{pol}*'))
        if len(gf)<3:
            dirlist = f'{datadir}/{jd}_fhd_dirlist.txt'
            calfiles, obsfiles, layoutfiles, settingsfiles = getFhdGainDirs(dirlist)
            uvc = UVCal()
            uvc.read_fhd_cal(calfiles, obsfiles,layoutfiles,settingsfiles)
            uvc.file_name = dirlist
        elif len(gf)==1:
            uvc = UVCal()
            uvc.read_calfits(gf[0])
        
    else:
        dirlist=None
    if raw is None:
        raw = UVData()
        raw_file = f'{datadir}/{jd}_fhd_raw_data_{pol}.uvfits'
        raw.read(raw_file)
        raw.file_name = raw_file
    else:
        raw_file=None
    if cal is None:
        cal = UVData()
        cal_file = f'{datadir}/{jd}_fhd_calibrated_data_{pol}.uvfits'
        cal.read(cal_file)
        cal.file_name = cal_file
    else:
        cal_file=None
    if model is None:
        model = UVData()
        model_file = f'{datadir}/{jd}_fhd_model_data_{pol}.uvfits'
        model.read(model_file)
        model.file_name=model_file
    else:
        model_file=None
    
    times = np.unique(cal.time_array)
    raw.select(times=times)
    
    freqs = raw.freq_array[0]*1e-6
    lstsRaw = raw.lst_array * 3.819719
    lstsRaw = np.reshape(lstsRaw,(raw.Ntimes,-1))[:,0]
    lstsFHD = uvc.lst_array * 3.819719
    lstsFHD = np.reshape(lstsFHD,(uvc.Ntimes,-1))[:,0]
    
    xticks = [int(i) for i in np.linspace(0, len(freqs) - 1, 5)]
    xticklabels = [int(freqs[x]) for x in xticks]
    yticksRaw = [int(i) for i in np.linspace(0, len(lstsRaw) - 1, 6)]
    yticklabelsRaw = [np.around(lstsRaw[ytick], 1) for ytick in yticksRaw]
    yticksFHD = [int(i) for i in np.linspace(0, len(lstsFHD) - 1, 6)]
    yticklabelsFHD = [np.around(lstsFHD[ytick], 1) for ytick in yticksFHD]
    
    allbls = np.asarray(np.load(f'{datadir}/{jd}_bl_array.npy'))  
    alllengths = np.asarray(getBaselineLength(raw,allbls))
    if sortBy == 'blLength':
        leninds = alllengths.argsort()
        alllengths = alllengths[leninds]
        allbls = allbls[leninds]
    if sortBy == 'blLengthRev':
        leninds = alllengths.argsort()
        alllengths = alllengths[leninds]
        allbls = allbls[leninds]
        allbls = np.flip(allbls)
    if NblsPlot != 'all':
        allbls = allbls[0:NblsPlot] 
    nbls = len(allbls)
    nplots = nbls//nsplit+1
    for n in range(nplots):
        if readOnly:
            continue
        bls = allbls[n*nsplit:n*nsplit+nsplit]
        axes = np.empty((len(bls),5),dtype=object)
        fig = plt.figure(figsize=(16,4*nsplit))
        gs = fig.add_gridspec(nsplit,5,width_ratios=[1,1,1,1,1])
        lengths = getBaselineLength(raw,bls)
        maxlength = np.max(lengths)
        for b,bl in enumerate(bls):
            if bl[0]==183 or bl[1]==183 or bl[0]==5 or bl[1]==5:
                continue
            g0 = uvc.get_gains(bl[0])
            g1 = uvc.get_gains(bl[1])
            g0 = np.transpose(gainNorm(g0[:,:,0]))
            g1 = np.transpose(gainNorm(g1[:,:,0]))
            gains = [g0,g1]
            axinds = [0,4]
            for i in [0,1]:
                if gainNorm == np.angle:
                    vmin = -np.pi
                    vmax = np.pi
                    cmap = 'twilight'
                else:
                    vmin = np.percentile(gains[i],100-percentile)
                    vmax = np.percentile(gains[i],percentile)
                    cmap = 'viridis'
                ax = fig.add_subplot(gs[b,axinds[i]])
                axes[b,axinds[i]] = ax
                im = ax.imshow(gains[i],aspect='auto',cmap=cmap,vmin=vmin,vmax=vmax,
                                 interpolation='nearest')
                ax.set_title(f'ant {bl[i]}',fontsize=14)
            r = rawNorm(raw.get_data((bl[0],bl[1],-5)))
            c = calNorm(cal.get_data((bl[0],bl[1],-5)))
            m = modelNorm(model.get_data((bl[0],bl[1],-5)))
            norms = [rawNorm,calNorm,modelNorm]
            dats = [r,c,m]
            for i in [0,1,2]:
                ax = fig.add_subplot(gs[b,i+1])
                axes[b,i+1] = ax
                norm = norms[i]
                if norm == np.angle:
                    vmin = -np.pi
                    vmax = np.pi
                    cmap = 'twilight'
                else:
                    vmin = np.percentile(dats[i],100-percentile)
                    vmax = np.percentile(dats[i],percentile)
                    cmap = 'viridis'
                im = ax.imshow(dats[i],aspect='auto',interpolation='nearest',vmin=vmin,vmax=vmax,cmap=cmap)
        
        for i in range(np.shape(axes)[0]):
            for j in range(np.shape(axes)[1]):
                ax = axes[i,j]
                ax.set_xticks(xticks)
                ax.set_xticklabels(xticklabels)
                a1 = bls[i][0]
                a2 = bls[i][1]
                plotnames = [f'ant {a1} gain - {gainNorm.__name__}',
                     f'raw vis - {rawNorm.__name__}',
                     f'cal vis - {calNorm.__name__}',
                     f'model vis - {modelNorm.__name__}',
                     f'ant {a2} gain - {gainNorm.__name__}']
                if j==0 or j==4:
                    ax.set_yticks(yticksFHD)
                    ax.set_yticklabels(yticklabelsFHD)
                else:
                    ax.set_yticks(yticksRaw)
                    ax.set_yticklabels(yticklabelsRaw)
                if i==(np.shape(axes)[0]-1):
                    ax.set_xlabel('Freq (MHz)')
                if j==0:
                    ax.set_ylabel(f'({a1}, {a2})\n LST',fontsize=16)
                if i==0:
                    ax.set_title(plotnames[j],fontsize=14)
        avgLen = int(np.mean(lengths))
        plt.suptitle(f'Avg Length this panel: {avgLen} meters',fontsize=18)
#         plt.subplots_adjust(top=0.65)
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        if savefig is True:
            if sortBy == 'blLength':
                outname = f'{outfig}_sortBy{sortBy}_{pol}_{str(avgLen).zfill(3)}m_{str(n).zfill(2)}.{file_ext}'
            else:
                outname = f'{outfig}_sortBy{sortBy}_{pol}_{str(n).zfill(2)}.{file_ext}'
            plt.savefig(outname,bbox_inches='tight')
            if write_params and n==0:
                curr_func = inspect.stack()[0][3]
                utils.write_params_to_text(outfig,args,curr_func=curr_func,curr_file=curr_file,githash=githash,dirlist=dirlist,
                                          raw_file=raw_file,cal_file=cal_file,model_file=model_file,
                                          gains=uvc,raw=raw,calibrated=cal,model=model)
            plt.close()
        else:
            plt.show()
            plt.close()
            
    return uvc, raw, cal, model

def plotCalVisAllBls(uv,datdir,jd=2459855,tmin=0,tmax=600,jd_label=False,savefig=False,outfig='',write_params=True,
                    split_plots=True,nsplit=3,file_ext='pdf',norm='abs'):
    args = locals()
    model = np.load(f'{datdir}/{jd}_fhd_model_data.npy')
    cal = np.load(f'{datdir}/{jd}_fhd_calibrated_data.npy')
    raw = np.load(f'{datdir}/{jd}_fhd_raw_data.npy')
    f = UVFlag()
    f.read(f'{datdir}/{jd}_ssins_flags.hdf5')
    flags = f.flag_array
    allbls = np.load(f'{datdir}/{jd}_bl_array.npy')
    jds = np.load(f'{datdir}/{jd}_jd_array.npy')
    lsts = np.load(f'{datdir}/{jd}_lst_array.npy')
    freqs = uv.freq_array[0]*1e-6

    nbls = len(allbls)
    nplots = nbls//nsplit+1
    for n in range(nplots):
        bls = allbls[n*nsplit:n*nsplit+nsplit]
        fig = plt.figure(figsize=(16,6*nsplit))
        gs = fig.add_gridspec(nsplit,4,width_ratios=[1,1,1,0.1])
        lengths = getBaselineLength(uv,bls)
        maxlength = np.max(lengths)
        for i,bl in enumerate(bls):
            ant1 = bl[0]
            ant2 = bl[1]
            ind = get_ind(allbls,ant1,ant2)
            length = lengths[i]

            xticks = [int(i) for i in np.linspace(0, len(freqs) - 1, 5)]
            xticklabels = [int(freqs[x]) for x in xticks]
            yticks = [int(i) for i in np.linspace(0, len(lsts[tmin:tmax]) - 1, 6)]
            yticklabels1 = [np.around(lsts[tmin:tmax][ytick], 1) for ytick in yticks]
            yticklabels2 = [str(jds[tmin:tmax][ytick])[4:11] for ytick in yticks]
            mask = np.ma.masked_where(flags==False, flags)[tmin:tmax,:,0]
            if norm == 'abs':
                m = np.abs(model[tmin:tmax,ind,:])
                c = np.abs(cal[tmin:tmax,ind,:])
                r = np.abs(raw[tmin:tmax,ind,:])
            elif norm == 'real':
                m = np.real(model[tmin:tmax,ind,:])
                c = np.real(cal[tmin:tmax,ind,:])
                r = np.real(raw[tmin:tmax,ind,:])
            elif norm == 'imag':
                m = np.imag(model[tmin:tmax,ind,:])
                c = np.imag(cal[tmin:tmax,ind,:])
                r = np.imag(raw[tmin:tmax,ind,:])
            elif norm == 'phase':
                m = np.angle(model[tmin:tmax,ind,:])
                c = np.angle(cal[tmin:tmax,ind,:])
                r = np.angle(raw[tmin:tmax,ind,:])
            else:
                print('ERROR! Options for norm parameter are abs, real, imag, or phase.')

            ax = fig.add_subplot(gs[i,0])
            if norm == 'phase':
                im = ax.imshow(m, 'twilight', vmin=-np.pi,vmax=np.pi, aspect='auto',interpolation='nearest')
            else:
                im = ax.imshow(m, vmin=np.percentile(c,10),vmax=np.percentile(c,90), aspect='auto',interpolation='nearest')
            im = ax.imshow(mask, 'binary', interpolation='none',alpha=1,aspect='auto')
            ax.set_title('Model Vis',fontsize=20)
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticklabels)
            ax.set_yticks(yticks)
            ax.set_yticklabels(yticklabels1,fontsize=12)
            ax.set_ylabel('LST',fontsize=15)

            ax = fig.add_subplot(gs[i,1])
            if norm == 'phase':
                im = ax.imshow(c, 'twilight', vmin=-np.pi,vmax=np.pi, aspect='auto',interpolation='nearest')
            else:
                im = ax.imshow(c, aspect='auto',vmin=np.percentile(c,10),vmax=np.percentile(c,90),interpolation='nearest')
            im = ax.imshow(mask, 'binary', interpolation='none',alpha=1,aspect='auto')
            ax.set_title('Calibrated Vis',fontsize=20)
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticklabels)
            ax.set_yticks([])

            ax = fig.add_subplot(gs[i,2])
            if norm == 'phase':
                im = ax.imshow(r, 'twilight', vmin=-np.pi,vmax=np.pi, aspect='auto',interpolation='nearest')
            else:
                im = ax.imshow(r, aspect='auto',vmin=np.percentile(r,10),vmax=np.percentile(r,90),interpolation='nearest')
            im = ax.imshow(mask, 'binary', interpolation='none',alpha=1,aspect='auto')
            ax.set_title('Raw Vis',fontsize=20)
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticklabels)
            if jd_label is True:
                ax.yaxis.tick_right()
                ax.set_yticks(yticks)
                ax.set_yticklabels(yticklabels2,fontsize=10)
            else:
                ax.set_yticks([])

            ax = fig.add_subplot(gs[i,3])
            frac = length/maxlength
            ax.fill_between([0,0.5],[length,length])
            ax.set_ylim(0,maxlength)
            ax.set_xlim(0,1)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position("right")
            ax.set_ylabel(f'({ant1}, {ant2})',rotation=270,fontsize=20)

        plt.tight_layout()
        if savefig is True:
            plt.savefig(f'{outfig}_{norm}_{n}.{file_ext}',bbox_inches='tight')
            if write_params and n==0:
                curr_func = inspect.stack()[0][3]
                utils.write_params_to_text(outfig,args,curr_func,curr_file,githash)
            plt.close()
        else:
            plt.show()
            plt.close()
            
def plotBlLengthHists(uv,use_ants=[],freq=170,bl_cut=25,nbins=10,savefig=False,outfig='',write_params=True,
                     title=''):
    """
    Plots a histogram of baseline lengths, and shows where a particular baseline cut or cuts is.
    
    Parameters:
    ----------
    uv: UVData
        UVData object to use for getting antenna set and baseline length information.
    use_ants: List
        Set of antennas to select on.
    freq: Int
        Frequency in MHz or HZ to calculate baseline cut at.
    bl_cut: Int or List
        Baseline cut or list of baseline cuts to make histograms for - should be in units of n lambda.
    nbins: Int
        Number of bins to use in histogram.
    savefig: Boolean
    outfig: String
    write_params: Boolean
    """
    import scipy
    args = locals()
    if len(use_ants)>0:
        uv.select(use_ants=use_ants)
    baseline_groups, vec_bin_centers, lengths = uv.get_redundancies(
        use_antpos=False, include_autos=False
    )
    antpairs, lengths = unpackBlLengths(uv, baseline_groups,lengths)
    if type(bl_cut)==int:
        ncuts=1
        bl_cut = [bl_cut]
    else:
        ncuts = len(bl_cut)
    if freq < 1000:
        freq = freq*1e6
    wl = scipy.constants.speed_of_light/freq
    bl_cut_m=np.asarray(bl_cut)*wl
    for i,cut in enumerate(bl_cut):
        print(f'Baseline cut of {cut} lambda is at {np.round(bl_cut_m[i],1)} meters at {int(freq*1e-6)}MHz')
        numOver = np.count_nonzero(lengths>=bl_cut_m[i])
        print(f'Data has {numOver} baselines above the {np.around(bl_cut_m[i],1)}m cut')
    
    fig = plt.figure(figsize=(8,6))
    plt.hist(lengths,bins=nbins,histtype='step')
    plt.xlabel('Baseline Length(m)')
    plt.ylabel('Count')
    plt.xlim((0,max(lengths)+20))
    colors=['black','red','cyan','green','gold']
    for i,cut in enumerate(bl_cut_m):
        plt.axvline(cut,linestyle='--',label=f'{np.round(bl_cut[i],1)} lambda cut',color=colors[i])
    plt.legend()
    plt.title(title)
    if savefig:
        plt.savefig(outfig)
        if write_params:
            curr_func = inspect.stack()[0][3]
            utils.write_params_to_text(outfig,args,curr_func,curr_file,githash)
            
def plotBlLengthHists_perAnt(uv,use_ants=[],freq=170,bl_cut=25,nbins=10,savefig=False,outfig='',
                             write_params=True,title='',xlim=None):
    """
    Plots a histogram of the number of baselines each antenna has that survive a certain baseline cut or cuts.
    
    Parameters:
    -----------
    uv: UVData
        UVData object to use for getting antenna set and baseline length information
    use_ants: List
        Set of antennas to select on.
    freq: Int
        Frequency in MHz or HZ to calculate baseline cut at
    bl_cut: Int or List
        Baseline cut or list of baseline cuts to make histograms for - should be in units of n lambda.
    nbins: Int
        Number of bins to use in histogram
    savefig: Boolean
    Outfig: String
    write_params: Boolean
    title: String
    """
    import scipy
    args = locals()
    if len(use_ants)>0:
        uv.select(use_ants=use_ants)
    baseline_groups, vec_bin_centers, lengths = uv.get_redundancies(
        use_antpos=False, include_autos=False
    )
    nants = len(uv.get_ants())
    if type(bl_cut)==int:
        ncuts=1
    else:
        ncuts=len(bl_cut)
    counts = np.zeros((nants,ncuts))
    antpairs, lengths = unpackBlLengths(uv, baseline_groups,lengths)
    if freq < 1000:
        freq = freq*1e6
    wl = scipy.constants.speed_of_light/freq
    bl_cut_m=np.asarray(bl_cut)*wl
    print(f'Baseline cut of {bl_cut} lambda is {np.round(bl_cut_m,1)} meters at {int(freq*1e-6)}MHz')
    for i,ant in enumerate(uv.get_ants()):
        bls = [(bl[0],bl[1]) for bl in antpairs if (bl[0]==ant or bl[1]==ant)]
        lens = np.asarray(getBaselineLength(uv,bls))
        if ncuts==1:
            c = np.count_nonzero(lens>=bl_cut_m)
            counts[i] = c
        else:
            for j,cut in enumerate(bl_cut_m):
                c = np.count_nonzero(lens>=bl_cut_m[j])
                counts[i,j] = c

    if ncuts == 1:
        fig = plt.figure(figsize=(8,6))
        plt.hist(counts,bins=nbins,histtype='step')
        plt.xlabel('# Baselines above cut per antenna')
        plt.ylabel('Count')
        plt.xlim((0,max(counts)+2))
    else:
        ymax = 0
        for n in range(ncuts):
            hist, edges = np.histogram(counts[:,n],bins=nbins)
            if np.max(hist) > ymax:
                ymax = np.max(hist)
        ymax = np.round(ymax*1.1,0)
        fig, ax = plt.subplots(1,ncuts,figsize=(10,4))
        for i,cut in enumerate(bl_cut_m):
            ax[i].hist(counts[:,i],bins=nbins,histtype='step')
#             ax[i].plot(hist)
            ax[i].set_xlabel('# Baselines above cut per antenna')
            ax[i].set_ylabel('Count')
            if xlim == None:
                ax[i].set_xlim((0,np.max(counts)+2))
            else:
                ax[i].set_xlim(xlim)
            ax[i].set_title(f'{np.round(cut,1)}m/{bl_cut[i]}lambda baseline cut')
            ax[i].set_ylim((0,ymax))
    plt.suptitle(title)
    plt.tight_layout()
#     plt.axvline(bl_cut_m,linestyle='--',color='k')
    if savefig:
        plt.savefig(outfig)
        if write_params:
            curr_func = inspect.stack()[0][3]
            utils.write_params_to_text(outfig,args,curr_func,curr_file,githash)
            
def plot_antenna_positions(uv, badAnts=[], flaggedAnts={}, use_ants="all",hexsize=35,inc_outriggers=False):
    """
    Plots the positions of all antennas that have data, colored by node.

    Parameters
    ----------
    uv: UVData object
        Observation to extract antenna numbers and positions from
    badAnts: List
        A list of flagged or bad antennas. These will be outlined in black in the plot.
    flaggedAnts: Dict
        A dict of antennas flagged by ant_metrics with value corresponding to color in ant_metrics plot
    use_ants: List or 'all'
        List of antennas to include, or set to 'all' to include all antennas.

    Returns:
    ----------
    None

    """
    from hera_mc import geo_sysdef

    plt.figure(figsize=(12, 10))
    nodes, antDict, inclNodes = utils.generate_nodeDict(uv)
    if use_ants == "all":
        use_ants = uv.get_ants()
    N = len(inclNodes)
    cmap = plt.get_cmap("tab20")
    i = 0
    ants = geo_sysdef.read_antennas()
    nodes = geo_sysdef.read_nodes()
    firstNode = True
    for n, info in nodes.items():
        firstAnt = True
        if n > 9:
            n = str(n)
        else:
            n = f"0{n}"
        if n in inclNodes:
            color = cmap(round(20 / N * i))
            color = 'blue'
            i += 1
            for a in info["ants"]:
                width = 0
                widthf = 0
                if a in badAnts:
                    width = 2
                if a in flaggedAnts.keys():
                    widthf = 6
                station = "HH{}".format(a)
                try:
                    this_ant = ants[station]
                except KeyError:
                    if inc_outriggers:
                        try:
                            station = "HA{}".format(a)
                            this_ant = ants[station]
                        except:
                            try:
                                station = "HB{}".format(a)
                                this_ant = ants[station]
                            except:
                                continue
                    else:
                        continue
                x = this_ant["E"]
                y = this_ant["N"]
                if a in use_ants:
                    falpha = 0.7
                else:
                    falpha = 0.1
                if firstAnt:
                    if a in badAnts or a in flaggedAnts.keys():
                        if falpha == 0.1:
                            plt.plot(
                                x,
                                y,
                                marker="h",
                                markersize=hexsize,
                                color=color,
                                alpha=falpha,
                                markeredgecolor="black",
                                markeredgewidth=0,
                            )
                            plt.annotate(a, [x - 2, y-1])
                            continue
                        plt.plot(
                            x,
                            y,
                            marker="h",
                            markersize=hexsize,
                            color=color,
                            alpha=falpha,
                            label=str(n),
                            markeredgecolor="black",
                            markeredgewidth=0,
                        )
                    else:
                        if falpha == 0.1:
                            plt.plot(
                                x,
                                y,
                                marker="h",
                                markersize=hexsize,
                                color=color,
                                alpha=falpha,
                                markeredgecolor="black",
                                markeredgewidth=0,
                            )
                            plt.annotate(a, [x - 2, y-1])
                            continue
                        plt.plot(
                            x,
                            y,
                            marker="h",
                            markersize=hexsize,
                            color=color,
                            alpha=falpha,
                            label=str(n),
                            markeredgecolor="black",
                            markeredgewidth=width,
                        )
                    firstAnt = False
                else:
                    plt.plot(
                        x,
                        y,
                        marker="h",
                        markersize=hexsize,
                        color=color,
                        alpha=falpha,
                        markeredgecolor="black",
                        markeredgewidth=0,
                    )
                    if a in flaggedAnts.keys() and a in use_ants:
                        plt.plot(
                            x,
                            y,
                            marker="h",
                            markersize=hexsize,
                            color=color,
                            markeredgecolor=flaggedAnts[a],
                            markeredgewidth=widthf,
                            markerfacecolor="None",
                        )
                    if a in badAnts and a in use_ants:
                        plt.plot(
                            x,
                            y,
                            marker="h",
                            markersize=hexsize,
                            color=color,
                            markeredgecolor="black",
                            markeredgewidth=width,
                            markerfacecolor="None",
                        )
                plt.annotate(a, [x - 2, y-1])
    plt.xlabel("East")
    plt.ylabel("North")
    plt.show()
    plt.close()
    
def get_ind(bls,ant1,ant2):
    ind = -1
    for i in range(len(bls)):
#         if ant1==41 or ant2==41:
#                 print(bls[i])
        if bls[i][0] in [ant1,ant2] and bls[i][1] in [ant1,ant2]:
            if bls[i][0] == bls[i][1]:
                if ant1==ant2:
                    ind = i
                else:
                    continue
            else:
                ind = i
    if ind == -1:
        raise Exception(f'Couldnt find baseline ({ant1}, {ant2})')
    return ind

def unpackBlLengths(uv, baseline_groups, lengths):
    antpairs = []
    lens = []
    for i in range(len(baseline_groups)):
        bls = baseline_groups[i]
        for bl in bls:
            apair = uv.baseline_to_antnums(bl)
            antpairs.append(apair)
            lens.append(lengths[i])
    return antpairs, lens

def getBaselineLength(uv,bls):
    baseline_groups, vec_bin_centers, lengths = uv.get_redundancies(
        use_antpos=False, include_autos=False
    )
    antpairs, lengths = unpackBlLengths(uv, baseline_groups,lengths)
    lens = []
    if type(bls) is list or type(bls) == np.ndarray:
        for bl in bls:
            ind = get_ind(antpairs,bl[0],bl[1])
            l = lengths[ind]
            lens.append(l)
    elif type(bls) is tuple:
        ind = get_ind(antpairs,bls[0],bls[1])
        lens = lengths[ind]
    return lens

def getFhdGainDirs(dirlist):
    import glob
    
    d = open(dirlist,'r')
    dirlist = d.read().split('\n')[0:-1]

    calfiles = []
    obsfiles = []
    layoutfiles = []
    settingsfiles = []
    for path in dirlist:
        cf = sorted(glob.glob(f'{path}calibration/*cal.sav'))
    #     print(cf)
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
    return calfiles, obsfiles,layoutfiles,settingsfiles
