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
    
def trimVisTimes(raw,cal,model,uvc,printTimes=False):
    if printTimes:
        print('Before select:')
        print(f'Raw has LSTS {raw.lst_array[0]* 3.819719} to {raw.lst_array[-1]* 3.819719}')
        if model != None:
            print(f'Model has LSTS {model.lst_array[0]* 3.819719} to {model.lst_array[-1]* 3.819719}')
        print(f'Cal has LSTS {cal.lst_array[0]* 3.819719} to {cal.lst_array[-1]* 3.819719}')
        print(f'Gains has LSTS {uvc.lst_array[0]* 3.819719} to {uvc.lst_array[-1]* 3.819719}')
    
    times = np.unique(cal.time_array)
    rawtimes = np.unique(raw.time_array)
    gaintimes = np.unique(uvc.time_array)
    
    mintime = np.max([times[0],rawtimes[0],gaintimes[0]])
    maxtime = np.min([times[-1],rawtimes[-1],gaintimes[-1]])
        
    minind = np.argmin(abs(np.subtract(times,mintime)))
    maxind = np.argmin(abs(np.subtract(times,maxtime)))
    times = times[minind:maxind]
    
    minind = np.argmin(abs(np.subtract(rawtimes,mintime)))
    maxind = np.argmin(abs(np.subtract(rawtimes,maxtime)))
    rawtimes = rawtimes[minind:maxind]
    
    minind = np.argmin(abs(np.subtract(gaintimes,mintime)))
    maxind = np.argmin(abs(np.subtract(gaintimes,maxtime)))
    gaintimes = gaintimes[minind:maxind]
    
    raw.select(times=times)
    if model != None:
        model.select(times=times)
    cal.select(times=times)
    uvc.select(times=gaintimes)
    if printTimes:
        print('After select:')
        print(f'Raw has LSTS {raw.lst_array[0]* 3.819719} to {raw.lst_array[-1]* 3.819719}')
        if model != None:
            print(f'Model has LSTS {model.lst_array[0]* 3.819719} to {model.lst_array[-1]* 3.819719}')
        print(f'Cal has LSTS {cal.lst_array[0]* 3.819719} to {cal.lst_array[-1]* 3.819719}')
        print(f'Gains has LSTS {uvc.lst_array[0]* 3.819719} to {uvc.lst_array[-1]* 3.819719}')
    return uvc, raw, cal, model

def removePhaseWrap(data,freqs):
    import scipy
    #Take average across times and do linear fit
    dslice = np.mean(data,axis=0)
    unwrap = np.unwrap(np.angle(dslice))
    res = scipy.stats.linregress(freqs,unwrap,alternative='less')
    sfit = res.intercept + res.slope*freqs
    #Subtract fit from all times
    whole_sub = np.subtract(np.angle(data),sfit)
    #Re-wrap phase
    whole_sub = ( whole_sub + np.pi) % (2 * np.pi ) - np.pi
    #Re-combine real and imag parts
    comb = np.cos(whole_sub) + 1j*np.sin(whole_sub)
    data = np.multiply(abs(data),comb)
    return data

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
               nframes='all',file_ext='jpeg',outsuffix='',write_params=True,plot_range='all',
               reverse_ra=False,fontsize=16,plotBeam=False,beam_color_scale=[0,0.005],
               beamGradient=False,fieldType='streamplot',**kwargs):
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
                                 ra_range=ra_range,dec_range=_dec_range,title='Calibrated',fontsize=fontsize)
        data = plot_fits.load_image(model_files[ind])
        im = plot_fits.plot_fits_image(data, axes[0][1], color_scale, output_path, prefix, write_pixel_coordinates, log_scale,
                                 ra_range=ra_range,dec_range=_dec_range,title='Model',fontsize=fontsize)
        data = plot_fits.load_image(residual_files[ind])
        vmin = np.percentile(data.signal_arr,5)
        vmax = np.percentile(data.signal_arr,95)
        im = plot_fits.plot_fits_image(data, axes[1][0], [vmin,vmax], output_path, prefix, write_pixel_coordinates, log_scale,
                                 ra_range=ra_range,dec_range=_dec_range,title='Residual',fontsize=fontsize)
        sources = plot_fits.gather_source_list(inc_flux=True)
        ax = axes[1][1]
        if plotBeam:
            beamFile = beam_files[ind]
            data = plot_fits.load_image(beamFile)
            if beamGradient:
                if fieldType == 'streamplot':
                    im = ax.imshow(np.ones(np.shape(data.signal_arr))*1e10,vmin=0,vmax=1,cmap='Greys_r',
                                  extent=[xlim[0],xlim[1],ylim[0],ylim[1]])
                    signal_arr = np.gradient(data.signal_arr)
                    x,y = np.meshgrid(data.ra_axis,data.dec_axis)
                    ax.streamplot(x,y,signal_arr[0],signal_arr[1])
                    cbar = plt.colorbar(im,ax=ax,pad=0.0)
                    cbar.set_ticks([])
                    cbar.outline.set_visible(False)
                    ax.set_xlim(xlim)
                    ax.set_ylim(ylim)
                    ax.axis('equal')
                    ax.grid(which='both', zorder=10, lw=0.5)
                    ax.margins(0)
                elif fieldType == 'quiver':
                    im = ax.imshow(np.ones(np.shape(data.signal_arr))*1e10,vmin=0,vmax=1,cmap='Greys_r',
                                  extent=[xlim[0],xlim[1],ylim[0],ylim[1]])
                    signal_arr = np.gradient(data.signal_arr)
                    x,y = np.meshgrid(data.ra_axis,data.dec_axis)
                    ax.quiver(x,y,signal_arr[0],signal_arr[1],scale=100)
                    cbar = plt.colorbar(im,ax=ax,pad=0.0)
                    cbar.set_ticks([])
                    cbar.outline.set_visible(False)
                    ax.set_xlim(xlim)
                    ax.set_ylim(ylim)
                    ax.axis('equal')
                    ax.grid(which='both', zorder=10, lw=0.5)
                    ax.margins(0)
            else:
                im = plot_fits.plot_fits_image(data, ax, beam_color_scale, output_path, prefix, write_pixel_coordinates, log_scale,
                                         ra_range=ra_range,dec_range=dec_range,title='Beam',fontsize=fontsize)
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            for s in sources:
                if s[1] > ylim[0] and s[1] < ylim[1]:
                    if s[0] > 180:
                        s = (s[0]-360,s[1],s[2],s[3])
                    if s[0] > xlim[0] and s[0] < xlim[1]:
                        if s[2] == 'LMC' or s[2] == 'SMC':
                            ax.annotate(s[2],xy=(s[0],s[1]),xycoords='data',fontsize=8,xytext=(20,-20),
                                         textcoords='offset points',arrowprops=dict(facecolor='red', shrink=2,width=1,
                                                                                    headwidth=4))
                        else:
                            ax.scatter(s[0],s[1],c='r',s=s[3])
                            if len(s[2]) > 0:
                                if reverse_ra:
                                    ax.annotate(s[2],xy=(s[0]-3,s[1]-4),xycoords='data',fontsize=6)
                                else:
                                    ax.annotate(s[2],xy=(s[0]+3,s[1]-4),xycoords='data',fontsize=6)
        else:
            im = plot_fits.plot_sky_map(uv,ax,dec_pad=55,ra_pad=55,clip=False,sources=sources,fontsize=fontsize,
                                       fwhm=ra_range,reverse_ra=reverse_ra)
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

def plotBeam(fhd_dir,pol='XX',color_scale=[-1763758,1972024],output_path='',prefix='beam',
             write_pixel_coordinates=False,log_scale=False,ra_range=40,dec_range=40,fontsize=16):
    from djs_fhd_pipeline import plot_fits, plot_deconvolution
    beamFile = glob.glob(f'{fhd_dir}/output_data/*_Beam_{pol}.fits')[0]
    data = plot_fits.load_image(beamFile)
    fig, ax = plt.subplots(1,1,figsize=(12,12))
#     _dec_range = [pos[1]-dec_range/2,pos[1]+dec_range/2]
    im = plot_fits.plot_fits_image(data, ax, color_scale, output_path, prefix, write_pixel_coordinates, log_scale,
                                 ra_range=ra_range,dec_range=dec_range,title='Beam',fontsize=fontsize)
    sources = plot_fits.gather_source_list()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    print(xlim)
    for s in sources:
        if s[1] > ylim[0] and s[1] < ylim[1]:
            if s[0] > xlim[0] and s[0] < xlim[1]:
                if s[0] > 180:
                    s = (s[0]-360,s[1],s[2])
                if s[2] == 'LMC' or s[2] == 'SMC':
                    ax.annotate(s[2],xy=(s[0],s[1]),xycoords='data',fontsize=8,xytext=(20,-20),
                                 textcoords='offset points',arrowprops=dict(facecolor='red', shrink=2,width=1,
                                                                            headwidth=4))
                else:
                    ax.scatter(s[0],s[1],c='r',s=10)
                    if len(s[2]) > 0:
                        if reverse_ra:
                            ax.annotate(s[2],xy=(s[0]-3,s[1]-4),xycoords='data',fontsize=6)
                        else:
                            ax.annotate(s[2],xy=(s[0]+3,s[1]-4),xycoords='data',fontsize=6)
        
def make_frames_withVis(datadir,fhd_path,uvfits_path,outdir,jd,pol='XX',savefig=True,ra_range=9,dec_range=9,
               nframes='all',file_ext='jpeg',outsuffix='',write_params=True,plot_range='all',
               bl=(62,333),gainNorm=np.abs,calNorm=np.abs,modelNorm=np.abs,rawNorm=np.abs,
               raw=None,uvc=None,cal=None,model=None,readVis=True,percentile=90,fontsize=16,
               visPlotType='waterfall',reverse_ra=True,**kwargs):
    # To compile frames into movie after this script runs, on terminal execute: 
    # ffmpeg 2459911_wholeImage_XX.mp4 -r 5 -i 2459911_frame_XX_wholeImage_%3d.jpeg
    from astropy.coordinates import Galactic
    from astropy.coordinates import EarthLocation, AltAz, Angle
    from astropy.coordinates import SkyCoord as sc
    from astropy import units as u
    from astropy.time import Time
    import warnings
    warnings.filterwarnings("ignore", message="Telescope location derived from obs lat/lon/alt values does not match the location in the layout file. Using the value from known_telescopes.")
    
#     print(f'{fhd_path}/fhd_*/output_data/*Dirty_{pol}.fits')
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
    
    if readVis:
        print('Reading visibilities')
        uvc, raw, cal, model = plotVisAndGains(datadir,readOnly=True,jd=2459906, pol=pol)
        print('Finished reading visibilities')
    if cal is None or raw is None or model is None or uvc is None:
        print('ERROR: Must either supply all of uvc, raw, cal, and model UVData objects, or set readVis=True.')
    uvc.select(antenna_nums=[bl[0],bl[1]])
    raw.select(bls=[bl])
    cal.select(bls=[bl])
    model.select(bls=[bl])
    
    args = locals()
    
    print(f'Creating images for {len(dirty_files)} files')
    if nframes=='all':
        nframes = len(dirty_files)
    if plot_range=='all':
        plot_range = [0,nframes]
    if type(rawNorm)==list:
        Nnorms=2
    else:
        Nnorms=1
        rawNorm = [rawNorm]
        gainNorm = [gainNorm]
        calNorm = [calNorm]
        modelNorm = [modelNorm]
    for ind in range(plot_range[0],plot_range[1]):
        if ind%20==0:
            print(str(ind).zfill(3))
        fhd_files = [layout_files[ind],obs_files[ind],params_files[ind],dirty_vis_files[ind],flag_files[ind],settings_files[ind]]
        uv = UVData()
        uv.read_fhd(fhd_files,read_data=False)
        lst = uv.lst_array[0] * 3.819719 
        pos = [uv.phase_center_app_ra*57.2958,-31]
        _dec_range = [pos[1]-dec_range/2,pos[1]+dec_range/2]
        prefix = f'{fhd_path}/output_data'
        color_scale=[-1763758,1972024]
        output_path = ''
        write_pixel_coordinates=False
        log_scale=False

        fig = plt.figure(figsize=(20,24))
        if Nnorms==1:
            height_ratios=[1,1,0.7]
        else:
            height_ratios=[1,1,0.7,0.7]
        gs = fig.add_gridspec(2+Nnorms,5,wspace=0.2,hspace=0.3,width_ratios=[1,1,1,1,0.08],
                             height_ratios=height_ratios)
        data = plot_fits.load_image(dirty_files[ind])
        
        ax = fig.add_subplot(gs[0, 0:2])
        im = plot_fits.plot_fits_image(data, ax, color_scale, output_path, prefix, write_pixel_coordinates, log_scale,
                                 ra_range=ra_range,dec_range=_dec_range,title='Calibrated',fontsize=fontsize)
        data = plot_fits.load_image(model_files[ind])
        ax = fig.add_subplot(gs[0, 2:4])
        im = plot_fits.plot_fits_image(data, ax, color_scale, output_path, prefix, write_pixel_coordinates, log_scale,
                                 ra_range=ra_range,dec_range=_dec_range,title='Model',fontsize=fontsize)
        data = plot_fits.load_image(residual_files[ind])
        vmin = np.percentile(data.signal_arr,5)
        vmax = np.percentile(data.signal_arr,95)
        ax = fig.add_subplot(gs[1, 0:2])
        im = plot_fits.plot_fits_image(data, ax, [vmin,vmax], output_path, prefix, write_pixel_coordinates, log_scale,
                                 ra_range=ra_range,dec_range=_dec_range,title='Residual',fontsize=fontsize)
        sources = plot_fits.gather_source_list()
        ax = fig.add_subplot(gs[1,2:4])
        im = plot_fits.plot_sky_map(uv,ax,dec_pad=55,ra_pad=55,clip=False,sources=sources,fontsize=fontsize,
                                   fwhm=ra_range,reverse_ra=reverse_ra)
        
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
        
        for n in range(Nnorms):
            g0 = uvc.get_gains(bl[0])
            g1 = uvc.get_gains(bl[1])
            g0 = np.transpose((g0[:,:,0]))
            g1 = np.transpose((g1[:,:,0]))
            g = gainNorm[n](np.multiply(g0,np.conj(g1)))
            r = rawNorm[n](raw.get_data((bl[0],bl[1],pol)))
            c = calNorm[n](cal.get_data((bl[0],bl[1],pol)))
            m = modelNorm[n](model.get_data((bl[0],bl[1],pol)))
#             if ind==0:
#                 print(f'Inds: {range(plot_range[0],plot_range[1])}')
#                 print(f'Raw: {np.shape(r)}')
#                 print(f'Cal: {np.shape(c)}')
#                 print(f'Model: {np.shape(m)}')
#                 print(f'Gains: {np.shape(g)}')

            hline_r = np.argmin(abs(np.subtract(lstsRaw,lst)))
            hline_g = np.argmin(abs(np.subtract(lstsFHD,lst)))

            norms = [gainNorm[n],rawNorm[n],calNorm[n],modelNorm[n]]
            dats = [g,r,c,m]
            titles = [f'g{bl[0]} x g{bl[1]}*','Raw','Calibrated','Model']
            hlines = [hline_g,hline_r,hline_r,hline_r]
            ylims = [(0,12000),(0,120000),(0,18),(0,8)]
            for i in [0,1,2,3]:
                ax = fig.add_subplot(gs[2+n,i])
                norm = norms[i]
                if visPlotType=='waterfall':
                    if norm == np.angle:
                        vmin = -np.pi
                        vmax = np.pi
                        cmap = 'twilight'
                    else:
                        vmin = np.percentile(dats[i],100-percentile)
                        vmax = np.percentile(dats[i],percentile)
                        cmap = 'viridis'
                    im = ax.imshow(dats[i],aspect='auto',interpolation='nearest',vmin=vmin,vmax=vmax,cmap=cmap)
                    ax.set_title(titles[i],fontsize=fontsize)
                    if i==0:
                        ax.set_ylabel(f'Time (LST) \n {bl}: {gainNorm[n].__name__}',fontsize=fontsize)
                        ax.set_yticks(yticksFHD)
                        ax.set_yticklabels(yticklabelsFHD)
                    else:
                        ax.set_yticks([])
                        ax.set_yticklabels([])
                    if i==3:
                        cax = fig.add_subplot(gs[2+n,4])
                        fig.colorbar(im,cax=cax)
                    ax.axhline(hlines[i],color='white',linewidth=2)
                elif visPlotType=='line':
                    ax.plot(dats[i][hlines[i],:])
                    if norm==np.angle:
                        ax.set_ylim((-np.pi,np.pi))
                    elif norm==np.abs:
                        ax.set_ylim(ylims[i])
                    if i==0:
                        ax.set_ylabel(f'{bl}: {gainNorm[n].__name__}',fontsize=fontsize)
                else:
                    print('ERROR: visPlotType must be either waterfall or line')
                ax.set_xticks(xticks)
                ax.set_xticklabels(xticklabels)
                ax.set_xlabel('Freq (MHz)',fontsize=fontsize)
        
        
        ax = plt.gca()
        loc = EarthLocation.from_geocentric(*uv.telescope_location, unit='m')
        obstime = Time(uv.time_array[len(uv.time_array)//2],format='jd',location=loc)
        zenith = sc(Angle(0, unit='deg'),Angle(90,unit='deg'),frame='altaz',obstime=obstime,location=loc)
        zenith = zenith.transform_to('icrs')
        ra = zenith.ra.degree
        plt.suptitle(f'LST {np.around(lst,2)}          RA {np.around(ra,2)}$^\circ$',fontsize=30,y=0.92)
        
        if savefig == True:
            if len(outsuffix) > 0:
                outfig = f'{outdir}/{jd}_frame_{pol}_{outsuffix}_{str(ind).zfill(3)}.{file_ext}'
            else:
                outfig = f'{outdir}/{jd}_frame_{pol}_{str(ind).zfill(3)}.{file_ext}'
            plt.savefig(outfig,facecolor='white')
            if write_params and ind==0:
                utils.write_params_to_text(outfig,args,curr_func='make_frames',
                       curr_file=curr_file,**kwargs)
        else:
            plt.show()
        plt.close()
    return uvc, raw, cal, model



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

def plotDelayOfGains(uvc,baseline=(124,35),pol='XX',phaseWrapRemoval=True,savefig=False,write_params=True,outfig='',
                    ds_lim=1500,gsliceInd=20,**kwargs):
    from scipy import optimize
    import scipy
    args = locals()
    freqs = uvc.freq_array[0]*1e-6
    taus = np.fft.fftshift(np.fft.fftfreq(freqs.size, np.diff(freqs)[0]*1e6))*1e9
    inds = np.arange(0,len(freqs))
    g1 = np.transpose(uvc.get_gains(baseline[0],pol))
    g2 = np.transpose(uvc.get_gains(baseline[1],pol))
    gplot = np.multiply(g2,np.conj(g1))
    freqs *=1e6
    
    lsts = uvc.lst_range * 3.819719
    lsts = np.reshape(lsts,(uvc.Ntimes,-1))[:,0]
    yticks = [int(i) for i in np.linspace(0, len(lsts) - 1, 6)]
    yticklabels = [np.around(lsts[ytick], 1) for ytick in yticks]

    if phaseWrapRemoval:
        g1 = removePhaseWrap(g1,freqs)
        g2 = removePhaseWrap(g2,freqs)

    gplot = np.multiply(g2,np.conj(g1))
    gslice = gplot[gsliceInd,:]

    ds1 = np.fft.fftshift(np.fft.ifft(g1,axis=1))
    ds2 = np.fft.fftshift(np.fft.ifft(g2,axis=1))
    tind1 = np.argmin(abs(np.subtract(taus,-ds_lim)))
    tind2 = np.argmin(abs(np.subtract(taus,ds_lim)))
    if (tind2-tind1)%2==0:
        tind2+=1
    taus = taus[tind1:tind2]
    ds1 = ds1[:,tind1:tind2]
    ds2 = ds2[:,tind1:tind2]
    tticks = [int(i) for i in np.linspace(0, len(taus) - 1, 5)]
    tticklabels = [int(taus[x]) for x in tticks]

    fig, ax = plt.subplots(1,2,figsize=(16,8))
    im = ax[0].imshow(np.angle(ds1),vmin=-np.pi,vmax=np.pi,cmap='twilight',aspect='auto',interpolation='nearest')
    plt.colorbar(im,ax=ax[0])
    ax[0].set_xticks(tticks)
    ax[0].set_xticklabels(tticklabels)
    ax[0].set_title(f'Ant {baseline[0]} gain')
    ax[0].set_yticks(yticks)
    ax[0].set_yticklabels(yticklabels)
    im = ax[1].imshow(np.angle(ds2),vmin=-np.pi,vmax=np.pi,cmap='twilight',aspect='auto',interpolation='nearest')
    plt.colorbar(im,ax=ax[1])
    ax[1].set_xticks(tticks)
    ax[1].set_xticklabels(tticklabels)
    ax[1].set_yticks([])
    ax[1].set_title(f'Ant {baseline[1]} gain')
    for a in ax:
        a.set_xlabel('Delay (ns)')
        
    if savefig == True:
        plt.savefig(outfig,facecolor='white')
        if write_params:
            utils.write_params_to_text(outfig,args,curr_func='plotDelayOfGains',
                   curr_file=curr_file,**kwargs)
    else:
        plt.show()
    plt.close()
    
def plotDelayOfGainsAndAutos(uvc,uvd,baseline=(124,35),pol='XX',phaseWrapRemoval=True,savefig=False,write_params=True,
                             outfig='',ds_lim=1500,gsliceInd=20,**kwargs):
    from scipy import optimize
    import scipy
    args = locals()
    freqs = uvc.freq_array[0]*1e-6
    taus = np.fft.fftshift(np.fft.fftfreq(freqs.size, np.diff(freqs)[0]*1e6))*1e9
    inds = np.arange(0,len(freqs))
    g1 = np.transpose(uvc.get_gains(baseline[0],pol))
    g2 = np.transpose(uvc.get_gains(baseline[1],pol))
    a1 = uvd.get_data(baseline[0],baseline[0],pol)
    a2 = uvd.get_data(baseline[1],baseline[1],pol)
    freqs *=1e6
    
    lsts = uvc.lst_range * 3.819719
    lsts = np.reshape(lsts,(uvc.Ntimes,-1))[:,0]
    yticks = [int(i) for i in np.linspace(0, len(lsts) - 1, 6)]
    yticklabels = [np.around(lsts[ytick], 1) for ytick in yticks]
    lstsA = uvd.lst_array * 3.819719
    lstsA = np.reshape(lstsA,(uvd.Ntimes,-1))[:,0]
    yticksA = [int(i) for i in np.linspace(0, len(lstsA) - 1, 6)]
    yticklabelsA = [np.around(lstsA[ytick], 1) for ytick in yticksA]

    if phaseWrapRemoval:
        g1 = removePhaseWrap(g1,freqs)
        g2 = removePhaseWrap(g2,freqs)

    gplot = np.multiply(g2,np.conj(g1))
    gslice = gplot[gsliceInd,:]

    dsG1 = np.fft.fftshift(np.fft.ifft(g1,axis=1))
    dsG2 = np.fft.fftshift(np.fft.ifft(g2,axis=1))
    dsGp = np.fft.fftshift(np.fft.ifft(gplot,axis=1))
    dsA1 = np.fft.fftshift(np.fft.ifft(a1,axis=1))
    dsA2 = np.fft.fftshift(np.fft.ifft(a2,axis=1))
    tind1 = np.argmin(abs(np.subtract(taus,-ds_lim)))
    tind2 = np.argmin(abs(np.subtract(taus,ds_lim)))
    if (tind2-tind1)%2==0:
        tind2+=1
    taus = taus[tind1:tind2]
    dsG1 = dsG1[:,tind1:tind2]
    dsG2 = dsG2[:,tind1:tind2]
    dsA1 = dsA1[:,tind1:tind2]
    dsA2 = dsA2[:,tind1:tind2]
    dsgp = dsgp[:,tind1:tind2]
    tticks = [int(i) for i in np.linspace(0, len(taus) - 1, 5)]
    tticklabels = [int(taus[x]) for x in tticks]

    fig, ax = plt.subplots(1,5,figsize=(16,8))
    im = ax[0].imshow(np.angle(dsA1),vmin=-np.pi,vmax=np.pi,cmap='twilight',aspect='auto',interpolation='nearest')
    plt.colorbar(im,ax=ax[0])
    ax[0].set_xticks(tticks)
    ax[0].set_xticklabels(tticklabels)
    ax[0].set_title(f'Ant {baseline[0]} Auto')
    ax[0].set_yticks(yticks)
    ax[0].set_yticklabels(yticklabels)
    im = ax[1].imshow(np.angle(dsG1),vmin=-np.pi,vmax=np.pi,cmap='twilight',aspect='auto',interpolation='nearest')
    plt.colorbar(im,ax=ax[1])
    ax[1].set_xticks(tticks)
    ax[1].set_xticklabels(tticklabels)
    ax[1].set_title(f'Ant {baseline[0]} Gain')
    ax[1].set_yticks(yticks)
    ax[1].set_yticklabels(yticklabels)
    im = ax[2].imshow(np.angle(dsgp),vmin=-np.pi,vmax=np.pi,cmap='twilight',aspect='auto',interpolation='nearest')
    plt.colorbar(im,ax=ax[2])
    ax[2].set_xticks(tticks)
    ax[2].set_xticklabels(tticklabels)
    ax[2].set_yticks([])
    ax[2].set_title(f'g{baseline[0]} x g{baseline[1]}*')
    im = ax[3].imshow(np.angle(dsG2),vmin=-np.pi,vmax=np.pi,cmap='twilight',aspect='auto',interpolation='nearest')
    plt.colorbar(im,ax=ax[3])
    ax[3].set_xticks(tticks)
    ax[3].set_xticklabels(tticklabels)
    ax[3].set_title(f'Ant {baseline[1]} Gain')
    ax[3].set_yticks(yticks)
    ax[3].set_yticklabels(yticklabels)
    im = ax[4].imshow(np.angle(dsA2),vmin=-np.pi,vmax=np.pi,cmap='twilight',aspect='auto',interpolation='nearest')
    plt.colorbar(im,ax=ax[4])
    ax[4].set_xticks(tticks)
    ax[4].set_xticklabels(tticklabels)
    ax[4].set_title(f'Ant {baseline[1]} Auto')
    ax[4].set_yticks(yticks)
    ax[4].set_yticklabels(yticklabels)
    
    for a in ax:
        a.set_xlabel('Delay (ns)')
        
    if savefig == True:
        plt.savefig(outfig,facecolor='white')
        if write_params:
            utils.write_params_to_text(outfig,args,curr_func='plotDelayOfGainsAndAutos',
                   curr_file=curr_file,**kwargs)
    else:
        plt.show()
    plt.close()
    
def plotGainComponentsByTime(uvc,baseline=(124,35),pol='XX',phaseWrapRemoval=True,savefig=False,write_params=True,outfig='',
                    ds_lim=1500,gsliceType='singleSlice',gsliceInd=20,**kwargs):
    from scipy import optimize
    import scipy
    args = locals()
    freqs = uvc.freq_array[0]*1e-6
    taus = np.fft.fftshift(np.fft.fftfreq(freqs.size, np.diff(freqs)[0]*1e6))*1e9
    inds = np.arange(0,len(freqs))
    g1 = np.transpose(uvc.get_gains(baseline[0],pol))
    g2 = np.transpose(uvc.get_gains(baseline[1],pol))
    gplot = np.multiply(g2,np.conj(g1))
    freqs *=1e6
    
    lsts = uvc.lst_range * 3.819719
    lsts = np.reshape(lsts,(uvc.Ntimes,-1))[:,0]
    yticks = [int(i) for i in np.linspace(0, len(lsts) - 1, 6)]
    yticklabels = [np.around(lsts[ytick], 1) for ytick in yticks]

    if phaseWrapRemoval:
        g1 = removePhaseWrap(g1,freqs)
        g2 = removePhaseWrap(g2,freqs)

    gplot = np.multiply(g2,np.conj(g1))
    gslice = gplot[gsliceInd,:]

    fig, axp = plt.subplots(1,3,figsize=(16,8))
    if gsliceType=='singleSlice':
        g1slice = g1[20,:]
    elif gsliceType=='average':
        gslice = np.average(g1,axis=0)
    for i in range(90):
        axp[0].plot(abs(g1[i,:]),label='abs',color='gray',alpha=0.3)
        axp[1].plot(np.real(g1[i,:]),label='real',color='gray',alpha=0.3)
        axp[2].plot(np.imag(g1[i,:]),label='imag',color='gray',alpha=0.3)
    axp[0].plot(abs(g1slice),label='abs',color='black')
    axp[1].plot(np.real(g1slice),label='real',color='black')
    axp[2].plot(np.imag(g1slice),label='imag',color='black')
    axp[0].set_title('g1: abs')
    axp[1].set_title('g1: real')
    axp[2].set_title('g1: imag')

    fig, axp = plt.subplots(1,3,figsize=(16,8))
    g2slice = g2[20,:]
    for i in range(90):
        axp[0].plot(abs(g2[i,:]),label='abs',color='gray',alpha=0.3)
        axp[1].plot(np.real(g2[i,:]),label='real',color='gray',alpha=0.3)
        axp[2].plot(np.imag(g2[i,:]),label='imag',color='gray',alpha=0.3)
    axp[0].plot(abs(g2slice),label='abs',color='black')
    axp[1].plot(np.real(g2slice),label='real',color='black')
    axp[2].plot(np.imag(g2slice),label='imag',color='black')
    axp[0].set_title('g2: abs')
    axp[1].set_title('g2: real')
    axp[2].set_title('g2: imag')
        
    if savefig == True:
        plt.savefig(outfig,facecolor='white')
        if write_params:
            utils.write_params_to_text(outfig,args,curr_func='plotGainComponentsByTime',
                   curr_file=curr_file,**kwargs)
    else:
        plt.show()
    plt.close()
    
def plotGainFit(fit='ones',raw=None,cal=None,gain=None,readVis=False,baseline=(124,35),savefig=False,outfig='',
               write_params=True,pol='XX',datadir=None,jd=None,title='',tophatWidth=10,polyDeg=2,sigma=[2,2],
               useUvcal=True,subPhaseWrap=False,plot_ft=False,ds_lim=None,gain_conv='divide',
               flip_gain_conj=True,**kwargs):
    from matplotlib.colors import LogNorm
    args = locals()
    
    if readVis:
        raw, cal, _, gain = plots.readFiles(datadir,jd,pol=pol)
        uvc, raw, cal, _ = trimVisTimes(raw,cal,None,gain)
    
    
    r = np.asarray(raw.get_data(baseline[0],baseline[1],pol))
    c = cal.get_data(baseline[0],baseline[1],pol)
    g1 = np.transpose(gain.get_gains(baseline[0],pol))
    g2 = np.transpose(gain.get_gains(baseline[1],pol))
    gplot = np.multiply(g2,np.conj(g1))
    fhdgains = np.multiply(g2,np.conj(g1))
    
    freqs = raw.freq_array[0]*1e-6
    taus = np.fft.fftshift(np.fft.fftfreq(freqs.size, np.diff(freqs)[0]*1e6))*1e9
    if ds_lim is not None:
        tind1 = np.argmin(abs(np.subtract(taus,-ds_lim)))
        tind2 = np.argmin(abs(np.subtract(taus,ds_lim)))
        if (tind2-tind1)%2==0:
            tind2+=1
        taus = taus[tind1:tind2]
    else:
        tind1=0
        tind2=len(taus)-1
    xticks = [int(i) for i in np.linspace(0, len(freqs) - 1, 5)]
    xticklabels = [int(freqs[x]) for x in xticks]
    tticks = [int(i) for i in np.linspace(0, len(taus) - 1, 5)]
    tticklabels = [int(taus[x]) for x in tticks]
    
    if subPhaseWrap:
#         #Take average across times and do linear fit
#         gslice = np.mean(gplot,axis=0)
#         unwrap = np.unwrap(np.angle(gslice))
#         res = scipy.stats.linregress(freqs,unwrap,alternative='less')
#         sfit = res.intercept + res.slope*freqs
#         #Subtract fit from all times
#         whole_sub = np.subtract(np.angle(gplot),sfit)
#         #Re-wrap phase
#         whole_sub = ( whole_sub + np.pi) % (2 * np.pi ) - np.pi
#         #Re-combine real and imag parts
#         comb = np.cos(whole_sub) + 1j*np.sin(whole_sub)
#         gplot = np.multiply(abs(gplot),comb)
        gplot = removePhaseWrap(gplot,freqs)
    if fit=='ones':
        fitArray = np.ones(np.shape(gplot))
    elif fit=='tophatConvolve':
        box = np.ones((1,tophatWidth))/tophatWidth
        gplot = scipy.signal.convolve2d(gplot, box,mode='same')
    elif fit=='polynomial':
        X = np.arange(len(freqs))
        gplot = np.array([np.polyval(np.polyfit(X, gplot[i,:], polyDeg),X) for i in range(len(gplot))])
    elif fit=='gauss2d':
        gplot = scipy.ndimage.gaussian_filter(gplot, sigma, mode='constant')
    elif fit=='harmonic':
        from scipy import optimize
        optimize.curve_fit(func, x, train_data)
    if subPhaseWrap:
        #Add phase wrap back in
        an = np.unwrap(np.angle(gplot))
        an_add = np.add(an,sfit)
        an_add = ( an_add + np.pi) % (2 * np.pi ) - np.pi
        comb = np.cos(an_add) + 1j*np.sin(an_add)
        gplot = np.multiply(abs(gplot),comb)
        
    if useUvcal:
        gain.gain_convention=gain_conv
        cfit = pyutils.uvcalibrate(raw,gain,inplace=False,flip_gain_conj=flip_gain_conj)
        calFit = cfit.get_data(baseline[0],baseline[1],pol)
        gmult = np.repeat(gplot,repeats=5,axis=0)
    else:
        gmult = np.repeat(gplot,repeats=5,axis=0)
        calFit = np.divide(r,gmult)

    
    fig = plt.figure(figsize=(16,8))
    gs = fig.add_gridspec(2,6,wspace=0.2)
    
    # RAW
    ax = fig.add_subplot(gs[0,0])
    
    im = ax.imshow(np.abs(r),interpolation='nearest',aspect='auto',cmap='viridis',
             vmin=np.percentile(np.abs(r),8),vmax=np.percentile(np.abs(r),92))
    ax.set_title('Raw')
    ax.set_xticks([])
    plt.colorbar(im)
    ax = fig.add_subplot(gs[1,0])
    if plot_ft:
        ds = np.fft.fftshift(np.fft.ifft(r,axis=1))
        ds = ds[:,tind1:tind2]
        im = ax.imshow(np.abs(ds),interpolation='nearest',aspect='auto',cmap='viridis',
                 vmin=np.percentile(np.abs(ds),8),vmax=np.percentile(np.abs(ds),92))
#         im = ax.imshow(np.abs(ds),interpolation='nearest',aspect='auto',cmap='viridis',
#                  norm=LogNorm(vmin=1000, vmax=np.percentile(np.abs(ds),98)))
        ax.set_xticks(tticks)
        ax.set_xticklabels(tticklabels)
        ax.set_xlabel('Delay (ns)')
    else:
        im = ax.imshow(np.angle(r),interpolation='nearest',aspect='auto',vmin=-np.pi,vmax=np.pi,cmap='twilight')
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
        ax.set_xlabel('Freq (MHz)')
    plt.colorbar(im)
    ax.set_yticks([])
    
    # GAINS
    ax = fig.add_subplot(gs[0,1])
    im = ax.imshow(np.abs(fhdgains),interpolation='nearest',aspect='auto',cmap='viridis',
             vmin=np.percentile(np.abs(fhdgains),8),vmax=np.percentile(np.abs(fhdgains),92))
    ax.set_yticks([])
    plt.colorbar(im)
    ax.set_title('g1 x g2*')
    ax.set_xticks([])
    ax = fig.add_subplot(gs[1,1])
    if plot_ft:
        ds = np.fft.fftshift(np.fft.ifft(fhdgains,axis=1))
        ds = ds[:,tind1:tind2]
        im = ax.imshow(np.abs(ds),interpolation='nearest',aspect='auto',cmap='viridis',
                 vmin=np.percentile(np.abs(ds),8),vmax=np.percentile(np.abs(ds),92))
        ax.set_xticks(tticks)
        ax.set_xticklabels(tticklabels)
        ax.set_xlabel('Delay (ns)')
    else:
        im = ax.imshow(np.angle(fhdgains),interpolation='nearest',aspect='auto',vmin=-np.pi,vmax=np.pi,cmap='twilight')
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
        ax.set_xlabel('Freq (MHz)')
    plt.colorbar(im)
    ax.set_yticks([])
    
    #FIT
    ax = fig.add_subplot(gs[0,2])
    im = ax.imshow(np.abs(gmult),interpolation='nearest',aspect='auto',cmap='viridis',
             vmin=np.percentile(np.abs(gmult),8),vmax=np.percentile(np.abs(gmult),92))
    ax.set_yticks([])
    plt.colorbar(im)
    ax.set_title('Fitted Gains')
    ax.set_xticks([])
    ax = fig.add_subplot(gs[1,2])
    if plot_ft:
        ds = np.fft.fftshift(np.fft.ifft(gmult,axis=1))
        ds = ds[:,tind1:tind2]
        im = ax.imshow(np.abs(ds),interpolation='nearest',aspect='auto',cmap='viridis',
                 vmin=np.percentile(np.abs(ds),8),vmax=np.percentile(np.abs(ds),92))
        ax.set_xticks(tticks)
        ax.set_xticklabels(tticklabels)
        ax.set_xlabel('Delay (ns)')
    else:
        im = ax.imshow(np.angle(gmult),interpolation='nearest',aspect='auto',vmin=-np.pi,vmax=np.pi,cmap='twilight')
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
        ax.set_xlabel('Freq (MHz)')
    plt.colorbar(im)
    ax.set_yticks([])
    
    #INITIAL CAL
    ax = fig.add_subplot(gs[0,3])
    im = ax.imshow(np.abs(c),interpolation='nearest',aspect='auto',cmap='viridis',
             vmin=np.percentile(np.abs(c),8),vmax=np.percentile(np.abs(c),92))
    ax.set_yticks([])
    plt.colorbar(im)
    ax.set_title('Raw Cal')
    ax.set_xticks([])
    ax = fig.add_subplot(gs[1,3])
    if plot_ft:
        ds_raw = np.fft.fftshift(np.fft.ifft(c,axis=1))
        ds_raw = ds_raw[:,tind1:tind2]
        im = ax.imshow(np.abs(ds_raw),interpolation='nearest',aspect='auto',cmap='viridis',
                 vmin=np.percentile(np.abs(ds_raw),8),vmax=np.percentile(np.abs(ds_raw),92))
        ax.set_xticks(tticks)
        ax.set_xticklabels(tticklabels)
        ax.set_xlabel('Delay (ns)')
    else:
        im = ax.imshow(np.angle(c),interpolation='nearest',aspect='auto',vmin=-np.pi,vmax=np.pi,cmap='twilight')
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
        ax.set_xlabel('Freq (MHz)')
    plt.colorbar(im)
    ax.set_yticks([])
    
    #CAL WITH FIT
    ax = fig.add_subplot(gs[0,4])
    im = ax.imshow(np.abs(calFit),interpolation='nearest',aspect='auto',cmap='viridis',
             vmin=np.percentile(np.abs(calFit),8),vmax=np.percentile(np.abs(calFit),92))
    ax.set_yticks([])
    plt.colorbar(im)
    ax.set_title('Cal With Fit')
    ax.set_xticks([])
    ax = fig.add_subplot(gs[1,4])
    if plot_ft:
        ds_fit = np.fft.fftshift(np.fft.ifft(calFit,axis=1))
        ds_fit = ds_fit[:,tind1:tind2]
        im = ax.imshow(np.abs(ds_fit),interpolation='nearest',aspect='auto',cmap='viridis',
                 vmin=np.percentile(np.abs(ds_fit),8),vmax=np.percentile(np.abs(ds_fit),92))
        ax.set_xticks(tticks)
        ax.set_xticklabels(tticklabels)
        ax.set_xlabel('Delay (ns)')
    else:
        im = ax.imshow(np.angle(calFit),interpolation='nearest',aspect='auto',vmin=-np.pi,vmax=np.pi,cmap='twilight')
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
        ax.set_xlabel('Freq (MHz)')
    plt.colorbar(im)
    ax.set_yticks([])
    
    #RESIDUAL
    ax = fig.add_subplot(gs[0,5])
    residAbs = np.subtract(np.abs(calFit),np.abs(c))
    residAngle = np.subtract(np.angle(calFit),np.angle(c))
    vmax = np.max([abs(np.percentile(residAbs,8)),abs(np.percentile(residAbs,92))])
    vmin=-vmax
    im = ax.imshow(residAbs,interpolation='nearest',aspect='auto',cmap='coolwarm',
             vmin=vmin,vmax=vmax)
    ax.set_yticks([])
    plt.colorbar(im)
    ax.set_title('Residual Cal')
    ax.set_xticks([])
    ax = fig.add_subplot(gs[1,5])
    if plot_ft:
        resid = np.subtract(abs(ds_raw),abs(ds_fit))
        vmax = np.max([abs(np.percentile(resid,8)),abs(np.percentile(resid,92))])
        vmin=-vmax
        im = ax.imshow(resid,interpolation='nearest',aspect='auto',cmap='coolwarm',
                 vmin=vmin,vmax=vmax)
        ax.set_xticks(tticks)
        ax.set_xticklabels(tticklabels)
        ax.set_xlabel('Delay (ns)')
    else:
        im = ax.imshow(residAngle,interpolation='nearest',aspect='auto',vmin=-np.pi,vmax=np.pi,cmap='twilight')
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
        ax.set_xlabel('Freq (MHz)')
    plt.colorbar(im)
    ax.set_yticks([])
    
    plt.suptitle(title)
    
    if savefig == True:
        plt.savefig(outfig,facecolor='white')
        if write_params:
            utils.write_params_to_text(outfig,args,curr_func='plotGainFit',
                   curr_file=curr_file,**kwargs)
    else:
        plt.show()
    plt.close()
    

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

            
def plotSsins(datadir, jd, pol='XX',freq_range=[0,-1]):
    fname = f'{datadir}/{jd}_ssins_flags_{pol}.hdf5'
    f = UVFlag()
    if os.path.isfile(fname):
        f.read(fname)
    else:
        fnames = sorted(glob.glob(f'{datadir}/*flags.h5'))
        f.read(fnames)
    freqs = f.freq_array
    if not (freq_range[0]==0 and freq_range[1]==-1):
        minind = np.argmin(abs(np.subtract(freqs*1e-6,freq_range[0])))
        maxind = np.argmin(abs(np.subtract(freqs*1e-6,freq_range[-1])))
        freq_range = [minind,maxind]
    
    freqs = freqs[freq_range[0]:freq_range[1]]
    f.select(frequencies=freqs)
    
    fig, ax = plt.subplots(1,1,figsize=(12,10))
    im = plt.imshow(f.flag_array[:,:,0],aspect='auto',interpolation='nearest')
    fig.colorbar(im)
    
    freqs *= 1e-6
    lsts = np.reshape(f.lst_array* 3.819719,(f.Ntimes,-1))[:,0]
    xticks = [int(i) for i in np.linspace(0, len(freqs) - 1, 5)]
    xticklabels = [int(freqs[x]) for x in xticks]
    yticks = [int(i) for i in np.linspace(0, len(lsts) - 1, 6)]
    yticklabels = [np.around(lsts[ytick], 1) for ytick in yticks]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    plt.xlabel('Freq')
    plt.ylabel('Time (LST)')
    
    
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
                   jd=2459855,readOnly=False,NblsPlot='all',sortBy='blLength',pol='XX',percentile=90,
                   readAllFiles=False):
    from pyuvdata import UVCal
    args = locals()
    
    if uvc is None:
        gf = sorted(glob.glob(f'{datadir}/{jd}_fhd_gains_{pol}*'))
        print('Reading Gains')
        if len(gf)<3:
            dirlist = f'{datadir}/{jd}_fhd_dirlist.txt'
            calfiles, obsfiles, layoutfiles, settingsfiles = getFhdGainDirs(dirlist)
            uvc = UVCal()
            uvc.read_fhd_cal(calfiles, obsfiles,layoutfiles,settingsfiles)
            uvc.file_name = dirlist
        elif len(gf)==1:
            uvc = UVCal()
            uvc.read_calfits(gf[0])
            dirlist=None
        else:
            uvc = UVCal()
            uvc.read_calfits(gf)
            dirlist=None
    else:
        dirlist=None
    if raw is None:
        print('Reading Raw')
        raw = UVData()
        rf = sorted(glob.glob(f'{datadir}/{jd}_fhd_raw_data_{pol}*'))
#         raw_file = f'{datadir}/{jd}_fhd_raw_data_{pol}_0.uvfits'
        if readAllFiles is False:
            rf = [rf[0]]
        print('Raw files are: \n')
        print(rf)
        raw.read(rf)
        raw.file_name = rf
    else:
        raw_file=None
    if cal is None:
        print('Reading Cal')
        cal = UVData()
        cal_file = f'{datadir}/{jd}_fhd_calibrated_data_{pol}.uvfits'
        cal.read(cal_file)
        cal.file_name = cal_file
    else:
        cal_file=None
    if model is None:
        print('Reading Model')
        model = UVData()
        model_file = f'{datadir}/{jd}_fhd_model_data_{pol}.uvfits'
        model.read(model_file)
        model.file_name=model_file
    else:
        model_file=None
    print('Before select:')
    print(f'Raw has LSTS {raw.lst_array[0]* 3.819719} to {raw.lst_array[-1]* 3.819719}')
    print(f'Model has LSTS {model.lst_array[0]* 3.819719} to {model.lst_array[-1]* 3.819719}')
    print(f'Cal has LSTS {cal.lst_array[0]* 3.819719} to {cal.lst_array[-1]* 3.819719}')
    print(f'Gains has LSTS {uvc.lst_array[0]* 3.819719} to {uvc.lst_array[-1]* 3.819719}')
    
#     print('Cal times:')
#     print(np.unique(cal.time_array))
#     print('Raw times:')
#     print(np.unique(raw.time_array))
    
    times = np.unique(cal.time_array)
    rawtimes = np.unique(raw.time_array)
    gaintimes = np.unique(uvc.time_array)
    if times[0]<rawtimes[0]:
        ind = np.argmin(abs(np.subtract(times,rawtimes[0])))
        times = times[ind:]
    if times[-1]>rawtimes[-1]:
        ind = np.argmin(abs(np.subtract(times,rawtimes[-1])))
        times = times[0:ind]
    gmin = np.argmin(abs(np.subtract(gaintimes,times[0])))
    gmax = np.argmin(abs(np.subtract(gaintimes,times[-1])))
    raw.select(times=times)
    model.select(times=times)
    cal.select(times=times)
    uvc.select(times=gaintimes[gmin:gmax])
    print('After select:')
    print(f'Raw has LSTS {raw.lst_array[0]* 3.819719} to {raw.lst_array[-1]* 3.819719}')
    print(f'Model has LSTS {model.lst_array[0]* 3.819719} to {model.lst_array[-1]* 3.819719}')
    print(f'Cal has LSTS {cal.lst_array[0]* 3.819719} to {cal.lst_array[-1]* 3.819719}')
    print(f'Gains has LSTS {uvc.lst_array[0]* 3.819719} to {uvc.lst_array[-1]* 3.819719}')
    if readOnly:
        return uvc, raw, cal, model
    
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
        if NblsPlot>=1:
                allbls = allbls[0:NblsPlot]
        else:
            inds = np.linspace(0,len(allbls)-1,int(len(allbls)*NblsPlot))
            allbls = [allbls[int(i)] for i in inds]
            
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
        if len(lengths)==0:
            continue
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
            r = rawNorm(raw.get_data((bl[0],bl[1],pol)))
            c = calNorm(cal.get_data((bl[0],bl[1],pol)))
            m = modelNorm(model.get_data((bl[0],bl[1],pol)))
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
                if ax is None:
                    continue
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
