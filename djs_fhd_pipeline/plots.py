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
from djs_fhd_pipeline import utils as djs_utils

dirpath = os.path.dirname(os.path.realpath(__file__))
githash = utils.get_git_revision_hash(dirpath)
curr_file = __file__

warnings.filterwarnings("ignore", message='This figure includes Axes that are not compatible with tight_layout, so results might be incorrect.')

import json

def getFhdFiles(fhd_files):
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
            print(f'Stopping at obs {path} and reading gains up to this point')
            break
    return fhd_file_array, calfiles, obsfiles, layoutfiles, settingsfiles

def readRawFiles(fhdnames,rawnames,jd,raw=None, cal=None, model=None, gains=None, pol='XX',ind_range='all'):
    f = open(fhdnames, "r")
    fhd_files = f.read().split('\n')[0:-1]
    if ind_range == 'all':
        ind_range = [0,-1]
    fhd_files = fhd_files[ind_range[0]:ind_range[1]]
    if raw is None:
        print('Reading raw')
        raw = UVData()
        f = open(rawnames, "r")
        raw_data = f.read().split('\n')[0:-1]
        raw.read(raw_data[ind_range[0]:ind_range[1]],ignore_name=True)
        raw.phase_to_time(raw.time_array[len(raw.time_array)//2])
    if cal is None:
        fhd_file_array,_,_,_,_ = getFhdFiles(fhd_files)
        cal = UVData()
        cal.read(fhd_file_array[startind:stopind],use_model=False,ignore_name=True,polarizations=pol)
        cal.phase_to_time(cal.time_array[len(model.time_array)//2])
    if model is None:
        fhd_file_array = getFhdFiles(fhd_files)
        model = UVData()
        model.read(fhd_file_array[startind:stopind],use_model=True,ignore_name=True,polarizations=pol)
        model.phase_to_time(model.time_array[len(model.time_array)//2])
    if gains is None:
        _, calfiles, obsfiles, layoutfiles, settingsfiles = getFhdFiles(fhd_files)
        gains = UVCal()
        gains.read_fhd_cal(cal_file=calfiles,obs_file=obsfiles,layout_file=layoutfiles,settings_file=settingsfiles,
                        run_check=False,run_check_acceptability=False)
    return raw, cal, model, gains
        

def readFiles(datadir, jd, raw=None, cal=None, model=None, gains=None, extraCal=False, incGains=True, pol='XX'):
    if raw is None:
        print('Reading raw')
        raw = UVData()
        raw_file = sorted(glob.glob(f'{datadir}/*_raw_data_{pol}*uvfits'))
        # if not os.path.isfile(raw_file):
        #     raw_file = f'{datadir}/{jd}_fhd_raw_data_{pol}_0.uvfits'
        raw.read(raw_file)
        raw.file_name = raw_file
    else:
        raw_file=None
    if cal is None:
        print('Reading cal')
        cal = UVData()
        cal_file = f'{datadir}/{jd}_fhd_calibrated_data_{pol}.uvfits'
        if not os.path.isfile(cal_file):
            cal_file = f'{datadir}/{jd}_fhd_calibrated_data_{pol}_0.uvfits'
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
        if not os.path.isfile(model_file):
            model_file = f'{datadir}/{jd}_fhd_model_data_{pol}_0.uvfits'
        model.read(model_file)
        model.file_name=model_file
    else:
        model_file=None
    if incGains is True and gains is None:
        print('Reading Gains')
        gains = UVCal()
        gains_file = f'{datadir}/{jd}_fhd_gains_{pol}.uvfits'
        if not os.path.isfile(gains_file):
            gains_file = f'{datadir}/{jd}_fhd_gains_data_{pol}_0.uvfits'
        if not os.path.isfile(gains_file):
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
    a1s,a2s = pyutils.baseline_to_antnums(bls, Nants_telescope=uv.Nants_data)
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
        if plot_range[1]-plot_range[0]>50:
            if ind%20==0:
                print(str(ind).zfill(3))
        else:
            print(str(ind).zfill(3))
        # uv = UVData()
        # fhd_files = [layout_files[ind],obs_files[ind],params_files[ind],dirty_vis_files[ind],flag_files[ind],settings_files[ind]]
        _,uv,_,_ = djs_utils.read_fhd(fhd_path,file_range=[ind,ind+1],rawfiles='',
                               readModel=False,readRaw=False,readGains=False)
        # try:
        #     # uv.read_fhd(fhd_files,read_data=False)
        #     uv = utils.read_fhd(fhd_path,file_range=[ind,ind+1])
        # except:
        #     print(f'COULD NOT READ FILE: {fhd_files}')
        #     continue
        lst = uv.lst_array[0] * 3.819719
        if np.unique(uv.time_array)[0] < 2459906.26 and np.unique(uv.time_array)[0]>2459906.24:
            print(lst)
        pos = [uv.phase_center_app_ra*57.2958,-31]
        _dec_range = [pos[1]-dec_range/2,pos[1]+dec_range/2]
        prefix = f'{fhd_path}/output_data'
        # color_scale=[-1763758,1972024]
        color_scale=[-1e6,1e6]
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
        im = plot_fits.plot_fits_image(data, axes[1][0], color_scale, output_path, prefix, write_pixel_coordinates, log_scale,
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
        
def plotGainDelays(uvc,divideByK=True,baseline=(124,35),pol='XX',savefig=False,write_params=True,
                  outfig='',subPhaseWrap=False, nb_path='', **kwargs):
    args = locals()
    from scipy import optimize
    freqs = uvc.freq_array[0]*1e-6
    taus = np.fft.fftshift(np.fft.fftfreq(freqs.size, np.diff(freqs)[0]*1e6))*1e9
    inds = np.arange(0,len(freqs))
    g1 = np.transpose(uvc.get_gains(baseline[0],pol))
    g2 = np.transpose(uvc.get_gains(baseline[1],pol))
    gplot = np.multiply(g2,np.conj(g1))
    freqs *=1e6
    
    if subPhaseWrap:
        g1 = removePhaseWrap(g1,freqs)
        g2 = removePhaseWrap(g2,freqs)

    gplot = np.multiply(g2,np.conj(g1))
    gslice = gplot[20,:]


    ds_lim=1500
    # ds1 = np.fft.fftshift(np.fft.ifft(g1,axis=1))
    ds1 = []
    for row in g1:
        d = np.fft.fftshift(np.fft.ifft(row))
        ds1.append(d)
    ds2 = []
    for row in g1:
        d = np.fft.fftshift(np.fft.ifft(row))
        ds2.append(d)
    ds1 = np.asarray(ds1)
    ds2 = np.asarray(ds2)
    # ds2 = np.fft.fftshift(np.fft.ifft(g2,axis=1))
    tind1 = np.argmin(abs(np.subtract(taus,-ds_lim)))
    tind2 = np.argmin(abs(np.subtract(taus,ds_lim)))
    if (tind2-tind1)%2==0:
        tind2+=1
    taus = taus[tind1:tind2]
    ds1 = ds1[:,tind1:tind2]
    ds2 = ds2[:,tind1:tind2]
    tticks = [int(i) for i in np.linspace(0, len(taus) - 1, 5)]
    tticklabels = [int(taus[x]) for x in tticks]

    if divideByK:
        ds_div = np.exp(1j*taus)
        ds1_div = np.divide(ds1,ds_div)
        ds2_div = np.divide(ds2,ds_div)

    fig, ax = plt.subplots(2,2,figsize=(16,10))
    im = ax[0][0].imshow(np.angle(ds1),vmin=-np.pi,vmax=np.pi,cmap='twilight',aspect='auto',interpolation='nearest')
    ax[0][0].set_title(f'Ant {baseline[0]}: Phase of delay of gains')
    plt.colorbar(im,ax=ax[0])
    im = ax[0][1].imshow(np.angle(ds1_div),vmin=-np.pi,vmax=np.pi,cmap='twilight',aspect='auto',interpolation='nearest')
    ax[0][1].set_title(f'Ant {baseline[0]}: Normalized by bin')
    plt.colorbar(im,ax=ax[1])
    for a in ax[0]:
        a.set_xticks(tticks)
        a.set_xticklabels(tticklabels)
    fig.suptitle(f'Ant {baseline[0]} gains')

#     fig, ax = plt.subplots(1,2,figsize=(16,10))
    im = ax[1][0].imshow(np.angle(ds2),vmin=-np.pi,vmax=np.pi,cmap='twilight',aspect='auto',interpolation='nearest')
    ax[1][0].set_title(f'Ant {baseline[1]}: Phase of delay of gains')
#     plt.colorbar(im,ax=ax[0])
    im = ax[1][1].imshow(np.angle(ds2_div),vmin=-np.pi,vmax=np.pi,cmap='twilight',aspect='auto',interpolation='nearest')
    ax[1][1].set_title(f'Ant {baseline[1]}: Normalized by bin')
#     plt.colorbar(im,ax=ax[1])
    for a in ax[1]:
        a.set_xticks(tticks)
        a.set_xticklabels(tticklabels)
    fig.suptitle(f'Ant {baseline[1]} gains')
    if savefig:
        plt.savefig(outfig,bbox_inches='tight')
        if write_params:
            curr_func = inspect.stack()[0][3]
            utils.write_params_to_text(outfig,args,curr_file=curr_file,
                                       curr_func="plotGainDelays",githash=githash,nb_path=nb_path)
        plt.close()
    else:
        plt.show()
        plt.close()

def plotBeamSliceComparisons(fhddirs,labels,savefig=False,write_params=True,outfig='',
                            nb_path='',peakNorm=False):
    args = locals()
    fig = plt.figure()
    for i,d in enumerate(fhddirs):
        beamdir = sorted(glob.glob(f'{d}/fhd_*'))[0]
        beamFile = glob.glob(f'{beamdir}/output_data/*_Beam_XX.fits')[0]
        data = plot_fits.load_image(beamFile).signal_arr
        dslice = data[511,:]
    #     fig = plt.figure()
    #     plt.imshow(data,aspect='auto',interpolation='nearest')
        plt.plot(dslice,label=labels[i])
    plt.legend()
    if savefig:
        plt.savefig(outfig,bbox_inches='tight')
        if write_params:
            curr_func = inspect.stack()[0][3]
            utils.write_params_to_text(outfig,args,curr_file=curr_file,
                                       curr_func="plotBeamSliceComparisons",githash=githash,nb_path=nb_path)
    else:
        plt.show()
        plt.close()
        
def plotBeam(fhd_dir,pol='XX',color_scale=[-1763758,1972024],output_path='',prefix='beam',
             write_pixel_coordinates=False,log_scale=False,ra_range=40,dec_range=40,fontsize=16,
             savefig=False,write_params=True,outfig=''):
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
    if savefig:
        plt.savefig(outfig,bbox_inches='tight')
        if write_params:
            curr_func = inspect.stack()[0][3]
            utils.write_params_to_text(outfig,args,curr_file=curr_file,
                                       curr_func="plotBeam",githash=githash,nb_path=nb_path)
    else:
        plt.show()
        plt.close()
        
def plotGainsSimple(raw=None,cal=None,gain=None,readVis=False,baseline=(124,35),savefig=False,outfig='',
               write_params=True,pol='XX',datadir=None,jd=None,title='',subPhaseWrap=False,plot_ft=False,
               ds_lim=None,gain_conv='divide',flip_gain_conj=True,**kwargs):
    from matplotlib.colors import LogNorm
    args = locals()
    
    if readVis:
        raw, cal, _, gain = readFiles(datadir,jd,pol=pol)
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
    lsts = np.reshape(raw.lst_array* 3.819719,(raw.Ntimes,-1))[:,0]
    yticks = [int(i) for i in np.linspace(0, len(lsts) - 1, 6)]
    yticklabels = [np.around(lsts[ytick], 1) for ytick in yticks]
    
    if subPhaseWrap:
        gplot = removePhaseWrap(gplot,freqs)

    
    fig = plt.figure(figsize=(16,8))
    gs = fig.add_gridspec(2,3,wspace=0.2)
    
    # RAW
    ax = fig.add_subplot(gs[0,0])
    ax.set_ylabel('LST (hours)')
    
    im = ax.imshow(np.abs(r),interpolation='nearest',aspect='auto',cmap='viridis',
             vmin=np.percentile(np.abs(r),8),vmax=np.percentile(np.abs(r),92))
    ax.set_title('Raw')
    ax.set_xticks([])
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    plt.colorbar(im)
    ax = fig.add_subplot(gs[1,0])
    ax.set_ylabel('LST (hours)')
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
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    
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
    
    #INITIAL CAL
    ax = fig.add_subplot(gs[0,2])
    im = ax.imshow(np.abs(c),interpolation='nearest',aspect='auto',cmap='viridis',
             vmin=np.percentile(np.abs(c),8),vmax=np.percentile(np.abs(c),92))
    ax.set_yticks([])
    plt.colorbar(im)
    ax.set_title('Calibrated')
    ax.set_xticks([])
    ax = fig.add_subplot(gs[1,2])
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
    
    plt.suptitle(title)
    
    if savefig == True:
        plt.savefig(outfig,facecolor='white')
        if write_params:
            utils.write_params_to_text(outfig,args,curr_func='plotGainsSimple',
                   curr_file=curr_file,**kwargs)
    else:
        plt.show()
    plt.close()
        
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
    
    
    if nframes=='all':
        nframes = len(dirty_files)
    if plot_range=='all':
        plot_range = [0,nframes]
    print(f'Creating images for {plot_range[1]-plot_range[0]} files')
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
                             outfig='',ds_lim=1500,gsliceInd=20,norm=np.angle,**kwargs):
    from scipy import optimize
    import scipy
    args = locals()
    freqs = uvc.freq_array[0]*1e-6
    taus = np.fft.fftshift(np.fft.fftfreq(freqs.size, np.diff(freqs)[0]*1e6))*1e9
    inds = np.arange(0,len(freqs))
    g1 = np.transpose(uvc.get_gains(baseline[0],pol))
    g2 = np.transpose(uvc.get_gains(baseline[1],pol))
    a1 = abs(uvd.get_data(baseline[0],baseline[0],pol))
    a2 = abs(uvd.get_data(baseline[1],baseline[1],pol))
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
    dsGp = dsGp[:,tind1:tind2]
    tticks = [int(i) for i in np.linspace(0, len(taus) - 1, 5)]
    tticklabels = [int(taus[x]) for x in tticks]
    if norm==np.angle:
        vminA=-np.pi
        vminG=-np.pi
        vminGp=-np.pi
        vmaxA=np.pi
        vmaxG=np.pi
        vmaxGp=np.pi
        cmap='twilight'
    else:
        vminG = np.percentile(norm(dsG1),2)
        vmaxG = np.percentile(norm(dsG1),98)
        vminGp = np.percentile(norm(dsGp),2)
        vmaxGp = np.percentile(norm(dsGp),98)
        vminA = np.percentile(norm(dsA1),2)
        vmaxA = np.percentile(norm(dsA1),98)
        cmap='viridis'

    fig, ax = plt.subplots(1,5,figsize=(16,8))
    im = ax[0].imshow(norm(dsA1),vmin=vminA,vmax=vmaxA,cmap=cmap,aspect='auto',interpolation='nearest')
    plt.colorbar(im,ax=ax[0])
    ax[0].set_xticks(tticks)
    ax[0].set_xticklabels(tticklabels)
    ax[0].set_title(f'Ant {baseline[0]} Auto')
    ax[0].set_yticks(yticksA)
    ax[0].set_yticklabels(yticklabelsA)
    im = ax[1].imshow(norm(dsG1),vmin=vminG,vmax=vmaxG,cmap=cmap,aspect='auto',interpolation='nearest')
    plt.colorbar(im,ax=ax[1])
    ax[1].set_xticks(tticks)
    ax[1].set_xticklabels(tticklabels)
    ax[1].set_title(f'Ant {baseline[0]} Gain')
    ax[1].set_yticks(yticks)
    ax[1].set_yticklabels(yticklabels)
    im = ax[2].imshow(norm(dsGp),vmin=vminGp,vmax=vmaxGp,cmap=cmap,aspect='auto',interpolation='nearest')
    plt.colorbar(im,ax=ax[2])
    ax[2].set_xticks(tticks)
    ax[2].set_xticklabels(tticklabels)
    ax[2].set_yticks([])
    ax[2].set_title(f'g{baseline[0]} x g{baseline[1]}*')
    im = ax[3].imshow(norm(dsG2),vmin=vminG,vmax=vmaxG,cmap=cmap,aspect='auto',interpolation='nearest')
    plt.colorbar(im,ax=ax[3])
    ax[3].set_xticks(tticks)
    ax[3].set_xticklabels(tticklabels)
    ax[3].set_title(f'Ant {baseline[1]} Gain')
    ax[3].set_yticks(yticks)
    ax[3].set_yticklabels(yticklabels)
    im = ax[4].imshow(norm(dsA2),vmin=vminA,vmax=vmaxA,cmap=cmap,aspect='auto',interpolation='nearest')
    plt.colorbar(im,ax=ax[4])
    ax[4].set_xticks(tticks)
    ax[4].set_xticklabels(tticklabels)
    ax[4].set_title(f'Ant {baseline[1]} Auto')
    ax[4].set_yticks(yticksA)
    ax[4].set_yticklabels(yticklabelsA)
    
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
    axp[0].set_title(f'g{baseline[0]}: abs')
    axp[1].set_title(f'g{baseline[0]}: real')
    axp[2].set_title(f'g{baseline[0]}: imag')

    fig, axp = plt.subplots(1,3,figsize=(16,8))
    g2slice = g2[20,:]
    for i in range(90):
        axp[0].plot(abs(g2[i,:]),label='abs',color='gray',alpha=0.3)
        axp[1].plot(np.real(g2[i,:]),label='real',color='gray',alpha=0.3)
        axp[2].plot(np.imag(g2[i,:]),label='imag',color='gray',alpha=0.3)
    axp[0].plot(abs(g2slice),label='abs',color='black')
    axp[1].plot(np.real(g2slice),label='real',color='black')
    axp[2].plot(np.imag(g2slice),label='imag',color='black')
    axp[0].set_title(f'g{baseline[1]}: abs')
    axp[1].set_title(f'g{baseline[1]}: real')
    axp[2].set_title(f'g{baseline[1]}: imag')
        
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
        raw, cal, _, gain = readFiles(datadir,jd,pol=pol)
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
        gain.check()
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
        if not os.path.isfile(raw_file):
            raw_file = f'{datadir}/{jd}_fhd_raw_data_{pol}_0.uvfits'
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

            
def plotSsins(datadir=None, jd=None, fname='', pol='XX',freq_range=[0,-1]):
    if len(fname)==0:
        if datadir is None or jd is None:
            raise Exception('Must either provide fname or both of datadir and jd')
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
                   nb_path=None,fixLsts=False,clipTime=True):
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
    if clipTime:
        gains, raw, cal, model, = trimVisTimes(rawFull,calFull,modelFull,gainsFull,printTimes=True)
    
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

def plotBaselineMap(uv,bls='withData',use_ants=[],p_inc=100):
    allbls = []
    use_bls = []
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
                        allbls.append((a2,a1))
                        if a1 in use_ants and a2 in use_ants:
                            use_bls.append((a1,a2))
                            use_bls.append((a2,a1))
                    except:
                        continue
                elif bls=='all':
                    allbls.append((a1,a2))
                    allbls.append((a2,a1))
                    if a1 in use_ants and a2 in use_ants:
                        use_bls.append((a1,a2))
                        use_bls.append((a2,a1))
    allbls = np.asarray(allbls)
    print(len(allbls))
    allangs = []
    alldisps = []
    useangs = []
    usedisps = []
    pos = uv.antenna_positions + uv.telescope_location
    pos = pyutils.ENU_from_ECEF(pos, *uv.telescope_location_lat_lon_alt)
    for bl in allbls:
        p1 = pos[np.argwhere(uv.antenna_numbers == bl[0])]
        p2 = pos[np.argwhere(uv.antenna_numbers == bl[1])]
        disp = (p2 - p1)[0][0][0:2]
        alldisps.append(disp)
        # allangs.append(np.arctan(disp[1]/disp[0])*57.2958)
        if bl[0] in use_ants and bl[1] in use_ants:
            usedisps.append(disp)
            # useangs.append(np.arctan(disp[1]/disp[0])*57.2958)
    allangs = np.asarray(allangs)
    alldisps = np.asarray(alldisps)
    # useangs = np.asarray(useangs)
    usedisps = np.asarray(usedisps)
    fig = plt.figure(figsize=(10,10))
    for b,bl in enumerate(allbls):
        r = random.uniform(0,1)
        if r < p_inc/100:
            plt.scatter(alldisps[b][0],alldisps[b][1],color='blue',alpha=0.3,s=1)
    for b,bl in enumerate(use_bls):
        plt.scatter(usedisps[b][0],usedisps[b][1],color='green',s=1)
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
                   readAllFiles=False,clipTime=True):
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

    if clipTime:
        uvc, raw, cal, model, = trimVisTimes(raw,cal,model,uvc,printTimes=True)
    # print('Before select:')
    # print(f'Raw has LSTS {raw.lst_array[0]* 3.819719} to {raw.lst_array[-1]* 3.819719}')
    # print(f'Model has LSTS {model.lst_array[0]* 3.819719} to {model.lst_array[-1]* 3.819719}')
    # print(f'Cal has LSTS {cal.lst_array[0]* 3.819719} to {cal.lst_array[-1]* 3.819719}')
    # print(f'Gains has LSTS {uvc.lst_array[0]* 3.819719} to {uvc.lst_array[-1]* 3.819719}')
    
    # times = np.unique(cal.time_array)
    # rawtimes = np.unique(raw.time_array)
    # gaintimes = np.unique(uvc.time_array)
    # if times[0]<rawtimes[0]:
    #     ind = np.argmin(abs(np.subtract(times,rawtimes[0])))
    #     times = times[ind:]
    # if times[-1]>rawtimes[-1]:
    #     ind = np.argmin(abs(np.subtract(times,rawtimes[-1])))
    #     times = times[0:ind]
    # gmin = np.argmin(abs(np.subtract(gaintimes,times[0])))
    # gmax = np.argmin(abs(np.subtract(gaintimes,times[-1])))
    # raw.select(times=times)
    # model.select(times=times)
    # cal.select(times=times)
    # uvc.select(times=gaintimes[gmin:gmax])
    # print('After select:')
    # print(f'Raw has LSTS {raw.lst_array[0]* 3.819719} to {raw.lst_array[-1]* 3.819719}')
    # print(f'Model has LSTS {model.lst_array[0]* 3.819719} to {model.lst_array[-1]* 3.819719}')
    # print(f'Cal has LSTS {cal.lst_array[0]* 3.819719} to {cal.lst_array[-1]* 3.819719}')
    # print(f'Gains has LSTS {uvc.lst_array[0]* 3.819719} to {uvc.lst_array[-1]* 3.819719}')
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
                     title='',plotFullAntSet=True, yscale='linear'):
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
    if plotFullAntSet:
        baseline_groups, vec_bin_centers, lengths = uv.get_redundancies(
            use_antpos=False, include_autos=False
        )
        antpairs, lengths_full = unpackBlLengths(uv, baseline_groups,lengths)
    if len(use_ants)>0:
        uv_sub = uv.select(antenna_nums=use_ants,inplace=False)
        baseline_groups, vec_bin_centers, lengths = uv_sub.get_redundancies(
            use_antpos=False, include_autos=False
        )
        antpairs, lengths = unpackBlLengths(uv_sub, baseline_groups,lengths)
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
    plt.hist(lengths,bins=nbins,histtype='step',label='Included ants',color='blue')
    if yscale == 'log':
        plt.yscale('log')
    if plotFullAntSet:
        plt.hist(lengths_full,bins=nbins,histtype='step',label='All ants',color='green')
        plt.legend()
        if yscale == 'log':
            plt.yscale('log')
    plt.xlabel('Baseline Length(m)')
    plt.ylabel('Count')
    plt.xlim((0,max(lengths)+20))
    colors=['black','red','cyan','gold']
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
            
def plot_antenna_positions(uv, badAnts=[], flaggedAnts={}, use_ants="all",hexsize=35,inc_outriggers=False,
                          fontsize=10,title='',savefig=False,outfig='',write_params=True):
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
    args = locals()

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
                            plt.annotate(a, [x - 4, y-1],fontsize=fontsize)
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
                            plt.annotate(a, [x - 4, y-1],fontsize=fontsize)
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
                plt.annotate(a, [x - 4, y-1],fontsize=fontsize)
    plt.xlabel("East")
    plt.ylabel("North")
    plt.title(title)
    if savefig:
        plt.savefig(outfig)
        if write_params:
            curr_func = inspect.stack()[0][3]
            utils.write_params_to_text(outfig,args,curr_func,curr_file,githash)
        plt.close()
    else:
        plt.show()
    
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

def getFhdFilenames(datadir,pol,data_type='gains', dir_range='all'):
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
            vis_files.append(sorted(glob.glob(f'{dir}/vis_data/*vis_{pol}.sav'))[0])
            model_files.append(sorted(glob.glob(f'{dir}/vis_data/*model_{pol}.sav'))[0])
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

    if data_type == 'gains':
        return cal_files, obs_files, layout_files, settings_files
    elif data_type == 'data':
        return vis_files, params_files, obs_files, flag_files, layout_files, settings_files
    elif data_type == 'model':
        return model_files, params_files, obs_files, flag_files, layout_files, settings_files
    elif data_type == 'all':
        return model_files, vis_files, params_files, obs_files, flag_files, layout_files, settings_files, cal_files

def plotAllWaterfalls_partialIO(rawnames,fhd_dir,bl,uvtimes=None,nfiles='all',flagfile=None,plotFlags=False,
                               nfiles_sim=1, savefig=False, outdir='', write_params=True):
    import numpy.ma as ma
    import warnings
    warnings.filterwarnings("ignore", message="Telescope location derived from obs lat/lon/alt values does not match the location in the layout file. Using the value from known_telescopes.")
    warnings.filterwarnings("ignore", message="UVParameter Nsources does not match. Combining anyway.")

    args = locals()
    if nfiles != 'all':
        if len(rawnames) != (nfiles[1]-nfiles[0]):
            rawnames = rawnames[nfiles[0]:nfiles[1]]
    if uvtimes==None:
        print('Reading metadata for all files')
        uvtimes = UVData()
        uvtimes.read(rawnames,read_data=False)
    modf_x, visf_x, paramsf, obsf, flagsf, layoutf, settingsf, gainsf = getFhdFilenames(fhd_dir,'XX',dir_range=nfiles,data_type='all')
    modf_y, visf_y, _,_,_,_,_,_ = getFhdFilenames(fhd_dir,'YY',dir_range=nfiles,data_type='all')
    if len(rawnames) != len(obsf):
        raise "ERROR: Must have same number of raw and FHD files"
    fig, ax = plt.subplots(2,5,figsize=(16,16))
    if plotFlags:
        fig2, ax2 = plt.subplots(2,5,figsize=(16,16))
    # grange = np.zeros((2,2))
    # rrange = np.zeros((2,2))
    # crange = np.zeros((2,2))
    # mrange = np.zeros((2,2))
    for i in np.arange(0,len(rawnames),nfiles_sim):
        if i%100==0:
            print(i)
        raw = UVData()
        raw.read(rawnames[i:i+nfiles_sim])
        # if i==0:
        Nfreqs = len(raw.freq_array[0])
        if plotFlags:
            uvf = UVFlag()
            uvf.read(flagfile)
            uvf.select(times=np.unique(raw.time_array))
        try:
            gains = UVCal()
            gains.read_fhd_cal(gainsf[i],obs_file=obsf[i],layout_file=layoutf[i],settings_file=settingsf[i])
            mask = np.zeros((5,Nfreqs))
        except:
            g1 = np.tile([1e8]*Nfreqs,(5,1))
            g2 = np.tile([1e8]*Nfreqs,(5,1))
            mask = np.ones((5,Nfreqs))
        for p, pol in enumerate(['XX','YY']):
            try:
                cal = UVData()
                calf = [visf_x,visf_y][p]
                cal.read_fhd(calf[i], params_file=paramsf[i], obs_file=obsf[i], flags_file=flagsf[i], 
                             layout_file=layoutf[i], settings_file=settingsf[i])
                modelf = [modf_x,modf_y][p]
                model = UVData()
                model.read_fhd(modelf[i], params_file=paramsf[i], obs_file=obsf[i], flags_file=flagsf[i], 
                             layout_file=layoutf[i], settings_file=settingsf[i])
                c = abs(cal.get_data(bl[0],bl[1],pol))
                m = abs(model.get_data(bl[0],bl[1],pol))
                g1 = np.transpose(abs(gains.get_gains(bl[0],pol)))
                g2 = np.transpose(abs(gains.get_gains(bl[1],pol)))
                g1 = np.tile(g1,(5,1))
                g2 = np.tile(g2,(5,1))
            except:
                if p==0:
                    print(f'No data for obs ind {i}')
                c = np.tile([1e8]*Nfreqs,(5,1))
                m = np.tile([1e8]*Nfreqs,(5,1))
                g1 = np.tile([1e8]*Nfreqs,(5,1))
                g2 = np.tile([1e8]*Nfreqs,(5,1))
            if plotFlags:
                if np.shape(uvf.flag_array)[2]>1:
                    mask_f = uvf.flag_array[:,:,p]
            r = abs(raw.get_data(bl[0],bl[1],pol))
            r = ma.masked_array(r,mask=mask)
            c = ma.masked_array(c,mask=mask)
            m = ma.masked_array(m,mask=mask)
            g1 = ma.masked_array(g1,mask=mask)
            g2 = ma.masked_array(g2,mask=mask)
            # if i==0:
            #     grange[0,p] = np.min(g1)
            #     grange[1,p] = np.max(g1)
            #     mrange[0,p] = np.min(m)
            #     mrange[1,p] = np.max(m)
            #     crange[0,p] = np.min(c)
            #     crange[1,p] = np.max(c)
            #     rrange[0,p] = np.min(r)
            #     rrange[1,p] = np.max(r)
            # else:
            #     for v,vals in enumerate([grange,mrange,crange,rrange]):
            #         mn = np.percentile([g1,m,c,r][v],5)
            #         mx = np.percentile([g1,m,c,r][v],95)
            #         if mn < vals[0,p]:
            #             vals[0,p]=mn
            #         if mx > vals[1,p]:
            #             vals[1,p]=mx
            im = ax[p][0].imshow(g1,aspect='auto',extent=[0,245,i*5+5,i*5],origin='upper',interpolation='nearest',
                     vmin=0,vmax=160)
            im = ax[p][1].imshow(r,aspect='auto',extent=[0,245,i*5+5,i*5],origin='upper',interpolation='nearest',
                     vmin=0,vmax=1.1e5)
            im = ax[p][2].imshow(c,aspect='auto',extent=[0,245,i*5+5,i*5],origin='upper',interpolation='nearest',
                     vmin=0,vmax=80)
            im = ax[p][3].imshow(m,aspect='auto',extent=[0,245,i*5+5,i*5],origin='upper',interpolation='nearest',
                     vmin=0,vmax=20)
            im = ax[p][4].imshow(g2,aspect='auto',extent=[0,245,i*5+5,i*5],origin='upper',interpolation='nearest',
                     vmin=0,vmax=160)
            if plotFlags:
                # r = ma.masked_array(r,mask=mask_f)
                # c = ma.masked_array(c,mask=mask_f)
                # m = ma.masked_array(m,mask=mask_f)
                # g1 = ma.masked_array(g1,mask=mask_f)
                # g2 = ma.masked_array(g2,mask=mask_f)
                im = ax2[p][0].imshow(g1,aspect='auto',extent=[0,245,i*5+5,i*5],origin='upper',interpolation='nearest',
                         vmin=0,vmax=160)
                im = ax2[p][1].imshow(r,aspect='auto',extent=[0,245,i*5+5,i*5],origin='upper',interpolation='nearest',
                         vmin=0,vmax=1.1e5)
                im = ax2[p][2].imshow(c,aspect='auto',extent=[0,245,i*5+5,i*5],origin='upper',interpolation='nearest',
                         vmin=0,vmax=80)
                im = ax2[p][3].imshow(m,aspect='auto',extent=[0,245,i*5+5,i*5],origin='upper',interpolation='nearest',
                         vmin=0,vmax=20)
                im = ax2[p][4].imshow(g2,aspect='auto',extent=[0,245,i*5+5,i*5],origin='upper',interpolation='nearest',
                         vmin=0,vmax=160)
    
    
    ax[0][0].set_title(f'Ant {bl[0]} gains')
    ax[0][1].set_title('Raw')
    ax[0][2].set_title('Calibrated')
    ax[0][3].set_title('Model')
    ax[0][4].set_title(f'Ant {bl[1]} gains')
    if plotFlags:
        ax2[0][0].set_title(f'Ant {bl[0]} gains')
        ax2[0][1].set_title('Raw')
        ax2[0][2].set_title('Calibrated')
        ax2[0][3].set_title('Model')
        ax2[0][4].set_title(f'Ant {bl[1]} gains')
    freqs = uvtimes.freq_array[0]*1e-6
    pols=['XX','YY']
    xticks = [int(i) for i in np.linspace(0, Nfreqs - 1, 6)]
    xticklabels = [int(freqs[x]) for x in xticks]
    lsts = uvtimes.lst_array * 3.819719
    lsts = np.reshape(lsts,(uvtimes.Ntimes,-1))[:,0][0:len(rawnames)*5]
    yticks = [int(i) for i in np.linspace(0, len(lsts) - 1, 6)]
    yticklabels = [np.around(lsts[ytick], 1) for ytick in yticks]
    for a in ax:
        for b in a:
            b.set_ylim(len(rawnames)*5,0)
            b.set_xlim(0,Nfreqs)
            b.set_yticks(yticks)
    for a in ax[1]:
        a.set_xticklabels(xticklabels)
        a.set_xticks(xticks)
        a.set_xlabel('Freq (MHz)')
    for a in ax[0]:
        a.set_xticklabels([])
        a.set_xticks([])
    for p in [0,1]:
        ax[p][0].set_yticks(yticks)
        ax[p][0].set_yticklabels(yticklabels)
        ax[p][0].set_ylabel(f'LST (hours) | {pols[p]} pol')
        for i in [1,2,3,4]:
            ax[p][i].set_yticks([])
            ax[p][i].set_yticklabels([])
    if plotFlags:
        for a in ax2:
            for b in a:
                b.set_ylim(len(rawnames)*5,0)
                b.set_xlim(0,Nfreqs)
                b.set_yticks(yticks)
        for a in ax2[1]:
            a.set_xticklabels(xticklabels)
            a.set_xticks(xticks)
            a.set_xlabel('Freq (MHz)')
        for a in ax2[0]:
            a.set_xticklabels([])
            a.set_xticks([])
        for p in [0,1]:
            ax2[p][0].set_yticks(yticks)
            ax2[p][0].set_yticklabels(yticklabels)
            ax2[p][0].set_ylabel(f'LST (hours) | {pols[p]} pol')
            for i in [1,2,3,4]:
                ax2[p][i].set_yticks([])
                ax2[p][i].set_yticklabels([])
    length = int(np.around(getBaselineLength(uvtimes,[bl])[0],0))
    fig.suptitle(f'Baseline ({bl[0]},{bl[1]}), {length}m')
    fig.tight_layout()
    if plotFlags:
        fig2.tight_layout()
    if savefig:
        outfig = f'{outdir}/{str(length).zfill(3)}m_{bl[0]}_{bl[1]}.jpeg'
        outfig_f = f'{outdir}/withFlags_{str(length).zfill(3)}m_{bl[0]}_{bl[1]}.jpeg'
        fig.savefig(outfig,bbox_inches='tight')
        if plotFlags:
            fig.suptitle(f'Baseline ({bl[0]},{bl[1]}), {length}m')
            fig2.savefig(outfig_f,bbox_inches='tight')
        if write_params:
            curr_func = inspect.stack()[0][3]
            utils.write_params_to_text(outfig,args,curr_func=curr_func)
        plt.close()
    else:
        plt.show()
        plt.close()


status_colors = dict(
    dish_maintenance="salmon",
    dish_ok="red",
    RF_maintenance="lightskyblue",
    RF_ok="royalblue",
    digital_maintenance="plum",
    digital_ok="mediumpurple",
    calibration_maintenance="lightgreen",
    calibration_ok="green",
    calibration_triage="lime",
    not_connected="gray",
)
status_abbreviations = dict(
    dish_maintenance="dish-M",
    dish_ok="dish-OK",
    RF_maintenance="RF-M",
    RF_ok="RF-OK",
    digital_maintenance="dig-M",
    digital_ok="dig-OK",
    calibration_maintenance="cal-M",
    calibration_ok="cal-OK",
    calibration_triage="cal-Tri",
    not_connected="No-Con",
)

def plot_autos(
    uvd,
    wrongAnts=[],
    ylim=None,
    logscale=True,
    savefig=False,
    write_params=True,
    outfig="",
    plot_nodes='all',
    time_slice=False,
    slice_freq_inds=[],
    dtype="sky",
    freq_range='all',
    freq_range_type='index',
    exants=[],
):
    """

    Function to plot autospectra of all antennas, with a row for each node, sorted by SNAP and within that by SNAP
    input. Spectra are chosen from a time in the middle of the observation.

    Parameters:
    -----------
    uvd: UVData Object
        UVData object containing data to plot.
    wrongAnts: List
        Optional, list of antennas that are identified as observing the wrong datatype (seeing the sky when we are
        trying to observe load, for example) or are severely broken/dead. These antennas will be greyed out and
        outlined in red.
    ylim: List
        The boundaries of the y-axis, formatted as [minimum value, maximum value].
    logscale:
        Option to plot the data on a logarithmic scale. Default is True.
    savefig: Boolean
        Option to write out the figure.
    outfig: String
        Path to full figure name, required if savefig is True.

    Returns:
    --------
    None

    """
    from astropy.time import Time
    from hera_mc import cm_active
    import math
    from matplotlib import pyplot as plt
    import yaml

    if type(exants)==str:
        # If only one file is passed, assume it applies to both polarizations.
        exants = [exants,exants]
    xants = {}
    if len(exants)>0:
        for i,f in enumerate(exants):
            with open(f, 'r') as xfile:
                x = np.asarray(yaml.safe_load(xfile))
                print(x)
            xants[i] = x

    nodes, antDict, inclNodes = utils.generate_nodeDict(uvd)
    sorted_ants, sortedSnapLocs, sortedSnapInputs = utils.sort_antennas(uvd)
    freqs = (uvd.freq_array[0]) * 10 ** (-6)
    if freq_range != 'all':
        if freq_range_type=='mhz':
            i1 = np.argmin(np.abs(np.subtract(freqs,freq_range[0])))
            i2 = np.argmin(np.abs(np.subtract(freqs,freq_range[1])))
            freqs_use = uvd.freq_array[0][i1:i2]
        elif freq_range_type=='index':
            freqs_use = uvd.freq_array[0][freq_range[0]:freq_range[1]]
        uvd = uvd.select(frequencies=freqs_use, inplace=False)
        freqs = uvd.freq_array[0] * 1e-6
    times = uvd.time_array
    maxants = 0
    for node in nodes:
        if plot_nodes != 'all' and node not in plot_nodes:
            continue
        n = len(nodes[node]["ants"])
        if n > maxants:
            maxants = n

    Nside = maxants
    if plot_nodes == 'all':
        Yside = len(inclNodes)
    else:
        Yside = len(plot_nodes)

    t_index = len(np.unique(times))//2
    jd = times[t_index]
    utc = Time(jd, format="jd").datetime

    h = cm_active.get_active(at_date=jd, float_format="jd")

    xlim = (np.min(freqs), np.max(freqs))
    colorsx = ['gold','darkorange','orangered','red','maroon']
    colorsy = ['powderblue','deepskyblue','dodgerblue','royalblue','blue']

    if ylim is None:
        if dtype == "sky":
            ylim = [60, 80]
        elif dtype == "load":
            ylim = [55, 75]
        elif dtype == "noise":
            ylim = [75, 75.2]

    fig, axes = plt.subplots(Yside, 12, figsize=(16, Yside * 3))

    ptitle = 1.92 / (Yside * 3)
    fig.suptitle("JD = {0}, time = {1} UTC".format(jd, utc), fontsize=10, y=1 + ptitle)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=1, wspace=0.05, hspace=0.3)
    k = 0
    i=-1
    yrange_set = False
    for _, n in enumerate(inclNodes):
        slots_filled = []
        if plot_nodes != 'all':
            inclNodes = plot_nodes
            if n not in plot_nodes:
                continue
        i += 1
        ants = nodes[n]["ants"]
        j = 0
        for _, a in enumerate(sorted_ants):
            if a not in ants:
                continue
            status = utils.get_ant_status(h, a)
            slot = utils.get_slot_number(uvd, a, sorted_ants, sortedSnapLocs, sortedSnapInputs)
            slots_filled.append(slot)
            ax = axes[i, slot]
            if time_slice is True:
                colors = ['r','b']
                lsts = uvd.lst_array * 3.819719
                inds = np.unique(lsts, return_index=True)[1]
                lsts = np.asarray([lsts[ind] for ind in sorted(inds)])
#                 if a == sorted_ants[0]:
#                     print(lsts)
#                     print(len(lsts))
                dx = np.log10(np.abs(uvd.get_data((a,a,'xx'))))
                for s,ind in enumerate(slice_freq_inds):
                    dslice = dx[:,ind]
                    dslice = np.subtract(dslice,np.nanmean(dslice))
                    (px,) = ax.plot(
                        dslice,
                        color=colorsx[s],
                        alpha=1,
                        linewidth=1.2,
                        label=f'XX - {int(freqs[ind])} MHz'
                    )
                dy = np.log10(np.abs(uvd.get_data((a,a,'yy'))))
                for s,ind in enumerate(slice_freq_inds):
                    dslice = dy[:,ind]
                    dslice = np.subtract(dslice,np.nanmean(dslice))
#                     print(dslice)
                    (py,) = ax.plot(
                        dslice,
                        color=colorsy[s],
                        alpha=1,
                        linewidth=1.2,
                        label=f'YY - {int(freqs[ind])} MHz'
                    )
                if yrange_set is False:
                    xdiff = np.abs(np.subtract(np.nanmax(dx),np.nanmin(dx)))
                    ydiff = np.abs(np.subtract(np.nanmax(dy),np.nanmin(dy)))
                    yrange = np.nanmax([xdiff,ydiff])
                    if math.isinf(yrange) or math.isnan(yrange):
                        yrange = 10
                    else:
                        yrange_set = True
                if ylim is None:
                    ymin = np.nanmin([np.nanmin(dx),np.nanmin(dy)])
                    if math.isinf(ymin):
                        ymin=0
                    if math.isnan(ymin):
                        ymin=0
                    ylim = (ymin, ymin + yrange)
                    if yrange > 3*np.abs(np.subtract(np.nanmax(dx),np.nanmin(dx))) or yrange > 3*np.abs(np.subtract(np.nanmax(dy),np.nanmin(dy))):
                        xdiff = np.abs(np.subtract(np.nanmax(dx),np.nanmin(dx)))
                        ydiff = np.abs(np.subtract(np.nanmax(dy),np.nanmin(dy)))
                        yrange_temp = np.nanmax([xdiff,ydiff])
                        ylim = (ymin, ymin + yrange_temp)
                        ax.tick_params(color='red', labelcolor='red')
                        for spine in ax.spines.values():
                            spine.set_edgecolor('red')
            elif logscale is True:
                (px,) = ax.plot(
                    freqs,
                    10 * np.log10(np.abs(uvd.get_data((a, a, "xx"))[t_index])),
                    color="r",
                    alpha=0.75,
                    linewidth=1,
                )
                (py,) = ax.plot(
                    freqs,
                    10 * np.log10(np.abs(uvd.get_data((a, a, "yy"))[t_index])),
                    color="b",
                    alpha=0.75,
                    linewidth=1,
                )
            else:
                (px,) = ax.plot(
                    freqs,
                    np.abs(uvd.get_data((a, a, "xx"))[t_index]),
                    color="r",
                    alpha=0.75,
                    linewidth=1,
                )
                (py,) = ax.plot(
                    freqs,
                    np.abs(uvd.get_data((a, a, "yy"))[t_index]),
                    color="b",
                    alpha=0.75,
                    linewidth=1,
                )
            if time_slice is False:
                ax.set_xlim(xlim)
            else:
                xticks = np.asarray([int(i) for i in np.linspace(0, len(lsts) - 1, 3)])
                xticklabels = [int(l) for l in lsts[xticks]]
                ax.set_xticks(xticks)
                ax.set_xticklabels(xticklabels)
            ax.set_ylim(ylim)
            ax.grid(False, which="both")
            abb = status_abbreviations[status]
            if (len(xants) == 1 and a in exants[0]) or (len(xants)>1 and a in xants[0] and a in xants[1]):
                ax.set_title(f"{a} ({abb})", fontsize=10, backgroundcolor="purple")
            elif len(xants)>1 and a in xants[0]:
                ax.set_title(f"{a} ({abb})", fontsize=10, backgroundcolor="red")
            elif len(xants)>1 and a in xants[1]:
                ax.set_title(f"{a} ({abb})", fontsize=10, backgroundcolor="blue")
            elif xants=={}:
                ax.set_title(
                    f"{a} ({abb})", fontsize=10, backgroundcolor=status_colors[status]
                )
            else:
                ax.set_title(f"{a} ({abb})", fontsize=10, backgroundcolor="green")
            if k == 0:
                if time_slice:
                    ax.legend(loc='upper left',bbox_to_anchor=(0, 1.7),ncol=2)
                else:
                    ax.legend([px, py], ["NN", "EE"])
            if i == len(inclNodes) - 1:
                [t.set_fontsize(10) for t in ax.get_xticklabels()]
                if time_slice is True:
                    ax.set_xlabel("LST (hours)", fontsize=10)
                else:
                    ax.set_xlabel("freq (MHz)", fontsize=10)
            else:
                ax.set_xticklabels([])
            if j != 0:
                ax.set_yticklabels([])
            else:
                [t.set_fontsize(10) for t in ax.get_yticklabels()]
                ax.set_ylabel(r"$10\cdot\log$(amp)", fontsize=10)
            if a in wrongAnts:
                for axis in ["top", "bottom", "left", "right"]:
                    ax.spines[axis].set_linewidth(2)
                    ax.spines[axis].set_color("red")
                    ax.set_facecolor("black")
                    ax.patch.set_alpha(0.2)
            j += 1
            k += 1
        for k in range(0, 12):
            if k not in slots_filled:
                axes[i, k].axis("off")
        axes[i, maxants - 1].annotate(
            f"Node {n}", (1.1, 0.3), xycoords="axes fraction", rotation=270
        )
    if savefig is True:
        plt.savefig(outfig,bbox_inches='tight')
        if write_params:
            args = locals()
            curr_func = inspect.stack()[0][3]
            utils.write_params_to_text(outfig,args,curr_func,curr_file,githash)
        plt.show()
    else:
        plt.show()
        plt.close()


def plot_wfs(
    uvd,
    pol="NN",
    plotType="raw",
    savefig=False,
    write_params=True,
    vmin=None,
    vmax=None,
    wrongAnts=[],
    plot_nodes='all',
    logscale=True,
    uvd_diff=None,
    metric=None,
    outfig="",
    dtype="sky",
    _data_cleaned_sq="auto",
    freq_range='all',
    freq_range_type='index',
    lst_range='all',
    return_times=False,
    plotFlags=False,
    exants=[],
):
    """
    Function to plot auto waterfalls of all antennas, with a row for each node, sorted by SNAP and within that by
    SNAP input.

    Parameters:
    -----------
    uvd: UVData Object
        UVData object containing all sum data to plot.
    pol: String
        Polarization to plot. Can be any polarization string accepted by pyuvdata.
    plotType: String
        Option to specify what data to plot. Can be 'raw' (raw visibilities), 'mean_sub' (the average spectrum over the night is subtracted out), or 'delay' (delay spectra).
    savefig: Boolean
        Option to write out the figure
    vmin: float
        Colorbar minimum value. Set to None to use default values, which vary depending on dtype and plotType.
    vmax: float
        Colorbar maximum value. Set to None to use default values, which vary depending on dtype and plotType.
    wrongAnts: List
        Optional, list of antennas that are identified as observing the wrong datatype (seeing the sky when we are
        trying to observe load, for example) or are severely broken/dead. These antennas will be greyed out and
        outlined in red.
    logscale: Boolean
        Option to use a logarithmic colorbar.
    uvd_diff: UVData Object
        Diff data corresponding to the sum data in uvd. Required when metric is set.
    metric: String or None
        When metric is None the standard sum data is plot. Set metric to 'even' or 'odd' to plot those values instead.
        Providing uvd_diff is required when this parameter is used.
    outfig: String
        Path to write out the figure if savefig is True.
    dtype: String or None
        Can be 'sky', 'load', 'noise', or None. If set to 'load' or 'noise' the vmin and vmax parameters will be
        automatically changed to better suit those datatypes. If you want to manually set vmin and vmax, set this
        parameter to None.

    Returns:
    --------
    None

    """
    from hera_mc import cm_active
    import numpy.ma as ma
    import yaml

    if type(exants)==str:
        # If only one file is passed, assume it applies to both polarizations.
        exants = [exants,exants]
    xants = {}
    if len(exants)>0:
        for i,f in enumerate(exants):
            with open(f, 'r') as xfile:
                x = np.asarray(yaml.safe_load(xfile))
                print(x)
            xants[i] = x
    
    nodes, _, inclNodes = utils.generate_nodeDict(uvd)
    sorted_ants, sortedSnapLocs, sortedSnapInputs = utils.sort_antennas(uvd)
    freqs = (uvd.freq_array[0]) * 10 ** (-6)
    if freq_range != 'all':
        if freq_range_type=='mhz':
            i1 = np.argmin(np.abs(np.subtract(freqs,freq_range[0])))
            i2 = np.argmin(np.abs(np.subtract(freqs,freq_range[1])))
            freqs_use = uvd.freq_array[0][i1:i2]
        elif freq_range_type=='index':
            freqs_use = uvd.freq_array[0][freq_range[0]:freq_range[1]]
        uvd = uvd.select(frequencies=freqs_use, inplace=False)
        freqs = uvd.freq_array[0] * 1e-6
    times = uvd.time_array
    lsts = uvd.lst_array * 3.819719
    inds = np.unique(lsts, return_index=True)[1]
    lsts = [lsts[ind] for ind in sorted(inds)]
    if lst_range != 'all':
        i1 = np.argmin(np.abs(np.subtract(lsts,lst_range[0])))
        i2 = np.argmin(np.abs(np.subtract(lsts,lst_range[1])))
        lsts_use = lsts[i1:i2]
        times_use = np.unique(times)[i1:i2]
        uvd = uvd.select(times=times_use, inplace=False)
        times = uvd.time_array
        lsts = uvd.lst_array * 3.819719
        inds = np.unique(lsts, return_index=True)[1]
        lsts = [lsts[ind] for ind in sorted(inds)]
    maxants = 0
    if dtype == "sky":
        if plotType == "raw":
            vminAuto = 6.5
            vmaxAuto = 8
        elif plotType == "mean_sub":
            vminAuto = -0.07
            vmaxAuto = 0.07
        elif plotType == "delay":
            vminAuto = -50
            vmaxAuto = -30
    elif dtype == "load":
        if plotType == "raw":
            vminAuto = 5.5
            vmaxAuto = 7.5
        elif plotType == "mean_sub":
            vminAuto = -0.04
            vmaxAuto = 0.04
        elif plotType == "delay":
            vminAuto = -50
            vmaxAuto = -30
    elif dtype == "noise":
        if plotType == "raw":
            vminAuto = 7.5
            vmaxAuto = 7.52
        elif plotType == "mean_sub":
            vminAuto = -0.0005
            vmaxAuto = 0.0005
        elif plotType == "delay":
            vminAuto = -50
            vmaxAuto = -30
    else:
        print(
            "##################### dtype must be one of sky, load, or noise #####################"
        )
    if vmin is None:
        vmin = vminAuto
    if vmax is None:
        vmax = vmaxAuto

    for node in nodes:
        if plot_nodes != 'all' and node not in plot_nodes:
            continue
        n = len(nodes[node]["ants"])
        if n > maxants:
            maxants = n

    Nside = maxants
    if plot_nodes == 'all':
        Yside = len(inclNodes)
    else:
        Yside = len(plot_nodes)

    t_index = 0
    jd = times[t_index]

    h = cm_active.get_active(at_date=jd, float_format="jd")
    ptitle = 1.92 / (Yside * 3)
    fig, axes = plt.subplots(Yside, 12, figsize=(16, Yside * 3))
    fig.suptitle(f"{pol} Polarization", fontsize=14, y=1 + ptitle)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.subplots_adjust(left=0, bottom=0.1, right=0.9, top=1, wspace=0.1, hspace=0.3)
    i = -1
    for _, n in enumerate(inclNodes):
        slots_filled = []
        if plot_nodes != 'all':
            inclNodes = plot_nodes
            if n not in plot_nodes:
                continue
        i += 1
        ants = nodes[n]["ants"]
        j = 0
        for _, a in enumerate(sorted_ants):
            if a not in ants:
                continue
            status = utils.get_ant_status(h, a)
            slot = utils.get_slot_number(uvd, a, sorted_ants, sortedSnapLocs, sortedSnapInputs)
            slots_filled.append(slot)
            abb = status_abbreviations[status]
            ax = axes[i, slot]
            if metric is None:
                if logscale is True:
                    dat = np.log10(np.abs(uvd.get_data(a, a, pol)))
                else:
                    dat = np.abs(uvd.get_data(a, a, pol))
            else:
                dat_diff = uvd_diff.get_data(a, a, pol)
                dat = uvd.get_data(a, a, pol)
                if metric == "even":
                    dat = (dat + dat_diff) / 2
                elif metric == "odd":
                    dat = (dat - dat_diff) / 2
                if logscale is True:
                    dat = np.log10(np.abs(dat))
            if plotFlags:
                mask = uvd.get_flags(a,a,pol)
                dat = ma.masked_array(dat,mask=mask)
            if plotType == "mean_sub":
                ms = np.subtract(dat, np.nanmean(dat, axis=0))
                xticks = [int(i) for i in np.linspace(0, len(freqs) - 1, 3)]
                xticklabels = np.around(freqs[xticks], 0)
                xlabel = "Freq (MHz)"
                im = ax.imshow(
                    ms,
                    vmin=vmin,
                    vmax=vmax,
                    aspect="auto",
                    interpolation="nearest",
                )
            elif plotType == "delay":
                bls = bls = [(ant, ant) for ant in uvd.get_ants()]
                if _data_cleaned_sq == "auto":
                    _data_cleaned_sq, _, _ = utils.clean_ds(
                        bls, uvd, uvd_diff, N_threads=14
                    )
                key = (a, a, pol)
                norm = np.abs(_data_cleaned_sq[key]).max(axis=1)[:, np.newaxis]
                ds = 10.0 * np.log10(np.sqrt(np.abs(_data_cleaned_sq[key]) / norm))
                taus = (
                    np.fft.fftshift(np.fft.fftfreq(freqs.size, np.diff(freqs)[0])) * 1e3
                )
                xticks = [int(i) for i in np.linspace(0, len(taus) - 1, 4)]
                xticklabels = np.around(taus[xticks], 0)
                xlabel = "Tau (ns)"
                print(np.min(ds))
                print(np.max(ds))
                im = ax.imshow(
                    ds,
                    vmin=vmin,
                    vmax=vmax,
                    aspect="auto",
                    interpolation="nearest",
                )
            elif plotType == "raw":
                xticks = [int(i) for i in np.linspace(0, len(freqs) - 1, 3)]
                xticklabels = np.around(freqs[xticks], 0)
                xlabel = "Freq (MHz)"
                im = ax.imshow(
                    dat, vmin=vmin, vmax=vmax, aspect="auto", interpolation="nearest"
                )
            else:
                print(
                    "##### plotType parameter must be either raw, mean_sub, or delay #####"
                )
            if (len(xants) == 1 and a in exants[0]) or (len(xants)>1 and a in xants[0] and a in xants[1]):
                ax.set_title(f"{a} ({abb})", fontsize=10, backgroundcolor="purple")
            elif len(xants)>1 and a in xants[0]:
                ax.set_title(f"{a} ({abb})", fontsize=10, backgroundcolor="red")
            elif len(xants)>1 and a in xants[1]:
                ax.set_title(f"{a} ({abb})", fontsize=10, backgroundcolor="blue")
            elif xants=={}:
                ax.set_title(
                    f"{a} ({abb})", fontsize=10, backgroundcolor=status_colors[status]
                )
            else:
                ax.set_title(f"{a} ({abb})", fontsize=10, backgroundcolor="green")
            if i == len(inclNodes) - 1:
                ax.set_xticks(xticks)
                ax.set_xticklabels(xticklabels)
                ax.set_xlabel(xlabel, fontsize=10)
                [t.set_rotation(70) for t in ax.get_xticklabels()]
            else:
                ax.set_xticklabels([])
            if j != 0 or slot!=0:
                ax.set_yticklabels([])
            else:
                yticks = [int(i) for i in np.linspace(0, len(lsts) - 1, 6)]
                yticklabels = [np.around(lsts[ytick], 1) for ytick in yticks]
                [t.set_fontsize(12) for t in ax.get_yticklabels()]
                ax.set_yticks(yticks)
                ax.set_yticklabels(yticklabels)
                ax.set_ylabel("Time(LST)", fontsize=10)
            if a in wrongAnts:
                for axis in ["top", "bottom", "left", "right"]:
                    ax.spines[axis].set_linewidth(2)
                    ax.spines[axis].set_color("red")
            if j==0 and slot !=0:
                ax = axes[i, 0]
                yticks = [int(i) for i in np.linspace(0, len(lsts) - 1, 6)]
                yticklabels = [np.around(lsts[ytick], 1) for ytick in yticks]
                [t.set_fontsize(12) for t in ax.get_yticklabels()]
                ax.set_ylabel("Time(LST)", fontsize=10)
                ax.set_yticks([])
                ax.set_yticks(yticks)
                ax.set_yticklabels(yticklabels)
                ax.set_frame_on(False)
                ax.axes.get_xaxis().set_visible(False)
            j += 1
        for k in range(1, 12):
            if k not in slots_filled:
                axes[i, k].axis("off")
        pos = ax.get_position()
        cbar_ax = fig.add_axes([0.91, pos.y0, 0.01, pos.height])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label(f"Node {n}", rotation=270, labelpad=15)
    if savefig is True:
        plt.savefig(outfig, bbox_inches="tight", dpi=100)
        if write_params:
            args = locals()
            curr_func = inspect.stack()[0][3]
            utils.write_params_to_text(outfig,args,curr_func,curr_file,githash)
    plt.show()
    plt.close()
    if return_times:
        return times


def plot_gain_wfs(
    datadir,
    pol="XX",
    plotType="raw",
    savefig=False,
    write_params=True,
    vmin=0,
    vmax=100,
    wrongAnts=[],
    plot_nodes='all',
    logscale=False,
    uvd_diff=None,
    outfig="",
    dtype="sky",
    _data_cleaned_sq="auto",
    freq_range='all',
    freq_range_type='index',
    lst_range='all',
    return_times=False,
    plotFlags=False,
    file_range='all',
    exants=[],
):
    """
    Function to plot auto waterfalls of all antennas, with a row for each node, sorted by SNAP and within that by
    SNAP input.

    Parameters:
    -----------
    uvd: UVData Object
        UVData object containing all sum data to plot.
    pol: String
        Polarization to plot. Can be any polarization string accepted by pyuvdata.
    plotType: String
        Option to specify what data to plot. Can be 'raw' (raw visibilities), 'mean_sub' (the average spectrum over the night is subtracted out), or 'delay' (delay spectra).
    savefig: Boolean
        Option to write out the figure
    vmin: float
        Colorbar minimum value. Set to None to use default values, which vary depending on dtype and plotType.
    vmax: float
        Colorbar maximum value. Set to None to use default values, which vary depending on dtype and plotType.
    wrongAnts: List
        Optional, list of antennas that are identified as observing the wrong datatype (seeing the sky when we are
        trying to observe load, for example) or are severely broken/dead. These antennas will be greyed out and
        outlined in red.
    logscale: Boolean
        Option to use a logarithmic colorbar.
    uvd_diff: UVData Object
        Diff data corresponding to the sum data in uvd. Required when metric is set.
    metric: String or None
        When metric is None the standard sum data is plot. Set metric to 'even' or 'odd' to plot those values instead.
        Providing uvd_diff is required when this parameter is used.
    outfig: String
        Path to write out the figure if savefig is True.
    dtype: String or None
        Can be 'sky', 'load', 'noise', or None. If set to 'load' or 'noise' the vmin and vmax parameters will be
        automatically changed to better suit those datatypes. If you want to manually set vmin and vmax, set this
        parameter to None.

    Returns:
    --------
    None

    """
    from hera_mc import cm_active
    import numpy.ma as ma
    import yaml

    if type(exants)==str:
        # If only one file is passed, assume it applies to both polarizations.
        exants = [exants,exants]
    xants = {}
    if len(exants)>0:
        for i,f in enumerate(exants):
            with open(f, 'r') as xfile:
                x = np.asarray(yaml.safe_load(xfile))
            xants[i] = x
    print(len(xants))
    print(xants)
    
    
    calfiles = sorted(glob.glob(f'{datadir}/*/calibration/*cal.sav'))
    dirnames = sorted(glob.glob(f'{datadir}/fhd_*'))
    file_inds = [int(c.split('_')[-1]) for c in dirnames]
    
    print(f'Found {len(calfiles)} cal files ranging from index {file_inds[0]} to {file_inds[-1]}')
    _,uvd,_,_ = djs_utils.read_fhd(datadir,pol=['XX','YY'],readModel=False,readGains=False,readRaw=False,file_range=file_inds[0])
    jd = dirnames[0].split('/')[-1][4:11]
    print(f'Plotting for jd {jd}')
    if file_range != 'all':
        print('Clipping file range')
        ind0 = np.argmin(np.abs(np.subtract(file_inds,file_range[0])))
        ind1 = np.argmin(np.abs(np.subtract(file_inds,file_range[1])))
        # print(np.subtract(file_inds,file_range[1]))
        # print(f'({ind0},{ind1})')
        calfiles=calfiles[ind0:ind1]
        file_inds=file_inds[ind0:ind1]
    nodes, _, inclNodes = utils.generate_nodeDict(uvd)
    sorted_ants, sortedSnapLocs, sortedSnapInputs = utils.sort_antennas(uvd)
    freqs = (uvd.freq_array[0]) * 10 ** (-6)
    if freq_range != 'all':
        if freq_range_type=='mhz':
            i1 = np.argmin(np.abs(np.subtract(freqs,freq_range[0])))
            i2 = np.argmin(np.abs(np.subtract(freqs,freq_range[1])))
            freqs_use = uvd.freq_array[0][i1:i2]
        elif freq_range_type=='index':
            freqs_use = uvd.freq_array[0][freq_range[0]:freq_range[1]]
        uvd = uvd.select(frequencies=freqs_use, inplace=False)
        freqs = uvd.freq_array[0] * 1e-6
    maxants = 0
    if pol=='XX':
        polname='NN'
    elif pol=='YY':
        polname='EE'
    else:
        polname=''

    for node in nodes:
        if plot_nodes != 'all' and node not in plot_nodes:
            continue
        n = len(nodes[node]["ants"])
        if n > maxants:
            maxants = n
    Nside = maxants
    if plot_nodes == 'all':
        Yside = len(inclNodes)
    else:
        Yside = len(plot_nodes)
    h = cm_active.get_active(at_date=jd, float_format="jd")
    ptitle = 1.92 / (Yside * 3)
    fig, axes = plt.subplots(Yside, 12, figsize=(16, Yside * 3))
    fig.suptitle(f"FHD Gains: {polname} Polarization", fontsize=14, y=1 + ptitle)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.subplots_adjust(left=0, bottom=0.1, right=0.9, top=1, wspace=0.1, hspace=0.3)
    _,_,_,gains0 = djs_utils.read_fhd(datadir,pol=['XX','YY'],readModel=False,readGains=True,readRaw=False,readCal=False,file_range=file_inds[0])
    d0 = gains0.get_gains(sorted_ants[0],pol)
    lsts=[]
    for t in file_inds:
        t_ind = t - file_inds[0]
        if t%20==0:
            print(f't={t}')
        i=-1
        try:
            _,_,_,uvg = djs_utils.read_fhd(datadir,pol=['XX','YY'],readModel=False,readGains=True,readRaw=False,readCal=False,file_range=t)
            # return uvg
        except:
            print(f'oops at index {t}')
            dat = np.zeros(np.shape(d0))
        # print(cal.lst_array)
        lsts.append(uvg.lst_range[0][0]*3.819719)
        for _, n in enumerate(inclNodes):
            slots_filled = []
            if plot_nodes != 'all':
                inclNodes = plot_nodes
                if n not in plot_nodes:
                    continue
            i += 1
            # print(f'i: {i}, n: {n}')
            ants = nodes[n]["ants"]
            j = 0
            for _, a in enumerate(sorted_ants):
                if a not in ants:
                    continue
                # print(a)
                status = utils.get_ant_status(h, a)
                slot = utils.get_slot_number(uvd, a, sorted_ants, sortedSnapLocs, sortedSnapInputs)
                if slot > 12:
                    print(f'ant {a}, slot {slot}')
                slots_filled.append(slot)
                abb = status_abbreviations[status]
                ax = axes[i, slot]
                try:
                    if logscale is True:
                        dat = np.transpose(np.log10(np.abs(uvg.get_gains(a, pol))))
                    else:
                        dat = np.transpose(np.abs(uvg.get_gains(a, pol)))
                    if plotFlags:
                        if t==file_inds[0]:
                            print('Plotting flags')
                        mask = np.transpose(uvg.get_flags(a,pol))
                        dat = ma.masked_array(dat,mask=mask)
                except:
                    print(f'No gains found for index {t}')
                    dat = np.zeros(np.shape(dat))
                if plotType == "mean_sub":
                    ms = np.subtract(dat, np.nanmean(dat, axis=0))
                    xticks = [int(i) for i in np.linspace(0, len(freqs) - 1, 3)]
                    xticklabels = np.around(freqs[xticks], 0)
                    xlabel = "Freq (MHz)"
                    im = ax.imshow(
                        ms,
                        vmin=vmin,
                        vmax=vmax,
                        aspect="auto",
                        interpolation="nearest",
                        extent=[0,245,t_ind,t_ind-1],
                        origin='upper'
                    )
                elif plotType == "raw":
                    xticks = [int(i) for i in np.linspace(0, len(freqs) - 1, 3)]
                    xticklabels = np.around(freqs[xticks], 0)
                    xlabel = "Freq (MHz)"
                    im = ax.imshow(
                        dat, vmin=vmin, vmax=vmax, aspect="auto", interpolation="nearest",
                        extent=[0,245,t_ind-1,t_ind],origin='upper'
                    )
                else:
                    print(
                        "##### plotType parameter must be either raw or mean_sub #####"
                    )
                if (len(xants) == 1 and a in exants[0]) or (len(xants)>1 and a in xants[0] and a in xants[1]):
                    ax.set_title(f"{a} ({abb})", fontsize=10, backgroundcolor="purple")
                elif len(xants)>1 and a in xants[0]:
                    ax.set_title(f"{a} ({abb})", fontsize=10, backgroundcolor="red")
                elif len(xants)>1 and a in xants[1]:
                    ax.set_title(f"{a} ({abb})", fontsize=10, backgroundcolor="blue")
                elif xants=={}:
                    ax.set_title(
                        f"{a} ({abb})", fontsize=10, backgroundcolor=status_colors[status]
                    )
                else:
                    ax.set_title(f"{a} ({abb})", fontsize=10, backgroundcolor="green")
                ax.set_ylim(len(file_inds),0)
                if i == len(inclNodes) - 1:
                    ax.set_xticks(xticks)
                    ax.set_xticklabels(xticklabels)
                    ax.set_xlabel(xlabel, fontsize=10)
                    [t.set_rotation(70) for t in ax.get_xticklabels()]
                else:
                    ax.set_xticklabels([])
                if j != 0 or slot!=0:
                    ax.set_yticklabels([])
                if t==file_inds[-1] and (j==0 and slot==0):
                    yticks = [int(i) for i in np.linspace(0, len(file_inds)-1, 6)]
                    yticklabels = [np.around(lsts[ytick], 1) for ytick in yticks]
                    [t.set_fontsize(12) for t in ax.get_yticklabels()]
                    ax.set_yticks(yticks)
                    ax.set_yticklabels(yticklabels)
                    ax.set_ylabel("Time(LST)", fontsize=10)
                elif j==0 and slot !=0 and t==file_inds[-1]:
                    ax = axes[i, 0]
                    yticks = [int(i) for i in np.linspace(0, len(file_inds)-1, 6)]
                    yticklabels = [np.around(lsts[ytick], 1) for ytick in yticks]
                    [t.set_fontsize(12) for t in ax.get_yticklabels()]
                    ax.set_ylabel("Time(LST)", fontsize=10)
                    ax.set_yticks(yticks)
                    ax.set_yticklabels(yticklabels)
                    ax.set_frame_on(False)
                    ax.axes.get_xaxis().set_visible(False)
                else:
                    yticks = [int(i) for i in np.linspace(0, len(file_inds)-1, 6)]
                    ax.set_yticks(yticks)
                if a in wrongAnts:
                    for axis in ["top", "bottom", "left", "right"]:
                        ax.spines[axis].set_linewidth(2)
                        ax.spines[axis].set_color("red")
                # ax.invert_yaxis()
                
                j += 1
            for k in range(1, 12):
                if k not in slots_filled:
                    axes[i, k].axis("off")
            if t==file_inds[-1]:
                pos = ax.get_position()
                cbar_ax = fig.add_axes([0.91, pos.y0, 0.01, pos.height])
                cbar = fig.colorbar(im, cax=cbar_ax)
                cbar.set_label(f"Node {n}", rotation=270, labelpad=15)
    if savefig is True:
        plt.savefig(outfig, bbox_inches="tight", dpi=100)
        if write_params:
            args = locals()
            curr_func = inspect.stack()[0][3]
            utils.write_params_to_text(outfig,args,curr_func,curr_file,githash)
    plt.show()
    plt.close()
    if return_times:
        return times
