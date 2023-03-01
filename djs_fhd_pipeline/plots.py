"""Licensed under the MIT License"""
"""Written by Dara Storer"""

import numpy as np
from pyuvdata import UVData, UVFlag
from matplotlib import pyplot as plt
import os
import inspect
from hera_commissioning_tools import utils
import glob
import json

dirpath = os.path.dirname(os.path.realpath(__file__))
githash = utils.get_git_revision_hash(dirpath)
curr_file = __file__

import json

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
            
def plotBlLengthHists(uv,use_ants=[],freq=170,bl_cut=25,nbins=10,savefig=False,outfig='',write_params=True):
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
    args = locals()
    if len(use_ants)>0:
        uv.select(use_ants=use_ants)
    baseline_groups, vec_bin_centers, lengths = uv.get_redundancies(
        use_antpos=False, include_autos=False
    )
    antpairs, lengths = plots.unpackBlLengths(uv, baseline_groups,lengths)
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
    
    fig = plt.figure(figsize=(8,6))
    plt.hist(lengths,bins=nbins,histtype='step')
    plt.xlabel('Baseline Length(m)')
    plt.ylabel('Count')
    plt.xlim((0,max(lengths)+20))
    colors=['black','red','cyan','green','gold']
    for i,cut in enumerate(bl_cut_m):
        plt.axvline(cut,linestyle='--',label=f'{np.round(bl_cut[i],1)} lambda cut',color=colors[i])
    plt.legend()
    if savefig:
        plt.savefig(outfig)
        if write_params:
            curr_func = inspect.stack()[0][3]
            utils.write_params_to_text(outfig,args,curr_func,curr_file,githash)
            
def plotBlLengthHists_perAnt(uv,use_ants=[],freq=170,bl_cut=25,nbins=10,savefig=False,outfig='',
                             write_params=True,suptitle=''):
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
    suptitle: String
    """
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
    antpairs, lengths = plots.unpackBlLengths(uv, baseline_groups,lengths)
    if freq < 1000:
        freq = freq*1e6
    wl = scipy.constants.speed_of_light/freq
    bl_cut_m=np.asarray(bl_cut)*wl
    print(f'Baseline cut of {bl_cut} lambda is {np.round(bl_cut_m,1)} meters at {int(freq*1e-6)}MHz')
    for i,ant in enumerate(uv.get_ants()):
        bls = [(bl[0],bl[1]) for bl in antpairs if (bl[0]==ant or bl[1]==ant)]
        lens = np.asarray(plots.getBaselineLength(uv,bls))
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
            ax[i].set_xlim((0,np.max(counts)+2))
            ax[i].set_title(f'{np.round(cut,1)}m/{bl_cut[i]}lambda baseline cut')
            ax[i].set_ylim((0,ymax))
    plt.suptitle(suptitle)
    plt.tight_layout()
#     plt.axvline(bl_cut_m,linestyle='--',color='k')
    if savefig:
        plt.savefig(outfig)
        if write_params:
            curr_func = inspect.stack()[0][3]
            utils.write_params_to_text(outfig,args,curr_func,curr_file,githash)
    
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
