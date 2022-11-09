"""Licensed under the MIT License"""
"""Written by Dara Storer"""

import numpy as np
from pyuvdata import UVData, UVFlag
from matplotlib import pyplot as plt
import os
import inspect
from hera_commissioning_tools import utils

dirpath = os.path.dirname(os.path.realpath(__file__))
githash = utils.get_git_revision_hash(dirpath)
curr_file = os.path.dirname(os.path.abspath(__file__))

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
