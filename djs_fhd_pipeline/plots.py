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

def plotCalVisAllBls(uv,datdir,jd=2459855,tmin=0,tmax=600,jd_label=False,savefig=False,outfig=''):
    model = np.load(f'{datdir}/{jd}_fhd_model_data.npy')
    cal = np.load(f'{datdir}/{jd}_fhd_calibrated_data.npy')
    raw = np.load(f'{datdir}/{jd}_fhd_raw_data.npy')
    f = UVFlag()
    f.read(f'{datdir}/{jd}_ssins_flags.hdf5')
    flags = f.flag_array
    bls = np.load(f'{datdir}/{jd}_bl_array.npy')
    jds = np.load(f'{datdir}/{jd}_jd_array.npy')
    lsts = np.load(f'{datdir}/{jd}_lst_array.npy')
    freqs = uv.freq_array[0]*1e-6

    nbls = len(bls)
    fig = plt.figure(figsize=(16,8*nbls))
    gs = fig.add_gridspec(nbls,4,width_ratios=[1,1,1,0.1])
    lengths = getBaselineLength(uv,bls)
    maxlength = np.max(lengths)
    for i,bl in enumerate(bls):
        ant1 = bl[0]
        ant2 = bl[1]
        ind = get_ind(bls,ant1,ant2)
        length = lengths[i]

        xticks = [int(i) for i in np.linspace(0, len(freqs) - 1, 5)]
        xticklabels = [int(freqs[x]) for x in xticks]
        yticks = [int(i) for i in np.linspace(0, len(lsts[tmin:tmax]) - 1, 6)]
        yticklabels1 = [np.around(lsts[tmin:tmax][ytick], 1) for ytick in yticks]
        yticklabels2 = [str(jds[tmin:tmax][ytick])[4:11] for ytick in yticks]
        mask = np.ma.masked_where(flags==False, flags)[tmin:tmax,:,0]

        ax = fig.add_subplot(gs[i,0])
        im = ax.imshow(abs(model[tmin:tmax,ind,:]), aspect='auto',interpolation='nearest')
        im = ax.imshow(mask, 'binary', interpolation='none',alpha=1,aspect='auto')
        ax.set_title('Model Vis',fontsize=20)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels1,fontsize=12)
        ax.set_ylabel('LST',fontsize=15)

        ax = fig.add_subplot(gs[i,1])
        im = ax.imshow(abs(cal[tmin:tmax,ind,:]), aspect='auto',vmin=1,vmax=15,interpolation='nearest')
        im = ax.imshow(mask, 'binary', interpolation='none',alpha=1,aspect='auto')
        ax.set_title('Calibrated Vis',fontsize=20)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
        ax.set_yticks([])

        ax = fig.add_subplot(gs[i,2])
        r = abs(raw[tmin:tmax,ind,:])
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
        plt.savefig(outfig,bbox_inches='tight')
        if write_params:
            args = locals()
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
