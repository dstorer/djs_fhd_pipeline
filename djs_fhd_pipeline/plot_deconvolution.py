import numpy as np
from pyuvdata import UVData, UVFlag, UVCal
from matplotlib import pyplot as plt
import matplotlib.mlab as mlab
import matplotlib
import os
import inspect
from hera_commissioning_tools import utils
from scipy.spatial import distance
from astropy.io.votable import parse
from scipy.stats import norm
from djs_fhd_pipeline import plot_fits

dirpath = os.path.dirname(os.path.realpath(__file__))
githash = utils.get_git_revision_hash(dirpath)
curr_file = __file__

def read_catalogs(fhd_path, gleam_path):
    votable = parse(gleam_path)
    table = list(votable.iter_tables())[0]
    gleam = table.to_table()
    fhd = np.loadtxt(fhd_path)[1:, :]
    return fhd, gleam

def restrictGleam(gleam_ras, gleam_decs, gleam_flux, ra_range, dec_range):
    inds = np.ones(np.shape(gleam_ras))
    inds[gleam_ras<ra_range[0]] = 0
    inds[gleam_ras>ra_range[1]] = 0
    inds[gleam_decs>dec_range[1]] = 0
    inds[gleam_decs<dec_range[0]] = 0
    inds[gleam_flux<0.001] = 0
    inds = np.asarray(inds)
    inds = inds.astype(bool)
    gleam_ras = gleam_ras[inds]
    gleam_decs = gleam_decs[inds]
    gleam_flux = gleam_flux[inds]
    return gleam_ras, gleam_decs, gleam_flux

def getCols(fhd, gleam, sortBy=None,flux_type='stokes',beamCut=0):
    gleam_ras = gleam['RAJ2000']
    gleam_decs = gleam['DEJ2000']
    gleam_flux = gleam['Fint166']
    fhd_ras = fhd[:,3]
    fhd_decs = fhd[:,4]
    fhd_beam = fhd[:,7]
    if flux_type == 'stokes':
        fhd_flux = fhd[:,13]
    else:
        fhd_flux = fhd[:,9]
    ra_range = [np.min(fhd_ras),np.max(fhd_ras)]
    dec_range = [np.min(fhd_decs),np.max(fhd_decs)]
    gleam_ras, gleam_decs, gleam_flux = restrictGleam(gleam_ras, gleam_decs, gleam_flux,
                                                     ra_range, dec_range)
    if beamCut>0:
        beam_inds = fhd_beam>beamCut
        fhd_flux = fhd_flux[beam_inds]
        fhd_ras = fhd_ras[beam_inds]
        fhd_decs = fhd_decs[beam_inds]
        fhd_beam = fhd_beam[beam_inds]
        ra_range = [np.min(fhd_ras),np.max(fhd_ras)]
        dec_range = [np.min(fhd_decs),np.max(fhd_decs)]
        gleam_ras, gleam_decs, gleam_flux = restrictGleam(gleam_ras, gleam_decs, gleam_flux,
                                                     ra_range, dec_range)
    if sortBy == 'flux':
        inds = (-fhd_flux).argsort()
        ginds = (-gleam_flux).argsort()
    elif sortBy == 'ra':
        inds = fhd_ras.argsort()
        ginds = gleam_ras.argsort()
    elif sortBy == 'dec':
        inds = fhd_decs.argsort()
        ginds = gleam_decs.argsort()
    else:
        inds = np.arange(0,len(fhd_ras))
        ginds = np.arange(0,len(gleam_ras))
    flux = fhd_flux[inds]
    ra = fhd_ras[inds]
    dec = fhd_decs[inds]
    fhd_beam = fhd_beam[inds]
    gra = gleam_ras[ginds]
    gdec = gleam_decs[ginds]
    gflux = gleam_flux[ginds]

    return ra, dec, flux, gra, gdec, gflux

def plotFhdGleamRatioMap(fhd,gleam,delta=0.15,sortBy=None,ratioThresh=5,singleSourceThresh=0.1,
                     simple_pairs=True,complex_pairs=True,unmatched=False,combine_fhd=False,
                     logscale=False,savefig=False,outfig='',fluxCombType=np.sum, write_params=True,
                     beamCut=0,cmap='plasma_r',vmin=0,vmax=2,edgecolor='face'):
    args = locals()
    import matplotlib
    ra, dec, flux, gra, gdec, gflux = getCols(fhd,gleam,sortBy=sortBy,flux_type='stokes',beamCut=beamCut)
    spairs, cpairs, apairs, sinds, cinds, noinds = find_sources(fhd, gleam, sortBy=sortBy,
                                                       ratioThresh=ratioThresh,delta=delta,
                                                       singleSourceThresh=singleSourceThresh,
                                                       combine_fhd=combine_fhd,fluxCombType=fluxCombType,
                                                       beamCut=beamCut)
    fig = plt.figure(figsize=(16,10))
    if simple_pairs:
        fhd_srcs = [x[0][0:3] for x in spairs]
        fhd_x = [x[0] for x in fhd_srcs]
        fhd_y = [x[1] for x in fhd_srcs]
        pairs = [(np.average(x[1:],axis=0)[0],np.average(x[1:],axis=0)[1]) for x in spairs]
        fhd_f = [x[2] for x in fhd_srcs]
        gleam_f = [np.average(x[1:],axis=0)[2] for x in spairs]
        rats = np.divide(fhd_f,gleam_f)
        if logscale:
            im = plt.scatter(fhd_x,fhd_y,c=rats,cmap=cmap,marker='o',
                        norm=matplotlib.colors.LogNorm(vmin=10e-3,vmax=1.1),edgecolor=edgecolor)
        else:
            im = plt.scatter(fhd_x,fhd_y,c=rats,cmap=cmap,marker='o',
                        vmin=vmin,vmax=vmax,edgecolor=edgecolor)
    if complex_pairs:
        fhd_srcs = [x[0][0:3] for x in cpairs]
        fhd_x = [x[0] for x in fhd_srcs]
        fhd_y = [x[1] for x in fhd_srcs]
        pairs = [(np.average(x[1:],axis=0)[0],np.average(x[1:],axis=0)[1]) for x in cpairs]
        fhd_f = [x[2] for x in fhd_srcs]
        gleam_f = [fluxCombType(x[1:],axis=0)[2] for x in cpairs]
        rats = np.divide(fhd_f,gleam_f)
        if logscale:
            im = plt.scatter(fhd_x,fhd_y,c=rats,cmap=cmap,marker='s',
                        norm=matplotlib.colors.LogNorm(vmin=10e-3,vmax=1.1),edgecolor=edgecolor)
        else:
            im = plt.scatter(fhd_x,fhd_y,c=rats,cmap=cmap,marker='s',
                        vmin=vmin,vmax=vmax,edgecolor=edgecolor)
    plt.colorbar(im)
    plt.gca().set_aspect('equal')
    plt.xlabel('RA (deg)')
    plt.ylabel('DEC (deg)')
    plt.title('FHD flux / GLEAM flux')
    if savefig:
        plt.savefig(outfig)
        if write_params:
            curr_func = inspect.stack()[0][3]
            utils.write_params_to_text(outfig,args,curr_file=curr_file,
                                       curr_func=curr_func)

def plotSourceCompMap(fhd,gleam,delta=0.15,sortBy=None,ratioThresh=5,singleSourceThresh=0.1,

                     simple_pairs=True,complex_pairs=True,unmatched=False,combine_fhd=False,
                     savefig=False,write_params=True,outfig=''):
    args=locals()
    ra, dec, flux, gra, gdec, gflux = getCols(fhd,gleam,sortBy=sortBy)
    spairs, cpairs, apairs, sinds, cinds, noinds = find_sources(fhd, gleam, sortBy=sortBy,
                                                       ratioThresh=ratioThresh,delta=delta,
                                                       singleSourceThresh=singleSourceThresh,
                                                       combine_fhd=combine_fhd)
    fig = plt.figure(figsize=(14,14))
    
    if simple_pairs:
        print(f'Plotting {len(spairs)} simple pairs')
        fhd_srcs = [x[0][0:2] for x in spairs]
        fhd_x = [x[0] for x in fhd_srcs]
        fhd_y = [x[1] for x in fhd_srcs]
        pairs = [(np.average(x[1:],axis=0)[0],np.average(x[1:],axis=0)[1]) for x in spairs]
        diffs = np.subtract(fhd_srcs,pairs)
        plt.scatter(fhd_x,fhd_y,color='green',marker='o',s=20)
        if complex_pairs:
            plt.quiver(fhd_x,fhd_y,diffs[:,0],diffs[:,1],color='green',angles='xy',scale=0.2,scale_units='xy')
        else:
            plt.quiver(fhd_x,fhd_y,diffs[:,0],diffs[:,1],color='green',angles='xy',scale=0.1,scale_units='xy')
    
    if complex_pairs:
        print(f'Plotting {len(cpairs)} complex pairs')
        fhd_srcs = [x[0][0:2] for x in cpairs]
        fhd_x = [x[0] for x in fhd_srcs]
        fhd_y = [x[1] for x in fhd_srcs]
        pairs = [(np.average(x[1:],axis=0)[0],np.average(x[1:],axis=0)[1]) for x in cpairs]
        diffs = np.subtract(fhd_srcs,pairs)
        plt.scatter(fhd_x,fhd_y,color='blue',marker='o',s=20)
        plt.quiver(fhd_x,fhd_y,diffs[:,0],diffs[:,1],color='blue',angles='xy',scale=0.2,scale_units='xy')
    
    if unmatched:
        unmatched_srcs = np.asarray(apairs)[noinds]
        print(f'Plotting {len(unmatched_srcs)} unmatched sources')
        for x in unmatched_srcs:
            plt.scatter(x[0][0],x[0][1],color='k',marker='o',s=20)

    plt.gca().set_aspect('equal')
    plt.xlabel('RA (deg)')
    plt.ylabel('DEC (deg)')
    if savefig:
        plt.savefig(outfig)
        if write_params:
            curr_func = inspect.stack()[0][3]
            utils.write_params_to_text(outfig,args,curr_file=curr_file,
                                       curr_func=curr_func)


def plotFractionGleamFound(fhd,gleam,delta=0.15,sortBy=None,ratioThresh=5,singleSourceThresh=0.1,
                     combine_fhd=False,logscale=True,savefig=False,outfig='',
                     write_params=True,simple_pairs=True,complex_pairs=False,fluxCombType=np.sum,
                      density=False, nbins=20, beamCut=0):
    args = locals()
    ra, dec, flux, gra, gdec, gflux = getCols(fhd,gleam,sortBy=sortBy,flux_type='stokes',beamCut=beamCut)
    spairs, cpairs, apairs, sinds, cinds, noinds = find_sources(fhd, gleam, sortBy=sortBy,
                                                       ratioThresh=ratioThresh,delta=delta,
                                                       singleSourceThresh=singleSourceThresh,
                                                       combine_fhd=combine_fhd,fluxCombType=fluxCombType,
                                                       beamCut=beamCut)
    inds = get_unmatched_gleam(apairs,fhd,gleam,sortBy=sortBy,beamCut=beamCut)
    inds = np.array(inds,dtype=bool)
    gfound = gflux[inds]
    
    fig = plt.figure(figsize=(10,12))
    bins = np.linspace(0.01,18,nbins)
    if logscale:
        bins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
    hist1, bins1 = np.histogram(gflux, bins=bins)
    hist2, bins2 = np.histogram(gfound, bins=bins)
    centers = (bins[:-1] + bins[1:]) / 2
    ratio = np.divide(hist2,hist1)
    
    gs = fig.add_gridspec(2,1,height_ratios=[2,1],hspace=0)
    ax = fig.add_subplot(gs[0, 0])
    
    width = 0.7 * (bins[1] - bins[0])
    ax.hist(gflux,bins=bins,alpha=0.5,fill=False,edgecolor='red',linewidth=2,label='all GLEAM')
    ax.hist(gfound,bins=bins,alpha=0.5,fill=False,edgecolor='blue',linewidth=2, label='Found by FHD')
    ax.set_xscale('log')
    ax.set_xticks([])
    ax.legend(loc='upper right')
    ax.set_ylabel('Count')
    l,r = ax.get_xlim()
    ax.set_title('Fraction GLEAM sources found by FHD')
    
    ax = fig.add_subplot(gs[1, 0])
    ax.scatter(centers,ratio)
    ax.set_xscale('log')
    ax.set_xlabel('Flux')
    ax.set_ylabel('Fraction Found')
    ax.set_ylim(0,1)
    ax.set_xlim(l,r)

    if savefig:
        plt.savefig(outfig)
        if write_params:
            curr_func = inspect.stack()[0][3]
            utils.write_params_to_text(outfig,args,curr_file=curr_file,
                                       curr_func=curr_func)


def plotFhdVsGleamFlux(fhd,gleam,delta=0.15,sortBy=None,ratioThresh=5,singleSourceThresh=0.1,
                     combine_fhd=False,logscale=True,savefig=False,outfig='',
                     write_params=True,simple_pairs=True,complex_pairs=False,fluxCombType=np.sum,
                     beamCut=0):
    args = locals()
    ra, dec, flux, gra, gdec, gflux = getCols(fhd,gleam,sortBy=sortBy,flux_type='stokes',beamCut=beamCut)
    spairs, cpairs, apairs, sinds, cinds, noinds = find_sources(fhd, gleam, sortBy=sortBy,
                                                       ratioThresh=ratioThresh,delta=delta,
                                                       singleSourceThresh=singleSourceThresh,
                                                       combine_fhd=combine_fhd,fluxCombType=fluxCombType,
                                                       beamCut=beamCut)
    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(1,1,1)
    
    if simple_pairs:
        fhd_srcs = [x[0][0:3] for x in spairs]
        fhd_f = [x[2] for x in fhd_srcs]
        gleam_f = [fluxCombType(x[1:],axis=0)[2] for x in spairs]
        plt.scatter(fhd_f,gleam_f,label='simple pairs',color='blue')
    if complex_pairs:
        fhd_srcs = [x[0][0:3] for x in cpairs]
        fhd_f = [x[2] for x in fhd_srcs]
        gleam_f = [fluxCombType(x[1:],axis=0)[2] for x in cpairs]
        plt.scatter(fhd_f,gleam_f,label='complex pairs',color='red')
    plt.legend(loc='upper left',fontsize=15)
    if logscale:
        plt.xscale('log')
        plt.yscale('log')
    plt.xlim(0.03,60)
    plt.ylim(0.03,60)
    plt.gca().set_aspect('equal')
    plt.xlabel('Fhd Flux (Log)',fontsize=15)
    plt.ylabel('GLEAM Flux (Log)',fontsize=15)
    plt.title('FHD vs GLEAM source fluxes',fontsize=15)
    plt.plot([0, 1], [0, 1], transform=ax.transAxes,linestyle='--',color='k')
    
    if savefig:
        plt.savefig(outfig)
        if write_params:
            curr_func = inspect.stack()[0][3]
            utils.write_params_to_text(outfig,args,curr_file=curr_file,
                                       curr_func=curr_func)


def plotFluxHists(fhd,gleam,delta=0.15,sortBy=None,ratioThresh=5,singleSourceThresh=0.1,
                     combine_fhd=False,logscale=False,savefig=False,outfig='',density=False,
                     write_params=True,fluxCombType=np.sum):
    args = locals()
    ra, dec, flux, gra, gdec, gflux = getCols(fhd,gleam,sortBy=sortBy,flux_type='stokes')
    spairs, cpairs, apairs, sinds, cinds, noinds = find_sources(fhd, gleam, sortBy=sortBy,
                                                       ratioThresh=ratioThresh,delta=delta,
                                                       singleSourceThresh=singleSourceThresh,
                                                       combine_fhd=combine_fhd,fluxCombType=np.sum)

    fig, ax = plt.subplots(1,3,figsize=(16,6))
    bins = np.linspace(0,5,20)
    titles = ['Simple Pairs','Complex Pairs','Unmatched FHD Sources']
    for i,p in enumerate([spairs,cpairs,flux]):
        a = ax[i]
        if i<2:
            fhd_srcs = [x[0][0:3] for x in p]
            fhd_x = [x[0] for x in fhd_srcs]
            fhd_y = [x[1] for x in fhd_srcs]
            fhd_f = [x[2] for x in fhd_srcs]
            a.hist(fhd_f, bins=bins, alpha=0.5,fill=False,edgecolor='red',linewidth=2,density=density,label='FHD')
            gleam_f = [fluxCombType(x[1:],axis=0)[2] for x in p]
            a.hist(gleam_f, bins=bins, alpha=0.5,fill=False,edgecolor='blue',linewidth=2,density=density,label='GLEAM')
        else:
            fhd_f = p[noinds]
            a.hist(fhd_f, bins=bins, alpha=0.5,fill=False,edgecolor='red',linewidth=2,density=density,label='FHD')
        if i==1:
            a.legend(loc='upper right')
        a.set_title(titles[i])
        a.set_xlabel('Flux (Jy)')
        if i==0:
            if density:
                a.set_ylabel('Count (density)')
            else:
                a.set_ylabel('Count')
    if savefig:
        plt.savefig(outfig)
        if write_params:
            curr_func = inspect.stack()[0][3]
            utils.write_params_to_text(outfig,args,curr_file=curr_file,
                                       curr_func=curr_func)


def combine_sources(ras,decs,fluxes,fluxCombType=np.sum):
    ra = np.average(ras,weights=fluxes)
    dec = np.average(decs,weights=fluxes)
    flux = fluxCombType(fluxes)
    return ra, dec, flux

def find_sources(fhd, gleam, ratioThresh=5, delta=0.15, sortBy=None, singleSourceThresh=0.1,
                combine_fhd=False, combine_fhd_thresh=0.02, fluxCombType=np.sum, beamCut=0):
    ra, dec, flux, gra, gdec, gflux = getCols(fhd,gleam,sortBy=sortBy, beamCut=beamCut)
    print(f'Matching gleam to {len(ra)} fhd sources')
    
    src_pairs_simple = [] 
    src_pairs_complex = []
    simple_inds = []
    complex_inds = []
    nomatch_inds = []
    all_pairs = []
    inds_skip = []
    fluxes = []

    for i, r in enumerate(ra):
        if i in inds_skip:
            continue
        d = dec[i]
        f = flux[i]
        idx = (gra>(r-delta))*(gra<(r+delta))*(gdec>(d-delta))*(gdec<(d+delta))
        gr = np.asarray(gra[idx])
        gd = np.asarray(gdec[idx])
        gf = np.asarray(gflux[idx])
        # if combine_fhd turned on, check if there are other, overlapping FHD sources - if so, combine them into one source
        if combine_fhd:
            dists = np.asarray([distance.euclidean([r,d],[ra[i],dec[i]]) for i in range(len(ra))])
#             print(dists)
            comb_comp = (dists!=0) & (dists<combine_fhd_thresh)
            if np.count_nonzero(comb_comp)>0:
                inds = np.where(comb_comp)[0]
                for j in inds:
                    inds_skip.append(j)
#                 print(f'Combining {len(inds)+1} FHD sources')
                r, d, f = combine_sources(ra[inds],dec[inds],flux[inds],fluxCombType)
        # No gleam sources within specified box
        if len(gr)==0:
            nomatch_inds.append(i)
            all_pairs.append([(r,d,f)])
            continue
        diff = [distance.euclidean([r,d],[gr[i],gd[i]]) for i in range(len(gr))]
        # Only one gleam source within box - check if it's within specified range of fhd source
        if len(diff)==1 and diff[0]>0.08:
            nomatch_inds.append(i)
            all_pairs.append([(r,d,f)])
            continue
        match_ind = np.argmin(diff)
        match = (gr[match_ind],gd[match_ind],gf[match_ind])
        diff_rat = np.divide(diff,np.min(diff))
        src_comp = (1<diff_rat) & (diff_rat<ratioThresh)
        if np.count_nonzero(src_comp) == 0:
            src_pairs_simple.append([(r,d,f),match])
            all_pairs.append([(r,d,f),match])
            simple_inds.append(i)
        else:
            inds = np.where(diff_rat<2)[0]
            matches = []
            matches.append((r,d,f))
            for j in inds:
                src = (gr[j],gd[j],gf[j])
                matches.append(src)
            src_pairs_complex.append(matches)
            all_pairs.append(matches)
            complex_inds.append(i)
    return src_pairs_simple, src_pairs_complex, all_pairs, simple_inds, complex_inds, nomatch_inds

def get_unmatched_gleam(apairs,fhd,gleam,sortBy,beamCut=0):
    ra, dec, flux, gra, gdec, gflux = getCols(fhd,gleam,sortBy=sortBy,beamCut=beamCut)
    inds = np.zeros(np.shape(gra))
    count = 0
    for i,p in enumerate(apairs):
        for j,g in enumerate(p[1:]):
            r = g[0]
            aind = np.argmin(abs(np.subtract(gra,r)))
            inds[aind] = 1
            count += 1
    return inds


def plotSourcePairs(fhd,gleam,ncols=10,delta=0.15,sortBy=None,ratioThresh=5,singleSourceThresh=0.1,
                   plotAvgSrc=True,combine_fhd=False,sourcerange=[0,-1],savefig=False,outfig='',
                   write_params=True):
    args = locals()
    print(f'Sorting by {sortBy}')
    ra, dec, flux, gra, gdec, gflux = getCols(fhd,gleam,sortBy=sortBy)
    spairs, cpairs, apairs, sinds, cinds, noinds = find_sources(fhd, gleam, sortBy=sortBy,
                                                       ratioThresh=ratioThresh,delta=delta,
                                                       singleSourceThresh=singleSourceThresh,
                                                       combine_fhd=combine_fhd)
    apairs = apairs[sourcerange[0]:sourcerange[1]]
    print(f'Plotting {len(apairs)} source pairs')
    nrows = len(apairs)//ncols
    if len(apairs)%ncols !=0:
        nrows += 1
    fig, ax = plt.subplots(nrows,10,figsize=(20,nrows*2))
    for i,src in enumerate(apairs):
        if src in spairs:
            boxcolor = 'green'
        elif src in cpairs:
            boxcolor = 'cyan'
        else:
            boxcolor = 'black'
        r = src[0][0]
        d = src[0][1]
        f = np.log10(src[0][2])
        a = ax[i//ncols][i%ncols]
        idx = (gra>(r-delta))*(gra<(r+delta))*(gdec>(d-delta))*(gdec<(d+delta))
        im = a.scatter(r,d,c=f,marker=',',cmap='plasma_r',
                       vmin=np.min(flux)-2.5,vmax=np.percentile(flux,85))
        gr = gra[idx]
        gd = gdec[idx]
        gf = np.log10(gflux[idx])
        a.scatter(gr,gd,facecolor='None',marker='o',s=60,edgecolor='k')
        if len(apairs[i])>1:
            for j,src in enumerate(apairs[i][1:]):
                a.scatter(src[0],src[1],facecolor='blue',marker='o',s=60,edgecolor='k',alpha=0.3)
            if plotAvgSrc and len(apairs[i])>2:
                rc,dc,fc = combine_sources([x[0] for x in apairs[i][1:]],[x[1] for x in apairs[i][1:]],
                                        [x[2] for x in apairs[i][1:]])
#                 avg_src = (np.average(apairs[i][1:],axis=0)[0],np.average(apairs[i][1:],axis=0)[1])
                a.scatter(rc,dc,facecolor='red',marker='o',s=60,edgecolor='k',alpha=0.3)
#                 a.scatter(avg_src[0],avg_src[1],facecolor='red',marker='o',s=60,edgecolor='k',alpha=0.3)
        a.set_xticks([])
        a.set_yticks([])
        a.set_xlim([r-delta,r+delta])
        a.set_ylim([d-delta,d+delta])
        a.set(adjustable='box', aspect='equal')
        a.tick_params(color=boxcolor, labelcolor=boxcolor)
        for spine in a.spines.values():
            spine.set_edgecolor(boxcolor)
            spine.set_linewidth(1.5)
    while (i+1)%ncols!=0:
        i+=1
        a = ax[i//ncols][i%ncols]
        a.set_axis_off()
    if savefig:
        plt.savefig(outfig)
        if write_params:
            curr_func = inspect.stack()[0][3]
            utils.write_params_to_text(outfig,args,curr_file=curr_file,
                                       curr_func=curr_func)
            
def plotBeam(fhd_dir,pol='XX',color_scale=[-1763758,1972024],output_path='',prefix='beam',
             write_pixel_coordinates=False,log_scale=False,ra_range=40,dec_range=40,fontsize=16,
             savefig=False,write_params=True,outfig=''):
    import glob
    beamFile = glob.glob(f'{fhd_dir}/output_data/*_Beam_{pol}.fits')[0]
    data = plot_fits.load_image(beamFile)
    fig, ax = plt.subplots(1,1,figsize=(12,12))
    im = plot_fits.plot_fits_image(data, ax, color_scale, output_path, prefix, write_pixel_coordinates, log_scale,
                                 ra_range=ra_range,dec_range=dec_range,title='Beam',fontsize=fontsize)
    sources = plot_fits.gather_source_list()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
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
        plt.savefig(outfig)
        if write_params:
            curr_func = inspect.stack()[0][3]
            utils.write_params_to_text(outfig,args,curr_file=curr_file,
                                       curr_func=curr_func)
    else:
        plt.show()


def plotRaDecHists(fhd,gleam,param='RA',delta=0.15,sortBy=None,ratioThresh=5,singleSourceThresh=0.1,
                     combine_fhd=False,savefig=False,outfig='',write_params=True,units='deg',
                      inc_errorbar=False,beamCut=0):
    args = locals()
    ra, dec, flux, gra, gdec, gflux = getCols(fhd,gleam,sortBy=sortBy,flux_type='stokes',beamCut=beamCut)
    spairs, cpairs, apairs, sinds, cinds, noinds = find_sources(fhd, gleam, sortBy=sortBy,
                                                       ratioThresh=ratioThresh,delta=delta,
                                                       singleSourceThresh=singleSourceThresh,
                                                       combine_fhd=combine_fhd,fluxCombType=np.sum,
                                                       beamCut=beamCut)
    
    fig = plt.figure(figsize=(20,6))
    ax = fig.add_subplot(1,1,1)
    
    # ras_fhd = [a[0][0] for a in spairs]
    # decs_fhd = [a[0][1] for a in spairs]
    # ras_gleam = [a[i][0] for a in spairs for i in range(1,len(a))]
    # decs_gleam = [a[i][1] for a in spairs for i in range(1,len(a))]
    # fhd_srcs = [x[0][0:2] for x in spairs]
    # fhd_x = [x[0] for x in fhd_srcs]
    # fhd_y = [x[1] for x in fhd_srcs]
    # pairs = [(np.average(x[1:],axis=0)[0],np.average(x[1:],axis=0)[1]) for x in spairs]
    pairs = fhd + np.random.rand(np.shape(fhd))
    pairs[:,0] += np.random.rand(np.shape(fhd))
    pairs[:,1] += np.random.rand(np.shape(fhd))
    diffs = np.subtract(fhd_srcs,pairs)
    if units=='arcmin':
        diffs *=60

    if param=='RA':
        pind = 0
    elif param=='DEC':
        pind = 1
        
    bins = np.linspace(np.min(diffs),np.max(diffs),20)
    plt.hist(diffs[:,pind], bins=bins, alpha=0.5,fill=False,edgecolor='red',linewidth=2,density=True)
    plt.xlabel(f'{param} ({units})',fontsize=15)
    plt.ylabel('Count Density',fontsize=15)
    plt.title(f'{param} Pairwise Positional Difference Histogram',fontsize=15)

    (mu, sigma) = norm.fit(diffs[:,pind])
    y = norm.pdf( bins, mu, sigma)
    l = plt.plot(bins, y, 'b--', linewidth=2)

    plt.axvline(np.mean(diffs[:,pind]),linestyle='--',color='k')
    if inc_errorbar:
        err = np.divide(sigma,np.sqrt(len(diffs[:,pind])))
        plt.axvline(np.mean(diffs[:,pind])+err,linestyle='--',color='k',alpha=0.5)
        plt.axvline(np.mean(diffs[:,pind])-err,linestyle='--',color='k',alpha=0.5)

    plt.text(s=f'Mean: {np.around(mu,5)}',x=0.8,y=0.8,transform = ax.transAxes,fontsize=12)
    plt.text(s=f'Std: {np.around(sigma,5)}',x=0.8,y=0.7,transform = ax.transAxes,fontsize=12)
    if savefig:
        plt.savefig(outfig)
        if write_params:
            curr_func = inspect.stack()[0][3]
            utils.write_params_to_text(outfig,args,curr_file=curr_file,
                                       curr_func=curr_func)
    else:
        plt.show()