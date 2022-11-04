from pyuvdata import UVData, UVCal
from matplotlib import pyplot as plt
import numpy as np
import os
from astropy.coordinates import EarthLocation, SkyCoord, AltAz, Angle
from astropy.time import Time
import scipy.io
import astropy.io
from astropy.io import fits
import glob

def readRawVisibilities(data_path,ants,visrange='all'):
    data_files = sorted(glob.glob(f'{data_path}/*uvfits'))
    if visrange != 'all':
        data_files = data_files[visrange[0]:visrange[1]]
    
    uv = UVData()
    uv.read(data_files,antenna_nums=ants)
#     dat = uv.get_data(ant,ant,pol)
    return uv, uv.flag_array, uv.freq_array, uv.lst_array

def plot_gains(gains_array, ax, ant,set_alpha_mask=False,metric='amp',title='',inc_xlabels=True,vrange='auto',cmap='plasma',
              zero_mean=False,ylabel='LST',inc_ylabels=True,xlabel='Freq (MHz)'):
    lst_blacklists = [(0,1.3),(2.5,4.3),(6.5,9.1),(10.6,11.5),(11.9,14.3),(16.3,24)]
    if type(ant) is int:
        dat = gains_array[ant]['cal_array']
        lsts = np.unique(gains_array[ant]['lst_array'])
        freqs = gains_array[ant]['freq_array']
        vmax = 200
        vmin=0
    elif ant[0]==ant[1]:
        dat = gains_array[ant[0]]['cal_array']
        lsts = np.unique(gains_array[ant[0]]['lst_array'])
        freqs = gains_array[ant[0]]['freq_array']
        vmax = 200
        vmin=0
    else:
        d1 = gains_array[ant[0]]['cal_array']
        d2 = gains_array[ant[1]]['cal_array']
        dat = np.multiply(d1,np.conj(d2))
        lsts = np.unique(gains_array[ant[0]]['lst_array'])
        freqs = gains_array[ant[0]]['freq_array']
        vmax = 5e3
        vmin=0
    dat_complex = dat
    if vrange != 'auto':
        vmin = vrange[0]
        vmax = vrange[1]
    if metric == 'amp':
        dat = np.abs(dat)
    elif metric == 'real':
        dat = np.real(dat)
    elif metric == 'imag':
        dat = np.imag(dat)
    else:
        print('################ ERROR: metric must be one of: amp, real, imag ################')
#     print(len(lsts))
#     print(np.shape(dat))
    if zero_mean == True:
        dat = np.subtract(dat, np.nanmean(dat))
    if set_alpha_mask is True:
        alpha_mask = np.ones(np.shape(dat))
        if set_alpha_mask == True:
            for l,lst in enumerate(lsts):
                if l==len(lsts)-1:
                    continue
                mask = False
                for r in lst_blacklists:
                    if r[0]<=lst<=r[1]:
                        mask = True
                if mask == True:
                    alpha_mask[l,:] = np.ones((1,np.shape(dat)[1]),dtype=float)*0.4
                else:
                    alpha_mask[l,:] = np.ones((1,np.shape(dat)[1]),dtype=float)
    else:
        alpha_mask = 1
    im = ax.imshow(dat,aspect='auto',vmin=vmin,vmax=vmax,cmap=cmap,interpolation='nearest',alpha=alpha_mask)
#     print('Interpolation = nearest')
    if title == '':
        if type(ant) is int or ant[0]==ant[1]:
            ax.set_title(f'Ant {ant} gains')
        else:
            ax.set_title(f'|g{ant[0]} x g{ant[1]}*|')
    else:
        ax.set_title(title)
    inds = np.unique(lsts,return_index=True)[1]
    lsts = [lsts[ind] for ind in sorted(inds)]
    nticks = 6
    yticks = [int(i) for i in np.linspace(0,len(lsts)-1,nticks)]
    yticklabels = [np.around(lsts[ytick],1) for ytick in yticks]
    if set_alpha_mask == False:
        cbar = plt.colorbar(im,ax=ax)
    if inc_ylabels is True:
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)
    ax.set_ylabel(ylabel)
    if inc_xlabels is True:
        xticks = [int(i) for i in np.linspace(0,len(freqs)-1,nticks)]
        xticklabels = [int(freqs[i]) for i in xticks]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
        ax.set_xlabel(xlabel)
    return im, dat, dat_complex
            
def plot_vis(vis_array,ax,bl,dtype='dirty',alpha_mask=1,freqclip=None,inc_xlabels=False,title='',vrange='auto',cmap='plasma',
            ylabel='LST',inc_ylabels=True,xlabel='Freq (MHz)'):
    lsts = vis_array['lsts']
    freqs = vis_array['freqs']
    dat = vis_array['amp'][bl]
    if vrange == 'auto':
        if bl[0]==bl[1]:
            if dtype == 'model':
                vmin = 15
                vmax = 40
            elif dtype == 'dirty':
                vmin = 0
                vmax = 3e4
            else:
                print('INVALID DTYPE PROVIDED')
        else:
            if dtype == 'model':
                vmin = 0
                vmax = 8
            elif dtype == 'dirty':
                vmin = 0
                vmax = 0.4e2
            else:
                print('INVALID DTYPE PROVIDED')
    else:
        vmin = vrange[0]
        vmax = vrange[1]
    if freqclip is not None:
        dat = dat[:,freqclip[0]:freqclip[1]]
        freqs = freqs[freqclip[0]:freqclip[1]]
    im = ax.imshow(dat,aspect='auto',vmin=vmin,vmax=vmax,cmap=cmap,interpolation='nearest',alpha=alpha_mask)
    if title == '':
        ax.set_title(f'Baseline {bl} {dtype} vis amp')
    else:
        ax.set_title(title)
    inds = np.unique(lsts,return_index=True)[1]
    lsts = [lsts[ind] for ind in sorted(inds)]
    nyticks = 6
    yticks = [int(i) for i in np.linspace(0,len(lsts)-1,nyticks)]
    yticklabels = [np.around(lsts[ytick],1) for ytick in yticks]
    if type(alpha_mask) == int:
        cbar = plt.colorbar(im,ax=ax,aspect=20)
    if inc_ylabels is True:
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)
        ax.set_ylabel(ylabel)
    nxticks = 8
    if inc_xlabels is True:
        xticks = [int(i) for i in np.linspace(0,len(freqs)-1,nxticks)]
        xticklabels = [np.around(freqs[i],0) for i in xticks]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
        ax.set_xlabel(xlabel)
    else:
        ax.set_xticks([])
        ax.set_xticklabels([])
    return im

def plot_raw_vis(data_array,ax,ant,lsts,freqs,inc_xlabels=False,alpha_mask=1,title='',xlabel='Freq (MHz)',vmax='default',vmin=0,
                cmap='plasma',ylabel='LST',inc_yticks=True, inc_ylabels=True):
#     print(alpha_mask.shape)
#     print(np.shape(data_array))
#     print(alpha_mask)
    if vmax=='default':
        if ant[0]==ant[1]:
            vmax = 1.3e7
        else:
            vmax = 1e5
        im = ax.imshow(data_array,aspect='auto',cmap=cmap,interpolation='nearest',vmax=vmax,vmin=vmin,alpha=alpha_mask)
    elif vmax=='auto':
        im = ax.imshow(data_array,aspect='auto',cmap=cmap,interpolation='nearest',vmin=vmin,alpha=alpha_mask)
    else:
        im = ax.imshow(data_array,aspect='auto',cmap=cmap,interpolation='nearest',vmax=vmax,vmin=vmin,alpha=alpha_mask)
    if title == '':
        ax.set_title(f'Baseline {ant} Raw Vis Amp')
    else:
        ax.set_title(title)
    lsts = np.multiply(lsts,3.819719)
    inds = np.unique(lsts,return_index=True)[1]
    lsts = [lsts[ind] for ind in sorted(inds)]
    nyticks = 6
    yticks = [int(i) for i in np.linspace(0,len(lsts)-1,nyticks)]
    yticklabels = [np.around(lsts[ytick],1) for ytick in yticks]
    if type(alpha_mask)==int:
        cbar = plt.colorbar(im,ax=ax,aspect=20)
    if inc_yticks is True:
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)
    else:
        ax.set_yticks([])
    if inc_ylabels is True:
        ax.set_ylabel(ylabel)
    nxticks = 6
    if inc_xlabels is True:
        xticks = [int(i) for i in np.linspace(0,len(freqs)-1,nxticks)]
        xticklabels = [np.around(freqs[i],0) for i in xticks]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
        ax.set_xlabel(xlabel)
    else:
        ax.set_xticks([])
        ax.set_xticklabels([])
    return im

def readFHDVisibilities(path1, band, baselines='all', polarization='XX', use_model=False, visrange='all'):
#     print(f'Getting FHD files from {path1}')
    vis_array = {}
    vis = [ name for name in os.listdir(path1) if os.path.isdir(os.path.join(path1, name)) and 'fhd_' in name ]
    vis.sort()
#     print(vis[0])
#     print(len(vis))
    if visrange != 'all':
        vis = vis[visrange[0]:visrange[1]]
    obs1 = vis[3].split('_')[1] + '.sum'
    path = '%s/%s' % (path1,vis[3])
    uv = UVData()
    fhd_files = []
    fhd_files.append(glob.glob(f'{path}/metadata/*_params.sav')[0])
    fhd_files.append(glob.glob(f'{path}/metadata/*_settings.txt')[0])
    fhd_files.append(glob.glob(f'{path}/metadata/*_layout.sav')[0])
    vis_files = ['flags.sav','vis_XX.sav','vis_YY.sav','vis_model_XX.sav','vis_model_YY.sav']
    for f in vis_files:
        fhd_files.append(glob.glob(f'{path}/vis_data/*{f}')[0])
#     print(fhd_files)
    uv.read(fhd_files, use_model=use_model)
    print(uv.get_ants())
    loc = EarthLocation.from_geocentric(*uv.telescope_location, unit='m')
    autos = []
    crosses = []
    if baselines == 'all':
        ants = uv.get_ants()
        baselines = []
        for i,a in enumerate(ants):
            for j,b in enumerate(ants):
                if j<i:
                    baselines.append((a,b))
                    crosses.append((a,b))
                elif j==i:
                    baselines.append((a,b))
                    autos.append((a,b))
    nfreqs = len(uv.freq_array[0])
    ntimes = len(np.unique(uv.time_array))
    print(ntimes)
    freq_array = uv.freq_array[0]*10**(-6)
    flag_array = {}
    day_array = {}
    day_array_phase = {}
    for b in baselines:
        day_array[b] = np.zeros((len(vis)*ntimes,nfreqs))
        day_array_phase[b] = np.zeros((len(vis)*ntimes,nfreqs))
        flag_array[b] = np.zeros((len(vis)*ntimes,nfreqs))
    lst_array = []
    for i in range(len(vis)):
        v = vis[i]
        obs = v.split('_')[1] + '_' + band
        if i%50==0:
            print(obs)
        path = '%s/%s' % (path1,v)
        if os.path.exists(path + '/vis_data/') is False:
            if i == 0:
                continue
            mask_array[i*ntimes:(i+1)*ntimes,:] = np.ones(np.shape(vis_data))
            for b in baselines:
                day_array[b][i*ntimes:(i+1)*ntimes,:] = vis_data
                day_array_phase[b][i*ntimes:(i+1)*ntimes,:] = vis_data_phase
                flag_array[b][i*ntimes:(i+1)*ntimes,:] = np.ones(np.shape(vis_data))
        else:
            uv = UVData()
            fhd_files = []
            fhd_files.append(glob.glob(f'{path}/metadata/*_params.sav')[0])
            fhd_files.append(glob.glob(f'{path}/metadata/*_settings.txt')[0])
            fhd_files.append(glob.glob(f'{path}/metadata/*_layout.sav')[0])
            vis_files = ['flags.sav','vis_XX.sav','vis_YY.sav','vis_model_XX.sav','vis_model_YY.sav']
            for f in vis_files:
                fhd_files.append(glob.glob(f'{path}/vis_data/*{f}')[0])
            uv.read(fhd_files, use_model=use_model, bls=baselines, file_type='fhd')
            ntimes = len(np.unique(uv.time_array))
            for t in np.unique(uv.lst_array):
                lst_array.append(t*3.819719)
            if i == 0:
                mask_array = np.zeros((ntimes*len(vis),nfreqs))
            for b in baselines:
                vis_data = np.abs(uv.get_data(b[0], b[1], polarization))
                vis_data_phase = np.angle(uv.get_data(b[0], b[1], polarization))
                day_array[b][i*ntimes:(i+1)*ntimes,:] = vis_data
                day_array_phase[b][i*ntimes:(i+1)*ntimes,:] = vis_data_phase
                #mask_array[i*ntimes:(i+1)*ntimes,:] = uv.get_flags(b[0], b[1], polarization)
                flags = uv.get_flags(b[0],b[1],polarization)
                flag_array[b][i*ntimes:(i+1)*ntimes,:] = flags
    for b in baselines:
        day_array[b] = np.ma.masked_array(day_array[b],mask=mask_array)
        day_array_phase[b] = np.ma.masked_array(day_array_phase[b],mask=mask_array)
    vis_array['amp'] = day_array
    vis_array['phase'] = day_array_phase
    vis_array['flags'] = flag_array
    vis_array['freqs'] = freq_array
    vis_array['lsts'] = lst_array
    vis_array['autos'] = autos
    vis_array['crosses'] = crosses
    return vis_array, loc

def readCalSolutions(path1,loc,band,data_array={},ant_nums=[],pol='xx',visrange='all',source='sav',
              write_calfits=False, metric='amp',freqrange='all'):
    cal = UVCal()
    ant_nums.sort()
    nants = len(ant_nums)
    day_array = {}
    vis = [ name for name in os.listdir(path1) if os.path.isdir(os.path.join(path1, name)) and 'fhd_' in name ]
    vis.sort()
    if visrange != 'all':
        vis = vis[visrange[0]:visrange[1]]
    nocal = 0
    i=-1
    times = []
    lsts = []
    cal_dict = {}
    mask_dict = {}
    init_data=False
    for v in vis:
        i = i + 1
#         print(glob.glob(f'{path1}/{v}/vis_data/'))
        if os.path.exists(f'{path1}/{v}/vis_data/'):
            obs = glob.glob(f'{path1}/{v}/vis_data/*_vis_XX.sav')[0][0:-11]
        else:
            print(f'No data in {path1}/{v}/vis_data/')
            continue
#         print('Reading data for obs ' + str(obs))
        data = True
        path = path1 + '/' + v
        if os.path.exists(path + '/calibration/') is False:
                print('!!!!!!!!!!!!!!!!!!!!WARNING!!!!!!!!!!!!!!!!!!!')
                print('There is no calibration data for this observation. The gains for this observation will be entered as 0.')
                print(path + '/calibration/')
                nocal = nocal + 1
                data = False
                times.append(None)
        if data == True:
                obsfile = glob.glob(f'{path}/metadata/*_obs.sav')[0]
                calfile = glob.glob(f'{path}/calibration/*_cal.sav')[0]
                settingsfile = glob.glob(f'{path}/metadata/*_settings.txt')[0]
#                 try:
#                 print(calfile)
#                 print(obsfile)
                cal.read_fhd_cal(calfile,obsfile,settings_file=settingsfile)
#                 except:
#                     print(f'Could not read {obs}')
#                     break
        for j in range(nants):
            ant = ant_nums[j]
            if write_calfits == True and data == True:
                writepath = path + str(nums[i]) + '/calibration/' + obs + '_cal.fits'
                cal.write_calfits(writepath, clobber=True)
            if i == 0 and data==True:
                nobs = len(vis)
                nfreqs = cal.Nfreqs
                freq_array = cal.freq_array[0,:]
                freq_array = np.transpose(freq_array)
                freq_array = np.divide(freq_array,1000000)
                if freqrange != 'all':
                    minfreqind = np.argmax(freq_array>freqrange[0])
                    maxfreqind = np.argmin(freq_array<freqrange[1])
                else:
                    minfreqind = 0
                    maxfreqind = len(freq_array)-1
                cal_dict[ant] = np.empty((nfreqs,nobs))
                mask_dict[ant] = np.zeros((nfreqs,nobs))
                if j == 0:
                    ## Get the LST start and end times for this obsid ##
                    time_array = cal.time_array
                    obstime_start = Time(time_array[0],format='jd',location=loc)
                    startTime = obstime_start.sidereal_time('mean').hour
                    JD = int(obstime_start.jd)
            if v == vis[-1] and j == 0:
                time_array = cal.time_array
                obstime_end = Time(time_array[-1],format='jd',location=loc)
                endTime = obstime_end.sidereal_time('mean').hour
            if data is True:
                if metric == 'amp':
                    gain = np.abs(cal.get_gains(ant, pol))
                elif metric == 'phase':
                    gain = np.angle(cal.get_gains(ant, pol))
                elif metric == 'complex':
                    gain = cal.get_gains(ant,pol)
                else:
                    print('Invalid metric parameter')
                    break
            if data is False:
                if init_data is False:
                    continue
                else:
                    gain = np.zeros(np.shape(gain))
                    cal_dict[ant][:,i] = gain[:,0]
                    mask_dict[ant][:,i] = np.ones(np.shape(gain))[:,0]
                    continue
            init_data=True
            cal_dict[ant][:,i] = gain[:,0]
            tArr = cal.time_array
            t = Time(tArr[0],format='jd',location=loc)
            times.append(t)
    mx_dict = {}
    print('Doing per-antenna calculations')
    for j in range(nants):
        ant = ant_nums[j]
        print(ant)
        cal_dict[ant] = np.transpose(cal_dict[ant])
        mask_dict[ant] = np.transpose(mask_dict[ant])
        cal_dict[ant] = cal_dict[ant][:,minfreqind:maxfreqind]
        mask_dict[ant] = mask_dict[ant][:,minfreqind:maxfreqind]
        calsort = np.sort(cal_dict[ant])
        mx_dict[ant] = np.ma.masked_array(cal_dict[ant], mask=mask_dict[ant])
        freq_array = freq_array[minfreqind:maxfreqind]
        lst_array = get_LSTs(times)
        zenith_RAs = get_zenithRA(times,loc)
        res = {
                "ant_num": ant,
                "obsid": obs,
                "cal_array": cal_dict[ant],
                "time_array": times,
                "freq_array": freq_array,
                "masked_data": mx_dict[ant],
                "zenith_RA_array": zenith_RAs,
                "lst_array": lst_array,
                "pol": pol
            }
        day_array[ant] = res
    print('###############################################################')
    print('Out of ' + str(len(vis)) + ' observations, ' + str(nocal) + ' did not contain calibration solutions')
    print('###############################################################')
    return day_array

def plot_catalog(
    catalog_path, save_filename, flux_plot_max=None, title='',
    ra_range=None, dec_range=None, ra_cut_val=220, label_sources=False,
    label_gleam_supplement_sources=False, savefig=False
):

#     catalog = scipy.io.readsav(catalog_path)['catalog']
    catalog = scipy.io.readsav(catalog_path)
    print(catalog)

    source_ras = []
    source_decs = []
    source_fluxes = []
    for source in catalog:
        ra = source['ra']
        if ra > ra_cut_val:
            ra -= 360.
        if ra < ra_cut_val-360.:
            ra += 360.
        source_ras.append(ra)
        source_decs.append(source['dec'])
        source_fluxes.append(source['flux']['I'][0])

    if flux_plot_max is None:
        flux_plot_max = max(source_fluxes)
    source_markersizes = []
    markersize_range = [.02, 3.]
    for flux in source_fluxes:
        if flux >= flux_plot_max:
            flux = flux_plot_max
        source_markersizes.append(
            flux/flux_plot_max*(markersize_range[1] - markersize_range[0])
            + markersize_range[0]
        )

    if ra_range is None:
        ra_min = min(source_ras)
        ra_max = max(source_ras)
        ra_range = [
            ra_min-(ra_max-ra_min)/10., ra_max+(ra_max-ra_min)/10.
        ]
    if dec_range is None:
        dec_min = min(source_decs)
        dec_max = max(source_decs)
        dec_range = [
            dec_min-(dec_max-dec_min)/10., dec_max+(dec_max-dec_min)/10.
        ]

    plt.figure()
    ax = plt.gca()
    plt.scatter(
        source_ras, source_decs, s=source_markersizes, facecolors='blue',
        edgecolors='none'
        )

    if label_sources:
        source_names = [
            'Crab', 'PicA', 'HydA', 'CenA',
            'HerA', 'VirA', 'CygA', 'CasA',
            '3C161', '3C353', '3C409',
            '3C444', 'CasA', 'ForA', 'HerA',
            'NGC0253', 'PicA', 'VirA',
            'PKS0349-27', 'PKS0442-28', 'PKS2153-69', 'PKS2331-41',
            'PKS2356-61', 'PKSJ0130-2610'
        ]
        named_source_ras = [
            83.6331, 79.9572, 139.524, 201.365,
            252.784, 187.706, 299.868, 350.858,
            96.7921, 260.117, 303.615,
            333.607, 350.866, 50.6738, 252.793,
            11.8991, 79.9541, 187.706,
            57.8988, 71.1571, 329.275, 353.609,
            359.768, 22.6158
        ]
        named_source_decs = [
            22.0145, -45.7788, -12.0956, -43.0192,
            4.9925, 12.3911, 40.7339, 58.8,
            -5.88472, -0.979722, 23.5814,
            -17.0267, 58.8117, -37.2083, 4.99806,
            -25.2886, -45.7649, 12.3786,
            -27.7431, -28.1653, -69.6900, -41.4233,
            -60.9164, -26.1656
        ]
        for ind, ra in enumerate(named_source_ras):
            if ra > ra_cut_val:
                named_source_ras[ind] = ra-360.
            if ra < ra_cut_val-360.:
                named_source_ras[ind] = ra+360.
        plt.scatter(
            named_source_ras, named_source_decs, marker='o',
            s=markersize_range[1], facecolors='none', edgecolors='red',
            linewidth=.1
        )
        for i, name in enumerate(source_names):
            plt.annotate(
                name, (named_source_ras[i], named_source_decs[i]),
                fontsize=3.
            )

    if label_gleam_supplement_sources:
        source_names = [
            'PicA', 'HydA', 'CenA', 'HerA', 'VirA', 'CasA', '3C161',
            '3C409', 'ForA'
        ]
        named_source_ras = [
            79.9572, 139.524, 201.365, 252.784, 187.706, 350.858, 96.7921, 303.615, 50.6738
        ]
        named_source_decs = [
            -45.7788, -12.0956, -43.0192, 4.9925, 12.3911, 58.8, -5.88472,
            23.5814, -37.2083
        ]
        for ind, ra in enumerate(named_source_ras):
            if ra > ra_cut_val:
                named_source_ras[ind] = ra-360.
            if ra < ra_cut_val-360.:
                named_source_ras[ind] = ra+360.
        plt.scatter(
            named_source_ras, named_source_decs, marker='o',
            s=markersize_range[1], facecolors='none', edgecolors='red',
            linewidth=.1
        )
        for i, name in enumerate(source_names):
            plt.annotate(
                name, (named_source_ras[i], named_source_decs[i]),
                fontsize=3.
            )

    plt.xlim(ra_range[1], ra_range[0])
    plt.ylim(dec_range[0], dec_range[1])
    plt.xlabel('RA (deg)')
    plt.ylabel('Dec (deg)')
    plt.title(title)
    ax.set_aspect('equal', adjustable='box')
    ax.set_facecolor('white')
    if savefig is True:
        print(f'Saving plot to {save_filename}')
        plt.savefig(save_filename, format='png', dpi=500)
    else:
        plt.show()

def get_LSTs(time_array):
    lst_arr = []
    for i in range(len(time_array)):
        t = time_array[i]
        if t is not None:
            lst = t.sidereal_time('mean').hour
            lst_arr.append(lst)
        else:
            if i == 0:
                lst = 0
            else:
                lst = lst_arr[-1]+0.0139
            lst_arr.append(lst)
    return lst_arr

def get_zenithRA(time_array,telescope_location):
    zens = []
    for t in time_array:
        if t is None:
            zens.append(zens[-1] + (zens[-1]-zens[-2]))
        else:
            zen = SkyCoord(Angle(0, unit='deg'),Angle(90,unit='deg'),frame='altaz',obstime=t,location=telescope_location)
            zen = zen.transform_to('icrs')
            zen = zen.ra.degree
            zens.append(zen)
    return zens

