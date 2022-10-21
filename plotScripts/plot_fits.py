from astropy.io import fits
import numpy as np
import healpy as hp
import sys
import math
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
import matplotlib.colors as colors
from astropy_healpix import HEALPix
from astropy.coordinates import Galactic
from astropy.coordinates import EarthLocation, AltAz, Angle
from astropy.coordinates import SkyCoord as sc
from astropy.time import Time
from astropy import units as u
import healpy
import csv


class ImageFromFits:

    def __init__(self, signal_arr, ra_axis=None, dec_axis=None,
                 ra_range=None, dec_range=None):
        n_dec_vals, n_ra_vals = np.shape(signal_arr)
        if ra_axis is not None:
            ra_axis = list(ra_axis)
            if len(ra_axis) != n_ra_vals:
                print('ERROR: Number of axis elements does not match data axis. Exiting.')
                sys.exit(1)
        if dec_axis is not None:
            dec_axis = list(dec_axis)
            if len(dec_axis) != n_dec_vals:
                print('ERROR: Number of axis elements does not match data axis. Exiting.')
                sys.exit(1)
        if ra_axis is None and ra_range is not None:
            if len(ra_range) != 2:
                print('ERROR: parameters ra_range and dec_range must have the form [min, max]. Exiting.')
                sys.exit(1)
            ra_axis = list(np.linspace(ra_range[0], ra_range[1], n_ra_vals))
        if dec_axis is None and dec_range is not None:
            if len(dec_range) != 2:
                print('ERROR: parameters ra_range and dec_range must have the form [min, max]. Exiting.')
                sys.exit(1)
            dec_axis = list(
                np.linspace(dec_range[0], dec_range[1], n_dec_vals)
            )
        self.signal_arr = signal_arr
        for i in range(len(ra_axis)):
            if ra_axis[i]>180: ra_axis[i] = np.subtract(ra_axis[i],360)
        self.ra_axis = ra_axis
        self.dec_axis = dec_axis

    def limit_data_range(self, ra_range=None, dec_range=None):
#         print('ra_range:')
#         print(ra_range)
#         print('self.ra_axis:')
#         print(f'{self.ra_axis[0]} - {self.ra_axis[-1]}')
        
        if ra_range is not None:
            use_ra_inds = [i for i in range(len(self.ra_axis))
                           if ra_range[0] < self.ra_axis[i] < ra_range[1]
                           ]
        else:
            use_ra_inds = range(len(self.ra_axis))
            ra_range = [self.ra_axis[0],self.ra_axis[1]]
#             print('use_ra_inds:' + use_ra_inds)
#         print('use_ra_inds')
#         print(f'{use_ra_inds[0]} - {use_ra_inds[-1]}')
        if dec_range is not None:
#             print(dec_range[0])
#             print(dec_range[1])
            use_dec_inds = [i for i in range(len(self.dec_axis))
                            if dec_range[0] < self.dec_axis[i] < dec_range[1]
                            ]
#             print(use_dec_inds)
        else:
            use_dec_inds = range(len(self.dec_axis))
            print('use_dec_inds:' + use_dec_inds)
#         print(use_ra_inds)
#         print(use_dec_inds)
        self.signal_arr = self.signal_arr[
            use_dec_inds[0]:use_dec_inds[-1]+1,
            use_ra_inds[0]:use_ra_inds[-1]+1
        ]
        self.ra_axis = list(
            np.linspace(ra_range[0], ra_range[1], len(use_ra_inds))
        )
        self.dec_axis = list(
            np.linspace(dec_range[0], dec_range[1], len(use_dec_inds))
        )


def load_image(data_filename):

    contents = fits.open(data_filename)
    use_hdu = 0
    data = contents[use_hdu].data
    header = contents[use_hdu].header

    if 'CD1_1' in header.keys() and 'CD2_2' in header.keys():  # FHD convention
        cdelt1 = header['CD1_1']
        cdelt2 = header['CD2_2']
#         print ('WARNING: Ignoring curved sky effects.')
    elif 'CDELT1' in header.keys() and 'CDELT2' in header.keys():
        cdelt1 = header['CDELT1']
        cdelt2 = header['CDELT2']
    else:
        print('ERROR: Header format not recognized.')
        sys.exit(1)

    ra_axis = [
        header['crval1'] +
        cdelt1*(i-header['crpix1'])
        for i in range(header['naxis1'])
        ]
    dec_axis = [
        header['crval2'] +
        cdelt2*(i-header['crpix2'])
        for i in range(header['naxis2'])
        ]

    fits_image = ImageFromFits(data, ra_axis=ra_axis, dec_axis=dec_axis)
    return fits_image


def load_gaussian_source_model_as_image(
    catalog_path, source_ind=0, resolution=.01, ra_range=None, dec_range=None,
    reference_image=None
):
    # THE NORMALIZATION IS WRONG!!! DON'T USE THIS FUNCTION!

    # If reference image is supplied, use the same ra and dec locations
    if reference_image is not None:
        print('ref image is not none')
        ra_axis = reference_image.ra_axis
        dec_axis = reference_image.dec_axis
        grid_ra = np.tile(np.array(ra_axis), (len(dec_axis), 1))
        grid_dec = np.tile(np.array([dec_axis]).T, (1, len(ra_axis)))
        ra_range = [min(ra_axis), max(ra_axis)]
        dec_range = [min(dec_axis), max(dec_axis)]
    else:
        if ra_range is None:
            ra_range = [50, 51.25]
        if dec_range is None:
            dec_range = [-37.8, -36.7]

        grid_dec, grid_ra = np.mgrid[
            dec_range[0]:dec_range[1]:resolution,
            ra_range[0]:ra_range[1]:resolution
            ]

    plot_signal = np.zeros_like(grid_dec)

    source = scipy.io.readsav(catalog_path)['catalog'][source_ind]
    source_ra = source['ra']
    source_dec = source['dec']
    components = source['extend']
    if len(components) == 0:
        print('WARNING: Source is not extended.')

    total_flux = 0.
    for comp in components:
        comp_ra = comp['ra']
        comp_dec = comp['dec']
        comp_flux = comp['flux']['I'][0]
        total_flux += comp_flux
        print('total flux is' + total_flux)
        comp_size_x = comp['shape']['x'][0]/(7200.*np.sqrt(2*np.log(2.)))
        print(comp_size_x)
        comp_size_y = comp['shape']['y'][0]/(7200.*np.sqrt(2*np.log(2.)))
        comp_size_angle = comp['shape']['angle'][0]

        if comp_size_x == 0:
            comp_size_x = resolution
        if comp_size_y == 0:
            comp_size_y = resolution

        for i in range(np.shape(grid_dec)[0]):
            for j in range(np.shape(grid_dec)[1]):
                pixel_val = (
                    comp_flux*np.pi/(2.*(180.)**2.*comp_size_x*comp_size_y)
                    * np.exp(-(grid_ra[i, j]-comp_ra)**2./(2*comp_size_x**2.))
                    * np.exp(-(grid_dec[i, j]-comp_dec)**2./(2*comp_size_y**2.))
                )
                plot_signal[i, j] += pixel_val

    image = ImageFromFits(plot_signal, ra_range=ra_range, dec_range=dec_range)
    return image


def difference_images(image1, image2):

    if image1.ra_axis == image2.ra_axis and image1.dec_axis == image2.dec_axis:
        data_diff = ImageFromFits(
            np.subtract(image1.signal_arr, image2.signal_arr),
            ra_axis=image1.ra_axis, dec_axis=image1.dec_axis
            )
    else:
        print ('WARNING: Image axes do not match. Interpolating image2 to image1 axes.')
        image2_signal_array_interp = griddata(  # This doesn't work
            (image2.ra_axis, image2.dec_axis),
            image2.signal_arr,
            (image1.ra_axis, image1.dec_axis)
            )
        data_diff = ImageFromFits(
            np.subtract(image1.signal_arr, image2_signal_array_interp),
            ra_axis=image1.ra_axis, dec_axis=image1.dec_axis
            )
    return data_diff


def plot_fits_image(
    fits_image, ax, color_scale, output_path, prefix, write_pixel_coordinates, log_scale, title='', ra_range=None, dec_range=None, log=False,
    colorbar_label='Flux Density (Jy/beam)', plot_grid=True,
    xlabel='RA (deg.)', ylabel='Dec. (deg.)'
):

#     ra_range = [45,75]
#     dec_range = [-45,-15]
    colorbar_range = color_scale
    save_filename = output_path
    if ra_range is not None or dec_range is not None:
        fits_image.limit_data_range(ra_range=ra_range, dec_range=dec_range)

#     print('Minimum image value {}'.format(np.min(fits_image.signal_arr)))
#     print('Maximum image value {}'.format(np.max(fits_image.signal_arr)))

#     fig = plt.subplots(figsize=(16,16))
    if log_scale is True:
        data = np.log10(fits_image.signal_arr)
        print('Plotting on logarithmic scale')
        colorbar_label = 'log(Flux Density)'
    else:
        data = fits_image.signal_arr
    print(np.min(data))
    print(np.max(data))
    im = ax.imshow(
        data, origin='lower', interpolation='none',
        cmap='Greys_r',
        extent=[
            fits_image.ra_axis[0], fits_image.ra_axis[-1],
            fits_image.dec_axis[0], fits_image.dec_axis[-1]
            ],
        vmin=colorbar_range[0], vmax=colorbar_range[1], aspect='auto'
    )
    ax.axis('equal')
#     ax.set_facecolor('gray')  # make plot background gray
#     ax.set_facecolor('black')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if plot_grid:
        ax.grid(which='both', zorder=10, lw=0.5)
    cbar = plt.colorbar(im,ax=ax,pad=0.01)
    # Label colorbar:
    cbar.ax.set_ylabel(colorbar_label, rotation=270, labelpad=15)
    bbox = ax.get_window_extent()
    width, height = bbox.width, bbox.height
    ax.margins(0)
#     ax.set_size_inches(2, 2)
#     print(f'{width},{height}')
#     if save_filename == '':
#         plt.show()
#     else:
#         print('Saving figure to {}'.format(save_filename))
#         plt.savefig(save_filename, format='png', dpi=500)
#     if write_pixel_coordinates is True:
#         coords = np.array([fits_image.ra_axis, fits_image.dec_axis])
#         coords = coords.T
#         coordpath = prefix + 'pixel_coordinates.txt'
#         print('Writing out pixel coordinates in RA/DEC to ' + coordpath)
#         with open(coordpath, 'w+') as f:
#             np.savetxt(f, coords, fmt=['%f','%f'], header='RA        DEC')
    return im

def plot_sky_map(uvd,ax,ra_pad=20,dec_pad=30,clip=True,fwhm=11,nx=300,ny=200,sources=[]):
    map_path = f'/lustre/aoc/projects/hera/dstorer/Setup/djsScripts/Scripts/haslam408_dsds_Remazeilles2014.fits'
    hdulist = fits.open(map_path)

    # Set up the HEALPix projection
    nside = hdulist[1].header['NSIDE']
    order = hdulist[1].header['ORDERING']
    hp = HEALPix(nside=nside, order=order, frame=Galactic())
    
    #Get RA/DEC coords of observation
    loc = EarthLocation.from_geocentric(*uvd.telescope_location, unit='m')
    time_array = uvd.time_array
    obstime_start = Time(time_array[0],format='jd',location=loc)
    obstime_end = Time(time_array[-1],format='jd',location=loc)
    zenith_start = sc(Angle(0, unit='deg'),Angle(90,unit='deg'),frame='altaz',obstime=obstime_start,location=loc)
    zenith_start = zenith_start.transform_to('icrs')
    zenith_end = sc(Angle(0, unit='deg'),Angle(90,unit='deg'),frame='altaz',obstime=obstime_end,location=loc)
    zenith_end = zenith_end.transform_to('icrs')
    lst_start = obstime_start.sidereal_time('mean').hour
    lst_end = obstime_end.sidereal_time('mean').hour
    start_coords = [zenith_start.ra.degree,zenith_start.dec.degree]
    if start_coords[0] > 180:
        start_coords[0] = start_coords[0] - 360
    end_coords = [zenith_end.ra.degree,zenith_end.dec.degree]
    if end_coords[0] > 180:
        end_coords[0] = end_coords[0] - 360
    
    # Sample a 300x200 grid in RA/Dec
    ra_range = [zenith_start.ra.degree-ra_pad, zenith_end.ra.degree+ra_pad]
    dec_range = [zenith_start.dec.degree-ra_pad, zenith_end.dec.degree+ra_pad]
    if clip == True:
        ra = np.linspace(ra_range[0],ra_range[1], nx)
        dec = np.linspace(dec_range[0],dec_range[1], ny)
    else:
        ra = np.linspace(-180,180,nx)
        dec = np.linspace(-90,zenith_start.dec.degree+90,ny)
    ra_grid, dec_grid = np.meshgrid(ra * u.deg, dec * u.deg)
    
    #Create alpha grid
    alphas = np.ones(ra_grid.shape)
    alphas = np.multiply(alphas,0.5)
    ra_min = np.argmin(np.abs(np.subtract(ra,start_coords[0]-fwhm/2)))
    ra_max = np.argmin(np.abs(np.subtract(ra,end_coords[0]+fwhm/2)))
    dec_min = np.argmin(np.abs(np.subtract(dec,start_coords[1]-fwhm/2)))
    dec_max = np.argmin(np.abs(np.subtract(dec,end_coords[1]+fwhm/2)))
    alphas[dec_min:dec_max, ra_min:ra_max] = 1

    # Set up Astropy coordinate objects
    coords = sc(ra_grid.ravel(), dec_grid.ravel(), frame='icrs')

    # Interpolate values
    temperature = healpy.read_map(map_path)
    tmap = hp.interpolate_bilinear_skycoord(coords, temperature)
    tmap = tmap.reshape((ny, nx))
    tmap = np.flip(tmap,axis=1)
    alphas = np.flip(alphas,axis=1)

    # Make a plot of the interpolated temperatures
#     plt.figure(figsize=(12, 7))
    im = ax.imshow(tmap, extent=[ra[-1], ra[0], dec[0], dec[-1]], 
                    cmap=plt.cm.viridis, aspect='auto', vmin=10,vmax=40,alpha=alphas,origin='lower')
    ax.set_xlabel('RA (ICRS)')
    ax.set_ylabel('DEC (ICRS)')
    ax.hlines(y=start_coords[1]-fwhm/2,xmin=ra[-1],xmax=ra[0],linestyles='dashed')
    ax.hlines(y=start_coords[1]+fwhm/2,xmin=ra[-1],xmax=ra[0],linestyles='dashed')
    ax.vlines(x=start_coords[0],ymin=start_coords[1],ymax=dec[-1],linestyles='dashed')
    ax.vlines(x=end_coords[0],ymin=start_coords[1],ymax=dec[-1],linestyles='dashed')
    ax.annotate(np.around(lst_start,2),xy=(start_coords[0],dec[-1]),xytext=(0,8),
                 fontsize=10,xycoords='data',textcoords='offset points',horizontalalignment='center')
#     plt.annotate(np.around(lst_end,2),xy=(end_coords[0],dec[-1]),xytext=(0,8),
#                  fontsize=10,xycoords='data',textcoords='offset points',horizontalalignment='center')
    ax.annotate('LST (hours)',xy=(np.average([start_coords[0],end_coords[0]]),dec[-1]),
                xytext=(0,22),fontsize=10,xycoords='data',textcoords='offset points',horizontalalignment='center')
    for s in sources:
        if s[1] > dec[0] and s[1] < dec[-1]:
            if s[0] > 180:
                s = (s[0]-360,s[1],s[2])
            if s[2] == 'LMC' or s[2] == 'SMC':
                ax.annotate(s[2],xy=(s[0],s[1]),xycoords='data',fontsize=8,xytext=(20,-20),
                             textcoords='offset points',arrowprops=dict(facecolor='black', shrink=2,width=1,
                                                                        headwidth=4))
            else:
                ax.scatter(s[0],s[1],c='k',s=6)
                if len(s[2]) > 0:
                    ax.annotate(s[2],xy=(s[0]+3,s[1]-4),xycoords='data',fontsize=6)
#     plt.show()
#     plt.close()
#     hdulist.close()
    return im

def gather_source_list():
    sources = []
    sources.append((50.6750,-37.2083,'Fornax A'))
    sources.append((201.3667,-43.0192,'Cen A'))
    # sources.append((83.6333,22.0144,'Taurus A'))
    sources.append((252.7833,4.9925,'Hercules A'))
    sources.append((139.5250,-12.0947,'Hydra A'))
    sources.append((79.9583,-45.7789,'Pictor A'))
    sources.append((187.7042,12.3911,'Virgo A'))
    sources.append((83.8208,-59.3897,'Orion A'))
    sources.append((80.8958,-69.7561,'LMC'))
    sources.append((13.1875,-72.8286,'SMC'))
    sources.append((201.3667,-43.0192,'Cen A'))
    sources.append((83.6333,20.0144,'Crab Pulsar'))
    sources.append((128.8375,-45.1764,'Vela SNR'))
    cat_path = f'/lustre/aoc/projects/hera/dstorer/Setup/djsScripts/Scripts/G4Jy_catalog.tsv'
    cat = open(cat_path)
    f = csv.reader(cat,delimiter='\n')
    for row in f:
        if len(row)>0 and row[0][0]=='J':
            s = row[0].split(';')
            tup = (float(s[1]),float(s[2]),'')
            sources.append(tup)
    return sources

# if __name__ == '__main__':

#     #prefix = '/Users/dstorer/Files/FHD_Pyuvsim_comp/TestingSuite/AdjustedPsfDim/56/fhd_djs_simComp_offzen5d_gauss_psfDim56_Aug2019/output_data/'
#     #filename = 'offzenith5d_gauss_Beam_XX'
#     prefix = '/Users/dstorer/Files/FHD_Pyuvsim_comp/fhd_djs_simComp_ref_1.1_gauss_beamAdjustmens_Aug2019/output_data/'
#     filename = 'ref_1.1_gauss_uniform_Residual_XX'
#     log_scale = False
#     data = load_image(prefix + filename + '.fits')
#     if log_scale is True:
#         output_path = prefix + filename + '_LOGIMAGE.png'
#     else:
#         output_path = prefix + filename + '_IMAGE.png'
#     color_scale = [-0.01,0.01]
#     write_pixel_coordinates=False
#     plot_fits_image(data, color_scale, output_path, prefix, write_pixel_coordinates, log_scale)