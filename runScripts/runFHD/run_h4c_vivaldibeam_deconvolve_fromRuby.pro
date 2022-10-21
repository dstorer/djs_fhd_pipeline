PRO run_h4c_vivaldibeam,_Extra=extra
  except=!except
  !except=0
  heap_gc

  args = command_line_args(count=nargs)
  print, nargs
  print, args[0]
  print, args[1]  

  instrument = 'hera'
  output_directory = args[2]
  
  recalculate_all = 1
  uvfits_version = 5
  uvfits_subversion = 1
  max_deconvolution_components = 200000
  calibration_catalog_file_path = "/lustre/aoc/projects/hera/dstorer/Setup/FHD/catalog_data/GLEAM_v2_plus_rlb2019.sav"
  max_cal_iter = 1000L ;increase max calibration iterations to ensure convergence
  gain_factor = 0.1
  deconvolve = 1
  return_decon_visibilities = 1
  deconvolution_filter = 'filter_uv_uniform'
  filter_background = 1
  return_cal_visibilities = 1
  diffuse_calibrate = 0
  diffuse_model = 0
  cal_bp_transfer = 0
  rephase_weights = 0
  restrict_hpx_inds = 0
  hpx_radius = 15
  subtract_sidelobe_catalog = "/lustre/aoc/projects/hera/dstorer/Setup/FHD/catalog_data/GLEAM_v2_plus_rlb2019.sav"
  return_sidelobe_catalog = 1
  dft_threshold = 0
  ring_radius = 0
  write_healpix_fits = 1
  debug_region_grow = 0
  n_pol = 4
  time_cut = -4 ;flag an extra 4 seconds from the end of each obs
  snapshot_healpix_export = 0 ;don't need eppsilon inputs

  ;beam_nfreq_avg=1
  ;recalculate_all=1
  no_png=1
  version=args[1]
  ;psf_dim=28
  ;image_filter_fn='filter_uv_uniform' ;applied ONLY to output images

  vis_file_list=args[0]

  ;dimension=1024.
  ;max_sources=100000.
  ;pad_uv_image=1.
  ;precess=0 ;set to 1 ONLY for X16 PXX scans (i.e. Drift_X16.pro)
  ;FoV=45
  no_ps=1 ;don't save postscript copy of images
  ;gain_factor=0.1
  ;min_baseline=1.
  ;min_cal_baseline=25.
  ;silent=0
  ;smooth_width=32.
  ;ps_kbinsize=0.5
  ;ps_kspan=600.
  ;split_ps=1
  ;save_vis=1
  ;no_rephase=1
  ;mark_zenith=1
  ;n_pol=2
  ;restore_vis_savefile=0
  ;max_cal_iter=1000L
  ;use Fagnoni vivaldi efield beam model
  beam_model_version=4
  ;dft_threshold=1
  ;init_healpix
  fhd_file_list=fhd_path_setup(vis_file_list,version=version,output_directory=output_directory,_Extra=extra)
  healpix_path=fhd_path_setup(output_dir=output_directory,subdir='Healpix',output_filename='Combined_obs',version=version,_Extra=extra)

  IF N_Elements(extra) GT 0 THEN cmd_args=extra
  extra=var_bundle()
  general_obs,_Extra=extra
  !except=except
END
