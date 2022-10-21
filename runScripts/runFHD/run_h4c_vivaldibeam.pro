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

  beam_nfreq_avg=1
  no_frequency_flagging=1

  calibrate_visibilities=1
  flag_calibration=0
  recalculate_all=1
  no_png=1
  cleanup=0
  ps_export=0
  version=args[1]
  psf_dim=28
  image_filter_fn='filter_uv_uniform' ;applied ONLY to output images

  vis_file_list=args[0]

  catalog_file_path="/lustre/aoc/projects/hera/dstorer/Setup/FHD/catalog_data/GLEAM_v2_plus_rlb2019.sav"
  calibration_catalog_file_path="/lustre/aoc/projects/hera/dstorer/Setup/FHD/catalog_data/GLEAM_v2_plus_rlb2019.sav"
  ;flag numbers should be =  HERA ant # + 1 to account for indexing differences
  ;tile_flag_list = ['3','12','13','14','24','25','37','38','39','44','47','51','52','53','54','59','60','61','66','67','68','69','75','76','93','94','95','109','110','111','112','127','128','129','130','81','82','83','84','85','86','87','101','102','116','120','121','123','136','137','138','139','140','141','142','143','145','146','155','156','157','158','159','160','161','165','166','176','178','179','180','183','184','185','186','187'] ;adjusted for indexing differences
  ;tile_flag_list = ['3', '12', '13', '14', '24', '25', '37', '38', '39', '47', '51', '52', '53', '54', '59', '60', '66', '67', '68', '69', '76', '82', '83', '84', '85', '87', '88', '89', '91', '92', '93', '94', '95', '102', '106', '108', '109', '110', '111', '112', '113', '117', '118', '119', '120', '121', '122', '123', '124', '125', '128', '129', '130', '131', '136', '137', '138', '139', '141', '142', '143', '144', '145', '146', '156', '157', '158', '162', '163', '164', '165', '166', '167', '177', '178', '179', '180', '181', '182', '183', '184', '185', '186', '187', '188']
  ;ant flags for 2459122 (w adjusted inds)
  ;tile_flag_list = ['177', '179', '182', '186', '157', '158', '159', '161', '163','164', '165', '166', '128', '129', '130', '131', '110', '93', '1','2', '3', '12', '13', '14', '15', '24', '25', '26', '27', '38','39', '40', '45', '46', '47', '52', '53', '60', '66', '68', '74','76', '82', '85', '87', '88', '89', '90', '91', '94', '95', '102','108', '111', '112', '113', '117', '120', '122', '123', '124','137', '138', '139', '143', '146', '156', '162', '167', '178','180', '181', '183', '184', '185', '187', '188']

  calibration_auto_initialize=1
  
  allow_sidelobe_cal_sources=1

  combine_obs=0
  dimension=1024.
  max_sources=100000.
  pad_uv_image=1.
  IF dimension GE 2048 THEN pad_uv_image=1.
  precess=0 ;set to 1 ONLY for X16 PXX scans (i.e. Drift_X16.pro)
  FoV=45
  no_ps=1 ;don't save postscript copy of images
  gain_factor=0.1
  min_baseline=1.
  min_cal_baseline=25.
  silent=0
  smooth_width=32.
  ;nfreq_avg=16
  ps_kbinsize=0.5
  ps_kspan=600.
  split_ps=1
  bandpass_calibrate=0
  calibration_polyfit=0
  cal_amp_degree_fit=2
  cal_phase_degree_fit=1
  save_vis=1
  no_restrict_cal_sources=1
  no_rephase=1
  mark_zenith=1
  psf_resolution=64.
  beam_diff_image=0
  beam_residual_threshold=0.1
  output_residual_histogram=1
  show_beam_contour=1
  contour_level=[0,0.01,0.05,0.1,0.2,0.5,0.67,0.9]
  contour_color='blue'

  default_diffuse='D:\MWA\IDL_code\FHD\catalog_data\EoR0_polarized_diffuse_2.sav'
  IF N_Elements(extra) GT 0 THEN IF Tag_exist(extra,'diffuse_calibrate') THEN IF extra.diffuse_calibrate EQ 1 THEN $
    extra=structure_update(extra,diffuse_calibrate=default_diffuse)
  IF N_Elements(extra) GT 0 THEN IF Tag_exist(extra,'diffuse_model') THEN IF extra.diffuse_model EQ 1 THEN BEGIN
    extra=structure_update(extra,diffuse_model=default_diffuse)
    IF ~(Tag_exist(extra,'model_visibilities') OR (N_Elements(model_visibilities) GT 0)) THEN model_visibilities=1
  ENDIF
  undefine_fhd,default_diffuse
  n_pol=2
  restore_vis_savefile=0
  firstpass=1
  max_cal_iter=1000L
  ;use Fagnoni vivaldi efield beam model
  beam_model_version=4
  dft_threshold=1
  init_healpix
  fhd_file_list=fhd_path_setup(vis_file_list,version=version,output_directory=output_directory,_Extra=extra)
  healpix_path=fhd_path_setup(output_dir=output_directory,subdir='Healpix',output_filename='Combined_obs',version=version,_Extra=extra)

  IF N_Elements(extra) GT 0 THEN cmd_args=extra
  extra=var_bundle()
  general_obs,_Extra=extra
  !except=except
END
