PRO run_h4c_vivaldibeam_versions,_Extra=extra
  except=!except
  !except=0
  heap_gc

  ; parse command line args
  args = command_line_args(count=nargs)
  print, nargs
  vis_file_list=args[0]
  print, args[0]
  version=args[1]
  print, args[1]
  output_directory = args[2]
  case_name=args[3]

  instrument = 'hera'

  beam_nfreq_avg=1
  no_frequency_flagging=1

  calibrate_visibilities=1
  cal_time_average=0
  flag_calibration=0
  recalculate_all=1
  no_png=1
  cleanup=0
  ps_export=0
  
  psf_dim=28
  image_filter_fn='filter_uv_uniform' ;applied ONLY to output images

  catalog_file_path="/lustre/aoc/projects/hera/dstorer/Setup/FHD/catalog_data/GLEAM_v2_plus_rlb2019.sav"
  calibration_catalog_file_path="/lustre/aoc/projects/hera/dstorer/Setup/FHD/catalog_data/GLEAM_v2_plus_rlb2019.sav"

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
  
  case case_name of
  
    'test_versions': begin
        min_cal_baseline=200
    end
    
    'medResFlaggingOnly_ISO_perSetPhasing_40lambdaCut': begin
        min_cal_baseline=40
    end
    
    'medResFlaggingOnly_ISO_perSetPhasing_25lambdaCut': begin
        min_cal_baseline=25
    end
    
    'medResFlaggingOnly_ISO_perSetPhasing_40lambdaCut_gainInit': begin
        min_cal_baseline=40
        calibration_auto_initialize=0
        cal_gain_init=10
    end
    
    'incAutos_forEllie': begin
        min_cal_baseline=25
        calibration_auto_initialize=1
    end
    
    'incAutos_testAnts': begin
        min_cal_baseline=25
        calibration_auto_initialize=1
    end
    
    'excAutos_testAnts': begin
        min_cal_baseline=25
        calibration_auto_initialize=1
    end
    
    'testLowerConvThresh': begin
        min_cal_baseline=25
        calibration_auto_initialize=0
        cal_convergence_threshold=1E-6
    end
    
    'lowerConvThresh_minCal25': begin
        min_cal_baseline=25
        calibration_auto_initialize=0
        cal_convergence_threshold=1E-6
    end
    
    'lowerConvThresh_minCal40': begin
        min_cal_baseline=40
        calibration_auto_initialize=0
        cal_convergence_threshold=1E-6
    end
    
    'lowerConvThresh_minCal40_higherPhaseFitIter': begin
        min_cal_baseline=40
        calibration_auto_initialize=0
        cal_convergence_threshold=1E-6
        phase_fit_iter=16
    end
    
    'lowerConvThresh_minCal40_higherPhaseIter': begin
        min_cal_baseline=40
        calibration_auto_initialize=0
        cal_convergence_threshold=1E-6
        phase_iter=16
    end
    
    'calGainInit200': begin
        min_cal_baseline=40
        calibration_auto_initialize=0
        cal_gain_init=200.
        cal_convergence_threshold=1E-6
        phase_iter=16
    end
    
    'calPolyfitOn_deg1': begin
        min_cal_baseline=40
        calibration_auto_initialize=0
        cal_convergence_threshold=1E-6
        phase_iter=16
        calibration_polyfit=1
        cal_phase_degree_fit=1
    end
    
    'calPolyfitOn_deg2': begin
        min_cal_baseline=40
        calibration_auto_initialize=0
        cal_convergence_threshold=1E-6
        phase_iter=16
        calibration_polyfit=1
        cal_phase_degree_fit=2
    end
    
    'mediumRestrictive_intersnap_caltimeaverage1': begin
        cal_time_average=1
    end
    
    'medRes_IS_startFromPrevGains': begin
        cal_time_average=1
        start_gain_file = '/lustre/aoc/projects/hera/dstorer/Projects/updatedHeraOnFHD/2459855/fhdOutput/mediumRestrictive_intersnapOnly/fhd_2459855.6033653542_mediumRestrictive_intersnap_caltimeaverage1_mid_3/calibration/zen.2459855.6033653547_mid_10obs_1_cal.sav'
        cal=getvar_savefile(start_gain_file,'cal')
        gain_arr_ptr=cal.gain
    end
    
    'medRes_IS_startFromGains1': begin
        cal_time_average=1
    end
    
    'medRes_IS_useInitialCalibration': begin
        cal_time_average=1
        initial_calibration='/lustre/aoc/projects/hera/dstorer/Projects/updatedHeraOnFHD/2459855/fhdOutput/mediumRestrictive_intersnapOnly/fhd_2459855.6033653542_mediumRestrictive_intersnap_caltimeaverage1_mid_3/calibration/zen.2459855.6033653547_mid_10obs_1_cal.sav'
    end

  endcase

  ;default_diffuse='D:\MWA\IDL_code\FHD\catalog_data\EoR0_polarized_diffuse_2.sav'
  ;IF N_Elements(extra) GT 0 THEN IF Tag_exist(extra,'diffuse_calibrate') THEN IF extra.diffuse_calibrate EQ 1 THEN $
  ;  extra=structure_update(extra,diffuse_calibrate=default_diffuse)
  ;IF N_Elements(extra) GT 0 THEN IF Tag_exist(extra,'diffuse_model') THEN IF extra.diffuse_model EQ 1 THEN BEGIN
  ;  extra=structure_update(extra,diffuse_model=default_diffuse)
  ;  IF ~(Tag_exist(extra,'model_visibilities') OR (N_Elements(model_visibilities) GT 0)) THEN model_visibilities=1
  ;ENDIF
  ;undefine_fhd,default_diffuse

  IF N_Elements(extra) GT 0 THEN cmd_args=extra
  extra=var_bundle()
  general_obs,_Extra=extra
  !except=except
END
