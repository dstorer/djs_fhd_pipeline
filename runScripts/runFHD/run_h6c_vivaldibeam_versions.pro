PRO run_h6c_vivaldibeam_versions,_Extra=extra
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
  save_beam_metadata_only=0
  no_frequency_flagging=1

  calibrate_visibilities=1
  cal_time_average=0
  flag_calibration=0
  recalculate_all=1
  no_png=1
  cleanup=0
  ;ps_export=0 ; Commented out on 3/25/24

  catalog_file_path="/lustre/aoc/projects/hera/dstorer/Setup/FHD/catalog_data/GLEAM_v2_plus_rlb2019.sav"
  calibration_catalog_file_path="/lustre/aoc/projects/hera/dstorer/Setup/FHD/catalog_data/GLEAM_v2_plus_rlb2019.sav"

  calibration_auto_initialize=0

  allow_sidelobe_cal_sources=1

  combine_obs=0
  dimension=1024.
  max_sources=100000.
  pad_uv_image=1.
  precess=0 ;set to 1 ONLY for X16 PXX scans (i.e. Drift_X16.pro)
  kbinsize=0.5
  no_ps=1 ;don't save postscript copy of images
  min_baseline=1.
  min_cal_baseline=25.
  bandpass_calibrate=0
  calibration_polyfit=0
  save_vis=1
  no_restrict_cal_sources=1
  no_rephase=1
  mark_zenith=1
  psf_resolution=64.
  ;beam_diff_image=0
  ;beam_residual_threshold=0.1
  output_residual_histogram=1
  show_beam_contour=0

  n_pol=2
  restore_vis_savefile=0
  max_cal_iter=1000L
  dft_threshold=0
  init_healpix

  print, 'Running case: ' + case_name
  
  case case_name of

    'matchingFlags_writeHealpix_withPixels_minCal40_nff0': begin
      beam_clip_floor=1
      mapfn_recalculate=0
      kbinsize=0.5
      dimension=1024.
      beam_threshold=0.1
      deconvolve=0
      n_pol=2
      save_beam_metadata_only=0
      write_healpix_fits=1
      snapshot_healpix_export=1
      save_uvf=1
      restrict_hpx_inds='HERA_field0_3x_inds.idlsave'
      split_ps_export=1
      ps_export=1
      no_frequency_flagging=0
      no_calibration_frequency_flagging=0
      min_baseline=1.
      min_cal_baseline=40.

      sidelobe_subtract=0
      export_images=1
      recalculate_all = 0
      return_cal_visibilities = 1
      rephase_weights = 01
      import_pyuvdata_beam_filepath='/lustre/aoc/projects/hera/dstorer/Setup/HERA-Beams/NicolasFagnoniBeams/NF_HERA_Vivaldi_efield_beam.fits'
      initial_calibration='/lustre/aoc/projects/hera/dstorer/Projects/updatedHeraOnFHD/2459906/855_initialCal.sav'
    end
    
    'matchingFlags_writeHealpix_withPixels_minCal40': begin
      beam_clip_floor=1
      mapfn_recalculate=0
      kbinsize=0.5
      dimension=1024.
      beam_threshold=0.1
      deconvolve=0
      n_pol=2
      save_beam_metadata_only=0
      write_healpix_fits=1
      snapshot_healpix_export=1
      save_uvf=1
      restrict_hpx_inds='HERA_field0_3x_inds.idlsave'
      split_ps_export=1
      ps_export=1
      no_frequency_flagging=1
      no_calibration_frequency_flagging=1
      min_baseline=1.
      min_cal_baseline=40.

      sidelobe_subtract=0
      export_images=1
      recalculate_all = 0
      return_cal_visibilities = 1
      rephase_weights = 01
      import_pyuvdata_beam_filepath='/lustre/aoc/projects/hera/dstorer/Setup/HERA-Beams/NicolasFagnoniBeams/NF_HERA_Vivaldi_efield_beam.fits'
      initial_calibration='/lustre/aoc/projects/hera/dstorer/Projects/updatedHeraOnFHD/2459906/855_initialCal.sav'
    end
    
    'matchingFlags_writeHealpix_withPixels': begin
      beam_clip_floor=1
      mapfn_recalculate=0
      kbinsize=0.5
      dimension=1024.
      beam_threshold=0.1
      deconvolve=0
      n_pol=2
      save_beam_metadata_only=0
      write_healpix_fits=1
      snapshot_healpix_export=1
      save_uvf=1
      restrict_hpx_inds='HERA_field0_3x_inds.idlsave'
      split_ps_export=1
      ps_export=1

      sidelobe_subtract=0
      export_images=1
      recalculate_all = 0
      return_cal_visibilities = 1
      rephase_weights = 01
      import_pyuvdata_beam_filepath='/lustre/aoc/projects/hera/dstorer/Setup/HERA-Beams/NicolasFagnoniBeams/NF_HERA_Vivaldi_efield_beam.fits'
      initial_calibration='/lustre/aoc/projects/hera/dstorer/Projects/updatedHeraOnFHD/2459906/855_initialCal.sav'
    end
    
    'perPolAntFlagging_writeHealpix_withPixels': begin
      beam_clip_floor=1
      mapfn_recalculate=0
      kbinsize=0.5
      dimension=1024.
      beam_threshold=0.1
      deconvolve=0
      n_pol=2
      save_beam_metadata_only=0
      write_healpix_fits=1
      snapshot_healpix_export=1
      save_uvf=1
      restrict_hpx_inds='HERA_field0_3x_inds.idlsave'
      split_ps_export=1
      ps_export=1

      sidelobe_subtract=0
      export_images=1
      recalculate_all = 0
      return_cal_visibilities = 1
      rephase_weights = 01
      import_pyuvdata_beam_filepath='/lustre/aoc/projects/hera/dstorer/Setup/HERA-Beams/NicolasFagnoniBeams/NF_HERA_Vivaldi_efield_beam.fits'
      initial_calibration='/lustre/aoc/projects/hera/dstorer/Projects/updatedHeraOnFHD/2459906/855_initialCal.sav'
    end
    
    'perPolAntFlagging_testFreqFlagging': begin
      beam_clip_floor=1
      mapfn_recalculate=0
      kbinsize=0.5
      dimension=1024.
      beam_threshold=0.1
      deconvolve=0
      n_pol=2
      save_beam_metadata_only=1
      no_frequency_flagging=0

      sidelobe_subtract=0
      export_images=1
      recalculate_all = 0
      return_cal_visibilities = 1
      rephase_weights = 01
      import_pyuvdata_beam_filepath='/lustre/aoc/projects/hera/dstorer/Setup/HERA-Beams/NicolasFagnoniBeams/NF_HERA_Vivaldi_efield_beam.fits'
      initial_calibration='/lustre/aoc/projects/hera/dstorer/Projects/updatedHeraOnFHD/2459906/855_initialCal.sav'
    end
    
    'perPolAntFlagging': begin
      beam_clip_floor=1
      mapfn_recalculate=0
      kbinsize=0.5
      dimension=1024.
      beam_threshold=0.1
      deconvolve=0
      n_pol=2
      save_beam_metadata_only=1

      sidelobe_subtract=0
      export_images=1
      recalculate_all = 0
      return_cal_visibilities = 1
      rephase_weights = 01
      import_pyuvdata_beam_filepath='/lustre/aoc/projects/hera/dstorer/Setup/HERA-Beams/NicolasFagnoniBeams/NF_HERA_Vivaldi_efield_beam.fits'
      initial_calibration='/lustre/aoc/projects/hera/dstorer/Projects/updatedHeraOnFHD/2459906/855_initialCal.sav'
    end
    
    'perPolAntFlagging_writeHealpix': begin
      beam_clip_floor=1
      mapfn_recalculate=0
      kbinsize=0.5
      dimension=1024.
      beam_threshold=0.1
      deconvolve=0
      n_pol=2
      save_beam_metadata_only=0
      write_healpix_fits=1
      snapshot_healpix_export=1
      save_uvf=1
      restrict_hpx_inds=0

      sidelobe_subtract=0
      export_images=1
      recalculate_all = 0
      return_cal_visibilities = 1
      rephase_weights = 01
      import_pyuvdata_beam_filepath='/lustre/aoc/projects/hera/dstorer/Setup/HERA-Beams/NicolasFagnoniBeams/NF_HERA_Vivaldi_efield_beam.fits'
      initial_calibration='/lustre/aoc/projects/hera/dstorer/Projects/updatedHeraOnFHD/2459906/855_initialCal.sav'
    end
    
    'uvbeam_support_import_az_za_antSize14_855initialCal_testStokesIHack_perPolAntFlagging': begin
      beam_clip_floor=1
      mapfn_recalculate=0
      kbinsize=0.5
      dimension=1024.
      beam_threshold=0.1
      deconvolve=0
      n_pol=2
      save_beam_metadata_only=1

      sidelobe_subtract=0
      export_images=1
      recalculate_all = 0
      return_cal_visibilities = 1
      rephase_weights = 01
      import_pyuvdata_beam_filepath='/lustre/aoc/projects/hera/dstorer/Setup/HERA-Beams/NicolasFagnoniBeams/NF_HERA_Vivaldi_efield_beam.fits'
      initial_calibration='/lustre/aoc/projects/hera/dstorer/Projects/updatedHeraOnFHD/2459906/855_initialCal.sav'
    end
    
    'uvbeam_support_import_az_za_antSize14_855initialCal_testStokesIHack': begin
      beam_clip_floor=1
      mapfn_recalculate=0
      kbinsize=0.5
      dimension=1024.
      beam_threshold=0.1
      deconvolve=0
      n_pol=2
      save_beam_metadata_only=1

      sidelobe_subtract=0
      export_images=1
      recalculate_all = 0
      return_cal_visibilities = 1
      rephase_weights = 01
      import_pyuvdata_beam_filepath='/lustre/aoc/projects/hera/dstorer/Setup/HERA-Beams/NicolasFagnoniBeams/NF_HERA_Vivaldi_efield_beam.fits'
      initial_calibration='/lustre/aoc/projects/hera/dstorer/Projects/updatedHeraOnFHD/2459906/855_initialCal.sav'
    end
    
    'uvbeam_support_import_az_za_antSize14_855initialCal': begin
      beam_clip_floor=1
      mapfn_recalculate=0
      kbinsize=0.5
      dimension=1024.
      beam_threshold=0.1
      deconvolve=0

      sidelobe_subtract=0
      export_images=1
      recalculate_all = 0
      return_cal_visibilities = 1
      rephase_weights = 01
      import_pyuvdata_beam_filepath='/lustre/aoc/projects/hera/dstorer/Setup/HERA-Beams/NicolasFagnoniBeams/NF_HERA_Vivaldi_efield_beam.fits'
      initial_calibration='/lustre/aoc/projects/hera/dstorer/Projects/updatedHeraOnFHD/2459906/855_initialCal.sav'
    end
    
    'uvbeam_support_import_az_za_antSize14_initialCalSmooth': begin
      beam_clip_floor=1
      mapfn_recalculate=0
      kbinsize=0.5
      dimension=1024.
      beam_threshold=0.1
      deconvolve=0

      sidelobe_subtract=0
      export_images=1
      recalculate_all = 0
      return_cal_visibilities = 1
      rephase_weights = 01
      import_pyuvdata_beam_filepath='/lustre/aoc/projects/hera/dstorer/Setup/HERA-Beams/NicolasFagnoniBeams/NF_HERA_Vivaldi_efield_beam.fits'
      initial_calibration='/lustre/aoc/projects/hera/dstorer/Projects/updatedHeraOnFHD/2459906/906_initialCal_smooth.sav'
    end
    
    'uvbeam_support_import_az_za_antSize14_initialCalv2': begin
      beam_clip_floor=1
      mapfn_recalculate=0
      kbinsize=0.5
      dimension=1024.
      beam_threshold=0.1
      deconvolve=0

      sidelobe_subtract=0
      export_images=1
      recalculate_all = 0
      return_cal_visibilities = 1
      rephase_weights = 01
      import_pyuvdata_beam_filepath='/lustre/aoc/projects/hera/dstorer/Setup/HERA-Beams/NicolasFagnoniBeams/NF_HERA_Vivaldi_efield_beam.fits'
      initial_calibration='/lustre/aoc/projects/hera/dstorer/Projects/updatedHeraOnFHD/2459906/906_initialCal_v2.sav'
    end
    
    'uvbeam_support_import_az_za_antSize14_deconvolution': begin
      beam_clip_floor=1
      mapfn_recalculate=0
      kbinsize=0.5
      dimension=1024.
      beam_threshold=0.1
      deconvolve=1

      sidelobe_subtract=0
      export_images=1
      recalculate_all = 0
      return_cal_visibilities = 1
      rephase_weights = 01
      import_pyuvdata_beam_filepath='/lustre/aoc/projects/hera/dstorer/Setup/HERA-Beams/NicolasFagnoniBeams/NF_HERA_Vivaldi_efield_beam.fits'
      initial_calibration='/lustre/aoc/projects/hera/dstorer/Projects/updatedHeraOnFHD/2459906/906_initialCal.sav'
    end
    
    'uvbeam_support_import_az_za_antSize14': begin
      beam_clip_floor=1
      mapfn_recalculate=0
      kbinsize=0.5
      dimension=1024.
      beam_threshold=0.1
      deconvolve=0

      sidelobe_subtract=0
      export_images=1
      recalculate_all = 0
      return_cal_visibilities = 1
      rephase_weights = 01
      import_pyuvdata_beam_filepath='/lustre/aoc/projects/hera/dstorer/Setup/HERA-Beams/NicolasFagnoniBeams/NF_HERA_Vivaldi_efield_beam.fits'
      initial_calibration='/lustre/aoc/projects/hera/dstorer/Projects/updatedHeraOnFHD/2459906/906_initialCal.sav'
    end
    
    'cal_forEllie_noFov_antenna_size_22_it2': begin
      deconvolve=0
      beam_clip_floor=1
      kbinsize=0.5
      dimension=1024.
      beam_threshold=0.1

      sidelobe_subtract=0
      export_images=0
      recalculate_all = 0
      return_cal_visibilities = 1
      rephase_weights = 01
      import_pyuvdata_beam_filepath='/lustre/aoc/projects/hera/dstorer/Setup/HERA-Beams/NicolasFagnoniBeams/NF_HERA_Vivaldi_efield_beam.fits'
      initial_calibration='/lustre/aoc/projects/hera/dstorer/Projects/updatedHeraOnFHD/2459864/fhdOutput/cal_forEllie_noFov_antenna_size_22/fhd_2459864.3021791037_cal_forEllie_noFov_antenna_size_22_mid_4/calibration/zen.2459864.3021791037_mid_5obs_1_cal.sav'
    end
    
    'cal_forEllie_noFov_antenna_size_22': begin
      beam_clip_floor=1
      kbinsize=0.5
      dimension=1024.
      beam_threshold=0.1

      sidelobe_subtract=0
      export_images=0
      recalculate_all = 0
      return_cal_visibilities = 1
      rephase_weights = 01
      import_pyuvdata_beam_filepath='/lustre/aoc/projects/hera/dstorer/Setup/HERA-Beams/NicolasFagnoniBeams/NF_HERA_Vivaldi_efield_beam.fits'
      initial_calibration='/lustre/aoc/projects/hera/dstorer/Projects/updatedHeraOnFHD/2459906/fhdOutput/initialCalFrom906/fhd_2459906.306753304_medRes_IS_useInitialCalibration_from906_mid_97/calibration/zen.2459906.306753304_mid_5obs_24_cal.sav'
    end
    
    'cal_noFov_antenna_size_22': begin
      beam_clip_floor=1
      kbinsize=0.5
      dimension=1024.
      beam_threshold=0.1

      sidelobe_subtract=0
      export_images=0
      recalculate_all = 0
      ;uvfits_version = 5
      ;uvfits_subversion = 1
      return_cal_visibilities = 1
      rephase_weights = 01
      import_pyuvdata_beam_filepath='/Users/dstorer/repos/HERA-Beams/NicolasFagnoniBeams/NF_HERA_Vivaldi_efield_beam.fits'
      initial_calibration='/Users/dstorer/Documents/_Files/Dara/School/Graduate/RadCos/H6C_onFHD/testInitialCal/zen.2459855.6033653547_mid_10obs_1_cal.sav'
      output_directory='/Users/dstorer/Documents/_Files/Dara/School/Graduate/RadCos/H6C_onFHD/investigateBeamParameters'
      ;transfer_psf='/Users/dstorer/Documents/_Files/Dara/School/Graduate/RadCos/H6C_onFHD/investigateBeamParameters/fhd_cal_noFov_antenna_size_14'
      vis_file_list = '/Users/dstorer/Documents/_Files/Dara/School/Graduate/RadCos/H6C_onFHD/investigateBeamParameters/zen.2459906.2580993776_mid_5obs_2.uvfits'
      version=case_name
    end
    
  endcase
  
  fhd_file_list=fhd_path_setup(vis_file_list,version=version,output_directory=output_directory,_Extra=extra)
  healpix_path=fhd_path_setup(output_dir=output_directory,subdir='Healpix',output_filename='Combined_obs',version=version,_Extra=extra)

  IF N_Elements(extra) GT 0 THEN cmd_args=extra
  extra=var_bundle()
  general_obs,_Extra=extra
  !except=except
END