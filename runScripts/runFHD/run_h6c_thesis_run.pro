PRO run_h6c_thesis_run,_Extra=extra
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
  save_beam_metadata_only=1
  no_frequency_flagging=1

  calibrate_visibilities=1
  cal_time_average=0
  flag_calibration=0
  recalculate_all=1
  no_png=1
  cleanup=0

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
  output_residual_histogram=1
  show_beam_contour=0

  n_pol=2
  restore_vis_savefile=0
  max_cal_iter=1000L
  dft_threshold=0
  deconvolve=0
  init_healpix

  print, 'Running case: ' + case_name
  
  case case_name of

    'thesis_v2': begin
      save_uvf=0
      snapshot_healpix_export=0
      ps_export=0
      allow_sidelobe_sources=1

      beam_clip_floor=1
      mapfn_recalculate=0
      kbinsize=0.5
      dimension=1024.
      save_beam_metadata_only=1

      calibration_polyfit=1
      cable_bandpass_fit=0
      cal_amp_degree_fit=2
      cal_phase_degree_fit=1

      export_images=1
      no_ps=0
      no_png=0
      recalculate_all = 1
      return_cal_visibilities = 1
      rephase_weights = 01
      import_pyuvdata_beam_filepath='/lustre/aoc/projects/hera/dstorer/Setup/HERA-Beams/NicolasFagnoniBeams/NF_HERA_Vivaldi_efield_beam.fits'
      initial_calibration='/lustre/aoc/projects/hera/dstorer/Projects/updatedHeraOnFHD/2459906/855_initialCal.sav'
      version=case_name
    end
    
    'thesis_v1_noPolyfit': begin
      ;no_calibration_frequency_flagging=0
      ;no_frequency_flagging=0
      ;flag_calibration=0
      ;flag_visibilities=0
      save_uvf=0
      snapshot_healpix_export=0
      ps_export=0

      beam_clip_floor=1
      mapfn_recalculate=0
      kbinsize=0.5
      dimension=1024.
      ;beam_threshold=0.1
      save_beam_metadata_only=1

      calibration_polyfit=0
      cable_bandpass_fit=0
      cal_amp_degree_fit=0
      cal_phase_degree_fit=0

      sidelobe_subtract=0
      export_images=1
      no_ps=0
      no_png=0
      recalculate_all = 1
      return_cal_visibilities = 1
      rephase_weights = 01
      import_pyuvdata_beam_filepath='/lustre/aoc/projects/hera/dstorer/Setup/HERA-Beams/NicolasFagnoniBeams/NF_HERA_Vivaldi_efield_beam.fits'
      initial_calibration='/lustre/aoc/projects/hera/dstorer/Projects/updatedHeraOnFHD/2459906/855_initialCal.sav'
      version=case_name
    end
    
    'thesis_v1': begin
      ;no_calibration_frequency_flagging=0
      ;no_frequency_flagging=0
      ;flag_calibration=0
      ;flag_visibilities=0
      save_uvf=0
      snapshot_healpix_export=0
      ps_export=0

      beam_clip_floor=1
      mapfn_recalculate=0
      kbinsize=0.5
      dimension=1024.
      ;beam_threshold=0.1
      save_beam_metadata_only=1

      calibration_polyfit=1
      cable_bandpass_fit=0
      cal_amp_degree_fit=2
      cal_phase_degree_fit=1

      sidelobe_subtract=0
      export_images=1
      no_ps=0
      no_png=0
      recalculate_all = 1
      return_cal_visibilities = 1
      rephase_weights = 01
      import_pyuvdata_beam_filepath='/lustre/aoc/projects/hera/dstorer/Setup/HERA-Beams/NicolasFagnoniBeams/NF_HERA_Vivaldi_efield_beam.fits'
      initial_calibration='/lustre/aoc/projects/hera/dstorer/Projects/updatedHeraOnFHD/2459906/855_initialCal.sav'
      version=case_name
    end
    
    'og_version': begin
      no_calibration_frequency_flagging=0
      no_frequency_flagging=0
      flag_calibration=0
      flag_visibilities=0
      save_uvf=0
      snapshot_healpix_export=0
      ps_export=0

      beam_clip_floor=1
      mapfn_recalculate=0
      kbinsize=0.5
      dimension=1024.
      beam_threshold=0.1
      deconvolve=0
      save_beam_metadata_only=1
      n_pol=2

      calibration_polyfit=1
      cable_bandpass_fit=0
      cal_amp_degree_fit=2
      cal_phase_degree_fit=1

      sidelobe_subtract=0
      export_images=1
      no_ps=0
      no_png=0
      recalculate_all = 1
      return_cal_visibilities = 1
      rephase_weights = 01
      import_pyuvdata_beam_filepath='/lustre/aoc/projects/hera/dstorer/Setup/HERA-Beams/NicolasFagnoniBeams/NF_HERA_Vivaldi_efield_beam.fits'
      initial_calibration='/lustre/aoc/projects/hera/dstorer/Projects/updatedHeraOnFHD/2459906/855_initialCal.sav'
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