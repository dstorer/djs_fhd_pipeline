pro plot_uvf_cubes

    test=4
    print, test
    ;jds=["2459890"]
    ;jds=["2459890","2459891","2459892","2459893","2459894","2459895","2459896","2459897","2459898","2459900","2459902","2459903",$
    ;"2459904","2459905","2459906","2459907","2459908"]
    ;obs=["zen.2459890.3513525105_mid_clip_4obs_17","zen.2459891.348600048_mid_clip_4obs_17","zen.2459892.3458685395_mid_clip_4obs_17",$
    ;"zen.2459893.3431259487_mid_clip_4obs_17","zen.2459894.3403953565_mid_clip_4obs_17","zen.2459895.3376620007_mid_clip_4obs_17",$
    ;"zen.2459896.334926757_mid_clip_4obs_17","zen.2459897.3321848707_mid_clip_4obs_17","zen.2459898.329438017_mid_clip_4obs_17",$
    ;"zen.2459900.3239666633_mid_clip_4obs_17","zen.2459902.318492469_mid_clip_4obs_17","zen.2459903.31576264_mid_clip_4obs_17",$
    ;"zen.2459904.313456405_mid_clip_4obs_17","zen.2459905.310278673_mid_clip_4obs_17","zen.2459906.3075362407_mid_clip_4obs_17",$
    ;"zen.2459907.304804443_mid_clip_4obs_17","zen.2459908.3020635047_mid_clip_4obs_17"]
    obs=["Combined_obs_thesis_v2_MR_freqClip_withInit_wholeNight_2459890","Combined_obs_thesis_v2_MR_freqClip_withInit_wholeNight_2459891",$
    "Combined_obs_thesis_v2_MR_freqClip_withInit_wholeNight_2459892","Combined_obs_thesis_v2_MR_freqClip_withInit_wholeNight_2459893",$
    "Combined_obs_thesis_v2_MR_freqClip_withInit_wholeNight_2459894","Combined_obs_thesis_v2_MR_freqClip_withInit_wholeNight_2459895",$
    "Combined_obs_thesis_v2_MR_freqClip_withInit_wholeNight_2459896","Combined_obs_thesis_v2_MR_freqClip_withInit_wholeNight_2459897",$
    "Combined_obs_thesis_v2_MR_freqClip_withInit_wholeNight_2459898","Combined_obs_thesis_v2_MR_freqClip_withInit_wholeNight_2459900",$
    "Combined_obs_thesis_v2_MR_freqClip_withInit_wholeNight_2459902","Combined_obs_thesis_v2_MR_freqClip_withInit_wholeNight_2459903",$
    "Combined_obs_thesis_v2_MR_freqClip_withInit_wholeNight_2459904","Combined_obs_thesis_v2_MR_freqClip_withInit_wholeNight_2459905",$
    "Combined_obs_thesis_v2_MR_freqClip_withInit_wholeNight_2459906","Combined_obs_thesis_v2_MR_freqClip_withInit_wholeNight_2459907",$
    "Combined_obs_thesis_v2_MR_freqClip_withInit_wholeNight_2459908"]
    jds=[2459890,2459891,2459892,2459893,2459894,2459895,2459896,2459897,2459898,2459900,2459902,2459903,2459904,2459905,2459906,2459907,2459908]
    cubes=["cubeXX_noimgclip_model_uvf","cubeYY_noimgclip_model_uvf",$
    "cubeXX_noimgclip_weights_uvf","cubeYY_noimgclip_weights_uvf",$
    "cubeXX_noimgclip_dirty_uvf","cubeYY_noimgclip_dirty_uvf"]
    exdir="/lustre/aoc/projects/hera/dstorer/Projects/thesis/fhdRuns"
    version = "fhd_thesis_v2_MR_freqClip_withInit"
    njds = n_elements(jds)
    ;print, "running for " + njds + "jds"
    ncubes = n_elements(cubes)
    print, "ncubes:"
    print, ncubes
    FOR j=0,njds-1 DO BEGIN
        jd=Strn(jds[j])
        print, "JD:"
        print, jd
        path_base = exdir + '/' + jd + '/fhdOutput/' + version
        print, "path base:"
        print, path_base
        FOR c=0,ncubes-1 DO BEGIN
            cube = cubes[c]
            print, cube
            
            path = path_base + '/ps/data/uvf_cubes/' + obs[j] + '_' + 'even_' + cube + '.idlsave'
            print, path
            restore, path
            de0 = data_cube[*,*,0]
            savefile = path_base + '/ps/plots/uvf_cube_plots/' + obs[j] + '_even_' + cube + '_slice_freq0'
            print, "savefile:"
            print, savefile
            quick_image, abs(de0), savefile=savefile, data_range=[100,100000000], log=1
            de1 = data_cube[*,*,100]
            savefile = path_base + '/ps/plots/uvf_cube_plots/' + obs[j] + '_even_' + cube + '_slice_freq100'
            quick_image, abs(de1), savefile=savefile, data_range=[100,100000000], log=1, png=1
            
            path = path_base + '/ps/data/uvf_cubes/' + obs[j] + '_' + 'odd_' + cube + '.idlsave'
            restore, path
            do0 = data_cube[*,*,0]
            savefile = path_base + '/ps/plots/uvf_cube_plots/' + obs[j] + '_odd_' + cube + '_slice_freq0'
            quick_image, abs(do0), savefile=savefile, data_range=[100,100000000], log=1, png=1
            do1 = data_cube[*,*,100]
            savefile = path_base + '/ps/plots/uvf_cube_plots/' + obs[j] + '_odd_' + cube + '_slice_freq100'
            quick_image, abs(do1), savefile=savefile, data_range=[100,100000000], log=1, png=1
            
            df0 = de0 - do0
            df1 = de1 - do1
            savefile = path_base + '/ps/plots/uvf_cube_plots/' + obs[j] + '_' + cube + '_evenOddDiff_slice_freq0'
            quick_image, abs(df0), savefile=savefile, data_range=[100,100000000], log=1, png=1
            savefile = path_base + '/ps/plots/uvf_cube_plots/' + obs[j] + '_' + cube + '_evenOddDiff_slice_freq100'
            quick_image, abs(df1), savefile=savefile, data_range=[100,100000000], log=1, png=1
        ENDFOR
    ENDFOR
END