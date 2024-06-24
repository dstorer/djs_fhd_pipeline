PRO plot_kxky_slices

    base_path="/lustre/aoc/projects/hera/dstorer/Projects/thesis/fhdRuns/intCrossNights/"
    ps_foldername="ps_1-64/"
    ;obs_name="Combined_obs_updatedHpx_64_cubeXX"
    obs_name="updatedHpx_1-64"
    ind=22

    cubes=["dirty","model","res"]
    pols=["xx","yy"]
    slice_files_str=""

    FOR p=0,1 DO BEGIN
        FOR c=0,2 DO BEGIN
            out_path = base_path + ps_foldername + "data/kspace_cubes/slices/" + obs_name + "__even_odd_joint_noimgclip_" + cubes[c] + "_" +$ 
            pols[p] + "_dft_averemove_swbh_incoherent_avg_xy_plane_sliceInd" + strn(ind) + ".idlsave"
            in_path = base_path + ps_foldername + "data/kspace_cubes/" + obs_name + "__even_odd_joint_noimgclip_" + cubes[c] + "_" +$ 
            pols[p] + "_dft_averemove_swbh_incoherent_avg_power.idlsave"
            print, in_path
            restore, in_path
            pslices = kpower_slice(power_3d, kx_mpc, ky_mpc, kz_mpc, kperp_lambda_conv,delay_params, hubble_param, slice_axis=2, slice_inds=ind,$ 
            slice_savefile=out_path, noise_3d=noise_3d, noise_expval_3d=noise_expval_3d, weights_3d=weights_3d) 

            slice_files_str = slice_files_str + out_path + ","
        ENDFOR
    ENDFOR
    print, slice_files_str
END

