function to_hdf(filename::String, sol::Sol, vl::VideoLoader)
    HDF5.h5open(filename, "w") do fid
        fid["/footprints"] = Array(sol.A)
        fid["/initImg"] = reshape(Array(sol.I), framesize(vl))
        for video_i = 1:nvideos(vl)
            seg_i = first(vl.source_loader.source_loader.segs_per_video[video_i])
            key = match(r"animal[0-9].*day[0-9]+",
                        VideoLoaders.filename(vl, seg_i)).match
            fr = framerange_video(vl, video_i)
            fid["/traces/$key/raw"] = Array(sol.R[fr, :])
            fid["/traces/$key/fitted"] = Array(sol.C[fr, :])
            fid["/traces/$key/deconvolved"] = Array(sol.S[fr, :])
        end
        fid["/params/gammas"] = sol.gammas
        fid["/params/lambdas"] = sol.lambdas
        #fid["/params/colors"] = sol.colors
        #TODO: save backgrounds?
    end
end