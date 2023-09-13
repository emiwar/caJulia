import CSV
import HDF5
import CUDA
import Colors
import Distributions
import LinearAlgebra
using ProgressMeter
include("videoLoaders/videoLoaders.jl")
s="/mnt/dmclab/Emil/arrowmaze_raw_downsampled/animal6_v2_with_mcwindows.csv"
s="/mnt/dmclab/Emil/arrowmaze_raw_downsampled/animal4_with_mcwindows.csv"
nsplits=100
pathPrefix = s[1:end-length(split(s, "/")[end])]
videolist = CSV.File(s,comment="#")#DataFrames.DataFrame(CSV.File(s,comment="#"))
sources = map(videolist) do r #map(eachrow(videolist))
    hdfLoader = VideoLoaders.HDFLoader(pathPrefix * r.filename, r.hdfKey)
    VideoLoaders.SplitLoader(hdfLoader, nsplits)
end
segs_per_video = [((i-1)*nsplits+1):i*nsplits for i=1:length(sources)]
output_labels = String.(map(r->r.resultKey, videolist))
baseLoader = VideoLoaders.MultiVideoLoader(sources, segs_per_video, output_labels)
hostCache = VideoLoaders.CachedHostLoader(baseLoader; max_memory=3.2e10)
filterLoader = VideoLoaders.BandpassFilterLoader(hostCache, 1, 300)
#filterloader = VideoLoaders.FilterLoader(hostcache, filterkernel)
#filterloader = BandpassFilterLoader(hostcache, 2, 100)
mcwindows = map(enumerate(videolist)) do (i, r) #map(eachrow(videolist))
    (r.mcwindowxmin:r.mcwindowxmax,
     r.mcwindowymin:r.mcwindowymax,
     VideoLoaders.framerange_video(filterLoader, i)[1:1000])
end
mcLoader = VideoLoaders.MotionCorrectionLoader(filterLoader, mcwindows)
deviceCache = VideoLoaders.CachedDeviceLoader(mcLoader, max_memory=5e9)

HDF5.h5open("animal6_semimanual_mc.h5", "r") do fid
    VideoLoaders.loadfromhdf(mcLoader, fid)
end

using .VideoLoaders
include("solution_struct.jl")
include("negentropy_img.jl")
include("viz.jl")
include("initROIs.jl")
include("fastHALS.jl")
include("oasis_opt.jl")
include("merge_split.jl")
include("backgrounds/backgrounds.jl")
include("save_result.jl")

#sol = Sol(deviceCache);
#sol.I = negentropy_img(deviceCache);
#to_hdf("animal6_semimanual_mc_initImg.h5", sol, deviceCache)
sol = from_hdf("animal6_semimanual_mc_initImg.h5", deviceCache)

initA!(sol; callback=(m,i,n)->"$m ($i/$N)")
Ah = Array(sol.A)
sol.A = SparseArrays.sparse(Ah) |> CUDA.cu
selection = .!any(isnan.(Ah), dims=2)[:]
sol.A = CUDA.cu(SparseArrays.sparse(Ah[selection, :]))
zeroTraces!(sol)
to_hdf("animal6_semimanual_mc_inited_cells.h5", sol, deviceCache)
#sol = from_hdf("animal6_semimanual_mc_inited_cells.h5", deviceCache)

callback=(m,i,N)->println("$m ($i/$N)")
initBackgrounds!(deviceCache, sol; callback);
to_hdf("animal6_semimanual_mc_bgs_inited.h5", sol, deviceCache)
updateTraces!(deviceCache, sol, deconvFcn! = oasis_opt!; callback);
to_hdf("animal6_semimanual_mc_first_temporal.h5", sol, deviceCache)
#sol = from_hdf("animal6_semimanual_mc_first_temporal.h5", deviceCache)
updateROIs!(deviceCache, sol; callback);
for i=1:3
    merge!(sol, thres=.6)
end
to_hdf("animal6_semimanual_mc_first_spatial.h5", sol, deviceCache)

updateTraces!(deviceCache, sol, deconvFcn! = oasis_opt!; callback);
to_hdf("animal6_semimanual_mc_second_temporal.h5", sol, deviceCache)
updateROIs!(deviceCache, sol; callback);
for i=1:3
    merge!(sol, thres=.6)
end
to_hdf("animal6_semimanual_mc_second_spatial.h5", sol, deviceCache)

updateTraces!(deviceCache, sol, deconvFcn! = oasis_opt!; callback);
to_hdf("animal6_semimanual_mc_third_temporal.h5", sol, deviceCache)
updateROIs!(deviceCache, sol; callback);
for i=1:3
    merge!(sol, thres=.6)
end
to_hdf("animal6_semimanual_mc_third_spatial.h5", sol, deviceCache)






import Colors
import GLMakie
include("gui_module.jl")
#sol = from_hdf("animal6_semimanual_mc_second_temporal.h5", deviceCache)

gui = GUI.GUIState(deviceCache);
display(gui.fig)
#gui.sol[] = sol

mc_result = VideoLoaders.fitMotionCorrection_v2!(mcLoader;
                                                 callback=(m,i,n)->println("$m ($i/$n)"));
to_hdf("animal4_semimanual_mc.h5", gui.sol[], deviceCache)


new_windows = [(450:600, 280:400, framerange_video(mcLoader, i)) for i=1:nvideos(mcLoader)]

vl = mcLoader
first_frames = map(new_windows) do win
    VideoLoaders.readframe(vl.source_loader, win[3][1])[win[1:2]...]
end
callback=(m,i,n)->println("$m ($i/$n)")
prelim_callback = (m, i, N)->callback("[Special] $m", i, N)
special_results = VideoLoaders.fitToFrame(vl, first_frames, new_windows;
                                         callback=prelim_callback)
to_hdf("animal4_semimanual_mc_v2.h5", gui.sol[], deviceCache)