import Statistics
import CUDA
import cuDNN
import Images
import SparseArrays
import HDF5
import Colors
#import GLMakie
import Distributions
import LinearAlgebra
using ProgressMeter
#include("motion_correction.jl")
include("videoLoaders/videoLoaders.jl"); using .VideoLoaders
include("solution_struct.jl")
include("negentropy_img.jl")
include("viz.jl")
include("initROIs.jl")
include("fastHALS.jl")
include("oasis_opt.jl")
include("merge_split.jl")
include("backgrounds/backgrounds.jl")
#include("gui_module.jl")
include("save_result.jl")


example_files = ["20211016_163921_animal1learnday1.nwb",
                 "20211016_173112_animal3learnday1.nwb"]
nwbLoader = VideoLoaders.NWBLoader("../data/"*example_files[1])
#nwbLoader = VideoLoaders.NWBLoader("/mnt/dmclab/Vasiliki/motion corrected video calim inscopix example.nwb")
splitLoader = VideoLoaders.SplitLoader(nwbLoader, 50)
#alignedLoader = VideoLoaders.AlignedHDFLoader("../data/aligned_videos_animal3.csv", 10;
#                                 pathPrefix = "../data/")
hostCache = VideoLoaders.CachedHostLoader(splitLoader; max_memory=3.2e10)
#minSubtr = VideoLoaders.SubtractMinLoader(hostCache)
#VideoLoaders.calcmin!(minSubtr)
deviceCache = VideoLoaders.CachedDeviceLoader(hostCache; max_memory=1.6e10)

sol = Sol(deviceCache);
sol.I = negentropy_img(deviceCache);
initA!(sol; callback=(_,_,_)->nothing);
zeroTraces!(sol)

pows = mapreduce(y->(y, y^2, y^3, y^4), .+, deviceCache, (0.0, 0.0, 0.0, 0.0), dims=2)
maxproj = mapreduce(y->y, max, deviceCache, 0.0, dims=2)
N = Float64(nframes(deviceCache))

meanIm = reshape(Array(getindex.(pows, 1) ./ N), framesize(deviceCache))
stdIm = reshape(map(p->sqrt(p[2]/N - (p[1]/N)^2),Array(pows)), framesize(deviceCache))
maxIm = reshape(Array(maxproj), framesize(deviceCache))
pnrImg = (maxIm .- meanIm) ./ stdIm
negentIm = reshape(Array(sol.I), framesize(deviceCache))
roiIm = roiImg(sol)[1]

import Plots
Plots.gr()

allIms = [meanIm, stdIm, maxIm, pnrImg]
dpi = 200
clims = extrema.(allIms)
hms = Plots.heatmap.(allIms, colorbar=:bottom)
#Plots.colorbar!(hms, position=(0.5, -0.1))
Plots.plot(hms...; clims=reshape(clims, 1, 4),
           layout=(1, 4), aspect_ratio=:equal, xticks=[], yticks=[],
           dpi, size=(7.5*dpi, 3*dpi))




nwbLoader = VideoLoaders.NWBLoader("../data/"*example_files[2])
splitLoader = VideoLoaders.SplitLoader(nwbLoader, 50)
hostCache = VideoLoaders.CachedHostLoader(splitLoader; max_memory=3.2e10)
deviceCache = VideoLoaders.CachedDeviceLoader(hostCache; max_memory=1.6e10)

sol2 = Sol(deviceCache);
sol2.I = negentropy_img(deviceCache);
initA!(sol2; callback=(_,_,_)->nothing);
zeroTraces!(sol2)
negentIm2 = reshape(Array(sol2.I), framesize(deviceCache))
roiIm2 = roiImg(sol2)[1]

import HDF5
HDF5.h5open("manuscript_data_fig2.h5", "w") do fid
    fid["meanIm"] = meanIm
    fid["stdIm"] = stdIm
    fid["maxIm"] = maxIm
    fid["pnrIm"] = pnrImg
    fid["negentIm"] = negentIm
    fid["roiIm"] = roiIm
    fid["negentIm2"] = negentIm2
    fid["roiIm2"] = roiIm2
end