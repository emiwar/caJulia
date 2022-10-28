import Statistics
import CUDA
import Images
import SparseArrays
import HDF5
using ProgressMeter
CUDA.allowscalar(false)
include("solution_struct.jl")
include("initROIs.jl")
include("fastHALS.jl")
include("oasis_opt.jl")
include("merge_split.jl")

video = HDF5.h5open("../data/20211016_163921_animal1learnday1.nwb", "r") do fid
    HDF5.read(fid["analysis/recording_20211016_163921-PP-BP-MC/data"])
end
const frame_size = size(video)[1:2]
Y = Float32.(CUDA.cu(reshape(video, :, size(video, 3))));
sol = Sol(Y, frame_size);
initA!(Y, sol);
zeroTraces!(sol);
initBackground!(Y, sol);
altRoiPlot(sol.A, sol.frame_size, labels=false) |> display

updateTraces!(Y, sol; deconvFcn! = oasis_opt!);
updateROIs!(Y, sol);
while sum(merge!(sol; thres=0.6)) > 0
    println("Number of neurons ", size(sol.A, 1))
end
altRoiPlot(sol.A, sol.frame_size, labels=false) |> display