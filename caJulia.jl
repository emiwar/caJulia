import Statistics
import CUDA
import Images
import SparseArrays
CUDA.allowscalar(false)
include("initROIs.jl")
include("fastHALS.jl")

using HDF5
video = h5open("../data/20211016_163921_animal1learnday1.nwb", "r") do fid
    read(fid["analysis/recording_20211016_163921-PP-BP-MC/data"])
end
frame_size = size(video)[1:2]
Y = Float32.(CUDA.cu(reshape(video, :, size(video, 3))));
A = initROIs(Y, frame_size);
C = CUDA.zeros(Float32, (size(video, 3), size(A, 1)));
b0 = sum(Y, dims=2) ./ Float32(sqrt(size(Y, 2)));
b1 = CUDA.ones(Float32, size(Y, 1)) ./ Float32(sqrt(size(Y, 1)));
f1 = CUDA.zeros(Float32, size(Y, 2));

updateTraces!(Y, A, C, b0);
A = updateROIs!(Y, A, C, b0);


function roiPlot(rois)
    a = Float64.(getindex.(argmax(Array(rois); dims=1), 1))
    a[(maximum(Array(rois); dims=1)) .== 0] .= NaN
    Plots.heatmap(reshape(a, frame_size...), c=:tab20)
end
