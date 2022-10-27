import Statistics
import CUDA
import Images
import SparseArrays
import HDF5
CUDA.allowscalar(false)
include("initROIs.jl")
include("fastHALS.jl")
include("merge_split.jl")

video = HDF5.h5open("../data/20211016_163921_animal1learnday1.nwb", "r") do fid
    HDF5.read(fid["analysis/recording_20211016_163921-PP-BP-MC/data"])
end
const frame_size = size(video)[1:2]
Y = Float32.(CUDA.cu(reshape(video, :, size(video, 3))));
#Y_mean = reshape(sum(Y; dims=2) / size(Y, 2), size(Y, 1))
#Y .-= Y_mean;
A = initROIs(Y, frame_size);
C = CUDA.zeros(Float32, (size(video, 3), size(A, 1)));
b0 = sum(Y, dims=2) ./ Float32(sqrt(size(Y, 2)));
b1 = view(Float32.(maximum(CUDA.CuArray(A); dims=1) .== 0), :);
b1 ./= CUDA.norm(b1)
f1 = CUDA.zeros(Float32, size(Y, 2));

for i=1:1
    updateTraces!(Y, A, C, b0, b1, f1);
    A = updateROIs!(Y, A, C, b0, b1, f1, frame_size);
    #A, C = merge(A, C; thres=.6)
    #print(i, size(C))
    #display(altRoiPlot(A; labels=false, axis=false, xticks=[], yticks=[]))
end

