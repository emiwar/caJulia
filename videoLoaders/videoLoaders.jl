module VideoLoaders

export VideoLoader, readseg, readframe, optimalorder, nframes, nsegs, nvideos, framesize, video_idx, framerange, framerange_video, frame2seg

using OrderedCollections, CUDA, HDF5, DataFrames, CSV

abstract type VideoLoader end
abstract type FileLoader <: VideoLoader end
abstract type SegmentLoader <: VideoLoader end

include("hdfLoader.jl")
include("segmentLoader.jl")
include("cachedLoader.jl")


function Base.mapreduce(f::Function, op::Function, vl::VideoLoader, init; dims=2)
    dims == 2 || error("Only dims=2 is implemented for mapreduce")
    res = CUDA.fill(init, (prod(framesize(vl)), 1))
    for i = optimalorder(vl)
        seg = readseg(vl, i)
        res .= op.(res, CUDA.mapreduce(f, op, seg; init=init, dims=dims))
    end
    CUDA.synchronize()
    return res
end

end
