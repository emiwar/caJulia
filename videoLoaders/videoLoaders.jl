module VideoLoaders

export VideoLoader, readseg, readframe, optimalorder, nframes, nsegs, nvideos, framesize, video_idx, framerange, framerange_video, frame2seg

using OrderedCollections, CUDA, HDF5, DataFrames, CSV, ProgressMeter

abstract type VideoLoader end
abstract type FileLoader <: VideoLoader end
abstract type SegmentLoader <: VideoLoader end

include("hdfLoader.jl")
include("segmentLoader.jl")
include("cachedLoader.jl")
include("subtractMinLoader.jl")

struct EmptyLoader <: VideoLoader end
nframes(::EmptyLoader) = 1
nvideos(::EmptyLoader) = 1
framesize(::EmptyLoader) = (100, 100)
readframe(vl::EmptyLoader, i) = zeros(Int16, framesize(vl))
location(::EmptyLoader) = :nowhere
frame2seg(::EmptyLoader, _) = 1
video_idx(::EmptyLoader, _) = 1

function Base.mapreduce(f::Function, op::Function, vl::VideoLoader, init; dims=2)
    dims == 2 || error("Only dims=2 is implemented for mapreduce")
    res = CUDA.fill(init, (prod(framesize(vl)), 1))
    @showprogress "mapreduce" for i = optimalorder(vl)
        seg = readseg(vl, i)
        res .= op.(res, CUDA.mapreduce(f, op, seg; init=init, dims=dims))
    end
    CUDA.synchronize()
    return res
end

function openvideo(s::String; nsplits=10, hostCacheSize=3.2e10,
                              deviceCacheSize=1.0e10)
    if !isfile(s)
        #TODO: show error message
        return EmptyLoader()
    end
    if endswith(s, ".nwb")
        baseloader = NWBLoader(s)
    elseif endswith(s, ".hdf5")
        baseloader = HDFLoader(s, "images")
    else
        #TODO: show error message
        return EmptyLoader()
    end
    splitloader = SplitLoader(baseloader, nsplits)
    hostcache = CachedHostLoader(splitloader; max_memory=hostCacheSize)
    devicecache = CachedDeviceLoader(hostcache, max_memory=deviceCacheSize)
end

end
