module VideoLoaders

export VideoLoader, readseg, readframe, optimalorder, nframes, nsegs, nvideos, framesize, video_idx, framerange, framerange_video, frame2seg

using OrderedCollections, CUDA, HDF5, DataFrames, CSV, ProgressMeter
import Images

abstract type VideoLoader end
abstract type FileLoader <: VideoLoader end
abstract type SegmentLoader <: VideoLoader end

include("hdfLoader.jl")
include("segmentLoader.jl")
include("cachedLoader.jl")
include("subtractMinLoader.jl")
include("filterLoader.jl")
include("motionCorrectionLoader.jl")

struct EmptyLoader <: VideoLoader end
nframes(::EmptyLoader) = 1
nvideos(::EmptyLoader) = 1
framesize(::EmptyLoader) = (100, 100)
readframe(vl::EmptyLoader, i) = zeros(Int16, framesize(vl))
location(::EmptyLoader) = :nowhere
frame2seg(::EmptyLoader, _) = 1
video_idx(::EmptyLoader, _) = 1

function Base.mapreduce(f::Function, op::Function, vl::VideoLoader, init; dims=2,
                        callback=(_,_)->nothing)
    dims == 2 || error("Only dims=2 is implemented for mapreduce")
    res = CUDA.fill(init, (prod(framesize(vl)), 1))
    callback(0, nsegs(vl))
    @showprogress "mapreduce" for i = optimalorder(vl)
        seg = readseg(vl, i)
        res .= op.(res, CUDA.mapreduce(f, op, seg; init=init, dims=dims))
        callback(i, nsegs(vl))
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
        baseloader = HDFLoader(s, "images")#, (1:1440, 1:1080,  1:1000))
    elseif endswith(s, ".h5")
        baseloader = HDFLoader(s, "data")
    elseif endswith(s, ".csv")
        return openmultivideo(s; nsplits, hostCacheSize, deviceCacheSize)
    else
        #TODO: show error message
        return EmptyLoader()
    end
    splitloader = SplitLoader(baseloader, nsplits)
    hostcache = CachedHostLoader(splitloader; max_memory=hostCacheSize)
    if endswith(s, ".hdf5") || endswith(s, ".h5")
        filterkernel = Images.OffsetArrays.no_offset_view(Images.Kernel.DoG(5.0))
        filterloader = FilterLoader(hostcache, filterkernel)
        #filterloader = BandpassFilterLoader(hostcache, 2, 100)
        mcloader = MotionCorrectionLoader(filterloader, (300:600, 125:300))#(600:1200, 250:600))
        return CachedDeviceLoader(mcloader, max_memory=deviceCacheSize)
    else
        return CachedDeviceLoader(hostcache, max_memory=deviceCacheSize)
    end
end

function openmultivideo(s; nsplits, hostCacheSize, deviceCacheSize)
    pathPrefix = s[1:end-length(split(s, "/")[end])]
    videolist = CSV.File(s,comment="#")#DataFrames.DataFrame(CSV.File(s,comment="#"))
    sources = map(videolist) do r #map(eachrow(videolist))
        hdfLoader = HDFLoader(pathPrefix * r.filename, r.hdfKey)
        SplitLoader(hdfLoader[:, :, 1:100], nsplits)
    end
    segs_per_video = [((i-1)*nsplits+1):i*nsplits for i=1:length(sources)]
    baseloader = MultiVideoLoader(sources, segs_per_video)
    hostcache = CachedHostLoader(baseloader; max_memory=hostCacheSize)

    filterkernel = Images.OffsetArrays.no_offset_view(Images.Kernel.DoG(5.0))
    filterloader = FilterLoader(hostcache, filterkernel)
    #filterloader = BandpassFilterLoader(hostcache, 2, 100)
    mcloader = MotionCorrectionLoader(filterloader, (300:600, 125:300))

    return CachedDeviceLoader(mcloader, max_memory=deviceCacheSize)
end

end
