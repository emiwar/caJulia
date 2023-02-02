abstract type CachedLoader <: SegmentLoader end

struct CachedHostLoader{L <: SegmentLoader, T} <: CachedLoader
    source_loader::L
    max_memory::Int64
    cache::OrderedDict{Int64, Array{T, 3}}
end

struct CachedDeviceLoader{T <: SegmentLoader} <: CachedLoader
    source_loader::T
    max_memory::Int64
    cache::OrderedDict{Int64, CUDA.CuMatrix{Float32, CUDA.Mem.DeviceBuffer}}
end

function CachedHostLoader(source::SegmentLoader; max_memory=4e9)
    CachedHostLoader(source, Int64(max_memory),
                     OrderedDict{Int64, Array{eltype(source), 3}}())
end
function CachedDeviceLoader(source::SegmentLoader; max_memory=4e9)
    CachedDeviceLoader(source, Int64(max_memory),
                       OrderedDict{Int64, CUDA.CuMatrix{Float32, CUDA.Mem.DeviceBuffer}}())
end

function Base.show(io::IO, vl::CachedLoader)
    Base.show(io, vl.source_loader)
    println(io, "Cached segs ($(location(vl))): $(keys(vl.cache))")
    percent_memory = round(1000 * used_memory(vl) / vl.max_memory) / 10
    println(io, "Used memory: $(used_memory(vl)) of $(vl.max_memory) ($(percent_memory)%)")
end

location(::CachedHostLoader) = :host
location(::CachedDeviceLoader) = :device
Base.eltype(vl::CachedLoader) = Base.eltype(Base.eltype(values(vl.cache)))
nsegs(vl::CachedLoader) = nsegs(vl.source_loader)
nframes(vl::CachedLoader) = nframes(vl.source_loader)
framesize(vl::CachedLoader) = framesize(vl.source_loader)
framerange(vl::CachedLoader, i) = framerange(vl.source_loader, i) 
framerange_video(vl::CachedLoader, i) = framerange_video(vl.source_loader, i)
multivideo(vl::CachedLoader) = multivideo(vl.source_loader)
nvideos(vl::CachedLoader) = nvideos(vl.source_loader)
video_idx(vl::CachedLoader, i) = video_idx(vl.source_loader, i) 

used_memory(vl::CachedLoader) = sum(sizeof.(values(vl.cache)))

convertseg(vl::CachedLoader, seg) = seg
function convertseg(vl::CachedDeviceLoader, seg::AbstractArray)
    if ndims(seg) == 3
        return convertseg(vl, reshape(seg, :, size(seg, 3)))
    end
    moved = CUDA.cu(seg)
    return eltype(vl).(moved)
end
convertseg(vl::CachedDeviceLoader, seg::CUDA.CuMatrix{Float32}) = seg

function readseg(vl::CachedLoader, i::Int64)
    if i in keys(vl.cache)
        return vl.cache[i]
    end
    newseg = readseg(vl.source_loader, i)
    required_memory = length(newseg)*(sizeof(eltype(vl)))
    if eltype(vl) != eltype(vl.source_loader)
        required_memory += length(newseg)*(sizeof(eltype(vl.source_loader)))
    end
    if required_memory > vl.max_memory
        error("Segment $i requires $required_memory bytes but cache only has $(vl.max_memory).")
    end
    available_memory = vl.max_memory - used_memory(vl)
    while required_memory > available_memory
        seg_id = first(keys(vl.cache))
        available_memory += sizeof(vl.cache[seg_id])
        clearseg!(vl, seg_id)
        GC.gc()
    end
    return vl.cache[i] = convertseg(vl, newseg)
end

function clearseg!(vl::CachedLoader, i::Int64)
    delete!(vl.cache, i)
end

function clear!(vl::CachedLoader)
    empty!(vl.cache)
end

function optimalorder(vl::CachedLoader)
    union(OrderedSet(keys(vl.cache)), optimalorder(vl.source_loader))
end

function filename(vl::CachedLoader, i::Integer)
    filename(vl.source_loader, i)
end