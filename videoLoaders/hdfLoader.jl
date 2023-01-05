struct HDFLoader{T} <: FileLoader
    filename::String
    hdfkey::String
    subset::NTuple{3, UnitRange{Int64}}
end

function Base.show(io::IO, vl::FileLoader)
    println(io, "VideoLoader")
    if !get(io, :compact, false)
        println(io, "$(filename(vl))")
        println(io, "$(nframes(vl)) frames $(framesize(vl))")
    end
end

Base.eltype(::HDFLoader{T}) where T = T
Base.eltype(::Type{HDFLoader{T}}) where T = T
nframes(vl::HDFLoader) = length(vl.subset[3])
framesize(vl::HDFLoader) = (length(vl.subset[1]), length(vl.subset[2]))
filename(vl::HDFLoader) = vl.filename
nvideos(vl::HDFLoader) = 1
video_idx(vl::HDFLoader) = 1
multivideo(vl::HDFLoader) = false

function NWBLoader(filename, subset=nothing)
    fid = HDF5.h5open(filename, "r")
    all_keys = HDF5.keys(fid["/analysis"])
    if length(all_keys) != 1
        error("$filename does not contain exactly 1 key under '/analysis'.")
    end
    key = "/analysis/"*first(all_keys)*"/data"
    HDF5.close(fid)
    HDFLoader(filename, key, subset)
end

function HDFLoader(filename, key, subset=nothing)
    fid = HDF5.h5open(filename, "r")
    dataset = fid[key]    
    if subset === nothing
        subset = Tuple(1:d for d in size(dataset))
    end
    HDFLoader{Base.eltype(dataset)}(filename, key, subset)
end

function readseg(vl::HDFLoader)
    HDF5.h5open(vl.filename, "r") do f
        f[vl.hdfkey][vl.subset...]
    end
end

function Base.getindex(vl::HDFLoader{T}, a, b, c) where T
    s = vl.subset
    HDFLoader(vl.filename, vl.hdfkey, (s[1][a], s[2][b], s[3][c]))
end

struct MultiVideoLoader{T} <: SegmentLoader
    sources::Vector{T}
    segs_per_video::Vector{UnitRange{Int64}}
end
Base.eltype(::MultiVideoLoader{T}) where T = eltype(T)
Base.eltype(::Type{MultiVideoLoader{T}}) where T = eltype(T)
framesize(vl::MultiVideoLoader) = framesize(first(vl.sources))
nframes(vl::MultiVideoLoader) = sum(nframes.(vl.sources))
nsegs(vl::MultiVideoLoader) = sum(length.(vl.segs_per_video))
multivideo(vl::MultiVideoLoader) = true
nvideos(vl::MultiVideoLoader) = length(vl.segs_per_video)
optimalorder(vl::MultiVideoLoader) = union((vl.segs_per_video)...)
function framerange_video(vl::MultiVideoLoader, video_id)
    sum(nframes.(view(vl.sources, 1:(video_id-1)))) .+ 
        (1:nframes(vl.sources[video_id]))
end
function framerange(vl::MultiVideoLoader, i::Integer)
    video_i = video_idx(vl, i)
    local_idx = i - first(vl.segs_per_video[video_i]) + 1
    sum(nframes.(view(vl.sources, 1:(video_i-1)))) .+
        framerange(vl.sources[video_i], local_idx)
end

function AlignedHDFLoader(alignmentFile, segsPerFile; pathPrefix="")
    alignment = DataFrames.DataFrame(CSV.File(alignmentFile))
    sources = map(eachrow(alignment)) do r
        hdfLoader = NWBLoader(pathPrefix * r.filename)
        SplitLoader(hdfLoader[r.left:r.right, r.top:r.bottom, :], segsPerFile)
    end
    @assert all([framesize(sources[1])] .== framesize.(sources))
    @assert all(eltype(sources[1]) .== eltype.(sources))
    segs_per_video = [((i-1)*segsPerFile+1):i*segsPerFile for i=1:size(alignment, 1)]
    MultiVideoLoader(sources, segs_per_video)
end

video_idx(vl::MultiVideoLoader, seg_idx) = findfirst(s->(seg_idx in s),
                                                     vl.segs_per_video) 
function readseg(vl::MultiVideoLoader, i::Integer)
    video_i = video_idx(vl, i)
    local_idx = i - first(vl.segs_per_video[video_i]) + 1
    readseg(vl.sources[video_i], local_idx)
end

function filename(vl::MultiVideoLoader, i::Integer)
    video_i = video_idx(vl, i)
    local_idx = i - first(vl.segs_per_video[video_i]) + 1
    filename(vl.sources[video_i], local_idx)
end

