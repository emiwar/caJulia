struct SplitLoader{T} <: SegmentLoader
    sources::Vector{T}
    frame_ranges::Vector{UnitRange{Int64}}
end

function SplitLoader(source::FileLoader, n_segs::Int64)
    n_frames = nframes(source)
    frames_per_seg = Int64(ceil(n_frames / n_segs))
    frame_ranges = map(1:n_segs) do seg_id
        start_frame = (seg_id-1)*frames_per_seg + 1
        end_frame = min(seg_id*frames_per_seg, n_frames)
        return start_frame:end_frame
    end
    SplitLoader([source[:, :, r] for r=frame_ranges], frame_ranges)
end

function Base.show(io::IO, vl::SegmentLoader)
    println(io, string(typeof(vl)))
    if !get(io, :compact, false)
        println(io, "$(nframes(vl)) frames $(framesize(vl))")
        println(io, "$(nsegs(vl)) segments")
    end
end

Base.eltype(vl::SplitLoader) = eltype(first(vl.sources))
Base.eltype(::Type{SplitLoader{T}}) where T = eltype(T)
nsegs(vl::SplitLoader) = length(vl.frame_ranges)
nframes(vl::SplitLoader) = sum(length.(vl.frame_ranges))
framesize(vl::SplitLoader) = framesize(first(vl.sources))
optimalorder(vl::SplitLoader) = 1:nsegs(vl)
framerange(vl::SplitLoader, i::Int64) = vl.frame_ranges[i]
readseg(vl::SplitLoader, i::Int64) = readseg(vl.sources[i])
filename(vl::SplitLoader, i::Int64) = filename(vl.sources[i])
nvideos(vl::SplitLoader) = length(unique(filename.(vl.sources)))
video_idx(vl::SplitLoader, i::Int64) = video_idx(vl.sources[i])

function framerange_video(vl::SplitLoader, video_i)
    frs = [vl.frame_ranges[i] for i=1:nsegs(vl) if video_idx(vl, i) == video_i]
    minimum(first.(frs)):maximum(last.(frs))
end

function frame2seg(vl::SegmentLoader, frame_idx)
    for i=1:nsegs(vl)
        if frame_idx in framerange(vl, i)
            return i
        end
    end
    error("Frame $frame_idx not in video.")
end

function readframe(vl::SegmentLoader, frame_idx)
    seg_id = frame2seg(vl, frame_idx)
    seg = readseg(vl, seg_id)
    local_frame = frame_idx - first(framerange(vl, seg_id)) + 1
    if ndims(seg) == 3
        return seg[:, :, local_frame]
    elseif ndims(seg) == 2
        return seg[:, local_frame]
    end
end