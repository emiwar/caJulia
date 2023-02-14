struct SubtractMinLoader{T <: VideoLoader} <: SegmentLoader
    source_loader::T
    min::CUDA.CuVector{Float32, CUDA.Mem.DeviceBuffer}
end

function SubtractMinLoader(source_loader)
    SubtractMinLoader(source_loader, CUDA.zeros(prod(framesize(source_loader))))
end

function readseg(vl::SubtractMinLoader, i)
    seg = readseg(vl.source_loader, i)
    if ndims(seg) == 3
        seg = reshape(seg, :, size(seg, 3))
    end
    moved = CUDA.cu(seg)
    return eltype(vl).(moved) .- vl.min
end

function calcmin!(vl::SubtractMinLoader; callback=(_,_,_)->nothing)
    cb(i, N) = callback("Subtracting min", i, N)
    vl.min .= 0.0f0
    vl.min .= view(mapreduce(x->x, min, vl, Inf32; callback=cb), :)
    nothing
end

location(::SubtractMinLoader) = :device
Base.eltype(vl::SubtractMinLoader) = Float32
nsegs(vl::SubtractMinLoader) = nsegs(vl.source_loader)
nframes(vl::SubtractMinLoader) = nframes(vl.source_loader)
framesize(vl::SubtractMinLoader) = framesize(vl.source_loader)
framerange(vl::SubtractMinLoader, i) = framerange(vl.source_loader, i) 
framerange_video(vl::SubtractMinLoader, i) = framerange_video(vl.source_loader, i) 
nvideos(vl::SubtractMinLoader) = nvideos(vl.source_loader)
video_idx(vl::SubtractMinLoader, i) = video_idx(vl.source_loader, i)
optimalorder(vl::SubtractMinLoader) = optimalorder(vl.source_loader)

function readframe(vl::SubtractMinLoader, frame_idx)
    seg_id = frame2seg(vl.source_loader, frame_idx)
    seg = readseg(vl.source_loader, seg_id)
    local_frame = frame_idx - first(framerange(vl, seg_id)) + 1
    if ndims(seg) == 3
        return seg[:, :, local_frame] .- reshape(eltype(seg).(Array(vl.min)), framesize(vl))
    elseif ndims(seg) == 2
        return seg[:, local_frame] .- eltype(seg).(Array(vl.min))
    end
end