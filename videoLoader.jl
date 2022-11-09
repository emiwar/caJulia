struct VideoLoader
    nFrames::Int64
    frameSize::Tuple{Int64, Int64}
    nSegs::Int64
    fileReader::Function
    frameToSeg::Vector{Int64}
    hostArrays::Dict{Int64, Array{Int16, 3}}
    hostArraysOrder::Vector{Int64}
    deviceArrays::Dict{Int64, CUDA.CuMatrix{Float32, CUDA.Mem.DeviceBuffer}}
    deviceArraysOrder::Vector{Int64}
    hostMemory::Int64
    deviceMemory::Int64
end

function VideoLoader(nFrames, frameSize, nSegs, fileReader, frameToSeg,
                     hostMemory, deviceMemory)
    hostArrays = Dict{Int64, Array{Int16, 3}}()
    hostArraysOrder = Vector{Int64}()
    deviceArrays = Dict{Int64, CUDA.CuMatrix{Float32}}()
    deviceArraysOrder = Vector{Int64}()
    VideoLoader(nFrames, frameSize, Int64(nSegs), fileReader, frameToSeg,
                hostArrays, hostArraysOrder, deviceArrays, deviceArraysOrder,
                Int64(hostMemory), Int64(deviceMemory))
end

function HDFLoader(fileName; hostMemory=1e10, deviceMemory=1e10, deviceType=Float32)
    fid = HDF5.h5open(fileName, "r")
    all_keys = HDF5.keys(fid["/analysis"])
    @assert length(all_keys)==1
    key = "/analysis/"*all_keys[1]*"/data"
    dataset = fid[key]
    w, h, nFrames = size(dataset)
    hostType = eltype(dataset)
    est_host_memory = sizeof(hostType) * length(dataset)
    est_device_memory = (sizeof(deviceType)+sizeof(hostType)) * length(dataset)
    #println("Est. device memory: $est_device_memory")
    #println("Limit device memory: $deviceMemory")
    min_host_segs = Int64(ceil(est_host_memory / hostMemory))
    min_dev_segs = Int64(ceil(est_device_memory / deviceMemory))
    n_segs = max(min_host_segs, min_dev_segs)
    frames_per_seg = Int64(ceil(nFrames / n_segs))
    frame_to_seg = div.((1:nFrames) .- 1, frames_per_seg) .+ 1
    function fileReader(seg_id)
        start_frame = (seg_id-1)*frames_per_seg + 1
        end_frame = min(seg_id*frames_per_seg, nFrames)
        HDF5.h5open(fileName, "r") do f
            f[key][:, :, start_frame:end_frame]
        end
    end
    VideoLoader(nFrames, (w, h), n_segs, fileReader, frame_to_seg, hostMemory, deviceMemory)
end

function eachSegment(f::Function, vl::VideoLoader)
    processed = zeros(Bool, vl.nSegs)
    for seg_id in vl.deviceArraysOrder
        f(seg_id, vl.deviceArrays[seg_id])
        processed[seg_id] = true
    end
    for seg_id in vl.hostArraysOrder
        if !processed[seg_id]
            f(seg_id, loadToDevice!(vl, seg_id))
            processed[seg_id] = true
        end
    end
    for seg_id=1:vl.nSegs
        if !processed[seg_id]
            f(seg_id, loadToDevice!(vl, seg_id))
            processed[seg_id] = true
        end
    end
end

hostType(vl::VideoLoader) = eltype(eltype(values(vl.hostArrays)))
deviceType(vl::VideoLoader) = eltype(eltype(values(vl.deviceArrays)))


function loadToDevice!(vl::VideoLoader, seg_id::Int64)
    if seg_id in keys(vl.deviceArrays)
        return vl.deviceArrays[seg_id]
    end
    hostArray = loadToHost!(vl, seg_id)
    required_memory  = length(hostArray)*(sizeof(hostType(vl))+sizeof(deviceType(vl)))
    reduceToFit!(vl.deviceArrays, vl.deviceArraysOrder, required_memory, vl.deviceMemory)
    vl.deviceArrays[seg_id] = deviceType(vl).(CUDA.cu(reshape(hostArray, :, size(hostArray, 3))))
    insert!(vl.deviceArraysOrder, 1, seg_id)
    #println("Loaded segment $seg_id into device memory")
    return vl.deviceArrays[seg_id]
end

function loadToHost!(vl::VideoLoader, seg_id::Int64)
    if seg_id in keys(vl.hostArrays)
        return vl.hostArrays[seg_id]
    end
    driveArray = vl.fileReader(seg_id)
    required_memory  = length(driveArray)*(sizeof(hostType(vl)))
    reduceToFit!(vl.hostArrays, vl.hostArraysOrder, required_memory, vl.hostMemory)
    vl.hostArrays[seg_id] = Array(driveArray)
    insert!(vl.hostArraysOrder, 1, seg_id)
    #println("Loaded segment $seg_id into host memory")
    return vl.hostArrays[seg_id]
end

function reduceToFit!(arrays, arraysOrder, required_memory, max_memory)
    cuml_memory = 0
    i = 1
    while i <= length(arraysOrder)
        seg_id = arraysOrder[i]
        seg_size = sizeof(arrays[seg_id])
        if (seg_size + cuml_memory + required_memory) <= max_memory
            i += 1
            cuml_memory += seg_size
        else
            popat!(arraysOrder, i)
            delete!(arrays, seg_id)
            #if eltype(values(arrays)) <: CUDA.CuArray
            #    println("Dropped segment $seg_id from device memory")
            #else
            #    println("Dropped segment $seg_id from host memory")
            #end
        end
    end
end

function getFrameHost(vl::VideoLoader, frame_id::Int64)
    seg_id = vl.frameToSeg[frame_id]
    offset = (seg_id-1)*Int64(ceil(vl.nFrames / vl.nSegs))
    seg = loadToHost!(vl, seg_id)
    return seg[:, :, frame_id-offset]
end

n_frames(vl::VideoLoader) = vl.nFrames

function extrema(vl::VideoLoader)
    Y_min = Inf
    Y_max = -Inf
    eachSegment(vl) do _, seg
        Y_min = min(Y_min, minimum(seg))
        Y_max = max(Y_max, maximum(seg))
    end
    return Y_min, Y_max
end

function Base.mapreduce(f::Function, op::Function, vl::VideoLoader, init; dims=2)
    dims == 2 || error("Only dims=2 is implemented for mapreduce")
    res = CUDA.fill(init, (prod(vl.frameSize), 1))
    eachSegment(vl) do _, seg
        res .= op.(res, CUDA.mapreduce(f, op, seg; init=init, dims=dims))
    end
    CUDA.synchronize()
    return res
end

function leftMul(mats::Tuple, vl::VideoLoader)
    T = vl.nFrames 
    M = prod(vl.frameSize)
    for m in mats
        size(m, 2) == M || error("Matrix size doesn't match video")
    end
    res = Tuple(CUDA.zeros(size(m, 1), T) for m in mats)
    frames_per_seg = Int64(ceil(T / vl.nSegs))
    eachSegment(vl) do seg_id, seg
        start_frame = (seg_id-1)*frames_per_seg + 1
        end_frame = min(seg_id*frames_per_seg, T)
        for (i, m) in enumerate(mats)
            res[i][:, start_frame:end_frame] .= m*seg
        end
    end
    return res
end

function rightMul(vl::VideoLoader, mats::Tuple)
    T = vl.nFrames
    M = prod(vl.frameSize)
    for m in mats
        size(m, 1) == T || error("Matrix size doesn't match video")
    end
    res = Tuple(CUDA.zeros(M, size(m, 2)) for m in mats)
    frames_per_seg = Int64(ceil(T / vl.nSegs))
    eachSegment(vl) do seg_id, seg
        start_frame = (seg_id-1)*frames_per_seg + 1
        end_frame = min(seg_id*frames_per_seg, T)
        for (i, m) in enumerate(mats)
            res[i] .+= seg*view(m, start_frame:end_frame, :)
        end
    end
    return res
end