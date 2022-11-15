using OrderedCollections
struct VideoLoader{HostType, MCPlanType}
    nFrames::Int64
    frameSize::Tuple{Int64, Int64}
    nSegs::Int64
    frameRanges::Vector{UnitRange{Int64}}
    fileReader::Function
    motionCorrecter::MotionCorrecter{MCPlanType}
    hostArrays::OrderedDict{Int64, Array{HostType, 3}}
    deviceArrays::OrderedDict{Int64, CUDA.CuMatrix{Float32, CUDA.Mem.DeviceBuffer}}
    deviceArraysShifted::Vector{Bool}
    hostMemory::Int64
    deviceMemory::Int64
end

function VideoLoader(nFrames, frameSize, nSegs, frameRanges, fileReader,
                     hostMemory, deviceMemory, HostType)
    hostArrays = OrderedDict{Int64, Array{HostType, 3}}()
    deviceArrays = OrderedDict{Int64, CUDA.CuMatrix{Float32, CUDA.Mem.DeviceBuffer}}()
    motionCorrecter = MotionCorrecter(nFrames, frameSize)
    VideoLoader(nFrames, frameSize, Int64(nSegs), frameRanges, fileReader,
                motionCorrecter, hostArrays,
                deviceArrays, fill(false, nSegs),
                Int64(hostMemory), Int64(deviceMemory))
end

function HDFLoader(fileName; key=nothing, hostMemory=1e10, deviceMemory=1e10, deviceType=Float32)
    fid = HDF5.h5open(fileName, "r")
    if key === nothing
        all_keys = HDF5.keys(fid["/analysis"])
        @assert length(all_keys)==1
        key = "/analysis/"*all_keys[1]*"/data"
    end
    key = string(key)
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
    frameRanges = map(1:n_segs) do seg_id
        start_frame = (seg_id-1)*frames_per_seg + 1
        end_frame = min(seg_id*frames_per_seg, nFrames)
        return start_frame:end_frame
    end
    function fileReader(seg_id)::Array{hostType, 3}
        HDF5.h5open(fileName, "r") do f
            f[key][:, :, frameRanges[seg_id]]
        end
    end
    VideoLoader(nFrames, (w, h), n_segs, frameRanges, fileReader,
                hostMemory, deviceMemory, hostType)
end

function eachSegment(f::Function, vl::VideoLoader; shift::Bool=true)
    processed = zeros(Bool, vl.nSegs)
    @showprogress "Device segs" for seg_id in keys(vl.deviceArrays)
        f(seg_id, loadToDevice!(vl, seg_id, shift))
        processed[seg_id] = true
    end
    @showprogress "Host segs" for seg_id in keys(vl.hostArrays)
        if !processed[seg_id]
            f(seg_id, loadToDevice!(vl, seg_id, shift))
            processed[seg_id] = true
        end
    end
    @showprogress "Disk segs" for seg_id=1:vl.nSegs
        if !processed[seg_id]
            f(seg_id, loadToDevice!(vl, seg_id, shift))
            processed[seg_id] = true
        end
    end
end

hostType(vl::VideoLoader) = eltype(eltype(values(vl.hostArrays)))
deviceType(vl::VideoLoader) = eltype(eltype(values(vl.deviceArrays)))


function loadToDevice!(vl::VideoLoader, seg_id::Int64, shift::Bool=true)
    if seg_id in keys(vl.deviceArrays) &&
       vl.deviceArraysShifted[seg_id] == shift
          return vl.deviceArrays[seg_id]
    end
    hostArray = loadToHost!(vl, seg_id)
    required_memory  = length(hostArray)*(sizeof(hostType(vl))+sizeof(deviceType(vl)))
    reduceToFit!(vl.deviceArrays, vl.deviceMemory - required_memory)
    vl.deviceArrays[seg_id] = deviceType(vl).(CUDA.cu(reshape(hostArray, :, size(hostArray, 3))))
    if shift
        shiftSegment!(vl.motionCorrecter, vl.deviceArrays[seg_id],
                      vl.frameSize, vl.frameRanges[seg_id])
    end
    vl.deviceArraysShifted[seg_id] = shift
    #println("Loaded segment $seg_id into device memory")
    return vl.deviceArrays[seg_id]
end

function loadToHost!(vl::VideoLoader, seg_id::Int64)
    if seg_id in keys(vl.hostArrays)
        return vl.hostArrays[seg_id]
    end
    driveArray = vl.fileReader(seg_id)
    required_memory  = length(driveArray)*(sizeof(hostType(vl)))
    reduceToFit!(vl.hostArrays, vl.hostMemory - required_memory)
    vl.hostArrays[seg_id] = driveArray#hostType(vl).(driveArray)
    #println("Loaded segment $seg_id into host memory")
    return vl.hostArrays[seg_id]
end

function reduceToFit!(arrayDict, max_memory)
    available_memory = max_memory
    for seg_id in reverse(collect(keys(arrayDict)))
        array_size = sizeof(arrayDict[seg_id])
        if array_size < available_memory
            available_memory -= array_size
        else
            delete!(arrayDict, seg_id)
        end
    end
end

function getFrameHost(vl::VideoLoader, frame_id::Int64)
    seg_id = frame_to_seg(vl, frame_id)
    offset = (seg_id-1)*Int64(ceil(vl.nFrames / vl.nSegs))
    seg = loadToHost!(vl, seg_id)
    return seg[:, :, frame_id-offset]
end

n_frames(vl::VideoLoader) = vl.nFrames

function frame_to_seg(vl::VideoLoader, frame_id::Int64)
    #TODO: Could do a binary search here
    for seg_id=1:vl.nSegs
        if frame_id in vl.frameRanges[seg_id]
            return seg_id
        end
    end
    error("Did not find frame $frame_id in the video.")
end

function Base.extrema(vl::VideoLoader)
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