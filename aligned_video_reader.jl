function assert_all_equal(arr)
    @assert all(arr .== arr[1])
end

function AlignedHDFLoader(alignmentFile, segsPerFile; pathPrefix="",
                          hostMemory=1e10, deviceMemory=1e10, deviceType=Float32)
    alignment = DataFrames.DataFrame(CSV.File(alignmentFile))
    keys = String[]
    n_frames_per_video = Int64[]
    eltypes = Type[]
    for fn in alignment.filename
        HDF5.h5open(pathPrefix * fn, "r") do fid
            available_keys = HDF5.keys(fid["/analysis"])
            @assert length(available_keys)==1
            push!(keys, "/analysis/"*available_keys[1]*"/data")
            push!(n_frames_per_video, size(fid[keys[end]], 3))
            push!(eltypes, eltype(fid[keys[end]]))
        end
    end
    assert_all_equal(alignment.right .- alignment.left)
    assert_all_equal(alignment.bottom .- alignment.top)
    assert_all_equal(eltypes)
    width = alignment.right[1] - alignment.left[1] + 1
    height = alignment.bottom[1] - alignment.top[1] + 1
    nFrames = sum(n_frames_per_video)
    hostType = eltypes[1]
    n_segs = segsPerFile*length(n_frames_per_video)
    localFrameRanges = UnitRange{Int64}[]
    frameRanges = UnitRange{Int64}[]
    cumlFrame = 0
    for N in n_frames_per_video
        for f = 1:segsPerFile
            start = (f-1)*(div(N, segsPerFile)+1)+1
            stop = min(f*(div(N, segsPerFile)+1), N)
            push!(localFrameRanges, start:stop)
            push!(frameRanges, (cumlFrame+start):(cumlFrame+stop))
        end
        cumlFrame += N
    end

    function fileReader(seg_id)::Array{hostType, 3}
        video_id = div(seg_id-1, segsPerFile) + 1
        fileName = alignment.filename[video_id]
        left = alignment.left[video_id]
        right = alignment.right[video_id]
        top = alignment.top[video_id]
        bottom = alignment.bottom[video_id]
        F_offset = hostType(alignment.F_offset[video_id])
        HDF5.h5open(pathPrefix*fileName, "r") do f
            f[keys[video_id]][left:right, top:bottom, localFrameRanges[seg_id]] .+ F_offset
        end
    end
    VideoLoader(nFrames, (width, height), n_segs, frameRanges, fileReader,
                hostMemory, deviceMemory, hostType)
end