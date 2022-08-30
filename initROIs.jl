function initROIs(video, frame_size)
    N = size(video, 2)
    m = sum(video, dims=2) ./ N
    m2 = mapreduce(x->Float64(x)^2, +, video; dims=2) ./ N
    sd = sqrt.(m2 .- m.^2)
    max_proj = maximum(video; dims=2)
    projection = reshape(Array((max_proj .- m) ./ sd), frame_size...)
    filtered = Images.mapwindow(Statistics.median, projection[:,:,1], (5, 5))
    rois_list = segment_peaks(filtered)
    rois = SparseArrays.sparse(hcat(rois_list...)')
    return CUDA.cu(rois)
end

function segment_peaks(projection; threshold=5)
    segmentation = zeros(Int64, size(projection))
    sorted_pixels = CartesianIndices(projection)[sortperm(projection[:]; rev=true)]
    lin_inds = LinearIndices(segmentation)
    seg_id = 0
    q = CartesianIndex{2}[]
    rois_list = SparseArrays.SparseVector{Float32, Int32}[]
    for start_coord in sorted_pixels
        if segmentation[start_coord] == 0 &&
           projection[start_coord] > threshold
            seg_id += 1
            segmentation[start_coord] = seg_id
            push!(q, start_coord)
            push!(rois_list, SparseArrays.spzeros(Float32, length(segmentation)))
            rois_list[seg_id][lin_inds[start_coord]] = projection[start_coord]
            while !isempty(q)
                coord = popfirst!(q)
                for offs in [(0,1), (0,-1), (1,0), (-1,0)]
                    offs_coord = coord + CartesianIndex(offs...)
                    if checkbounds(Bool, projection, offs_coord) &&
                       segmentation[offs_coord] == 0 &&
                       projection[coord] > projection[offs_coord]
                            segmentation[offs_coord] = seg_id
                            push!(q, offs_coord)
                            linear_index = lin_inds[offs_coord]
                            rois_list[seg_id][linear_index] = projection[offs_coord]
                    end
                end
            end
            rois_list[seg_id] ./= sqrt(sum(rois_list[seg_id].^2))
        end
    end
    return rois_list
end