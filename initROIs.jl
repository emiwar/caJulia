include("negentropy_img.jl")
function initA!(Y, sol::Sol; threshold=5.0, median_wnd=5)
    N = size(Y, 2)
    m = sum(Y, dims=2) ./ N
    m2 = mapreduce(x->Float64(x)^2, +, Y; dims=2) ./ N
    sd = sqrt.(m2 .- m.^2)
    max_proj = maximum(Y; dims=2)
    projection = reshape(Array((max_proj .- m) ./ sd), sol.frame_size...) 
    rois_list = segment_peaks(projection, threshold, median_wnd)
    rois = SparseArrays.sparse(hcat(rois_list...)')
    sol.A = CUDA.cu(rois)
end

function initA_negentropy!(Y, sol::Sol; threshold=5e-3, median_wnd=5)
    projection = reshape(Array(negentropy_img(Y)), sol.frame_size...)
    rois_list = segment_peaks_alt(projection, 0.5)
    rois = SparseArrays.sparse(hcat(rois_list...)')
    sol.A = CUDA.cu(rois)
end

function segment_peaks(projection, threshold, median_wnd)
    filtered = Images.mapwindow(Statistics.median, projection[:,:,1], (median_wnd, median_wnd))
    segmentation = zeros(Int64, size(projection))
    sorted_pixels = CartesianIndices(filtered)[sortperm(filtered[:]; rev=true)]
    lin_inds = LinearIndices(segmentation)
    seg_id = 0
    q = CartesianIndex{2}[]
    rois_list = SparseArrays.SparseVector{Float32, Int32}[]
    for start_coord in sorted_pixels
        if segmentation[start_coord] == 0 &&
            filtered[start_coord] > threshold
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
                       filtered[coord] > filtered[offs_coord]
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


function segment_peaks_alt(projection, tol=0.5)
    segmentation = zeros(Int64, size(projection))
    sorted_pixels = CartesianIndices(filtered)[sortperm(filtered[:]; rev=true)]
    lin_inds = LinearIndices(segmentation)
    seg_id = 0
    q = CartesianIndex{2}[]
    rois_list = SparseArrays.SparseVector{Float32, Int32}[]
    for start_coord in sorted_pixels
        if segmentation[start_coord] == 0
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
                       filtered[coord] > tol*filtered[start_coord]
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

function initROIs_alt(video, frame_size; min_size=50)
    M, N = size(video)
    mean_frame = sum(video, dims=2) ./ N
    mean_sqrd_frame = mapreduce(x->Float64(x)^2, +, video; dims=2) ./ N
    sd_frame = sqrt.(mean_sqrd_frame .- mean_frame.^2)
    max_frame, argmax_frame = findmax(video; dims=2)
    projection_frame = dropdims(Array((max_frame .- mean_frame) ./ sd_frame); dims=2)
    frame_nos = dropdims(Array(getindex.(argmax_frame, 2)); dims=2)
    
    sorted_pixels = sortperm(projection_frame; rev=true)
    cart_inds = CartesianIndices(frame_size)
    lin_inds = LinearIndices(frame_size)
    q = eltype(sorted_pixels)[]
    blocked = zeros(Bool, M)
    rois_list = SparseArrays.SparseVector{Float32, Int32}[]
    seg_id = 0
    @showprogress for start_pixel in sorted_pixels
        if !blocked[start_pixel]
            seg_id += 1
            blocked[start_pixel] = true
            push!(q, start_pixel)
            local_frame = Array((view(Y, :, frame_nos[start_pixel]) .- mean_frame) ./ sd_frame)
            #local_frame = Images.mapwindow(Statistics.median, reshape(local_frame, frame_size...), (5, 5))[:]
            threshold = -minimum(local_frame)
            roi = SparseArrays.spzeros(Float32, M)
            roi[start_pixel] = local_frame[start_pixel]
            while !isempty(q)
                pixel = popfirst!(q)
                for offs in [(0,1), (0,-1), (1,0), (-1,0)]
                    offs_coord = cart_inds[pixel] + CartesianIndex(offs...)
                    if checkbounds(Bool, lin_inds, offs_coord)
                        neighbour_pixel = lin_inds[offs_coord]
                        local_value = local_frame[neighbour_pixel]
                        if local_value > threshold && !blocked[neighbour_pixel]
                            #roi[neighbour_pixel] == 0.0
                            #Maybe checked blocked here as well - that would imply no ROI overlap
                            blocked[neighbour_pixel] = true
                            push!(q, neighbour_pixel)
                            roi[neighbour_pixel] = local_value
                        end
                    end
                end
            end
            if SparseArrays.nnz(roi) > min_size
                #roi ./= sqrt(sum(roi.^2))
                push!(rois_list, roi)
            end
        end
    end
    println("$seg_id, $(length(rois_list)), $(length(sorted_pixels))")
    rois_list
end