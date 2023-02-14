import Images

function initA!(sol::Sol; threshold=2e-2, median_wnd=1, callback)
    projection = reshape(Array(sol.I), sol.frame_size...) 
    if median_wnd > 1
        projection = Images.mapwindow(Statistics.median, projection, (median_wnd, median_wnd))
    end
    callback("Detecting peaks", 0, 1)
    rois_list = segment_peaks_unionfind(projection; callback)#segment_peaks(projection, threshold)
    callback("Moving footprints to device", 0, 1)
    rois = SparseArrays.sparse(hcat(rois_list...)')
    callback("Moving footprints to device", 0.5, 1)
    sol.A = CUDA.cu(rois)
end

function initA_PNR!(Y, sol::Sol; threshold=5.0, median_wnd=5)
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

function segment_peaks(I, threshold)#, median_wnd)
    #filtered = Images.mapwindow(Statistics.median, projection[:,:,1], (median_wnd, median_wnd))
    segmentation = zeros(Int64, size(I))
    sorted_pixels = CartesianIndices(I)[sortperm(I[:]; rev=true)]
    lin_inds = LinearIndices(segmentation)
    seg_id = 0
    q = CartesianIndex{2}[]
    rois_list = SparseArrays.SparseVector{Float32, Int32}[]
    for start_coord in sorted_pixels
        if segmentation[start_coord] == 0 &&
            I[start_coord] > threshold
            seg_id += 1
            segmentation[start_coord] = seg_id
            push!(q, start_coord)
            push!(rois_list, SparseArrays.spzeros(Float32, length(segmentation)))
            rois_list[seg_id][lin_inds[start_coord]] = I[start_coord]
            while !isempty(q)
                coord = popfirst!(q)
                for offs in [(0,1), (0,-1), (1,0), (-1,0)]
                    offs_coord = coord + CartesianIndex(offs...)
                    if checkbounds(Bool, I, offs_coord) &&
                       segmentation[offs_coord] == 0 &&
                       I[coord] > I[offs_coord] &&
                       I[offs_coord] > 0
                            segmentation[offs_coord] = seg_id
                            push!(q, offs_coord)
                            linear_index = lin_inds[offs_coord]
                            rois_list[seg_id][linear_index] = I[offs_coord]
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



function getparent(parents, i)::Int
    if i == 0
        return 0
    elseif parents[i] == i
        return i
    else
        return getparent(parents, parents[i])
    end
end

function segment_peaks_unionfind(I, min_size=20, sens=50.0, min_n=1000; callback=(_,_,_)->nothing)
    segmentation = zeros(Int64, size(I))
    sorted_pixels = CartesianIndices(I)[sortperm(I[:]; rev=true)]
    parents = Int64[]
    sizes = Int64[]

    #Step 1: cluster all pixels
    for (i, coord) in enumerate(sorted_pixels)
        neighbours = Int64[]
        for offs in [(0,1), (0,-1), (1,0), (-1,0)]
            offs_coord = coord + CartesianIndex(offs...)
            if checkbounds(Bool, I, offs_coord) && segmentation[offs_coord] != 0
                push!(neighbours, segmentation[offs_coord])
            end
        end
        if length(neighbours) == 0
            push!(parents, length(parents)+1)
            push!(sizes, 1)
            segmentation[coord] = length(parents)
        elseif length(neighbours) == 1
            p = getparent(parents, neighbours[1])
            segmentation[coord] = p
            sizes[p] += 1
        else
            p_sizes = [sizes[getparent(parents, n)] for n in neighbours]
            largest_i = findmax(p_sizes)[2]
            largest_p = getparent(parents, neighbours[largest_i])
            segmentation[coord] = largest_p
            sizes[largest_p] += 1
            for n in neighbours
                p = getparent(parents, n)
                if sizes[p] < min_size && p != largest_p
                    parents[p] = largest_p
                    sizes[largest_p] += sizes[p]
                end
            end
        end
        if i%500==0
            callback("Processing pixels", i, length(sorted_pixels))
        end
    end
    callback("Assigning blob IDs", 0, 1)
    seg = map(segmentation) do i
        p = getparent(parents, i)
        (p==0 || sizes[p] < min_size) ? 0 : p
    end

    #Step 2: Go through all candidate clusters and filter out the ones that
    #are unlikely to be noise
    rois_list = SparseArrays.SparseVector{Float32, Int32}[]
    c_s = 0.0
    c_ss = 0.0
    c_n = 0
    callback("Finding unique blobs", 0, 1)
    segs = unique(seg)
    callback("Sorting by size", 0, 1)
    #TODO: this can be _much_ more efficient
    segs = sort(segs, by=i->Statistics.mean(I[seg .== i] .^ 2))
    for (i, seg_id) = enumerate(segs)
        patch = I[seg .== seg_id]
        n = length(patch)
        keep = false
        if c_n > min_n
            c_mean = c_s / c_n
            c_std  = sqrt(c_ss / c_n - (c_mean .^ 2))
            xi = sum(((patch .- c_mean) ./ c_std) .^ 2)
            #TODO: change this so that instead of sens, we have false positive rate,
            #and use multiple comparisons to determine the threshold
            keep = xi > Distributions.invlogccdf(Distributions.Chisq(n), -sens)
        end
        if keep
            new_roi = SparseArrays.sparse(reshape(clamp.(0, I .* (seg .== seg_id), Inf), :))
            new_roi ./= LinearAlgebra.norm(new_roi)
            push!(rois_list, new_roi)
        else
            c_s += sum(patch)
            c_ss += sum(patch .^ 2)
            c_n += n
            #seg[seg .== seg_id] .= 0
        end
        callback("Filtering detected blobs", i, length(segs))
    end
    return rois_list
end