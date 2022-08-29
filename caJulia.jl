import Statistics
import CUDA
import Images
import SparseArrays
CUDA.allowscalar(false)

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


function updateTraces!(rois, traces, back_kernel, back_trace, B, video; n_iter=5)
    #FastHALS
    rois_dense = CUDA.CuArray(rois)
    W = (rois*video)
    V = rois*rois_dense'
    #WB = video'*back_kernel
    #VB = rois_dense*back_kernel
    #VB_host = Array(VB)
    B_proj = Array(rois*B)
    #BB_proj = sum(back_kernel.*B)
    for it=1:n_iter
        for j=1:size(traces, 2)
            traces[:, j] .= max.(traces[:, j] .+ W[j, :] .- traces*V[j, :] .-
                                 B_proj[j], 0) #.- back_trace*VB_host[j]
        end
        #back_trace .= max.(WB .- traces*VB .- BB_proj, 0)
    end
end

function updateROIs!(rois, traces, back_kernel, back_trace, B, video; n_iter=5)
    #FastHALS
    rois_dense = CUDA.CuArray(rois)
    P = video*traces .- B*sum(traces, dims=1)
    Q = traces'*traces
    Q_host = Array(Q)
    #PB = video*back_trace .- B[:,1] .* sum(back_trace)
    #QB = traces'*back_trace
    #QB_host = Array(QB)
    for it=1:n_iter
        for j=1:size(rois, 1)
            rois_dense[j, :] .= max.(rois_dense[j, :].*Q_host[j,j] .+ P[:, j] .-
                                     (Q[j:j, :]*rois_dense)[:], 0)#- back_kernel*QB_host[j]
            rois_dense[j, :] ./= sqrt(sum(rois_dense[j, :].^2))
        end
        #back_kernel .= PB .- rois_dense'*QB
        #back_kernel ./= (sqrt(sum(back_kernel .^ 2)))
    end
    return CUDA.cu(SparseArrays.sparse(Array(rois_dense)))
end

using HDF5
video = h5open("../data/20211016_163921_animal1learnday1.nwb", "r") do fid
    read(fid["analysis/recording_20211016_163921-PP-BP-MC/data"])
end
frame_size = size(video)[1:2]
flatV = Float32.(CUDA.cu(reshape(video, :, size(video, 3))));
rois = initROIs(flatV, frame_size);
traces = CUDA.zeros(Float32, (size(video, 3), size(rois, 1)));
back_kernel = CUDA.ones(Float32, size(flatV, 1)) ./ Float32(sqrt(size(flatV, 1)))
back_trace = CUDA.zeros(Float32, size(flatV, 2))
B = sum(flatV, dims=2) ./ size(flatV, 2)#minimum(flatV, dims=2)

updateTraces!(rois, traces, back_kernel, back_trace, B, flatV; n_iter=5);
rois = updateROIs!(rois, traces, back_kernel, back_trace, B, flatV; n_iter=5)


function roiPlot(rois)
    a = Float64.(getindex.(argmax(Array(rois); dims=1), 1))
    a[(maximum(Array(rois); dims=1)) .== 0] .= NaN
    Plots.heatmap(reshape(a, frame_size...), c=:tab20)
end
