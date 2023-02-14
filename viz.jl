#import Colors
#import Plots

function spacemap(arr; kwargs...)
    arr_h = reshape(Array(arr), frame_size...)
    Plots.heatmap(arr_h; fmt=:png, kwargs...)
end

function roiPlot(rois)
    a = Float64.(getindex.(argmax(Array(rois); dims=1), 1))
    a[(maximum(Array(rois); dims=1)) .== 0] .= NaN
    Plots.heatmap(reshape(a, frame_size...), c=:tab20)
end

function rand_color(i)
    col_vec = rand(3)
    col_vec ./= 0.5*sum(col_vec)
    clamp!(col_vec, 0.0, 1.0)
    return Colors.RGB{Colors.N0f8}(col_vec...)
end

function roiImg(A, frame_size, cols)
    Ah = Array(A)
    combined = zeros(Colors.RGB{Colors.N0f8}, frame_size...)
    peaks = zeros(CartesianIndex{2}, size(A, 1))
    for i=1:size(Ah, 1)
        if sum(Ah[i, :] .^ 2) < 0.99
            continue
        end
        mask = reshape(Ah[i, :], frame_size...)
        max_val, peaks[i] = findmax(mask)
        mask ./= max_val
        combined .= cols[i].*mask .+ (1 .- mask).*combined
    end
    return combined, peaks
end
roiImg(sol::Sol) = roiImg(sol.A, sol.frame_size, sol.colors)

function altRoiPlot(A, frame_size; labels=true, col_fcn::Function=rand_color, kwargs...)
    combined, peaks, cols = roiImg(A, frame_size; col_fcn)
    hm = Plots.heatmap(combined; format=:png, kwargs...)
    if labels
        for i=1:length(peaks)
            Plots.annotate!(peaks[i][2], peaks[i][1], ("$i", cols[i]))
        end
    end
    return hm
end

altRoiPlot(sol::Sol; kwargs...) = altRoiPlot(sol.A, sol.frame_size; kwargs...)

function tracePlot(traces...; fps=20, window=20)
    maxval, argmax = findmax(traces[1])
    p1 = plot()
    p2 = plot()
    for trace=traces
        plot!(p1, (1:length(trace))./fps, trace)
        plot!(p2, (1:length(trace))./fps, trace,
                xlim=(argmax/fps-window/2, argmax/fps+window/2))
    end
    plot(p1, p2, layout=(2,1), legend=false)
end

tracePlot(sol::Sol, j) = tracePlot(Array(sol.R[:, j]), Array(sol.C[:, j]))

function strongestAMap(A)
    Amax, Aargmax = Array.(findmax(CUDA.CuArray(A), dims=1));
    peaks = getindex.(Aargmax, 1)
    peaks[Amax .== 0.0] .= 0
    return peaks
end
function strongestAMap(sol::Sol) 
    if ncells(sol) == 0
        return zeros(Int, sol.frame_size)
    else
        return reshape(strongestAMap(sol.A), sol.frame_size)
    end
end
