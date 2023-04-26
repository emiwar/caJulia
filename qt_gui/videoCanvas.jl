function canvas_func(f::Function)
    function apply_func(buffer::Array{UInt32, 1},
        width32::Int32,
        height32::Int32)
        width::Int = width32
        height::Int = height32
        buffer = reshape(buffer, width, height)
        buffer = reinterpret(ARGB32, buffer)
        f(buffer)
        return
    end
    return CxxWrap.@safe_cfunction($apply_func, Cvoid, 
                        (Array{UInt32,1}, Int32, Int32))
end

function video_canvas(frame::Observable, cmin, cmax, xmin, xmax, ymin, ymax, cmap=:grays)
    colorscheme = ColorSchemes.colorschemes[cmap]
    function update_canvas(buffer)
        x1 = clamp(xmin[], 1, size(frame[], 1)-5)
        x2 = clamp(xmax[], x1+5, size(frame[], 1))
        y1 = clamp(ymin[], 1, size(frame[], 2)-5)
        y2 = clamp(ymax[], y1+5, size(frame[], 2))
        f = Images.imresize(view(frame[], x1:x2, y1:y2), size(buffer))
        range = clamp(cmax[] - cmin[], 1e-10, Inf)
        for ind in eachindex(buffer)
            pixel = f[ind]
            if isfinite(pixel)
                scaled = (pixel - cmin[]) / range
                buffer[ind] = ARGB32(colorscheme[scaled])
            else
                buffer[ind] = ARGB32(0.0, 0.0, 0.0, 1.0)
            end
        end
    end
    return canvas_func(update_canvas)
end

function video_canvas_raw(frame::Observable, xmin, xmax, ymin, ymax)
    @assert eltype(frame) == Matrix{Colors.ARGB32}
    function update_canvas(buffer)
        x1 = clamp(xmin[], 1, size(frame[], 1)-5)
        x2 = clamp(xmax[], x1+5, size(frame[], 1))
        y1 = clamp(ymin[], 1, size(frame[], 2)-5)
        y2 = clamp(ymax[], y1+5, size(frame[], 2))
        buffer .= Images.imresize(view(frame[], x1:x2, y1:y2), size(buffer))
    end
    return canvas_func(update_canvas)
end

function video_canvas_raw(frame::Observable)
    @assert eltype(frame) == Matrix{Colors.ARGB32}
    function update_canvas(buffer)
        buffer .= Images.imresize(frame[], size(buffer))
    end
    return canvas_func(update_canvas)
end