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

function video_canvas(frame::Observable, cmin, cmax, cmap=:grays)
    colorscheme = ColorSchemes.colorschemes[cmap]
    function update_canvas(buffer)
        f = Images.imresize(frame[], size(buffer))
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

function video_canvas_raw(frame::Observable)
    @assert eltype(frame) == Matrix{Colors.ARGB32}
    function update_canvas(buffer)
        buffer .= Images.imresize(frame[], size(buffer))
    end
    return canvas_func(update_canvas)
end