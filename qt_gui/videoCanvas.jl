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

function video_canvas(frame::Observable, cmin, cmax)
    function update_canvas(buffer)
        f = Images.imresize(frame[], size(buffer))
        range = clamp(cmax[] - cmin[], 1e-10, Inf)
        g = clamp.((f .- cmin[]) ./ range, 0.0, 1.0)
        buffer .= ARGB32.(g, g, g, 1.0)
    end
    return canvas_func(update_canvas)
end