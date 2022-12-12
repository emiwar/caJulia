mutable struct PerVideoRank1Background <: Background
    raw_f::CUDA.CuVector{Float32}
    f::CUDA.CuVector{Float32}
    raw_b::CUDA.CuMatrix{Float32}
    b::CUDA.CuMatrix{Float32}
    bAf::CUDA.CuMatrix{Float32}
    bfC::CUDA.CuMatrix{Float32}
end

function PerVideoRank1Background(vl::VideoLoader)
    npixels = prod(framesize(vl))
    nvid = nvideos(vl)
    PerVideoRank1Background(CUDA.zeros(nframes(vl)),
                            CUDA.zeros(nframes(vl)),
                            CUDA.zeros(npixels, nvid),
                            CUDA.zeros(npixels, nvid),
                            CUDA.zeros(0, 0),
                            CUDA.zeros(0, 0))
end

function updatepixelcorrections!(bg::PerVideoRank1Background, sol, vl)
    bg.bfC = CUDA.zeros(prod(framesize(vl)), ncells(sol))
    for video_id = 1:nvideos(vl)
        fr = framerange_video(vl, video_id)
        bg.bfC .+= view(bg.b, :, video_id) * (view(sol.C, fr, :)' * view(bg.f, fr))'
    end
end

function prepupdate!(bg::PerVideoRank1Background, sol, vl)
    bg.raw_b .= 0.0f0
end

function updateseg!(bg::PerVideoRank1Background, Y_seg, i, vl)
    video_id = video_idx(vl, i)
    fr = framerange(vl, i)
    bg.raw_b[:, video_id] .+= Y_seg*view(bg.f, fr)
end

function update!(bg::PerVideoRank1Background, sol, vl)
    bg.b .= bg.raw_b 
    for video_id = 1:nvideos(vl)
        fr = framerange_video(vl, video_id)
        bg.b[:, video_id] .-= sol.A'*(view(sol.C, fr, :)'*view(bg.f, fr))
    end
    for other_bg in sol.backgrounds
        if other_bg !== bg
            pixelcorrection!(bg, other_bg, vl)
        end
    end
    for video_id = 1:nvideos(vl)
        bg.b[:, video_id] ./= CUDA.norm(view(bg.b, :, video_id))
    end
end

function pixelcorrections(bg::PerVideoRank1Background, j)
    view(bg.bfC, :, j)
end

function pixelcorrection!(bg::PerVideoRank1Background, other_bg::PerVideoBackground, vl)
    for video_id = 1:nvideos(vl)
        fr = framerange_video(vl, video_id)
        bg.b[:, video_id] .-= view(other_bg.b, :, video_id) .* sum(view(bg.f, fr))
    end
end

function pixelcorrection!(bg::PerVideoRank1Background, other_bg::StaticBackground, vl)
    for video_id = 1:nvideos(vl)
        fr = framerange_video(vl, video_id)
        bg.b[:, video_id] .-= other_bg.b .* sum(view(bg.f, fr))
    end
end

function traceupdateseg!(bg::PerVideoRank1Background, Y_seg, i, vl)
    video_id = video_idx(vl, i)
    bg.raw_f[framerange(vl, i)] .= Y_seg' * view(bg.b, :, video_id)
end

function updatetracecorrections!(bg::PerVideoRank1Background, sol, vl)
    bg.bAf = CUDA.zeros(nframes(vl), ncells(sol))
    for video_id = 1:nvideos(vl)
        fr = framerange_video(vl, video_id)
        bg.bAf[fr, :] .= bg.f[fr] * (sol.A * view(bg.b, :, video_id))'
    end
end

function traceupdate!(bg::PerVideoRank1Background, sol, vl)
    bg.f .= bg.raw_f
    for video_id = 1:nvideos(vl)
        fr = framerange_video(vl, video_id)
        bg.f[fr, :] .-= sol.C[fr, :] * (sol.A * view(bg.b, :, video_id))
    end
    for other_bg in sol.backgrounds
        if other_bg !== bg
            tracecorrection!(bg, other_bg, vl)
        end
    end
end

function tracecorrection(bg::PerVideoRank1Background, j)
    view(bg.bAf, :, j)
end


function tracecorrection!(bg::PerVideoRank1Background, other_bg::PerVideoBackground, vl)
    for video_id = 1:nvideos(vl)
        overlap = view(bg.b, :, video_id)' * view(other_bg.b, :, video_id)
        fr = framerange_video(vl, video_id)
        bg.f[fr] .-= overlap
    end
end

function tracecorrection!(bg::PerVideoRank1Background, other_bg::StaticBackground, vl)
    for video_id = 1:nvideos(vl)
        overlap = view(bg.b, :, video_id)' * other_bg.b
        fr = framerange_video(vl, video_id)
        bg.f[fr] .-= overlap
    end
end

function prepinit!(bg::PerVideoRank1Background, sol, vl)
    bg.f .= 0.0f0
    bg.b[:, 1] .= view(Float32.(maximum(CUDA.CuArray(sol.A), dims=1) .== 0), :)
    bg.b[:, 1] ./= CUDA.norm(view(bg.b, :, 1))
    for i=2:nvideos(vl)
        bg.b[:, i] = view(bg.b, :, 1)
    end
end

function initseg!(bg::PerVideoRank1Background, seg, seg_id, vl)
    nothing
end


function reconstruct_frame(bg::PerVideoRank1Background, frame_id, vl)
    seg_id = frame2seg(vl, frame_id)
    video_id = video_idx(vl, seg_id)
    return @CUDA.allowscalar view(bg.b, :, video_id) * bg.f[frame_id]
end