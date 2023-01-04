mutable struct PerVideoBackground <: Background
    m::CUDA.CuMatrix{Float32}
    b::CUDA.CuMatrix{Float32}
    bA::CUDA.CuMatrix{Float32}
    bC::CUDA.CuMatrix{Float32}
end

function PerVideoBackground(vl::VideoLoader)
    npixels = prod(framesize(vl))
    nvid = nvideos(vl)
    PerVideoBackground(CUDA.zeros(npixels, nvid),
                       CUDA.zeros(npixels, nvid),
                       CUDA.zeros(0, 0),
                       CUDA.zeros(0, 0))
end

function updatepixelcorrections!(bg::PerVideoBackground, sol, vl)
    bg.bC = CUDA.zeros(prod(framesize(vl)), ncells(sol))
    for video_id = 1:nvideos(vl)
        fr = framerange_video(vl, video_id)
        bg.bC .+= view(bg.b, :, video_id) * sum(view(sol.C, fr, :); dims=1)
    end
end

function pixelcorrections(bg::PerVideoBackground, j)
    view(bg.bC, :, j)
end

function updatetracecorrections!(bg::PerVideoBackground, sol, vl)
    bg.bA = CUDA.zeros(nframes(vl), ncells(sol))
    for video_id = 1:nvideos(vl)
        fr = framerange_video(vl, video_id)
        bg.bA[fr, :] .= reshape(sol.A * view(bg.b, :, video_id), (1, ncells(sol)))
    end
end

function tracecorrection(bg::PerVideoBackground, j)
    bg.bA[:, j]
end

function prepinit!(bg::PerVideoBackground, sol, vl)
    bg.m .= 0.0f0
end

function initseg!(bg::PerVideoBackground, seg, seg_id, vl)
    video_i = VideoLoaders.video_idx(vl, seg_id)
    n_frames_in_video = length(framerange_video(vl, video_i))
    bg.m[:, video_i] .+= view(sum(seg; dims=2), :) ./ n_frames_in_video 
    bg.b .= bg.m #Stupid, should fix...
end

function update!(bg::PerVideoBackground, sol, vl)
    for video_id = 1:nvideos(vl)
        fr = framerange_video(vl, video_id)
        bg.b[:, video_id] .= bg.m[:, video_id] .-
            view(sol.A'*sum(view(sol.C, fr, :); dims=1)' ./ nframes(vl), :)
    end
    for other_bg in sol.backgrounds
        if other_bg !== bg
            pixelcorrection!(bg, other_bg, vl)
        end
    end
end

function reconstruct_frame(bg::PerVideoBackground, frame_id, vl)
    seg_id = frame2seg(vl, frame_id)
    video_id = video_idx(vl, seg_id)
    return bg.b[:, video_id]
end