function pixelcorrection!(bg::PerVideoBackground, other_bg::PerVideoRank1Background, vl)
    for video_id = 1:nvideos(vl)
        fr = framerange_video(vl, video_id)
        bg.b[:, video_id] .-= view(other_bg.b, :, video_id) .* sum(view(other_bg.f, fr)) ./ length(fr)
    end
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
