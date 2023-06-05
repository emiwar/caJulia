struct MotionCorrectionLoader{T <: VideoLoader} <: SegmentLoader
    source_loader::T
    window::NTuple{2, UnitRange{Int64}}
    shifts::Matrix{Int64}
    phaseDiffs::Vector{Float32}
end

function MotionCorrectionLoader(source::VideoLoader)
    window = framesize(source)
    shifts = zeros(Int64, nframes(source), 2)
    phaseDiffs = fill(0f0, nframes(source))
    MotionCorrectionLoader(source, (1:window[1], 1:window[2]), shifts, phaseDiffs)
end

function VideoLoaders.readseg(vl::VideoLoaders.MotionCorrectionLoader, i)
    seg = readseg(vl.source_loader, i)
    @assert ndims(seg) == 3
    freqs1 = CUDA.CUFFT.fftfreq(size(seg, 1), 1.0f0)
    freqs2 = CUDA.CUFFT.fftfreq(size(seg, 2), 1.0f0)
    #FFT along the w and h dimensions. The third dim is time.
    seg_freq = CUDA.CUFFT.fft(seg, (1, 2))
    
    fr = framerange(vl, i)
    for j=1:size(seg, 3)
        frame = fr[j]
        seg_freq[:, :, j] .*= cis.(-Float32(2pi) .* (freqs1 .* vl.shifts[frame, 1] .+ freqs2' .* vl.shifts[frame, 2]) .+ vl.phaseDiffs[frame])
    end
    return real(CUDA.CUFFT.ifft(seg_freq, (1, 2)))
end

#=
function fitMotionCorrection!(vl::VideoLoaders.MotionCorrectionLoader)
    first_frame = view(readframe(vl.source_loader, 1), vl.window...)
    fft_plan = CUDA.CUFFT.plan_fft(first_frame)
    cntr = (1 .+ size(first_frame)) ./ 2
    first_frame_freq = fft_plan * first_frame;
    @showprogress for seg_id = optimalorder(vl.source_loader)
        seg = readseg(vl.source_loader, seg_id)
        for (i, f) in enumerate(framerange(vl, seg_id))
            frame = view(seg, vl.window[1], vl.window[2], i)
            frame_freq = fft_plan * frame;
            cross_corr_freq = first_frame_freq .* conj(frame_freq)
            cross_corr = fft_plan \ cross_corr_freq
            _, max_flat_idx = CUDA.findmax(view(abs.(cross_corr), :))
            maxidx = CartesianIndices(cross_corr)[max_flat_idx]
            max_val = @CUDA.allowscalar cross_corr[maxidx]
            phaseDiff = atan(imag(max_val), real(max_val))
            shift = (ifelse.(maxidx.I .> cntr, maxidx.I .- size(frame), maxidx.I) .- 1)
            vl.shifts[f, :] .= shift
            vl.phaseDiffs[f] = phaseDiff
        end
    end
    return nothing
end
=#

function fitMotionCorrection!(vl::VideoLoaders.MotionCorrectionLoader)
    first_frame = view(readframe(vl.source_loader, 1), vl.window...)
    #fft_plan = CUDA.CUFFT.plan_fft(first_frame)
    cntr = (1 .+ size(first_frame)) ./ 2
    first_frame_freq = CUDA.CUFFT.fft(first_frame);
    @showprogress for seg_id = optimalorder(vl.source_loader)
        seg = readseg(vl.source_loader, seg_id)
        seg_freq = CUDA.CUFFT.fft(seg, (1, 2))
        for (i, f) in enumerate(framerange(vl, seg_id))
            frame = view(seg, vl.window[1], vl.window[2], i)
            frame_freq = fft_plan * frame;
            cross_corr_freq = first_frame_freq .* conj(frame_freq)
            cross_corr = fft_plan \ cross_corr_freq
            _, max_flat_idx = CUDA.findmax(view(abs.(cross_corr), :))
            maxidx = CartesianIndices(cross_corr)[max_flat_idx]
            max_val = @CUDA.allowscalar cross_corr[maxidx]
            phaseDiff = atan(imag(max_val), real(max_val))
            shift = (ifelse.(maxidx.I .> cntr, maxidx.I .- size(frame), maxidx.I) .- 1)
            vl.shifts[f, :] .= shift
            vl.phaseDiffs[f] = phaseDiff
        end
    end
    return nothing
end

location(::MotionCorrectionLoader) = :device
Base.eltype(vl::MotionCorrectionLoader) = Float32
nsegs(vl::MotionCorrectionLoader) = nsegs(vl.source_loader)
nframes(vl::MotionCorrectionLoader) = nframes(vl.source_loader)
framesize(vl::MotionCorrectionLoader) = framesize(vl.source_loader)
framerange(vl::MotionCorrectionLoader, i) = framerange(vl.source_loader, i) 
framerange_video(vl::MotionCorrectionLoader, i) = framerange_video(vl.source_loader, i)
multivideo(vl::MotionCorrectionLoader) = multivideo(vl.source_loader)
nvideos(vl::MotionCorrectionLoader) = nvideos(vl.source_loader)
video_idx(vl::MotionCorrectionLoader, i) = video_idx(vl.source_loader, i)
optimalorder(vl::MotionCorrectionLoader) = optimalorder(vl.source_loader)
