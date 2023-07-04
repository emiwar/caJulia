struct MotionCorrectionLoader{T <: VideoLoader} <: SegmentLoader
    source_loader::T
    window::NTuple{2, UnitRange{Int64}}
    shifts::Matrix{Int64}
    phaseDiffs::Vector{Float32}
    lensBg::CUDA.CuMatrix{Float32, CUDA.Mem.DeviceBuffer}
end

function MotionCorrectionLoader(source::VideoLoader)
    fs = framesize(source)
    MotionCorrectionLoader(source, (1:fs[1], 1:fs[2]))
end

function MotionCorrectionLoader(source::VideoLoader, window::NTuple{2, UnitRange{Int64}})
    shifts = zeros(Int64, nframes(source), 2)
    phaseDiffs = fill(0f0, nframes(source))
    MotionCorrectionLoader(source, window, shifts, phaseDiffs, CUDA.zeros(framesize(source)))
end

function VideoLoaders.readseg(vl::VideoLoaders.MotionCorrectionLoader, i)
    seg = readseg(vl.source_loader, i)
    @assert ndims(seg) == 3
    seg .-= vl.lensBg
    fr = framerange(vl, i)
    shifts = view(mcshifts(vl), fr, :)
    phasediffs = view(mcphasediffs(vl), fr, :)
    #TODO: crop seg using min and max shifts (to avoid wrap-around effects)
    return shiftseg(seg, shifts, phasediffs)
end

function fitMotionCorrection!(vl::MotionCorrectionLoader; callback=(_,_,_)->nothing)
    callback("Reading first frame", 0, 1)
    first_frame = view(VideoLoaders.readframe(vl.source_loader, 1), vl.window...)
    cntr = (1 .+ size(first_frame)) ./ 2
    first_frame_freq = CUDA.CUFFT.fft(first_frame)
    fs = framesize(vl)
    sm_shifts_forward = CUDA.zeros(ComplexF32, fs)
    sm_shifts_backward = CUDA.zeros(ComplexF32, fs)
    sm_frame_unshifted = CUDA.zeros(Float32, fs)
    sm_frame_shifted_freq = CUDA.zeros(ComplexF32, fs)
    freqs1 = -Float32(2pi) .* CUDA.CUFFT.fftfreq(fs[1], 1.0f0) |> CUDA.CuArray
    freqs2 = -Float32(2pi) .* CUDA.CUFFT.fftfreq(fs[2], 1.0f0) |> CUDA.CuArray
    for (i, seg_id) = enumerate(VideoLoaders.optimalorder(vl.source_loader))
        seg = VideoLoaders.readseg(vl.source_loader, seg_id)
        sm_frame_unshifted .+= CUDA.sum(seg, dims=3)
        seg_window = view(seg, vl.window..., :)
        seg_window_freq = CUDA.CUFFT.fft(seg_window, (1 ,2))
        seg_cross_corr_freq = first_frame_freq .* conj(seg_window_freq)
        seg_cross_corr = CUDA.CUFFT.ifft(seg_cross_corr_freq, (1,2))
        for (i, f) in enumerate(framerange(vl, seg_id))
            #Find the shift using max cross-corr
            cross_corr = view(seg_cross_corr, :, :, i)
            _, max_flat_idx = CUDA.findmax(view(abs.(cross_corr), :))
            maxidx = CartesianIndices(cross_corr)[max_flat_idx]
            max_val = @CUDA.allowscalar cross_corr[maxidx]
            phaseDiff = atan(imag(max_val), real(max_val))
            shift = (ifelse.(maxidx.I .> cntr, maxidx.I .- length.(vl.window), maxidx.I) .- 1)
            vl.shifts[f, :] .= shift
            vl.phaseDiffs[f] = phaseDiff

            #Calculate some statistics for lens artifact correction
            forward_shift = cis.((freqs1 .* shift[1] .+ freqs2' .* shift[2]) .+ phaseDiff)
            backward_shift = cis.(.-(freqs1 .* shift[1] .+ freqs2' .* shift[2]) .- phaseDiff)
            sm_shifts_forward .+= forward_shift
            sm_shifts_backward .+= backward_shift
            sm_frame_shifted_freq .+= forward_shift .* CUDA.CUFFT.fft(view(seg, :, :, i))
        end
        callback("Motion correcting", i, nsegs(vl))
    end
    #sm_frame_shifted = real.(CUDA.CUFFT.ifft(sm_frame_shifted_freq))
    sm_frame_unshifted_freq = CUDA.CUFFT.fft(sm_frame_unshifted)

    N = nframes(vl)
    unshifted_bg_freq = sm_frame_unshifted_freq ./ N
    shifted_bg_freq = zero(unshifted_bg_freq)

     for i=1:200
        shifted_bg_freq .= sm_frame_shifted_freq ./ N .- (unshifted_bg_freq .* sm_shifts_forward./ N)
        unshifted_bg_freq .= sm_frame_unshifted_freq ./ N .- (shifted_bg_freq .* sm_shifts_backward ./ N)
        callback("Correcting lens artifacts", i, 200)
    end

    vl.lensBg .= real.(CUDA.CUFFT.ifft(unshifted_bg_freq))
    #return shifted_bg_freq, unshifted_bg_freq, sm_frame_shifted_freq, sm_frame_unshifted_freq, sm_shifts_forward, sm_shifts_backward
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

mcshifts(vl::MotionCorrectionLoader) = vl.shifts
mcshifts(vl::VideoLoader) = mcshifts(vl.source_loader)

mcphasediffs(vl::MotionCorrectionLoader) = vl.phaseDiffs
mcphasediffs(vl::VideoLoader) = mcphasediffs(vl.source_loader)

function Base.show(io::IO, vl::MotionCorrectionLoader)
    Base.show(io, vl.source_loader)
    println(io, "Motion correction layer")
end

function shiftseg(seg, shifts, phaseDiffs)
    @assert ndims(seg) == 3
    freqs1 = -Float32(2pi) .* CUDA.CUFFT.fftfreq(size(seg, 1), 1.0f0)
    freqs2 = -Float32(2pi) .* CUDA.CUFFT.fftfreq(size(seg, 2), 1.0f0)
    #FFT along the w and h dimensions. The third dim is time.
    seg_freq = CUDA.CUFFT.fft(seg, (1, 2))
    
    for j=1:size(seg, 3)
        seg_freq[:, :, j] .*= cis.((freqs1 .* shifts[j, 1] .+ freqs2' .* shifts[j, 2])
                                    .+ phaseDiffs[j])
    end
    return real(CUDA.CUFFT.ifft(seg_freq, (1, 2)))
end

function shiftframe(frame, shift, phaseDiff)
    @assert ndims(seg) == 2
    freqs1 = -Float32(2pi) .* CUDA.CUFFT.fftfreq(size(seg, 1), 1.0f0)
    freqs2 = -Float32(2pi) .* CUDA.CUFFT.fftfreq(size(seg, 2), 1.0f0)
    
    frame_freq = CUDA.CUFFT.fft(frame, (1, 2))
    frame_freq .*= cis.((freqs1 .* shift[1] .+ freqs2' .* shift[2])
                         .+ phaseDiff)
    return real(CUDA.CUFFT.ifft(frame_freq, (1, 2)))
end

function readseg_bghack(vl::MotionCorrectionLoader, i::Integer,
                        lensbg)
    seg = readseg(vl.source_loader, i)
    @assert ndims(seg) == 3
    fr = framerange(vl, i)
    shifts = view(mcshifts(vl), fr, :)
    phasediffs = view(mcphasediffs(vl), fr, :)
    seg .-= reshape(lensbg, framesize(vl))
    #TODO: crop seg using min and max shifts (to avoid wrap-around effects)
    return shiftseg(seg, shifts, phasediffs)
end

function savetohdf(vl::MotionCorrectionLoader, hdfhandle)
    hdfhandle["/motion_correction/shifts"] = mcshifts(shifts)
    hdfhandle["/motion_correction/phaseDiffs"] = phasediffs(shifts)
    hdfhandle["/motion_correction/lensBg"] = Array(vl.lensBg)
    savetohdf(vl.source_loader, hdfhandle)
end
