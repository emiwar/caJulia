mutable struct LensBackground <: Background
    uncorrectedMeanFrame::CUDA.CuVector{Float32}
    b::CUDA.CuVector{Float32}
    sumShiftFreqs::CUDA.CuVector{ComplexF32}
    bC::CUDA.CuMatrix{Float32}
    bA::CUDA.CuMatrix{Float32}
end

function LensBackground(framesize::Int64)
    LensBackground(CUDA.zeros(framesize), CUDA.zeros(framesize),
                   CUDA.zeros(ComplexF32, framesize),
                   CUDA.zeros(framesize,0), CUDA.zeros(framesize,0))
end
function LensBackground(vl::VideoLoader)
    LensBackground(prod(framesize(vl)))
end

### !!!
#=
TODO: Check if signs of shifts are correct, and if order of matrix
multiplications are correct. Also fix inits. And check if the CuVector types
need any more parameters.
=# 
### !!!

function prepinit!(bg::LensBackground, sol, vl)
    bg.uncorrectedMeanFrame .= 0.0f0
    bg.b .= 0.0f0
    fs = VideoLoaders.framesize(vl)
    shifts = VideoLoaders.mcshifts(vl)
    phasediffs = VideoLoaders.mcphasediffs(vl)
    freqs1 = -Float32(2pi) .* CUDA.CUFFT.fftfreq(fs[1], 1.0f0)
    freqs2 = -Float32(2pi) .* CUDA.CUFFT.fftfreq(fs[2], 1.0f0)
    sumShiftFreqs = CUDA.zeros(ComplexF32, fs)
    for i = 1:nframes(vl)
        sumShiftFreqs .+= cis.((freqs1 .* shifts[i, 1] .+ freqs2' .* shifts[i, 2]) .+
                                    phasediffs[i])
    end
    bg.sumShiftFreqs .= view(sumShiftFreqs, :)
end

function initseg!(bg::LensBackground, seg, seg_id, vl)
    fr = VideoLoaders.framerange(vl, seg_id)
    fs = VideoLoaders.framesize(vl)

    #TODO: would be better not to shift the segment BACK here, but read the unshifted one.
    shifts = -(VideoLoaders.mcshifts(vl)[fr, :])
    phasediffs = -(VideoLoaders.mcphasediffs(vl)[fr, :])
    shifted = VideoLoaders.shiftseg(reshape(seg, fs..., length(fr)), shifts, phasediffs)
    bg.uncorrectedMeanFrame .+= reshape(sum(shifted; dims=3), prod(fs)) ./ nframes(vl)
    bg.b .= bg.uncorrectedMeanFrame
end

function reconstruct_frame(bg::LensBackground, frame_id, vl)
    shift = -(VideoLoaders.mcshifts(vl)[frame_id, :])
    phasediff = -(VideoLoaders.mcphasediffs(vl)[frame_id])
    return VideoLoaders.shiftframe(bg.b, shift, phasediff)
end

function update!(bg::LensBackground, sol, vl)
    fs = VideoLoaders.framesize(vl)
    shifts = -VideoLoaders.mcshifts(vl)
    phasediffs = -VideoLoaders.mcphasediffs(vl)
    freqs1 = -Float32(2pi) .* CUDA.CUFFT.fftfreq(fs[1], 1.0f0)
    freqs2 = -Float32(2pi) .* CUDA.CUFFT.fftfreq(fs[2], 1.0f0)
    sm = CUDA.zeros(ComplexF32, fs)
    @showprogress for i = 1:nframes(vl)
        if ncells(sol) == 0
            f = CUDA.zeros(Float32, prod(framesize(vl)))
        else
            f = (sol.C[i, :]' * sol.A)'
        end
        for other_bg in sol.backgrounds
            if !(typeof(other_bg) <: LensBackground)
                f .+= reconstruct_frame(other_bg, i, vl)
            end
        end
        f_freq = CUDA.CUFFT.fft(reshape(f, fs))
        sm .+= f_freq .* cis.((freqs1 .* shifts[i, 1] .+ freqs2' .* shifts[i, 2]) .+
                               phasediffs[i])
    end
    bg.b .= bg.uncorrectedMeanFrame .- real.(view(CUDA.CUFFT.ifft(sm), :)) ./ nframes(vl)
end

function updatetracecorrections!(bg::LensBackground, sol, vl)
    b_freq = CUDA.CUFFT.fft(bg.b)
    meanshifted = real.(CUDA.CUFFT.ifft(b_freq .* bg.sumShiftFreqs)) ./ nframes(vl)
    bg.bA = sol.A * meanshifted
end

function tracecorrection(bg::LensBackground, j)
    @CUDA.allowscalar bg.bA[j]
end

function updatepixelcorrections!(bg::LensBackground, sol, vl)
    fs = VideoLoaders.framesize(vl)
    shifts = -VideoLoaders.mcshifts(vl)
    phasediffs = -VideoLoaders.mcphasediffs(vl)
    freqs1 = -Float32(2pi) .* CUDA.CUFFT.fftfreq(fs[1], 1.0f0)
    freqs2 = -Float32(2pi) .* CUDA.CUFFT.fftfreq(fs[2], 1.0f0)
    sm = CUDA.zeros(ComplexF32, prod(fs), ncells(sol))
    for i = 1:nframes(vl)
        bg_shifted_freq .+= bg_freq .* cis.((freqs1 .* shifts[i, 1] .+ freqs2' .* shifts[i, 2]) .+
                               phasediffs[i])
        sm .+= view(bg_shifted_freq, :) * view(sol.C, i, :)
    end
    bg.bC = real.(CUDA.CUFFT.ifft(reshape(sm, fs..., nframes(vl)), (1,2)))
end

function pixelcorrections(bg::LensBackground, j)
    view(bg.bC, :, j)
end

#Updating using only one other BG. This could be much
#more efficient if f_freq multiplication is out of the loop.
#Should probably fix that, and then do a larger number (100-200)
#of quick iterations when it is initialized.
function quickupdate!(bg::LensBackground, otherBg::StaticBackground, vl)
    fs = VideoLoaders.framesize(vl)
    shifts = -VideoLoaders.mcshifts(vl)
    phasediffs = -VideoLoaders.mcphasediffs(vl)
    freqs1 = -Float32(2pi) .* CUDA.CUFFT.fftfreq(fs[1], 1.0f0)
    freqs2 = -Float32(2pi) .* CUDA.CUFFT.fftfreq(fs[2], 1.0f0)
    sm = CUDA.zeros(ComplexF32, fs)
    f_freq = CUDA.CUFFT.fft(reshape(otherBg.b, fs))
    @showprogress for i = 1:nframes(vl)
        sm .+= f_freq .* cis.((freqs1 .* shifts[i, 1] .+ freqs2' .* shifts[i, 2]) .+
                               phasediffs[i])
    end
    bg.b .= bg.uncorrectedMeanFrame .- real.(view(CUDA.CUFFT.ifft(sm), :)) ./ nframes(vl)
end