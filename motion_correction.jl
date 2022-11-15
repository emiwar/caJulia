import CUDA
import HDF5
using ProgressMeter
#import FFTW
CUDA.allowscalar(false)

#include("videoLoader.jl")

struct MotionCorrecter{PlanT}
    shifts::Vector{Tuple{Int64, Int64}}
    phaseDiffs::Vector{Float32}
    freqs1::CUDA.CUFFT.AbstractFFTs.Frequencies{Float32}
    freqs2::CUDA.CUFFT.AbstractFFTs.Frequencies{Float32}
    plan::PlanT
end

function MotionCorrecter(nFrames, frameSize)
    shifts = fill((0, 0), nFrames)
    phaseDiffs = fill(0.0f0, nFrames)
    dummyFrame = CUDA.CuMatrix{Float32}(undef, frameSize...)
    freqs1 = CUDA.CUFFT.fftfreq(frameSize[1], 1.0f0)
    freqs2 = CUDA.CUFFT.fftfreq(frameSize[2], 1.0f0)
    plan = CUDA.CUFFT.plan_fft(dummyFrame)
    MotionCorrecter(shifts, phaseDiffs, freqs1, freqs2, plan)
end

#MotionCorrecter(vl::VideoLoader) = MotionCorrecter(vl.nFrames, vl.frameSize)

function shiftSegment!(motionCorrecter::MotionCorrecter, seg, frame_size,
                       frame_inds)
    shifted_freq = CUDA.CuMatrix{Complex{Float32}}(undef, frame_size...)
    shifted = CUDA.CuVector{Complex{Float32}}(undef, prod(frame_size))
    @showprogress "Shifting video" for i=axes(seg, 2)
        frame_id = frame_inds[i]
        freq = motionCorrecter.plan * reshape(view(seg, :, i), frame_size...)
        shift = motionCorrecter.shifts[frame_id]
        phaseDiff = motionCorrecter.phaseDiffs[frame_id]
        shifted_freq .= freq .* cis.(-Float32(2pi) .* (motionCorrecter.freqs1 .* shift[1] .+
                                                       motionCorrecter.freqs2' .* shift[2]) .+
                                                       phaseDiff)
        shifted .= view(motionCorrecter.plan \ shifted_freq, :)
        seg[:, i] .= real.(shifted)
    end
end

function fitMotionCorrection!(vl::VideoLoader, sol::Sol)
    mc = vl.motionCorrecter
    eachSegment(vl; shift=false) do seg_id, seg
        frameRange = vl.frameRanges[seg_id]
        frameSize = vl.frameSize
        cntr = (1 .+ frameSize) ./ 2
        @showprogress "Motion correcting" for i=axes(seg, 2)
            frame_id = frameRange[i]
            raw_frame = reshape(view(seg, :, i), frameSize...)
            raw_freq = mc.plan * raw_frame;
            target_frame = reshape(reconstruct_frame(sol, frame_id), frameSize...)
            target_freq = mc.plan * target_frame
            cross_corr_freq = raw_freq .* conj(target_freq)
            cross_corr = mc.plan \ cross_corr_freq
            _, max_flat_idx = CUDA.findmax(view(abs.(cross_corr), :))
            maxidx = CartesianIndices(cross_corr)[max_flat_idx]
            max_val = @CUDA.allowscalar cross_corr[maxidx]
            phaseDiff = atan(imag(max_val), real(max_val))
            shift = (ifelse.(maxidx.I .> cntr, maxidx.I .- frameSize, maxidx.I) .- 1)
            shift = (-shift[1], -shift[2])
            shifted_freq = raw_freq .* cis.(-Float32(2pi) .* (mc.freqs1 .* shift[1] .+
                                                          mc.freqs2' .* shift[2]) .+
                                                          phaseDiff)
            shifted = real(mc.plan \ shifted_freq)
            seg[:, i] .= view(shifted, :)
            mc.shifts[frame_id] = shift
            mc.phaseDiffs[frame_id] = phaseDiff
        end
        vl.deviceArraysShifted[seg_id] = true
    end
end



function fitMotionCorrection!(vl::VideoLoader)
    mc = vl.motionCorrecter
    eachSegment(vl; shift=false) do seg_id, seg
        frameRange = vl.frameRanges[seg_id]
        frameSize = vl.frameSize
        cntr = (1 .+ frameSize) ./ 2
        step_size = 1
        mc.shifts[frameRange] .= [(0, 0)]
        while step_size < size(seg, 2)
            for i=1:(2*step_size):(size(seg, 2)-step_size)
                raw_frame = reshape(view(seg, :, i), frameSize...)
                raw_freq = mc.plan * raw_frame;
                target_frame = reshape(view(seg, :, i+step_size), frameSize...)
                target_freq = mc.plan * target_frame
                cross_corr_freq = raw_freq .* conj(target_freq)
                cross_corr = mc.plan \ cross_corr_freq
                _, max_flat_idx = CUDA.findmax(view(abs.(cross_corr), :))
                maxidx = CartesianIndices(cross_corr)[max_flat_idx]
                max_val = @CUDA.allowscalar cross_corr[maxidx]
                phaseDiff = atan(imag(max_val), real(max_val))
                shift = (ifelse.(maxidx.I .> cntr, maxidx.I .- frameSize, maxidx.I) .- 1)
                shifted_freq = target_freq .* cis.(-Float32(2pi) .* (mc.freqs1 .* shift[1] .+
                                                                     mc.freqs2' .* shift[2]) .+
                                                                     phaseDiff)
                shifted = real(mc.plan \ shifted_freq)
                seg[:, i] .+= view(shifted, :)
                
                start_frame = frameRange[min(i+step_size, size(seg, 2))]
                end_frame = frameRange[min(i+2*step_size-1, size(seg, 2))]
                for j = start_frame:end_frame
                    mc.shifts[j] = mc.shifts[j] .+ (shift[1], shift[2])
                    mc.phaseDiffs[j] = phaseDiff
                end
            end
            step_size *= 2
        end
        #The video segment has been modified, so should force reloading before
        #using it again.
        delete!(vl.deviceArrays, seg_id)
    end
end
