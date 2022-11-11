import CUDA
import HDF5
using ProgressMeter
#import FFTW
CUDA.allowscalar(false)

include("videoLoader.jl")

function motionCorrectSegment(seg, target_frame)
    plan = CUDA.CUFFT.plan_fft(target_frame);
    target_frame_freq = plan * target_frame
    shape = size(target_frame)
    cntr = (1 .+ shape) ./ 2
    freqs1 = CUDA.CUFFT.fftfreq(shape[1], 1.0f0) |> collect |> CUDA.cu
    freqs2 = CUDA.CUFFT.fftfreq(shape[2], 1.0f0)' |> collect |> CUDA.cu
    function alignFrame(frame)
        source_freq = plan * frame;
        prod = source_freq .* conj(target_frame_freq);
        cross_corr = plan \ prod;
        _, maxidx = CUDA.findmax(abs.(cross_corr))
        max_val = @CUDA.allowscalar cross_corr[maxidx]
        phasediff = atan(imag(max_val), real(max_val))
        shift = ifelse.(maxidx.I .> cntr, maxidx.I .- shape, maxidx.I) .- 1
        shifted_freq = source_freq .* cis.(-Float32(2pi) .* (freqs1 .* shift[1] .+ freqs2 .* shift[2]) .+ phasediff)
        return real(plan \ shifted_freq), shift[1], shift[2]
    end
    T = size(seg, 2)
    aligned_frames = CUDA.zeros(shape[1], shape[2], T)
    shifts = zeros(Int64, 2, T)
    @showprogress "Motion correcting" for t=1:T
        frame = reshape(seg[:, t], shape...)
        aligned_frames[:, :, t], shifts[1, t], shifts[2, t] = alignFrame(frame)
    end
    return aligned_frames, shifts
end

function motionCorrect(fileName)
    hdfLoader = HDFLoader(fileName; key="/images",
                          deviceMemory=4e9, hostMemory=1.0e10);
    resultFileName = ".."*join(split(fileName, ".")[1:end-1])*"-MC.h5"
    HDF5.h5open(resultFileName, "w") do resultFile
        resultType = hostType(hdfLoader)
        T = hdfLoader.nFrames
        H_orig, W_orig = hdfLoader.frameSize
        H_crop = 150:830
        W_crop = 1:600
        H = length(H_crop)
        W = length(W_crop)
        resultDataSet = HDF5.create_dataset(resultFile, "/images",
                                            HDF5.datatype(resultType),
                                            HDF5.dataspace(H, W, T))
        shiftsDataSet = HDF5.create_dataset(resultFile, "/mc-shifts",
                                            HDF5.datatype(Int64),
                                            HDF5.dataspace(2, T))
        target_frame = Float32.(CUDA.cu(getFrameHost(hdfLoader, div(T, 2))))
        frames_per_seg = Int64(ceil(T / hdfLoader.nSegs))
        eachSegment(hdfLoader) do seg_id, seg
            start_frame = (seg_id-1)*frames_per_seg + 1
            end_frame = min(seg_id*frames_per_seg, T)
            aligned_frames, shifts = motionCorrectSegment(seg, target_frame)
            resultDataSet[:, :, start_frame:end_frame] = resultType.(round.(Array(aligned_frames[H_crop, W_crop, :])))
            shiftsDataSet[:, start_frame:end_frame] = shifts
        end
    end
end
