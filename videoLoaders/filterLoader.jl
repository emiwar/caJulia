import NNlib
import cuDNN

struct FilterLoader{T <: VideoLoader} <: SegmentLoader
    source_loader::T
    filterkernel::CUDA.CuArray{Float32, 4, CUDA.Mem.DeviceBuffer}
end

function FilterLoader(source_loader::VideoLoader, filterkernel::CUDA.CuMatrix)
    FilterLoader(source_loader, reshape(filterkernel,
                                        size(filterkernel)...,
                                        1, 1))
end

function FilterLoader(source_loader::VideoLoader, filterkernel::Matrix)
    FilterLoader(source_loader, CUDA.cu(filterkernel))
end

function readseg(vl::FilterLoader, i)
    if location(vl.source_loader) == :device
        seg = readseg(vl.source_loader, i)
    else
        seg = eltype(vl).(CUDA.cu(readseg(vl.source_loader, i)))
    end
    nframes = size(seg, ndims(seg))
    kernelsize = maximum(size(vl.filterkernel))
    reshaped = reshape(seg, framesize(vl)..., 1, nframes)
    padded = NNlib.pad_reflect(reshaped, div(kernelsize, 2))
    reshape(cuDNN.cudnnConvolutionForward(vl.filterkernel, padded), framesize(vl)..., nframes)
end

location(::FilterLoader) = :device
Base.eltype(vl::FilterLoader) = Float32
nsegs(vl::FilterLoader) = nsegs(vl.source_loader)
nframes(vl::FilterLoader) = nframes(vl.source_loader)
framesize(vl::FilterLoader) = framesize(vl.source_loader)
framerange(vl::FilterLoader, i) = framerange(vl.source_loader, i) 
framerange_video(vl::FilterLoader, i) = framerange_video(vl.source_loader, i)
multivideo(vl::FilterLoader) = multivideo(vl.source_loader)
nvideos(vl::FilterLoader) = nvideos(vl.source_loader)
video_idx(vl::FilterLoader, i) = video_idx(vl.source_loader, i)
optimalorder(vl::FilterLoader) = optimalorder(vl.source_loader)