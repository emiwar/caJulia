#import NNlib
#import cuDNN

struct FilterLoader{T <: VideoLoader} <: SegmentLoader
    source_loader::T
    filter::CUDA.CuArray{ComplexF32, 2, CUDA.Mem.DeviceBuffer}
end

function FilterLoader(source_loader::VideoLoader, filterkernel::CUDA.CuMatrix)
    @assert size(filterkernel) == framesize(source_loader)
    filter = CUDA.CUFFT.fft(filterkernel)
    FilterLoader(source_loader, filter)
end

function FilterLoader(source_loader::VideoLoader, filterkernel::Matrix)
    filterkernel = pad_filter(filterkernel, framesize(source_loader))
    FilterLoader(source_loader, CUDA.cu(filterkernel))
end

function readseg(vl::FilterLoader, i)
    if location(vl.source_loader) == :device
        seg = readseg(vl.source_loader, i)
    else
        seg = eltype(vl).(CUDA.cu(readseg(vl.source_loader, i)))
    end
    @assert ndims(seg) == 3
    #nframes = size(seg, ndims(seg))
    #kernelsize = maximum(size(vl.filterkernel))
    #reshaped = reshape(seg, framesize(vl)..., 1, nframes)
    #TODO: this is wrong, must pass dims=(...) to pad_reflect
    #padded = NNlib.pad_reflect(reshaped, div(kernelsize, 2))
    #reshape(cuDNN.cudnnConvolutionForward(vl.filterkernel, padded), framesize(vl)..., nframes)
    seg_freq = CUDA.CUFFT.fft(seg, (1, 2))
    seg_freq .*= vl.filter
    CUDA.CUFFT.ifft!(seg_freq, (1,2))
    seg .= real.(seg_freq)
    return seg
end

function setfilterkernel(vl::FilterLoader, filterkernel::CUDA.CuMatrix)
    if eltype(filterkernel) <: Complex
        vl.filter .= filterkernel
    else
        vl.filter .= CUDA.CUFFT.fft(filterkernel)
    end
end

function setfilterkernel(vl::FilterLoader, filterkernel::Matrix)
    filterkernel = pad_filter(filterkernel, framesize(vl))
    setfilterkernel(vl, CUDA.cu(filterkernel))
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

function Base.show(io::IO, vl::FilterLoader)
    Base.show(io, vl.source_loader)
    println(io, "Filtered")
end

function pad_filter(kernel, framesize)
    padded = zeros(eltype(kernel), framesize)
    for idx=CartesianIndices(kernel)
        r = (idx[1]+framesize[1]-1)%framesize[1] + 1
        c = (idx[2]+framesize[2]-1)%framesize[2] + 1
        padded[r, c] = kernel[idx]
    end
    return padded
end