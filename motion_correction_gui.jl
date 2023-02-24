import GLMakie
import CUDA
import cuDNN
using ProgressMeter
include("videoLoaders/videoLoaders.jl")#include("motion_correction.jl")

example_files = ["../data/recording_20211016_163921.hdf5",
                 "../data/recording_20220919_135612.hdf5"]

    
nwbLoader = VideoLoaders.HDFLoader("../data/"*example_files[1],
                                "images", (1:1440, 1:1080,  1:2000))
splitLoader = VideoLoaders.SplitLoader(nwbLoader, 5)
hostCache = VideoLoaders.CachedHostLoader(splitLoader; max_memory=3.2e10)
deviceCache = VideoLoaders.CachedDeviceLoader(hostCache; max_memory=1.0e10)
#mc = MotionCorrecter(vl)
#mc.shifts .= [(round(25*cos(2pi*t/40)), round(25*sin(2pi*t/40))) for t=1:mc.nFrames]
#@CUDA.time shiftedVideoDevice = loadCorrectedSeg!(mc, vl, 1);

fig = GLMakie.Figure()

fig[2, 1] = playerControls = GLMakie.GridLayout()
nFrames = VideoLoaders.nframes(hostCache)
timeSlider = GLMakie.SliderGrid(playerControls[1, 1],
                        (label="Frame", format="{:d}",
                        range=1:nFrames,
                        startvalue=1)).sliders[1]
playing = GLMakie.Observable(false)
play_button = GLMakie.Button(playerControls[1, 2],
                     label=(GLMakie.@lift($playing ? "||" : "▷")))
GLMakie.on(play_button.clicks) do _
    playing[] = !(playing[])
end
function stepTime!(time_diff)
    t = timeSlider.value[]
    new_time = (t - 1 + time_diff + nFrames)%nFrames + 1
    GLMakie.set_close_to!(timeSlider, new_time)
end
prevFrameButton = GLMakie.Button(playerControls[1, 3], label="<")
GLMakie.on((_)->stepTime!(-1), prevFrameButton.clicks)
nextFrameButton = GLMakie.Button(playerControls[1, 4], label=">")
GLMakie.on((_)->stepTime!(1), nextFrameButton.clicks)

fig[1, 1] = topRow = GLMakie.GridLayout()

minFrame = CUDA.fill(Float32(Inf), (prod(VideoLoaders.framesize(deviceCache)), 1))
@showprogress "mapreduce" for i = VideoLoaders.optimalorder(deviceCache)
    seg = VideoLoaders.readseg(deviceCache, i)
    minFrame .= min.(minFrame, minimum(seg, dims=2))
end

sumFrame = CUDA.fill(Float32(0), (prod(VideoLoaders.framesize(deviceCache)), 1))
@showprogress "mapreduce" for i = VideoLoaders.optimalorder(deviceCache)
    seg = VideoLoaders.readseg(deviceCache, i)
    sumFrame .+= sum(seg, dims=2)
end
meanFrame = sumFrame ./ VideoLoaders.nframes(deviceCache)

#CUDA.synchronize()
videoDisplayType = GLMakie.Observable(:original)
current_frame = GLMakie.lift( timeSlider.value, videoDisplayType) do t, vdt
    if vdt == :original
        return Float32.(VideoLoaders.readframe(hostCache, t))#reshape(Array(view(originalVideoDevice, :, t)), vl.frameSize...))
    elseif vdt == :smoothed_sub
        rawframe = VideoLoaders.readframe(deviceCache, t)
        diff = rawframe .- view(smoothed, :)
        return reshape(Array(diff), VideoLoaders.framesize(deviceCache))
    else
        error("Unknown videoDisplayType $vdt")
    end
end

contrast_range = -128:1024#Int64(maximum(originalVideoDevice))
contrast_slider = GLMakie.IntervalSlider(topRow[1, 1], range=contrast_range)
ax1 = GLMakie.Axis(topRow[2, 1])
GLMakie.image!(ax1, current_frame, colorrange=contrast_slider.interval,
       interpolate=false)
ax1.aspect = GLMakie.DataAspect()

contrast_slider_min = GLMakie.IntervalSlider(topRow[1, 2], range=contrast_range)
ax2 = GLMakie.Axis(topRow[2, 2])
#reshape(Array(minFrame), VideoLoaders.framesize(deviceCache)...)
#reshape(meanFrame, frame_size) .-
GLMakie.image!(ax2, Array(smoothed), colorrange=contrast_slider_min.interval,
               interpolate=false)
ax2.aspect = GLMakie.DataAspect()

GLMakie.linkaxes!(ax1, ax2)
GLMakie.display(fig)



frame_size = VideoLoaders.framesize(deviceCache)
m_reshaped = reshape(meanFrame, frame_size..., 1, 1); #size(Ad, 1)
s = 10
kernel = -Images.Kernel.gaussian(s)
kernel 
kernel = Images.OffsetArrays.no_offset_view(kernel)
kernel = CUDA.cu(reshape(kernel, size(kernel)..., 1, 1))#CUDA.ones(1+2*growth, 1+2*growth, 1, 1)
conved = cuDNN.cudnnConvolutionForward(kernel, m_reshaped; padding=2*s);
smoothed = reshape(conved, frame_size)

function filter(arr, kernel)
    kernel_d = CUDA.cu(reshape(kernel, size(kernel)..., 1, 1))
    arr_d = CUDA.cu(reshape(arr, size(arr)..., 1, 1))
    padding = div(size(kernel, 1) - 1, 2)
    conved = cuDNN.cudnnConvolutionForward(kernel_d, arr_d; padding);
    reshape(conved, size(arr))
end


#import VideoIO
#encoder_options = (color_range=2, crf=0, preset="slow")
#first_frame = reshape(Array(view(originalVideoDevice, :, 1) .- minFrame), vl.frameSize...)
#uint_first_frame = UInt8.(floor.(clamp.(first_frame .- 78, 0, 255)))
#VideoIO.open_video_out("example_video.mp4", uint_first_frame, framerate=20; encoder_options) do writer
#    @showprogress "Writing video" for t = axes(originalVideoDevice,2)
#        frame = reshape(Array(view(originalVideoDevice, :, t) .- minFrame), vl.frameSize...)
#        uint_frame = UInt8.(floor.(clamp.(frame .- 78, 0, 255)))
#        VideoIO.write(writer, uint_frame)
#    end
#end