import GLMakie
import CUDA
import cuDNN
import Images
using ProgressMeter
GLMakie.activate!(inline=false)
include("videoLoaders/videoLoaders.jl")
#include("motion_correction.jl")

#example_files = "../data/" .* ["../data/recording_20211016_163921.hdf5",
#                 "../data/recording_20220919_135612.hdf5"]

#server_folder = "/mnt/dmclab/Vasiliki/Striosomes experiments/Calcium imaging in oprm1/2nd batch sep 2022/Oprm1 calimaging 2nd batch sept 2022 - inscopix data/oprm1 day1 20220919/"
#example_files_server = server_folder .* [
#    "recording_20220919_135612.hdf5",
#    "recording_20220919_143826.hdf5",
#    "recording_20220919_144807.hdf5"
#]
example_file = "/mnt/dmclab/Emil/arrowmaze_raw_downsampled/oprm1 day1 20220919/recording_20220919_144807.hdf5"

#Full: (1:1440, 1:1080, :)
#manual_cropping_server = [
#    (750:1350, 250:850, 1:1000),
#    (1:1440, 1:1080, 1:1000),
#    (1:1440, 1:1080, 1:1000)
#]

nwbLoader = VideoLoaders.HDFLoader(example_file, "data", (1:720, 1:540, 1:7000))
splitLoader = VideoLoaders.SplitLoader(nwbLoader, 20)
hostCache = VideoLoaders.CachedHostLoader(splitLoader; max_memory=3.2e10)
#filterLoader = VideoLoaders.FilterLoader(hostCache, Images.OffsetArrays.no_offset_view(Images.Kernel.DoG(5.0)))
filterLoader = VideoLoaders.BandpassFilterLoader(hostCache, 1, 300)
#(300:600, 125:300, 1:3000))#(320:380, 80:140, 1:3000)#(600:1200, 250:600))
mcLoader = VideoLoaders.MotionCorrectionLoader(filterLoader, [(320:380, 80:140, 1:3000)])
deviceCache = VideoLoaders.CachedDeviceLoader(mcLoader; max_memory=1.0e10)
#mc = MotionCorrecter(vl)
#mc.shifts .= [(round(25*cos(2pi*t/40)), round(25*sin(2pi*t/40))) for t=1:mc.nFrames]
#@CUDA.time shiftedVideoDevice = loadCorrectedSeg!(mc, vl, 1);

deviceCache = VideoLoaders.openvideo("/mnt/dmclab/Emil/arrowmaze_raw_downsampled/animal6_v2_with_mcwindows.csv"; nsplits=100)
mcLoader = deviceCache.source_loader
filterLoader = mcLoader.source_loader
hostCache = filterLoader.source_loader

mc_result = VideoLoaders.fitMotionCorrection_v2!(mcLoader;
                callback=(m,i,n)->println("$m ($i/$n)"));
VideoLoaders.clear!(deviceCache)


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

#CUDA.synchronize()
videoDisplayType = GLMakie.Observable(:motionCorrected)

freq_slider = GLMakie.IntervalSlider(topRow[1, 1], range=0:500)

#fs = VideoLoaders.framesize(filterLoader)
#newFilter = CUDA.cu(VideoLoaders.generate_bandpass_filter_no_shift(fs, 1, 300))
#filterLoader.filter .= newFilter
#    filterLoader.filter .= newFilter
#GLMakie.on(freq_slider.interval) do (lo, hi)
#    fs = VideoLoaders.framesize(filterLoader)
#    newFilter = CUDA.cu(VideoLoaders.generate_bandpass_filter_no_shift(fs, lo, hi))
#    filterLoader.filter .= newFilter
#    videoDisplayType[] = videoDisplayType[]
#end
current_frame = GLMakie.lift(timeSlider.value, videoDisplayType) do t, vdt
    if vdt == :original
        return Float32.(VideoLoaders.readframe(hostCache, t))
    elseif vdt == :filtered
        frame_dev = VideoLoaders.readframe(filterLoader, t)
        return reshape(Array(frame_dev), VideoLoaders.framesize(filterLoader))
    elseif vdt == :motionCorrected
        frame_dev = VideoLoaders.readframe(deviceCache, t)
        return reshape(Array(frame_dev), VideoLoaders.framesize(deviceCache))
    else
        error("Unknown videoDisplayType $vdt")
    end
end

contrast_range = -256:1024#4096#Int64(maximum(originalVideoDevice))
contrast_slider = GLMakie.IntervalSlider(topRow[2, 1], range=contrast_range)
ax1 = GLMakie.Axis(topRow[3, 1])
GLMakie.image!(ax1, current_frame, colorrange=contrast_slider.interval,
       interpolate=false)
ax1.aspect = GLMakie.DataAspect()

#contrast_slider_min = GLMakie.IntervalSlider(topRow[1, 2], range=contrast_range)
#ax2 = GLMakie.Axis(topRow[2, 2])
#reshape(Array(minFrame), VideoLoaders.framesize(deviceCache)...)
#reshape(meanFrame, frame_size) .-
#GLMakie.image!(ax2, Array(smoothed), colorrange=contrast_slider_min.interval,
#               interpolate=false)
#ax2.aspect = GLMakie.DataAspect()

#GLMakie.linkaxes!(ax1, ax2)
GLMakie.display(fig)

VideoLoaders.fitMotionCorrection!(mcLoader)
VideoLoaders.clear!(deviceCache)

meanFrame = VideoLoaders.mapreduce(x->x, +, deviceCache, 0.0, dims=2)
meanFrame ./= VideoLoaders.nframes(deviceCache)


import Plots
import Images
import NNlib

function videoSum(vl::VideoLoaders.VideoLoader)
    sm = CUDA.zeros(prod(VideoLoaders.framesize(vl)))
    @showprogress for seg_id = VideoLoaders.optimalorder(vl)
        seg = VideoLoaders.readseg(vl, seg_id)
        sm .+= sum(reshape(seg, prod(VideoLoaders.framesize(vl)), size(seg, ndims(seg))), dims=2)
    end
    return sm
end

function shiftSum(bg, shifts, phaseDiffs)
    @assert ndims(bg) == 2
    freqs1 = CUDA.CUFFT.fftfreq(size(bg, 1), 1.0f0)
    freqs2 = CUDA.CUFFT.fftfreq(size(bg, 2), 1.0f0)
    bg_freq = CUDA.CUFFT.fft(bg)
    sm = zero(bg_freq)
    for i = 1:size(shifts, 1)
        sm .+= bg_freq .* cis.(-Float32(2pi) .* (freqs1 .* shifts[i, 1] .+
                                                 freqs2' .* shifts[i, 2]) .+
                               phaseDiffs[i])
    end
    return real(CUDA.CUFFT.ifft(sm))
end

frame_size = VideoLoaders.framesize(deviceCache)
function frameHeatmaps(frames...; kwargs...)
    hms = map(frames) do f
        Plots.heatmap(Array(reshape(f, frame_size))')
    end
    Plots.plot(hms..., fmt=:png, aspect=:equal; kwargs...)
end

sumUnshifted = reshape(videoSum(filterLoader), VideoLoaders.framesize(filterLoader));
sumShifted =  reshape(videoSum(deviceCache), VideoLoaders.framesize(deviceCache));
#Plots.heatmap(Array(reshape(sm, frame_size))', fmt=:png, aspect=:equal)

bgUnshifted = sumUnshifted ./ VideoLoaders.nframes(mcLoader);

for i=1:10
    shiftedSm = shiftSum(bgUnshifted, VideoLoaders.mcshifts(mcLoader), mcLoader.phaseDiffs);
    bgShifted = (sumShifted .- shiftedSm) ./ VideoLoaders.nframes(mcLoader);

    shiftedSm = shiftSum(bgShifted, -VideoLoaders.mcshifts(mcLoader), -mcLoader.phaseDiffs);
    bgUnshifted = (sumUnshifted .- shiftedSm) ./ VideoLoaders.nframes(mcLoader);
end
frameHeatmaps(bgUnshifted, bgShifted)



bgShifted = sumShifted ./ VideoLoaders.nframes(mcLoader);

@showprogress for i=1:100
    shiftedSm = shiftSum(bgShifted, -VideoLoaders.mcshifts(mcLoader), -mcLoader.phaseDiffs);
    bgUnshifted .= (sumUnshifted .- shiftedSm) ./ VideoLoaders.nframes(mcLoader);

    shiftedSm = shiftSum(bgUnshifted, VideoLoaders.mcshifts(mcLoader), mcLoader.phaseDiffs);
    bgShifted .= (sumShifted .- shiftedSm) ./ VideoLoaders.nframes(mcLoader);
end
frameHeatmaps(bgUnshifted, bgShifted, clim=(-10, 20))



vl = deviceCache
sm2 = CUDA.zeros(prod(VideoLoaders.framesize(vl)))
@showprogress for seg_id = VideoLoaders.optimalorder(vl)
    seg = VideoLoaders.readseg(vl, seg_id)
    sm2 .+= sum(reshape(seg, prod(VideoLoaders.framesize(vl)), size(seg, ndims(seg))), dims=2)
end
sm2 .-= shiftedSm
Plots.heatmap(Array(reshape(sm2, frame_size))', fmt=:png, aspect=:equal)

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

#frame = CUDA.zeros(size(current_frame[]) .+ 60)
#frame[31:end-30, 31:end-30] = current_frame[]
frame = VideoLoaders.readframe(hostCache, 1); #current_frame[]
frame = NNlib.pad_reflect(Float32.(CUDA.cu(frame)), (30, 30, 30, 30))
framesize = size(frame)

#frame = CUDA.ones(size(current_frame[]))
#framesize = size(frame)

dog = Images.Kernel.DoG(10.0)#(10.0, 10.0), (14.0, 14.0), framesize)

function pad_filter(kernel, framesize)
    padded = zeros(eltype(kernel), framesize)
    for idx=CartesianIndices(kernel)
        r = (idx[1]+framesize[1]-1)%framesize[1] + 1
        c = (idx[2]+framesize[2]-1)%framesize[2] + 1
        padded[r, c] = kernel[idx]
    end
    return padded
end

Plots.heatmap(Array(frame)', fmt=:png)
Plots.heatmap(Images.imfilter(Array(frame[31:end-30, 31:end-30]), dog)', fmt=:png, clim=(-10, 10))

padded_kernel = CUDA.cu(pad_filter(dog, framesize))
plan = CUDA.CUFFT.plan_fft(frame)

fft_kernel = plan * padded_kernel
fft_frame = plan * frame
fft_product = fft_kernel .* fft_frame

result = real(plan \ fft_product)
Plots.heatmap(Array(result[31:end-30, 31:end-30])', fmt=:png, clim=(-10, 10))




function generate_bandpass_filter_no_shift(size, low_radius, high_radius)
    h, w = size
    filter_mask = zeros(Float32, h, w)

    cy, cx = div.(size, 2)

    for i in 1:h
        for j in 1:w
            idist = i <= cy ? i - 1 : h - i
            jdist = j <= cx ? j - 1 : w - j
            dist = sqrt(idist^2 + jdist^2)
            if low_radius <= dist <= high_radius
                filter_mask[i, j] = 1.0
            end
        end
    end

    return filter_mask
end
plan = CUDA.CUFFT.plan_fft(CUDA.cu(current_frame[]))
fft_frame = plan * CUDA.cu(current_frame[])
framesize = size(fft_frame)
fft_frame .*= CUDA.cu(generate_bandpass_filter_no_shift(framesize, 10, 100))
#fft_frame[1:5, :] .= 0.0
#fft_frame[:, 1:5] .= 0.0
#fft_frame[1:5, :] .= 0.0
#fft_frame[:, 1:5] .= 0.0
Plots.heatmap(Array(real(plan \ fft_frame))', fmt=:png, clim=(-100, 100))



function fitMotionCorrection(vl::VideoLoaders.VideoLoader, subframe)
    nframes = VideoLoaders.nframes(vl)
    framesize = VideoLoaders.framesize(vl)
    shifts = fill((0, 0), nframes)
    phaseDiffs = fill(0.0f0, nframes)
    dummyFrame = CUDA.CuMatrix{Float32}(undef, framesize...)
    freqs1 = CUDA.CUFFT.fftfreq(framesize[1], 1.0f0)
    freqs2 = CUDA.CUFFT.fftfreq(framesize[2], 1.0f0)
    plan = CUDA.CUFFT.plan_fft(dummyFrame)
    @showprogress "mapreduce" for seg_id = VideoLoaders.optimalorder(deviceCache)
        seg = VideoLoaders.readseg(deviceCache, seg_id)
        frameRange = VideoLoaders.framerange(deviceCache, seg_id)
        cntr = (1 .+ framesize) ./ 2
        step_size = 1
        shifts[frameRange] .= [(0, 0)]
        while step_size < size(seg, 2)
            for i=1:(2*step_size):(size(seg, 2)-step_size)
                raw_frame = reshape(view(seg, :, i) .- subframe, framesize...)
                raw_freq = plan * raw_frame;
                target_frame = reshape(view(seg, :, i+step_size) .- subframe, framesize...)
                target_freq = plan * target_frame
                cross_corr_freq = raw_freq .* conj(target_freq)
                cross_corr = plan \ cross_corr_freq
                _, max_flat_idx = CUDA.findmax(view(abs.(cross_corr), :))
                maxidx = CartesianIndices(cross_corr)[max_flat_idx]
                max_val = @CUDA.allowscalar cross_corr[maxidx]
                phaseDiff = atan(imag(max_val), real(max_val))
                shift = (ifelse.(maxidx.I .> cntr, maxidx.I .- framesize, maxidx.I) .- 1)
                shifted_freq = target_freq .* cis.(-Float32(2pi) .* (freqs1 .* shift[1] .+
                                                                     freqs2' .* shift[2]) .+
                                                                     phaseDiff)
                shifted = real(plan \ shifted_freq)
                #seg[:, i] .+= view(shifted, :)
                
                start_frame = frameRange[min(i+step_size, size(seg, 2))]
                end_frame = frameRange[min(i+2*step_size-1, size(seg, 2))]
                for j = start_frame:end_frame
                    shifts[j] = shifts[j] .+ (shift[1], shift[2])
                    phaseDiffs[j] = phaseDiff
                end
            end
            step_size *= 2
        end
        #The video segment has been modified, so should force reloading before
        #using it again.
        VideoLoaders.clearseg!(vl, seg_id)
    end
    return shifts, phaseDiffs
end

shifts, phaseDiffs = fitMotionCorrection(deviceCache, meanFrame)
shifts = map(s->.-s, shifts)

maxFrame = CUDA.fill(Float32(-Inf), (prod(VideoLoaders.framesize(deviceCache)), 1))
@showprogress "mapreduce" for i = VideoLoaders.optimalorder(deviceCache)
    seg = VideoLoaders.readseg(deviceCache, i)
    maxFrame .= max.(minFrame, maximum(seg, dims=2))
end

sqrFrame = CUDA.fill(Float32(0.0), (prod(VideoLoaders.framesize(deviceCache)), 1))
@showprogress "mapreduce" for i = VideoLoaders.optimalorder(deviceCache)
    seg = VideoLoaders.readseg(deviceCache, i)
    sqrFrame .+= sum(seg .^ 2, dims=2)
end

function autoCorr(frame)
    framesize = size(frame)
    plan = CUDA.CUFFT.plan_fft(frame)
    freq = plan * frame
    cross_corr_freq = freq .* conj(freq)
    cross_corr = plan \ cross_corr_freq
    return abs.(cross_corr)
end


function findShiftPairwise(frame1, frame2, framesize)
    frame1 = reshape(frame1, framesize)
    frame2 = reshape(frame2, framesize)
    #freqs1 = CUDA.CUFFT.fftfreq(framesize[1], 1.0f0)
    #freqs2 = CUDA.CUFFT.fftfreq(framesize[2], 1.0f0)
    plan = CUDA.CUFFT.plan_fft(frame1)
    cntr = (1 .+ framesize) ./ 2
    frame1_freq = plan * frame1;
    frame2_freq = plan * frame2;
    cross_corr_freq = frame1_freq .* conj(frame2_freq)
    cross_corr = plan \ cross_corr_freq
    _, max_flat_idx = CUDA.findmax(view(abs.(cross_corr), :))
    maxidx = CartesianIndices(cross_corr)[max_flat_idx]
    #max_val = @CUDA.allowscalar cross_corr[maxidx]
    #phaseDiff = atan(imag(max_val), real(max_val))
    shift = (ifelse.(maxidx.I .> cntr, maxidx.I .- framesize, maxidx.I) .- 1)
    return shift
end


GLMakie.set_close_to!(timeSlider, 1)
window = (600:1200, 250:600)
#window = (600:1200, 50:800)
first_frame = view(CUDA.cu(current_frame[][window...]), :)
for t = 2:300
    GLMakie.set_close_to!(timeSlider, t)
    frame = view(CUDA.cu(current_frame[][window...]), :)
    println(findShiftPairwise(first_frame, frame, length.(window)))
end

frame1 = VideoLoaders.readframe(deviceCache, 1);
frame2 = VideoLoaders.readframe(deviceCache, 2);
frame3 = VideoLoaders.readframe(deviceCache, 3);
frame4 = VideoLoaders.readframe(deviceCache, 4);
frame5 = VideoLoaders.readframe(deviceCache, 5);
frame10 = VideoLoaders.readframe(deviceCache, 10);
frame20 = VideoLoaders.readframe(deviceCache, 20);
seg1 = VideoLoaders.readseg(deviceCache, 1);

plotFrame(frame; kwargs...) = Plots.heatmap(reshape(Array(frame), VideoLoaders.framesize(deviceCache)), fmt=:png; kwargs...)

subFrame(frame) = view(reshape(frame, VideoLoaders.framesize(deviceCache)), 600:950, 180:500)

foreground = CUDA.fill(Float32(0), (prod(VideoLoaders.framesize(deviceCache)), 1));
background = sum(view(seg1, :, 1:20), dims=2) ./ 20;
#hardcoded_mc = [(0,0), (-1, 5), (10, -5), (10, 5), (0, 10)]
for i=1:20
    background_subtracted = VideoLoaders.readframe(deviceCache, i) .- background
end

kernel = Images.OffsetArrays.no_offset_view(Images.Kernel.DoG(10.0)) |> CUDA.cu
seg_reshaped = reshape(seg1, framesize..., 1, size(seg1, 2));
padded = NNlib.pad_reflect(seg_reshaped, 30);
seg_smoothed = cuDNN.cudnnConvolutionForward(reshape(kernel, 61, 61, 1, 1), padded);



Plots.heatmap(seg_smoothed[:,:,1,1] |> Array, fmt=:png)







hdfLoader = VideoLoaders.HDFLoader("../fake_videos/fake3.h5", "images")
splitLoader = VideoLoaders.SplitLoader(hdfLoader, 10)
hostCache = VideoLoaders.CachedHostLoader(splitLoader; max_memory=3.2e10)
filterLoader = VideoLoaders.FilterLoader(hostCache, Images.OffsetArrays.no_offset_view(Images.Kernel.DoG(5.0)))
mcLoader = VideoLoaders.MotionCorrectionLoader(filterLoader, (200:400, 200:400, 1:500))#(300:600, 125:300, 1:2000))#(600:1200, 250:600))
deviceCache = VideoLoaders.CachedDeviceLoader(mcLoader; max_memory=1.0e10)


function makieims(vl, images...; clim=(-20, 50))
    nims = length(images)
    fig = GLMakie.Figure()
    obs_ims = map(im->GLMakie.lift(im->reshape(Array(im), VideoLoaders.framesize(mcLoader)), im), images)
    for (i, im)=enumerate(obs_ims)
        ax = GLMakie.Axis(fig[1, i] )
        GLMakie.image!(ax, im, colorrange=clim)
        ax.aspect = GLMakie.DataAspect()
    end
    GLMakie.display(fig)
end

mc_result = VideoLoaders.fitMotionCorrection_v2!(mcLoader;
                callback=(m,i,n)->println("$m ($i/$n)"));
N = Float32(length(mcLoader.windows[1][3]))#3000.0)
sm_frame_unshifted_freq = copy(mc_result.prelim_results.sm_frame_unshifted_freq)
sm_frame_shifted_freq = copy(mc_result.prelim_results.sm_frame_shifted_freq)
sm_shifts_forward = copy(mc_result.prelim_results.sm_shifts_forward)
sm_shifts_backward = copy(mc_result.prelim_results.sm_shifts_backward)
unshifted_bg_freq = sm_frame_unshifted_freq ./ N
shifted_bg_freq = zero(unshifted_bg_freq)
#shifted_bg_freq = sm_frame_shifted_freq ./ N
#unshifted_bg_freq = zero(shifted_bg_freq)
im1 = GLMakie.Observable(real.(CUDA.CUFFT.ifft(sm_frame_unshifted_freq)) ./ N);
im2 = GLMakie.Observable(real.(CUDA.CUFFT.ifft(sm_frame_shifted_freq)) ./ N);
#im2 = GLMakie.Observable(lensback.b);
makieims(mcLoader, im1, im2; clim=(-200, 400));


for i=1:200
    shifted_bg_freq .= sm_frame_shifted_freq ./ N .- (unshifted_bg_freq .* sm_shifts_forward./ N)
    unshifted_bg_freq .= sm_frame_unshifted_freq ./ N .- (shifted_bg_freq .* sm_shifts_backward ./ N)    
end
im2[] = real.(CUDA.CUFFT.ifft(shifted_bg_freq))
im1[] = real.(CUDA.CUFFT.ifft(unshifted_bg_freq))




vl = mcLoader
first_frame = VideoLoaders.readframe(vl.source_loader, vl.window[3][1])[vl.window[1:2]...]
prelim_results = VideoLoaders.fitToFrame(vl, first_frame, vl.window)
VideoLoaders.shiftlensbgbetweenvideos!(mcLoader)
im1 = GLMakie.Observable(real.(CUDA.CUFFT.ifft(prelim_results.sm_frame_unshifted_freq[:,:,1])) ./ 1000);
im2 = GLMakie.Observable(real.(CUDA.CUFFT.ifft(prelim_results.sm_frame_shifted_freq[:,:,1])) ./ 1000);
makieims(vl, im1, im2)
im2[] = real.(CUDA.CUFFT.ifft(prelim_results.shifted_bg_freq[:,:,1]))
im1[] = real.(CUDA.CUFFT.ifft(prelim_results.unshifted_bg_freq[:,:,1]))


target_frame = real.(CUDA.CUFFT.ifft(prelim_results.shifted_bg_freq[:,:,1]))
w, h = VideoLoaders.framesize(vl)
T = VideoLoaders.nframes(vl)

full_results = VideoLoaders.fitToFrame(vl, target_frame, (1:w, 1:h, 1:T);
                          applyLensBg=true, callback=(m,i,n)->println("$m ($i/$n)"))
#N = [length(intersect(1:1000, VideoLoaders.framerange(vl, vi))) for vi=1:VideoLoaders.nvideos(vl)]
#N = Float32.(CUDA.CuArray(reshape(max.(N, 1), 1, 1, VideoLoaders.nvideos(vl))))


sm = CUDA.zeros(Float32, VideoLoaders.framesize(filterLoader)...,
                VideoLoaders.nvideos(filterLoader))
sqrsm = zero(sm)
@showprogress for seg_id = 1:20#VideoLoaders.optimalorder(filterLoader)
    vid_id = VideoLoaders.video_idx(filterLoader, seg_id)
    seg = VideoLoaders.readseg(filterLoader, seg_id)
    sm[:, :, vid_id] .+= sum(seg, dims=3)
    sqrsm[:, :, vid_id] .+= sum(seg .^ 2, dims=3)
end

std = sqrsm ./ 3000 - ((sm ./ 3000.0) .^ 2)





vl = mcLoader
first_frames = map(vl.windows) do win
    VideoLoaders.readframe(vl.source_loader, win[3][1])[win[1:2]...]
end
callback=(m,i,n)->println("$m ($i/$n)")
prelim_callback = (m, i, N)->callback("[Prelim] $m", i, N)
prelim_results = VideoLoaders.fitToFrame(vl, first_frames, vl.windows;
                                         callback=prelim_callback)

HDF5.h5open("mc_test2_anim6.h5", "w") do fid
    fid["/motion_correction/shifts"] = VideoLoaders.mcshifts(mcLoader)
    fid["/motion_correction/phaseDiffs"] = VideoLoaders.mcphasediffs(mcLoader)
    fid["/motion_correction/lensBg"] = Array(mcLoader.lensBg)
end


vl = mcLoader
first_frames = map(vl.windows) do win
    VideoLoaders.readframe(vl.source_loader, win[3][1])[win[1:2]...]
end
prelim_callback = (m, i, N)->callback("[Prelim] $m", i, N)
prelim_results = VideoLoaders.fitToFrame(vl, first_frames, vl.windows;
                            callback=prelim_callback)

target_frame = real.(CUDA.CUFFT.ifft(prelim_results.shifted_bg_freq[:,:,1]))
#extra_wins = [(290:600, 200:450, 1:7000)]
#target_frames = [target_frame[extra_wins[i][1:2]...] for i=1:VideoLoaders.nvideos(vl)]
#prelim_results2 = VideoLoaders.fitToFrame(vl, target_frames, extra_wins;
#                            callback=prelim_callback, applyLensBg=true)
#VideoLoaders.clear!(deviceCache)
GLMakie.lines(mcLoader.shifts[:,1])
GLMakie.lines!(mcLoader.shifts[:,2])

zeroPaddedTarget = zero(target_frame)
zeroPaddedTarget[extra_wins[1][1:2]...] = target_frame[extra_wins[1][1:2]...]

w, h = VideoLoaders.framesize(vl)
full_window = [(1:w,1:h,VideoLoaders.framerange_video(vl, i))
                for i=1:VideoLoaders.nvideos(vl)]
prelim_results3 = VideoLoaders.fitToFrame(vl, [zeroPaddedTarget], full_window;
                            callback=prelim_callback, applyLensBg=true)






function recalcLensBg(vl::VideoLoaders.MotionCorrectionLoader, callback=(_,_,_)->nothing)
    nvids = VideoLoaders.nvideos(vl)
    fullfs = VideoLoaders.framesize(vl)
    sm_shifts_forward = CUDA.zeros(ComplexF32, fullfs..., nvids)
    sm_shifts_backward = CUDA.zeros(ComplexF32, fullfs..., nvids)
    sm_frame_unshifted = CUDA.zeros(Float32, fullfs..., nvids)
    sm_frame_shifted_freq = CUDA.zeros(ComplexF32, fullfs..., nvids)
    freqs1 = -Float32(2pi) .* CUDA.CUFFT.fftfreq(fullfs[1], 1.0f0) |> CUDA.CuArray
    freqs2 = -Float32(2pi) .* CUDA.CUFFT.fftfreq(fullfs[2], 1.0f0) |> CUDA.CuArray
    for (i, seg_id) = enumerate(VideoLoaders.optimalorder(vl.source_loader))
        vid_idx = VideoLoaders.video_idx(vl, seg_id)
        seg = VideoLoaders.readseg(vl.source_loader, seg_id)
        for (j, f) in enumerate(VideoLoaders.framerange(vl, seg_id))
            sm_frame_unshifted[:, :, vid_idx] .+= view(seg, :, :, j)
            shiftx = vl.shifts[f, 1]
            shifty = vl.shifts[f, 2]
            phaseDiff = vl.phaseDiffs[f]
            forward_shift = cis.((freqs1 .* shiftx .+ freqs2' .* shifty) .+ phaseDiff)
            sm_shifts_forward[:, :, vid_idx] .+= forward_shift
            sm_shifts_backward[:, :, vid_idx] .+= 1.0f0 ./ forward_shift
            sm_frame_shifted_freq[:, :, vid_idx] .+= forward_shift .* CUDA.CUFFT.fft(view(seg, :, :, j))
        end
        if i%10==0
            GC.gc()
        end
        callback("Recalculating artifact corrections.", i, VideoLoaders.nsegs(vl))
    end
    sm_frame_unshifted_freq = CUDA.CUFFT.fft(sm_frame_unshifted, (1,2))

    #N = Float32(length(window[3]))#nframes(vl)
    N = [length(VideoLoaders.framerange_video(vl, vi)) for vi=1:nvids]
    N = Float32.(CUDA.CuArray(reshape(max.(N, 1), 1, 1, nvids)))
    unshifted_bg_freq = sm_frame_unshifted_freq ./ N
    shifted_bg_freq = zero(unshifted_bg_freq)
    for i=1:500
        shifted_bg_freq .= sm_frame_shifted_freq ./ N .- (unshifted_bg_freq .* sm_shifts_forward./ N)
        unshifted_bg_freq .= sm_frame_unshifted_freq ./ N .- (shifted_bg_freq .* sm_shifts_backward ./ N)
        if i%100==0
            callback("Correcting lens artifacts", i, 500)
        end
    end

    vl.lensBg .= real.(CUDA.CUFFT.ifft(unshifted_bg_freq, (1, 2)))
    return (;shifted_bg_freq, unshifted_bg_freq, sm_frame_shifted_freq,
             sm_frame_unshifted_freq, sm_shifts_forward, sm_shifts_backward)
end

s="/mnt/dmclab/Emil/arrowmaze_raw_downsampled/animal6_v2_with_mcwindows.csv"
nsplits=100
pathPrefix = s[1:end-length(split(s, "/")[end])]
videolist = CSV.File(s,comment="#")#DataFrames.DataFrame(CSV.File(s,comment="#"))
sources = map(videolist) do r #map(eachrow(videolist))
    hdfLoader = VideoLoaders.HDFLoader(pathPrefix * r.filename, r.hdfKey)
    VideoLoaders.SplitLoader(hdfLoader, nsplits)
end
segs_per_video = [((i-1)*nsplits+1):i*nsplits for i=1:length(sources)]
output_labels = String.(map(r->r.resultKey, videolist))
baseLoader = VideoLoaders.MultiVideoLoader(sources, segs_per_video, output_labels)
hostCache = VideoLoaders.CachedHostLoader(baseLoader; max_memory=3.2e10)
filterLoader = VideoLoaders.BandpassFilterLoader(hostCache, 1, 300)
#filterloader = VideoLoaders.FilterLoader(hostcache, filterkernel)
#filterloader = BandpassFilterLoader(hostcache, 2, 100)
mcwindows = map(enumerate(videolist)) do (i, r) #map(eachrow(videolist))
    (r.mcwindowxmin:r.mcwindowxmax,
     r.mcwindowymin:r.mcwindowymax,
     VideoLoaders.framerange_video(filterLoader, i)[1:3000])
end
mcLoader = VideoLoaders.MotionCorrectionLoader(filterLoader, mcwindows)
deviceCache = VideoLoaders.CachedDeviceLoader(mcLoader, max_memory=5e10)

shifts = HDF5.h5open(fid->read(fid, "/motion_correction/shifts"), "animal6_test_init.h5")
phaseDiffs = HDF5.h5open(fid->read(fid, "/motion_correction/phaseDiffs"), "animal6_test_init.h5")

mcLoader.shifts .= shifts
mcLoader.phaseDiffs .= phaseDiffs

refitted_res = recalcLensBg(mcLoader, callback)

HDF5.h5open("animal6_recalced_lens_bg.h5", "w") do fid
    VideoLoaders.savetohdf(mcLoader, fid)
    fid["/motion_correction/extra/unshifted_bg"] = (real.(CUDA.CUFFT.ifft(refitted_res.unshifted_bg_freq, (1, 2))) |> Array)
    fid["/motion_correction/extra/shifted_bg"] = (real.(CUDA.CUFFT.ifft(refitted_res.shifted_bg_freq, (1, 2))) |> Array)
end


shifted_bgs = real.(CUDA.CUFFT.ifft(refitted_res.shifted_bg_freq, (1, 2))) |> Array

i = 0
i += 1; GLMakie.image!(shifted_bgs[:, :, i])

GLMakie.image!(shifted_bgs[:, :, 1])
GLMakie.image!(shifted_bgs[:, :, 8])
GLMakie.image!(shifted_bgs[:, :, 10])

extra_wins[1]
target_frame = shifted_bgs[:, :, 8]
zeroPaddedTarget = zero(target_frame)
zeroPaddedTarget[extra_wins[1][1:2]...] = target_frame[extra_wins[1][1:2]...]
GLMakie.image!(zeroPaddedTarget)

w, h = VideoLoaders.framesize(mcLoader)
full_window = [(1:w,1:h,VideoLoaders.framerange_video(mcLoader, i))
                for i=1:VideoLoaders.nvideos(mcLoader)]
targ = CUDA.cu(zeroPaddedTarget)
windowFitResults = VideoLoaders.fitToFrame(mcLoader, [targ for _=1:size(shifted_bgs, 3)],
                                           full_window; callback=callback, applyLensBg=true);

HDF5.h5open("animal6_re_motion_corrected.h5", "w") do fid
    VideoLoaders.savetohdf(mcLoader, fid)
    fid["/motion_correction/extra/unshifted_bg"] = (real.(CUDA.CUFFT.ifft(windowFitResults.unshifted_bg_freq, (1, 2))) |> Array)
    fid["/motion_correction/extra/shifted_bg"] = (real.(CUDA.CUFFT.ifft(windowFitResults.shifted_bg_freq, (1, 2))) |> Array)
end


HDF5.h5open("animal6_recalced_lens_bg.h5", "r") do fid
    VideoLoaders.loadfromhdf(mcLoader, fid)
end
GLMakie.lines(mcLoader.shifts[:, 1])
GLMakie.lines!(mcLoader.shifts[:, 2])

new_shifts = copy(mcLoader.shifts)
for i=2:size(shifts, 1)
    d = mcLoader.shifts[i, :] .- new_shifts[i-1, :]
    if any(d .> 50)
        new_shifts[i, :] .= new_shifts[i-1,:]
    end
end
GLMakie.lines(new_shifts[:, 1])
GLMakie.lines!(new_shifts[:, 2])

mcLoader.shifts .= new_shifts
HDF5.h5open("animal6_semimanual_mc.h5", "w") do fid
    VideoLoaders.savetohdf(mcLoader, fid)
end
