import GLMakie
include("motion_correction.jl")

example_files = ["../data/recording_20211016_163921.hdf5",
                 "../data/recording_20220919_135612.hdf5"]

vl = HDFLoader(example_files[1], key="/images",
               deviceMemory=5e9);

originalVideoHost = loadToHost!(vl, 1);
originalVideoDevice = loadToDevice!(vl, 1);

mc = MotionCorrecter(vl)
mc.shifts .= [(round(25*cos(2pi*t/40)), round(25*sin(2pi*t/40))) for t=1:mc.nFrames]
@CUDA.time shiftedVideoDevice = loadCorrectedSeg!(mc, vl, 1);

fig = GLMakie.Figure()

fig[2, 1] = playerControls = GLMakie.GridLayout()
nFrames = size(originalVideoHost, 3)
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
minFrame = minimum(originalVideoDevice; dims=2);
videoDisplayType = GLMakie.Observable(:original)
current_frame = GLMakie.lift( timeSlider.value, videoDisplayType) do t, vdt
    if vdt == :original
        return reshape(Array(view(originalVideoDevice, :, t)), vl.frameSize...)
    elseif vdt == :motion_corrected
        return reshape(Array(view(shiftedVideoDevice, :, t)), vl.frameSize...)
    elseif vdt == :minimum_subtracted
        return reshape(Array(view(originalVideoDevice, :, t) .- minFrame), vl.frameSize...)
    else
        error("Unknown videoDisplayType $vdt")
    end
end

contrast_range = 0:Int64(maximum(originalVideoDevice))
contrast_slider = GLMakie.IntervalSlider(topRow[1, 1], range=contrast_range)
ax1 = GLMakie.Axis(topRow[2, 1])
GLMakie.image!(ax1, current_frame, colorrange=contrast_slider.interval,
       interpolate=false)
ax1.aspect = GLMakie.DataAspect()

contrast_slider_min = GLMakie.IntervalSlider(topRow[1, 2], range=contrast_range)
ax2 = GLMakie.Axis(topRow[2, 2])
GLMakie.image!(ax2, reshape(Array(minFrame), vl.frameSize...), colorrange=contrast_slider_min.interval,
               interpolate=false)
ax2.aspect = GLMakie.DataAspect()

GLMakie.linkaxes!(ax1, ax2)
GLMakie.display(fig)
