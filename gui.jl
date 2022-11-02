using GLMakie
using ProgressMeter
import HDF5
include("solution_struct.jl")
include("initROIs.jl")
include("fastHALS.jl")
include("oasis_opt.jl")
include("merge_split.jl")
include("viz.jl")

video = HDF5.h5open("../data/20211016_163921_animal1learnday1.nwb", "r") do fid
    HDF5.read(fid["analysis/recording_20211016_163921-PP-BP-MC/data"])
end
const frame_size = size(video)[1:2]
const n_frames = size(video, 3)
Y = Float32.(CUDA.cu(reshape(video, :, size(video, 3))));
sol = Sol(Y, frame_size);

fig = Figure()
fig[3, 1] = playerControls = GridLayout()
time_slider = SliderGrid(playerControls[1, 1],
                         (label="Frame", format="{:d}",
                         range=1:n_frames, startvalue=1))
current_time = time_slider.sliders[1].value
function step_time(steps=1)
    t = current_time[]
    next_t = (t-1+steps + n_frames)%n_frames + 1
    set_close_to!(time_slider.sliders[1], next_t)
end

playing = Observable(false)
play_button = Button(playerControls[1, 2], label=(@lift($playing ? "||" : "▷")))
on(play_button.clicks) do _
    playing[] = !(playing.val)
end
prev_frame_button = Button(playerControls[1, 3], label="<")
on((_)->step_time(-1), prev_frame_button.clicks)
next_frame_button = Button(playerControls[1, 4], label=">")
on((_)->step_time(1), next_frame_button.clicks)

run_task = Task() do 
    while true
        playing.val && step_time()
        sleep(0.05)
    end
end |> schedule

fig[1, 1] = topRow = GridLayout()
current_frame = lift(current_time) do t
    reshape(Array(Y[:, t]), frame_size...)
end
Y_min = minimum(Y)
Y_max = maximum(Y)
contrast_slider = IntervalSlider(topRow[1, 1], range=LinRange(Y_min, Y_max, 200))
ax1 = Axis(topRow[2, 1])
image!(ax1, current_frame, colorrange=contrast_slider.interval, interpolate=false)
ax1.aspect = DataAspect()

ax2 = Axis(topRow[2, 2])
summaryImg = Observable(reshape(Array(log10.(negentropy_img(Y))), sol.frame_size))
summary_range_slider = IntervalSlider(topRow[1, 2], range=LinRange(-7, 2, 200), startvalues=(-2.5, 0.5))
image!(ax2, summaryImg; colorrange=summary_range_slider.interval,
                        colormap=:inferno, nan_color=:black, interpolate=false)
ax2.aspect = DataAspect()

ax3 = Axis(topRow[2, 3])
footprintSummary = Observable(roiImg(sol)[1])
heatmap!(ax3, footprintSummary)
ax3.aspect = DataAspect()
linkaxes!(ax1, ax2)
linkaxes!(ax1, ax3)

traceAxis = Axis(fig[2, 1]; yticksvisible = false, yticks=([0], [""]))
function updateTracePlots()
    empty!(traceAxis)
    Ch = Array(sol.C)
    Rh = Array(sol.R)
    for i=1:size(Ch, 2)
        lines!(traceAxis, Rh[:, i] .- 200*i, color=:gray)
        lines!(traceAxis, Ch[:, i] .- 200*i, color=:black)
    end
    xlims!(traceAxis, 1, n_frames)
    vlines!(traceAxis, lift(t->[t], current_time), color=:red)
end

on(events(traceAxis).mousebutton) do event
    if event.button == Mouse.left && event.action == Mouse.press && is_mouseinside(traceAxis.scene)
        mouse_pos = events(traceAxis).mouseposition[] .- (52, 0)
        t = to_world(traceAxis.scene, Point2f(mouse_pos...))[1]
        set_close_to!(time_slider.sliders[1], t)
    end
end


updateTracePlots()
display(fig)

initA!(Y, sol);
zeroTraces!(sol);
initBackground!(Y, sol);
footprintSummary[] = roiImg(sol)[1]

updateTraces!(Y, sol; deconvFcn! = oasis_opt!);
updateROIs!(Y, sol);
footprintSummary[] = roiImg(sol)[1]
updateTracePlots()

while sum(merge!(sol; thres=0.8)) > 0
    println("Number of neurons ", size(sol.A, 1))
end
footprintSummary[] = roiImg(sol)[1]
updateTracePlots()