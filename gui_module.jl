module GUI
using GLMakie

struct GUIState
    fig::Figure
    sol::Observable{Main.Sol}
    vid::Observable{Main.VideoLoader}
    timeSlider::Slider
    playing::Observable{Bool}
end

function GUIState(vl::Main.VideoLoader)
    fig = Figure()
    vid = Observable(vl)
    sol = lift(v->Main.Sol(v), vid)
    fig[3, 1] = playerControls = GridLayout()
    nFrames = lift(Main.n_frames, vid)
    timeSlider = SliderGrid(playerControls[1, 1],
                            (label="Frame", format="{:d}",
                            range=@lift(1:$nFrames),
                            startvalue=1)).sliders[1]
    playing = Observable(false)
    guiState = GUIState(fig, sol, vid, timeSlider, playing)
    setupButtons(guiState, playerControls)
    setupImages(guiState)
    setupTraces(guiState)
    setupPlayTask(guiState)
    return guiState
end

function setupButtons(guiState::GUIState, playerControls)
    play_button = Button(playerControls[1, 2],
                         label=(@lift($(guiState.playing) ? "||" : "▷")))
    on(play_button.clicks) do _
        guiState.playing[] = !(guiState.playing[])
    end
    prevFrameButton = Button(playerControls[1, 3], label="<")
    on((_)->stepTime!(guiState, -1), prevFrameButton.clicks)
    nextFrameButton = Button(playerControls[1, 4], label=">")
    on((_)->stepTime!(guiState, 1), nextFrameButton.clicks)
end

function setupImages(guiState::GUIState)

    guiState.fig[1, 1] = topRow = GridLayout()
    current_frame = lift(Main.getFrameHost, guiState.vid, currentTime(guiState))
    Y_extrema = lift(Main.extrema, guiState.vid)
    contrast_range = lift((ex)->LinRange(ex[1], ex[2], 200), Y_extrema)
    contrast_slider = IntervalSlider(topRow[1, 1],
                                     range=contrast_range)
    ax1 = Axis(topRow[2, 1])
    image!(ax1, current_frame, colorrange=contrast_slider.interval,
                               interpolate=false)
    ax1.aspect = DataAspect()

    ax2 = Axis(topRow[2, 2])
    summaryImg = lift(s->reshape(Array(log10.(s.I)), s.frame_size...), guiState.sol)
    summary_range_slider = IntervalSlider(topRow[1, 2],
                    range=LinRange(-7, 2, 200), startvalues=(-2.5, 0.5))
    image!(ax2, summaryImg; colorrange=summary_range_slider.interval,
                            colormap=:inferno, nan_color=:black,
                            interpolate=false)
    ax2.aspect = DataAspect()

    ax3 = Axis(topRow[2, 3])
    footprintSummary = lift(Main.roiImg, guiState.sol)
    heatmap!(ax3, @lift($footprintSummary[1]))
    ax3.aspect = DataAspect()
    linkaxes!(ax1, ax2)
    linkaxes!(ax1, ax3)
end

function setupTraces(guiState::GUIState)
    traceAxis = Axis(guiState.fig[2, 1]; 
                     yticksvisible = false, yticks=([0], [""]))
    on(guiState.sol) do s
        empty!(traceAxis)
        Ch = Array(s.C)
        Rh = Array(s.R)
        for i=1:size(Ch, 2)
            lines!(traceAxis, Rh[:, i] .- 200*i, color=:gray)
            lines!(traceAxis, Ch[:, i] .- 200*i, color=:black)
        end
        xlims!(traceAxis, 1, size(Ch, 1))
        vlines!(traceAxis, lift(t->[t], currentTime(guiState)), color=:red)
    end

    on(events(traceAxis).mousebutton) do event
        if event.button == Mouse.left &&
           event.action == Mouse.press &&
           is_mouseinside(traceAxis.scene)
                #TODO: the to_world transform is somehow not correct here
                mouse_pos = events(traceAxis).mouseposition[] .- (52, 0)
                t = to_world(traceAxis.scene, Point2f(mouse_pos...))[1]
                setTime!(guiState, t)
        end
    end
end

function setupPlayTask(guiState::GUIState)
    run_task = Task() do 
        while true
            guiState.playing.val && stepTime!(guiState, 1)
            sleep(0.05)
        end
    end |> schedule
end

function currentTime(guiState::GUIState)
    return guiState.timeSlider.value
end

function setTime!(guiState::GUIState, new_time)
    set_close_to!(guiState.timeSlider, new_time)
end

function stepTime!(guiState::GUIState, time_diff)
    t = currentTime(guiState)[]
    nf = Main.n_frames(guiState.vid[])
    new_time = (t - 1 + time_diff + nf)%nf + 1
    setTime!(guiState, new_time)
end

function calcI!(guiState::GUIState)
    I = Main.negentropy_img(guiState.vid[])
    guiState.sol[].I = I
    guiState.sol[] = guiState.sol[]
end

function initA!(guiState::GUIState)
    Main.initA!(guiState.sol[])
    Main.zeroTraces!(guiState.sol[]);
    guiState.sol[] = guiState.sol[]
end

function initBackground!(guiState::GUIState)
    Main.initBackground!(guiState.vid[], guiState.sol[])
    guiState.sol[] = guiState.sol[]
end

function updateTraces!(guiState::GUIState)
    Main.updateTraces!(guiState.vid[], guiState.sol[], deconvFcn! = Main.oasis_opt!)
    guiState.sol[] = guiState.sol[]
end

function updateFootprints!(guiState::GUIState)
    Main.updateROIs!(guiState.vid[], guiState.sol[])
    guiState.sol[] = guiState.sol[]
end

function mergeCells!(guiState::GUIState; thres=0.6)
    Main.merge!(guiState.sol[]; thres=thres)
    guiState.sol[] = guiState.sol[]
end

end