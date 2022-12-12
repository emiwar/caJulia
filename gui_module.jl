module GUI
using GLMakie

struct GUIState
    fig::Figure
    sol::Observable{Main.Sol}
    vid::Observable{Main.VideoLoaders.VideoLoader}
    timeSlider::Slider
    playing::Observable{Bool}
    selectedNeuron::Observable{Int64}
end

function GUIState(vl::Main.VideoLoaders.VideoLoader)
    fig = Figure()
    vid = Observable(vl)
    sol = lift(v->Main.Sol(v), vid)
    fig[3, 1] = playerControls = GridLayout()
    nFrames = lift(Main.nframes, vid)
    timeSlider = SliderGrid(playerControls[1, 1],
                            (label="Frame", format="{:d}",
                            range=@lift(1:$nFrames),
                            startvalue=1)).sliders[1]
    playing = Observable(false)
    selectedNeuron = Observable(0)
    guiState = GUIState(fig, sol, vid, timeSlider, playing, selectedNeuron)
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

    current_frame = lift((vl, t)->Main.readframe(vl.source_loader, t), guiState.vid, currentTime(guiState))
    Y_extrema = Observable((0, 500))#lift(Main.extrema, guiState.vid)
    contrast_range = lift((ex)->LinRange(ex[1], ex[2], 200), Y_extrema)
    contrast_slider = IntervalSlider(topRow[2, 1],
                                     range=contrast_range)
    ax1 = Axis(topRow[3, 1])
    image!(ax1, current_frame, colorrange=contrast_slider.interval,
                               interpolate=false)
    ax1.aspect = DataAspect()

    second_video_menu = GLMakie.Menu(topRow[1, 2], options=[:reconstructed,
                                                            :residual,
                                                            :background_subtracted])
    current_mc_frame = lift(guiState.vid,
                            guiState.sol,
                            currentTime(guiState),
                            second_video_menu.selection) do vl, sol, t, sel
        #seg_id = Main.VideoLoaders.frame2seg(vl, t)
        #frame_id = t - (first(Main.framerange(vl, seg_id)) - 1)
        frame_id = t
        #video_id = Main.seg2video(vl, frame_id)
        #if sel==:motion_corrected
        #    seg = Main.loadToDevice!(vl, seg_id, true)     
        #    reshape(Array(seg[:, frame_id]), vl.frameSize...)
        if sel==:reconstructed
            reshape(Array(Main.reconstruct_frame(sol, frame_id, vl)),
                    Main.framesize(vl)...)
        elseif sel==:residual
            reshape(Array(Main.residual_frame(sol, frame_id, vl)),
                    Main.framesize(vl)...)
        elseif sel==:background_subtracted
            reshape(Array(Main.bg_subtracted_frame(sol, frame_id, vl)),
                    Main.framesize(vl)...)
        else
            zeros(Float32, Main.framesize(vl)...)
        end
    end
    contrast_slider_mc = IntervalSlider(topRow[2, 2],
                                     range=contrast_range)
    ax2 = Axis(topRow[3, 2])
    image!(ax2, current_mc_frame, colorrange=contrast_slider_mc.interval,
                               interpolate=false)
    ax2.aspect = DataAspect()

    ax3 = Axis(topRow[3, 3])
    summaryImg = lift(s->reshape(Array(log10.(s.I)), s.frame_size...), guiState.sol)
    summary_range_slider = IntervalSlider(topRow[2, 3],
                    range=LinRange(-7, 2, 200), startvalues=(-2.5, 0.5))
    image!(ax3, summaryImg; colorrange=summary_range_slider.interval,
                            colormap=:inferno, nan_color=:black,
                            interpolate=false)
    ax3.aspect = DataAspect()

    ax4 = Axis(topRow[3, 4])
    footprintSummary = lift(Main.roiImg, guiState.sol)
    heatmap!(ax4, @lift($footprintSummary[1]))
    ax4.aspect = DataAspect()
    selectionMarker = lift(footprintSummary, guiState.selectedNeuron) do fpSm, sel
        if checkbounds(Bool, fpSm[2], sel)
            cartInd = fpSm[2][sel]
            return [Point2f(cartInd[1], cartInd[2]-20)]
        else
            return Point2f[]
        end
    end
    scatter!(ax4, selectionMarker, marker=:utriangle,
             color=:white, markersize=20,)

    linkaxes!(ax1, ax2)
    linkaxes!(ax1, ax3)
    linkaxes!(ax1, ax4)
end

function setupTraces(guiState::GUIState)
    traceAxis = Axis(guiState.fig[2, 1]; 
                     yticksvisible = false, yticks=([0], [""]))

    objToI = IdDict{Lines{Tuple{Vector{GLMakie.GeometryBasics.Point{2, Float32}}}}, Int64}()
    on(guiState.sol) do s
        empty!(traceAxis)
        Ch = Array(s.C)
        Rh = Array(s.R)
        for i=1:size(Ch, 2)
            lines!(traceAxis, Rh[:, i] .- 200*i, color=:gray, linewidth=0.5)
            l = lines!(traceAxis, Ch[:, i] .- 200*i, color=s.colors[i], linewidth=1.0)
            objToI[l] = i
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

    on(events(traceAxis).mouseposition) do event
        obj, _ = GLMakie.pick(traceAxis)
        if obj !== nothing && (obj in keys(objToI))
            guiState.selectedNeuron[] = objToI[obj]
        else
            guiState.selectedNeuron[] = 0
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
    nf = Main.nframes(guiState.vid[])
    new_time = (t - 1 + time_diff + nf)%nf + 1
    setTime!(guiState, new_time)
end

function calcI!(guiState::GUIState)
    I = Main.negentropy_img_per_video(guiState.vid[])
    guiState.sol[].I = I
    guiState.sol[] = guiState.sol[]
end

function initA!(guiState::GUIState; kwargs...)
    Main.initA!(guiState.sol[]; kwargs...)
    Main.zeroTraces!(guiState.sol[]);
    guiState.sol[] = guiState.sol[]
end

function initBackground!(guiState::GUIState)
    Main.initBackgrounds!(guiState.vid[], guiState.sol[])
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