function handle_response(response_type::Symbol, data, observables)
    if response_type == :videoloaded
        send_request(conn, :rawframe, observables["frame_n"][])
        send_request(conn, :reconstructedframe, observables["frame_n"][])
        send_request(conn, :initframe)
        send_request(conn, :footprints)
        observables["traceS"][] = zero(observables["traceS"][])
        observables["traceC"][] = zero(observables["traceC"][])
        observables["traceR"][] = zero(observables["traceR"][])
        observables["traceCol"][] = "#000000"
        observables["selected_cell"][] = 0
    elseif response_type == :framesize
        w, h = data
        observables["xmin"][] = 1
        observables["xmax"][] = w
        observables["ymin"][] = 1
        observables["ymax"][] = h
    elseif response_type == :initbackgrounds
        send_request(conn, :reconstructedframe, observables["frame_n"][])
    elseif response_type == :rawframe
        raw_frame[] = data
    elseif response_type == :reconstructedframe
        rec_frame[] = data
    elseif response_type == :initframe
        init_frame[] = data
        mask = isfinite.(data)
        if any(mask)
            data_min, data_max = extrema(view(data, mask))
            observables["cmin3"][] = data_min
            observables["cmax3"][] = data_max
        end
    elseif response_type == :footprints
        img, peaks = data
        footprints_frame[] = Colors.ARGB32.(img)
        footprints_peaks[] = peaks
        send_request(conn, :reconstructedframe, observables["frame_n"][])
    elseif response_type == :behaviorframe
        behavior_frame[] = data
    elseif response_type == :nframes
        observables["n_frames"][] = data
        observables["tmin"][] = 1
        observables["tmax"][] = data
    elseif response_type == :trace
        traceS, traceC, traceR, col = data
        yscale = max(maximum(traceR), maximum(traceC))
        observables["traceS"][] = traceS ./ yscale
        observables["traceC"][] = traceC ./ yscale
        observables["traceR"][] = traceR ./ yscale
        observables["traceCol"][] = "#"*Colors.hex(col)
    elseif response_type == :updatedtraces
        cell_id = observables["selected_cell"][] 
        if cell_id > 0
            send_request(conn, :trace, cell_id)
        end
    elseif response_type == :subtractedmin
        send_request(conn, :raw, observables["frame_n"][])
        send_request(conn, :reconstructedframe, observables["frame_n"][])
    else
        println("Unhandled response: $response_type")
    end
end