function handle_response(response_type::Symbol, data, observables)
    if response_type == :videoloaded
        send_request(conn, :reconstructedframe, observables["frame_n"][])
        send_request(conn, :initframe)
        send_request(conn, :footprints)
    elseif response_type == :framesize
        w, h = data
        observables["xmin"][] = 1
        observables["xmax"][] = w
        observables["ymin"][] = 1
        observables["ymax"][] = h
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
        #TODO: need to send per-pixel-id, not center-per-id
        footprints_peaks[] = peaks
        send_request(conn, :reconstructedframe, observables["frame_n"][])
    elseif response_type == :nframes
        observables["n_frames"][] = data
        println("nframes = $data")
    elseif response_type == :trace
        traceS, traceC, traceR, col = data
        yscale = max(maximum(traceR), maximum(traceC))
        observables["traceS"][] = traceS ./ yscale
        observables["traceC"][] = traceC ./ yscale
        observables["traceR"][] = traceR ./ yscale
        observables["traceCol"][] = "#"*Colors.hex(col)
    else
        println("Unhandled response: $response_type")
    end
end