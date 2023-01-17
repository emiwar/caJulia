function handle_response(response_type::Symbol, data, observables)
    if response_type == :rawframe
        raw_frame[] = data
    elseif response_type == :reconstructedframe
        rec_frame[] = data
    elseif response_type == :initframe
        init_frame[] = data
        data_min, data_max = extrema(view(data, isfinite.(data)))
        observables["cmin3"][] = data_min
        observables["cmax3"][] = data_max
    elseif response_type == :footprints
        img, peaks, = data
        footprints_frame[] = Colors.ARGB32.(img)
    elseif response_type == :nframes
        observables["n_frames"][] = data
        println("nframes = $data")
    else
        println("Unhandled response: $response_type")
    end
end