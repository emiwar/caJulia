function handle_response(response_type::Symbol, data, observables)
    if response_type == :rawframe
        raw_frame[] = data
    elseif response_type == :nframes
        observables["n_frames"][] = data
    else
        println("Unhandled response: $response_type")
    end
end