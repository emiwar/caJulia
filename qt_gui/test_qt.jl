ENV["QSG_RENDER_LOOP"] = "basic"
import CxxWrap
using Images
using Colors
using Observables
using QML
using Qt5QuickControls_jll
using Qt5QuickControls2_jll
include("videoCanvas.jl")
include("worker_connection.jl")
include("handle_responses.jl")

function checkworkerstatus(observables)
    yield() #Appearently needed to refresh the channels
    while isready(status)
        st, sp = take!(status)
        observables["status_text"][] = st
        observables["status_progress"][] = sp
    end
    while isready(responses)
        response_type, data = take!(responses)
        handle_response(response_type, data, observables)
    end
end

function pingworker()
    put!(jobs, (false, :ping, nothing))
end

function openvideo(Qfilename)
    #TODO: must be possible to parse this safer
    filename = String(QString(Qfilename))[8:end]
    submit_job(:loadvideo, filename)
end

function init_observables()
    observables = JuliaPropertyMap()
    observables["fractional_time"] = Observable(0.0)
    observables["n_frames"] = Observable{Int}(1)
    observables["frame_n"] = map((tf, N)->Int(clamp(round(tf*N), 1, N)),
                                 observables["fractional_time"],
                                 observables["n_frames"])
    observables["status_text"] = Observable("Starting worker process...")
    observables["status_progress"] = Observable(0.5)
    observables["frame_n"][] = observables["frame_n"][]
    return observables
end

const raw_frame = Observable(zeros(Int16, 5, 5))
on(raw_frame) do _
    @emit updateDisplay(1)
end
function run_gui()
    @qmlfunction openvideo
    @qmlfunction checkworkerstatus
    @qmlfunction pingworker
    QML.loadqml("qt_gui/gui.qml",
        paint_cfunction1 = video_canvas(raw_frame),
        paint_cfunction2 = video_canvas(raw_frame),
        paint_cfunction3 = video_canvas(raw_frame),
        paint_cfunction4 = video_canvas(raw_frame),
        timer = QTimer(),
        observables = init_observables()
    )
    QML.exec()
    send_request(:rawframe, 1)
end
run_gui()
