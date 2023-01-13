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
    if Sys.iswindows()
        filename = String(QString(Qfilename))[9:end]
    else
        filename = String(QString(Qfilename))[8:end]
    end
    submit_job(:loadvideo, filename)
end

function init_observables()
    observables = JuliaPropertyMap()
    observables["frame_n_float"] = Observable(1.0)
    observables["n_frames"] = Observable{Int}(1)
    observables["frame_n"] = map((tf, N)->Int(clamp(round(tf*N), 1, N)),
                            observables["frame_n_float"],
                            observables["n_frames"])
    observables["status_text"] = Observable("Starting worker process...")
    observables["status_progress"] = Observable(-1.0)
    observables["xmin"] = Observable(1)
    observables["xmax"] = Observable(5)
    observables["ymin"] = Observable(1)
    observables["ymax"] = Observable(5)
    for i=1:4
        observables["cmin$i"] = Observable(0.0)
        observables["cmax$i"] = Observable(512.0)
        on((f)->send_request(:rawframe, f), observables["frame_n"])
        on((_)->(@emit updateDisplay(i)), observables["cmin$i"])
        on((_)->(@emit updateDisplay(i)), observables["cmax$i"])
        on((_)->(@emit updateDisplay(i)), observables["xmin"])
        on((_)->(@emit updateDisplay(i)), observables["xmax"])
        on((_)->(@emit updateDisplay(i)), observables["ymin"])
        on((_)->(@emit updateDisplay(i)), observables["ymax"])
    end
    return observables
end

const raw_frame = Observable(zeros(Int16, 5, 5))
const rec_frame = Observable(zeros(Int16, 5, 5))
const init_frame = Observable(zeros(Int16, 5, 5))
const footprints_frame = Observable(zeros(Int16, 5, 5))
function updateDisplays()
    @emit updateDisplay(1)
end
on(updateDisplays, raw_frame)
function run_gui()
    @qmlfunction openvideo
    @qmlfunction checkworkerstatus
    @qmlfunction pingworker
    observables = init_observables()
    QML.loadqml("qt_gui/gui.qml",
        paint_cfunction1 = video_canvas(raw_frame, observables["cmin1"], observables["cmax1"]),
        paint_cfunction2 = video_canvas(rec_frame),
        paint_cfunction3 = video_canvas(init_frame),
        paint_cfunction4 = video_canvas(footprints_frame),
        timer = QTimer(),
        observables = observables
    )
    QML.exec()
    send_request(:rawframe, 1)
end
run_gui()
