ENV["QSG_RENDER_LOOP"] = "basic"
import CxxWrap
using Images
using Colors
using ColorSchemes
using Observables
using QML
using Qt5QuickControls_jll
using Qt5QuickControls2_jll
include("worker_connection.jl")
include("videoCanvas.jl")
include("handle_responses.jl")
include("actions.jl")


function checkworkerstatus(observables)
    yield() #Appearently needed to refresh the channels
    while isready(conn.status)
        st, sp = take!(conn.status)
        observables["status_text"][] = st
        observables["status_progress"][] = sp
    end
    while isready(conn.responses)
        response_type, data = take!(conn.responses)
        handle_response(response_type, data, observables)
    end
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
    on(observables["frame_n"]) do f
        send_request(conn, :rawframe, f)
        send_request(conn, :reconstructedframe, f)
    end
    for i=1:2
        observables["cmin$i"] = Observable(0.0)
        observables["cmax$i"] = Observable(512.0)
    end
    observables["cmin3"] = Observable(-4.0)
    observables["cmax3"] = Observable(2.0)
    for i=1:3
        on((_)->(@emit updateDisplay(i)), observables["cmin$i"])
        on((_)->(@emit updateDisplay(i)), observables["cmax$i"])
    end
    for i=1:4
        on((_)->(@emit updateDisplay(i)), observables["xmin"])
        on((_)->(@emit updateDisplay(i)), observables["xmax"])
        on((_)->(@emit updateDisplay(i)), observables["ymin"])
        on((_)->(@emit updateDisplay(i)), observables["ymax"])
    end

    observables["traceS"] = Observable(zeros(Float32, 10))
    observables["traceC"] = Observable(zeros(Float32, 10))
    observables["traceR"] = Observable(zeros(Float32, 10))
    observables["traceCol"] = Observable("#000000")
    on((_)->(@emit updateDisplay(5)), observables["traceS"])
    on((_)->(@emit updateDisplay(5)), observables["traceC"])
    on((_)->(@emit updateDisplay(5)), observables["traceR"])
    return observables
end

const raw_frame = Observable(zeros(Int16, 100, 100));
const rec_frame = Observable(zeros(Float32, 100, 100));
const init_frame = Observable(zeros(Float32, 100, 100));
const footprints_frame = Observable(ones(Colors.ARGB32, 100, 100));
const footprints_peaks = Observable(zeros(Int, 100, 100));
#function updateDisplays()
#    @emit updateDisplay(1)
#end
on((_)->(@emit updateDisplay(1)), raw_frame);
on((_)->(@emit updateDisplay(2)), rec_frame);
on((_)->(@emit updateDisplay(3)), init_frame);
on((_)->(@emit updateDisplay(4)), footprints_frame);

conn = WorkerConnection()
function run_gui()
    @qmlfunction openvideo
    @qmlfunction checkworkerstatus
    @qmlfunction pingworker
    @qmlfunction calcinitframe
    @qmlfunction initfootprints
    @qmlfunction initbackgrounds
    @qmlfunction updatetraces
    @qmlfunction updatefootprints
    @qmlfunction mergecells
    @qmlfunction footprintclick
    observables = init_observables()
    QML.loadqml("qt_gui/gui.qml",
        paint_cfunction1 = video_canvas(raw_frame, observables["cmin1"], observables["cmax1"]),
        paint_cfunction2 = video_canvas(rec_frame, observables["cmin2"], observables["cmax2"]),
        paint_cfunction3 = video_canvas(init_frame, observables["cmin3"], observables["cmax3"], :inferno),
        paint_cfunction4 = video_canvas_raw(footprints_frame),
        timer = QTimer(),
        observables = observables
    )
    send_request(conn, :rawframe, 1)
    QML.exec()
    return observables
end
obs = run_gui();
