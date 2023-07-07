ENV["QSG_RENDER_LOOP"] = "basic"
ENV["QT_QUICK_CONTROLS_STYLE"] = "Material"
ENV["QT_QUICK_CONTROLS_MATERIAL_THEME"] = "Dark"
ENV["QT_QUICK_CONTROLS_MATERIAL_VARIANT"] = "Dense"
#ENV["QT_QUICK_CONTROLS_UNIVERSAL_VARIANT"] = "Dark"
include("worker_connection.jl")
conn = WorkerConnection()
import CxxWrap
using Images
using Colors
using ColorSchemes
using Observables
using QML
using Qt5QuickControls_jll
using Qt5QuickControls2_jll
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
    observables["frame_n_float"] = Observable(0.0)
    observables["n_frames"] = Observable{Int}(1)
    observables["frame_n"] = map((tf, N)->Int(clamp(round(tf*N), 1, N)),
                            observables["frame_n_float"],
                            observables["n_frames"])
    observables["status_text"] = Observable("Starting worker process...")
    observables["status_progress"] = Observable(-1.0)
    observables["xmin"] = Observable(1)
    observables["xmax"] = Observable(100)
    observables["ymin"] = Observable(1)
    observables["ymax"] = Observable(100)
    on(observables["frame_n"]) do f
        send_request(conn, :rawframe, f)
        send_request(conn, :reconstructedframe, f)
        send_request(conn, :behaviorframe, f)
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
    observables["tmin"] = Observable(0)
    observables["tmax"] = Observable(1)
    observables["crangemin"] = Observable(-256.0)
    observables["crangemax"] = Observable(256.0)
    on((_)->(@emit updateDisplay(6)), observables["traceS"])
    on((_)->(@emit updateDisplay(6)), observables["traceC"])
    on((_)->(@emit updateDisplay(6)), observables["traceR"])
    on((_)->(@emit updateDisplay(6)), observables["frame_n_float"])
    on((_)->(@emit updateDisplay(6)), observables["tmin"])
    on((_)->(@emit updateDisplay(6)), observables["tmax"])
    observables["selected_cell"] = Observable(0)
    return observables
end

const raw_frame = Observable(zeros(Float32, 100, 100));
const rec_frame = Observable(zeros(Float32, 100, 100));
const init_frame = Observable(zeros(Float32, 100, 100) .* NaN);
const footprints_frame = Observable(zeros(Colors.ARGB32, 100, 100));
const footprints_peaks = Observable(zeros(Int, 100, 100));
const behavior_frame = Observable(ones(Colors.ARGB32, 100, 100));
#function updateDisplays()
#    @emit updateDisplay(1)
#end
on((_)->(@emit updateDisplay(1)), raw_frame);
on((_)->(@emit updateDisplay(2)), rec_frame);
on((_)->(@emit updateDisplay(3)), init_frame);
on((_)->(@emit updateDisplay(4)), footprints_frame);
on((_)->(@emit updateDisplay(5)), behavior_frame);


function run_gui()
    @qmlfunction openvideo
    @qmlfunction openbehaviorvideo
    @qmlfunction saveresult
    @qmlfunction checkworkerstatus
    @qmlfunction pingworker
    @qmlfunction resetworker
    @qmlfunction calcinitframe
    @qmlfunction initfootprints
    @qmlfunction initbackgrounds
    @qmlfunction updatetraces
    @qmlfunction updatefootprints
    @qmlfunction mergecells
    @qmlfunction subtractmin
    @qmlfunction footprintclick
    @qmlfunction zoomscroll
    @qmlfunction pandrag
    @qmlfunction zoomscrolltrace
    @qmlfunction pandragtrace
    @qmlfunction motioncorrect
    observables = init_observables()
    QML.loadqml("qt_gui/gui.qml",
        paint_cfunction1 = video_canvas(raw_frame, observables["cmin1"], observables["cmax1"], observables["xmin"], observables["xmax"], observables["ymin"], observables["ymax"]),
        paint_cfunction2 = video_canvas(rec_frame, observables["cmin2"], observables["cmax2"], observables["xmin"], observables["xmax"], observables["ymin"], observables["ymax"]),
        paint_cfunction3 = video_canvas(init_frame, observables["cmin3"], observables["cmax3"], observables["xmin"], observables["xmax"], observables["ymin"], observables["ymax"], :inferno),
        paint_cfunction4 = video_canvas_raw(footprints_frame, observables["xmin"], observables["xmax"], observables["ymin"], observables["ymax"]),
        paint_cfunction5 = video_canvas_raw(behavior_frame),
        timer = QTimer(),
        observables = observables
    )
    on(observables["xmin"]) do xmin
        println("Frame $(observables["frame_n"][])")
        println("X: [$(observables["xmin"][]), $(observables["xmax"][])]")
        println("Y: [$(observables["ymin"][]), $(observables["ymax"][])]")
        println("")
    end
    send_request(conn, :rawframe, 1)
    QML.exec()
    return observables
end
obs = run_gui();
send_request(conn, :refreshvideo)





#ex_video = "/mnt/dmclab/Vasiliki/Striosomes experiments/Calcium imaging in oprm1/1st batch oct 2021/Inscopix/oprm1_learn_day2/recording_20211017_154731.hdf5";
