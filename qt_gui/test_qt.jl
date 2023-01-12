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

const current_time_frac = Observable(0.0)
const current_frame_n = map(current_time_frac) do ctf
    N::Int = 1#VideoLoaders.nframes(vl)
    cf = round(ctf*N)
    Int(clamp(cf, 1, N))
end
const current_frame = Observable(zeros(Int16, 5, 5))
function checkworkerstatus()
    while isready(status)
        status_text, status_progress = take!(status)
        println("$status_text ($status_progress*100)%")
    end
    while isready(responses)
        response_type, data = take!(responses)
        println("Got response $response_type")
    end
end

function pingworker()
    put!(jobs, (false, :ping, nothing))
end
function openvideo(Qfilename)
    #TODO: must be possible to parse this safer
    println(typeof(Qfilename))
    return
    filename = String(QString(Qfilename))[9:end]
    println("Loading $filename")
    println(isfile(filename))
    try
        video_loader[] = VideoLoaders.openvideo(filename)
    catch e
        display(e)
    end
    println("Loaded $filename")
end
function run_gui()
    @qmlfunction openvideo
    @qmlfunction checkworkerstatus
    @qmlfunction pingworker
    QML.loadqml("qt_gui/gui.qml",
        paint_cfunction1 = video_canvas(current_frame),
        paint_cfunction2 = video_canvas(current_frame),
        paint_cfunction3 = video_canvas(current_frame),
        paint_cfunction4 = video_canvas(current_frame),
        timer = QTimer(), #TODO: start timer
        observables = JuliaPropertyMap("current_time" => current_time_frac)
    )
    QML.exec()
end
run_gui()

