#ENV["QSG_RENDER_LOOP"] = "basic"
import CxxWrap
using Images
using Colors
using Observables
using QML
using Qt5QuickControls_jll
using Qt5QuickControls2_jll
include("videoCanvas.jl")
include("../videoLoaders/videoLoaders.jl")

const current_time_frac = Observable(0.0)
const video_loader = Observable{VideoLoaders.VideoLoader}(VideoLoaders.EmptyLoader())
const current_frame_n = map(video_loader, current_time_frac) do vl, ctf
    N::Int = VideoLoaders.nframes(vl)
    cf = round(ctf*N)
    Int(clamp(cf, 1, N))
end
const current_frame = map(video_loader, current_frame_n) do vl, cfn
    if VideoLoaders.location(vl) == :device
        return VideoLoaders.readframe(vl.source_loader, cfn)::Matrix{Int16}
    else
        return VideoLoaders.readframe(vl, cfn)::Matrix{Int16}
    end
end
function openvideo(Qfilename)
    #TODO: must be possible to parse this safer
    filename = String(QString(Qfilename))[8:end]
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
    QML.loadqml("qt_gui/gui.qml",
        paint_cfunction1 = video_canvas(current_frame),
        paint_cfunction2 = video_canvas(current_frame),
        paint_cfunction3 = video_canvas(current_frame),
        paint_cfunction4 = video_canvas(current_frame),
        observables = JuliaPropertyMap("current_time" => current_time_frac)
    )
    QML.exec()
end
run_gui()