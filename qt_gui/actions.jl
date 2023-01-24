
function pingworker()
    send_request(conn, :ping)
end

function openvideo(Qfilename)
    #TODO: must be possible to parse this safer
    if Sys.iswindows()
        filename = String(QString(Qfilename))[9:end]
    else
        filename = String(QString(Qfilename))[8:end]
    end
    submit_job(conn, :loadvideo, filename)
end

function saveresult(Qfilename)
    #TODO: must be possible to parse this safer
    if Sys.iswindows()
        filename = String(QString(Qfilename))[9:end]
    else
        filename = String(QString(Qfilename))[8:end]
    end
    submit_job(conn, :saveresult, filename)
end

function calcinitframe()
    submit_job(conn, :calcinitframe)
end

function initfootprints()
    submit_job(conn, :initfootprints)
end

function initbackgrounds()
    submit_job(conn, :initbackgrounds)
end

function updatetraces()
    submit_job(conn, :updatetraces)
end

function updatefootprints()
    submit_job(conn, :updatefootprints)
end

function mergecells()
    submit_job(conn, :mergecells)
end

function footprintclick(rel_x, rel_y, observables)
    w, h = size(footprints_peaks[])
    xmin = observables["xmin"][]
    xmax = observables["xmax"][]
    ymin = observables["ymin"][]
    ymax = observables["ymax"][]
    x = clamp(Int(round(xmin + rel_x*(xmax - xmin))), 1, w)
    y = clamp(Int(round(ymin + rel_y*(ymax - ymin))), 1, h)
    cell_id = footprints_peaks[][x, y]
    println("Clicked on cell #$cell_id at ($x, $y)")
    observables["traceS"][] = zero(observables["traceS"][])
    observables["traceC"][] = zero(observables["traceC"][])
    observables["traceR"][] = zero(observables["traceR"][])
    observables["traceCol"][] = "#000000"
    if cell_id > 0
        send_request(conn, :trace, cell_id)
    end
end

function zoomscroll(rel_x, rel_y, delta, observables)
    w, h = size(raw_frame[])
    xmin = observables["xmin"][]
    xmax = observables["xmax"][]
    ymin = observables["ymin"][]
    ymax = observables["ymax"][]
    xmin += rel_x*delta * (xmax - xmin) / 100
    xmax -= (1-rel_x)*delta * (xmax - xmin) / 100
    ymin += rel_y*delta * (ymax - ymin) / 100
    ymax -= (1-rel_y)*delta * (ymax - ymin) / 100
    xmin = clamp(Int(round(xmin)), 1, w)
    xmax = clamp(Int(round(xmax)), 1, w)
    ymin = clamp(Int(round(ymin)), 1, h)
    ymax = clamp(Int(round(ymin + (xmax-xmin)*(h/w))), 1, h)
    if xmax-xmin > 20 && ymax-ymin > 20
        observables["xmin"][] = xmin
        observables["xmax"][] = xmax
        observables["ymin"][] = ymin
        observables["ymax"][] = ymax
    end
end

function pandrag(rel_x, rel_y, observables)
    w, h = size(raw_frame[])
    xmin = observables["xmin"][]
    xmax = observables["xmax"][]
    ymin = observables["ymin"][]
    ymax = observables["ymax"][]
    xmin -= rel_x * (xmax - xmin)
    xmax -= rel_x * (xmax - xmin)
    ymin -= rel_y * (ymax - ymin)
    ymax -= rel_y * (ymax - ymin)
    xmin = Int(round(xmin))
    xmax = Int(round(xmax))
    ymin = Int(round(ymin))
    ymax = Int(round(ymin + (xmax-xmin)*(h/w)))
    if xmin >= 1 && ymin >= 1 && xmin < xmax && ymin < ymax && xmax <=w && ymax <=h
        observables["xmin"][] = xmin
        observables["xmax"][] = xmax
        observables["ymin"][] = ymin
        observables["ymax"][] = ymax
    end
end
