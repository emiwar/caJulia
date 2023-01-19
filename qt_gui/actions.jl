
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