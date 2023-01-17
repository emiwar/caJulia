
function pingworker()
    send_request(:ping)
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

function calcinitframe()
    submit_job(:calcinitframe)
end

function initfootprints()
    submit_job(:initfootprints)
end

function initbackgrounds()
    submit_job(:initbackgrounds)
end

function updatetraces()
    submit_job(:updatetraces)
end

function updatefootprints()
    submit_job(:updatefootprints)
end

function mergecells()
    submit_job(:mergecells)
end