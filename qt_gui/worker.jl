using Distributed
using DataStructures
include("../videoLoaders/videoLoaders.jl")

videoloader = VideoLoaders.EmptyLoader()
const jobqueue = Queue{Tuple{Symbol, Any}}()

function listenforjobs(jobs, status, responses)
    while isready(jobs)
        toqueue, requesttype, data = take!(jobs)
        if toqueue
            enqueue!(jobqueue, (requesttype, data))
        else
            processrequest(requesttype, data, status, responses)
        end
    end
end

function processrequest(requesttype, data, status, responses)
    if requesttype == :ping
        put!(responses, (:ping, "Hello"))
        put!(status, ("Worker replied to ping", -1.0))
    elseif requesttype == :killworker
        put!(responses, (:killingworker, nothing))
        exit()
    elseif requesttype == :rawframe
        frameno = data
        put!(status, ("Loading frame $frameno", -1.0))
        if VideoLoaders.location(videoloader) == :device
            frame = VideoLoaders.readframe(videoloader.source_loader, frameno)::Matrix{Int16} 
        else
            frame = VideoLoaders.readframe(videoloader, frameno)::Matrix{Int16} 
        end
        put!(responses, (:rawframe, frame))
        put!(status, ("Loaded frame $frameno", 1.0))
    end
end

function processjob(jobtype, data, status, responses)
    if jobtype == :loadvideo
        filename = data
        put!(status, ("Loading $filename", -1.0))
        global videoloader = VideoLoaders.openvideo(filename)
        put!(responses, (:videoloaded, filename))
        put!(responses, (:nframes, VideoLoaders.nframes(videoloader)))
        put!(responses, (:framesize, VideoLoaders.framesize(videoloader)))
        put!(status, ("Loaded $filename", 1.0))
    end
end

#println("Code loaded at process $(myid())")
