using Distributed
using DataStructures
import Statistics
import CUDA
import Images
import SparseArrays
import HDF5
import Colors
import GLMakie
import Distributions
import LinearAlgebra
using ProgressMeter #TODO: should remove this dependency

include("../videoLoaders/videoLoaders.jl"); using .VideoLoaders
include("../solution_struct.jl")
include("../negentropy_img.jl")
include("../viz.jl")
include("../initROIs.jl")
include("../fastHALS.jl")
include("../oasis_opt.jl")
include("../merge_split.jl")
include("../backgrounds/backgrounds.jl")
include("../save_result.jl")

videoloader = VideoLoaders.EmptyLoader()
solution = Sol(videoloader)
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
    elseif requesttype == :reconstructedframe
        frameno = data
        put!(status, ("Reconstructing frame $frameno", -1.0))
        frame_device = reconstruct_frame(solution, frameno, videoloader)
        frame = reshape(Array(frame_device), framesize(videoloader))
        put!(responses, (:reconstructedframe, frame))
        put!(status, ("Reconstructed frame $frameno", 1.0))
    elseif requesttype == :initframe
        frame = reshape(Array(log10.(solution.I)), solution.frame_size)
        put!(responses, (:initframe, frame))
    end
end

function processjob(jobtype, data, status, responses)
    if jobtype == :loadvideo
        filename = data
        put!(status, ("Loading $filename", -1.0))
        global videoloader = VideoLoaders.openvideo(filename)
        global solution = Sol(videoloader)
        put!(responses, (:videoloaded, filename))
        put!(responses, (:nframes, VideoLoaders.nframes(videoloader)))
        put!(responses, (:framesize, VideoLoaders.framesize(videoloader)))
        put!(status, ("Loaded $filename", 1.0))
    elseif jobtype == :calcinitframe
        if VideoLoaders.location(videoloader) == :nowhere
            put!(status, ("Please load video before calculating init frame", -1.0))
        else
            put!(status, ("Calculating init image", 0.0))
            solution.I = negentropy_img_per_video(videoloader)
            frame = reshape(Array(log10.(solution.I)), solution.frame_size)
            put!(responses, (:initframe, frame))
            put!(status, ("Calculated init image", 1.0))
        end
    elseif jobtype == :initfootprints
        put!(status, ("Initiating footprints", 0.0))
        initA!(solution)
        put!(status, ("Initiated footprints.", 1.0))
        put!(status, ("Creating empty traces", 0.0))
        zeroTraces!(solution)
        send_footprints(status, responses)
    elseif jobtype == :initbackgrounds
        put!(status, ("Initiating backgrounds", 0.0))
        initBackgrounds!(videoloader, solution)
        put!(status, ("Initiated backgrounds", 1.0))
    elseif jobtype == :updatetraces
        put!(status, ("Updating traces", 0.0))
        updateTraces!(videoloader, solution, deconvFcn! = oasis_opt!)
        put!(status, ("Updated traces", 1.0))
    elseif jobtype == :updatefootprints
        put!(status, ("Updating footprints", 0.0))
        updateROIs!(videoloader, solution)
        put!(status, ("Updated footprints", 1.0))
        send_footprints(status, responses)
    elseif jobtype == :mergecells
        put!(status, ("Merging cells", 0.0))
        merge!(solution, thres=.6)
        put!(status, ("Merged cells", 1.0))
        send_footprints(status, responses)
    end
end


function send_footprints(status, responses)
    put!(status, ("Drawing plot of footprints", 0.0))
    put!(responses, (:footprints, roiImg(solution)))
    put!(status, ("Drawed plot of footprints", 1.0))
end

#println("Code loaded at process $(myid())")
