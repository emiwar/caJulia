using Distributed
using DataStructures
import Statistics
import CUDA
import Images
import SparseArrays
import HDF5
import Colors
import Distributions
import LinearAlgebra
using ProgressMeter #TODO: should remove this dependency

println("Imports done")

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

println("Includes done")

mutable struct WorkerState
    videoloader::VideoLoaders.VideoLoader
    solution::Sol
end
WorkerState(videoloader) = WorkerState(videoloader, Sol(videoloader))
WorkerState() = WorkerState(VideoLoaders.EmptyLoader())

function work(jobs, status, responses)
    jobqueue = Queue{Tuple{Symbol, Any}}()
    workerstate = WorkerState()
    put!(status, ("Worker ready", -1.0))
    put!(responses, (:nframes, VideoLoaders.nframes(workerstate.videoloader)))
    put!(responses, (:framesize, VideoLoaders.framesize(workerstate.videoloader)))
    while true
        listenforjobs(jobs, status, responses, jobqueue, workerstate)
        while !isempty(jobqueue)
            jobtype, jobdata = dequeue!(jobqueue)
            processjob(jobtype, jobdata, jobs, status, responses, jobqueue, workerstate)
            listenforjobs(jobs, status, responses, jobqueue, workerstate)
        end
        wait(jobs)
    end
end

function listenforjobs(jobs, status, responses, jobqueue, workerstate)
    while isready(jobs)
        toqueue, requesttype, data = take!(jobs)
        if toqueue
            enqueue!(jobqueue, (requesttype, data))
        else
            processrequest(requesttype, data, status, responses, workerstate)
        end
    end
end

function processrequest(requesttype, data, status, responses, workerstate)
    videoloader = workerstate.videoloader
    solution = workerstate.solution
    if requesttype == :ping
        put!(responses, (:ping, "Hello"))
        put!(status, ("Worker replied to ping", -1.0))
    elseif requesttype == :killworker
        put!(responses, (:killingworker, nothing))
        exit()
    elseif requesttype == :refreshvideo
        put!(responses, (:nframes, VideoLoaders.nframes(videoloader)))
        put!(responses, (:framesize, VideoLoaders.framesize(videoloader)))
    elseif requesttype == :rawframe
        frameno = Int(data)
        put!(status, ("Loading frame $frameno", -1.0))
        if VideoLoaders.location(videoloader) == :device
            frame = VideoLoaders.readframe(videoloader.source_loader, frameno)#::Matrix{Int16} 
        else
            frame = VideoLoaders.readframe(videoloader, frameno)#::Matrix{Int16} 
        end
        put!(responses, (:rawframe, frame))
        put!(status, ("Loaded frame $frameno", 1.0))
    elseif requesttype == :reconstructedframe
        frameno = Int(data)
        if frameno >= 1 && frameno <= nframes(videoloader)
            put!(status, ("Reconstructing frame $frameno", -1.0))
            frame_device = reconstruct_frame(solution, frameno, videoloader)
            frame = reshape(Array(frame_device), framesize(videoloader))
            put!(responses, (:reconstructedframe, frame))
            put!(status, ("Reconstructed frame $frameno", 1.0))
        else
            put!(status, ("Invalid frame no $frameno", -1.0))
        end
    elseif requesttype == :initframe
        frame = reshape(Array(log10.(solution.I)), solution.frame_size)
        put!(responses, (:initframe, frame))
    elseif requesttype == :footprints
        send_footprints(workerstate, status, responses)
    elseif requesttype == :trace
        cellid = Int(data)
        S = workerstate.solution.S
        C = workerstate.solution.C
        R = workerstate.solution.R
        if cellid > 0 && cellid <= size(C, 2)
            traceS = Array(view(S, :, cellid))
            traceC = Array(view(C, :, cellid))
            traceR = Array(view(R, :, cellid))
            col = workerstate.solution.colors[cellid]
            put!(responses, (:trace, (traceS, traceC, traceR, col)))
        else
            put!(status, ("Invalid cell $cellid", -1))
        end
    end
end

function processjob(jobtype, data, jobs, status, responses, jobqueue, workerstate)
    videoloader = workerstate.videoloader
    solution = workerstate.solution
    function callback(label, i, N)
        put!(status, (label, i/N))
        listenforjobs(jobs, status, responses, jobqueue, workerstate)
    end
    if jobtype == :loadvideo
        filename = data
        put!(status, ("Loading $filename", -1.0))
        workerstate.videoloader = VideoLoaders.openvideo(filename; nsplits=50)
        workerstate.solution = Sol(workerstate.videoloader)
        put!(responses, (:videoloaded, filename))
        put!(responses, (:nframes, VideoLoaders.nframes(workerstate.videoloader)))
        put!(responses, (:framesize, VideoLoaders.framesize(workerstate.videoloader)))
        put!(status, ("Loaded $filename", 1.0))
    elseif jobtype == :calcinitframe
        if VideoLoaders.location(videoloader) == :nowhere
            put!(status, ("Please load video before calculating init frame", -1.0))
        else
            put!(status, ("Calculating init image", 0.0))
            solution.I = negentropy_img_per_video(videoloader; callback)
            frame = reshape(Array(log10.(solution.I)), solution.frame_size)
            put!(responses, (:initframe, frame))
            put!(status, ("Calculated init image", 1.0))
        end
    elseif jobtype == :initfootprints
        put!(status, ("Initializing footprints", 0.0))
        initA!(solution; callback)
        put!(status, ("Initialized footprints.", 1.0))
        put!(status, ("Creating empty traces", 0.0))
        zeroTraces!(solution)
        send_footprints(workerstate, status, responses)
    elseif jobtype == :initbackgrounds
        put!(status, ("Initializing backgrounds", 0.0))
        initBackgrounds!(videoloader, solution; callback)
        put!(responses, (:initbackgrounds, nothing))
        put!(status, ("Initialized backgrounds", 1.0))
    elseif jobtype == :updatetraces
        put!(status, ("Updating traces", 0.0))
        updateTraces!(videoloader, solution; deconvFcn! = oasis_opt!, callback)
        put!(responses, (:updatedtraces, nothing))
        put!(status, ("Updated traces", 1.0))
    elseif jobtype == :updatefootprints
        put!(status, ("Updating footprints", 0.0))
        updateROIs!(videoloader, solution; callback)
        put!(status, ("Updated footprints", 1.0))
        send_footprints(workerstate, status, responses)
    elseif jobtype == :mergecells
        put!(status, ("Merging cells", 0.0))
        merge!(solution, thres=.6; callback)
        put!(status, ("Merged cells", 1.0))
        send_footprints(workerstate, status, responses)
    elseif jobtype == :saveresult
        filename = data
        put!(status, ("Saving result to $filename", 0.0))
        to_hdf(filename, solution, videoloader)
        put!(status, ("Saved result to $filename", 1.0))
    end
end

function send_footprints(workerstate, status, responses)
    put!(status, ("Drawing plot of footprints", 0.0))
    img, peaks = roiImg(workerstate.solution)
    amap = strongestAMap(workerstate.solution)
    put!(responses, (:footprints, (img, amap)))
    put!(status, ("Drawed plot of footprints", 1.0))
end

nothing

#println("Code loaded at process $(myid())")
