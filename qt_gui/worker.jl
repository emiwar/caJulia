using Distributed
using DataStructures
import Statistics
import CUDA
import cuDNN
import Images
import SparseArrays
import HDF5
import Colors
import Distributions
import LinearAlgebra
import VideoIO
import CSV
import DataFrames
using InteractiveUtils
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
    behaviorvideo::String
end
WorkerState(videoloader) = WorkerState(videoloader, Sol(videoloader), "")
WorkerState() = WorkerState(VideoLoaders.EmptyLoader())

const frame_mapping = CSV.read("../data/frame_mapping.csv", DataFrames.DataFrame);

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
        if frameno < 1 || frameno > VideoLoaders.nframes(videoloader)
            put!(status, ("Invalid frame no $frameno", -1.0))
        else
            put!(status, ("Loading frame $frameno", -1.0))
            if VideoLoaders.location(videoloader) == :device
                frame = Array(VideoLoaders.readframe(videoloader, frameno))#::Matrix{Int16} 
            else
                frame = VideoLoaders.readframe(videoloader, frameno)#::Matrix{Int16} 
            end
            frame = reshape(frame, VideoLoaders.framesize(videoloader))
            put!(responses, (:rawframe, frame))
            put!(status, ("Loaded frame $frameno", 1.0))
        end
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
    elseif requesttype == :behaviorframe
        frameno = Int(data)
        if workerstate.behaviorvideo != ""
            v = VideoIO.openvideo(workerstate.behaviorvideo);
            fps = 60.0#VideoIO.framerate(v)
            b_frameno = frame_mapping.behavior_frame_no[frameno]
            t = float((b_frameno-1) / fps)
            #println("data=$data, frameno=$frameno, fps=$fps, t=$t")
            VideoIO.seek(v, t)
            frame = VideoIO.read(v)
            put!(responses, (:behaviorframe, frame))
        end
    elseif requesttype == :framerange_estimate
        ex_seg_id = VideoLoaders.optimalorder(videoloader)[1]
        range = extrema(VideoLoaders.readseg(videoloader, ex_seg_id))
        put!(responses, (:framerange_estimate, range))
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
        workerstate.videoloader = VideoLoaders.openvideo(filename; nsplits=100)
        workerstate.solution = Sol(workerstate.videoloader)
        put!(responses, (:videoloaded, filename))
        put!(responses, (:nframes, VideoLoaders.nframes(workerstate.videoloader)))
        put!(responses, (:framesize, VideoLoaders.framesize(workerstate.videoloader)))
        GC.gc()
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
    elseif jobtype == :subtractmin
        put!(status, ("Subtracting min", 0.0))
        VideoLoaders.calcmin!(videoloader.source_loader; callback)
        VideoLoaders.clear!(videoloader)
        put!(responses, (:subtractedmin, nothing))
        put!(status, ("Subtracted min", 1.0))
    elseif jobtype == :motioncorrect
        put!(status, ("Motion correcting", 0.0))
        VideoLoaders.fitMotionCorrection_v2!(videoloader.source_loader; callback)
        VideoLoaders.clear!(videoloader)
        put!(responses, (:motioncorrected, nothing))
        put!(status, ("Motion corrected", 1.0))
    elseif jobtype == :clearfilter
        videoloader.source_loader.source_loader.filter .= 1.0
        VideoLoaders.clear!(videoloader)
        put!(responses, (:filterchanged, nothing))
        put!(status, ("Filter cleared", 1.0))
    elseif jobtype == :setbandpassfilter
        low, high = data
        fs = VideoLoaders.framesize(videoloader)
        filter = VideoLoaders.generate_bandpass_filter_no_shift(fs, low, high)
        VideoLoaders.setfilterkernel(videoloader.source_loader.source_loader, filter)
        VideoLoaders.clear!(videoloader)
        put!(responses, (:filterchanged, nothing))
        put!(status, ("Filter set to band pass ($low -- $high)", 1.0))
    elseif jobtype == :saveresult
        filename = data
        put!(status, ("Saving result to $filename", 0.0))
        to_hdf(filename, solution, videoloader)
        put!(status, ("Saved result to $filename", 1.0))
    elseif jobtype == :reset
        workerstate.videoloader = VideoLoaders.EmptyLoader()
        workerstate.solution =  Sol(workerstate.videoloader)
        send_footprints(workerstate, status, responses)
        put!(responses, (:updatedtraces, nothing))
        put!(responses, (:videoloaded, nothing))
        put!(responses, (:nframes, VideoLoaders.nframes(workerstate.videoloader)))
        put!(responses, (:framesize, VideoLoaders.framesize(workerstate.videoloader)))
    elseif jobtype == :loadbehavior
        workerstate.behaviorvideo = data
        put!(responses, (:behaviorloaded, data))
        put!(status, ("Opened $data for behavior.", -1.0))
    elseif jobtype == :loadsolution
        filename = data
        put!(status, ("Loading $filename", -1.0))
        workerstate.solution = from_hdf(filename, workerstate.videoloader)
        put!(status, ("Solution read sucessfully from $filename, plotting data.", -1.0))
        frame = reshape(Array(log10.(solution.I)), solution.frame_size)
        put!(responses, (:initframe, frame))
        send_footprints(workerstate, status, responses)
        put!(responses, (:solutionloaded, filename))
        GC.gc()
        put!(status, ("Loaded $filename", 1.0))
    elseif jobtype == :deletecell
        cell_id = data
        put!(status, ("Deleting cell $cell_id", 0.0))
        deleteCell!(solution, cell_id)
        put!(status, ("Deleted cell $cell_id", 1.0))
        send_footprints(workerstate, status, responses)
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
