using Distributed
using DataStructures

const worker_proc_id = addprocs(1, exeflags="--project")[1]

const jobs = RemoteChannel(()->Channel{Tuple{Bool, Symbol, Any}}(128))

const status = RemoteChannel(()->Channel{Tuple{String, Float64}}(128))
const responses = RemoteChannel(()->Channel{Tuple{Symbol, Any}}(128))

@everywhere function work(jobs, status, responses)
    put!(status, ("Worker ready", -1.0))
    while true
        listenforjobs(jobs, status, responses)
        while !isempty(jobqueue)
            jobtype, jobdata = dequeue!(jobqueue)
            processjob(jobtype, jobdata, status, responses)
            listenforjobs(jobs, status, responses)
        end
        wait(jobs)
    end
end

function send_request(type::Symbol, data=nothing)
    put!(jobs, (false, type, data))
end
function submit_job(type::Symbol, data=nothing)
    put!(jobs, (true, type, data))
end

loadcodefuture = remotecall(()->include("qt_gui/worker.jl"), worker_proc_id)
remote_do(work, worker_proc_id, jobs, status, responses)

