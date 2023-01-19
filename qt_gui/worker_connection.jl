using Distributed

mutable struct WorkerConnection
    proc_id::Int
    jobs::RemoteChannel{Channel{Tuple{Bool, Symbol, Any}}}
    status::RemoteChannel{Channel{Tuple{String, Float64}}}
    responses::RemoteChannel{Channel{Tuple{Symbol, Any}}}
    function WorkerConnection()
        jobs = RemoteChannel(()->Channel{Tuple{Bool, Symbol, Any}}(128))
        status = RemoteChannel(()->Channel{Tuple{String, Float64}}(128))
        responses = RemoteChannel(()->Channel{Tuple{Symbol, Any}}(128))
        proc_id = startworker(jobs, status, responses)
        connection = new(proc_id, jobs, status, responses)
        finalizer((conn)->(@async rmprocs(conn.proc_id)), connection)
    end
end

function startworker(jobs, status, responses)
    put!(status, ("Starting worker process", -1.0))
    proc_id = addprocs(1, exeflags="--project")[1]
    errormonitor(@async begin
        remotecall_fetch(proc_id, status) do status
            put!(status, ("Initializing worker", 0.0))
            include("qt_gui/worker.jl")
        end
        remotecall_fetch((j,s,r)->work(j,s,r), proc_id, jobs, status, responses)
    end)
    return proc_id
end

function send_request(connection::WorkerConnection, type::Symbol, data=nothing)
    put!(connection.jobs, (false, type, data))
end
function submit_job(connection::WorkerConnection, type::Symbol, data=nothing)
    put!(connection.jobs, (true, type, data))
end


