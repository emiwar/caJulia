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
            put!(status, ("Importing CUDA", 0.1)); Base.eval(Main, :(import CUDA))
            put!(status, ("Importing Images", 0.2)); Base.eval(Main, :(import Images))
            put!(status, ("Importing HDF5", 0.3)); Base.eval(Main, :(import HDF5))
            put!(status, ("Importing Statistics", 0.35)); Base.eval(Main, :(import Statistics))
            put!(status, ("Importing Colors", 0.4)); Base.eval(Main, :(import Colors))
            put!(status, ("Loading code", 0.5))
            include("qt_gui/worker.jl")
            put!(status, ("Starting job queue", 0.7))
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

function restartworker(connection)
    jobs = connection.jobs
    status = connection.status
    responses = connection.responses
    proc_id = connection.proc_id
    put!(connection.status, ("Restarting worker process", -1.0))

    errormonitor(@async begin
        remotecall_fetch(proc_id, status) do status
            put!(status, ("Reinitializing worker", 0.0))
            #include("qt_gui/worker.jl")
            
        end
        remotecall_fetch((j,s,r)->work(j,s,r), proc_id, jobs, status, responses)
    end)
end