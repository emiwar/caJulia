abstract type Background end

prepupdate!(bg::Background, sol, vl) = nothing
updateseg!(bg::Background, sol, vl) = nothing
update!(bg::Background, sol, vl) = nothing
updateseg!(bg::Background, seg, seg_id, vl) = nothing

prepbackgroundtraceupdate!(bg::Background, sol, vl) = nothing
traceupdate!(bg::Background, sol, vl) = nothing
traceupdateseg!(bg::Background, seg, i, vl) = nothing

mutable struct StaticBackground <: Background
    m::CUDA.CuVector{Float32}
    b::CUDA.CuVector{Float32}
    bA::CUDA.CuVector{Float32}
    bC::CUDA.CuMatrix{Float32}
end

function StaticBackground(framesize::Int64)
    StaticBackground(CUDA.zeros(framesize), CUDA.zeros(framesize),
                     CUDA.zeros(0), CUDA.zeros(0, 0))
end

function StaticBackground(vl::VideoLoader)
    StaticBackground(prod(framesize(vl)))
end

function updatepixelcorrections!(bg::StaticBackground, sol, vl)
    bg.bC = bg.b * sum(sol.C; dims=1)
end

function pixelcorrections(bg::StaticBackground, j)
    view(bg.bC, :, j)
end

function updatetracecorrections!(bg::StaticBackground, sol, vl)
    bg.bA = sol.A * bg.b
end

function tracecorrection(bg::StaticBackground, j)
    @CUDA.allowscalar bg.bA[j]
end

function prepinit!(bg::StaticBackground, sol, vl)
    bg.m .= 0.0f0
end

function initseg!(bg::StaticBackground, seg, seg_id, vl)
    bg.m .+= view(sum(seg; dims=2), :) ./ nframes(vl)
    bg.b .= bg.m #Stupid, should fix...
end

function update!(bg::StaticBackground, sol, vl)
    bg.b .= bg.m .- view(sol.A'*sum(sol.C; dims=1)' ./ nframes(vl), :)
end

function reconstruct_frame(bg::StaticBackground, frame_id)
    return bg.b
end