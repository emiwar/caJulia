abstract type Background end
include("staticBackground.jl")
include("perVideoBackground.jl")
include("perVideoRank1Background.jl")
include("interactions.jl")

prepupdate!(bg::Background, sol, vl) = nothing
updateseg!(bg::Background, sol, vl) = nothing
update!(bg::Background, sol, vl) = nothing
updateseg!(bg::Background, seg, seg_id, vl) = nothing

prepbackgroundtraceupdate!(bg::Background, sol, vl) = nothing
traceupdate!(bg::Background, sol, vl) = nothing
traceupdateseg!(bg::Background, seg, i, vl) = nothing
