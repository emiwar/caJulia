import OrderedCollections
function to_hdf(filename::String, sol::Sol, vl::VideoLoader; videofilename=nothing)
    HDF5.h5open(filename, "w") do fid
        fid["/footprints"] = Array(sol.A)
        fid["/initImage"] = reshape(Array(sol.I), framesize(vl))
        if VideoLoaders.multivideo(vl)
            #TODO: this was written to enable exporting to python,
            #not reloading into CaJulia. Should extend!
            for video_i = 1:nvideos(vl)
                seg_i = first(vl.source_loader.source_loader.segs_per_video[video_i])
                key = match(r"animal[0-9].*day[0-9]+",
                            VideoLoaders.filename(vl, seg_i)).match
                fr = framerange_video(vl, video_i)
                fid["/traces/$key/raw"] = Array(sol.R[fr, :])
                fid["/traces/$key/fitted"] = Array(sol.C[fr, :])
                fid["/traces/$key/deconvolved"] = Array(sol.S[fr, :])
            end
        else
            fid["/traces/raw"] = Array(sol.R)
            fid["/traces/fitted"] = Array(sol.C)
            fid["/traces/deconvolved"] = Array(sol.S)
        end
        fid["/params/gammas"] = sol.gammas
        fid["/params/lambdas"] = sol.lambdas
        fid["/meta/displayColors"] = Colors.hex.(sol.colors, :RRGGBB)

        for (i, background) in enumerate(sol.backgrounds)
            bg_type = typeof(background)
            fid["/backgrounds/bg$i/type"] = string(bg_type)
            for field in fieldnames(bg_type)
                value = getfield(background, field)
                if value isa CUDA.CuArray
                    value = Array(value)
                end
                fid["/backgrounds/bg$i/$field"] = value
            end
        end

        fid["/meta/CaJulia/git"] = readchomp(`git rev-parse --short HEAD`)
        if videofilename !== nothing
            fid["/meta/videoFilename"] = string(videoFilename)
        elseif !VideoLoaders.multivideo(vl)
            fid["/meta/videoFilename"] = VideoLoaders.filename(vl, 1)
        end
        fid["/meta/nsegs"] = nsegs(vl)
    end
    nothing
end

function from_hdf(filename::String)
    HDF5.h5open(filename, "r") do fid
        A = SparseArrays.sparse(Array(fid["/footprints"])) |> CUDA.cu
        R = Array(fid["/traces/raw"]) |> CUDA.cu
        C = Array(fid["/traces/fitted"]) |> CUDA.cu
        S = Array(fid["/traces/deconvolved"]) |> CUDA.cu
        I = Array(fid["/initImage"]) |> CUDA.cu
        frame_size = size(I)
        gammas = fid["/params/gammas"] |> Array
        lambdas = fid["/params/lambdas"] |> Array
        hex_colors = fid["/meta/displayColors"]
        colors = [Colors.parse(Colors.RGB, "#$c") for c in Array(hex_colors)]
        back_vector = Background[]
        all_background_types = Dict(string(bt)=>bt for bt in subtypes(Background))
        for bg_group in keys(fid["/backgrounds/"])
            bg_type = all_background_types[read(fid["/backgrounds/$bg_group/type"])]
            field_dict = Dict{Symbol, Any}()
            for field_key in keys(fid["/backgrounds/$bg_group/"])
                if field_key != "type"
                    field_dict[Symbol(field_key)] = CUDA.cu(Array(fid["/backgrounds/$bg_group/$field_key"]))
                end
            end
            push!(back_vector, bg_type((field_dict[f] for f in fieldnames(bg_type))...))
        end
        Sol(A, R, C, S, I, Tuple(back_vector), gammas, lambdas, frame_size, colors)
    end
end