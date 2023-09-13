import OrderedCollections
function to_hdf(filename::String, sol::Sol, vl::VideoLoader; videofilename=nothing)
    HDF5.h5open(filename, "w") do fid
        fid["/footprints"] = Array(sol.A)
        fid["/initImage"] = reshape(Array(sol.I), framesize(vl))
        if VideoLoaders.multivideo(vl)
            #TODO: this was written to enable exporting to python,
            #not reloading into CaJulia. Should extend!
            for video_i = 1:nvideos(vl)
                #seg_i = first(vl.source_loader.source_loader.segs_per_video[video_i])
                #key = match(r"animal[0-9].*day[0-9]+",
                #            VideoLoaders.filename(vl, seg_i)).match
                key = outputlabel(vl, video_i)
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
        fid["/meta/displayColors"] = Vector{String}(Colors.hex.(sol.colors, :RRGGBB))

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

        VideoLoaders.savetohdf(vl, fid)

        fid["/meta/CaJulia/git"] = readchomp(`git rev-parse --short HEAD`)
        #if videofilename !== nothing
        #    fid["/meta/videoFilename"] = string(videoFilename)
        #elseif !VideoLoaders.multivideo(vl)
        #    fid["/meta/videoFilename"] = VideoLoaders.filename(vl, 1)
        #end
    end
    nothing
end

function from_hdf(filename::String, vl::VideoLoader)
    HDF5.h5open(filename, "r") do fid
        A = SparseArrays.sparse(Array(fid["/footprints"])) |> CUDA.cu
        I = Array(fid["/initImage"]) |> CUDA.cu
        ncells = size(A, 1)
        nf = nframes(vl)
        R = CUDA.zeros(Float32, nf, ncells)
        C = CUDA.zeros(Float32, nf, ncells)
        S = CUDA.zeros(Float32, nf, ncells)
        if VideoLoaders.multivideo(vl)
            for video_i = 1:nvideos(vl)
                key = outputlabel(vl, video_i)
                fr = framerange_video(vl, video_i)
                R[fr, :] .= fid["/traces/$key/raw"] |> Array |> CUDA.cu
                C[fr, :] .= fid["/traces/$key/fitted"] |> Array |> CUDA.cu
                S[fr, :] .= fid["/traces/$key/deconvolved"] |> Array |> CUDA.cu
            end
        else
            R .= fid["/traces/$key/raw"]
            C .= fid["/traces/$key/fitted"]
            S .= fid["/traces/$key/deconvolved"]
        end
        gammas = fid["/params/gammas"] |> Array
        lambdas = fid["/params/lambdas"] |> Array
        hex_colors = fid["/meta/displayColors"]
        colors = Colors.RGB{Colors.N0f8}[Colors.parse(Colors.RGB, "#$c") for c in Array(hex_colors)]
        back_vector = Background[]
        all_background_types = Dict(string(bt)=>bt for bt in InteractiveUtils.subtypes(Background))
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
        VideoLoaders.loadfromhdf(vl, fid)
        Sol(A, R, C, S, I, Tuple(back_vector), gammas, lambdas, framesize(vl), colors)
    end
end