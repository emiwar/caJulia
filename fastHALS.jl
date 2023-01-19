function updateTraces!(vl::VideoLoader, sol::Sol; deconvFcn! = zeroClamp!, callback)
    for bg in sol.backgrounds
        prepbackgroundtraceupdate!(bg, sol, vl)
    end
    AY = CUDA.zeros(ncells(sol), nframes(vl)) #similar(sol.C')
    N = VideoLoaders.nsegs(vl)
    for (p, i) = enumerate(optimalorder(vl))
        callback("Calculating traces", p-1, N) 
        Y_seg = readseg(vl, i)
        AY[:, framerange(vl, i)] .= sol.A*Y_seg
        for bg in sol.backgrounds
            traceupdateseg!(bg, Y_seg, i, vl)
        end
    end
    
    Ad = CUDA.CuArray(sol.A)
    AA = sol.A*Ad';
    for bg in sol.backgrounds
        updatetracecorrections!(bg, sol, vl)
    end
    for j=1:ncells(sol)
        callback("Deconvolving", j-1, ncells(sol)) 
        sol.R[:, j] .= view(AY, j, :) .- sol.C*view(AA, j, :) .+ view(sol.C, :, j)
        for bg in sol.backgrounds
            sol.R[:, j] .-= tracecorrection(bg, j)
        end
        deconvFcn!(sol, j)
    end
    callback("Deconvolving", ncells(sol), ncells(sol)) 
    for bg in sol.backgrounds
        traceupdate!(bg, sol, vl)
    end
end

function updateROIs!(vl::VideoLoader, sol::Sol; roi_growth=1, callback)
    for bg in sol.backgrounds
        prepupdate!(bg, sol, vl)
    end
    YC = CUDA.zeros(prod(framesize(vl)), ncells(sol))
    N = VideoLoaders.nsegs(vl)
    for (p, i) = enumerate(optimalorder(vl))
        callback("Calculating footprints", p-1, N)
        Y_seg = readseg(vl, i)
        YC .+= Y_seg*sol.C[framerange(vl, i), :]
        for bg in sol.backgrounds
            updateseg!(bg, Y_seg, i, vl)
        end
    end
    Ad = CUDA.CuArray(sol.A)
    CC = sol.C'*sol.C
    CCh = Array(CC)
    for bg in sol.backgrounds
        updatepixelcorrections!(bg, sol, vl)
    end
    #b0C = CUDA.zeros(size(Ad, 2), size(Ad, 1))
    #for video_id = 1:n_videos(vl)
    #    b0C .+= reshape(view(b0, :, video_id), size(b0, 1), 1) *
    #                sum(view(C, video_range(vl, video_id), :), dims=1)
    #end
    #f1Ch = Array(C'*sol.f1)
    
    Amask = maskA(Ad, sol.frame_size; growth=roi_growth)
    for j=1:size(sol.A, 1)
        callback("Updating footprints", j-1, size(sol.A, 1))
        Ad[j, :] .= view(YC, :, j) .- view((view(CC, j:j, :)*Ad), :) .+
                    view(Ad, j, :).*CCh[j,j]
        for bg in sol.backgrounds
            Ad[j, :] .-= pixelcorrections(bg, j)
        end
        Ad[j, :] .= max.(view(Ad, j, :), 0) .* view(Amask, j, :)
        Ad[j, :] ./= CUDA.norm(Ad[j, :]) .+ 1.0f-10
    end
    for bg in sol.backgrounds
        update!(bg, sol, vl)
    end
    #for video_id = 1:n_videos(vl)
    #    r = video_range(vl, video_id)
    #    f0C = sum(view(C, r, :), dims=1) ./ length(r)
    #    f1f0 = sum(view(sol.f1, r, :))
    #    sol.b0[:, video_id] .= view(sol.mean_frame, :, video_id) .- view(sol.A'*f0C', :) .- f1f0*b1
    #end
    #sol.b0 = max.(sol.b0, 0.0f0)
    #sol.b1 .= view(Yf1 .- (f0C*Ad)' .- f1f0*b0, :)
    #sol.b1 ./= CUDA.norm(b1) .+ 1.0f-10
    to_delete = Array(CUDA.mapreduce(x->x^2, +, Ad; dims=2) .== 0.0)[:]
    sol.A = CUDA.cu(SparseArrays.sparse(Array(Ad[.!to_delete, :])))
    sol.R = sol.R[:, .!to_delete]
    sol.C = sol.C[:, .!to_delete]
    sol.S = sol.S[:, .!to_delete]
    sol.gammas = sol.gammas[.!to_delete]
    sol.lambdas = sol.lambdas[.!to_delete]
    sol.colors = sol.colors[.!to_delete]
end

function zeroClamp!(sol::Sol, j)
    sol.C[:, j] .= max.(view(sol.R, :, j), 0.0f0)
end

function maskA(Ad, frame_size; growth=1)
    kernel = CUDA.ones(1+2*growth, 1+2*growth, 1, 1)
    A_reshaped = reshape(Float32.(Ad' .> 0.0), frame_size..., 1, size(Ad, 1));
    conved = CUDA.CUDNN.cudnnConvolutionForward(kernel, A_reshaped; padding=growth)
    Float32.(reshape(conved, size(Ad, 2), size(Ad, 1))' .> 0.0)
    # CUDA.CuArray(CUDA.CUSPARSE.CuSparseMatrixCSR)
end

function initBackground!(vl::VideoLoader, sol::Sol)
    sol.mean_frame .= 0.0
    for i=1:n_videos(vl)
        frames_in_video = 0.0f0
        eachSegment(vl, vl.segsPerVideo[i]) do seg_id, seg
            sol.mean_frame[:, i] .+= view(sum(seg; dims=2), :)
            frames_in_video += size(seg, 2)
        end
        sol.mean_frame[:, i] ./= frames_in_video;
    end
    sol.b0 .= sol.mean_frame
    sol.b1 .= view(Float32.(maximum(CUDA.CuArray(sol.A); dims=1) .== 0), :);
    sol.b1 ./= CUDA.norm(sol.b1)
    sol.f1 .= 0.0f0
end

function initBackgrounds!(vl::VideoLoader, sol::Sol; callback)
    for bg in sol.backgrounds
        prepinit!(bg, sol, vl)
    end
    N = VideoLoaders.nsegs(vl)
    for (p, i)=enumerate(optimalorder(vl))
        callback("Initializing backgrounds", p-1, N)
        seg = readseg(vl, i)
        for bg in sol.backgrounds
            initseg!(bg, seg, i, vl)
        end
    end
    callback("Initializing backgrounds", N, N)
end
