function updateTraces!(vl::VideoLoader, sol::Sol; deconvFcn! = zeroClamp!)
    AY, b1Y = leftMul((sol.A, sol.b1'), vl);
    A = sol.A;
    C = sol.C;
    f1 = sol.f1;
    Ad = CUDA.CuArray(A)
    AA = A*Ad';
    Ab0 = similar(C);
    b0b1 = similar(f1);
    for video_id = 1:n_videos(vl)
        r = video_range(vl, video_id)
        Ab0[r, :] .= reshape(A*view(sol.b0, :, video_id), 1, size(C, 2))
        b0b1[r] .= sol.b0[:, video_id]'*sol.b1
    end
    Ab1 = A*sol.b1;
    Ab1h = Array(Ab1);
    @showprogress "Deconvolving" for j=1:size(C, 2)
        sol.R[:, j] .= view(C, :, j) .+ view(AY, j, :) .- C*view(AA, j, :) .- 
                    view(Ab0, :, j) .- Ab1h[j]*f1
        deconvFcn!(sol, j)
    end
    #sol.f1 .= view(b1Y, :) .- C*Ab1 .- b0b1
end

function updateROIs!(vl::VideoLoader, sol::Sol; roi_growth=1)
    YC, Yf1 = rightMul(vl, (sol.C, sol.f1))
    C = sol.C
    b0 = sol.b0
    b1 = sol.b1
    Ad = CUDA.CuArray(sol.A)
    CC = C'*C
    CCh = Array(CC)
    
    b0C = CUDA.zeros(size(Ad, 2), size(Ad, 1))
    for video_id = 1:n_videos(vl)
        b0C .+= reshape(view(b0, :, video_id), size(b0, 1), 1) *
                    sum(view(C, video_range(vl, video_id), :), dims=1)
    end
    f1Ch = Array(C'*sol.f1)
    
    Amask = maskA(Ad, sol.frame_size; growth=roi_growth)
    @showprogress "Updating footprints" for j=1:size(sol.A, 1)
        Ad[j, :] .= max.(view(Ad, j, :).*CCh[j,j] .+ view(YC, :, j)
                      .- view((view(CC, j:j, :)*Ad), :)
                      .- view(b0C, :, j)
                      .- view(b1, :)*f1Ch[j], 0) .* view(Amask, j, :)
        Ad[j, :] ./= CUDA.norm(Ad[j, :]) .+ 1.0f-10
    end
    for video_id = 1:n_videos(vl)
        r = video_range(vl, video_id)
        f0C = sum(view(C, r, :), dims=1) ./ length(r)
        f1f0 = sum(view(sol.f1, r, :))
        sol.b0[:, video_id] .= view(sol.mean_frame, :, video_id) .- view(sol.A'*f0C', :) .- f1f0*b1
    end
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

