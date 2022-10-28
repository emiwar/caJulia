function updateTraces!(Y, sol::Sol; deconvFcn! = zeroClamp!)
    AY = sol.A*Y
    b1Y = (sol.b1'*Y)'
    A = sol.A
    C = sol.C
    f1 = sol.f1
    Ad = CUDA.CuArray(A)
    AA = A*Ad'
    Ab0 = Array(A*sol.b0) ./ sqrt(size(C, 1))
    Ab1 = A*sol.b1
    Ab1h = Array(Ab1)
    b0b1 = sol.b0'*sol.b1 ./ sqrt(size(C, 1))
    @showprogress "Deconvolving" for j=1:size(C, 2)
        sol.R[:, j] .= view(C, :, j) .+ view(AY, j, :) .- C*view(AA, j, :) .- Ab0[j] .- Ab1h[j]*f1
        deconvFcn!(sol, j)
    end
    sol.f1 .= b1Y .- C*Ab1 .- b0b1
end

function updateROIs!(Y, sol::Sol; roi_growth=1)
    YC = Y*sol.C
    Yf1 = Y*sol.f1
    C = sol.C
    b0 = sol.b0
    b1 = sol.b1
    Yf0 = sum(Y, dims=2) ./ sqrt(size(C, 1))
    Ad = CUDA.CuArray(sol.A)
    CC = C'*C
    CCh = Array(CC)
    f0C = sum(C, dims=1) ./ sqrt(size(C, 1))
    f0Ch = Array(f0C)
    f1C = C'*sol.f1
    f1Ch = Array(C'*sol.f1)
    f1f0 = sum(sol.f1) ./ sqrt(size(C, 1))
    Amask = maskA(Ad, sol.frame_size; growth=roi_growth)
    @showprogress "Updating footprints" for j=1:size(sol.A, 1)
        Ad[j, :] .= max.(view(Ad, j, :).*CCh[j,j] .+ view(YC, :, j)
                      .- view((view(CC, j:j, :)*Ad), :)
                      .- view(b0, :)*f0Ch[j]
                      .- view(b1, :)*f1Ch[j], 0) .* view(Amask, j, :)
        Ad[j, :] ./= CUDA.norm(Ad[j, :]) .+ 1.0f-10
    end
    sol.b0 .= Yf0 .- (f0C*Ad)' .- f1f0*b1
    sol.b1 .= view(Yf1 .- (f0C*Ad)' .- f1f0*b0, :)
    sol.b1 ./= CUDA.norm(b1) .+ 1.0f-10
    to_delete = Array(CUDA.mapreduce(x->x^2, +, Ad; dims=2) .== 0.0)[:]
    sol.A = CUDA.cu(SparseArrays.sparse(Array(Ad[.!to_delete, :])))
    sol.R = sol.R[:, .!to_delete]
    sol.C = sol.C[:, .!to_delete]
    sol.S = sol.S[:, .!to_delete]
    sol.gammas = sol.gammas[.!to_delete]
    sol.lambdas = sol.lambdas[.!to_delete]
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


function initBackground!(Y, sol::Sol)
    sol.b0 .= sum(Y, dims=2) ./ Float32(sqrt(size(Y, 2)));
    sol.b1 .= view(Float32.(maximum(CUDA.CuArray(sol.A); dims=1) .== 0), :);
    sol.b1 ./= CUDA.norm(sol.b1)
    sol.f1 .= 0.0f0
end
