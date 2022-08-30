function updateTraces!(Y, A, C, b0, b1, f1; max_iter=100)
    Ad = CUDA.CuArray(A)
    AY = A*Y
    AA = A*Ad'
    Ab0 = Array(A*b0) ./ sqrt(size(C, 1))
    b1Y = (b1'*Y)'
    Ab1 = A*b1
    Ab1h = Array(Ab1)
    b0b1 = b0'*b1 ./ sqrt(size(C, 1))
    c_new = CUDA.CuVector{Float32}(undef, size(C, 1))
    c_diff = CUDA.CuVector{Float32}(undef, size(C, 1))
    last_sqr_err = Inf
    for it=1:max_iter
        sqr_err = 0.0
        for j=1:size(C, 2)
            c_new .= max.(view(C, :, j) .+ view(AY, j, :) .- C*view(AA, j, :) .- Ab0[j] .- Ab1h[j]*f1, 0)
            c_diff .= c_new .- view(C, :, j)
            sqr_err += CUDA.norm(c_diff)
            C[:, j] .= c_new
        end
        f1 .= b1Y .- C*Ab1 .- b0b1
        if sqr_err < 0.9 * last_sqr_err
            break
        end
        last_sqr_err = sqr_err
    end
end

function updateROIs!(Y, A, C, b0, b1, f1; max_iter=100)
    Ad = CUDA.CuArray(A)
    YC = Y*C
    CC = C'*C
    CCh = Array(CC)
    Yf0 = sum(Y, dims=2) ./ sqrt(size(C, 1))
    f0C = sum(C, dims=1) ./ sqrt(size(C, 1))
    f0Ch = Array(f0C)
    Yf1 = Y*f1
    f1C = C'*f1
    f1Ch = Array(C'*f1)
    f1f0 = sum(f1) ./ sqrt(size(C, 1))
    a_new = CUDA.CuVector{Float32}(undef, size(A, 2))
    a_diff = CUDA.CuVector{Float32}(undef, size(A, 2))
    last_sqr_err = Inf
    for it=1:max_iter
        sqr_err = 0.0
        for j=1:size(A, 1)
            a_new .= max.(view(Ad, j, :).*CCh[j,j] .+ view(YC, :, j) 
                          .- (view(CC, j:j, :)*Ad)[:]
                          .- view(b0, :)*f0Ch[j]
                          .- view(b1, :)*f1Ch[j], 0)
            a_new ./= CUDA.norm(a_new) .+ 1.0f-10
            a_diff .= a_new .- view(Ad, j, :)
            sqr_err += CUDA.norm(a_diff)
            Ad[j, :] = a_new
        end
        b0 .= Yf0 .- (f0C*Ad)' .- f1f0*b1
        b1 .= view(Yf1 .- (f0C*Ad)' .- f1f0*b0, :)
        b1 ./= CUDA.norm(b1) .+ 1.0f-10
        if sqr_err < 0.9 * last_sqr_err
            break
        end
    end
    return CUDA.cu(SparseArrays.sparse(Array(Ad)))
end