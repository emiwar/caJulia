function updateTraces!(Y, A, C, b0; max_iter=100)
    Ad = CUDA.CuArray(A)
    AY = A*Y
    AA = A*Ad'
    Ab0 = Array(A*b0) ./ sqrt(size(C, 1))
    c_new = CUDA.CuVector{Float32}(undef, size(C, 1))
    c_diff = CUDA.CuVector{Float32}(undef, size(C, 1))
    last_sqr_err = Inf
    for it=1:max_iter
        sqr_err = 0.0
        for j=1:size(C, 2)
            c_new .= max.(view(C, :, j) .+ view(AY, j, :) .- C*view(AA, j, :) .- Ab0[j], 0)
            c_diff .= (c_new .- view(C, :, j)).^2
            sqr_err += sum(c_diff)
            C[:, j] .= c_new
        end
        if sqr_err < 0.9 * last_sqr_err
            break
        end
        last_sqr_err = sqr_err
    end
end

function updateROIs!(Y, A, C, b0; max_iter=100)
    Ad = CUDA.CuArray(A)
    YC = Y*C
    CC = C'*C
    CCh = Array(CC)
    Yf0 = sum(Y, dims=2) ./ sqrt(size(C, 1))
    f0C = sum(C, dims=1) ./ sqrt(size(C, 1))
    f0Ch = Array(f0C)
    a_new = CUDA.CuVector{Float32}(undef, size(A, 2))
    a_diff = CUDA.CuVector{Float32}(undef, size(A, 2))
    last_sqr_err = Inf
    for it=1:max_iter
        sqr_err = 0.0
        for j=1:size(A, 1)
            a_new .= max.(view(Ad, j, :).*CCh[j,j] .+ view(YC, :, j) .- view(b0, :)*f0Ch[j]
                          .- (view(CC, j:j, :)*Ad)[:], 0)
            a_new ./= sqrt(sum(a_new.^2)) .+ 1.0f-10
            a_diff .= (a_new .- view(Ad, j, :)).^2
            sqr_err += sum(a_diff)
            Ad[j, :] = a_new
        end
        b0 .= Yf0 .- (f0C*Ad)'
        if sqr_err < 0.9 * last_sqr_err
            break
        end
    end
    return CUDA.cu(SparseArrays.sparse(Array(Ad)))
end