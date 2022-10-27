import Optim

struct Pool
    v::Float64
    w::Float64
    t::Int64
    l::Int64
end

function oasis(y, gamma, lambda)
    T = length(y)
    pools = Pool[]
    i = 0
    for t=1:T
        yt = y[t]
        v = yt - lambda*(1 - gamma + (t==T)*gamma)
        push!(pools, Pool(v,1,t,1))
        while i>0 && pools[i+1].v < (gamma^pools[i].l) * pools[i].v
            v = pools[i].w*pools[i].v
            v += gamma^pools[i].l*pools[i+1].w*pools[i+1].v
            v /= pools[i].w + gamma^(2*pools[i].l)*pools[i+1].w
            w = pools[i].w + gamma^(2*pools[i].l) + pools[i+1].w
            pools[i] = Pool(v, w, pools[i].t, pools[i].l + pools[i+1].l)
            pop!(pools)
            i -= 1
        end
        i += 1
    end
    s = zero(y)
    c = zero(y)
    for pool in pools
        for t=0:pool.l-1
            c[pool.t + t] = gamma^t*max(0, pool.v)
        end
        if pool.t > 1
            s[pool.t] = c[pool.t] - gamma*c[pool.t-1]
        end
    end
    return c, s
end

function kurtosis(x)
    N = length(x)
    x_m = sum(x) / N
    return sum((x .- x_m).^4) ./ N ./ ((sum((x .- x_m).^2)./N).^2)
end

function negentropy_est(x)
    N = length(x)
    x_m = sum(x) / N
    sd = sqrt((sum((x .- x_m).^2)./N))
    return (sum(((x .- x_m)./sd).^3) ./ N)^2/12 + (kurtosis(x)-3)^2/48
end

function opt_oasis(raw, gamma_0=0.8, lambda_0=50.0)
    loss(x) = negentropy_est(oasis(raw, x[1], x[2])[1] .- raw)
    #TODO constrain gamma, lambda > 0 (maybe for gamma something like > 0.5?)
    sol = Optim.optimize(loss, [gamma_0, lambda_0]; iterations=10)
    gamma, lambda = sol.minimizer
    c, s = oasis(raw, gamma, lambda)
    return c, s, gamma, lambda
end

function updateTracesOasis(Y, A, C, b0, b1, f1, gammas, lambdas)
    Ad = CUDA.CuArray(A)
    AY = A*Y |> Array
    AA = A*Ad' |> Array
    Ab0 = Array(A*b0) ./ sqrt(size(C, 1))
    Ab1 = A*b1 |> Array
    new_C = C |> Array
    f1h = f1 |> Array
    raw_Y = similar(new_C)
    new_S = similar(new_C)
    @showprogress for j=1:size(new_C, 2)
        raw_Y[:, j] .= view(new_C, :, j) .+ view(AY, j, :) .- new_C*view(AA, j, :) .- Ab0[j] .- Ab1[j]*f1h
        c, s, gamma, lambda = opt_oasis(view(raw_Y, :, j), gammas[j], lambdas[j])
        new_C[:, j] = c
        new_S[:, j] = s
        gammas[j] = gamma
        lambdas[j] = lambda
    end
    return new_C, raw_Y, new_S, gammas, lambdas
end

