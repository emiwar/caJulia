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
    return c, s, pools
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

function oasis_opt!(sol::Sol, j)
    Rj = Array(sol.R[:, j])
    Rj_zeroclamp = clamp.(Rj, 0.0, Inf)
    loss(x) = negentropy_est(oasis(Rj_zeroclamp, x[1], x[2])[1] .- Rj)
    #TODO constrain gamma, lambda > 0 (maybe for gamma something like > 0.5?)
    opt = Optim.optimize(loss, [sol.gammas[j], sol.lambdas[j]]; iterations=10)
    gamma, lambda = opt.minimizer

    #Temporary hack:
    gamma = clamp(gamma, 0.3, 0.9999)
    lambda = clamp(lambda, 0.0, Inf)
    
    sol.C[:, j], sol.S[:, j] = oasis(Rj_zeroclamp, gamma, lambda)
    sol.gammas[j] = gamma
    sol.lambdas[j] = lambda
end
