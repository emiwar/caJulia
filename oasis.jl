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

function sweep_lambda(y, c, s, lambdas, gamma)
    res = Float64[]
    err = Float64[]
    kur = Float64[]
    c_cor = Float64[]
    s_cor = Float64[]
    for lambda in lambdas
        c_est, s_est = oasis(y, gamma, lambda)
        push!(res, sum((c_est .- y).^2))
        push!(err, sum((c_est .- c).^2))
        push!(kur, kurtosis(c_est .- y))
        push!(c_cor, Statistics.cor(c_est, c))
        push!(s_cor, Statistics.cor(s_est, s))
    end
    return res, err, kur, c_cor, s_cor
end

function kurtosis(x)
    N = length(x)
    x_m = sum(x) / N
    return sum((x .- x_m).^4) ./ N ./ ((sum((x .- x_m).^2)./N).^2)
end

function est_sigma(y)
    P = welch_pgram(y)
    f = freq(P)
    p = power(P)
    sqrt(mean(p[f .> 0.25]) / 2)
end

T = 25000
gamma = 0.9
sigma = 100*0.5
fake_s = 100 .* (randn(T).^2) .* (rand(T) .> 0.95)
fake_c = similar(fake_s)
fake_c[1] = fake_s[1]
for i=2:T
    fake_c[i] = gamma*fake_c[i-1] + fake_s[i]
end
fake_y = fake_c + sigma*randn(T);
lambdas = (0:500)
res, err, kur, c_cor, s_cor = sweep_lambda(fake_y, fake_c, fake_s, lambdas, gamma)
ested_sigma = est_sigma(fake_y)
plot(lambdas, sqrt.(res ./ T), label="Residual")
plot!([lambdas[1], lambdas[end]], [sigma, sigma], label="True sigma")
plot!([lambdas[1], lambdas[end]], [ested_sigma, ested_sigma], label="Estimated sigma")
plot!(lambdas, sqrt.(err ./ T), label="Actual error")
plot!(lambdas, 100*(kur .- 3), label="Kurtosis")
plot(lambdas, c_cor, label="Correlation true C vs estimated C")
plot!(lambdas, s_cor, label="Correlation true S vs estimated S")