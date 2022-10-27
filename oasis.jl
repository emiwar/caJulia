import Statistics
import DSP

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
    negentropy = Float64[]
    for lambda in lambdas
        c_est, s_est = oasis(y, gamma, lambda)
        push!(res, sum((c_est .- y).^2))
        push!(err, sum((c_est .- c).^2))
        push!(kur, kurtosis(c_est .- y))
        push!(negentropy, negentropy_est(c_est .- y))
        push!(c_cor, Statistics.cor(c_est, c))
        push!(s_cor, Statistics.cor(s_est, s))
    end
    return res, err, kur, negentropy, c_cor, s_cor
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

function est_sigma(y)
    P = DSP.welch_pgram(y)
    f = DSP.freq(P)
    p = DSP.power(P)
    sqrt(Statistics.mean(p[f .> 0.25]) / 2)
end

function joint_opt(y, lambdas, gammas, c_true, s_true)
    negent = zeros(length(lambdas), length(gammas))
    c_cor = zeros(length(lambdas), length(gammas))
    s_cor = zeros(length(lambdas), length(gammas))
    @showprogress "Running OASIS" for i=1:length(lambdas)
        for j=1:length(gammas)
            lambda = lambdas[i]
            gamma = gammas[j]
            c_est, s_est = oasis(y, gamma, lambda)
            negent[i, j] = negentropy_est(c_est .- y)
            c_cor[i, j] = Statistics.cor(c_est, c_true)
            s_cor[i, j] = Statistics.cor(s_est, s_true)
        end
    end
    return negent, c_cor, s_cor
end
   
T = 25000
gamma = 0.9
sigma = 10*1.0#0.2
fake_s = 10 .* (randn(T).^4) .* (rand(T) .> 0.98)
fake_c = similar(fake_s)
fake_c[1] = fake_s[1]
for i=2:T
    fake_c[i] = gamma*fake_c[i-1] + fake_s[i]
end
fake_y = fake_c + sigma*randn(T);
lambdas = (0:1000)/2
res, err, kur, negent, c_cor, s_cor = sweep_lambda(fake_y, fake_c, fake_s, lambdas, gamma)
ested_sigma = est_sigma(fake_y)
plot(lambdas, sqrt.(res ./ T), label="Residual")
plot!([lambdas[1], lambdas[end]], [sigma, sigma], label="True sigma")
plot!([lambdas[1], lambdas[end]], [ested_sigma, ested_sigma], label="Estimated sigma")
plot!(lambdas, sqrt.(err ./ T), label="Actual error")
plot!(lambdas, negent*10000, label="Negentropy")
p=plot!(lambdas, 100*(kur .- 3), label="Kurtosis", ylim=(-20, 50))
display(p)
plot(lambdas, c_cor, label="Correlation true C vs estimated C")
plot!(lambdas, s_cor, label="Correlation true S vs estimated S")


lambdas = 0:200
gammas = 0.8:0.0025:0.95
negent, c_cor, s_cor = joint_opt(fake_y, lambdas, gammas, fake_c, fake_s);
heatmap(gammas, lambdas, log10.(negent))
heatmap(gammas, lambdas, log10.(1.0 .- c_cor))
heatmap(gammas, lambdas, log10.(1.0 .- s_cor))


loss(x) = negentropy_est(oasis(fake_y, x[1], x[2])[1] .- fake_y)
sol = Optim.optimize(loss, [0.89, 50.0])
sol.minimizer

import HDF5
real_c = HDF5.h5open("example_traces.h5", "r") do fid
    HDF5.read(fid, "just_init")
end
ex_c = real_c[:,32]
real_gamma= 0.88#0.9#0.85#0.88
lambdas = (0:1000)/10
res, err, kur, negent, c_cor, s_cor = sweep_lambda(ex_c, ex_c, ex_c, lambdas, real_gamma)
ested_sigma = est_sigma(ex_c)

p1 = plot(lambdas, sqrt.(res ./ T), label="Residual")
#plot!([lambdas[1], lambdas[end]], [sigma, sigma], label="True sigma")
plot!(p1, [lambdas[1], lambdas[end]], [ested_sigma, ested_sigma], label="Estimated sigma")
#plot!(lambdas, sqrt.(err ./ T), label="Actual error")
p2 = plot(lambdas, negent*10000, label="Negentropy")
p3 = plot(lambdas, kur, label="Kurtosis")
plot!(p3, [extrema(lambdas)...], [3, 3], label="3")
display(plot(p1,p2,p3, layout=(3,1)))