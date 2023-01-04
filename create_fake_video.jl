import HDF5

w, h = (600, 800)
T = 2000
n_neurons = 50

gammas = 0.7 .+ 0.25*rand(n_neurons)
sparsity = 0.98
S = (randn(Float32, T, n_neurons).^2) .* (rand(Float32, T, n_neurons) .> sparsity)
C = similar(S)
for neuron=1:n_neurons
    C[1, neuron] = S[1, neuron]
    for t=2:T
        C[t, neuron] = gammas[neuron]*C[t-1, neuron] + S[t, neuron]
    end
end
offsets = rand(Float32, n_neurons)*5
C .+= offsets'

A = zeros(Float32, h, w, n_neurons)
for neuron=1:n_neurons
    c = [50 + rand()*(w - 100), 50 + rand()*(h - 100)]
    sigma = 5+5*rand()
    cov = -10 + 20*rand()
    sigInv = inv([sigma^2 cov; cov sigma^2])
    A[:, :, neuron] .= [exp(-0.5*(([x,y]-c)'*sigInv*([x,y]-c))) for y=1:h, x=1:w]
end

shifts = zeros(Int64, T, 2)
x = [0.0, 0.0]
for t=1:T
    x = 0.7 .* x .+ 10 .* (1 .- 2*rand(2))
    shifts[t, :] = Int64.(round.(x))
end

#reshape(reshape(A, w*h, n_neurons) * C', w, h, n_neurons)
Y = zeros(Float32, h-100, w-100, T)
for t=1:T
    frame = reshape(reshape(A, w*h, n_neurons) * C[t, :], h, w)
    sx, sy = shifts[t, :]
    Y[:, :, t] .= frame[50+sy:h-51+sy, 50+sx:w-51+sx]
end

Y .+= randn(Float32, h-100, w-100, T);

HDF5.h5open("../fake_videos/fake2.h5", "w") do fid
    fid["/images"] = Int16.(round.(10.0 .+ 10 .* Y))
    fid["/ground_truth/A"] = A
    fid["/ground_truth/C"] = C
    fid["/ground_truth/S"] = S
    fid["/ground_truth/gammas"] = gammas
    fid["/ground_truth/shifts"] = shifts
end