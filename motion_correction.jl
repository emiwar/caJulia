import Images
import TestImages
import Colors
import CUDA
import Plots
import FFTW

CUDA.allowscalar(false)
testImgCol = TestImages.testimage("lighthouse")
testImg = CUDA.cu(Float32.(Colors.Gray.(testImgCol)))
testImgSource = testImg[10:end, 1:end-4]
testImgTarget = testImg[1:end-9, 5:end]

plan = CUDA.CUFFT.plan_fft(testImgSource);
source_freq = plan * testImgSource;
target_freq = plan * testImgTarget;
img_product = source_freq .* conj(target_freq);
cross_corr = plan \ img_product;
_, maxidx = CUDA.findmax(abs.(cross_corr))
max_val = @CUDA.allowscalar cross_corr[maxidx]
phasediff = atan(imag(max_val), real(max_val))
shape = size(source_freq)
cntr = (1 .+ shape) ./ 2
shift = ifelse.(maxidx.I .> cntr, maxidx.I .- shape, maxidx.I) .- 1
freqs1 = CUDA.CUFFT.fftfreq(shape[1], 1.0f0) |> collect |> CUDA.cu
freqs2 = CUDA.CUFFT.fftfreq(shape[2], 1.0f0)' |> collect |> CUDA.cu
shifted_freq = source_freq .* cis.(-Float32(2pi) * (freqs1 .* shift[1] .+ freqs2 .* shift[2]) .+ phasediff)
shifted = real(plan \ shifted_freq)
Plots.heatmap(Array(shifted), color=:grays)