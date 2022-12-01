import SparseArrays
import CUDA
mutable struct Sol
    A::CUDA.CUSPARSE.CuSparseMatrixCSC{Float32, Int32}
    R::CUDA.CuMatrix{Float32}
    C::CUDA.CuMatrix{Float32}
    S::CUDA.CuMatrix{Float32}
    I::CUDA.CuMatrix{Float32}
    b0::CUDA.CuMatrix{Float32}
    b1::CUDA.CuVector{Float32}
    f1::CUDA.CuVector{Float32}
    mean_frame::CUDA.CuMatrix{Float32}
    gammas::Vector{Float64}
    lambdas::Vector{Float64}
    frame_size::Tuple{Int64, Int64}
    colors::Vector{Colors.RGB{Colors.N0f8}}
end

function Sol(height, width, length, nVideos=1)
    T = length
    M = width*height
    A = CUDA.cu(SparseArrays.spzeros(Float32, 0, M))
    R = CUDA.zeros(Float32, T, 0)
    C = CUDA.zeros(Float32, T, 0)
    S = CUDA.zeros(Float32, T, 0)
    I = CUDA.zeros(Float32, height, width)
    b0 = CUDA.zeros(Float32, M, nVideos)
    b1 = CUDA.zeros(Float32, M)
    f1 = CUDA.zeros(Float32, T)
    mean_frame = CUDA.zeros(Float32, M, nVideos)
    gammas = fill(0.8, 0)
    lambdas = fill(50.0, 0)
    colors = Colors.RGB{Colors.N0f8}[]
    Sol(A, R, C, S, I, b0, b1, f1, mean_frame, gammas, lambdas, (height, width), colors)
end

Sol(vl::VideoLoader) = Sol(vl.frameSize..., vl.nFrames, n_videos(vl))


function zeroTraces!(sol::Sol; gamma_guess=0.8, lambda_guess=50.0)
    N = size(sol.A, 1)
    T = size(sol.R, 1)
    sol.R = CUDA.zeros(Float32, T, N)
    sol.C = CUDA.zeros(Float32, T, N)
    sol.S = CUDA.zeros(Float32, T, N)
    sol.gammas = fill(gamma_guess, N)
    sol.lambdas = fill(lambda_guess, N)
    sol.colors = map(rand_color, 1:N)
end

function reconstruct_frame(sol::Sol, frame_id::Int64, video_id::Int64)
    f = (sol.C[frame_id, :]' * sol.A)'
    f .+= view(sol.b0[:, video_id], :)
    f .+= @CUDA.allowscalar sol.b1 .* sol.f1[frame_id]
    return f
end

function load_ground_truth(sol::Sol, filename::String)
    HDF5.h5open(filename, "r") do fid
         A = Array(fid["/ground_truth/A"])
         A = reshape(A[50:end-51, 50:end-51, :], prod(size(A)[1:2] .- 100), size(A, 3))'
         norms = sqrt.(sum(A .^ 2, dims=2))
         A ./= norms
         A[A .< 1e-5] .= 0.0
         sol.A = CUDA.cu(SparseArrays.sparse(A))

         zeroTraces!(sol)
         C = Array(fid["/ground_truth/C"])
         sol.C = CUDA.cu(C .* norms')
    end
end