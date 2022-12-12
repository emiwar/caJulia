import SparseArrays
import CUDA
mutable struct Sol{BackgroundsT <: Tuple}
    A::CUDA.CUSPARSE.CuSparseMatrixCSC{Float32, Int32}
    R::CUDA.CuMatrix{Float32}
    C::CUDA.CuMatrix{Float32}
    S::CUDA.CuMatrix{Float32}
    I::CUDA.CuMatrix{Float32}
    backgrounds::BackgroundsT
    gammas::Vector{Float64}
    lambdas::Vector{Float64}
    frame_size::Tuple{Int64, Int64}
    colors::Vector{Colors.RGB{Colors.N0f8}}
end

function Sol(vl::VideoLoader)
    T = nframes(vl)
    height, width = framesize(vl)
    M = width*height
    A = CUDA.cu(SparseArrays.spzeros(Float32, 0, M))
    R = CUDA.zeros(Float32, T, 0)
    C = CUDA.zeros(Float32, T, 0)
    S = CUDA.zeros(Float32, T, 0)
    I = CUDA.zeros(Float32, height, width)
    gammas = fill(0.8, 0)
    lambdas = fill(50.0, 0)
    colors = Colors.RGB{Colors.N0f8}[]
    backgrounds = (PerVideoBackground(vl), PerVideoRank1Background(vl))
    Sol(A, R, C, S, I, backgrounds, gammas, lambdas, (height, width), colors)
end

ncells(sol::Sol) = length(sol.colors)

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

function reconstruct_frame(sol::Sol, frame_id, vl)
    f = (sol.C[frame_id, :]' * sol.A)'
    for bg in sol.backgrounds
        f .+= reconstruct_frame(bg, frame_id, vl)
    end
    return f
end

function residual_frame(sol::Sol, frame_id, vl)
    readframe(vl, frame_id) .- reconstruct_frame(sol, frame_id, vl)
end

function bg_subtracted_frame(sol::Sol, frame_id, vl)
    f = readframe(vl, frame_id)
    for bg in sol.backgrounds
        f .-= reconstruct_frame(bg, frame_id, vl)
    end
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