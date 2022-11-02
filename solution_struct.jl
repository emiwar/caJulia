import SparseArrays
import CUDA
mutable struct Sol
    A::CUDA.CUSPARSE.CuSparseMatrixCSC{Float32, Int32}
    R::CUDA.CuMatrix{Float32}
    C::CUDA.CuMatrix{Float32}
    S::CUDA.CuMatrix{Float32}
    b0::CUDA.CuMatrix{Float32}
    b1::CUDA.CuVector{Float32}
    f1::CUDA.CuVector{Float32}
    gammas::Vector{Float64}
    lambdas::Vector{Float64}
    frame_size::Tuple{Int64, Int64}
end

function Sol(Y, frame_size)
    M, T = size(Y)
    M == prod(frame_size) || error("Y and frame_size do not match.")
    A = CUDA.cu(SparseArrays.spzeros(Float32, 0, M))
    R = CUDA.zeros(Float32, T, 0)
    C = CUDA.zeros(Float32, T, 0)
    S = CUDA.zeros(Float32, T, 0)
    b0 = CUDA.zeros(Float32, M, 1)
    b1 = CUDA.zeros(Float32, M)
    f1 = CUDA.zeros(Float32, T)
    gammas = fill(0.8, 0)
    lambdas = fill(50.0, 0)
    Sol(A, R, C, S, b0, b1, f1, gammas, lambdas, frame_size)
end


function zeroTraces!(sol::Sol; gamma_guess=0.8, lambda_guess=50.0)
    N = size(sol.A, 1)
    T = size(sol.R, 1)
    sol.R = CUDA.zeros(Float32, T, N)
    sol.C = CUDA.zeros(Float32, T, N)
    sol.S = CUDA.zeros(Float32, T, N)
    sol.gammas = fill(gamma_guess, N)
    sol.lambdas = fill(lambda_guess, N)
end