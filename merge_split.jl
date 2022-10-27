import Statistics

function merge(A, C; thres=.8)
    Ad = CUDA.CuArray(A)
    AA = Array(A*Ad')
    corrs = Array(Statistics.cor(C))
    N = size(A, 1)
    to_delete = zeros(Bool, N)
    for i=1:N
        if to_delete[i]
            continue
        end
        for j=(i+1):N
            if !(to_delete[j]) && AA[i, j] > 0 && corrs[i, j] > thres
                Ad[i, :] .+= view(Ad, j, :)
                Ad[i, :] ./= CUDA.norm(Ad[i, :]) .+ 1.0f-10
                C[:, i] .+= view(C, :, j)
                to_delete[j] = true
            end
        end
    end
    A_new = CUDA.cu(SparseArrays.sparse(Array(Ad)[.!to_delete, :]))
    C_new = C[:, .!to_delete]
    A_new, C_new
end


function merge_loss_increase(Y, A, C)
    #Y_sqr_sm = mapreduce(y -> y^2, +, Y)
    Ad = CUDA.CuArray(A)
    AY = A*Y
    AA = A*Ad'
    AAh = Array(AA)
    CC = C'*C
    full_loss = -2*sum(AY.*C') + sum(AA .* CC)
    loss_mat = SparseArrays.spzeros(size(AA)...)
    for i=1:size(AA, 1), j=1:size(AA, 2)
        if i==j || AAh[i, j] == 0
            continue
        end
        C_temp = copy(C)
        C_temp[:, j] .= view(C, :, i)
        CC_temp = C_temp'*C_temp
        loss_mat[i, j] = -2*sum(AY.*C_temp') + sum(AA .* CC_temp) - full_loss
    end
    loss_mat
end