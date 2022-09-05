function res_norm(Y, A, C)
    #Calculate ||Y - AC||_F without allocating a Y-shaped matrix
    
    #||Y - AC||^2 = sum(Y.^2) - 2*sum((A'*Y) .* C) + sum((A*C).^2)
    #sum((A*C).^2) = sum((A'*A).*(C*C'))
    Y_sqr_sm = mapreduce(y -> y^2, +, Y)
    AY = A'*Y
    AA = A'*A
    CC = C*C'
    return sqrt(Y_sqr_sm - 2*sum(AY.*C) + sum(AA .* CC))
end