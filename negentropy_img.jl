
macro powerTuple(variable, n)
    Meta.parse("("*join(("$(string(variable))^$i" for i=1:n), ", ")*")")
end


function negentropy_img(vl::VideoLoader)
    pows = mapreduce(y->(y, y^2, y^3, y^4), .+, vl,
                        (0.0, 0.0, 0.0, 0.0), dims=2)
    T = nframes(vl)
    map(yp->negentropy_approx((yp ./ T)...), pows)
end

function negentropy_img(Y)
    pows = mapreduce(y->(y, y^2, y^3, y^4), .+, Y;
                     init=(0.0, 0.0, 0.0, 0.0), dims=2)
    map(yp->negentropy_approx((yp ./ size(Y,2))...), pows)
end

function negentropy_approx(x1, x2, x3, x4)
    var = x2 - x1^2
    third_moment = x3 - 3*x1*x2 + 2*(x1^3)
    forth_moment = x4 - 4*x1*x3 + 6*(x1^2)*x2 - 3*(x1^4)
    skew2 = (third_moment^2) / (var^3)
    kurtosis = forth_moment / (var^2) - 3.0
    negentropy = skew2 / 12.0 + kurtosis / 48.0
end

function negentropy_approx_no_mean(Y)
    N = size(Y,2)
    y2, y3, y4 = (mapreduce(y->y^i, +, Y; dims=2) ./ N for i=2:4)
    @. (y3^2 / y2^3)/12.0 + y4/(y2^2) / 48.0
end

