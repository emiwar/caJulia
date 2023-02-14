import Distributed
import DataStructures
import Statistics
import Distributions

import CUDA
import cuDNN
import Images
import SparseArrays
import HDF5
import Colors
import ColorSchemes
import CxxWrap
import Observables
import QML
import Qt5QuickControls_jll
import Qt5QuickControls2_jll

#CUDA stuff
CUDA.allowscalar(false)
a = randn(10)
b = CUDA.cu(a)
c = Array(b)

a = zeros(Int16, 5, 5)
b = CUDA.cu(a)
c = Float32.(b)
d = c .^ 2
e = Array(b)
f = CUDA.zeros(5, 5)
g = d .+ f
kernel = CUDA.ones(3, 3, 1, 1)
c_reshaped = reshape(Float32.(c' .> 0.0), 5, 5, 1, 1)
conved = cuDNN.cudnnConvolutionForward(kernel, c_reshaped; padding=1)
h = view(g, :, 1)

#Colors / images
blank = zeros(Colors.RGB{Colors.N0f8}, (100, 100))
small_blank = Images.imresize(blank, 50, 50)
colorscheme = ColorSchemes.colorschemes[:grays]
small_blank[1, 1] = colorscheme[0.5]

#SparseArrays
a = Float32[0 0 0.5 0; 0 1 0 0]
b = SparseArrays.sparse(a)
c = CUDA.cu(b)
d = CUDA.CuArray(c)
e = Array(c)
f = d .^ 2
g = c .* c
h = f .== g
any(h)

#List of strings
a = ["a", "b", "c"]
push!(a, string(:d))
path = ENV["PATH"]
println("A string: ", a[1])
