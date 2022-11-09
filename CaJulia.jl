import Statistics
import CUDA
import Images
import SparseArrays
import HDF5
import Colors
import GLMakie
using ProgressMeter
include("videoLoader.jl")
include("solution_struct.jl")
include("negentropy_img.jl")
include("viz.jl")
include("initROIs.jl")
include("fastHALS.jl")
include("oasis_opt.jl")
include("merge_split.jl")
include("gui_module.jl")

example_files = ["20211016_163921_animal1learnday1.nwb",
                 "20211017_154731_animal1learnday2.nwb",
                 "20211018_154044_animal1learnday3.nwb",
                 "20211019_161956_animal1learnday4.nwb",
                 "20211101_171346_animal1reversalday15.nwb"]
hdfLoader = HDFLoader("../data/"*example_files[5];
                      deviceMemory=2e10, hostMemory=2e10);
gui = GUI.GUIState(hdfLoader);
display(gui.fig)
GUI.calcI!(gui);
GUI.initA!(gui);
GUI.initBackground!(gui);
GUI.updateTraces!(gui);
GUI.updateFootprints!(gui);
GUI.mergeCells!(gui);
