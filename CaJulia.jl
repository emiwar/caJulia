import Statistics
import CUDA
import Images
import SparseArrays
import HDF5
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

hdfLoader = HDFLoader("../data/20211016_163921_animal1learnday1.nwb",
                      "analysis/recording_20211016_163921-PP-BP-MC/data";
                      deviceMemory=2e10, hostMemory=2e10);
gui = GUI.GUIState(hdfLoader);
display(gui.fig)
GUI.calcI!(gui);
GUI.initA!(gui);
GUI.initBackground!(gui);
GUI.updateTraces!(gui);
GUI.updateFootprints!(gui);
GUI.mergeCells!(gui);

