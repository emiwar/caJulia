import Statistics
import CUDA
import Images
import SparseArrays
import HDF5
import Colors
import GLMakie
using ProgressMeter
include("motion_correction.jl")
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
example_files_mc = ["recording_20211016_163921-MC.h5",
                    "recording_20220919_135612-MC.h5"]
example_files_huge = ["recording_20211016_163921.hdf5",
                      "recording_20220919_135612.hdf5"]
#hdfLoader = HDFLoader("../data/"*example_files_mc[1]; key="/images",
#                      deviceMemory=8e9, hostMemory=4.0e10);
hdfLoader = HDFLoader("../fake_videos/fake1.h5"; key="/images",
                      deviceMemory=8e9, hostMemory=4.0e10);
gui = GUI.GUIState(hdfLoader);
display(gui.fig)
GUI.calcI!(gui);
GUI.initA!(gui; median_wnd=5);
GUI.initBackground!(gui);
GUI.updateTraces!(gui);
GUI.updateFootprints!(gui);
GUI.mergeCells!(gui);
