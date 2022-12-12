import Statistics
import CUDA
import Images
import SparseArrays
import HDF5
import Colors
import GLMakie
using ProgressMeter
#include("motion_correction.jl")
include("videoLoaders/videoLoaders.jl"); using .VideoLoaders
include("solution_struct.jl")
include("negentropy_img.jl")
include("viz.jl")
include("initROIs.jl")
include("fastHALS.jl")
include("oasis_opt.jl")
include("merge_split.jl")
include("backgrounds/backgrounds.jl")
include("gui_module.jl")
#include("aligned_video_reader.jl")

example_files = ["20211016_163921_animal1learnday1.nwb",
                 "20211017_154731_animal1learnday2.nwb",
                 "20211018_154044_animal1learnday3.nwb",
                 "20211019_161956_animal1learnday4.nwb",
                 "20211020_150705_animal1learnday5.nwb",
                 "20211021_172832_animal1learnday6.nwb",
                 "20211101_171346_animal1reversalday15.nwb"]
example_files_mc = ["recording_20211016_163921-MC.h5",
                    "recording_20220919_135612-MC.h5"]
example_files_huge = ["recording_20211016_163921.hdf5",
                      "recording_20220919_135612.hdf5"]

#nwbLoader = VideoLoaders.NWBLoader("../data/"*example_files[6])
#splitLoader = VideoLoaders.SplitLoader(nwbLoader, 5)
alignedLoader = VideoLoaders.AlignedHDFLoader("../data/aligned_videos_first_3.csv", 5;
                                 pathPrefix = "../data/")
hostCache = VideoLoaders.CachedHostLoader(alignedLoader; max_memory=4e10)
deviceCache = VideoLoaders.CachedDeviceLoader(hostCache; max_memory=1e10)


gui = GUI.GUIState(deviceCache);
display(gui.fig)
GUI.calcI!(gui);
GUI.initA!(gui; threshold=3e-3, median_wnd=5);
GUI.initBackground!(gui);
GUI.updateTraces!(gui);
GUI.updateFootprints!(gui);
GUI.mergeCells!(gui);


#sol = Sol(deviceCache);
#sol.I = negentropy_img(deviceCache);
#initA!(sol; threshold=1e-3, median_wnd=5);
#zeroTraces!(sol);
#initBackground!(hdfLoader, sol);
#updateTraces!(deviceCache, sol)#, deconvFcn! = oasis_opt!);
#updateROIs!(deviceCache, sol);
#merge!(sol, thres=.6) |> length
