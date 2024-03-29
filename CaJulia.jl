import Statistics
import CUDA
import Images
import SparseArrays
import HDF5
import Colors
import GLMakie
import Distributions
import LinearAlgebra
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
include("save_result.jl")
#include("aligned_video_reader.jl")

example_files = ["20211016_163921_animal1learnday1.nwb",
                 "20211017_154731_animal1learnday2.nwb",
                 "20211018_154044_animal1learnday3.nwb",
                 "20211019_161956_animal1learnday4.nwb",
                 "20211020_150705_animal1learnday5.nwb",
                 "20211021_172832_animal1learnday6.nwb",
                 "20211101_171346_animal1reversalday15.nwb",
                 "20211016_173112_animal3learnday1.nwb"]
example_files_huge = ["recording_20211016_163921.hdf5",
                      "recording_20220919_135612.hdf5"]
#folder = "/mnt/dmclab/Vasiliki/Striosomes experiments/Calcium imaging in oprm1"
#folder *= "/1st batch oct 2021/Oprm1_cal_im_gCamp8_1221_dataanalysis/exported videos/"
#nwbLoader = VideoLoaders.HDFLoader("../data/"*example_files_huge[1], "images")
nwbLoader = VideoLoaders.NWBLoader("../data/"*example_files[end])
splitLoader = VideoLoaders.SplitLoader(nwbLoader, 5)
#alignedLoader = VideoLoaders.AlignedHDFLoader("../data/aligned_videos_animal3.csv", 10;
#                                 pathPrefix = "../data/")
hostCache = VideoLoaders.CachedHostLoader(splitLoader; max_memory=3.2e10)
#minSubtr = VideoLoaders.SubtractMinLoader(hostCache)
#VideoLoaders.calcmin!(minSubtr);
deviceCache = VideoLoaders.CachedDeviceLoader(hostCache; max_memory=1.0e10)


gui = GUI.GUIState(deviceCache);
display(gui.fig)
GUI.calcI!(gui);

GUI.initA!(gui);
println("Done initing A ($(ncells(gui.sol[])) cells found.)")
GUI.initBackground!(gui);
GUI.updateTraces!(gui);
GUI.updateFootprints!(gui);
GUI.mergeCells!(gui);

#sol = Sol(deviceCache);
#sol.I = negentropy_img(deviceCache);
sol = gui.sol[];
initA!(sol);
zeroTraces!(sol);
initBackgrounds!(deviceCache, sol);
updateTraces!(deviceCache, sol, deconvFcn! = oasis_opt!);

for it = 3:6
    updateROIs!(deviceCache, sol);
    for i=1:3
        merge!(sol, thres=.6)
        println(ncells(sol))
    end
    updateTraces!(deviceCache, sol, deconvFcn! = oasis_opt!);
    to_hdf("animal3_it$(it).h5", sol, deviceCache)
end