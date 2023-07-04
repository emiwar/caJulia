import Images
import HDF5
using ProgressMeter


server_folder = "/mnt/dmclab/Vasiliki/Striosomes experiments/Calcium imaging in oprm1/2nd batch sep 2022/Oprm1 calimaging 2nd batch sept 2022 - inscopix data/oprm1 day1 20220919/"
filenames = [
    "recording_20220919_135612.hdf5",
    "recording_20220919_143826.hdf5",
    "recording_20220919_144807.hdf5"
]
orig_files = server_folder .* filenames
orig_video = HDF5.h5open(orig_files[3], "r")

results_folder = "/mnt/dmclab/Emil/arrowmaze_raw_downsampled/"
downsampled_video = HDF5.h5open(results_folder * filenames[3], "w")

w, h, T = size(orig_video["images"])
HDF5.create_dataset(downsampled_video, "data", UInt16, (div(w,2), div(h, 2), T))
@showprogress for t = 1:T
    im = orig_video["images"][:, :, t]
    downsampled_video["data"][:, :, t] = UInt16.(Images.imresize(im, ratio=0.5) .* 4)
end