import Glob
import HDF5
import Plots
using ProgressMeter


folder = "/mnt/dmclab/Emil/arrowmaze_raw_downsampled/"
for fo in Glob.glob("*", folder)
    for fi in Glob.glob("*.hdf5", fo)
        a, b = split(fi, "/")[end-1:end]
        println("$a/$b")
    end
end

first_frames = Matrix{UInt16}[]
titles = String[]
for fo in Glob.glob("*", folder)
    for fi in Glob.glob("*.hdf5", fo)
        a, b = split(fi, "/")[end-1:end]
        println("$a/$b")
        push!(titles, "$a/$b")
        HDF5.h5open(fi, "r") do fid
            push!(first_frames, fid["data"][:, : ,1])
            #push!(plots, Plots.heatmap(fid["data"][:, : ,1]))
        end
    end
end

ps = [Plots.plot(Images.Gray.((Float32.(f) .- 1000) ./ 2500)') for f=first_frames];
p = Plots.plot(ps..., aspect=:equal, axis=:off, grid=:off, fmt=:png, margin=-5mm)
Plots.savefig(p, "../data/first_frames_animal6.png")



baseFolder = "/mnt/dmclab/Vasiliki/Striosomes experiments/Calcium imaging in oprm1/2nd batch sep 2022/Oprm1 calimaging 2nd batch sept 2022 - inscopix data/"
first_frames_large = Matrix{UInt16}[]
filenames_large = String[]
@showprogress for f in Glob.glob("*/*.hdf5", baseFolder)
    if endswith(f, "_gpio.hdf5")
        continue
    end
    HDF5.h5open(f, "r") do fid
        push!(first_frames_large, fid["images"][:, : ,1])
    end
    push!(filenames_large, f)
end

subplots = Plots.plot(layout=(18, 6), grid=:off, axis=:off, fmt=:png, size=(2000, 5000));
i = 0
j = 0
last_folder_name = ""
for (fn, ff) in zip(filenames_large, first_frames_large)
    folder_name, file_name = split(fn, "/")[end-1:end]
    if last_folder_name != folder_name
        i += 1
        j = 1
        last_folder_name = folder_name
    else
        j += 1
    end
    Plots.plot!(subplots[i, j], Images.Gray.((Float32.(ff) .- 250) ./ 1000)',
                title= folder_name[6:end-9] * "\n" * file_name[11:end-5],
                xticks=[], yticks=[])
    println(folder_name, " ", file_name)
end
Plots.savefig(subplots, "../data/first_frames_all_videos_full_size.png")
