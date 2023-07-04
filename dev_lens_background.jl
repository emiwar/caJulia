import GLMakie
import CUDA
import cuDNN
import Images
import Colors
using ProgressMeter

GLMakie.activate!(inline=false)
include("videoLoaders/videoLoaders.jl");# using .VideoLoaders;
include("solution_struct.jl")
include("backgrounds/backgrounds.jl")
include("negentropy_img.jl")
include("initROIs.jl")

function makieims(vl, images...; clim=(-20, 50))
    nims = length(images)
    fig = GLMakie.Figure()
    obs_ims = map(im->GLMakie.lift(im->reshape(Array(im), VideoLoaders.framesize(vl)), im), images)
    for (i, im)=enumerate(obs_ims)
        ax = GLMakie.Axis(fig[1, i] )
        GLMakie.image!(ax, im, colorrange=clim)
        ax.aspect = GLMakie.DataAspect()
    end
    GLMakie.display(fig)
end

nwbLoader = VideoLoaders.HDFLoader(example_files_server[3],
                                   "images", manual_cropping_server[3])
splitLoader = VideoLoaders.SplitLoader(nwbLoader, 20)
hostCache = VideoLoaders.CachedHostLoader(splitLoader; max_memory=3.2e10)
filterLoader = VideoLoaders.FilterLoader(hostCache, Images.OffsetArrays.no_offset_view(Images.Kernel.DoG(10.0)))
mcLoader = VideoLoaders.MotionCorrectionLoader(filterLoader, (600:1200, 250:600))
deviceCache = VideoLoaders.CachedDeviceLoader(mcLoader; max_memory=1.0e10)

#VideoLoaders.fitMotionCorrection!
shifted_bg_freq, unshifted_bg_freq, sm_frame_shifted_freq, sm_frame_unshifted_freq, sm_shifts_forward, sm_shifts_backward = VideoLoaders.fitMotionCorrection!(mcLoader);

N = VideoLoaders.nframes(mcLoader)
im1 = GLMakie.Observable(real.(CUDA.CUFFT.ifft(sm_frame_unshifted_freq)) ./ N);
im2 = GLMakie.Observable(real.(CUDA.CUFFT.ifft(sm_frame_shifted_freq)) ./ N);
#im2 = GLMakie.Observable(lensback.b);
makieims(mcLoader, im1, im2);
im1[] = real.(CUDA.CUFFT.ifft(unshifted_bg_freq))
im2[] = real.(CUDA.CUFFT.ifft(shifted_bg_freq))

neg_ent_im = negentropy_img(deviceCache);
neg_ent_im_host = reshape(Array(neg_ent_im), VideoLoaders.framesize(deviceCache)) ;
segment_peaks_unionfind(neg_ent_im_host, 80, 2000.0; callback=(m, i, N)->println("$m ($(round(100*i/N))%)")) |> length

lensback = LensBackground(deviceCache)
staticbg = StaticBackground(deviceCache);
sol = Sol(deviceCache, (lensback, staticbg));

im1 = GLMakie.Observable(staticbg.b);
im2 = GLMakie.Observable(lensback.b);
makieims(deviceCache, im1, im2);


vl = deviceCache
prepinit!(lensback, sol, vl);
prepinit!(staticbg, sol, vl);
@showprogress for i=optimalorder(vl)
    seg = readseg(vl, i)
    for bg in sol.backgrounds
        initseg!(bg, seg, i, vl)
    end
end
im1[] = staticbg.b;
im2[] = lensback.b;

#=
for bg in sol.backgrounds
    prepupdate!(bg, sol, vl)
end
@showprogress for i = optimalorder(vl)
    Y_seg = readseg(vl, i)
    for bg in sol.backgrounds
        updateseg!(bg, Y_seg, i, vl)
    end
end
im1[] = staticbg.b;
im2[] = lensback.b;
=#
for i=1:20
    update!(staticbg, sol, vl);
    quickupdate!(lensback, staticbg, vl);
end
im1[] = staticbg.b;
im2[] = lensback.b;

update!(lensback, sol, vl);
im2[] = lensback.b;


#for bg in sol.backgrounds
#    updatepixelcorrections!(bg, sol, vl)
#end
#for bg in sol.backgrounds
#    update!(bg, sol, vl)
#end

#Problem: how to compute the negentropy while correcting for the bg?
#Maybe hack like this:
@showprogress for i=optimalorder(mcLoader)
    seg = VideoLoaders.readseg_bghack(mcLoader, i, lensback.b)

end