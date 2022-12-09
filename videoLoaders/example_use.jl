#fileLoader = NWBLoader("../data/20211016_163921_animal1learnday1.nwb")
#splitLoader = SplitLoader(fileLoader, 5)
#hostCache = CachedHostLoader(splitLoader; max_memory=4e10)
#deviceCache = CachedDeviceLoader(hostCache; max_memory=2e10)
#seg1 = readseg(deviceCache, 2);


#alignedLoader = AlignedHDFLoader("../data/aligned_videos_first_3.csv", 5; pathPrefix = "../data/")
#hostCache = CachedHostLoader(alignedLoader, max_memory=4e10)
#deviceCache = CachedDeviceLoader(hostCache; max_memory=2e10)