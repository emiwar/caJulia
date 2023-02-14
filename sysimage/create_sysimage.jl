import PackageCompiler
packages = ["DataStructures",
            "Distributions",
            "CUDA",
            "cuDNN",
            "Images",
            "HDF5",
            "Colors",
            "ColorSchemes",
            "CxxWrap",
            "Observables",
            "QML",
            "Qt5QuickControls_jll",
            "Qt5QuickControls2_jll"]

built_in = ["SparseArrays", "Distributed", "Statistics"]
PackageCompiler.create_sysimage(packages, sysimage_path="sysimage/custom_sys2.so",
       precompile_execution_file="sysimage/precompile_execution_file.jl")