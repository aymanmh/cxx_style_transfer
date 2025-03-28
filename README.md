# [Fast Neural Style Transfer](https://github.com/jcjohnson/fast-neural-style) Inference C++ implementation

### This repo rewrites the [fns_candy_style_transfer](https://github.com/microsoft/onnxruntime-inference-examples/blob/main/c_cxx/fns_candy_style_transfer/README.md) example from onnxruntime samples, the change are:
- Use C++ instead for C.
- Use OpenCV (4.x) for image manipulation instead of raw C functions.
- Use OnnxRuntime with DirectML from Visual Studio NuGet packages instead of manual download.
- Cuda code is not implemented.
- The models are included in the assets folder

### Before building:
- Reinstall Microsoft.ML.OnnxRuntime.DirectML from Nuget package manager
- Update the 'include' and 'library' paths for OpenCV (from project settings).

the use models are from [here](https://github.com/microsoft/Windows-Machine-Learning/tree/master/Samples/FNSCandyStyleTransfer/UWP/cs/Assets), if other versions are used, input and output data might need to be changed accordingly.


#### for the us of other fnst models see the other branch of this repo.