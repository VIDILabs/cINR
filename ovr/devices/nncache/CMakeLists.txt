
message(STATUS "Enable Neural Data Cache Device")
add_library(device_nncache SHARED
  renderer/method_pathtracing.cu
  renderer/method_raymarching.cu
  renderer/method_megakernel.cu
  renderer/renderer.cu
  renderer/renderer_nncache.cu

  array.cpp
  device.cu
  cachemanager.cu
  space_partition.cu
  volume.cu

  $<$<BOOL:${ENABLE_OPENGL}>:ui_controls.cpp>

  ../../serializer.cpp
  ../../serializer.h
  ../../api.cpp
  ../../api.h
)

# execute_process(
#   COMMAND python -c 
#     "import torch, os; print(os.path.dirname(torch.__file__))"
#   OUTPUT_VARIABLE PYTHON_TORCH_SITE
#   OUTPUT_STRIP_TRAILING_WHITESPACE
# )
# list(APPEND CMAKE_PREFIX_PATH "${PYTHON_TORCH_SITE}")
# find_package(Torch REQUIRED)

# find_package(pybind11 REQUIRED)

# add_library(pytorch_api OBJECT ../../pyTorch_bindings/pyTorch_api.cpp)

# target_link_libraries(pytorch_api PRIVATE 
#   rendercommon
#   pybind11::embed
# )

set_target_properties(device_nncache 
PROPERTIES 
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  CUDA_SEPARABLE_COMPILATION ON
)
target_link_libraries (device_nncache PRIVATE 
  rendercommon 
  nncache_core 
  TBB::tbb 
  nvtx3-cpp
)
target_compile_options(device_nncache PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>)

# --- Link GcCore libraries ---
target_link_libraries(device_nncache PRIVATE
  Common Data Math TinyXml Preprocessor
  cudaCommon cudaPreprocessor cudaGPUCache
  $<$<BOOL:${ENABLE_OPENGL}>:imgui>
  Threads::Threads
)
