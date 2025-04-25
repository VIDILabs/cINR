# ======================================================================== #
# Copyright 2019-2020 Qi Wu                                                #
#                                                                          #
# Licensed under the Apache License, Version 2.0 (the "License");          #
# you may not use this file except in compliance with the License.         #
# You may obtain a copy of the License at                                  #
#                                                                          #
#     http://www.apache.org/licenses/LICENSE-2.0                           #
#                                                                          #
# Unless required by applicable law or agreed to in writing, software      #
# distributed under the License is distributed on an "AS IS" BASIS,        #
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. #
# See the License for the specific language governing permissions and      #
# limitations under the License.                                           #
# ======================================================================== #
# ======================================================================== #
# Copyright 2018-2019 Ingo Wald                                            #
#                                                                          #
# Licensed under the Apache License, Version 2.0 (the "License");          #
# you may not use this file except in compliance with the License.         #
# You may obtain a copy of the License at                                  #
#                                                                          #
#     http://www.apache.org/licenses/LICENSE-2.0                           #
#                                                                          #
# Unless required by applicable law or agreed to in writing, software      #
# distributed under the License is distributed on an "AS IS" BASIS,        #
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. #
# See the License for the specific language governing permissions and      #
# limitations under the License.                                           #
# ======================================================================== #

set(target nncache_core)

include(configure_cuda) # skip optix here

# ======================================================================== #
# renderer core
# ======================================================================== #

add_library(${target} SHARED
  samplers/neural_sampler.cu
  samplers/neural_sampler.cpp
  samplers/neural_sampler.h
  samplers/sampler.cu
  samplers/sampler.h

  networks/tcnn_impl.cu
  networks/tcnn_network.h
  networks/nnapi.h

  volumes/macrocell.cu
  volumes/macrocell.h
  volumes/neural_volume.cu
  volumes/simple_volume.cu
  volumes/volumes.h

  instantvnr_types.cu
  instantvnr_types.h  
  array.h
  ../app.cu
)

set_target_properties(${target} 
PROPERTIES 
  CXX_STANDARD 17
  CUDA_STANDARD 17
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON
)

target_include_directories(${target} 
PRIVATE 
  ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_LIST_DIR}
PUBLIC 
  ${CMAKE_CURRENT_LIST_DIR}/..
  ${CMAKE_CURRENT_LIST_DIR}/../..
)

target_compile_options(${target} 
PRIVATE 
  $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe="--diag_suppress=177">   # -> variable declared but not referenced
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe="--diag_suppress=20044"> # -> extern variable '...' treated as static
)

target_link_libraries(${target} 
PRIVATE 
  TBB::tbb
  $<$<BOOL:${ENABLE_OUT_OF_CORE}>:$<$<PLATFORM_ID:Linux>:aio>>
PUBLIC # intentionally keep these library public
  rendercommon tcnn
)

target_compile_definitions(${target} 
PRIVATE
  $<$<BOOL:${ENABLE_OUT_OF_CORE}>:ENABLE_OUT_OF_CORE=1>
PUBLIC
  $<$<BOOL:${ENABLE_LOGGING}>:ENABLE_LOGGING=1>
  $<IF:$<BOOL:${MACROCELL_SIZE_MIP}>,MACROCELL_SIZE_MIP=${MACROCELL_SIZE_MIP},MACROCELL_SIZE_MIP=4>
  $<IF:$<BOOL:${DISABLE_ADAPTIVE_SAMPLING}>,ADAPTIVE_SAMPLING=0,ADAPTIVE_SAMPLING=1>
)
