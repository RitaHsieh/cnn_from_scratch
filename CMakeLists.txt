cmake_minimum_required(VERSION 3.10) # CUDA 需要至少 3.8，但最好用更高版本
project(neural_net_in_cpp_test LANGUAGES CXX CUDA)

# 設定 C++ 標準
set(CMAKE_CXX_STANDARD 11)

# 設定 CUDA 標準（可選，根據需求選擇合適的標準）
find_package(CUDA  REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")

set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# # CUDA 編譯器架構（根據目標 GPU 修改）
set(CMAKE_CUDA_ARCHITECTURES 61)  # 適用於 NVIDIA Turing 架構，75 = RTX 20 系列

# 如果需要，可以取消註解此行來啟用性能分析
# set (CMAKE_CXX_FLAGS "-pg")

# 定義執行檔案目標，包括 C++ 和 CUDA 檔案
add_executable(neural_net_in_cpp_test 
    src/test.cu
    src/NetworkModel.cu 
    src/MNISTDataLoader.cpp 
    src/Tensor.cpp 
    src/FullyConnected.cu
    src/Sigmoid.cpp 
    src/SoftmaxClassifier.cpp 
    src/Dropout.cpp 
    src/ReLU.cu 
    src/MaxPool.cu
    src/LinearLRScheduler.cpp 
    src/Conv2d.cu  # 添加 CUDA 檔案
)

# 包含 CUDA 頭文件目錄
target_include_directories(neural_net_in_cpp_test PRIVATE ${CMAKE_SOURCE_DIR}/include)

# 啟用 CUDA 可分離編譯（適合多檔案大型專案）
set_target_properties(neural_net_in_cpp_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
