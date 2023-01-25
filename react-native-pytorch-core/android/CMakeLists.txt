# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.4.1)

set (CMAKE_VERBOSE_MAKEFILE ON)

# This should be matched with the Unit Test Build. Please keep them in sync. See ../cxx/test/CMakeLists.txt
set (CMAKE_CXX_STANDARD 14)
set (CMAKE_CXX_FLAGS "-DFOLLY_NO_CONFIG=1 -DFOLLY_HAVE_CLOCK_GETTIME=1 -DFOLLY_HAVE_MEMRCHR=1 -DFOLLY_USE_LIBCPP=1 -DFOLLY_MOBILE=1 -DON_ANDROID -DONANDROID -DUSE_CAFFA2=0 -DBUILD_CAFFA2=0")
# set (CMAKE_CXX_FLAGS "-DFOLLY_NO_CONFIG=1 -DFOLLY_HAVE_CLOCK_GETTIME=1 -DFOLLY_HAVE_MEMRCHR=1 -DFOLLY_USE_LIBCPP=1 -DFOLLY_MOBILE=1 -DON_ANDROID -DONANDROID -DUSE_KINETO=0 -DBUILD_CAFFE2=0 -DUSE_DISTRIBUTED=0 -DUSE_NCCL=0 -DBUILD_TEST=0 -DUSE_XNNPACK=0 -DUSE_FBGEMM=0 -DUSE_QNNPACK=0 -DUSE_MKLDNN=0 -DUSE_MIOPEN=0 -DUSE_NNPACK=0 -DBUILD_CAFFE2_OPS=0 -DUSE_TENSORPIPE=0")
set (PACKAGE_NAME "torchlive")
set (BUILD_DIR ${CMAKE_SOURCE_DIR}/build)
set (RN_SO_DIR ${NODE_MODULES_DIR}/react-native/ReactAndroid/src/main/jni/first-party/react/jni)

# set (TorchVision_DIR "/home/leruo/torchvision/build")
set (TorchVision_DIR "/home/leruo/torchvision/build/")
set (Torch_DIR "/home/leruo/libs/libtorch/build")
set (Caffe2_DIR "/home/leruo/libs/libtorch/torch/share/cmake/Caffe2")
set (TORCH_LIBRARY "/home/leruo/libs/libtorch/build")
set (C10_LIBRARY "/home/leruo/libs/libtorch/build/libs")
# set (Torch_DIR "/home/leruo/libs/libtorch/share/cmake/Torch")
# set (Caffe2_DIR "/home/leruo/libs/libtorch/share/cmake/Caffe2")
# set (ATen_DIR "/home/leruo/libs/libtorch/share/cmake/ATen")
# set (Tensorpipe_DIR "/home/leruo/libs/libtorch/share/cmake/Tensorpipe")
list(APPEND CMAKE_PREFIX_PATH "/home/leruo/libs/libtorch/build/libs")
list(APPEND CMAKE_PREFIX_PATH "/home/leruo/libs/libtorch/build")
list(APPEND CMAKE_PREFIX_PATH "/home/leruo/torchvision/build")

# PyTorch Core shared

add_library(
        ${PACKAGE_NAME}
        SHARED
        ../cxx/src/torchlive/torchlive.cpp
        ../cxx/src/torchlive/common/BaseHostObject.cpp
        ../cxx/src/torchlive/experimental/ExperimentalNamespace.cpp
        ../cxx/src/torchlive/filesystem/FilesystemNamespace.cpp
        ../cxx/src/torchlive/media/Blob.cpp
        ../cxx/src/torchlive/media/BlobHostObject.cpp
        ../cxx/src/torchlive/media/MediaNamespace.cpp
        ../cxx/src/torchlive/media/audio/AudioHostObject.cpp
        ../cxx/src/torchlive/media/image/ImageHostObject.cpp
        ../cxx/src/torchlive/Promise.cpp
        ../cxx/src/torchlive/ThreadPool.cpp
        ../cxx/src/torchlive/torch/DictHostObject.cpp
        ../cxx/src/torchlive/torch/IValueHostObject.cpp
        ../cxx/src/torchlive/torch/jit/JITNamespace.cpp
        ../cxx/src/torchlive/torch/jit/mobile/ModuleHostObject.cpp
        ../cxx/src/torchlive/torch/TensorHostObject.cpp
        ../cxx/src/torchlive/torch/TorchNamespace.cpp
        ../cxx/src/torchlive/torch/utils/ArgumentParser.cpp
        ../cxx/src/torchlive/torch/utils/constants.cpp
        ../cxx/src/torchlive/torch/utils/converter.cpp
        ../cxx/src/torchlive/torch/utils/helpers.cpp
        ../cxx/src/torchlive/torchvision/AbstractScriptModule.cpp
        ../cxx/src/torchlive/torchvision/CenterCropModule.cpp
        ../cxx/src/torchlive/torchvision/GrayscaleModule.cpp
        ../cxx/src/torchlive/torchvision/NormalizeModule.cpp
        ../cxx/src/torchlive/torchvision/ResizeModule.cpp
        ../cxx/src/torchlive/torchvision/TorchvisionHostObject.cpp
        ../cxx/src/torchlive/torchvision/VisionTransformHostObject.cpp
        ../cxx/src/torchlive/vision/TransformsHostObject.cpp
        ../cxx/src/torchlive/vision/VisionHostObject.cpp
        src/main/cpp/OnLoad.cpp
        src/main/cpp/torchlive/media/NativeJSRefBridge.cpp
        src/main/cpp/torchlive/media/audio/Audio.cpp
        src/main/cpp/torchlive/media/audio/JIAudio.cpp
        src/main/cpp/torchlive/media/image/Image.cpp
        src/main/cpp/torchlive/media/image/JIImage.cpp
)

# includes

file (GLOB LIBFBJNI_INCLUDE_DIR "${BUILD_DIR}/fbjni-*-headers.jar/")
file (GLOB PYTORCH_INCLUDE_DIRS "${BUILD_DIR}/pytorch_android_lite*.aar/headers")

target_include_directories(
        ${PACKAGE_NAME}
        PRIVATE
        "${LIBFBJNI_INCLUDE_DIR}"
        "${NODE_MODULES_DIR}/react-native/React"
        "${NODE_MODULES_DIR}/react-native/React/Base"
        "${NODE_MODULES_DIR}/react-native/ReactAndroid/src/main/jni"
        "${NODE_MODULES_DIR}/react-native/ReactAndroid/src/main/java/com/facebook/react/turbomodule/core/jni"
        "${NODE_MODULES_DIR}/react-native/ReactCommon"
        "${NODE_MODULES_DIR}/react-native/ReactCommon/callinvoker"
        "${NODE_MODULES_DIR}/react-native/ReactCommon/jsi"
        "${NODE_MODULES_DIR}/react-native/ReactCommon/runtimeexecutor"
        "${PYTORCH_INCLUDE_DIRS}"
        "../cxx/src"
        "src/main/cpp"
)

# find libraries

file (GLOB LIBRN_DIR "${BUILD_DIR}/react-native-0*/jni/${ANDROID_ABI}")
file (GLOB PYTORCH_LINK_DIRS "${BUILD_DIR}/pytorch_android_lite*.aar/jni/${ANDROID_ABI}")
# file (GLOB TORCHVISION_LIBRARY_DIRS "/home/leruo/torchvision")
# file (GLOB TORCHVISION_LIBRARY_DIRS "${NODE_MODULES_DIR}/torchvision/build/lib.linux-x86_64-3.10/torchvision")
# file (GLOB TORCHVISION_LIBRARY_DIRS "/usr/local/lib/python3.10/dist-packages/torchvision/_C.so")



find_library(
        LOG_LIB
        log
)

find_library(
        FBJNI_LIBRARY
        fbjni
        PATHS ${LIBRN_DIR}
        NO_CMAKE_FIND_ROOT_PATH
)

find_library(
        REACT_NATIVE_JNI_LIB
        reactnativejni
        PATHS ${LIBRN_DIR}
        NO_CMAKE_FIND_ROOT_PATH
)


find_library(
        PYTORCH_LIBRARY
        pytorch_jni_lite
        PATHS ${PYTORCH_LINK_DIRS}
        NO_CMAKE_FIND_ROOT_PATH
)
# find_package(TorchVision REQUIRED)
# target_link_libraries(torchvision PUBLIC TorchVision::TorchVision)
# find_package(
#         TorchVision
#         TorchVision_DIR "/home/leruo/torchvision/cmake"
#         REQUIRED
# )
# add_library(
#         "torchvision"
#         SHARED
#         "/home/leruo/torchvision/CMakeLists.txt"
#        "/home/leruo/torchvision/torchvision/csrc/vision.cpp"
#        "/home/leruo/torchvision/torchvision/csrc/ops/cpp"
#        ${TORCHVISION_LIBRARY_DIRS}
# )
#find_library(
#        TORCHVISION_LIBRARY
#        NAMES torchvision
#        PATHS ${TORCHVISION_LIBRARY_DIRS}
#        NO_CMAKE_FIND_ROOT_PATH
#)

# For RN 0.66+, use the distributed libjsi.so
if(${REACT_NATIVE_MINOR_VERSION} GREATER_EQUAL 66)
        find_library(
                JSI_LIB
                jsi
                PATHS ${LIBRN_DIR}
                NO_CMAKE_FIND_ROOT_PATH
        )
endif()

# linking

target_link_libraries(
        ${PACKAGE_NAME}
        ${LOG_LIB}
        ${REACT_NATIVE_JNI_LIB}
        ${PYTORCH_LIBRARY}
        ${FBJNI_LIBRARY}
        ${JSI_LIB}
        # TorchVision::TorchVision
        android
)
