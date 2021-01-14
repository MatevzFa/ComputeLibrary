#!/bin/bash

#
# Set these variables
#

export ANDROID_ABI="arm64-v8a"
export ANDROID_ARCH="aarch64"
export ANDROID_API_LEVEL="27"

#
# Setup Android NDK
#

export NDK="$HOME/Android/Sdk/ndk/21.1.6352462"
export NDK_TOOLCHAIN="$NDK/toolchains/llvm/prebuilt/linux-x86_64"
export NDK_TOOLCHAIN_BIN="$NDK_TOOLCHAIN/bin"

#
# Build
#

export CC="$NDK_TOOLCHAIN_BIN/$ANDROID_ARCH-linux-android$ANDROID_API_LEVEL-clang"
export CXX="$NDK_TOOLCHAIN_BIN/$ANDROID_ARCH-linux-android$ANDROID_API_LEVEL-clang++"

bear scons \
    Werror=1 \
    -j8 \
    debug=0 assert=1 \
    neon=0 opencl=1 gles_compute=1 \
    embed_kernels=1 \
    os=android arch=$ANDROID_ABI
