#!/bin/bash
# > ./etc/build.sh
 
# exit when any command fails
set -e

# Clean build directory
rm -rf ./build
mkdir -p build || echo 0
cd build

# Build LEAP
# Honor an optional LEAP_USE_NOTEX override (ON/OFF); otherwise GPU builds use
# texture objects, except on AMD GPUs without image support, which sample from
# global memory.
CMAKE_ARGS=""
if [ -n "${LEAP_USE_NOTEX}" ]; then
  CMAKE_ARGS="-DLEAP_USE_NOTEX=${LEAP_USE_NOTEX}"
fi
cmake ${CMAKE_ARGS} ..
cmake --build . -j8
