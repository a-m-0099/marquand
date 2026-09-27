#!/bin/sh
# Builds llama-cpp-python against Vulkan (RX 7700S) into .venv. Vulkan/SPIR-V headers are vendored; no sudo needed.
set -e
cd "$(dirname "$0")/.."
SDK=$PWD/vendor/sdk
if [ ! -d "$SDK/include/vulkan" ]; then
  for r in Vulkan-Headers SPIRV-Headers; do
    [ -d vendor/$r ] || git clone --depth 1 https://github.com/KhronosGroup/$r vendor/$r
    cmake -S vendor/$r -B vendor/$r/build -DCMAKE_INSTALL_PREFIX="$SDK" -DSPIRV_HEADERS_ENABLE_TESTS=OFF >/dev/null
    cmake --install vendor/$r/build >/dev/null
  done
fi
CMAKE_ARGS="-DGGML_VULKAN=on -DCMAKE_PREFIX_PATH=$SDK -DVulkan_INCLUDE_DIR=$SDK/include -DGGML_NATIVE=on" \
CMAKE_BUILD_PARALLEL_LEVEL=16 \
  uv pip install --python .venv --no-cache --reinstall --no-binary llama-cpp-python "llama-cpp-python==0.3.35" pytest
