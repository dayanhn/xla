#!/bin/bash

# Build script for Ascend PJRT client
clear
set -e

# 设置ascend环境变量
source ~/Ascend8.5REL/ascend-toolkit/latest/set_env.sh

echo "Building Ascend PJRT client..."

# Build the Ascend PJRT client
./bazel-7.4.1-linux-arm64 build --compilation_mode=dbg --copt=-g --copt=-O0 --strip=never --action_env=ASCEND_TOOLKIT_HOME=$ASCEND_TOOLKIT_HOME   --linkopt=-L$ASCEND_TOOLKIT_HOME/lib64 --linkopt=-Wl,-rpath,$ASCEND_TOOLKIT_HOME/lib64 --linkopt=-lascendcl --linkopt=-lnnopbase --linkopt=-lopapi_nn //xla/pjrt/plugin/xla_npu:xla_npu_pjrt_client_test

echo "Build completed successfully!"
./bazel-bin/xla/pjrt/plugin/xla_npu/xla_npu_pjrt_client_test
