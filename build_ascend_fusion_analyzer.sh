#!/bin/bash

# Build script for Ascend ACLNN Fusion Analyzer
clear
set -e

source ~/Ascend8.5REL/ascend-toolkit/latest/set_env.sh

echo "Building Ascend ACLNN Fusion Analyzer..."

./bazel-7.4.1-linux-arm64 build \
    --compilation_mode=dbg --copt=-g --copt=-O0 --strip=never \
    --action_env=ASCEND_TOOLKIT_HOME=$ASCEND_TOOLKIT_HOME   \
    --linkopt=-L$ASCEND_TOOLKIT_HOME/lib64 \
    --linkopt=-Wl,-rpath,$ASCEND_TOOLKIT_HOME/lib64 \
    --linkopt=-lascendcl --linkopt=-lnnopbase --linkopt=-lopapi_nn \
    --linkopt=-lhccl --linkopt=-lhcomm  --linkopt=-lopapi_math \
    //xla/backends/ascend/transforms:aclnn_fusion_analyzer

echo "Build completed successfully!"

#export PATH=/data3/zhongzhw/code/uni_ai/google/xla/bazel-bin/xla/backends/ascend/transforms:$PATH
#bazel-bin/xla/backends/ascend/transforms/aclnn_fusion_analyzer xla/backends/ascend/transforms/test_conv.hlo
