#!/bin/bash

# Build script for Ascend PJRT client
clear
set -e

# 设置ascend环境变量
source ~/Ascend8.5REL/ascend-toolkit/latest/set_env.sh


./bazel-bin/xla/pjrt/plugin/xla_npu/xla_npu_pjrt_client_test \
     /data3/zhongzhw/code/uni_ai/google/xla/xla/pjrt/plugin/xla_npu/test_jnp_full_stablehlo.mlir
