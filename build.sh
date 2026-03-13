#!/bin/bash
# Ascend ??????
# ???????????????????: source ~/Ascend8.5REL/ascend-toolkit/latest/set_env.sh
source ~/Ascend8.5REL/ascend-toolkit/latest/set_env.sh

#./bazel-7.4.1-linux-x86_64 build --compilation_mode=dbg --copt=-g --copt=-O0 --strip=never //xla/examples/axpy:priority_fusion_analyzer //xla/examples/axpy:stablehlo_gpu_compiler //xla/examples/axpy:gpu_run_hlopasses //xla/examples/axpy:gpu_run_backend //xla/examples/axpy:gpu_run_executable

./bazel-7.4.1-linux-arm64 build --compilation_mode=dbg --copt=-g --copt=-O0 --strip=never --action_env=ASCEND_TOOLKIT_HOME=$ASCEND_TOOLKIT_HOME --linkopt=-L$ASCEND_TOOLKIT_HOME/lib64 --linkopt=-Wl,-rpath,$ASCEND_TOOLKIT_HOME/lib64 --linkopt=-lascendcl //xla/stream_executor/ascend:test_ascend_platform

./bazel-bin/xla/stream_executor/ascend/test_ascend_platform