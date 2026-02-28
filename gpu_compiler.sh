#!/bin/bash

export CUDA_VISIBLE_DEVICES="6,7"
export XLA_FLAGS="--xla_gpu_experimental_enable_triton_heroless_priority_fusion=true --xla_gpu_force_compilation_parallelism=1 --xla_dump_hlo_as_dot=true --xla_dump_fusion_visualization=true --xla_dump_hlo_as_text=true --xla_dump_hlo_as_proto=false --xla_dump_hlo_pass_re=.* --xla_dump_hlo_module_re=.* --xla_dump_emitter_re=triton-fusion --xla_gpu_dump_autotune_logs_to=./tmp/autotune_logs.txt --xla_dump_to=./tmp/xla_dump"

clear
rm -rf ./tmp/xla_dump/*

# 检查命令行参数
if [ $# -ne 1 ]; then
  echo "Usage: $0 <EXCUTE_TYPE>"
  echo "EXCUTE_TYPE options:"
  echo "  1: run stablehlo_gpu_compiler"
  echo "  2: run priority_fusion_analyzer"
  echo "  3: run gpu_run_hlopasses"
  echo "  4: run gpu_run_backend"
  echo "  5: run gpu_run_executable"
  exit 1
fi

# 获取命令行参数作为 EXCUTE_TYPE
EXCUTE_TYPE=$1

# 验证 EXCUTE_TYPE 的值
if ! [[ $EXCUTE_TYPE =~ ^[1-5]$ ]]; then
  echo "Error: EXCUTE_TYPE must be between 1 and 5"
  echo "Usage: $0 <EXCUTE_TYPE>"
  echo "EXCUTE_TYPE options:"
  echo "  1: run stablehlo_gpu_compiler"
  echo "  2: run priority_fusion_analyzer"
  echo "  3: run gpu_run_hlopasses"
  echo "  4: run gpu_run_backend"
  echo "  5: run gpu_run_executable"
  exit 1
fi

if [ $EXCUTE_TYPE -eq 1 ]; then
  echo "run stablehlo_gpu_compiler"
  ./bazel-bin/xla/examples/axpy/stablehlo_gpu_compiler ./xla/examples/axpy/stablehlo_gemm.mlir
elif [ $EXCUTE_TYPE -eq 2 ]; then
  echo "run priority_fusion_analyzer"
  ./bazel-bin/xla/examples/axpy/priority_fusion_analyzer ./xla/examples/axpy/hlomodule_priority_fusion.mlir
elif [ $EXCUTE_TYPE -eq 3 ]; then
  echo "run gpu_run_hlopasses"
  ./bazel-bin/xla/examples/axpy/gpu_run_hlopasses ./xla/examples/axpy/hlomodule_ori.mlir
elif [ $EXCUTE_TYPE -eq 4 ]; then
  echo "run gpu_run_backend"
  ./bazel-bin/xla/examples/axpy/gpu_run_backend ./xla/examples/axpy/hlomodule_opti.mlir
elif [ $EXCUTE_TYPE -eq 5 ]; then
  echo "run gpu_run_executable"
  ./bazel-bin/xla/examples/axpy/gpu_run_executable ./xla/examples/axpy/hlomodule_opti.mlir
fi
