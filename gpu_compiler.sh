#!/bin/bash

export CUDA_VISIBLE_DEVICES="6,7"
export XLA_FLAGS="--xla_gpu_experimental_enable_triton_heroless_priority_fusion=true --xla_gpu_force_compilation_parallelism=1 --xla_dump_hlo_as_text=true --xla_dump_hlo_as_proto=false --xla_dump_hlo_pass_re=.* --xla_dump_hlo_module_re=.* --xla_dump_emitter_re=triton-fusion --xla_gpu_dump_autotune_logs_to=./tmp/autotune_logs.txt --xla_dump_to=./tmp/xla_dump"

clear
rm -rf ./tmp/xla_dump/*
./bazel-bin/xla/examples/axpy/stablehlo_gpu_compiler ./xla/examples/axpy/stablehlo_gemm.mlir