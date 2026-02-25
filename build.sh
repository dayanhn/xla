#!/bin/bash

./bazel-7.4.1-linux-x86_64 build   --compilation_mode=dbg --copt=-g --copt=-O0 --strip=never  //xla/examples/axpy:stablehlo_gpu_compiler
# ./bazel-bin/xla/examples/axpy/stablehlo_gpu_compiler ./xla/examples/axpy/stablehlo_axpy.mlir