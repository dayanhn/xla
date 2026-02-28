#!/bin/bash

./bazel-7.4.1-linux-x86_64 build --compilation_mode=dbg --copt=-g --copt=-O0 --strip=never //xla/examples/axpy:priority_fusion_analyzer //xla/examples/axpy:stablehlo_gpu_compiler //xla/examples/axpy:gpu_run_hlopasses //xla/examples/axpy:gpu_run_backend
