#!/bin/bash
clear

./bazel-7.4.1-linux-arm64 build \
--compilation_mode=dbg --copt=-g --copt=-O0 --strip=never \
//xla/backends/npu/collectives:hccl_collectives