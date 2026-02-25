#!/bin/bash

./bazel-7.4.1-linux-x86_64 build   --compilation_mode=dbg --copt=-g --copt=-O0 --strip=never  //xla/examples/axpy:stablehlo_compiler