/* Copyright 2026 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef XLA_SERVICE_ASCEND_ASCEND_COMPILER_H_
#define XLA_SERVICE_ASCEND_ASCEND_COMPILER_H_

#include "xla/service/gpu/gpu_compiler.h"
#include "xla/stream_executor/ascend/ascend_platform_id.h"

namespace xla {

// Compiler for Ascend devices.
class AscendCompiler : public GpuCompiler {
 public:
  AscendCompiler();
  ~AscendCompiler() override;

  // GpuCompiler interface
  std::unique_ptr<HloModule> RunHloPasses(
      std::unique_ptr<HloModule> module,
      se::StreamExecutor* stream_executor,
      const CompileOptions& options) override;

  std::unique_ptr<Executable> CreateExecutable(
      std::unique_ptr<HloModule> module,
      se::StreamExecutor* stream_executor) override;

  se::Platform::Id PlatformId() const override;

  HloCostAnalysis::ShapeSizeFunction ShapeSizeBytesFunction() const override;
};

}  // namespace xla

#endif  // XLA_SERVICE_ASCEND_ASCEND_COMPILER_H_