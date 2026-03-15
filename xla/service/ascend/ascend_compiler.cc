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

#include "xla/service/ascend/ascend_compiler.h"

#include "xla/service/ascend/ascend_executable.h"
#include "xla/stream_executor/ascend/ascend_platform_id.h"

namespace xla {

AscendCompiler::AscendCompiler() {
}

AscendCompiler::~AscendCompiler() {
}

std::unique_ptr<HloModule> AscendCompiler::RunHloPasses(
    std::unique_ptr<HloModule> module,
    se::StreamExecutor* stream_executor,
    const CompileOptions& options) {
  // TODO: Implement Ascend-specific HLO passes
  return module;
}

std::unique_ptr<Executable> AscendCompiler::CreateExecutable(
    std::unique_ptr<HloModule> module,
    se::StreamExecutor* stream_executor) {
  return std::make_unique<AscendExecutable>(std::move(module), stream_executor);
}

se::Platform::Id AscendCompiler::PlatformId() const {
  return stream_executor::ascend::kAscendPlatformId;
}

HloCostAnalysis::ShapeSizeFunction AscendCompiler::ShapeSizeBytesFunction() const {
  // TODO: Implement Ascend-specific shape size function
  return GpuCompiler::ShapeSizeBytesFunction();
}

}  // namespace xla