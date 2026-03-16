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

#include "xla/stream_executor/ascend/ascend_platform_id.h"

namespace xla {

AscendCompiler::AscendCompiler() {
}

absl::StatusOr<std::vector<std::unique_ptr<Executable>>> AscendCompiler::Compile(
    std::unique_ptr<HloModule> hlo_module,
    std::vector<se::StreamExecutor*> stream_execs,
    const CompileOptions& options) {
  // TODO: Implement Ascend-specific compilation
  return absl::UnimplementedError("Compile not implemented for Ascend");
}

absl::StatusOr<std::unique_ptr<HloModule>> AscendCompiler::RunHloPasses(
    std::unique_ptr<HloModule> module, se::StreamExecutor* stream_exec,
    const CompileOptions& options) {
  // TODO: Implement Ascend-specific HLO passes
  return std::move(module);
}

absl::StatusOr<std::unique_ptr<Executable>> AscendCompiler::RunBackend(
    std::unique_ptr<HloModule> module, se::StreamExecutor* stream_exec,
    const CompileOptions& options) {
  // TODO: Implement Ascend-specific backend
  return absl::UnimplementedError("RunBackend not implemented for Ascend");
}

absl::StatusOr<std::vector<std::unique_ptr<CompiledModule>>> AscendCompiler::CompileAheadOfTime(
    std::unique_ptr<HloModule> hlo_module,
    const AotCompilationOptions& options) {
  // TODO: Implement Ascend-specific AOT compilation
  return absl::UnimplementedError("CompileAheadOfTime not implemented for Ascend");
}

se::Platform::Id AscendCompiler::PlatformId() const {
  return stream_executor::ascend::kAscendPlatformId;
}

HloCostAnalysis::ShapeSizeFunction AscendCompiler::ShapeSizeBytesFunction() const {
  // TODO: Implement Ascend-specific shape size function
  return [](const Shape& shape) {
    return ShapeUtil::ByteSizeOf(shape);
  };
}

absl::StatusOr<std::unique_ptr<CompiledModule>> AscendCompiler::Export(
    Executable* executable) {
  // TODO: Implement Ascend-specific export
  return absl::UnimplementedError("Export not implemented for Ascend");
}

absl::StatusOr<std::unique_ptr<CompiledModule>> AscendCompiler::LoadAotCompilationResult(
    const std::string& serialized_aot_result) {
  // TODO: Implement Ascend-specific AOT result loading
  return absl::UnimplementedError("LoadAotCompilationResult not implemented for Ascend");
}

std::vector<std::string> AscendCompiler::GetLLVMCommandLineOptions(
    const DebugOptions& debug_options) const {
  // TODO: Add Ascend-specific LLVM options
  return {};
}

}  // namespace xla
