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
#include "xla/service/executable.h"
#include "xla/stream_executor/stream_executor.h"

namespace xla {

// An Ascend compiler implementation based on GpuCompiler.
class AscendCompiler : public gpu::GpuCompiler {
 public:
  AscendCompiler();
  ~AscendCompiler() override = default;

  se::Platform::Id PlatformId() const override;

  HloCostAnalysis::ShapeSizeFunction ShapeSizeBytesFunction() const override;

  std::vector<std::string> GetLLVMCommandLineOptions(
      const DebugOptions& debug_options) const override;

  absl::StatusOr<std::unique_ptr<Executable>> RunBackend(
      std::unique_ptr<HloModule> module, se::StreamExecutor* stream_exec,
      const CompileOptions& options) override;

  absl::Status OptimizeHloConvolutionCanonicalization(
      HloModule* hlo_module, const se::GpuComputeCapability& gpu_version,
      se::dnn::VersionInfo dnn_version,
      const se::SemanticVersion& toolkit_version) override;

  absl::StatusOr<BackendCompileResult> CompileTargetBinary(
      const HloModuleConfig& module_config, llvm::Module* llvm_module,
      const stream_executor::DeviceDescription& device_description,
      bool relocatable, const HloModule* debug_module,
      const CompileOptions& options, std::optional<int> shard_number) override;
};

}  // namespace xla

#endif  // XLA_SERVICE_ASCEND_ASCEND_COMPILER_H_
