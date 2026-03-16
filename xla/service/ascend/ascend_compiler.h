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

#include "xla/service/llvm_compiler.h"

namespace xla {

// An Ascend compiler implementation based on LLVMCompiler.
class AscendCompiler : public LLVMCompiler {
 public:
  AscendCompiler();
  ~AscendCompiler() override = default;

  absl::StatusOr<std::vector<std::unique_ptr<Executable>>> Compile(
      std::unique_ptr<HloModule> hlo_module,
      std::vector<se::StreamExecutor*> stream_execs,
      const CompileOptions& options) override;

  absl::StatusOr<std::unique_ptr<HloModule>> RunHloPasses(
      std::unique_ptr<HloModule> module, se::StreamExecutor* stream_exec,
      const CompileOptions& options) override;

  absl::StatusOr<std::unique_ptr<Executable>> RunBackend(
      std::unique_ptr<HloModule> module, se::StreamExecutor* stream_exec,
      const CompileOptions& options) override;

  absl::StatusOr<std::vector<std::unique_ptr<CompiledModule>>>
  CompileAheadOfTime(std::unique_ptr<HloModule> hlo_module,
                     const AotCompilationOptions& options) override;

  se::Platform::Id PlatformId() const override;

  HloCostAnalysis::ShapeSizeFunction ShapeSizeBytesFunction() const override;

  absl::StatusOr<std::unique_ptr<CompiledModule>> Export(
      Executable* executable) override;

  absl::StatusOr<std::unique_ptr<CompiledModule>> LoadAotCompilationResult(
      const std::string& serialized_aot_result) override;

  std::vector<std::string> GetLLVMCommandLineOptions(
      const DebugOptions& debug_options) const;
};

}  // namespace xla

#endif  // XLA_SERVICE_ASCEND_ASCEND_COMPILER_H_
