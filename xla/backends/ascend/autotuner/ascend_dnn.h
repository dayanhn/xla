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

#ifndef XLA_BACKENDS_ASCEND_AUTOTUNER_ASCEND_DNN_H_
#define XLA_BACKENDS_ASCEND_AUTOTUNER_ASCEND_DNN_H_

#include "xla/backends/autotuner/backends.pb.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/compiler.h"
#include "xla/stream_executor/stream_executor.h"

namespace xla {
namespace ascend {

class AscendDnnBackend : public CodegenBackend {
 public:
  explicit AscendDnnBackend(stream_executor::StreamExecutor* stream_executor,
                           const DebugOptions* debug_options, Compiler* compiler,
                           const Compiler::GpuTargetConfig* target_config);

  absl::string_view name() const override;
  autotuner::Backend backend() const override;
  absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>> GetSupportedConfigs(
      const HloInstruction& instr) override;
  absl::StatusOr<std::unique_ptr<BackendConfig>> GetDefaultConfig(
      const HloInstruction& instr) override;
  absl::StatusOr<std::unique_ptr<Executable>> Compile(
      const HloInstruction& instr, const BackendConfig& config) override;
  absl::Status ApplyConfig(HloInstruction& instr,
                           const BackendConfig& config) override;
  bool CanProduceWrongResults() const override;

 private:
  bool IsSupported(const HloInstruction& instr);

  stream_executor::StreamExecutor* stream_executor_;
  const DebugOptions* debug_options_;
  Compiler* compiler_;
  const Compiler::GpuTargetConfig* target_config_;
};

}  // namespace ascend
}  // namespace xla

#endif  // XLA_BACKENDS_ASCEND_AUTOTUNER_ASCEND_DNN_H_
